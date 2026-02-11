"""Patient data endpoints."""

import io
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Response

# Ensure the project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from triage.config import INBOX_REPORTS_DIR, INBOX_VOLUMES_DIR, OUTPUT_DIR
from triage.ct_processor import extract_slice_as_multiwindow_image, load_nifti_volume
from triage.fhir_janitor import FHIRJanitor
from triage.output_generator import load_triage_result
from api.models import (
    PRIORITY_COLORS,
    PRIORITY_NAMES,
    PatientCondition,
    PatientDemographics,
    PatientFHIRContext,
    TriageResult,
)
from api.services.demo_service import demo_service

router = APIRouter(prefix="/api/patients", tags=["patients"])

_volume_cache: dict[str, tuple] = {}
_VOLUME_CACHE_MAX = 2


def _get_cached_volume(path: Path):
    """Return (volume_hu, metadata) from cache, loading if necessary."""
    key = str(path.resolve())
    if key in _volume_cache:
        return _volume_cache[key]
    volume, metadata = load_nifti_volume(path)
    if len(_volume_cache) >= _VOLUME_CACHE_MAX:
        oldest_key = next(iter(_volume_cache))
        del _volume_cache[oldest_key]
    _volume_cache[key] = (volume, metadata)
    return volume, metadata


@lru_cache(maxsize=512)
def _encode_slice_jpeg(volume_path_str: str, slice_index: int) -> bytes:
    """Encode a CT slice as JPEG bytes with caching (~512 entries × ~40KB = ~20MB)."""
    volume, _metadata = _get_cached_volume(Path(volume_path_str))
    image = extract_slice_as_multiwindow_image(volume, slice_index)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _clear_slice_cache():
    """Invalidate the slice JPEG cache (call if volume data changes)."""
    _encode_slice_jpeg.cache_clear()


def _find_report_path(patient_id: str) -> Optional[Path]:
    """Find the report JSON file for a patient."""
    # Check inbox first
    inbox_path = INBOX_REPORTS_DIR / f"{patient_id}.json"
    if inbox_path.exists():
        return inbox_path

    # Check with _report suffix
    inbox_path_alt = INBOX_REPORTS_DIR / f"{patient_id}_report.json"
    if inbox_path_alt.exists():
        return inbox_path_alt

    return None


def _find_volume_path(patient_id: str) -> Optional[Path]:
    """Find the volume file for a patient."""
    inbox_path = INBOX_VOLUMES_DIR / f"{patient_id}.nii.gz"
    if inbox_path.exists():
        return inbox_path

    inbox_path_alt = INBOX_VOLUMES_DIR / f"{patient_id}.nii"
    if inbox_path_alt.exists():
        return inbox_path_alt

    return None


@router.get("/{patient_id}/triage", response_model=TriageResult)
async def get_patient_triage(patient_id: str):
    """Get full triage result for a patient."""
    try:
        result = load_triage_result(patient_id, OUTPUT_DIR)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Triage result not found")

    priority_level = result.get("priority_level", 3)

    return TriageResult(
        patient_id=result["patient_id"],
        priority_level=priority_level,
        priority_name=PRIORITY_NAMES.get(priority_level, "UNKNOWN"),
        priority_color=PRIORITY_COLORS.get(priority_level, "#6B7280"),
        rationale=result.get("rationale", ""),
        key_slice_index=result.get("key_slice_index", 0),
        key_slice_thumbnail=result.get("key_slice_thumbnail", ""),
        processed_at=result.get("processed_at", ""),
        conditions_considered=result.get("conditions_considered", []),
        findings_summary=result.get("findings_summary", ""),
        visual_findings=result.get("visual_findings", ""),
        delta_analysis=result.get("delta_analysis", []),
        phase1_raw=result.get("phase1_raw", ""),
        phase2_raw=result.get("phase2_raw", ""),
        headline=result.get("headline", ""),
        reasoning=result.get("reasoning", ""),
    )


@router.get("/{patient_id}/fhir", response_model=PatientFHIRContext)
async def get_patient_fhir(patient_id: str):
    """Get FHIR context for a patient."""
    report_path = _find_report_path(patient_id)

    if not report_path:
        raise HTTPException(status_code=404, detail="Patient report not found")

    # Load FHIR bundle and process with FHIRJanitor
    try:
        with open(report_path, "r") as f:
            fhir_bundle = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load report: {str(e)}")

    janitor = FHIRJanitor()
    clinical_stream = janitor.process_bundle(fhir_bundle)

    # Build conditions with risk factor flag
    risk_factors_set = set(clinical_stream.risk_factors)
    conditions = [
        PatientCondition(
            name=cond,
            is_risk_factor=cond in risk_factors_set
        )
        for cond in clinical_stream.conditions
    ]

    return PatientFHIRContext(
        patient_id=patient_id,
        demographics=PatientDemographics(
            patient_id=patient_id,
            age=clinical_stream.age,
            gender=clinical_stream.gender,
        ),
        conditions=conditions,
        medications=clinical_stream.medications,
        risk_factors=clinical_stream.risk_factors,
        findings=clinical_stream.findings,
        impressions=clinical_stream.impressions,
    )


@router.get("/{patient_id}/slices/{slice_index}")
async def get_patient_slice(patient_id: str, slice_index: int):
    """Get a specific CT slice image for a patient."""
    volume_path = _find_volume_path(patient_id)

    if not volume_path:
        raise HTTPException(status_code=404, detail="Patient volume not found")

    try:
        # Load volume (cached) for bounds check
        volume, metadata = _get_cached_volume(volume_path)

        # Check slice index bounds
        total_slices = volume.shape[2]
        if slice_index < 0 or slice_index >= total_slices:
            raise HTTPException(
                status_code=400,
                detail=f"Slice index out of range. Valid range: 0-{total_slices - 1}"
            )

        # Get JPEG bytes from LRU cache (or encode + cache on miss)
        jpeg_bytes = _encode_slice_jpeg(str(volume_path.resolve()), slice_index)

        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={
                "X-Total-Slices": str(total_slices),
                "X-Current-Slice": str(slice_index),
                "Cache-Control": "private, max-age=3600, immutable",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process slice: {str(e)}")


@router.get("/{patient_id}/volume-info")
async def get_patient_volume_info(patient_id: str):
    """Get volume metadata for a patient."""
    volume_path = _find_volume_path(patient_id)

    if not volume_path:
        raise HTTPException(status_code=404, detail="Patient volume not found")

    try:
        volume, metadata = _get_cached_volume(volume_path)

        return {
            "patient_id": patient_id,
            "total_slices": volume.shape[2],
            "dimensions": list(volume.shape),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load volume: {str(e)}")
