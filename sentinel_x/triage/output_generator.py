"""Output generation for triage results including JSON and thumbnails.

Supports the Serial Late Fusion pipeline: merges Phase 1 VisualFactSheet
and Phase 2 DeltaAnalysisResult into a single triage_result.json while
maintaining backward-compatible fields.
"""

import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .config import OUTPUT_DIR
from .ct_processor import get_thumbnail
from .medgemma_analyzer import VisualFactSheet
from .medgemma_reasoner import DeltaAnalysisResult

logger = logging.getLogger(__name__)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL Image as a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _get_key_slice_index(
    visual_fact_sheet: VisualFactSheet,
    delta_result: DeltaAnalysisResult,
    num_images: int,
) -> int:
    """Pick the most diagnostically important slice from Phase 1 findings."""
    if visual_fact_sheet.findings:
        idx = visual_fact_sheet.findings[0].slice_index
        if 0 <= idx < num_images:
            return idx
    return num_images // 2


def generate_triage_result(
    patient_id: str,
    visual_fact_sheet: VisualFactSheet,
    delta_result: DeltaAnalysisResult,
    images: List[Image.Image],
    slice_indices: List[int],
    conditions_from_context: List[str],
) -> Dict[str, Any]:
    """Generate the final triage result JSON from both pipeline phases.

    Args:
        patient_id: Patient identifier
        visual_fact_sheet: Phase 1 output
        delta_result: Phase 2 output
        images: CT slice PIL Images
        slice_indices: Original volume slice indices
        conditions_from_context: Conditions extracted from FHIR

    Returns:
        Dict ready to be serialized as triage_result.json
    """
    # Determine key slice from highest-priority finding
    key_slice_idx = _get_key_slice_index(visual_fact_sheet, delta_result, len(images))

    # Generate thumbnail
    key_image = images[key_slice_idx] if key_slice_idx < len(images) else images[len(images) // 2]
    thumbnail = get_thumbnail(key_image)
    thumbnail_base64 = image_to_base64(thumbnail)

    # Map sampled index to original volume index
    original_slice_index = (
        slice_indices[key_slice_idx]
        if key_slice_idx < len(slice_indices)
        else key_slice_idx
    )

    # Build visual findings text from fact sheet
    def _format_finding(f):
        parts = [f.finding]
        if f.location and f.location != "unspecified":
            parts.append(f"in {f.location}")
        if f.size:
            parts.append(f"({f.size})")
        parts.append(f"\u2014 {f.description}")
        return " ".join(parts)

    visual_findings_text = "; ".join(
        _format_finding(f) for f in visual_fact_sheet.findings
    ) or "No abnormalities detected"

    # Build rationale combining both phases
    rationale = f"Visual analysis: {visual_findings_text}"
    if conditions_from_context:
        rationale += f" EHR Context: Patient has {', '.join(conditions_from_context)}."
    rationale += f" Delta: {delta_result.headline}"

    # Build delta_analysis serializable list
    delta_analysis_list = [
        {
            "finding": de.finding,
            "classification": de.classification,
            "priority": de.priority,
            "history_match": de.history_match,
            "reasoning": de.reasoning,
        }
        for de in delta_result.delta_analysis
    ]

    result = {
        # Core fields (backward compatible)
        "patient_id": patient_id,
        "priority_level": delta_result.overall_priority,
        "rationale": rationale,
        "key_slice_index": original_slice_index,
        "key_slice_thumbnail": thumbnail_base64,
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "conditions_considered": conditions_from_context,
        "findings_summary": delta_result.findings_summary,
        "visual_findings": visual_findings_text,
        # New Serial Late Fusion fields
        "delta_analysis": delta_analysis_list,
        "phase1_raw": visual_fact_sheet.raw_response,
        "phase2_raw": delta_result.raw_response,
        "headline": delta_result.headline,
        "reasoning": delta_result.priority_rationale,
    }

    return result


def save_triage_result(
    patient_id: str, result: Dict[str, Any], output_dir: Path = OUTPUT_DIR
) -> Path:
    """Save triage result JSON to disk."""
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    result_path = patient_dir / "triage_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved triage result: {result_path}")
    return result_path


def load_triage_result(
    patient_id: str, output_dir: Path = OUTPUT_DIR
) -> Dict[str, Any]:
    """Load a saved triage result from disk."""
    result_path = output_dir / patient_id / "triage_result.json"
    with open(result_path, "r") as f:
        return json.load(f)
