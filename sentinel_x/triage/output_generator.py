"""Output generation for triage results including JSON and thumbnails."""

import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .config import OUTPUT_DIR
from .ct_processor import get_thumbnail
from .medgemma_analyzer import AnalysisResult

logger = logging.getLogger(__name__)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string.

    Args:
        image: PIL Image
        format: Image format (PNG, JPEG)

    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generate_triage_result(
    patient_id: str,
    analysis: AnalysisResult,
    images: List[Image.Image],
    slice_indices: List[int],
    conditions_from_context: List[str],
) -> Dict[str, Any]:
    """Generate triage result JSON structure.

    Args:
        patient_id: Patient identifier
        analysis: MedGemma analysis result
        images: List of CT slice images
        slice_indices: Original slice indices from volume
        conditions_from_context: Conditions from FHIR context

    Returns:
        Triage result dictionary
    """
    # Get the key slice image
    key_slice_idx = analysis.key_slice_index
    if key_slice_idx >= len(images):
        key_slice_idx = len(images) // 2  # Default to middle slice

    key_image = images[key_slice_idx]
    thumbnail = get_thumbnail(key_image)
    thumbnail_base64 = image_to_base64(thumbnail)

    # Map sampled index to original slice index
    original_slice_index = slice_indices[key_slice_idx] if key_slice_idx < len(slice_indices) else key_slice_idx

    # Build rationale combining visual and context
    rationale = f"Visual analysis: {analysis.visual_findings}"
    if conditions_from_context:
        rationale += f" EHR Context: Patient has {', '.join(conditions_from_context)}."
    rationale += f" {analysis.priority_rationale}"

    result = {
        "patient_id": patient_id,
        "priority_level": analysis.priority_level,
        "rationale": rationale,
        "key_slice_index": original_slice_index,
        "key_slice_thumbnail": thumbnail_base64,
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "conditions_considered": analysis.conditions_considered,
        "findings_summary": analysis.findings_summary,
        "visual_findings": analysis.visual_findings,
    }

    return result


def save_triage_result(
    patient_id: str,
    result: Dict[str, Any],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save triage result to patient-specific directory.

    Args:
        patient_id: Patient identifier
        result: Triage result dictionary
        output_dir: Base output directory

    Returns:
        Path to saved result file
    """
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    result_path = patient_dir / "triage_result.json"

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved triage result: {result_path}")
    return result_path


def load_triage_result(patient_id: str, output_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    """Load triage result for a patient.

    Args:
        patient_id: Patient identifier
        output_dir: Base output directory

    Returns:
        Triage result dictionary
    """
    result_path = output_dir / patient_id / "triage_result.json"

    with open(result_path, "r") as f:
        return json.load(f)
