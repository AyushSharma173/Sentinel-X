"""Phase 1 Vision Analyzer — MedGemma 1.5 4B for unbiased visual detection.

Loads the 4B multimodal model in BFloat16 (full precision for visual fidelity),
sends CT slice images WITHOUT any clinical context, and extracts a structured
VisualFactSheet of anatomical findings.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import List

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import VISION_MODEL_ID
from .json_repair import parse_json_safely
from .prompts import PHASE1_SYSTEM_PROMPT, build_phase1_user_prompt
from .vram_manager import (
    VRAM_MIN_FREE_PHASE1_MB,
    assert_vram_available,
    log_vram_status,
    unload_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VisualFinding:
    """A single anatomical finding detected in the CT images."""
    finding: str
    location: str
    size: str
    slice_index: int
    description: str


@dataclass
class VisualFactSheet:
    """Structured output from Phase 1 vision analysis."""
    findings: List[VisualFinding] = field(default_factory=list)
    raw_response: str = ""
    num_slices_analyzed: int = 0

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict for passing to Phase 2."""
        return {
            "findings": [asdict(f) for f in self.findings],
            "num_slices_analyzed": self.num_slices_analyzed,
        }

    def to_json(self) -> str:
        """Serialize to JSON string for Phase 2 prompt injection."""
        return json.dumps(self.to_dict(), indent=2)


# Legacy alias for backward compatibility
@dataclass
class AnalysisResult:
    """Legacy result structure — kept for backward compatibility."""
    visual_findings: str = ""
    key_slice_index: int = 0
    priority_level: int = 3
    priority_rationale: str = ""
    findings_summary: str = ""
    conditions_considered: List[str] = field(default_factory=list)
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Vision Analyzer (Phase 1)
# ---------------------------------------------------------------------------

class VisionAnalyzer:
    """Phase 1: MedGemma 1.5 4B vision-only analyzer for CT images.

    Loads the model in BFloat16 without quantization for maximum visual
    fidelity. Sends images WITHOUT clinical context to prevent bias.
    Outputs a structured VisualFactSheet.
    """

    def __init__(self, model_id: str = VISION_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the MedGemma 1.5 4B model and processor in BFloat16."""
        if self._loaded:
            logger.info("Vision model already loaded")
            return

        logger.info(f"Loading vision model: {self.model_id}")
        assert_vram_available(VRAM_MIN_FREE_PHASE1_MB, "Phase 1 vision load")
        log_vram_status("before vision model load")

        try:
            import accelerate  # noqa: F401
        except ImportError:
            raise ImportError(
                "The accelerate library is required. "
                "Install with: pip install accelerate>=0.9.0"
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._loaded = True
        log_vram_status("after vision model load")
        logger.info("Vision model loaded successfully")

    def _build_messages(self, images: List[Image.Image]) -> List[dict]:
        """Build message format matching Google's official CT notebook.

        Format: single 'user' message with instruction text, then interleaved
        [image, "SLICE N"] pairs, then the query. No 'system' role.
        No clinical context — this prevents cognitive bias.
        """
        content = []

        # Instruction at the start (acts as system prompt within user message)
        content.append({"type": "text", "text": PHASE1_SYSTEM_PROMPT})

        # Interleaved images with slice labels (Google's CT notebook format)
        for i, image in enumerate(images, 1):
            content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": f"SLICE {i}"})

        # Query at the end
        user_prompt = build_phase1_user_prompt(len(images))
        content.append({"type": "text", "text": user_prompt})

        messages = [{"role": "user", "content": content}]
        return messages

    def _parse_response(self, response: str, num_slices: int) -> VisualFactSheet:
        """Parse JSON response into a VisualFactSheet."""
        parsed = parse_json_safely(response)

        if parsed is None:
            logger.warning("Failed to parse Phase 1 response as JSON, returning empty fact sheet")
            return VisualFactSheet(
                findings=[],
                raw_response=response,
                num_slices_analyzed=num_slices,
            )

        findings = []
        for item in parsed.get("findings", []):
            try:
                findings.append(VisualFinding(
                    finding=str(item.get("finding", "")),
                    location=str(item.get("location", "")),
                    size=str(item.get("size", "")),
                    slice_index=int(item.get("slice_index", 0)),
                    description=str(item.get("description", "")),
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed finding: {e}")

        return VisualFactSheet(
            findings=findings,
            raw_response=response,
            num_slices_analyzed=num_slices,
        )

    def analyze(self, images: List[Image.Image], max_new_tokens: int = 1024) -> VisualFactSheet:
        """Run Phase 1 vision analysis on CT images (no clinical context).

        Args:
            images: List of PIL Images (multi-window RGB encoded CT slices)
            max_new_tokens: Maximum generation length

        Returns:
            VisualFactSheet with detected findings
        """
        if not self._loaded:
            self.load_model()

        logger.info(f"Phase 1: Analyzing {len(images)} CT slices (vision-only)")
        messages = self._build_messages(images)

        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the generated portion (skip input tokens)
        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        del inputs
        del outputs
        torch.cuda.empty_cache()

        logger.info("Phase 1 analysis complete, parsing response")
        return self._parse_response(response, len(images))

    def unload(self) -> None:
        """Unload vision model and free GPU memory."""
        if self._loaded:
            logger.info("Unloading vision model...")
            unload_model(self.model, self.processor)
            self.model = None
            self.processor = None
            self._loaded = False
            logger.info("Vision model unloaded")

    # Legacy compatibility
    def unload_model(self) -> None:
        """Legacy method name — delegates to unload()."""
        self.unload()


# Legacy alias so existing imports still work
MedGemmaAnalyzer = VisionAnalyzer
