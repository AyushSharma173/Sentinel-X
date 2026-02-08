"""Phase 2 Clinical Reasoner — MedGemma 27B for delta analysis.

Loads the 27B text model with NF4 4-bit quantization via BitsAndBytes
(~13-14GB VRAM with double quantization). Receives the FHIR clinical
narrative + Phase 1 VisualFactSheet and performs Delta Analysis to
classify each finding as CHRONIC_STABLE, ACUTE_NEW, or DISCORDANT.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import REASONER_MODEL_ID, REASONER_USE_DOUBLE_QUANT, PRIORITY_ROUTINE
from .json_repair import parse_json_safely
from .prompts import PHASE2_SYSTEM_PROMPT, build_phase2_user_prompt
from .vram_manager import log_vram_status, unload_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DeltaEntry:
    """A single finding classification from Delta Analysis."""
    finding: str
    classification: str  # CHRONIC_STABLE, ACUTE_NEW, DISCORDANT
    priority: int
    history_match: Optional[str]
    reasoning: str


@dataclass
class DeltaAnalysisResult:
    """Structured output from Phase 2 clinical reasoning."""
    delta_analysis: List[DeltaEntry] = field(default_factory=list)
    overall_priority: int = PRIORITY_ROUTINE
    priority_rationale: str = ""
    findings_summary: str = ""
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Clinical Reasoner (Phase 2)
# ---------------------------------------------------------------------------

class ClinicalReasoner:
    """Phase 2: MedGemma 27B text-only reasoner for delta analysis.

    Loads the model with NF4 4-bit quantization to fit within VRAM budget.
    Receives clinical narrative + visual fact sheet (text only, no images)
    and performs Delta Analysis.
    """

    def __init__(self, model_id: str = REASONER_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self) -> None:
        """Load 27B model with 4-bit NF4 quantization via BitsAndBytes."""
        if self._loaded:
            logger.info("Reasoner model already loaded")
            return

        logger.info(f"Loading reasoner model: {self.model_id}")
        log_vram_status("before reasoner model load")

        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=REASONER_USE_DOUBLE_QUANT,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
        self._loaded = True
        log_vram_status("after reasoner model load")
        logger.info("Reasoner model loaded successfully")

    def _build_messages(
        self, clinical_narrative: str, visual_fact_sheet_json: str
    ) -> List[dict]:
        """Build text-only messages for Phase 2 Delta Analysis.

        Uses system + user role format for the 27B-it model.
        No images — this is pure text reasoning.
        """
        user_prompt = build_phase2_user_prompt(
            clinical_narrative, visual_fact_sheet_json
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": PHASE2_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]
        return messages

    def _parse_response(self, response: str) -> DeltaAnalysisResult:
        """Parse JSON response into a DeltaAnalysisResult."""
        parsed = parse_json_safely(response)

        if parsed is None:
            logger.warning(
                "Failed to parse Phase 2 response as JSON, returning default result"
            )
            return DeltaAnalysisResult(
                delta_analysis=[],
                overall_priority=PRIORITY_ROUTINE,
                priority_rationale="Unable to parse model response",
                findings_summary="Analysis parsing failed — manual review recommended",
                raw_response=response,
            )

        delta_entries = []
        for item in parsed.get("delta_analysis", []):
            try:
                delta_entries.append(DeltaEntry(
                    finding=str(item.get("finding", "")),
                    classification=str(item.get("classification", "ACUTE_NEW")),
                    priority=int(item.get("priority", 2)),
                    history_match=item.get("history_match"),
                    reasoning=str(item.get("reasoning", "")),
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed delta entry: {e}")

        overall_priority = int(parsed.get("overall_priority", PRIORITY_ROUTINE))
        if overall_priority not in (1, 2, 3):
            overall_priority = PRIORITY_ROUTINE

        return DeltaAnalysisResult(
            delta_analysis=delta_entries,
            overall_priority=overall_priority,
            priority_rationale=str(parsed.get("priority_rationale", "")),
            findings_summary=str(parsed.get("findings_summary", "")),
            raw_response=response,
        )

    def analyze(
        self,
        clinical_narrative: str,
        visual_fact_sheet: dict,
        max_new_tokens: int = 1024,
    ) -> DeltaAnalysisResult:
        """Run Phase 2 Delta Analysis: compare visual findings against clinical history.

        Args:
            clinical_narrative: Full FHIR clinical stream text
            visual_fact_sheet: Dict from VisualFactSheet.to_dict()
            max_new_tokens: Maximum generation length

        Returns:
            DeltaAnalysisResult with classified findings and overall priority
        """
        if not self._loaded:
            self.load_model()

        visual_json = json.dumps(visual_fact_sheet, indent=2)
        logger.info("Phase 2: Running delta analysis (text-only)")
        messages = self._build_messages(clinical_narrative, visual_json)

        # Tokenize (text-only, no images)
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        del inputs
        del outputs
        torch.cuda.empty_cache()

        logger.info("Phase 2 analysis complete, parsing response")
        return self._parse_response(response)

    def unload(self) -> None:
        """Unload reasoner model and free GPU memory."""
        if self._loaded:
            logger.info("Unloading reasoner model...")
            unload_model(self.model, self.processor)
            self.model = None
            self.processor = None
            self._loaded = False
            logger.info("Reasoner model unloaded")
