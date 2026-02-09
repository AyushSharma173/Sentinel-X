"""Phase 2 Clinical Reasoner — MedGemma 27B text-only for delta analysis.

Uses Unsloth's pre-quantized BnB NF4 4-bit version of medgemma-27b-text-it
(~16.6GB download, ~13-14GB VRAM). This is the text-only variant — ideal for
Phase 2 since it receives no images, only the FHIR narrative + Phase 1 fact sheet.

Receives the FHIR clinical narrative + Phase 1 VisualFactSheet and performs
Delta Analysis to classify each finding as CHRONIC_STABLE, ACUTE_NEW, or DISCORDANT.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import REASONER_MODEL_ID, REASONER_USE_DOUBLE_QUANT, PRIORITY_ROUTINE
from .json_repair import parse_json_safely
from .prompts import PHASE2_SYSTEM_PROMPT, build_phase2_user_prompt
from .vram_manager import (
    VRAM_MIN_FREE_PHASE2_MB,
    assert_vram_available,
    log_vram_status,
    unload_model,
)

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

    Uses Unsloth's pre-quantized BnB NF4 4-bit model — weights are already
    quantized on disk (16.6GB download), so no BitsAndBytesConfig is needed
    at load time. This avoids downloading the full 54GB BF16 weights.

    Receives clinical narrative + visual fact sheet (text only, no images)
    and performs Delta Analysis.
    """

    def __init__(self, model_id: str = REASONER_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self._loaded = False

    @property
    def processor(self):
        """Legacy alias — returns tokenizer for backward compatibility."""
        return self.tokenizer

    def load_model(self) -> None:
        """Load pre-quantized 27B text-only model."""
        if self._loaded:
            logger.info("Reasoner model already loaded")
            return

        logger.info(f"Loading reasoner model: {self.model_id}")
        assert_vram_available(VRAM_MIN_FREE_PHASE2_MB, "Phase 2 reasoner load")
        log_vram_status("before reasoner model load")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Cap GPU allocation to 20GB — leaves ~4GB headroom for KV cache
        # and activations, preventing OOM on 24GB cards.
        max_mem = {0: "20GiB", "cpu": "32GiB"}

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            max_memory=max_mem,
        )
        self._loaded = True
        log_vram_status("after reasoner model load")
        logger.info("Reasoner model loaded successfully")

    def _build_messages(
        self, clinical_narrative: str, visual_narrative: str
    ) -> List[dict]:
        """Build text-only messages for Phase 2 Delta Analysis.

        Uses system + user role format with plain string content
        (text-only model, no multimodal content dicts needed).
        """
        user_prompt = build_phase2_user_prompt(
            clinical_narrative, visual_narrative
        )

        messages = [
            {"role": "system", "content": PHASE2_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
        visual_narrative: str,
        max_new_tokens: int = 512,
    ) -> DeltaAnalysisResult:
        """Run Phase 2 Delta Analysis: compare visual findings against clinical history.

        Args:
            clinical_narrative: Full FHIR clinical stream text
            visual_narrative: Raw narrative from Phase 1 vision analysis
            max_new_tokens: Maximum generation length

        Returns:
            DeltaAnalysisResult with classified findings and overall priority
        """
        if not self._loaded:
            self.load_model()

        logger.info(
            "Phase 2: Running delta analysis (text-only) | "
            f"narrative_chars={len(clinical_narrative)}, "
            f"truncated={len(clinical_narrative) > 12_000}"
        )
        messages = self._build_messages(clinical_narrative, visual_narrative)

        # Tokenize (text-only, no images)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        logger.info(
            f"Phase 2 input budget: {input_len} tokens | "
            f"max_new_tokens={max_new_tokens} | "
            f"total_budget={input_len + max_new_tokens}"
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        response = self.tokenizer.decode(
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
            unload_model(self.model, self.tokenizer)
            self.model = None
            self.tokenizer = None
            self._loaded = False
            logger.info("Reasoner model unloaded")
