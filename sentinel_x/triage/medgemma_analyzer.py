"""MedGemma model interface for CT triage analysis."""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import MEDGEMMA_MODEL_ID, PRIORITY_CRITICAL, PRIORITY_HIGH_RISK, PRIORITY_ROUTINE
from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Structured result from MedGemma analysis."""
    visual_findings: str
    key_slice_index: int
    priority_level: int
    priority_rationale: str
    findings_summary: str
    conditions_considered: List[str]
    raw_response: str


class MedGemmaAnalyzer:
    """MedGemma model wrapper for CT triage analysis."""

    def __init__(self, model_id: str = MEDGEMMA_MODEL_ID):
        """Initialize the analyzer.

        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the MedGemma model and processor."""
        if self._loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading MedGemma model: {self.model_id}")

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self._loaded = True
        logger.info("MedGemma model loaded successfully")

    def _build_messages(
        self,
        images: List[Image.Image],
        context_text: str
    ) -> List[dict]:
        """Build message format for MedGemma.

        Args:
            images: List of CT slice images
            context_text: Formatted clinical context

        Returns:
            Messages in chat format
        """
        # Build content list with images and text
        content = []

        # Add all images first
        for _ in images:
            content.append({"type": "image"})

        # Add the user prompt
        user_prompt = build_user_prompt(context_text, len(images))
        content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]

        return messages

    def _parse_response(self, response: str) -> AnalysisResult:
        """Parse structured response from MedGemma.

        Args:
            response: Raw model response text

        Returns:
            Parsed AnalysisResult
        """
        # Default values
        visual_findings = ""
        key_slice_index = 0
        priority_level = PRIORITY_ROUTINE
        priority_rationale = ""
        findings_summary = ""
        conditions_considered = []

        # Parse VISUAL_FINDINGS
        match = re.search(r"VISUAL_FINDINGS:\s*(.+?)(?=\n\w+:|$)", response, re.DOTALL)
        if match:
            visual_findings = match.group(1).strip()

        # Parse KEY_SLICE
        match = re.search(r"KEY_SLICE:\s*(\d+)", response)
        if match:
            key_slice_index = int(match.group(1))

        # Parse PRIORITY_LEVEL
        match = re.search(r"PRIORITY_LEVEL:\s*(\d+)", response)
        if match:
            level = int(match.group(1))
            if level in (1, 2, 3):
                priority_level = level

        # Parse PRIORITY_RATIONALE
        match = re.search(r"PRIORITY_RATIONALE:\s*(.+?)(?=\n\w+:|$)", response, re.DOTALL)
        if match:
            priority_rationale = match.group(1).strip()

        # Parse FINDINGS_SUMMARY
        match = re.search(r"FINDINGS_SUMMARY:\s*(.+?)(?=\n\w+:|$)", response, re.DOTALL)
        if match:
            findings_summary = match.group(1).strip()

        # Parse CONDITIONS_CONSIDERED
        match = re.search(r"CONDITIONS_CONSIDERED:\s*(.+?)(?=\n\w+:|$)", response, re.DOTALL)
        if match:
            conditions_text = match.group(1).strip()
            conditions_considered = [c.strip() for c in conditions_text.split(",") if c.strip()]

        return AnalysisResult(
            visual_findings=visual_findings,
            key_slice_index=key_slice_index,
            priority_level=priority_level,
            priority_rationale=priority_rationale,
            findings_summary=findings_summary,
            conditions_considered=conditions_considered,
            raw_response=response,
        )

    def analyze(
        self,
        images: List[Image.Image],
        context_text: str,
        max_new_tokens: int = 1024,
    ) -> AnalysisResult:
        """Analyze CT images with clinical context.

        Args:
            images: List of CT slice images
            context_text: Formatted clinical context
            max_new_tokens: Maximum tokens to generate

        Returns:
            Structured analysis result
        """
        if not self._loaded:
            self.load_model()

        logger.info(f"Analyzing {len(images)} CT slices")

        # Build messages
        messages = self._build_messages(images, context_text)

        # Process inputs - apply_chat_template returns text, then process with images
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Now tokenize with the processor, passing images separately
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Generate response
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode response
        response = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )

        logger.info("Analysis complete, parsing response")

        # Parse and return
        result = self._parse_response(response)
        return result

    def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        if self._loaded:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._loaded = False
            torch.cuda.empty_cache()
            logger.info("Model unloaded")
