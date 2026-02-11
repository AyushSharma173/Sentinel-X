"""Prompt templates for the Serial Late Fusion triage pipeline.

Phase 1: Vision-only detection (4B model) — unbiased visual sensor
Phase 2: Clinical reasoning (27B model) — delta analysis against EHR
"""

# =============================================================================
# PHASE 1: VISION — "Unbiased Sensor" (MedGemma 1.5 4B)
# =============================================================================

PHASE1_SYSTEM_PROMPT = """You are an expert radiologist analyzing a volumetric Chest CT. Your goal is to detect ALL abnormalities, no matter how subtle. Assume the scan contains pathology until proven otherwise."""


PHASE1_USER_PROMPT_TEMPLATE = (
    "Please review these {num_slices} axial chest CT slices (ordered Top to Bottom).\n"
    "\n"
    "Perform a systematic 'Visual Inventory' by describing the appearance of these regions:\n"
    "1. VASCULAR SYSTEM: Describe the caliber and patency of the aorta and subclavian veins. Note any calcifications.\n"
    "2. LUNGS (AIRWAYS): Describe the bronchial walls. Are they normal, thickened, or dilated?\n"
    "3. LUNGS (PARENCHYMA): Describe any opacities. specifically looking for consolidation, atelectasis, or nodules.\n"
    "4. ABDOMEN: Assess the liver, gallbladder, and kidneys. Describe their size and contour.\n"
    "5. BONES: Describe the thoracic spine alignment and bone quality.\n"
    "\n"
    "Only AFTER describing these, provide your FINAL IMPRESSION."
)


def build_phase1_user_prompt(num_slices: int) -> str:
    """Build Phase 1 user prompt (no clinical context)."""
    return PHASE1_USER_PROMPT_TEMPLATE.format(num_slices=num_slices)

# =============================================================================
# PHASE 2: REASONING — "Delta Analyst" (MedGemma 27B)
# =============================================================================

import re

PHASE2_SYSTEM_PROMPT = """You are an expert Clinical Triage AI.
Your goal is to identify ACUTE RISKS by comparing new Visual Findings against Clinical History.

### INPUTS
1. CLINICAL HISTORY (Condensed)
2. VISUAL FINDINGS (CT Report)

### PROTOCOL
Perform a "Delta Analysis" in two parts:

PART 1: CLINICAL REASONING (The "Why")
- **FILTER:** Ignore findings described as "normal", "unremarkable", "clear", or "patent". Focus ONLY on pathology.
- **COMPARE:** For each abnormality, check the history. Is it NEW (Acute) or STABLE (Chronic)?
- **CONTEXTUALIZE:** Explain *why* a finding is risky given the patient's specific history (e.g. "New effusion in patient with recent heart surgery").

PART 2: FINAL TRIAGE (The "What")
- At the very bottom, output a strict summary block.
- **PRIORITY 1 (CRITICAL):** Life-threatening (PE, Pneumothorax, Aortic Dissection) OR New Mass.
- **PRIORITY 2 (URGENT):** New acute pathology (Pneumonia, Effusion, Fracture) requiring intervention.
- **PRIORITY 3 (ROUTINE):** Stable chronic disease or Normal scan.

### OUTPUT FORMAT
[Write your reasoning here in clear, concise bullet points.]

---
TRIAGE SUMMARY
PRIORITY: [1, 2, or 3]
HEADLINE: [5-10 word summary of the primary risk]
"""

PHASE2_USER_PROMPT_TEMPLATE = """## CLINICAL HISTORY

{clinical_narrative}

## VISUAL FINDINGS FROM CT SCAN

{visual_narrative}

Perform Delta Analysis. Compare abnormalities against history and assign a Priority."""

# Phase 2 narrative truncation — safety net. Smart compression targets ~800 tokens;
# this limit only activates for edge cases with unusually large histories.
PHASE2_MAX_NARRATIVE_CHARS = 16_000


def build_phase2_user_prompt(clinical_narrative: str, visual_narrative: str) -> str:
    """Build Phase 2 user prompt with clinical context and visual findings."""
    if len(clinical_narrative) > PHASE2_MAX_NARRATIVE_CHARS:
        clinical_narrative = (
            clinical_narrative[:PHASE2_MAX_NARRATIVE_CHARS]
            + "\n\n[... clinical history truncated for context window ...]"
        )
    return PHASE2_USER_PROMPT_TEMPLATE.format(
        clinical_narrative=clinical_narrative,
        visual_narrative=visual_narrative,
    )


def parse_phase2_response(response_text: str) -> dict:
    """
    Parses the text-based output from Phase 2 into structured data.
    Extracts PRIORITY and HEADLINE using Regex.
    """
    # 1. Default Safety Net
    result = {
        "priority": 3,  # Default to Routine if parsing fails
        "headline": "Assessment Pending",
        "reasoning": response_text.strip()
    }

    # 2. Extract Priority (Looking for "PRIORITY: X")
    # We look for the last occurrence to avoid false matches in the reasoning text
    priority_match = re.findall(r"PRIORITY:\s*(\d)", response_text)
    if priority_match:
        try:
            result["priority"] = int(priority_match[-1])
        except ValueError:
            pass

    # 3. Extract Headline (Looking for "HEADLINE: ...")
    headline_match = re.search(r"HEADLINE:\s*(.+)", response_text)
    if headline_match:
        result["headline"] = headline_match.group(1).strip()

    # 4. Clean the Reasoning (Separate the thinking from the tags)
    if "TRIAGE SUMMARY" in response_text:
        # Keep only the text BEFORE the summary block for the UI "Reasoning" tab
        result["reasoning"] = response_text.split("TRIAGE SUMMARY")[0].strip()

    return result


# =============================================================================
# Legacy prompts (kept for backward compatibility)
# =============================================================================

SYSTEM_PROMPT = PHASE1_SYSTEM_PROMPT

USER_PROMPT_TEMPLATE = (
    "Analyze the following chest CT scan and clinical information for triage "
    "prioritization.\n\n{context}\n\nPlease examine all {num_slices} CT slices "
    "provided and deliver your structured analysis."
)


def build_user_prompt(context_text: str, num_slices: int) -> str:
    """Build user prompt (legacy — kept for backward compatibility)."""
    return USER_PROMPT_TEMPLATE.format(context=context_text, num_slices=num_slices)
