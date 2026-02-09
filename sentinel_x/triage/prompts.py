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

PHASE2_SYSTEM_PROMPT = """You are an expert clinical reasoning system performing triage Delta Analysis. You will receive two inputs:

1. CLINICAL HISTORY: The patient's full EHR timeline (conditions, medications, labs)
2. VISUAL FINDINGS: A radiologist's narrative report of findings detected in a new CT scan

YOUR TASK — DELTA ANALYSIS:
For each visual finding, compare it against the clinical history and classify:

- CHRONIC_STABLE (Priority 3): Finding matches a known condition documented >3 months ago with no significant change. Example: "Known 4mm RUL nodule" + history shows "pulmonary nodule noted 2024-01" = Chronic/Stable.

- ACUTE_NEW (Priority 1 or 2): Finding has NO corresponding entry in clinical history. This is potentially new pathology. Priority 1 if life-threatening (PE, dissection, pneumothorax, hemorrhage). Priority 2 otherwise.

- DISCORDANT (Priority 2): Clinical history suggests acute presentation but imaging shows no corresponding visual findings, OR vice versa. Warrants prompt review.

PRIORITY ESCALATION RULES:
- Any single Priority 1 finding → Overall Priority 1
- Any Priority 2 finding (with no Priority 1) → Overall Priority 2
- All findings Chronic/Stable → Overall Priority 3
- Empty visual findings + acute clinical presentation → Priority 2 (DISCORDANT)

Respond with ONLY a JSON object in this exact format:
{
  "delta_analysis": [
    {
      "finding": "description of visual finding",
      "classification": "CHRONIC_STABLE | ACUTE_NEW | DISCORDANT",
      "priority": 1|2|3,
      "history_match": "matching history entry or null",
      "reasoning": "brief explanation of classification"
    }
  ],
  "overall_priority": 1|2|3,
  "priority_rationale": "1-2 sentence explanation of the overall triage decision",
  "findings_summary": "Brief worklist-ready summary"
}"""


PHASE2_USER_PROMPT_TEMPLATE = """## CLINICAL HISTORY

{clinical_narrative}

## VISUAL FINDINGS FROM CT SCAN

{visual_narrative}

Perform Delta Analysis. Compare each visual finding against the clinical history. Classify every finding and determine the overall triage priority."""

# Phase 2 narrative truncation — safety net. Smart compression targets ~800 tokens;
# this limit only activates for edge cases with unusually large histories.
PHASE2_MAX_NARRATIVE_CHARS = 16_000  # Safety net — smart compression targets ~800 tokens


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
