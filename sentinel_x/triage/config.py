"""Configuration constants for the Sentinel-X Triage Agent."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
INBOX_DIR = BASE_DIR / "inbox"
INBOX_VOLUMES_DIR = INBOX_DIR / "volumes"
INBOX_REPORTS_DIR = INBOX_DIR / "reports"
OUTPUT_DIR = BASE_DIR / "output" / "triage_results"
LOG_DIR = BASE_DIR / "logs"

# Combined folder paths (unified structure for all patient data)
DATA_DIR = BASE_DIR / "data" / "raw_ct_rate"
COMBINED_DIR = DATA_DIR / "combined"
COMBINED_MANIFEST = COMBINED_DIR / "manifest.json"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# MedGemma model configuration (legacy — kept for backward compat)
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MEDGEMMA_DTYPE = "bfloat16"

# Phase 1: Vision model (MedGemma 1.5 4B — trained on 3D CT volumes)
VISION_MODEL_ID = "google/medgemma-1.5-4b-it"
VISION_MODEL_DTYPE = "bfloat16"  # Full precision for visual fidelity

# Phase 2: Reasoning model (MedGemma 27B text-only — best text reasoning)
# Uses Unsloth's pre-quantized BnB 4-bit version (16.6GB download vs 54GB full-precision).
# The text-only variant is ideal for Phase 2 since it receives no images.
REASONER_MODEL_ID = "unsloth/medgemma-27b-text-it-unsloth-bnb-4bit"
REASONER_QUANTIZATION = "nf4"  # Pre-quantized NF4 4-bit
REASONER_USE_DOUBLE_QUANT = True  # Already applied in pre-quantized weights

# CT processing configuration
CT_NUM_SLICES = 85  # Number of slices to sample from volume
CT_WINDOW_CENTER = 40  # Soft tissue window center (HU) — legacy, kept for compat
CT_WINDOW_WIDTH = 400  # Soft tissue window width (HU) — legacy, kept for compat

# CT 3-channel windowing (EXACT values from Google's official CT notebook)
CT_WINDOW_WIDE = (-1024, 1024)   # R channel: full HU range (air to bone)
CT_WINDOW_SOFT = (-135, 215)     # G channel: soft tissue (fat to start of bone)
CT_WINDOW_BRAIN = (0, 80)        # B channel: brain (water to brain density)

# Inbox watcher configuration
INBOX_POLL_INTERVAL = 5  # Seconds between inbox scans

# Priority levels
PRIORITY_CRITICAL = 1  # Acute pathology (PE, aortic dissection, pneumothorax)
PRIORITY_HIGH_RISK = 2  # Contextual mismatch (nodule + cancer history)
PRIORITY_ROUTINE = 3  # No acute/contextual flags

PRIORITY_NAMES = {
    PRIORITY_CRITICAL: "CRITICAL",
    PRIORITY_HIGH_RISK: "HIGH RISK",
    PRIORITY_ROUTINE: "ROUTINE",
}

# High-risk conditions for contextual analysis
HIGH_RISK_CONDITIONS = {
    "cancer",
    "malignancy",
    "carcinoma",
    "tumor",
    "neoplasm",
    "diabetes",
    "diabetic",
    "copd",
    "pulmonary disease",
    "heart disease",
    "cardiac",
    "hypertension",
    "immunocompromised",
    "immunosuppressed",
}

# Acute pathology keywords
ACUTE_PATHOLOGY_KEYWORDS = {
    "pulmonary embolism",
    "pe",
    "aortic dissection",
    "dissection",
    "pneumothorax",
    "tension pneumothorax",
    "hemorrhage",
    "acute bleeding",
    "rupture",
    "perforation",
    "obstruction",
    "infarction",
    "stroke",
}

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "triage.log"

# Session-based logging configuration
LOG_SESSIONS_DIR = LOG_DIR / "sessions"
LOG_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FHIR Janitor Configuration
# =============================================================================

# Resource types to discard completely (noise resources)
JANITOR_DISCARD_RESOURCES = {
    "Provenance",
    "Organization",
    "PractitionerRole",
    "Coverage",
    "Device",
}

# Resource types to process conditionally (extract hidden diagnoses, then discard)
JANITOR_CONDITIONAL_RESOURCES = {"Claim", "ExplanationOfBenefit"}

# Label for entries without dates
JANITOR_UNDATED_LABEL = "[Historical/Undated]"

# Maximum length for narrative sections (findings, impressions)
JANITOR_MAX_NARRATIVE_LENGTH = 500

# Target maximum tokens for the entire clinical stream
JANITOR_TARGET_MAX_TOKENS = 16000
