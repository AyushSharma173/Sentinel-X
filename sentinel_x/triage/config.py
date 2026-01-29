"""Configuration constants for the Sentinel-X Triage Agent."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
INBOX_DIR = BASE_DIR / "inbox"
INBOX_VOLUMES_DIR = INBOX_DIR / "volumes"
INBOX_REPORTS_DIR = INBOX_DIR / "reports"
OUTPUT_DIR = BASE_DIR / "output" / "triage_results"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# MedGemma model configuration
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MEDGEMMA_DTYPE = "bfloat16"

# CT processing configuration
CT_NUM_SLICES = 85  # Number of slices to sample from volume
CT_WINDOW_CENTER = 40  # Soft tissue window center (HU)
CT_WINDOW_WIDTH = 400  # Soft tissue window width (HU)

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
