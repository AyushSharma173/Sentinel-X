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
JANITOR_TARGET_MAX_TOKENS = 4000

# =============================================================================
# Smart Compression Configuration (Stages 1-4)
# =============================================================================

# Stage 1: SNOMED codes to DROP (social, admin, dental, reproductive)
DROP_SNOMED_CODES = {
    "73595000",   # Stress (finding)
    "160903007",  # Full-time employment
    "160904001",  # Part-time employment
    "224299000",  # Received higher education
    "423315002",  # Limited social contact
    "446654005",  # Refugee
    "266948004",  # Has a criminal record
    "706893006",  # Victim of intimate partner abuse
    "314529007",  # Medication review due
    "267020005",  # History of tubal ligation
    "80583007",   # Severe anxiety (panic) — screening-related
    "66383009",   # Gingivitis
    "31642005",   # Acute gingivitis
    "109573003",  # Dental plaque
    "80967001",   # Dental caries
    "2556008",    # Periodontal disease
    "718052004",  # Asymptomatic periapical periodontitis
    "422650009",  # Social isolation (finding)
    "19169002",   # Miscarriage in first trimester (not chest-CT relevant)
    "281647001",  # Adverse reaction caused by drug (handled by AllergyIntolerance)
    "1172608001",  # Accretion on tooth
    "161744009",  # Past pregnancy history of miscarriage
}

# Stage 1: ICD-10 prefixes to DROP (dental)
DROP_ICD10_PREFIXES = ("K00", "K02", "K03", "K04", "K05", "K08")

# Stage 1: Observation categories to DROP
DROP_OBSERVATION_CATEGORIES = {"survey"}

# Stage 1: Resource types to DROP entirely (beyond existing JANITOR_DISCARD_RESOURCES)
DROP_RESOURCE_TYPES = {"Immunization", "SupplyDelivery", "CarePlan", "CareTeam", "DocumentReference"}

# Stage 1: Procedure coding systems to DROP (dental)
DROP_PROCEDURE_SYSTEMS = {"http://www.ada.org/cdt"}

# Stage 2: Lab LOINC whitelist (code -> short display name)
LAB_WHITELIST_LOINC = {
    # CBC
    "6690-2": "WBC", "718-7": "Hgb", "4544-3": "Hct", "777-3": "Plt",
    # Renal
    "2160-0": "Cr", "3094-0": "BUN", "33914-3": "eGFR",
    # Coag/PE
    "48065-7": "D-dimer", "6598-7": "Troponin", "6301-6": "INR",
    # Inflammatory
    "1988-5": "CRP",
    # Metabolic
    "4548-4": "HbA1c", "2345-7": "Glucose", "2951-2": "Na", "2823-3": "K",
    # ABG
    "2744-1": "pH", "2019-8": "pCO2", "2703-7": "pO2", "1960-4": "HCO3",
    # PFT
    "19926-5": "FEV1/FVC", "20150-9": "FEV1",
    # Lipids (cardiac context)
    "2093-3": "Chol", "18262-6": "LDL",
}

# Stage 2: Vital signs LOINC codes (for latest-set extraction)
VITAL_SIGNS_LOINC = {
    "8480-6": "SBP", "8462-4": "DBP", "8867-4": "HR",
    "9279-1": "RR", "2708-6": "SpO2", "8310-5": "Temp",
    "39156-5": "BMI",
}

LAB_DELTA_THRESHOLD_PERCENT = 10

# Stage 3: Time decay half-lives (days)
TIME_DECAY_HALF_LIVES = {
    "active_condition": float("inf"),
    "procedure": 365,
    "resolved_condition": 730,
}

RELEVANCE_THRESHOLD = 0.1
