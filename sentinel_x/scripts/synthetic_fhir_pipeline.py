#!/usr/bin/env python3
"""
Synthetic FHIR Generation Pipeline from Radiology Reports

This pipeline:
1. Extracts structured clinical data from radiology reports using OpenAI (gpt-4o)
2. Configures Synthea to generate matching patient profiles
3. Merges radiology DiagnosticReport into the FHIR bundle (US Core R4 compliant)
4. Outputs a unified folder structure with FHIR records and CT volumes together

Usage (run from workspace root):
    # Process all data in raw_ct_rate, output to raw_ct_rate/combined/ (symlinks by default)
    python sentinel_x/scripts/synthetic_fhir_pipeline.py

    # Copy volumes instead of symlinking (uses more disk space)
    python sentinel_x/scripts/synthetic_fhir_pipeline.py --copy-volumes

    # Custom data directory
    python sentinel_x/scripts/synthetic_fhir_pipeline.py --data-dir /path/to/data

    # Process single report
    python sentinel_x/scripts/synthetic_fhir_pipeline.py --report train_1_a_1.json

See docs/unified_fhir_pipeline.md for full documentation.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent  # sentinel_x/

SYNTHEA_JAR = PROJECT_DIR / "lib" / "synthea-with-dependencies.jar"
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "raw_ct_rate"
SYNTHEA_TEMP_OUTPUT = PROJECT_DIR / "data" / ".synthea_temp"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Pydantic Models for Structured Extraction
# -----------------------------------------------------------------------------


class ExtractedCondition(BaseModel):
    """A medical condition extracted from the radiology report."""
    condition_name: str = Field(description="Name of the medical condition")
    snomed_code: Optional[str] = Field(
        default=None,
        description="SNOMED-CT code for the condition (if known)"
    )
    severity: Literal["none", "mild", "moderate", "severe"] = Field(
        default="mild",
        description="Severity of the condition"
    )
    body_site: Optional[str] = Field(
        default=None,
        description="Body site affected (e.g., 'lung', 'heart', 'spine')"
    )


class DemographicInference(BaseModel):
    """Inferred demographics based on clinical findings."""
    estimated_age_min: int = Field(
        ge=18, le=100,
        description="Minimum estimated age based on clinical findings"
    )
    estimated_age_max: int = Field(
        ge=18, le=100,
        description="Maximum estimated age based on clinical findings"
    )
    gender_hint: Optional[Literal["M", "F"]] = Field(
        default=None,
        description="Inferred gender if determinable from findings"
    )
    reasoning: str = Field(
        description="Brief explanation of why these demographics were inferred"
    )


class RadiologyExtraction(BaseModel):
    """Complete extraction from a radiology report."""
    conditions: list[ExtractedCondition] = Field(
        default_factory=list,
        description="List of medical conditions identified in the report"
    )
    demographics: DemographicInference = Field(
        description="Inferred patient demographics"
    )
    smoking_history_likely: bool = Field(
        default=False,
        description="Whether findings suggest smoking history"
    )
    cardiovascular_risk: Literal["low", "moderate", "high"] = Field(
        default="low",
        description="Assessed cardiovascular risk level"
    )
    synthea_modules: list[str] = Field(
        default_factory=list,
        description="Synthea module names to use (e.g., 'copd', 'cardiovascular_disease')"
    )


# -----------------------------------------------------------------------------
# Condition to Synthea Module Mapping
# -----------------------------------------------------------------------------

CONDITION_TO_MODULE = {
    # Pulmonary conditions
    "emphysema": "copd",
    "bronchiectasis": "copd",
    "copd": "copd",
    "chronic bronchitis": "copd",
    "pulmonary fibrosis": "lung_cancer",  # closest available

    # Cardiovascular conditions
    "atheroma": "cardiovascular_disease",
    "atherosclerosis": "cardiovascular_disease",
    "calcification": "cardiovascular_disease",
    "coronary artery disease": "cardiovascular_disease",
    "cardiomegaly": "congestive_heart_failure",
    "heart failure": "congestive_heart_failure",
    "enlarged heart": "congestive_heart_failure",

    # Renal conditions
    "renal cyst": "chronic_kidney_disease",
    "kidney disease": "chronic_kidney_disease",
    "atrophic kidney": "chronic_kidney_disease",

    # Musculoskeletal
    "spondylosis": "osteoarthritis",
    "degenerative changes": "osteoarthritis",
    "osteoarthritis": "osteoarthritis",
    "osteophytes": "osteoarthritis",

    # Other
    "diabetes": "diabetes",
    "hypertension": "metabolic_syndrome_disease",
}


# Valid Synthea module names (verified against Synthea repository)
# Reference: https://github.com/synthetichealth/synthea/tree/master/src/main/resources/modules
VALID_SYNTHEA_MODULES: set[str] = {
    # Pulmonary
    "copd", "asthma", "bronchitis", "lung_cancer", "allergic_rhinitis",
    # Cardiovascular
    "atrial_fibrillation", "congestive_heart_failure", "hypertension",
    "myocardial_infarction", "stroke", "stable_ischemic_heart_disease",
    # Metabolic
    "diabetes", "metabolic_syndrome_disease", "metabolic_syndrome_care",
    "diabetic_retinopathy_treatment", "gout",
    # Renal
    "chronic_kidney_disease", "dialysis", "kidney_transplant",
    # Musculoskeletal
    "osteoarthritis", "osteoporosis", "rheumatoid_arthritis",
    "total_joint_replacement", "fibromyalgia",
    # Gastrointestinal
    "gallstones", "appendicitis", "colorectal_cancer",
    # Neurological
    "dementia", "epilepsy", "attention_deficit_disorder", "mTBI",
    # Cancer
    "breast_cancer", "acute_myeloid_leukemia",
    # Infectious
    "covid19", "sepsis", "urinary_tract_infections", "hiv_diagnosis", "hiv_care",
    # Other
    "allergies", "anemia___unknown_etiology", "sleep_apnea", "sinusitis",
    "ear_infections", "sore_throat", "dermatitis", "food_allergies",
    "hypothyroidism", "lupus", "cystic_fibrosis", "injuries",
    "opioid_addiction", "self_harm", "homelessness"
}


# -----------------------------------------------------------------------------
# Module Override Paths for Forcing Disease Conditions
# -----------------------------------------------------------------------------
# Each module lists transition paths to override for forcing disease conditions.
# Format: "module.json::JSONPath" - colons will be escaped when writing to properties file.
# Paths are extracted from Synthea's ModuleOverrides tool.
#
# Strategy: Set the first distribution (disease outcome) to 1.0 to guarantee condition.

MODULE_OVERRIDE_PATHS: dict[str, list[str]] = {
    # COPD - has separate smoker/non-smoker pathways with SES splits
    # Smoker pathway has 3 SES levels (low/middle/high), non-smoker has 1 pathway
    # First distribution in each = Emphysema, second = Chronic Bronchitis, third = no disease
    "copd": [
        # Smoker pathways (Low SES, Middle SES, High SES) → force Emphysema
        "$['states']['Potential_COPD_Smoker']['complex_transition'][0]['distributions'][0]['distribution']",
        "$['states']['Potential_COPD_Smoker']['complex_transition'][1]['distributions'][0]['distribution']",
        "$['states']['Potential_COPD_Smoker']['complex_transition'][2]['distributions'][0]['distribution']",
        # Non-smoker pathway → force Emphysema
        "$['states']['Potential_COPD_Nonsmoker']['complex_transition'][1]['distributions'][0]['distribution']",
    ],

    # Congestive Heart Failure - has gender splits (female/male) with 5 age brackets each
    # First 4 distributions = age-based CHF onset, 5th = no CHF
    "congestive_heart_failure": [
        # Female pathways (all age brackets) → force CHF onset
        "$['states']['Determine CHF']['complex_transition'][0]['distributions'][0]['distribution']",
        "$['states']['Determine CHF']['complex_transition'][0]['distributions'][1]['distribution']",
        "$['states']['Determine CHF']['complex_transition'][0]['distributions'][2]['distribution']",
        "$['states']['Determine CHF']['complex_transition'][0]['distributions'][3]['distribution']",
        # Male pathways (all age brackets) → force CHF onset
        "$['states']['Determine CHF']['complex_transition'][1]['distributions'][0]['distribution']",
        "$['states']['Determine CHF']['complex_transition'][1]['distributions'][1]['distribution']",
        "$['states']['Determine CHF']['complex_transition'][1]['distributions'][2]['distribution']",
        "$['states']['Determine CHF']['complex_transition'][1]['distributions'][3]['distribution']",
    ],

    # Osteoarthritis - has veteran/non-veteran and gender splits
    # First distribution = OA onset, second = no OA
    "osteoarthritis": [
        # Non-veteran pathways (male/female)
        "$['states']['Non_Veteran']['complex_transition'][0]['distributions'][0]['distribution']",
        "$['states']['Non_Veteran']['complex_transition'][1]['distributions'][0]['distribution']",
        # Veteran pathways (multiple demographics)
        "$['states']['Veteran']['complex_transition'][0]['distributions'][0]['distribution']",
        "$['states']['Veteran']['complex_transition'][1]['distributions'][0]['distribution']",
        "$['states']['Veteran']['complex_transition'][2]['distributions'][0]['distribution']",
    ],

    # Chronic Kidney Disease - entry via Initial_Kidney_Health state
    # Index 1 = early CKD onset (index 0 = normal, index 2 = late onset)
    "chronic_kidney_disease": [
        "$['states']['Initial_Kidney_Health']['distributed_transition'][1]['distribution']",
    ],

    # Stable Ischemic Heart Disease - entry via Chance_of_IHD state
    # Index 0 = no IHD (set to 0), Index 1 = IHD (currently 1.0, but annual check)
    "stable_ischemic_heart_disease": [
        "$['states']['Chance_of_IHD']['distributed_transition'][1]['distribution']",
    ],

    # Atrial Fibrillation - entry via Chance_of_AFib state
    # Index 1 = AFib onset (index 0 = no AFib delay)
    "atrial_fibrillation": [
        "$['states']['Chance_of_AFib']['distributed_transition'][1]['distribution']",
    ],
}


def validate_synthea_modules(modules: list[str]) -> list[str]:
    """Validate and normalize Synthea module names.

    Filters to only valid module names and normalizes formatting.

    Args:
        modules: List of module names to validate

    Returns:
        List of valid, normalized module names
    """
    valid = []
    for module in modules:
        # Normalize: lowercase, replace spaces/hyphens with underscores
        normalized = module.lower().strip().replace(" ", "_").replace("-", "_")

        if normalized in VALID_SYNTHEA_MODULES:
            valid.append(normalized)
        else:
            # Try partial matching for common mappings
            if "cardiovascular" in normalized:
                # Synthea doesn't have a generic "cardiovascular_disease" module
                # Use stable_ischemic_heart_disease as the closest match
                valid.append("stable_ischemic_heart_disease")
                logger.info(f"Mapped '{module}' to 'stable_ischemic_heart_disease'")
            else:
                logger.warning(f"Unknown Synthea module: '{module}' (normalized: '{normalized}')")

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for m in valid:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    return unique


def generate_module_overrides(modules: list[str], output_dir: Path) -> Optional[Path]:
    """
    Generate a .properties file to force disease transitions to 100%.

    Uses multi-path override strategy: ALL possible paths for each disease
    are set to 100% to guarantee the condition regardless of Synthea's
    random attribute assignments (smoker status, gender, age bracket, etc.)

    The output format follows Synthea's ModuleOverrides format:
    - Module filename prefix: "module.json\\:\\:"
    - JSONPath with escaped colons and spaces

    Args:
        modules: List of validated Synthea module names
        output_dir: Directory to write the override file

    Returns:
        Path to the generated .properties file, or None if no overrides needed
    """
    lines = [
        "# Auto-generated Synthea module overrides",
        "# Forces disease transition probabilities to 100%",
        "# Generated by Sentinel-X synthetic_fhir_pipeline",
        ""
    ]

    overrides_added = 0

    for module in modules:
        if module not in MODULE_OVERRIDE_PATHS:
            logger.warning(f"No override paths defined for module: {module}")
            continue

        paths = MODULE_OVERRIDE_PATHS[module]
        lines.append(f"# {module}")

        for json_path in paths:
            # Format: module.json\:\:JSONPath = value
            # Escape colons with backslashes and spaces in state names
            escaped_path = json_path.replace(" ", "\\ ")
            full_path = f"{module}.json\\:\\:{escaped_path}"
            lines.append(f"{full_path} = 1.0")
            overrides_added += 1

        lines.append("")

    if overrides_added == 0:
        logger.warning("No module overrides generated - no matching modules found")
        return None

    override_file = output_dir / "module_overrides.properties"
    override_file.write_text("\n".join(lines))

    logger.info(f"Generated module overrides: {overrides_added} paths for {len(modules)} modules")
    return override_file


# -----------------------------------------------------------------------------
# SNOMED-CT Code Mapping for Radiology Findings
# -----------------------------------------------------------------------------
# Reference: SNOMED CT International Edition
# Codes verified against https://browser.ihtsdotools.org/

SNOMED_MAPPING: dict[str, tuple[str, str]] = {
    # Pulmonary findings
    "emphysema": ("87433001", "Pulmonary emphysema"),
    "pulmonary emphysema": ("87433001", "Pulmonary emphysema"),
    "centrilobular emphysema": ("195963002", "Centrilobular emphysema"),
    "paraseptal emphysema": ("69120008", "Paraseptal emphysema"),
    "atelectasis": ("46621007", "Atelectasis"),
    "bronchiectasis": ("12295008", "Bronchiectasis"),
    "pulmonary nodule": ("427359005", "Solitary nodule of lung"),
    "lung nodule": ("427359005", "Solitary nodule of lung"),
    "pulmonary fibrosis": ("51615001", "Pulmonary fibrosis"),
    "interstitial lung disease": ("233703007", "Interstitial lung disease"),
    "pleural effusion": ("60046008", "Pleural effusion"),
    "pneumothorax": ("36118008", "Pneumothorax"),
    "pneumonia": ("233604007", "Pneumonia"),
    "consolidation": ("95436008", "Consolidation of lung"),
    "ground glass": ("50196008", "Ground-glass opacity on chest X-ray"),
    "ground-glass": ("50196008", "Ground-glass opacity on chest X-ray"),
    "bronchial wall thickening": ("26036001", "Bronchial wall thickening"),
    "bronchial thickening": ("26036001", "Bronchial wall thickening"),
    "copd": ("13645005", "Chronic obstructive lung disease"),
    "chronic obstructive": ("13645005", "Chronic obstructive lung disease"),
    "pulmonary edema": ("19242006", "Pulmonary edema"),
    "lung mass": ("363358000", "Malignant tumor of lung"),
    "pulmonary mass": ("363358000", "Malignant tumor of lung"),
    "reticulonodular": ("74853003", "Reticulonodular pattern"),
    "peribronchial": ("26036001", "Bronchial wall thickening"),

    # Cardiovascular findings
    "atherosclerosis": ("38716007", "Atherosclerosis"),
    "atheroma": ("38716007", "Atherosclerosis"),
    "atheromatous": ("38716007", "Atherosclerosis"),
    "calcific plaque": ("128305009", "Atherosclerotic plaque"),
    "calcific atheroma": ("128305009", "Atherosclerotic plaque"),
    "atheromatous plaque": ("128305009", "Atherosclerotic plaque"),
    "calcified plaque": ("128305009", "Atherosclerotic plaque"),
    "cardiomegaly": ("8186001", "Cardiomegaly"),
    "enlarged heart": ("8186001", "Cardiomegaly"),
    "cardiac enlargement": ("8186001", "Cardiomegaly"),
    "pericardial effusion": ("373945007", "Pericardial effusion"),
    "aortic aneurysm": ("67362008", "Aortic aneurysm"),
    "thoracic aortic aneurysm": ("54160000", "Thoracic aortic aneurysm"),
    "abdominal aortic aneurysm": ("233985008", "Abdominal aortic aneurysm"),
    "coronary calcification": ("194842008", "Coronary artery calcification"),
    "coronary artery calcification": ("194842008", "Coronary artery calcification"),
    "calcification": ("82650004", "Calcification"),
    "aortic calcification": ("440029008", "Calcification of aorta"),
    "mitral calcification": ("253382006", "Calcification of mitral valve"),
    "vascular calcification": ("128305009", "Atherosclerotic plaque"),

    # Musculoskeletal findings
    "spondylosis": ("75320002", "Spondylosis"),
    "thoracic spondylosis": ("75320002", "Spondylosis"),
    "lumbar spondylosis": ("75320002", "Spondylosis"),
    "cervical spondylosis": ("75320002", "Spondylosis"),
    "osteoarthritis": ("396275006", "Osteoarthritis"),
    "degenerative changes": ("396275006", "Osteoarthritis"),
    "degenerative change": ("396275006", "Osteoarthritis"),
    "degenerative disc": ("77547008", "Degenerative disc disease"),
    "disc degeneration": ("77547008", "Degenerative disc disease"),
    "osteophyte": ("88998003", "Osteophyte"),
    "osteophytes": ("88998003", "Osteophyte"),
    "scoliosis": ("298382003", "Scoliosis deformity of spine"),
    "kyphosis": ("414564002", "Kyphosis"),
    "compression fracture": ("207957008", "Compression fracture of vertebra"),
    "vertebral fracture": ("207957008", "Compression fracture of vertebra"),
    "fracture": ("125605004", "Fracture of bone"),
    "osteopenia": ("64859006", "Osteopenia"),
    "osteoporosis": ("64859006", "Osteopenia"),

    # Abdominal findings
    "cholelithiasis": ("235919008", "Cholelithiasis"),
    "gallstone": ("235919008", "Cholelithiasis"),
    "gallstones": ("235919008", "Cholelithiasis"),
    "hepatomegaly": ("80515008", "Hepatomegaly"),
    "enlarged liver": ("80515008", "Hepatomegaly"),
    "splenomegaly": ("16294009", "Splenomegaly"),
    "enlarged spleen": ("16294009", "Splenomegaly"),
    "renal cyst": ("36171008", "Renal cyst"),
    "kidney cyst": ("36171008", "Renal cyst"),
    "atrophic kidney": ("16395008", "Renal atrophy"),
    "renal atrophy": ("16395008", "Renal atrophy"),
    "kidney atrophy": ("16395008", "Renal atrophy"),
    "chronic kidney disease": ("709044004", "Chronic kidney disease"),
    "fatty liver": ("197321007", "Steatosis of liver"),
    "hepatic steatosis": ("197321007", "Steatosis of liver"),
    "steatosis": ("197321007", "Steatosis of liver"),
    "pancreatic cyst": ("37153006", "Pancreatic cyst"),
    "adrenal nodule": ("126873006", "Adrenal nodule"),
    "adrenal adenoma": ("93911001", "Adrenal adenoma"),
    "hiatal hernia": ("84089009", "Hiatal hernia"),
    "hernia": ("414403008", "Hernia of abdominal cavity"),

    # Vascular findings
    "venous collateral": ("234042006", "Collateral vessel"),
    "collateral vessel": ("234042006", "Collateral vessel"),
    "collaterals": ("234042006", "Collateral vessel"),
    "thrombosis": ("64156001", "Thrombosis"),
    "thrombus": ("64156001", "Thrombosis"),
    "pulmonary embolism": ("59282003", "Pulmonary embolism"),
    "pe": ("59282003", "Pulmonary embolism"),
    "embolism": ("414086009", "Embolism"),
    "dvt": ("128053003", "Deep venous thrombosis"),
    "deep vein thrombosis": ("128053003", "Deep venous thrombosis"),
    "aneurysm": ("432119003", "Aneurysm"),
    "subclavian": ("234044007", "Disorder of subclavian vein"),
    "collapsed vein": ("271299006", "Vein finding"),

    # Infectious/inflammatory findings
    "lymphadenopathy": ("30746006", "Lymphadenopathy"),
    "lymph node enlargement": ("30746006", "Lymphadenopathy"),
    "enlarged lymph node": ("30746006", "Lymphadenopathy"),
    "infectious process": ("40733004", "Infectious disease"),
    "infection": ("40733004", "Infectious disease"),
    "abscess": ("128477000", "Abscess"),
    "granuloma": ("45647009", "Granuloma"),
    "calcified granuloma": ("16003001", "Calcified granuloma"),

    # Other common findings
    "thyroid nodule": ("237495005", "Thyroid nodule"),
    "thyroid mass": ("237495005", "Thyroid nodule"),
    "breast mass": ("290078006", "Mass of breast"),
    "cyst": ("441457006", "Cyst"),
    "mass": ("4147007", "Mass"),
    "lesion": ("52988006", "Lesion"),
    "nodule": ("27925004", "Nodule"),
    "opacity": ("263837009", "Opacification"),
    "density": ("79365008", "Density"),
}


def lookup_snomed_code(condition_name: str) -> tuple[str, str] | None:
    """Look up SNOMED-CT code with fuzzy/partial matching.

    Args:
        condition_name: The condition name to look up

    Returns:
        Tuple of (snomed_code, display_name) or None if not found
    """
    name_lower = condition_name.lower().strip()

    # Try exact match first
    if name_lower in SNOMED_MAPPING:
        return SNOMED_MAPPING[name_lower]

    # Try partial match (keyword in condition name)
    # Sort by key length descending to match longer/more specific terms first
    sorted_keywords = sorted(SNOMED_MAPPING.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword in name_lower:
            return SNOMED_MAPPING[keyword]

    # No match found
    return None


def enrich_snomed_codes(extraction: RadiologyExtraction) -> RadiologyExtraction:
    """Enrich extraction with SNOMED codes for conditions that are missing them.

    This post-processing step attempts to fill in SNOMED codes that the LLM
    extraction may have missed, using the local SNOMED_MAPPING dictionary.

    Args:
        extraction: The RadiologyExtraction to enrich

    Returns:
        The same extraction object with enriched SNOMED codes
    """
    enriched_count = 0

    for condition in extraction.conditions:
        if condition.snomed_code is None:
            result = lookup_snomed_code(condition.condition_name)
            if result:
                condition.snomed_code = result[0]
                enriched_count += 1
                logger.debug(
                    f"SNOMED enrichment: '{condition.condition_name}' -> "
                    f"{result[0]} ({result[1]})"
                )

    if enriched_count > 0:
        logger.info(f"Enriched {enriched_count} conditions with SNOMED codes")

    return extraction


# -----------------------------------------------------------------------------
# Temporal Classification for Realistic Onset Dates
# -----------------------------------------------------------------------------

# Conditions classified by typical onset patterns relative to imaging discovery
CONDITION_TEMPORAL_CLASS: dict[str, list[str]] = {
    # Degenerative conditions: develop over 5-20 years
    "degenerative": [
        "spondylosis", "osteoarthritis", "degenerative", "osteophyte",
        "disc disease", "stenosis", "ddd", "arthritis"
    ],

    # Chronic conditions: typically present 2-10 years before discovery
    "chronic": [
        "emphysema", "copd", "fibrosis", "bronchiectasis", "cardiomegaly",
        "atherosclerosis", "atheroma", "calcification", "calcific",
        "chronic kidney", "diabetes", "hypertension", "cirrhosis",
        "hepatomegaly", "splenomegaly", "cholelithiasis", "gallstone",
        "scoliosis", "kyphosis", "fatty liver", "steatosis"
    ],

    # Subacute conditions: developing over 2 weeks to 6 months
    "subacute": [
        "consolidation", "effusion", "nodule", "mass", "lymphadenopathy",
        "thickening", "infiltrate", "opacity"
    ],

    # Acute conditions: recent onset, 0-14 days before scan
    "acute": [
        "pneumonia", "infection", "infectious", "pneumothorax", "embolism",
        "infarct", "hemorrhage", "edema", "thrombus", "thrombosis",
        "dissection", "fracture", "acute", "abscess"
    ],

    # Incidental findings: discovered at scan time, onset unknown
    "incidental": [
        "cyst", "hemangioma", "lipoma", "granuloma", "calcified granuloma",
        "benign", "incidental"
    ]
}


def classify_condition_temporality(condition_name: str) -> str:
    """Classify a condition by its typical temporal onset pattern.

    Args:
        condition_name: The name of the condition

    Returns:
        Temporal class: "degenerative", "chronic", "subacute", "acute", or "incidental"
    """
    name_lower = condition_name.lower()

    for temporal_class, keywords in CONDITION_TEMPORAL_CLASS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return temporal_class

    # Default to chronic for unknown conditions (conservative assumption)
    return "chronic"


def calculate_onset_date(
    scan_datetime: datetime,
    temporal_class: str,
    seed: int
) -> tuple[datetime, str]:
    """Calculate a realistic onset date based on condition temporal class.

    Uses deterministic random for reproducibility.

    Args:
        scan_datetime: When the scan was performed
        temporal_class: The temporal classification of the condition
        seed: Seed for deterministic random number generation

    Returns:
        Tuple of (onset_datetime, clinical_note)
    """
    import random
    rng = random.Random(seed)

    # Define offset ranges in days for each temporal class
    offset_ranges = {
        "degenerative": (365 * 5, 365 * 20),    # 5-20 years before scan
        "chronic": (365 * 2, 365 * 10),          # 2-10 years before scan
        "subacute": (14, 180),                    # 2 weeks to 6 months before scan
        "acute": (0, 14),                         # 0-14 days before scan
        "incidental": (0, 0),                     # Discovered at scan time
    }

    min_days, max_days = offset_ranges.get(temporal_class, (365, 365 * 5))

    if min_days == max_days == 0:
        # Incidental: use scan date as onset
        return scan_datetime, "Incidental finding discovered on imaging"

    # Calculate random offset within range
    days_before = rng.randint(min_days, max_days)
    onset = scan_datetime - timedelta(days=days_before)

    # Generate clinical note based on temporal class
    if temporal_class == "degenerative":
        years = days_before // 365
        note = f"Degenerative condition, estimated onset {years} years prior to imaging"
    elif temporal_class == "chronic":
        years = days_before // 365
        note = f"Chronic condition, estimated onset {years} years prior to imaging"
    elif temporal_class == "subacute":
        if days_before < 30:
            note = f"Subacute finding, estimated onset {days_before} days prior to imaging"
        else:
            months = days_before // 30
            note = f"Subacute finding, estimated onset {months} months prior to imaging"
    elif temporal_class == "acute":
        note = f"Acute finding, estimated onset {days_before} days prior to imaging"
    else:
        note = f"Estimated onset {days_before} days prior to imaging"

    return onset, note


# -----------------------------------------------------------------------------
# OpenAI Extraction
# -----------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a medical data extraction specialist. Your task is to analyze radiology reports and extract structured clinical information.

For each report, you must:
1. Identify all medical conditions mentioned in the findings and impressions
2. Provide SNOMED-CT codes for conditions when possible (common codes below)
3. Infer patient demographics (age range, gender if determinable) based on clinical patterns
4. Map conditions to Synthea modules for synthetic patient generation

Common SNOMED-CT codes (use these whenever applicable):

Pulmonary:
- Emphysema: 87433001
- Bronchiectasis: 12295008
- Atelectasis: 46621007
- Pulmonary nodule: 427359005
- Pulmonary fibrosis: 51615001
- Pleural effusion: 60046008
- Consolidation: 95436008
- Ground-glass opacity: 50196008
- Bronchial wall thickening: 26036001
- COPD: 13645005

Cardiovascular:
- Atherosclerosis/atheroma: 38716007
- Calcified plaque: 128305009
- Cardiomegaly: 8186001
- Pericardial effusion: 373945007
- Aortic aneurysm: 67362008
- Coronary calcification: 194842008
- Calcification (general): 82650004

Musculoskeletal:
- Spondylosis: 75320002
- Osteoarthritis/degenerative changes: 396275006
- Degenerative disc disease: 77547008
- Osteophytes: 88998003
- Scoliosis: 298382003

Abdominal:
- Cholelithiasis/gallstones: 235919008
- Hepatomegaly: 80515008
- Renal cyst: 36171008
- Renal atrophy: 16395008
- Chronic kidney disease: 709044004
- Fatty liver/steatosis: 197321007

Vascular:
- Venous collaterals: 234042006
- Thrombosis: 64156001
- Lymphadenopathy: 30746006

Other:
- Granuloma: 45647009
- Cyst: 441457006
- Hiatal hernia: 84089009

If you cannot find an exact SNOMED code, leave snomed_code as null.

Synthea module mapping:
- Pulmonary conditions (emphysema, bronchiectasis, COPD) → "copd"
- Atherosclerosis, calcifications → "cardiovascular_disease"
- Cardiomegaly, heart failure → "congestive_heart_failure"
- Kidney conditions → "chronic_kidney_disease"
- Degenerative spine/joint changes → "osteoarthritis"

Demographic inference guidelines:
- Degenerative changes, spondylosis, emphysema suggest older age (55-85)
- Multiple chronic conditions suggest middle-aged to elderly (50-80)
- Atherosclerosis/calcifications typically appear after 40
- If no age indicators, use a broader range (35-75)
- Gender is usually not determinable from chest CT unless specific anatomy mentioned"""


async def extract_with_openai(
    client: AsyncOpenAI,
    report: dict
) -> RadiologyExtraction:
    """Extract structured clinical data from a radiology report using OpenAI."""

    # Combine findings and impressions for analysis
    report_text = f"""
Clinical Information: {report.get('clinical_information', 'Not provided')}
Technique: {report.get('technique', 'Not provided')}
Findings: {report.get('findings', 'Not provided')}
Impressions: {report.get('impressions', 'Not provided')}
""".strip()

    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract clinical data from this radiology report:\n\n{report_text}"}
            ],
            response_format=RadiologyExtraction,
            temperature=0.1
        )

        extraction = response.choices[0].message.parsed

        # Enhance synthea_modules based on extracted conditions
        if extraction:
            enhanced_modules = set(extraction.synthea_modules)
            for condition in extraction.conditions:
                condition_lower = condition.condition_name.lower()
                for keyword, module in CONDITION_TO_MODULE.items():
                    if keyword in condition_lower:
                        enhanced_modules.add(module)
            extraction.synthea_modules = list(enhanced_modules)

            # Enrich missing SNOMED codes using local mapping
            extraction = enrich_snomed_codes(extraction)

        return extraction

    except Exception as e:
        logger.error(f"OpenAI extraction failed: {e}")
        # Return default extraction on failure
        return RadiologyExtraction(
            conditions=[],
            demographics=DemographicInference(
                estimated_age_min=40,
                estimated_age_max=70,
                gender_hint=None,
                reasoning="Default demographics due to extraction failure"
            ),
            smoking_history_likely=False,
            cardiovascular_risk="low",
            synthea_modules=[]
        )


# -----------------------------------------------------------------------------
# Synthea Configuration and Execution
# -----------------------------------------------------------------------------

@dataclass
class SyntheaConfig:
    """Configuration for running Synthea."""
    age_min: int
    age_max: int
    gender: Optional[str] = None
    modules: list[str] = field(default_factory=list)
    state: str = "Massachusetts"
    output_dir: Path = SYNTHEA_TEMP_OUTPUT
    seed: Optional[int] = None


def create_synthea_config(extraction: RadiologyExtraction, report_name: str) -> SyntheaConfig:
    """Create Synthea configuration from extracted data."""
    # Generate deterministic seed from report name using SHA256
    # This ensures reproducibility across different Python processes
    hash_obj = hashlib.sha256(report_name.encode('utf-8'))
    seed = int.from_bytes(hash_obj.digest()[:4], byteorder='big') & 0x7FFFFFFF  # Positive 32-bit integer

    return SyntheaConfig(
        age_min=extraction.demographics.estimated_age_min,
        age_max=extraction.demographics.estimated_age_max,
        gender=extraction.demographics.gender_hint,
        modules=extraction.synthea_modules,
        seed=seed
    )


def run_synthea(config: SyntheaConfig) -> Optional[dict]:
    """Run Synthea to generate a base FHIR bundle."""

    if not SYNTHEA_JAR.exists():
        logger.error(f"Synthea JAR not found: {SYNTHEA_JAR}")
        return None

    # Create temp output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous output from temp directory (including fhir subdirectory)
    for f in config.output_dir.glob("*.json"):
        f.unlink()
    fhir_subdir = config.output_dir / "fhir"
    if fhir_subdir.exists():
        for f in fhir_subdir.glob("*.json"):
            f.unlink()

    # Build Synthea command with = syntax for properties
    cmd = [
        "java", "-jar", str(SYNTHEA_JAR),
        "-p", "1",  # Single patient
        "-a", f"{config.age_min}-{config.age_max}",
        f"--exporter.baseDirectory={config.output_dir}",
        "--exporter.fhir.export=true",
        "--exporter.fhir.use_us_core_ig=true",
        "--exporter.ccda.export=false",
        "--exporter.csv.export=false",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        "--generate.only_alive_patients=true",
    ]

    # Add gender if specified
    if config.gender:
        cmd.extend(["-g", config.gender])

    # Add seed if specified (use -ps for single person seed)
    if config.seed is not None:
        cmd.extend(["-ps", str(config.seed)])

    # Generate module overrides to force disease conditions
    # This replaces the -m flag approach which broke base patient generation
    override_file = None
    validated_modules = []
    if config.modules:
        validated_modules = validate_synthea_modules(config.modules)
        if validated_modules:
            override_file = generate_module_overrides(
                modules=validated_modules,
                output_dir=config.output_dir
            )
            if override_file:
                cmd.extend(["--module_override", str(override_file)])
                logger.info(f"Using module overrides: {override_file}")

    # Add state
    cmd.append(config.state)

    override_str = f", overrides={len(validated_modules)} modules" if override_file else ""
    logger.info(f"Running Synthea: age={config.age_min}-{config.age_max}, gender={config.gender or 'any'}, seed={config.seed}{override_str}")
    logger.debug(f"Synthea command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            logger.error(f"Synthea failed: {result.stderr}")
            return None

        # Find the generated FHIR bundle in the fhir subdirectory
        fhir_output_dir = config.output_dir / "fhir"
        if not fhir_output_dir.exists():
            # Fallback to base directory
            fhir_output_dir = config.output_dir

        fhir_files = [
            f for f in fhir_output_dir.glob("*.json")
            if not f.name.startswith(("hospital", "practitioner"))
        ]

        if not fhir_files:
            logger.error(f"No FHIR bundle generated by Synthea in {fhir_output_dir}")
            # Log what files exist for debugging
            all_files = list(fhir_output_dir.glob("*")) if fhir_output_dir.exists() else []
            logger.error(f"Files in output dir: {all_files}")
            return None

        # Read the first patient bundle
        with open(fhir_files[0], "r") as f:
            bundle = json.load(f)

        logger.info(f"Generated Synthea bundle with {len(bundle.get('entry', []))} entries")
        return bundle

    except subprocess.TimeoutExpired:
        logger.error("Synthea execution timed out")
        return None
    except Exception as e:
        logger.error(f"Error running Synthea: {e}")
        return None


# -----------------------------------------------------------------------------
# FHIR Resource Creation (US Core R4 Compliant)
# -----------------------------------------------------------------------------

def generate_uuid() -> str:
    """Generate a new UUID."""
    return str(uuid.uuid4())


def create_imaging_study(
    patient_ref: str,
    volume_name: str,
    study_datetime: str
) -> dict:
    """Create an ImagingStudy resource linked to the CT volume."""

    study_uid = f"2.25.{uuid.uuid4().int}"
    series_uid = f"2.25.{uuid.uuid4().int}"

    return {
        "resourceType": "ImagingStudy",
        "id": generate_uuid(),
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/ImagingStudy"]
        },
        "identifier": [{
            "system": "urn:dicom:uid",
            "value": f"urn:oid:{study_uid}"
        }],
        "status": "available",
        "subject": {"reference": patient_ref},
        "started": study_datetime,
        "modality": [{
            "system": "http://dicom.nema.org/resources/ontology/DCM",
            "code": "CT",
            "display": "Computed Tomography"
        }],
        "description": f"CT Chest - {volume_name}",
        "series": [{
            "uid": series_uid,
            "modality": {
                "system": "http://dicom.nema.org/resources/ontology/DCM",
                "code": "CT",
                "display": "Computed Tomography"
            },
            "bodySite": {
                "system": "http://snomed.info/sct",
                "code": "51185008",
                "display": "Thorax"
            },
            "description": "Axial CT Chest",
            "numberOfInstances": 1,
            "instance": [{
                "uid": f"2.25.{uuid.uuid4().int}",
                "sopClass": {
                    "system": "urn:ietf:rfc:3986",
                    "code": "urn:oid:1.2.840.10008.5.1.4.1.1.2"
                },
                "title": volume_name
            }]
        }]
    }


def create_diagnostic_report(
    patient_ref: str,
    imaging_study_ref: str,
    report: dict,
    extraction: RadiologyExtraction,
    report_datetime: str,
    result_refs: list[str] | None = None
) -> dict:
    """Create a DiagnosticReport resource (US Core DiagnosticReport for Report and Note).

    Args:
        patient_ref: Reference to the patient
        imaging_study_ref: Reference to the ImagingStudy
        report: The original report dict with findings/impressions
        extraction: Extracted clinical data
        report_datetime: When the report was created
        result_refs: Optional list of Observation references (findings)
    """

    # Encode full report text as base64
    full_report_text = f"""
Clinical Information: {report.get('clinical_information', 'Not provided')}

Technique: {report.get('technique', 'Not provided')}

Findings:
{report.get('findings', 'Not provided')}

Impressions:
{report.get('impressions', 'Not provided')}
""".strip()

    report_base64 = base64.b64encode(full_report_text.encode()).decode()

    # Build conclusion codes from extracted conditions
    conclusion_codes = []
    for condition in extraction.conditions:
        if condition.snomed_code:
            conclusion_codes.append({
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": condition.snomed_code,
                    "display": condition.condition_name
                }],
                "text": condition.condition_name
            })

    diagnostic_report = {
        "resourceType": "DiagnosticReport",
        "id": generate_uuid(),
        "meta": {
            "profile": [
                "http://hl7.org/fhir/us/core/StructureDefinition/us-core-diagnosticreport-note"
            ]
        },
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://loinc.org",
                "code": "18748-4",
                "display": "Diagnostic imaging study"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "24627-2",
                "display": "Chest CT"
            }],
            "text": "CT Chest"
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": report_datetime,
        "issued": report_datetime,
        "conclusion": report.get("impressions", ""),
        "presentedForm": [{
            "contentType": "text/plain",
            "data": report_base64,
            "title": "Radiology Report"
        }]
    }

    # Add imaging study reference
    diagnostic_report["imagingStudy"] = [{"reference": imaging_study_ref}]

    # Add result references to finding observations
    if result_refs:
        diagnostic_report["result"] = [{"reference": ref} for ref in result_refs]

    # Add conclusion codes if available
    if conclusion_codes:
        diagnostic_report["conclusionCode"] = conclusion_codes

    return diagnostic_report


def create_smoking_observation(
    patient_ref: str,
    is_smoker: bool,
    effective_datetime: str
) -> dict:
    """Create US Core Smoking Status Observation.

    Profile: http://hl7.org/fhir/us/core/StructureDefinition/us-core-smokingstatus

    Args:
        patient_ref: Reference to the patient (e.g., "Patient/123")
        is_smoker: Whether the patient is a smoker (True) or non-smoker (False)
        effective_datetime: When the smoking status was assessed

    Returns:
        FHIR Observation resource dict
    """
    # SNOMED CT codes for smoking status
    # Reference: http://hl7.org/fhir/us/core/ValueSet/us-core-smoking-status-observation-codes
    if is_smoker:
        # "Current every day smoker" - most conservative assumption when findings suggest smoking
        snomed_code = "449868002"
        display = "Current every day smoker"
    else:
        # "Never smoked tobacco"
        snomed_code = "266919005"
        display = "Never smoked tobacco"

    return {
        "resourceType": "Observation",
        "id": generate_uuid(),
        "meta": {
            "profile": [
                "http://hl7.org/fhir/us/core/StructureDefinition/us-core-smokingstatus"
            ]
        },
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "social-history",
                "display": "Social History"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "72166-2",
                "display": "Tobacco smoking status"
            }],
            "text": "Tobacco smoking status"
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": effective_datetime,
        "valueCodeableConcept": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": snomed_code,
                "display": display
            }],
            "text": display
        }
    }


def create_cardiovascular_risk_assessment(
    patient_ref: str,
    risk_level: Literal["low", "moderate", "high"],
    occurrence_datetime: str,
    basis_condition_refs: list[str] | None = None
) -> dict:
    """Create a RiskAssessment resource for cardiovascular risk.

    FHIR RiskAssessment: http://hl7.org/fhir/riskassessment.html

    This captures the assessed cardiovascular risk based on imaging findings
    such as calcified atheromas, cardiomegaly, and vascular calcifications.

    Args:
        patient_ref: Reference to the patient
        risk_level: Assessed risk level ("low", "moderate", or "high")
        occurrence_datetime: When the assessment was made
        basis_condition_refs: Optional list of condition references that support the assessment

    Returns:
        FHIR RiskAssessment resource dict
    """
    # Map risk level to probability range
    probability_map = {
        "low": 0.1,
        "moderate": 0.35,
        "high": 0.65
    }

    risk_assessment = {
        "resourceType": "RiskAssessment",
        "id": generate_uuid(),
        "status": "final",
        "subject": {"reference": patient_ref},
        "occurrenceDateTime": occurrence_datetime,
        "method": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": "225338004",
                "display": "Risk assessment"
            }],
            "text": "Cardiovascular risk assessment based on imaging findings"
        },
        "code": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": "441829007",
                "display": "Assessment of cardiovascular system"
            }],
            "text": f"Cardiovascular Risk Assessment - {risk_level.upper()}"
        },
        "prediction": [{
            "outcome": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "49601007",
                    "display": "Disorder of cardiovascular system"
                }],
                "text": "Cardiovascular disease"
            },
            "qualitativeRisk": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                    "code": risk_level,
                    "display": risk_level.capitalize()
                }],
                "text": f"{risk_level.capitalize()} risk"
            },
            "probabilityDecimal": probability_map.get(risk_level, 0.35)
        }],
        "note": [{
            "text": f"Cardiovascular risk assessed as {risk_level} based on CT imaging findings including vascular calcifications and cardiac morphology."
        }]
    }

    # Add basis references if provided (conditions/observations that support the assessment)
    if basis_condition_refs:
        risk_assessment["basis"] = [{"reference": ref} for ref in basis_condition_refs]

    return risk_assessment


def create_finding_observation(
    patient_ref: str,
    condition: ExtractedCondition,
    effective_datetime: str,
    diagnostic_report_ref: str | None = None
) -> dict:
    """Create an Observation resource for an imaging finding.

    This creates a discrete representation of a finding from the radiology report,
    linking it to the DiagnosticReport it was derived from.

    Args:
        patient_ref: Reference to the patient
        condition: The extracted condition/finding
        effective_datetime: When the finding was observed
        diagnostic_report_ref: Reference to the source DiagnosticReport

    Returns:
        FHIR Observation resource dict
    """
    observation = {
        "resourceType": "Observation",
        "id": generate_uuid(),
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "imaging",
                "display": "Imaging"
            }]
        }],
        "code": {
            "text": condition.condition_name
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": effective_datetime
    }

    # Add SNOMED code if available
    if condition.snomed_code:
        observation["code"]["coding"] = [{
            "system": "http://snomed.info/sct",
            "code": condition.snomed_code,
            "display": condition.condition_name
        }]

    # Add body site if available
    if condition.body_site:
        observation["bodySite"] = {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": "51185008",  # Default to thorax for chest CT
                "display": "Thorax"
            }],
            "text": condition.body_site
        }

    # Add severity as interpretation
    if condition.severity and condition.severity != "none":
        severity_map = {
            "mild": ("L", "Low"),
            "moderate": ("N", "Normal"),  # Using normal as moderate baseline
            "severe": ("H", "High")
        }
        code, display = severity_map.get(condition.severity, ("N", "Normal"))
        observation["interpretation"] = [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": code,
                "display": display
            }],
            "text": f"Severity: {condition.severity}"
        }]

    # Add derivedFrom reference to the DiagnosticReport
    if diagnostic_report_ref:
        observation["derivedFrom"] = [{"reference": diagnostic_report_ref}]

    return observation


def create_condition_resource(
    patient_ref: str,
    condition: ExtractedCondition,
    onset_datetime: str,
    evidence_ref: str | None = None,
    recorded_datetime: str | None = None,
    onset_note: str | None = None
) -> dict:
    """Create a Condition resource (US Core Condition).

    Args:
        patient_ref: Reference to the patient
        condition: The extracted condition
        onset_datetime: When the condition started (estimated onset)
        evidence_ref: Optional reference to the supporting Observation
        recorded_datetime: When the condition was documented (scan date)
        onset_note: Clinical note about how onset was estimated
    """
    condition_resource = {
        "resourceType": "Condition",
        "id": generate_uuid(),
        "meta": {
            "profile": [
                "http://hl7.org/fhir/us/core/StructureDefinition/us-core-condition-encounter-diagnosis"
            ]
        },
        "clinicalStatus": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": "active",
                "display": "Active"
            }]
        },
        "verificationStatus": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "code": "confirmed",
                "display": "Confirmed"
            }]
        },
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                "code": "encounter-diagnosis",
                "display": "Encounter Diagnosis"
            }]
        }],
        "code": {
            "text": condition.condition_name
        },
        "subject": {"reference": patient_ref},
        "onsetDateTime": onset_datetime
    }

    # Add SNOMED code if available
    if condition.snomed_code:
        condition_resource["code"]["coding"] = [{
            "system": "http://snomed.info/sct",
            "code": condition.snomed_code,
            "display": condition.condition_name
        }]

    # Add severity if not 'none'
    if condition.severity != "none":
        severity_codes = {
            "mild": ("255604002", "Mild"),
            "moderate": ("6736007", "Moderate"),
            "severe": ("24484000", "Severe")
        }
        code, display = severity_codes.get(condition.severity, ("255604002", "Mild"))
        condition_resource["severity"] = {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": code,
                "display": display
            }]
        }

    # Add body site if available
    if condition.body_site:
        condition_resource["bodySite"] = [{
            "text": condition.body_site
        }]

    # Add evidence linking to the supporting Observation
    if evidence_ref:
        condition_resource["evidence"] = [{
            "detail": [{"reference": evidence_ref}]
        }]

    # Add recordedDate (when the condition was documented - typically scan date)
    if recorded_datetime:
        condition_resource["recordedDate"] = recorded_datetime

    # Add note about onset estimation
    if onset_note:
        condition_resource["note"] = [{
            "text": onset_note
        }]

    return condition_resource


# -----------------------------------------------------------------------------
# FHIR Bundle Merging
# -----------------------------------------------------------------------------

def get_patient_reference(bundle: dict) -> Optional[str]:
    """Extract patient reference from bundle."""
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            patient_id = resource.get("id")
            if patient_id:
                return f"Patient/{patient_id}"
    return None


def extract_temporal_value(resource: dict, field_path: str) -> Optional[str]:
    """
    Extract a temporal value from a resource given a field path.

    Args:
        resource: FHIR resource
        field_path: Dot-separated path (e.g., 'period.start')

    Returns:
        Temporal value as string or None
    """
    parts = field_path.split('.')
    value = resource
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value if isinstance(value, str) else None


def has_future_date(resource: dict, scan_date: str) -> bool:
    """
    Check if a resource has any temporal field after the scan date.

    This identifies Synthea-generated resources that represent "future" data
    relative to the scan acquisition time, which violates temporal simulation
    boundaries.

    Args:
        resource: FHIR resource to check
        scan_date: Reference scan date (ISO 8601 format)

    Returns:
        True if resource has any date after scan_date
    """
    resource_type = resource.get('resourceType')

    # Map of resource types to their temporal fields
    temporal_fields_map = {
        'Condition': ['onsetDateTime', 'abatementDateTime', 'recordedDate'],
        'Encounter': ['period.start', 'period.end'],
        'Observation': ['effectiveDateTime', 'issued'],
        'MedicationRequest': ['authoredOn'],
        'Procedure': ['performedDateTime', 'performedPeriod.start', 'performedPeriod.end'],
        'DiagnosticReport': ['effectiveDateTime', 'issued'],
        'ImagingStudy': ['started'],
        'Immunization': ['occurrenceDateTime'],
        'AllergyIntolerance': ['recordedDate'],
        'CarePlan': ['period.start', 'period.end'],
        'Claim': ['created'],
        'ExplanationOfBenefit': ['created'],
        'MedicationAdministration': ['effectiveDateTime', 'effectivePeriod.start', 'effectivePeriod.end'],
        'Device': ['manufactureDate'],
        'CareTeam': ['period.start', 'period.end'],
        'DocumentReference': ['date'],
        'SupplyDelivery': ['occurrenceDateTime'],
    }

    temporal_fields = temporal_fields_map.get(resource_type, [])

    for field_path in temporal_fields:
        date_value = extract_temporal_value(resource, field_path)
        if date_value and date_value > scan_date:
            return True

    return False


def is_manually_created(resource: dict, scan_date: str, volume_name: str) -> bool:
    """
    Check if a resource was manually created by the pipeline (not Synthea).

    Manually created resources are:
    - ImagingStudy with description "CT Chest - {volume_name}"
    - DiagnosticReport with effectiveDateTime exactly matching scan_date
    - Condition with onsetDateTime exactly matching scan_date

    Args:
        resource: FHIR resource to check
        scan_date: Reference scan date
        volume_name: Volume filename (e.g., "train_1_a_1.nii.gz")

    Returns:
        True if resource was manually created
    """
    resource_type = resource.get('resourceType')

    # Check ImagingStudy by description
    if resource_type == 'ImagingStudy':
        desc = resource.get('description', '')
        if desc == f"CT Chest - {volume_name}":
            return True

    # Check DiagnosticReport by temporal match and category
    if resource_type == 'DiagnosticReport':
        effective_dt = resource.get('effectiveDateTime')
        if effective_dt == scan_date:
            # Additional check: our reports have category "18748-4" (Diagnostic imaging study)
            categories = resource.get('category', [])
            for cat in categories:
                codings = cat.get('coding', [])
                for coding in codings:
                    if coding.get('code') == '18748-4':
                        return True

    # Check Condition by temporal match and category
    if resource_type == 'Condition':
        onset_dt = resource.get('onsetDateTime')
        if onset_dt == scan_date:
            # Additional check: our conditions have category "encounter-diagnosis"
            categories = resource.get('category', [])
            for cat in categories:
                codings = cat.get('coding', [])
                for coding in codings:
                    if coding.get('code') == 'encounter-diagnosis':
                        return True

    return False


def filter_future_events(bundle: dict, scan_date: str, volume_name: str) -> dict:
    """
    Filter out Synthea-generated resources with dates after the scan date.

    This enforces the temporal simulation boundary: FHIR data should only
    represent medical history up to the scan acquisition time, not beyond it.

    Args:
        bundle: FHIR bundle to filter
        scan_date: Reference scan date from ImagingStudy.started
        volume_name: Volume filename for identifying manually created resources

    Returns:
        Filtered FHIR bundle with temporal violations removed
    """
    filtered_entries = []
    removed_count = 0
    removed_by_type = {}

    for entry in bundle.get('entry', []):
        resource = entry.get('resource', {})
        resource_type = resource.get('resourceType')

        # Always keep: Patient, Practitioner, Organization, Location (no temporal data)
        if resource_type in ['Patient', 'Practitioner', 'Organization', 'Location', 'Provenance']:
            filtered_entries.append(entry)
            continue

        # Always keep manually created resources
        if is_manually_created(resource, scan_date, volume_name):
            filtered_entries.append(entry)
            continue

        # Check if Synthea resource has future dates
        if has_future_date(resource, scan_date):
            removed_count += 1
            removed_by_type[resource_type] = removed_by_type.get(resource_type, 0) + 1
            continue  # Skip this resource

        # Keep resource
        filtered_entries.append(entry)

    # Update bundle
    original_count = len(bundle.get('entry', []))
    bundle['entry'] = filtered_entries

    if removed_count > 0:
        logger.info(
            f"Filtered {removed_count} future-dated resources from bundle "
            f"(kept {len(filtered_entries)}/{original_count})"
        )
        logger.info(f"Removed by type: {removed_by_type}")
    else:
        logger.debug(f"No temporal violations found (all {original_count} resources valid)")

    return bundle


def merge_radiology_resources(
    synthea_bundle: dict,
    extraction: RadiologyExtraction,
    report: dict
) -> dict:
    """Merge radiology-specific resources into the Synthea bundle."""

    patient_ref = get_patient_reference(synthea_bundle)
    if not patient_ref:
        logger.error("Could not find patient reference in Synthea bundle")
        return synthea_bundle

    # Use current datetime for all resources
    now = datetime.utcnow().isoformat() + "Z"
    volume_name = report.get("volume_name", "unknown.nii.gz")

    # Create ImagingStudy
    imaging_study = create_imaging_study(patient_ref, volume_name, now)
    imaging_study_ref = f"ImagingStudy/{imaging_study['id']}"

    # Step 1: Create Finding Observations for each extracted condition
    # These will be linked from DiagnosticReport.result and to Condition.evidence
    finding_observations = []
    finding_obs_refs = []
    for condition in extraction.conditions:
        obs = create_finding_observation(
            patient_ref,
            condition,
            now,
            diagnostic_report_ref=None  # Will be updated after DiagnosticReport is created
        )
        finding_observations.append(obs)
        finding_obs_refs.append(f"Observation/{obs['id']}")

    # Step 2: Create DiagnosticReport with result references to finding observations
    diagnostic_report = create_diagnostic_report(
        patient_ref,
        imaging_study_ref,
        report,
        extraction,
        now,
        result_refs=finding_obs_refs
    )
    diagnostic_report_ref = f"DiagnosticReport/{diagnostic_report['id']}"

    # Step 3: Update Finding Observations with derivedFrom reference to DiagnosticReport
    for obs in finding_observations:
        obs["derivedFrom"] = [{"reference": diagnostic_report_ref}]

    # Step 4: Create Condition resources with realistic onset dates and evidence links
    # Parse scan datetime for temporal calculations
    scan_dt = datetime.fromisoformat(now.replace("Z", "+00:00"))

    conditions = []
    condition_refs = []  # Track refs for risk assessment basis
    for i, condition in enumerate(extraction.conditions):
        # Calculate realistic onset date based on condition type
        temporal_class = classify_condition_temporality(condition.condition_name)

        # Generate deterministic seed from volume_name and condition name
        seed_str = f"{volume_name}_{condition.condition_name}_{i}"
        seed = hash(seed_str) & 0x7FFFFFFF  # Positive 32-bit integer

        onset_dt, onset_note = calculate_onset_date(scan_dt, temporal_class, seed)
        onset_datetime_str = onset_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Link condition to its corresponding finding observation
        evidence_ref = finding_obs_refs[i] if i < len(finding_obs_refs) else None
        condition_resource = create_condition_resource(
            patient_ref,
            condition,
            onset_datetime_str,
            evidence_ref=evidence_ref,
            recorded_datetime=now,  # Documented at scan time
            onset_note=onset_note
        )
        conditions.append(condition_resource)
        condition_refs.append(f"Condition/{condition_resource['id']}")

        logger.debug(
            f"Condition '{condition.condition_name}': temporal_class={temporal_class}, "
            f"onset={onset_datetime_str}"
        )

    # Create Smoking Status Observation (uses extracted smoking_history_likely)
    smoking_observation = create_smoking_observation(
        patient_ref,
        extraction.smoking_history_likely,
        now
    )

    # Create Cardiovascular Risk Assessment (uses extracted cardiovascular_risk)
    # Link to cardiovascular-related conditions as basis
    cv_related_keywords = ["atheroma", "atherosclerosis", "calcif", "cardio", "coronary", "aortic"]
    cv_basis_refs = [
        ref for ref, cond in zip(condition_refs, extraction.conditions)
        if any(kw in cond.condition_name.lower() for kw in cv_related_keywords)
    ]

    cardiovascular_risk_assessment = create_cardiovascular_risk_assessment(
        patient_ref,
        extraction.cardiovascular_risk,
        now,
        basis_condition_refs=cv_basis_refs if cv_basis_refs else None
    )

    # Add new resources to bundle
    new_entries = [
        {
            "fullUrl": f"urn:uuid:{imaging_study['id']}",
            "resource": imaging_study,
            "request": {"method": "POST", "url": "ImagingStudy"}
        },
        {
            "fullUrl": f"urn:uuid:{diagnostic_report['id']}",
            "resource": diagnostic_report,
            "request": {"method": "POST", "url": "DiagnosticReport"}
        },
        {
            "fullUrl": f"urn:uuid:{smoking_observation['id']}",
            "resource": smoking_observation,
            "request": {"method": "POST", "url": "Observation"}
        },
        {
            "fullUrl": f"urn:uuid:{cardiovascular_risk_assessment['id']}",
            "resource": cardiovascular_risk_assessment,
            "request": {"method": "POST", "url": "RiskAssessment"}
        }
    ]

    # Add finding observations (linked to DiagnosticReport.result)
    for obs in finding_observations:
        new_entries.append({
            "fullUrl": f"urn:uuid:{obs['id']}",
            "resource": obs,
            "request": {"method": "POST", "url": "Observation"}
        })

    # Add conditions (linked to observations via evidence)
    for condition in conditions:
        new_entries.append({
            "fullUrl": f"urn:uuid:{condition['id']}",
            "resource": condition,
            "request": {"method": "POST", "url": "Condition"}
        })

    # Add to bundle
    if "entry" not in synthea_bundle:
        synthea_bundle["entry"] = []

    synthea_bundle["entry"].extend(new_entries)

    logger.info(
        f"Merged {len(new_entries)} resources: "
        f"ImagingStudy, DiagnosticReport, SmokingStatus, RiskAssessment, "
        f"{len(finding_observations)} FindingObs, {len(conditions)} Conditions"
    )

    # Apply temporal filtering to enforce simulation boundary
    # This removes any Synthea-generated resources with dates after the scan
    scan_date = now  # Use the scan date we just created
    filtered_bundle = filter_future_events(synthea_bundle, scan_date, volume_name)

    return filtered_bundle


# -----------------------------------------------------------------------------
# Bundle Validation
# -----------------------------------------------------------------------------

def validate_temporal_consistency(bundle: dict) -> list[str]:
    """Validate temporal ordering of resources.

    Checks that:
    - Condition.onsetDateTime <= Condition.recordedDate
    - DiagnosticReport.effectiveDateTime <= DiagnosticReport.issued
    - All dates are valid ISO format

    Args:
        bundle: The FHIR bundle to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id", "unknown")

        if resource_type == "Condition":
            onset = resource.get("onsetDateTime")
            recorded = resource.get("recordedDate")

            if onset and recorded:
                try:
                    onset_dt = datetime.fromisoformat(onset.replace("Z", "+00:00"))
                    recorded_dt = datetime.fromisoformat(recorded.replace("Z", "+00:00"))

                    if onset_dt > recorded_dt:
                        errors.append(
                            f"Condition/{resource_id}: onsetDateTime ({onset}) "
                            f"is after recordedDate ({recorded})"
                        )
                except ValueError as e:
                    errors.append(f"Condition/{resource_id}: invalid date format - {e}")

        elif resource_type == "DiagnosticReport":
            effective = resource.get("effectiveDateTime")
            issued = resource.get("issued")

            if effective and issued:
                try:
                    effective_dt = datetime.fromisoformat(effective.replace("Z", "+00:00"))
                    issued_dt = datetime.fromisoformat(issued.replace("Z", "+00:00"))

                    if effective_dt > issued_dt:
                        errors.append(
                            f"DiagnosticReport/{resource_id}: effectiveDateTime ({effective}) "
                            f"is after issued ({issued})"
                        )
                except ValueError as e:
                    errors.append(f"DiagnosticReport/{resource_id}: invalid date format - {e}")

    return errors


def validate_reference_integrity(bundle: dict) -> list[str]:
    """Validate that all references resolve to existing resources.

    Checks that all resource references point to resources that exist
    within the bundle.

    Args:
        bundle: The FHIR bundle to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Build set of available resource references
    available_refs = set()
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id")
        if resource_type and resource_id:
            available_refs.add(f"{resource_type}/{resource_id}")

    def check_reference(ref_obj: dict | None, source: str, field_name: str):
        """Check if a reference resolves to an existing resource."""
        if not isinstance(ref_obj, dict):
            return
        ref = ref_obj.get("reference", "")
        if ref and not ref.startswith("urn:uuid:"):
            if ref not in available_refs:
                errors.append(f"{source}.{field_name}: unresolved reference '{ref}'")

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "Unknown")
        resource_id = resource.get("id", "?")
        source = f"{resource_type}/{resource_id}"

        # Check common reference fields
        check_reference(resource.get("subject"), source, "subject")
        check_reference(resource.get("patient"), source, "patient")

        # Check array references
        for i, ref in enumerate(resource.get("result", [])):
            check_reference(ref, source, f"result[{i}]")

        for i, ref in enumerate(resource.get("derivedFrom", [])):
            check_reference(ref, source, f"derivedFrom[{i}]")

        for i, ref in enumerate(resource.get("imagingStudy", [])):
            check_reference(ref, source, f"imagingStudy[{i}]")

        for i, ref in enumerate(resource.get("basis", [])):
            check_reference(ref, source, f"basis[{i}]")

        # Check evidence references
        for i, evidence in enumerate(resource.get("evidence", [])):
            for j, detail in enumerate(evidence.get("detail", [])):
                check_reference(detail, source, f"evidence[{i}].detail[{j}]")

    return errors


def validate_synthea_radiology_overlap(
    bundle: dict,
    radiology_conditions: list[str],
    volume_name: str
) -> dict:
    """Measure overlap between Synthea-generated and radiology-extracted conditions.

    Helps assess whether Synthea modules generated relevant conditions.

    Args:
        bundle: The FHIR bundle
        radiology_conditions: List of condition names from radiology extraction
        volume_name: The volume name (used to identify pipeline-created resources)

    Returns:
        Dict with overlap metrics
    """
    # Identify Synthea conditions (those NOT manually created by the pipeline)
    synthea_conditions = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue

        # Check if this is a pipeline-created condition
        # Pipeline conditions have category "encounter-diagnosis" and were created at scan time
        is_pipeline_created = False
        for category in resource.get("category", []):
            for coding in category.get("coding", []):
                if coding.get("code") == "encounter-diagnosis":
                    is_pipeline_created = True
                    break

        if not is_pipeline_created:
            # This is a Synthea-generated condition
            code = resource.get("code", {})
            text = code.get("text", "")
            if not text:
                for coding in code.get("coding", []):
                    text = coding.get("display", "")
                    if text:
                        break
            if text:
                synthea_conditions.append(text.lower())

    # Normalize radiology conditions
    radiology_normalized = [c.lower() for c in radiology_conditions]

    # Find keyword overlaps (not exact match, but related terms)
    overlap_keywords = [
        "emphysema", "copd", "bronchi", "pulmonary", "lung",
        "cardio", "heart", "atherosclerosis", "calcif",
        "diabetes", "kidney", "renal",
        "arthritis", "osteo", "spondyl", "spine"
    ]

    overlaps = []
    for rad_cond in radiology_normalized:
        for synth_cond in synthea_conditions:
            # Check if they share significant keywords
            for keyword in overlap_keywords:
                if keyword in rad_cond and keyword in synth_cond:
                    overlaps.append({
                        "radiology": rad_cond,
                        "synthea": synth_cond,
                        "keyword": keyword
                    })
                    break

    # Calculate overlap percentage
    overlap_pct = (
        len(set(o["radiology"] for o in overlaps)) / max(len(radiology_conditions), 1) * 100
    )

    return {
        "synthea_condition_count": len(synthea_conditions),
        "radiology_condition_count": len(radiology_conditions),
        "overlap_count": len(overlaps),
        "overlap_percentage": round(overlap_pct, 1),
        "overlapping_conditions": overlaps[:10]  # Limit to first 10
    }


def validate_bundle(
    bundle: dict,
    radiology_conditions: list[str] | None = None,
    volume_name: str | None = None
) -> tuple[bool, dict]:
    """Perform comprehensive validation on the FHIR bundle.

    Args:
        bundle: The FHIR bundle to validate
        radiology_conditions: Optional list of condition names for overlap analysis
        volume_name: Optional volume name for identifying pipeline resources

    Returns:
        Tuple of (is_valid, validation_metrics)
    """
    metrics = {
        "is_valid": True,
        "entry_count": 0,
        "resource_types": {},
        "temporal_errors": [],
        "reference_errors": [],
        "overlap_metrics": None
    }

    if not isinstance(bundle, dict):
        logger.error("Bundle is not a valid JSON object")
        metrics["is_valid"] = False
        return False, metrics

    if bundle.get("resourceType") != "Bundle":
        logger.error("Resource is not a FHIR Bundle")
        metrics["is_valid"] = False
        return False, metrics

    entries = bundle.get("entry", [])
    if not entries:
        logger.error("Bundle has no entries")
        metrics["is_valid"] = False
        return False, metrics

    metrics["entry_count"] = len(entries)

    # Count resource types
    for entry in entries:
        resource_type = entry.get("resource", {}).get("resourceType", "Unknown")
        metrics["resource_types"][resource_type] = metrics["resource_types"].get(resource_type, 0) + 1

    # Check for required resources
    required = {"Patient", "DiagnosticReport", "ImagingStudy"}
    missing = required - set(metrics["resource_types"].keys())

    if missing:
        logger.warning(f"Bundle missing expected resources: {missing}")
        metrics["is_valid"] = False

    # Temporal consistency validation
    metrics["temporal_errors"] = validate_temporal_consistency(bundle)
    if metrics["temporal_errors"]:
        logger.warning(f"Temporal validation errors: {len(metrics['temporal_errors'])}")
        for error in metrics["temporal_errors"][:3]:  # Log first 3
            logger.warning(f"  - {error}")

    # Reference integrity validation
    metrics["reference_errors"] = validate_reference_integrity(bundle)
    if metrics["reference_errors"]:
        logger.warning(f"Reference integrity errors: {len(metrics['reference_errors'])}")
        for error in metrics["reference_errors"][:3]:  # Log first 3
            logger.warning(f"  - {error}")
        # Reference errors make the bundle invalid
        metrics["is_valid"] = False

    # Synthea-radiology overlap analysis (informational, doesn't affect validity)
    if radiology_conditions and volume_name:
        metrics["overlap_metrics"] = validate_synthea_radiology_overlap(
            bundle, radiology_conditions, volume_name
        )
        logger.info(
            f"Synthea-radiology overlap: {metrics['overlap_metrics']['overlap_percentage']}% "
            f"({metrics['overlap_metrics']['overlap_count']} matches)"
        )

    if metrics["is_valid"]:
        logger.info(
            f"Bundle validation passed: {metrics['entry_count']} entries, "
            f"types: {list(metrics['resource_types'].keys())}"
        )

    return metrics["is_valid"], metrics


# -----------------------------------------------------------------------------
# Pipeline Orchestration
# -----------------------------------------------------------------------------

@dataclass
class ProcessingResult:
    """Result of processing a single report."""
    report_name: str
    success: bool
    output_path: Optional[Path] = None
    extraction: Optional[dict] = None
    error: Optional[str] = None
    patient_fhir_id: Optional[str] = None
    conditions_count: int = 0
    validation_metrics: Optional[dict] = None


def get_patient_fhir_id(bundle: dict) -> Optional[str]:
    """Extract the Patient resource ID from a FHIR bundle."""
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            return resource.get("id")
    return None


async def process_single_report(
    client: AsyncOpenAI,
    report_path: Path,
    volume_path: Optional[Path],
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    copy_volumes: bool = False
) -> ProcessingResult:
    """Process a single radiology report through the pipeline.

    Creates a per-datapoint folder containing:
    - fhir.json: The FHIR bundle
    - volume.nii.gz: Symlink or copy of the CT volume (if volume exists)
    """

    report_name = report_path.stem
    logger.info(f"Processing: {report_name}")

    async with semaphore:
        try:
            # 1. Load radiology report
            with open(report_path, "r") as f:
                report = json.load(f)

            # 2. Extract clinical data via OpenAI
            extraction = await extract_with_openai(client, report)
            logger.info(
                f"Extracted {len(extraction.conditions)} conditions, "
                f"age range: {extraction.demographics.estimated_age_min}-{extraction.demographics.estimated_age_max}"
            )

            # 3. Create Synthea config
            config = create_synthea_config(extraction, report_name)

            # 4. Run Synthea (synchronous)
            synthea_bundle = run_synthea(config)

            if synthea_bundle is None:
                return ProcessingResult(
                    report_name=report_name,
                    success=False,
                    error="Synthea generation failed"
                )

            # 5. Merge radiology resources
            final_bundle = merge_radiology_resources(synthea_bundle, extraction, report)

            # 6. Validate bundle with comprehensive checks
            radiology_condition_names = [c.condition_name for c in extraction.conditions]
            is_valid, validation_metrics = validate_bundle(
                final_bundle,
                radiology_conditions=radiology_condition_names,
                volume_name=report.get("volume_name", report_name)
            )
            if not is_valid:
                logger.warning(f"Bundle validation failed for {report_name}")

            # 7. Create per-datapoint folder
            datapoint_dir = output_dir / report_name
            datapoint_dir.mkdir(parents=True, exist_ok=True)

            # 8. Save FHIR bundle as fhir.json
            fhir_path = datapoint_dir / "fhir.json"
            with open(fhir_path, "w") as f:
                json.dump(final_bundle, f, indent=2)

            logger.info(f"Saved FHIR bundle: {fhir_path}")

            # 9. Link or copy volume if it exists
            if volume_path and volume_path.exists():
                volume_dest = datapoint_dir / "volume.nii.gz"
                if volume_dest.exists() or volume_dest.is_symlink():
                    volume_dest.unlink()

                if copy_volumes:
                    shutil.copy2(volume_path, volume_dest)
                    logger.info(f"Copied volume: {volume_dest}")
                else:
                    # Create relative symlink
                    rel_path = os.path.relpath(volume_path, datapoint_dir)
                    volume_dest.symlink_to(rel_path)
                    logger.info(f"Symlinked volume: {volume_dest} -> {rel_path}")
            elif volume_path:
                logger.warning(f"Volume not found: {volume_path}")

            # 10. Copy radiology report files to combined folder
            report_json_dest = datapoint_dir / "report.json"
            report_txt_dest = datapoint_dir / "report.txt"

            # Copy the original report JSON
            shutil.copy2(report_path, report_json_dest)
            logger.info(f"Copied report.json: {report_json_dest}")

            # Copy or generate report.txt
            report_txt_source = report_path.with_suffix('.txt')
            if report_txt_source.exists():
                shutil.copy2(report_txt_source, report_txt_dest)
                logger.info(f"Copied report.txt: {report_txt_dest}")
            else:
                # Generate from JSON
                with open(report_txt_dest, 'w') as f:
                    f.write(f"Volume: {report.get('volume_name', 'Unknown')}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"CLINICAL INFORMATION:\n{report.get('clinical_information', 'N/A')}\n\n")
                    f.write(f"TECHNIQUE:\n{report.get('technique', 'N/A')}\n\n")
                    f.write(f"FINDINGS:\n{report.get('findings', 'N/A')}\n\n")
                    f.write(f"IMPRESSIONS:\n{report.get('impressions', 'N/A')}\n")
                logger.info(f"Generated report.txt: {report_txt_dest}")

            # Extract patient FHIR ID for manifest
            patient_fhir_id = get_patient_fhir_id(final_bundle)

            return ProcessingResult(
                report_name=report_name,
                success=True,
                output_path=datapoint_dir,
                extraction=extraction.model_dump(),
                patient_fhir_id=patient_fhir_id,
                conditions_count=len(extraction.conditions),
                validation_metrics=validation_metrics
            )

        except Exception as e:
            logger.error(f"Error processing {report_name}: {e}")
            return ProcessingResult(
                report_name=report_name,
                success=False,
                error=str(e)
            )


def find_volume_for_report(report_path: Path, volumes_dir: Path) -> Optional[Path]:
    """Find the corresponding volume file for a report."""
    report_stem = report_path.stem
    # Try common volume extensions
    for ext in [".nii.gz", ".nii"]:
        volume_path = volumes_dir / f"{report_stem}{ext}"
        if volume_path.exists():
            return volume_path
    return None


def generate_manifest(
    results: list[ProcessingResult],
    output_dir: Path
) -> Path:
    """Generate manifest.json with index of all data points."""
    manifest = {
        "created": datetime.utcnow().isoformat() + "Z",
        "total_patients": sum(1 for r in results if r.success),
        "patients": []
    }

    for result in results:
        if result.success:
            patient_entry = {
                "id": result.report_name,
                "folder": result.report_name,
                "fhir_path": f"{result.report_name}/fhir.json",
                "volume_path": f"{result.report_name}/volume.nii.gz",
                "report_json_path": f"{result.report_name}/report.json",
                "report_txt_path": f"{result.report_name}/report.txt",
                "patient_fhir_id": result.patient_fhir_id,
                "conditions_count": result.conditions_count
            }
            manifest["patients"].append(patient_entry)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated manifest: {manifest_path}")
    return manifest_path


async def process_all_reports(
    report_paths: list[Path],
    volumes_dir: Optional[Path],
    output_dir: Path,
    checkpoint_path: Path,
    copy_volumes: bool = False,
    max_concurrent: int = 3
) -> list[ProcessingResult]:
    """Process multiple reports with checkpoint/resume support."""

    # Load checkpoint if exists
    processed = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
            processed = set(checkpoint.get("processed", []))
        logger.info(f"Resuming from checkpoint: {len(processed)} already processed")

    # Filter out already processed
    remaining = [p for p in report_paths if p.stem not in processed]
    logger.info(f"Processing {len(remaining)} reports ({len(processed)} already done)")

    if not remaining:
        logger.info("All reports already processed")
        return []

    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process reports (sequentially due to Synthea being synchronous)
    results = []
    for report_path in remaining:
        # Find corresponding volume
        volume_path = None
        if volumes_dir and volumes_dir.exists():
            volume_path = find_volume_for_report(report_path, volumes_dir)

        result = await process_single_report(
            client,
            report_path,
            volume_path,
            output_dir,
            semaphore,
            copy_volumes=copy_volumes
        )
        results.append(result)

        # Update checkpoint after each report
        if result.success:
            processed.add(result.report_name)

        checkpoint_data = {
            "processed": list(processed),
            "last_updated": datetime.utcnow().isoformat()
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    return results


def save_processing_log(
    results: list[ProcessingResult],
    output_dir: Path
):
    """Save processing log with extraction results and errors."""

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_processed": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "results": []
    }

    for result in results:
        entry = {
            "report_name": result.report_name,
            "success": result.success
        }
        if result.output_path:
            entry["output_path"] = str(result.output_path)
        if result.extraction:
            entry["extraction"] = result.extraction
        if result.error:
            entry["error"] = result.error
        if result.validation_metrics:
            # Include key validation metrics
            entry["validation"] = {
                "is_valid": result.validation_metrics.get("is_valid"),
                "temporal_error_count": len(result.validation_metrics.get("temporal_errors", [])),
                "reference_error_count": len(result.validation_metrics.get("reference_errors", [])),
            }
            # Include overlap metrics if available
            overlap = result.validation_metrics.get("overlap_metrics")
            if overlap:
                entry["validation"]["synthea_radiology_overlap"] = {
                    "synthea_conditions": overlap.get("synthea_condition_count"),
                    "radiology_conditions": overlap.get("radiology_condition_count"),
                    "overlap_percentage": overlap.get("overlap_percentage")
                }
        log_data["results"].append(entry)

    log_path = output_dir / "processing_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Processing log saved: {log_path}")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic FHIR patient records from radiology reports"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Input directory containing volumes/ and reports/ (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for combined data (default: {data-dir}/combined)"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Process a single report (filename, e.g., train_1_a_1.json)"
    )
    parser.add_argument(
        "--copy-volumes",
        action="store_true",
        help="Copy volumes instead of symlinking (uses more disk space)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent OpenAI requests (default: 3)"
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Auto-detect subdirectories
    reports_dir = data_dir / "reports"
    volumes_dir = data_dir / "volumes"

    if not reports_dir.exists():
        logger.error(f"Reports directory not found: {reports_dir}")
        sys.exit(1)

    if not volumes_dir.exists():
        logger.warning(f"Volumes directory not found: {volumes_dir}")
        logger.warning("Proceeding without volume linking")
        volumes_dir = None

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "combined"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which reports to process
    if args.report:
        report_path = reports_dir / args.report
        if not report_path.exists():
            logger.error(f"Report not found: {report_path}")
            sys.exit(1)
        report_paths = [report_path]
    else:
        report_paths = sorted(reports_dir.glob("*.json"))

    if not report_paths:
        logger.error("No reports found to process")
        sys.exit(1)

    logger.info(f"Found {len(report_paths)} report(s) to process")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Volume handling: {'copy' if args.copy_volumes else 'symlink'}")

    # Check for Synthea JAR
    if not SYNTHEA_JAR.exists():
        logger.error(f"Synthea JAR not found: {SYNTHEA_JAR}")
        logger.error("Run fhir_creation.py first to download Synthea")
        sys.exit(1)

    # Check for Java
    try:
        subprocess.run(["java", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Java is required but not installed or not in PATH")
        sys.exit(1)

    # Create temp output directory for Synthea
    SYNTHEA_TEMP_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Checkpoint path
    checkpoint_path = output_dir / ".checkpoint.json"

    # Run the pipeline
    results = asyncio.run(
        process_all_reports(
            report_paths,
            volumes_dir,
            output_dir,
            checkpoint_path,
            copy_volumes=args.copy_volumes,
            max_concurrent=args.max_concurrent
        )
    )

    # Generate manifest for all successful results
    all_results = results.copy()

    # Include previously processed results from checkpoint for manifest
    if checkpoint_path.exists():
        # Re-scan output directory to include all successfully processed datapoints
        all_results = []
        for datapoint_dir in sorted(output_dir.iterdir()):
            if datapoint_dir.is_dir() and (datapoint_dir / "fhir.json").exists():
                # Load FHIR to get patient ID and conditions count
                with open(datapoint_dir / "fhir.json") as f:
                    bundle = json.load(f)
                patient_fhir_id = get_patient_fhir_id(bundle)
                conditions_count = sum(
                    1 for entry in bundle.get("entry", [])
                    if entry.get("resource", {}).get("resourceType") == "Condition"
                )
                all_results.append(ProcessingResult(
                    report_name=datapoint_dir.name,
                    success=True,
                    output_path=datapoint_dir,
                    patient_fhir_id=patient_fhir_id,
                    conditions_count=conditions_count
                ))

    if all_results:
        generate_manifest(all_results, output_dir)

    # Save processing log
    if results:
        save_processing_log(results, output_dir)

    # Summary
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    print("\n" + "=" * 50)
    print("Processing Complete")
    print("=" * 50)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
