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
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
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


# -----------------------------------------------------------------------------
# OpenAI Extraction
# -----------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a medical data extraction specialist. Your task is to analyze radiology reports and extract structured clinical information.

For each report, you must:
1. Identify all medical conditions mentioned in the findings and impressions
2. Provide SNOMED-CT codes for conditions when possible (common codes below)
3. Infer patient demographics (age range, gender if determinable) based on clinical patterns
4. Map conditions to Synthea modules for synthetic patient generation

Common SNOMED-CT codes:
- Emphysema: 87433001
- Bronchiectasis: 12295008
- Atherosclerosis: 38716007
- Cardiomegaly: 8186001
- Atelectasis: 46621007
- Spondylosis: 75320002
- Osteoarthritis: 396275006
- Pleural effusion: 60046008
- Pulmonary nodule: 427359005
- Calcification: 82650004
- Chronic kidney disease: 709044004
- Cholelithiasis: 235919008
- Scoliosis: 298382003

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


def create_synthea_config(extraction: RadiologyExtraction) -> SyntheaConfig:
    """Create Synthea configuration from extracted data."""
    return SyntheaConfig(
        age_min=extraction.demographics.estimated_age_min,
        age_max=extraction.demographics.estimated_age_max,
        gender=extraction.demographics.gender_hint,
        modules=extraction.synthea_modules
    )


def run_synthea(config: SyntheaConfig) -> Optional[dict]:
    """Run Synthea to generate a base FHIR bundle."""

    if not SYNTHEA_JAR.exists():
        logger.error(f"Synthea JAR not found: {SYNTHEA_JAR}")
        return None

    # Create temp output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous output from temp directory
    for f in config.output_dir.glob("*.json"):
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
    ]

    # Add gender if specified
    if config.gender:
        cmd.extend(["-g", config.gender])

    # Add state
    cmd.append(config.state)

    logger.info(f"Running Synthea: age={config.age_min}-{config.age_max}, gender={config.gender or 'any'}")
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
    report_datetime: str
) -> dict:
    """Create a DiagnosticReport resource (US Core DiagnosticReport for Report and Note)."""

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

    # Add conclusion codes if available
    if conclusion_codes:
        diagnostic_report["conclusionCode"] = conclusion_codes

    return diagnostic_report


def create_condition_resource(
    patient_ref: str,
    condition: ExtractedCondition,
    onset_datetime: str
) -> dict:
    """Create a Condition resource (US Core Condition)."""

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

    # Create DiagnosticReport
    diagnostic_report = create_diagnostic_report(
        patient_ref,
        imaging_study_ref,
        report,
        extraction,
        now
    )

    # Create Condition resources for extracted conditions
    conditions = []
    for condition in extraction.conditions:
        condition_resource = create_condition_resource(patient_ref, condition, now)
        conditions.append(condition_resource)

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
        }
    ]

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
        f"ImagingStudy, DiagnosticReport, {len(conditions)} Conditions"
    )

    return synthea_bundle


# -----------------------------------------------------------------------------
# Bundle Validation
# -----------------------------------------------------------------------------

def validate_bundle(bundle: dict) -> bool:
    """Perform basic validation on the FHIR bundle."""

    if not isinstance(bundle, dict):
        logger.error("Bundle is not a valid JSON object")
        return False

    if bundle.get("resourceType") != "Bundle":
        logger.error("Resource is not a FHIR Bundle")
        return False

    entries = bundle.get("entry", [])
    if not entries:
        logger.error("Bundle has no entries")
        return False

    # Check for required resources
    resource_types = {
        entry.get("resource", {}).get("resourceType")
        for entry in entries
    }

    required = {"Patient", "DiagnosticReport", "ImagingStudy"}
    missing = required - resource_types

    if missing:
        logger.warning(f"Bundle missing expected resources: {missing}")
        return False

    logger.info(f"Bundle validation passed: {len(entries)} entries, resources: {resource_types}")
    return True


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
            config = create_synthea_config(extraction)

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

            # 6. Validate bundle
            is_valid = validate_bundle(final_bundle)
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

            # Extract patient FHIR ID for manifest
            patient_fhir_id = get_patient_fhir_id(final_bundle)

            return ProcessingResult(
                report_name=report_name,
                success=True,
                output_path=datapoint_dir,
                extraction=extraction.model_dump(),
                patient_fhir_id=patient_fhir_id,
                conditions_count=len(extraction.conditions)
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
