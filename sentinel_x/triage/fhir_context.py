"""FHIR clinical context extraction for triage analysis."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import HIGH_RISK_CONDITIONS

logger = logging.getLogger(__name__)


@dataclass
class PatientContext:
    """Extracted patient clinical context."""
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    findings: str = ""
    impressions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "age": self.age,
            "gender": self.gender,
            "conditions": self.conditions,
            "risk_factors": self.risk_factors,
            "medications": self.medications,
            "allergies": self.allergies,
            "findings": self.findings,
            "impressions": self.impressions,
        }


def identify_risk_factors(conditions: List[str]) -> List[str]:
    """Identify high-risk conditions from patient history.

    Args:
        conditions: List of condition descriptions

    Returns:
        List of identified risk factors
    """
    risk_factors = []

    for condition in conditions:
        condition_lower = condition.lower()
        for risk_keyword in HIGH_RISK_CONDITIONS:
            if risk_keyword in condition_lower:
                risk_factors.append(condition)
                break

    return risk_factors


def extract_patient_demographics(fhir_bundle: Dict) -> tuple:
    """Extract age and gender from FHIR Patient resource.

    Args:
        fhir_bundle: FHIR Bundle containing Patient resource

    Returns:
        Tuple of (age, gender)
    """
    age = None
    gender = None

    entries = fhir_bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            gender = resource.get("gender")

            # Try to get age from birthDate
            birth_date = resource.get("birthDate")
            if birth_date:
                try:
                    from datetime import datetime
                    birth_year = int(birth_date.split("-")[0])
                    current_year = datetime.now().year
                    age = current_year - birth_year
                except (ValueError, IndexError):
                    pass

            # Also check extensions for age
            extensions = resource.get("extension", [])
            for ext in extensions:
                if "age" in ext.get("url", "").lower():
                    age = ext.get("valueInteger", ext.get("valueDecimal"))
                    break

            break

    return age, gender


def extract_conditions(fhir_bundle: Dict) -> List[str]:
    """Extract conditions from FHIR Condition resources.

    Args:
        fhir_bundle: FHIR Bundle containing Condition resources

    Returns:
        List of condition descriptions
    """
    conditions = []
    entries = fhir_bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Condition":
            # Get condition text
            code = resource.get("code", {})
            text = code.get("text")

            if not text:
                # Try to get from coding display
                codings = code.get("coding", [])
                for coding in codings:
                    if coding.get("display"):
                        text = coding["display"]
                        break

            if text:
                conditions.append(text)

    return conditions


def extract_medications(fhir_bundle: Dict) -> List[str]:
    """Extract medications from FHIR MedicationStatement/Request resources.

    Args:
        fhir_bundle: FHIR Bundle

    Returns:
        List of medication names
    """
    medications = []
    entries = fhir_bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "")

        if resource_type in ("MedicationStatement", "MedicationRequest"):
            med_ref = resource.get("medicationCodeableConcept", {})
            text = med_ref.get("text")

            if not text:
                codings = med_ref.get("coding", [])
                for coding in codings:
                    if coding.get("display"):
                        text = coding["display"]
                        break

            if text:
                medications.append(text)

    return medications


def extract_report_content(report_data: Dict) -> tuple:
    """Extract findings and impressions from radiology report.

    Args:
        report_data: Report JSON data

    Returns:
        Tuple of (findings, impressions)
    """
    findings = ""
    impressions = ""

    # Handle FHIR DiagnosticReport format
    if report_data.get("resourceType") == "DiagnosticReport":
        # Check presentedForm for text content
        presented = report_data.get("presentedForm", [])
        for form in presented:
            if form.get("contentType") == "text/plain":
                import base64
                data = form.get("data", "")
                try:
                    text = base64.b64decode(data).decode("utf-8")
                    if "FINDINGS:" in text.upper():
                        parts = text.upper().split("FINDINGS:")
                        if len(parts) > 1:
                            findings = parts[1].split("IMPRESSION")[0].strip()
                    if "IMPRESSION" in text.upper():
                        parts = text.upper().split("IMPRESSION")
                        if len(parts) > 1:
                            impressions = parts[1].strip()
                except Exception:
                    pass

        # Check conclusion field
        if not impressions:
            impressions = report_data.get("conclusion", "")

    # Handle simple JSON format
    if not findings:
        findings = report_data.get("findings", "")
    if not impressions:
        impressions = report_data.get("impressions", report_data.get("impression", ""))

    return findings, impressions


def parse_fhir_context(report_path: Path, patient_id: str) -> PatientContext:
    """Parse FHIR bundle and report to extract clinical context.

    Args:
        report_path: Path to report JSON file
        patient_id: Patient identifier

    Returns:
        PatientContext with extracted data
    """
    logger.info(f"Parsing FHIR context for patient: {patient_id}")

    context = PatientContext(patient_id=patient_id)

    try:
        with open(report_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load report: {e}")
        return context

    # Check if this is a FHIR Bundle
    if data.get("resourceType") == "Bundle":
        context.age, context.gender = extract_patient_demographics(data)
        context.conditions = extract_conditions(data)
        context.medications = extract_medications(data)
        context.risk_factors = identify_risk_factors(context.conditions)

        # Find DiagnosticReport in bundle
        for entry in data.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "DiagnosticReport":
                context.findings, context.impressions = extract_report_content(resource)
                break
    else:
        # Handle simple report format
        context.findings, context.impressions = extract_report_content(data)

        # Extract demographics if present
        context.age = data.get("age")
        context.gender = data.get("gender")
        context.conditions = data.get("conditions", [])
        context.risk_factors = identify_risk_factors(context.conditions)

    logger.info(f"Extracted context: {len(context.conditions)} conditions, "
                f"{len(context.risk_factors)} risk factors")

    return context


def format_context_for_prompt(context: PatientContext) -> str:
    """Format patient context for inclusion in MedGemma prompt.

    Args:
        context: Patient clinical context

    Returns:
        Formatted text string
    """
    lines = ["## EHR Clinical Context"]

    # Demographics
    demo_parts = []
    if context.age:
        demo_parts.append(f"{context.age} year old")
    if context.gender:
        demo_parts.append(context.gender)
    if demo_parts:
        lines.append(f"**Demographics:** {' '.join(demo_parts)}")

    # Conditions
    if context.conditions:
        lines.append(f"**Medical History:** {', '.join(context.conditions)}")

    # Risk factors
    if context.risk_factors:
        lines.append(f"**High-Risk Factors:** {', '.join(context.risk_factors)}")

    # Medications
    if context.medications:
        lines.append(f"**Current Medications:** {', '.join(context.medications)}")

    # Report content
    if context.findings:
        lines.append(f"\n## Radiology Report Findings\n{context.findings}")

    if context.impressions:
        lines.append(f"\n## Radiology Report Impressions\n{context.impressions}")

    return "\n".join(lines)
