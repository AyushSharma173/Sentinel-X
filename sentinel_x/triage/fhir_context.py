"""FHIR clinical context extraction for triage analysis."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import HIGH_RISK_CONDITIONS
from .logging import get_fhir_trace_logger

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


def extract_patient_demographics(
    fhir_bundle: Dict, patient_id: str = None
) -> Tuple[Optional[int], Optional[str], str]:
    """Extract age and gender from FHIR Patient resource.

    Args:
        fhir_bundle: FHIR Bundle containing Patient resource
        patient_id: Optional patient ID for logging

    Returns:
        Tuple of (age, gender, source_field)
    """
    trace_logger = get_fhir_trace_logger()
    age = None
    gender = None
    source_field = "not_found"

    entries = fhir_bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            gender = resource.get("gender")
            source_field = "Patient.gender"

            # Try to get age from birthDate
            birth_date = resource.get("birthDate")
            if birth_date:
                try:
                    from datetime import datetime
                    birth_year = int(birth_date.split("-")[0])
                    current_year = datetime.now().year
                    age = current_year - birth_year
                    source_field = "Patient.birthDate"
                except (ValueError, IndexError):
                    pass

            # Also check extensions for age
            extensions = resource.get("extension", [])
            for ext in extensions:
                if "age" in ext.get("url", "").lower():
                    age = ext.get("valueInteger", ext.get("valueDecimal"))
                    source_field = "Patient.extension[age]"
                    break

            break

    # Log demographics extraction
    if patient_id:
        trace_logger.log_demographics_extracted(
            patient_id=patient_id,
            age=age,
            gender=gender,
            source_field=source_field,
        )

    return age, gender, source_field


def extract_conditions(fhir_bundle: Dict, patient_id: str = None) -> List[str]:
    """Extract conditions from FHIR Condition resources.

    Args:
        fhir_bundle: FHIR Bundle containing Condition resources
        patient_id: Optional patient ID for logging

    Returns:
        List of condition descriptions
    """
    trace_logger = get_fhir_trace_logger()
    conditions = []
    entries = fhir_bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Condition":
            # Get condition text
            code = resource.get("code", {})
            text = code.get("text")
            source_field = "Condition.code.text"
            coding_system = None
            coding_code = None

            if not text:
                # Try to get from coding display
                codings = code.get("coding", [])
                for coding in codings:
                    if coding.get("display"):
                        text = coding["display"]
                        source_field = "Condition.code.coding.display"
                        coding_system = coding.get("system")
                        coding_code = coding.get("code")
                        break

            if text:
                conditions.append(text)

                # Log individual condition extraction
                if patient_id:
                    trace_logger.log_condition_extracted(
                        patient_id=patient_id,
                        condition=text,
                        source_field=source_field,
                        coding_system=coding_system,
                        coding_code=coding_code,
                    )

    # Log conditions summary
    if patient_id:
        trace_logger.log_conditions_summary(
            patient_id=patient_id,
            conditions=conditions,
            count=len(conditions),
        )

    return conditions


def extract_medications(fhir_bundle: Dict, patient_id: str = None) -> List[str]:
    """Extract medications from FHIR MedicationStatement/Request resources.

    Args:
        fhir_bundle: FHIR Bundle
        patient_id: Optional patient ID for logging

    Returns:
        List of medication names
    """
    trace_logger = get_fhir_trace_logger()
    medications = []
    entries = fhir_bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "")

        if resource_type in ("MedicationStatement", "MedicationRequest"):
            med_ref = resource.get("medicationCodeableConcept", {})
            text = med_ref.get("text")
            source_field = "medicationCodeableConcept.text"

            if not text:
                codings = med_ref.get("coding", [])
                for coding in codings:
                    if coding.get("display"):
                        text = coding["display"]
                        source_field = "medicationCodeableConcept.coding.display"
                        break

            if text:
                medications.append(text)

                # Log individual medication extraction
                if patient_id:
                    trace_logger.log_medication_extracted(
                        patient_id=patient_id,
                        medication=text,
                        source_field=source_field,
                        resource_type=resource_type,
                    )

    # Log medications summary
    if patient_id:
        trace_logger.log_medications_summary(
            patient_id=patient_id,
            medications=medications,
            count=len(medications),
        )

    return medications


def extract_report_content(
    report_data: Dict, patient_id: str = None
) -> Tuple[str, str, str]:
    """Extract findings and impressions from radiology report.

    Args:
        report_data: Report JSON data
        patient_id: Optional patient ID for logging

    Returns:
        Tuple of (findings, impressions, source_field)
    """
    trace_logger = get_fhir_trace_logger()
    findings = ""
    impressions = ""
    source_field = "not_found"

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
                            source_field = "DiagnosticReport.presentedForm"
                    if "IMPRESSION" in text.upper():
                        parts = text.upper().split("IMPRESSION")
                        if len(parts) > 1:
                            impressions = parts[1].strip()
                            source_field = "DiagnosticReport.presentedForm"
                except Exception:
                    pass

        # Check conclusion field
        if not impressions:
            impressions = report_data.get("conclusion", "")
            if impressions:
                source_field = "DiagnosticReport.conclusion"

    # Handle simple JSON format
    if not findings:
        findings = report_data.get("findings", "")
        if findings:
            source_field = "findings"
    if not impressions:
        impressions = report_data.get("impressions", report_data.get("impression", ""))
        if impressions and source_field == "not_found":
            source_field = "impressions"

    # Log report content extraction
    if patient_id:
        trace_logger.log_report_content_extracted(
            patient_id=patient_id,
            findings=findings,
            impressions=impressions,
            source_field=source_field,
        )

    return findings, impressions, source_field


def parse_fhir_context(report_path: Path, patient_id: str) -> PatientContext:
    """Parse FHIR bundle and report to extract clinical context.

    Args:
        report_path: Path to report JSON file
        patient_id: Patient identifier

    Returns:
        PatientContext with extracted data
    """
    logger.info(f"Parsing FHIR context for patient: {patient_id}")
    trace_logger = get_fhir_trace_logger()

    context = PatientContext(patient_id=patient_id)

    try:
        with open(report_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load report: {e}")
        trace_logger.log_parse_error(
            patient_id=patient_id,
            error=str(e),
            source=str(report_path),
        )
        return context

    # Check if this is a FHIR Bundle
    if data.get("resourceType") == "Bundle":
        # Analyze bundle structure
        entries = data.get("entry", [])
        resource_types: Dict[str, int] = {}
        for entry in entries:
            resource = entry.get("resource", {})
            rt = resource.get("resourceType", "unknown")
            resource_types[rt] = resource_types.get(rt, 0) + 1

        # Log bundle structure
        trace_logger.log_bundle_received(
            patient_id=patient_id,
            resource_types=resource_types,
            entry_count=len(entries),
        )

        # Extract demographics (updated to return source_field)
        age, gender, _ = extract_patient_demographics(data, patient_id)
        context.age = age
        context.gender = gender

        # Extract conditions
        context.conditions = extract_conditions(data, patient_id)

        # Extract medications
        context.medications = extract_medications(data, patient_id)

        # Identify risk factors
        context.risk_factors = identify_risk_factors(context.conditions)

        # Log risk factors
        trace_logger.log_risk_factors_summary(
            patient_id=patient_id,
            risk_factors=context.risk_factors,
            count=len(context.risk_factors),
        )

        # Find DiagnosticReport in bundle
        for entry in data.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "DiagnosticReport":
                findings, impressions, _ = extract_report_content(resource, patient_id)
                context.findings = findings
                context.impressions = impressions
                break
    else:
        # Handle simple report format
        findings, impressions, _ = extract_report_content(data, patient_id)
        context.findings = findings
        context.impressions = impressions

        # Extract demographics if present
        context.age = data.get("age")
        context.gender = data.get("gender")
        context.conditions = data.get("conditions", [])
        context.risk_factors = identify_risk_factors(context.conditions)

        # Log for simple format
        trace_logger.log_demographics_extracted(
            patient_id=patient_id,
            age=context.age,
            gender=context.gender,
            source_field="simple_json",
        )
        trace_logger.log_conditions_summary(
            patient_id=patient_id,
            conditions=context.conditions,
            count=len(context.conditions),
        )
        trace_logger.log_risk_factors_summary(
            patient_id=patient_id,
            risk_factors=context.risk_factors,
            count=len(context.risk_factors),
        )

    # Log context completion
    trace_logger.log_context_complete(
        patient_id=patient_id,
        age=context.age,
        gender=context.gender,
        conditions_count=len(context.conditions),
        medications_count=len(context.medications),
        risk_factors_count=len(context.risk_factors),
        has_findings=bool(context.findings),
        has_impressions=bool(context.impressions),
    )

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
