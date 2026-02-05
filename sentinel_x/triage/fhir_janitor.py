"""FHIR Janitor Module - Dense Clinical Stream Architecture.

This module transforms FHIR bundles into condensed, chronological narratives
that match MedGemma's training format (longitudinal patient narratives).

Components:
- GarbageCollector: Removes noise resources, extracts hidden diagnoses from claims
- NarrativeDecoder: Extracts text from Base64-encoded DiagnosticReports
- ResourceExtractors: Per-resource-type field extraction
- TimelineSerializer: Chronological output formatting
"""

import base64
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .age_utils import extract_age_from_patient_resource
from .config import (
    HIGH_RISK_CONDITIONS,
    JANITOR_CONDITIONAL_RESOURCES,
    JANITOR_DISCARD_RESOURCES,
    JANITOR_MAX_NARRATIVE_LENGTH,
    JANITOR_TARGET_MAX_TOKENS,
    JANITOR_UNDATED_LABEL,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TimelineEntry:
    """A single entry in the clinical timeline."""

    date: Optional[datetime]
    date_label: str  # "2024-01-15" or "[Historical/Undated]"
    category: str  # "Condition", "Medication", "Lab", etc.
    content: str  # Narrative text
    priority: int  # For same-date sorting (lower = higher priority)

    def __lt__(self, other: "TimelineEntry") -> bool:
        """Enable sorting by date then priority."""
        # Undated entries (date=None) come first
        if self.date is None and other.date is not None:
            return True
        if self.date is not None and other.date is None:
            return False
        if self.date == other.date:
            return self.priority < other.priority
        if self.date is None and other.date is None:
            return self.priority < other.priority
        return self.date < other.date


@dataclass
class ClinicalStream:
    """The complete processed clinical stream output."""

    patient_summary: str  # "71-year-old male"
    narrative: str  # Full chronological text
    token_estimate: int  # Approx token count
    extraction_warnings: List[str] = field(default_factory=list)
    active_medications: List[str] = field(default_factory=list)
    # Additional fields to replace PatientContext from fhir_context.py
    conditions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)  # Just names (no dosage)
    age: Optional[int] = None
    gender: Optional[str] = None
    findings: str = ""
    impressions: str = ""


@dataclass
class HistoricalDiagnosis:
    """A diagnosis extracted from billing records."""

    display: str
    code: Optional[str] = None
    system: Optional[str] = None


# =============================================================================
# Category Priority Constants
# =============================================================================

# Lower number = higher priority when sorting same-date entries
CATEGORY_PRIORITY = {
    "Encounter": 1,
    "Condition": 2,
    "Procedure": 3,
    "Medication": 4,
    "Lab": 5,
    "Narrative": 6,
}


# =============================================================================
# GarbageCollector
# =============================================================================


class GarbageCollector:
    """Removes noise resources and extracts hidden diagnoses from billing records."""

    def __init__(self) -> None:
        self.discard_types = JANITOR_DISCARD_RESOURCES
        self.conditional_types = JANITOR_CONDITIONAL_RESOURCES
        self.warnings: List[str] = []

    def process(
        self, entries: List[Dict[str, Any]], condition_codes: Set[str]
    ) -> Tuple[List[Dict[str, Any]], List[HistoricalDiagnosis]]:
        """Process bundle entries, removing noise and extracting hidden diagnoses.

        Args:
            entries: List of FHIR bundle entries
            condition_codes: Set of codes already present in Condition resources

        Returns:
            Tuple of (cleaned_entries, historical_diagnoses)
        """
        cleaned = []
        historical_diagnoses = []

        for entry in entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType", "")

            # Delete on sight
            if resource_type in self.discard_types:
                continue

            # Conditional handling for Claim/ExplanationOfBenefit
            if resource_type in self.conditional_types:
                hidden = self._extract_hidden_diagnoses(resource, condition_codes)
                historical_diagnoses.extend(hidden)
                continue  # Discard the claim itself

            cleaned.append(entry)

        return cleaned, historical_diagnoses

    def _extract_hidden_diagnoses(
        self, resource: Dict[str, Any], condition_codes: Set[str]
    ) -> List[HistoricalDiagnosis]:
        """Extract diagnoses from billing records not present in Conditions.

        Note: We only extract from the 'diagnosis' field, NOT from item[].productOrService
        which typically contains procedure codes, not diagnoses.

        Args:
            resource: Claim or ExplanationOfBenefit resource
            condition_codes: Codes already present in Condition resources

        Returns:
            List of historical diagnoses not found in Conditions
        """
        hidden = []

        # Only check diagnosis field (NOT item[].productOrService which contains procedures)
        diagnoses = resource.get("diagnosis", [])
        for diag in diagnoses:
            diag_codeable = diag.get("diagnosisCodeableConcept", {})
            codings = diag_codeable.get("coding", [])
            for coding in codings:
                code = coding.get("code", "")
                display = coding.get("display", "")
                system = coding.get("system", "")

                if code and code not in condition_codes and display:
                    hidden.append(
                        HistoricalDiagnosis(display=display, code=code, system=system)
                    )
                    condition_codes.add(code)

        return hidden

    def _is_diagnosis_code(self, system: str, code: str) -> bool:
        """Check if a coding system/code represents a diagnosis."""
        diagnosis_systems = {
            "http://snomed.info/sct",
            "http://hl7.org/fhir/sid/icd-10",
            "http://hl7.org/fhir/sid/icd-10-cm",
            "http://hl7.org/fhir/sid/icd-9",
            "http://hl7.org/fhir/sid/icd-9-cm",
        }
        return system in diagnosis_systems


# =============================================================================
# NarrativeDecoder
# =============================================================================


class NarrativeDecoder:
    """Extracts and parses Base64-encoded content from DiagnosticReports."""

    def __init__(self) -> None:
        self.warnings: List[str] = []

    def decode_report(
        self, resource: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
        """Decode a DiagnosticReport to extract findings and impressions.

        Args:
            resource: DiagnosticReport FHIR resource

        Returns:
            Tuple of (findings, impression, effective_date)
        """
        findings = None
        impression = None
        effective_date = None

        # Get effective date
        effective_dt = resource.get("effectiveDateTime")
        if effective_dt:
            effective_date = self._parse_datetime(effective_dt)

        # Check presentedForm for Base64 content
        presented_forms = resource.get("presentedForm", [])
        for form in presented_forms:
            content_type = form.get("contentType", "")
            if content_type.startswith("text/"):
                data = form.get("data", "")
                if data:
                    try:
                        text = base64.b64decode(data).decode("utf-8")
                        f, i = self._parse_report_text(text)
                        if f:
                            findings = f
                        if i:
                            impression = i
                    except Exception as e:
                        self.warnings.append(
                            f"Failed to decode Base64 in DiagnosticReport: {e}"
                        )

        # Fallback to conclusion field
        if not impression:
            conclusion = resource.get("conclusion", "")
            if conclusion:
                impression = conclusion

        return findings, impression, effective_date

    def _parse_report_text(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse FINDINGS and IMPRESSION sections from report text.

        Args:
            text: Full report text

        Returns:
            Tuple of (findings, impression)
        """
        findings = None
        impression = None

        # Case-insensitive parsing
        text_upper = text.upper()

        # Extract FINDINGS section
        if "FINDINGS:" in text_upper or "FINDINGS\n" in text_upper:
            # Find the start of findings
            findings_match = re.search(
                r"FINDINGS[:\s]*\n?(.*?)(?:IMPRESSION|CONCLUSION|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if findings_match:
                findings = findings_match.group(1).strip()
                # Truncate if too long
                if len(findings) > JANITOR_MAX_NARRATIVE_LENGTH:
                    findings = findings[:JANITOR_MAX_NARRATIVE_LENGTH] + "..."

        # Extract IMPRESSION section
        impression_match = re.search(
            r"(?:IMPRESSION|CONCLUSION)[:\s]*\n?(.*?)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if impression_match:
            impression = impression_match.group(1).strip()
            # Truncate if too long
            if len(impression) > JANITOR_MAX_NARRATIVE_LENGTH:
                impression = impression[:JANITOR_MAX_NARRATIVE_LENGTH] + "..."

        return findings, impression

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse a FHIR datetime string."""
        if not dt_str:
            return None
        try:
            # Extract just the date part (YYYY-MM-DD)
            if "T" in dt_str:
                date_part = dt_str.split("T")[0]
            else:
                date_part = dt_str[:10]
            return datetime.strptime(date_part, "%Y-%m-%d")
        except (ValueError, IndexError):
            return None


# =============================================================================
# Resource Extractors
# =============================================================================


class PatientExtractor:
    """Extracts patient demographics."""

    def extract(
        self, resource: Dict[str, Any]
    ) -> Tuple[str, Optional[int], Optional[str], List[str]]:
        """Extract patient summary from Patient resource.

        Args:
            resource: Patient FHIR resource

        Returns:
            Tuple of (patient_summary, age, gender, warnings)
        """
        warnings = []

        age, is_deceased, _ = extract_age_from_patient_resource(resource)
        gender = resource.get("gender", "")

        parts = []
        if age is not None:
            parts.append(f"{age}-year-old")
        if gender:
            parts.append(gender)

        if not parts:
            warnings.append("Unknown patient demographics")
            return "Unknown patient demographics", age, gender or None, warnings

        summary = " ".join(parts)
        if is_deceased:
            summary += " (deceased)"

        return summary, age, gender or None, warnings


class ConditionExtractor:
    """Extracts conditions from Condition resources."""

    def extract(
        self, resource: Dict[str, Any]
    ) -> Tuple[Optional[TimelineEntry], str, str]:
        """Extract a timeline entry from a Condition resource.

        Args:
            resource: Condition FHIR resource

        Returns:
            Tuple of (TimelineEntry or None, extracted_code, display_name)
        """
        # Get display text
        code = resource.get("code", {})
        display = code.get("text", "")
        extracted_code = ""

        if not display:
            codings = code.get("coding", [])
            for coding in codings:
                if coding.get("display"):
                    display = coding["display"]
                    extracted_code = coding.get("code", "")
                    break

        if not display:
            return None, "", ""

        # Get clinical status
        clinical_status = "unknown"
        status_obj = resource.get("clinicalStatus", {})
        status_codings = status_obj.get("coding", [])
        if status_codings:
            clinical_status = status_codings[0].get("code", "unknown")

        # Get onset date
        onset_dt = resource.get("onsetDateTime")
        date = self._parse_datetime(onset_dt) if onset_dt else None
        date_label = (
            date.strftime("%Y-%m-%d") if date else JANITOR_UNDATED_LABEL
        )

        # Build content
        content = f"Diagnosis: {display} ({clinical_status})"

        return (
            TimelineEntry(
                date=date,
                date_label=date_label,
                category="Condition",
                content=content,
                priority=CATEGORY_PRIORITY["Condition"],
            ),
            extracted_code,
            display,
        )

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse a FHIR datetime string."""
        if not dt_str:
            return None
        try:
            dt_str = dt_str.replace("Z", "+00:00")
            date_part = dt_str.split("T")[0] if "T" in dt_str else dt_str[:10]
            return datetime.strptime(date_part, "%Y-%m-%d")
        except (ValueError, IndexError):
            return None


class MedicationExtractor:
    """Extracts medications from MedicationRequest resources."""

    def __init__(self) -> None:
        self.warnings: List[str] = []

    def extract(
        self, resource: Dict[str, Any]
    ) -> Tuple[Optional[TimelineEntry], Optional[str], Optional[str]]:
        """Extract a timeline entry from a MedicationRequest resource.

        Args:
            resource: MedicationRequest FHIR resource

        Returns:
            Tuple of (TimelineEntry or None, active_medication_string or None, medication_name or None)
        """
        # Get medication name
        med_concept = resource.get("medicationCodeableConcept", {})
        name = med_concept.get("text", "")

        if not name:
            codings = med_concept.get("coding", [])
            for coding in codings:
                if coding.get("display"):
                    name = coding["display"]
                    break

        if not name:
            return None, None, None

        # Get status
        status = resource.get("status", "unknown")

        # Get dosage instruction
        dosage_text = self._extract_dosage(resource)

        # Get authored date
        authored_on = resource.get("authoredOn")
        date = self._parse_datetime(authored_on) if authored_on else None
        date_label = date.strftime("%Y-%m-%d") if date else JANITOR_UNDATED_LABEL

        # Build content
        if dosage_text:
            content = f"Medication: {name} {dosage_text} ({status})"
        else:
            content = f"Medication: {name} ({status})"

        # Build active medication string
        active_med_str = None
        if status == "active":
            if dosage_text:
                active_med_str = f"{name} {dosage_text}"
            else:
                active_med_str = name
            if date:
                active_med_str += f" (since {date_label})"

        return (
            TimelineEntry(
                date=date,
                date_label=date_label,
                category="Medication",
                content=content,
                priority=CATEGORY_PRIORITY["Medication"],
            ),
            active_med_str,
            name,
        )

    def _extract_dosage(self, resource: Dict[str, Any]) -> str:
        """Extract dosage instruction as human-readable string."""
        dosage_instructions = resource.get("dosageInstruction", [])
        if not dosage_instructions:
            return ""

        instruction = dosage_instructions[0]

        parts = []

        # Extract timing
        timing = instruction.get("timing", {})
        repeat = timing.get("repeat", {})

        frequency = repeat.get("frequency")
        period = repeat.get("period")
        period_unit = repeat.get("periodUnit", "")

        # Extract dose
        dose_and_rate = instruction.get("doseAndRate", [])
        if dose_and_rate:
            dose_qty = dose_and_rate[0].get("doseQuantity", {})
            dose_value = dose_qty.get("value")
            dose_unit = dose_qty.get("unit", "")

            if dose_value:
                parts.append(f"{dose_value} {dose_unit}".strip())

        # Build frequency string
        if frequency and period:
            unit_map = {"d": "daily", "wk": "weekly", "mo": "monthly", "h": "hourly"}
            period_str = unit_map.get(period_unit, period_unit)

            if frequency == 1 and period == 1:
                parts.append(f"once {period_str}")
            elif frequency == 1:
                parts.append(f"every {period} {period_unit}")
            else:
                parts.append(f"{frequency} times per {period_str}")

        if parts:
            return " ".join(parts)

        # Fallback to text field
        text = instruction.get("text", "")
        if text:
            return text

        self.warnings.append(f"Dosage unavailable for medication")
        return ""

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse a FHIR datetime string."""
        if not dt_str:
            return None
        try:
            dt_str = dt_str.replace("Z", "+00:00")
            date_part = dt_str.split("T")[0] if "T" in dt_str else dt_str[:10]
            return datetime.strptime(date_part, "%Y-%m-%d")
        except (ValueError, IndexError):
            return None


class ObservationExtractor:
    """Extracts labs from Observation resources."""

    # Interpretation code mapping
    INTERPRETATION_MAP = {
        "H": "High",
        "HH": "Critical High",
        "L": "Low",
        "LL": "Critical Low",
        "N": "Normal",
        "A": "Abnormal",
    }

    def extract(self, resource: Dict[str, Any]) -> Optional[TimelineEntry]:
        """Extract a timeline entry from an Observation resource.

        Args:
            resource: Observation FHIR resource

        Returns:
            TimelineEntry or None
        """
        # Get observation name
        code = resource.get("code", {})
        name = code.get("text", "")

        if not name:
            codings = code.get("coding", [])
            for coding in codings:
                if coding.get("display"):
                    name = coding["display"]
                    break

        if not name:
            return None

        # Get value
        value_qty = resource.get("valueQuantity", {})
        value = value_qty.get("value")
        unit = value_qty.get("unit", "")

        # Handle other value types
        if value is None:
            value_str = resource.get("valueString")
            if value_str:
                value = value_str
                unit = ""
            else:
                value_codeable = resource.get("valueCodeableConcept", {})
                value_text = value_codeable.get("text", "")
                if value_text:
                    value = value_text
                    unit = ""

        if value is None:
            return None

        # Get interpretation flag
        flag = ""
        interpretation = resource.get("interpretation", [])
        if interpretation:
            interp_codings = interpretation[0].get("coding", [])
            if interp_codings:
                interp_code = interp_codings[0].get("code", "")
                flag = self.INTERPRETATION_MAP.get(interp_code, "")

        # Get effective date
        effective_dt = resource.get("effectiveDateTime")
        date = self._parse_datetime(effective_dt) if effective_dt else None
        date_label = date.strftime("%Y-%m-%d") if date else JANITOR_UNDATED_LABEL

        # Build content
        if isinstance(value, float):
            value_str = f"{value:.2f}" if value != int(value) else str(int(value))
        else:
            value_str = str(value)

        content = f"Lab: {name} {value_str} {unit}".strip()
        if flag:
            content += f" ({flag})"

        return TimelineEntry(
            date=date,
            date_label=date_label,
            category="Lab",
            content=content,
            priority=CATEGORY_PRIORITY["Lab"],
        )

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse a FHIR datetime string."""
        if not dt_str:
            return None
        try:
            dt_str = dt_str.replace("Z", "+00:00")
            date_part = dt_str.split("T")[0] if "T" in dt_str else dt_str[:10]
            return datetime.strptime(date_part, "%Y-%m-%d")
        except (ValueError, IndexError):
            return None


class ProcedureExtractor:
    """Extracts procedures from Procedure resources."""

    def extract(self, resource: Dict[str, Any]) -> Optional[TimelineEntry]:
        """Extract a timeline entry from a Procedure resource.

        Args:
            resource: Procedure FHIR resource

        Returns:
            TimelineEntry or None
        """
        # Get procedure name
        code = resource.get("code", {})
        display = code.get("text", "")

        if not display:
            codings = code.get("coding", [])
            for coding in codings:
                if coding.get("display"):
                    display = coding["display"]
                    break

        if not display:
            return None

        # Get performed date
        performed_period = resource.get("performedPeriod", {})
        performed_dt = performed_period.get("start") or resource.get("performedDateTime")

        date = self._parse_datetime(performed_dt) if performed_dt else None
        date_label = date.strftime("%Y-%m-%d") if date else JANITOR_UNDATED_LABEL

        content = f"Procedure: {display}"

        return TimelineEntry(
            date=date,
            date_label=date_label,
            category="Procedure",
            content=content,
            priority=CATEGORY_PRIORITY["Procedure"],
        )

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse a FHIR datetime string."""
        if not dt_str:
            return None
        try:
            dt_str = dt_str.replace("Z", "+00:00")
            date_part = dt_str.split("T")[0] if "T" in dt_str else dt_str[:10]
            return datetime.strptime(date_part, "%Y-%m-%d")
        except (ValueError, IndexError):
            return None


class EncounterExtractor:
    """Extracts encounters from Encounter resources."""

    def extract(self, resource: Dict[str, Any]) -> Optional[TimelineEntry]:
        """Extract a timeline entry from an Encounter resource.

        Args:
            resource: Encounter FHIR resource

        Returns:
            TimelineEntry or None
        """
        # Get encounter type
        types = resource.get("type", [])
        type_display = ""
        if types:
            type_coding = types[0].get("coding", [])
            if type_coding:
                type_display = type_coding[0].get("display", "")
            if not type_display:
                type_display = types[0].get("text", "")

        if not type_display:
            # Fallback to class
            encounter_class = resource.get("class", {})
            type_display = encounter_class.get("display", encounter_class.get("code", ""))

        if not type_display:
            type_display = "Clinical encounter"

        # Get period start
        period = resource.get("period", {})
        start_dt = period.get("start")

        date = self._parse_datetime(start_dt) if start_dt else None
        date_label = date.strftime("%Y-%m-%d") if date else JANITOR_UNDATED_LABEL

        # Get reason if available
        reason_codes = resource.get("reasonCode", [])
        reason_text = ""
        if reason_codes:
            reason_coding = reason_codes[0].get("coding", [])
            if reason_coding:
                reason_text = reason_coding[0].get("display", "")

        content = f"Encounter: {type_display}"
        if reason_text:
            content += f" (reason: {reason_text})"

        return TimelineEntry(
            date=date,
            date_label=date_label,
            category="Encounter",
            content=content,
            priority=CATEGORY_PRIORITY["Encounter"],
        )

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse a FHIR datetime string."""
        if not dt_str:
            return None
        try:
            dt_str = dt_str.replace("Z", "+00:00")
            date_part = dt_str.split("T")[0] if "T" in dt_str else dt_str[:10]
            return datetime.strptime(date_part, "%Y-%m-%d")
        except (ValueError, IndexError):
            return None


# =============================================================================
# TimelineSerializer
# =============================================================================


class TimelineSerializer:
    """Serializes timeline entries into chronological narrative text."""

    def serialize(
        self,
        patient_summary: str,
        entries: List[TimelineEntry],
        active_medications: List[str],
    ) -> str:
        """Serialize timeline entries into formatted narrative.

        Args:
            patient_summary: Patient demographics summary
            entries: List of TimelineEntry objects
            active_medications: List of active medication strings

        Returns:
            Formatted narrative text
        """
        # Sort entries
        sorted_entries = sorted(entries)

        lines = []

        # Patient section
        lines.append("== PATIENT ==")
        lines.append(patient_summary)
        lines.append("")

        # Clinical timeline section
        lines.append("== CLINICAL TIMELINE ==")
        lines.append("")

        # Group by date label
        current_date_label = None
        for entry in sorted_entries:
            if entry.date_label != current_date_label:
                if current_date_label is not None:
                    lines.append("")
                lines.append(entry.date_label)
                current_date_label = entry.date_label

            lines.append(f"- {entry.content}")

        # Active medications section
        if active_medications:
            lines.append("")
            lines.append("== ACTIVE MEDICATIONS ==")
            for med in active_medications:
                lines.append(f"- {med}")

        return "\n".join(lines)


# =============================================================================
# Main FHIRJanitor Class
# =============================================================================


class FHIRJanitor:
    """Main orchestrator for FHIR bundle processing.

    Transforms FHIR bundles into condensed clinical narratives.
    """

    def __init__(self) -> None:
        self.garbage_collector = GarbageCollector()
        self.narrative_decoder = NarrativeDecoder()
        self.patient_extractor = PatientExtractor()
        self.condition_extractor = ConditionExtractor()
        self.medication_extractor = MedicationExtractor()
        self.observation_extractor = ObservationExtractor()
        self.procedure_extractor = ProcedureExtractor()
        self.encounter_extractor = EncounterExtractor()
        self.timeline_serializer = TimelineSerializer()

    def process_bundle(self, fhir_bundle: Dict[str, Any]) -> ClinicalStream:
        """Process a FHIR bundle into a clinical stream.

        Args:
            fhir_bundle: FHIR Bundle dictionary

        Returns:
            ClinicalStream with processed narrative
        """
        warnings: List[str] = []
        timeline_entries: List[TimelineEntry] = []
        active_medications: List[str] = []
        patient_summary = "Unknown patient demographics"
        condition_codes: Set[str] = set()

        # Additional fields for replacing PatientContext
        conditions: List[str] = []
        medications: List[str] = []
        age: Optional[int] = None
        gender: Optional[str] = None
        all_findings: List[str] = []
        all_impressions: List[str] = []

        # Validate bundle
        if not fhir_bundle or fhir_bundle.get("resourceType") != "Bundle":
            warnings.append("Invalid or missing FHIR Bundle")
            return ClinicalStream(
                patient_summary=patient_summary,
                narrative="",
                token_estimate=0,
                extraction_warnings=warnings,
            )

        entries = fhir_bundle.get("entry", [])

        # First pass: collect all condition codes
        for entry in entries:
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Condition":
                code_obj = resource.get("code", {})
                codings = code_obj.get("coding", [])
                for coding in codings:
                    if coding.get("code"):
                        condition_codes.add(coding["code"])

        # Run garbage collector
        cleaned_entries, historical_diagnoses = self.garbage_collector.process(
            entries, condition_codes
        )

        # Add historical diagnoses as timeline entries and to conditions list
        for hd in historical_diagnoses:
            timeline_entries.append(
                TimelineEntry(
                    date=None,
                    date_label=JANITOR_UNDATED_LABEL,
                    category="Condition",
                    content=f"Historical Diagnosis: {hd.display} (from billing records)",
                    priority=CATEGORY_PRIORITY["Condition"],
                )
            )
            conditions.append(hd.display)

        # Process each resource type
        for entry in cleaned_entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType", "")

            if resource_type == "Patient":
                patient_summary, age, gender, patient_warnings = (
                    self.patient_extractor.extract(resource)
                )
                warnings.extend(patient_warnings)

            elif resource_type == "Condition":
                timeline_entry, _, display_name = self.condition_extractor.extract(
                    resource
                )
                if timeline_entry:
                    timeline_entries.append(timeline_entry)
                if display_name:
                    conditions.append(display_name)

            elif resource_type == "MedicationRequest":
                timeline_entry, active_med, med_name = (
                    self.medication_extractor.extract(resource)
                )
                if timeline_entry:
                    timeline_entries.append(timeline_entry)
                if active_med:
                    active_medications.append(active_med)
                if med_name:
                    medications.append(med_name)
                warnings.extend(self.medication_extractor.warnings)
                self.medication_extractor.warnings.clear()

            elif resource_type == "Observation":
                timeline_entry = self.observation_extractor.extract(resource)
                if timeline_entry:
                    timeline_entries.append(timeline_entry)

            elif resource_type == "Procedure":
                timeline_entry = self.procedure_extractor.extract(resource)
                if timeline_entry:
                    timeline_entries.append(timeline_entry)

            elif resource_type == "Encounter":
                timeline_entry = self.encounter_extractor.extract(resource)
                if timeline_entry:
                    timeline_entries.append(timeline_entry)

            elif resource_type == "DiagnosticReport":
                findings, impression, effective_date = (
                    self.narrative_decoder.decode_report(resource)
                )
                if findings:
                    all_findings.append(findings)
                if impression:
                    all_impressions.append(impression)

                if findings or impression:
                    date_label = (
                        effective_date.strftime("%Y-%m-%d")
                        if effective_date
                        else JANITOR_UNDATED_LABEL
                    )

                    narrative_parts = []
                    if findings:
                        narrative_parts.append(f"FINDINGS: {findings}")
                    if impression:
                        narrative_parts.append(f"IMPRESSION: {impression}")

                    content = "Narrative (Report):\n  " + "\n  ".join(narrative_parts)

                    timeline_entries.append(
                        TimelineEntry(
                            date=effective_date,
                            date_label=date_label,
                            category="Narrative",
                            content=content,
                            priority=CATEGORY_PRIORITY["Narrative"],
                        )
                    )

        # Collect decoder warnings
        warnings.extend(self.narrative_decoder.warnings)

        # Identify risk factors from conditions
        risk_factors = self._identify_risk_factors(conditions)

        # Serialize timeline
        narrative = self.timeline_serializer.serialize(
            patient_summary, timeline_entries, active_medications
        )

        # Estimate token count (rough approximation: 1 token ~ 4 characters)
        token_estimate = len(narrative) // 4

        # Warn if over limit
        if token_estimate > JANITOR_TARGET_MAX_TOKENS:
            warnings.append(
                f"Narrative exceeds target token limit: ~{token_estimate} tokens"
            )

        return ClinicalStream(
            patient_summary=patient_summary,
            narrative=narrative,
            token_estimate=token_estimate,
            extraction_warnings=warnings,
            active_medications=active_medications,
            conditions=conditions,
            risk_factors=risk_factors,
            medications=medications,
            age=age,
            gender=gender,
            findings="\n\n".join(all_findings),
            impressions="\n\n".join(all_impressions),
        )

    def _identify_risk_factors(self, conditions: List[str]) -> List[str]:
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
