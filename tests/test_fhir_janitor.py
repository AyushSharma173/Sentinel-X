"""Unit tests for FHIR Janitor module."""

import base64
import pytest
from datetime import datetime

from sentinel_x.triage.fhir_janitor import (
    FHIRJanitor,
    GarbageCollector,
    NarrativeDecoder,
    PatientExtractor,
    ConditionExtractor,
    MedicationExtractor,
    ObservationExtractor,
    ProcedureExtractor,
    EncounterExtractor,
    TimelineSerializer,
    TimelineEntry,
    ClinicalStream,
    HistoricalDiagnosis,
    CATEGORY_PRIORITY,
)


class TestGarbageCollector:
    """Tests for GarbageCollector class."""

    def test_discard_provenance_resource(self):
        """Test that Provenance resources are discarded."""
        entries = [
            {"resource": {"resourceType": "Patient", "id": "123"}},
            {"resource": {"resourceType": "Provenance", "id": "456"}},
        ]
        gc = GarbageCollector()
        cleaned, _ = gc.process(entries, set())

        assert len(cleaned) == 1
        assert cleaned[0]["resource"]["resourceType"] == "Patient"

    def test_discard_organization_resource(self):
        """Test that Organization resources are discarded."""
        entries = [
            {"resource": {"resourceType": "Organization", "id": "org1"}},
            {"resource": {"resourceType": "Condition", "id": "cond1"}},
        ]
        gc = GarbageCollector()
        cleaned, _ = gc.process(entries, set())

        assert len(cleaned) == 1
        assert cleaned[0]["resource"]["resourceType"] == "Condition"

    def test_discard_all_noise_resources(self):
        """Test that all noise resource types are discarded."""
        noise_types = [
            "Provenance",
            "Organization",
            "PractitionerRole",
            "Coverage",
            "Device",
        ]
        entries = [{"resource": {"resourceType": rt, "id": f"{rt}_1"}} for rt in noise_types]
        entries.append({"resource": {"resourceType": "Patient", "id": "p1"}})

        gc = GarbageCollector()
        cleaned, _ = gc.process(entries, set())

        assert len(cleaned) == 1
        assert cleaned[0]["resource"]["resourceType"] == "Patient"

    def test_extract_hidden_diagnosis_from_eob(self):
        """Test extraction of hidden diagnoses from ExplanationOfBenefit.

        Note: Diagnoses are extracted from the 'diagnosis' field,
        NOT from item[].productOrService (which contains procedures).
        """
        entries = [
            {
                "resource": {
                    "resourceType": "ExplanationOfBenefit",
                    "id": "eob1",
                    "diagnosis": [
                        {
                            "diagnosisCodeableConcept": {
                                "coding": [
                                    {
                                        "system": "http://snomed.info/sct",
                                        "code": "38341003",
                                        "display": "Hypertensive disorder",
                                    }
                                ]
                            }
                        }
                    ],
                }
            },
        ]

        gc = GarbageCollector()
        existing_codes = set()
        cleaned, historical = gc.process(entries, existing_codes)

        # EOB should be discarded
        assert len(cleaned) == 0
        # But diagnosis should be extracted
        assert len(historical) == 1
        assert historical[0].display == "Hypertensive disorder"
        assert historical[0].code == "38341003"

    def test_skip_existing_condition_codes(self):
        """Test that diagnoses already in Conditions are not extracted."""
        entries = [
            {
                "resource": {
                    "resourceType": "ExplanationOfBenefit",
                    "id": "eob1",
                    "diagnosis": [
                        {
                            "diagnosisCodeableConcept": {
                                "coding": [
                                    {
                                        "system": "http://snomed.info/sct",
                                        "code": "38341003",
                                        "display": "Hypertensive disorder",
                                    }
                                ]
                            }
                        }
                    ],
                }
            },
        ]

        gc = GarbageCollector()
        existing_codes = {"38341003"}  # Already exists
        cleaned, historical = gc.process(entries, existing_codes)

        # Should not extract duplicate
        assert len(historical) == 0

    def test_discard_claim_after_extraction(self):
        """Test that Claim resources are discarded after extraction."""
        entries = [
            {
                "resource": {
                    "resourceType": "Claim",
                    "id": "claim1",
                    "diagnosis": [
                        {
                            "diagnosisCodeableConcept": {
                                "coding": [
                                    {
                                        "system": "http://snomed.info/sct",
                                        "code": "73211009",
                                        "display": "Diabetes mellitus",
                                    }
                                ]
                            }
                        }
                    ],
                }
            },
        ]

        gc = GarbageCollector()
        cleaned, historical = gc.process(entries, set())

        # Claim should be discarded
        assert len(cleaned) == 0
        # Diagnosis should be extracted
        assert len(historical) == 1
        assert historical[0].display == "Diabetes mellitus"


class TestNarrativeDecoder:
    """Tests for NarrativeDecoder class."""

    def test_decode_base64_findings_and_impression(self):
        """Test decoding Base64 report with both findings and impression."""
        report_text = """
FINDINGS:
Multiple filling defects in pulmonary arteries.
No pleural effusion.

IMPRESSION:
Acute pulmonary embolism.
"""
        encoded = base64.b64encode(report_text.encode("utf-8")).decode("utf-8")

        resource = {
            "resourceType": "DiagnosticReport",
            "effectiveDateTime": "2024-06-01T10:00:00Z",
            "presentedForm": [{"contentType": "text/plain", "data": encoded}],
        }

        decoder = NarrativeDecoder()
        findings, impression, date = decoder.decode_report(resource)

        assert findings is not None
        assert "filling defects" in findings
        assert impression is not None
        assert "pulmonary embolism" in impression
        assert date == datetime(2024, 6, 1)

    def test_fallback_to_conclusion(self):
        """Test fallback to conclusion field when no Base64 content."""
        resource = {
            "resourceType": "DiagnosticReport",
            "effectiveDateTime": "2024-06-01",
            "conclusion": "Normal chest CT.",
        }

        decoder = NarrativeDecoder()
        findings, impression, date = decoder.decode_report(resource)

        assert findings is None
        assert impression == "Normal chest CT."

    def test_invalid_base64_handling(self):
        """Test handling of invalid Base64 data."""
        resource = {
            "resourceType": "DiagnosticReport",
            "presentedForm": [{"contentType": "text/plain", "data": "not_valid_base64!!"}],
        }

        decoder = NarrativeDecoder()
        findings, impression, _ = decoder.decode_report(resource)

        # Should not crash, should add warning
        assert len(decoder.warnings) > 0

    def test_truncate_long_findings(self):
        """Test that very long findings are truncated."""
        long_text = "FINDINGS:\n" + "A" * 1000 + "\nIMPRESSION: Normal"
        encoded = base64.b64encode(long_text.encode("utf-8")).decode("utf-8")

        resource = {
            "resourceType": "DiagnosticReport",
            "presentedForm": [{"contentType": "text/plain", "data": encoded}],
        }

        decoder = NarrativeDecoder()
        findings, _, _ = decoder.decode_report(resource)

        assert findings is not None
        assert len(findings) <= 503  # 500 + "..."


class TestPatientExtractor:
    """Tests for PatientExtractor class."""

    def test_extract_demographics(self):
        """Test extraction of patient demographics."""
        resource = {
            "resourceType": "Patient",
            "id": "123",
            "birthDate": "1954-04-16",
            "gender": "male",
        }

        extractor = PatientExtractor()
        summary, warnings = extractor.extract(resource)

        assert "male" in summary
        # Age will vary based on current date
        assert "-year-old" in summary
        assert len(warnings) == 0

    def test_deceased_patient(self):
        """Test extraction for deceased patient."""
        resource = {
            "resourceType": "Patient",
            "id": "123",
            "birthDate": "1942-08-04",
            "deceasedDateTime": "2008-09-28T06:00:40+00:00",
            "gender": "male",
        }

        extractor = PatientExtractor()
        summary, warnings = extractor.extract(resource)

        assert "66-year-old" in summary
        assert "male" in summary
        assert "(deceased)" in summary

    def test_missing_demographics(self):
        """Test handling of missing demographics."""
        resource = {"resourceType": "Patient", "id": "123"}

        extractor = PatientExtractor()
        summary, warnings = extractor.extract(resource)

        assert summary == "Unknown patient demographics"
        assert len(warnings) == 1


class TestConditionExtractor:
    """Tests for ConditionExtractor class."""

    def test_extract_condition_with_status(self):
        """Test extraction of condition with clinical status."""
        resource = {
            "resourceType": "Condition",
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "38341003",
                        "display": "Essential hypertension",
                    }
                ]
            },
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "onsetDateTime": "2020-03-15",
        }

        extractor = ConditionExtractor()
        entry, code = extractor.extract(resource)

        assert entry is not None
        assert "Essential hypertension" in entry.content
        assert "(active)" in entry.content
        assert entry.date_label == "2020-03-15"
        assert code == "38341003"

    def test_extract_condition_without_date(self):
        """Test extraction of condition without onset date."""
        resource = {
            "resourceType": "Condition",
            "code": {"text": "Hypertension"},
            "clinicalStatus": {"coding": [{"code": "resolved"}]},
        }

        extractor = ConditionExtractor()
        entry, _ = extractor.extract(resource)

        assert entry is not None
        assert entry.date_label == "[Historical/Undated]"
        assert entry.date is None

    def test_extract_condition_from_text_field(self):
        """Test extraction when display is in text field."""
        resource = {
            "resourceType": "Condition",
            "code": {"text": "Chronic pain syndrome"},
        }

        extractor = ConditionExtractor()
        entry, _ = extractor.extract(resource)

        assert entry is not None
        assert "Chronic pain syndrome" in entry.content


class TestMedicationExtractor:
    """Tests for MedicationExtractor class."""

    def test_extract_medication_with_dosage(self):
        """Test extraction of medication with dosage instructions."""
        resource = {
            "resourceType": "MedicationRequest",
            "status": "active",
            "medicationCodeableConcept": {
                "coding": [{"display": "Lisinopril 10 MG Oral Tablet"}]
            },
            "authoredOn": "2020-03-15",
            "dosageInstruction": [
                {
                    "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}},
                    "doseAndRate": [{"doseQuantity": {"value": 1}}],
                }
            ],
        }

        extractor = MedicationExtractor()
        entry, active_med = extractor.extract(resource)

        assert entry is not None
        assert "Lisinopril" in entry.content
        assert "(active)" in entry.content
        assert active_med is not None
        assert "Lisinopril" in active_med

    def test_extract_medication_without_dosage(self):
        """Test extraction when dosage instruction is missing."""
        resource = {
            "resourceType": "MedicationRequest",
            "status": "stopped",
            "medicationCodeableConcept": {"text": "Warfarin"},
        }

        extractor = MedicationExtractor()
        entry, active_med = extractor.extract(resource)

        assert entry is not None
        assert "Warfarin" in entry.content
        assert active_med is None  # Not active

    def test_extract_active_medication_string(self):
        """Test active medication string generation."""
        resource = {
            "resourceType": "MedicationRequest",
            "status": "active",
            "medicationCodeableConcept": {"text": "Aspirin 81mg"},
            "authoredOn": "2024-01-01",
        }

        extractor = MedicationExtractor()
        _, active_med = extractor.extract(resource)

        assert active_med is not None
        assert "Aspirin 81mg" in active_med
        assert "(since 2024-01-01)" in active_med


class TestObservationExtractor:
    """Tests for ObservationExtractor class."""

    def test_extract_lab_with_interpretation(self):
        """Test extraction of lab with interpretation flag."""
        resource = {
            "resourceType": "Observation",
            "code": {"text": "D-dimer"},
            "valueQuantity": {"value": 1250, "unit": "ng/mL"},
            "interpretation": [{"coding": [{"code": "HH"}]}],
            "effectiveDateTime": "2024-06-01",
        }

        extractor = ObservationExtractor()
        entry = extractor.extract(resource)

        assert entry is not None
        assert "D-dimer" in entry.content
        assert "1250" in entry.content
        assert "ng/mL" in entry.content
        assert "(Critical High)" in entry.content

    def test_extract_lab_normal(self):
        """Test extraction of normal lab value."""
        resource = {
            "resourceType": "Observation",
            "code": {"coding": [{"display": "Creatinine"}]},
            "valueQuantity": {"value": 1.2, "unit": "mg/dL"},
            "interpretation": [{"coding": [{"code": "N"}]}],
            "effectiveDateTime": "2024-06-01",
        }

        extractor = ObservationExtractor()
        entry = extractor.extract(resource)

        assert entry is not None
        assert "Creatinine" in entry.content
        assert "1.2" in entry.content
        assert "(Normal)" in entry.content

    def test_extract_lab_without_value(self):
        """Test that labs without values are skipped."""
        resource = {
            "resourceType": "Observation",
            "code": {"text": "Some test"},
        }

        extractor = ObservationExtractor()
        entry = extractor.extract(resource)

        assert entry is None

    def test_extract_lab_string_value(self):
        """Test extraction of lab with string value."""
        resource = {
            "resourceType": "Observation",
            "code": {"text": "Blood Type"},
            "valueString": "A Positive",
            "effectiveDateTime": "2024-01-01",
        }

        extractor = ObservationExtractor()
        entry = extractor.extract(resource)

        assert entry is not None
        assert "A Positive" in entry.content


class TestProcedureExtractor:
    """Tests for ProcedureExtractor class."""

    def test_extract_procedure(self):
        """Test extraction of procedure."""
        resource = {
            "resourceType": "Procedure",
            "code": {"coding": [{"display": "Coronary angiography"}]},
            "performedPeriod": {"start": "2024-05-15T10:00:00Z"},
        }

        extractor = ProcedureExtractor()
        entry = extractor.extract(resource)

        assert entry is not None
        assert "Coronary angiography" in entry.content
        assert entry.date_label == "2024-05-15"

    def test_extract_procedure_without_date(self):
        """Test extraction of procedure without performed date."""
        resource = {
            "resourceType": "Procedure",
            "code": {"text": "Appendectomy"},
        }

        extractor = ProcedureExtractor()
        entry = extractor.extract(resource)

        assert entry is not None
        assert entry.date_label == "[Historical/Undated]"


class TestEncounterExtractor:
    """Tests for EncounterExtractor class."""

    def test_extract_encounter_with_reason(self):
        """Test extraction of encounter with reason code."""
        resource = {
            "resourceType": "Encounter",
            "type": [{"coding": [{"display": "Emergency department visit"}]}],
            "period": {"start": "2024-06-01T14:30:00Z"},
            "reasonCode": [{"coding": [{"display": "Chest pain"}]}],
        }

        extractor = EncounterExtractor()
        entry = extractor.extract(resource)

        assert entry is not None
        assert "Emergency department visit" in entry.content
        assert "(reason: Chest pain)" in entry.content
        assert entry.date_label == "2024-06-01"

    def test_extract_encounter_fallback_to_class(self):
        """Test extraction using class when type is missing."""
        resource = {
            "resourceType": "Encounter",
            "class": {"display": "Outpatient"},
            "period": {"start": "2024-01-15"},
        }

        extractor = EncounterExtractor()
        entry = extractor.extract(resource)

        assert entry is not None
        assert "Outpatient" in entry.content


class TestTimelineSerializer:
    """Tests for TimelineSerializer class."""

    def test_serialize_chronological_order(self):
        """Test that entries are serialized in chronological order."""
        entries = [
            TimelineEntry(
                date=datetime(2024, 6, 1),
                date_label="2024-06-01",
                category="Lab",
                content="Lab: D-dimer 1250 ng/mL (Critical High)",
                priority=CATEGORY_PRIORITY["Lab"],
            ),
            TimelineEntry(
                date=datetime(2020, 3, 15),
                date_label="2020-03-15",
                category="Condition",
                content="Diagnosis: Hypertension (active)",
                priority=CATEGORY_PRIORITY["Condition"],
            ),
        ]

        serializer = TimelineSerializer()
        output = serializer.serialize("71-year-old male", entries, [])

        # Check order - 2020 should come before 2024
        idx_2020 = output.index("2020-03-15")
        idx_2024 = output.index("2024-06-01")
        assert idx_2020 < idx_2024

    def test_serialize_undated_first(self):
        """Test that undated entries appear first."""
        entries = [
            TimelineEntry(
                date=datetime(2024, 6, 1),
                date_label="2024-06-01",
                category="Lab",
                content="Lab: Something",
                priority=CATEGORY_PRIORITY["Lab"],
            ),
            TimelineEntry(
                date=None,
                date_label="[Historical/Undated]",
                category="Condition",
                content="Historical Diagnosis: Diabetes",
                priority=CATEGORY_PRIORITY["Condition"],
            ),
        ]

        serializer = TimelineSerializer()
        output = serializer.serialize("Patient", entries, [])

        idx_undated = output.index("[Historical/Undated]")
        idx_2024 = output.index("2024-06-01")
        assert idx_undated < idx_2024

    def test_serialize_with_active_medications(self):
        """Test that active medications section is included."""
        entries = []
        active_meds = ["Lisinopril 10mg once daily (since 2020-03-15)"]

        serializer = TimelineSerializer()
        output = serializer.serialize("Patient", entries, active_meds)

        assert "== ACTIVE MEDICATIONS ==" in output
        assert "Lisinopril 10mg" in output

    def test_serialize_same_date_priority(self):
        """Test that same-date entries are sorted by priority."""
        entries = [
            TimelineEntry(
                date=datetime(2024, 6, 1),
                date_label="2024-06-01",
                category="Lab",
                content="Lab: Something",
                priority=CATEGORY_PRIORITY["Lab"],  # 5
            ),
            TimelineEntry(
                date=datetime(2024, 6, 1),
                date_label="2024-06-01",
                category="Encounter",
                content="Encounter: ER visit",
                priority=CATEGORY_PRIORITY["Encounter"],  # 1
            ),
        ]

        serializer = TimelineSerializer()
        output = serializer.serialize("Patient", entries, [])

        # Encounter should come before Lab on same date
        idx_encounter = output.index("Encounter: ER visit")
        idx_lab = output.index("Lab: Something")
        assert idx_encounter < idx_lab


class TestTimelineEntry:
    """Tests for TimelineEntry sorting behavior."""

    def test_undated_entry_sorts_first(self):
        """Test that undated entries sort before dated entries."""
        undated = TimelineEntry(None, "[Historical/Undated]", "Condition", "test", 1)
        dated = TimelineEntry(datetime(2024, 1, 1), "2024-01-01", "Condition", "test", 1)

        assert undated < dated
        assert not dated < undated

    def test_earlier_date_sorts_first(self):
        """Test that earlier dates sort before later dates."""
        earlier = TimelineEntry(datetime(2020, 1, 1), "2020-01-01", "Condition", "test", 1)
        later = TimelineEntry(datetime(2024, 1, 1), "2024-01-01", "Condition", "test", 1)

        assert earlier < later

    def test_same_date_sorts_by_priority(self):
        """Test that same dates sort by priority."""
        high_priority = TimelineEntry(datetime(2024, 1, 1), "2024-01-01", "Encounter", "test", 1)
        low_priority = TimelineEntry(datetime(2024, 1, 1), "2024-01-01", "Lab", "test", 5)

        assert high_priority < low_priority


class TestFHIRJanitor:
    """Integration tests for FHIRJanitor class."""

    def test_process_empty_bundle(self):
        """Test handling of empty bundle."""
        janitor = FHIRJanitor()
        result = janitor.process_bundle({})

        assert result.patient_summary == "Unknown patient demographics"
        assert "Invalid or missing FHIR Bundle" in result.extraction_warnings

    def test_process_invalid_bundle_type(self):
        """Test handling of non-Bundle resource."""
        janitor = FHIRJanitor()
        result = janitor.process_bundle({"resourceType": "Patient"})

        assert "Invalid or missing FHIR Bundle" in result.extraction_warnings

    def test_process_minimal_bundle(self):
        """Test processing a minimal valid bundle."""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "123",
                        "birthDate": "1954-04-16",
                        "gender": "male",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {"text": "Hypertension"},
                        "clinicalStatus": {"coding": [{"code": "active"}]},
                        "onsetDateTime": "2020-01-01",
                    }
                },
            ],
        }

        janitor = FHIRJanitor()
        result = janitor.process_bundle(bundle)

        assert "male" in result.patient_summary
        assert "Hypertension" in result.narrative
        assert result.token_estimate > 0

    def test_process_bundle_with_noise_resources(self):
        """Test that noise resources are filtered out."""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "123", "gender": "female"}},
                {"resource": {"resourceType": "Organization", "id": "org1"}},
                {"resource": {"resourceType": "Provenance", "id": "prov1"}},
            ],
        }

        janitor = FHIRJanitor()
        result = janitor.process_bundle(bundle)

        # Should not contain organization or provenance info
        assert "Organization" not in result.narrative
        assert "Provenance" not in result.narrative

    def test_process_bundle_extracts_hidden_diagnoses(self):
        """Test extraction of diagnoses from billing records."""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "123"}},
                {
                    "resource": {
                        "resourceType": "Claim",
                        "id": "claim1",
                        "diagnosis": [
                            {
                                "diagnosisCodeableConcept": {
                                    "coding": [
                                        {
                                            "system": "http://snomed.info/sct",
                                            "code": "73211009",
                                            "display": "Diabetes mellitus",
                                        }
                                    ]
                                }
                            }
                        ],
                    }
                },
            ],
        }

        janitor = FHIRJanitor()
        result = janitor.process_bundle(bundle)

        assert "Diabetes mellitus" in result.narrative
        assert "from billing records" in result.narrative

    def test_output_format_sections(self):
        """Test that output contains required sections."""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "birthDate": "1950-01-01",
                        "gender": "male",
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "status": "active",
                        "medicationCodeableConcept": {"text": "Aspirin"},
                        "authoredOn": "2024-01-01",
                    }
                },
            ],
        }

        janitor = FHIRJanitor()
        result = janitor.process_bundle(bundle)

        assert "== PATIENT ==" in result.narrative
        assert "== CLINICAL TIMELINE ==" in result.narrative
        assert "== ACTIVE MEDICATIONS ==" in result.narrative

    def test_clinical_stream_dataclass_fields(self):
        """Test that ClinicalStream has all expected fields."""
        bundle = {
            "resourceType": "Bundle",
            "entry": [{"resource": {"resourceType": "Patient", "id": "123"}}],
        }

        janitor = FHIRJanitor()
        result = janitor.process_bundle(bundle)

        # Verify all fields exist
        assert hasattr(result, "patient_summary")
        assert hasattr(result, "narrative")
        assert hasattr(result, "token_estimate")
        assert hasattr(result, "extraction_warnings")
        assert hasattr(result, "active_medications")
        assert isinstance(result.extraction_warnings, list)
        assert isinstance(result.active_medications, list)


class TestInterpretationMapping:
    """Tests for lab interpretation code mapping."""

    def test_all_interpretation_codes(self):
        """Test all interpretation code mappings."""
        codes = {
            "H": "High",
            "HH": "Critical High",
            "L": "Low",
            "LL": "Critical Low",
            "N": "Normal",
            "A": "Abnormal",
        }

        extractor = ObservationExtractor()

        for code, expected in codes.items():
            resource = {
                "resourceType": "Observation",
                "code": {"text": "Test"},
                "valueQuantity": {"value": 100},
                "interpretation": [{"coding": [{"code": code}]}],
            }
            entry = extractor.extract(resource)
            assert expected in entry.content, f"Failed for code {code}"
