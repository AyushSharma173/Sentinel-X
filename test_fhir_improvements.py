#!/usr/bin/env python3
"""Test script for FHIR generation improvements.

Tests the new functions added in the implementation without requiring
OpenAI API calls or Synthea execution.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sentinel_x.scripts.synthetic_fhir_pipeline import (
    # Phase 5: SNOMED
    SNOMED_MAPPING,
    lookup_snomed_code,
    enrich_snomed_codes,
    # Phase 2: Smoking/CV Risk
    create_smoking_observation,
    create_cardiovascular_risk_assessment,
    # Phase 4: Resource Linkages
    create_finding_observation,
    create_diagnostic_report,
    create_condition_resource,
    # Phase 3: Temporal
    CONDITION_TEMPORAL_CLASS,
    classify_condition_temporality,
    calculate_onset_date,
    # Phase 1: Synthea Modules
    VALID_SYNTHEA_MODULES,
    validate_synthea_modules,
    # Phase 6: Validation
    validate_temporal_consistency,
    validate_reference_integrity,
    validate_bundle,
    # Supporting types
    ExtractedCondition,
    RadiologyExtraction,
    DemographicInference,
)


def test_snomed_lookup():
    """Test Phase 5: SNOMED code lookup."""
    print("\n=== Phase 5: SNOMED Code Lookup ===")

    test_cases = [
        ("emphysema", "87433001"),
        ("Calcific atheromatous plaques", "128305009"),  # Partial match
        ("BRONCHIECTASIS", "12295008"),  # Case insensitive
        ("Venous collaterals", "234042006"),  # Gap case
        ("Degenerative changes", "396275006"),  # Gap case
        ("unknown xyz condition", None),  # No match
    ]

    passed = 0
    for condition, expected_code in test_cases:
        result = lookup_snomed_code(condition)
        actual_code = result[0] if result else None
        status = "PASS" if actual_code == expected_code else "FAIL"
        print(f"  {status}: '{condition}' -> {actual_code} (expected {expected_code})")
        if status == "PASS":
            passed += 1

    print(f"  Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_snomed_enrichment():
    """Test Phase 5: SNOMED enrichment of extraction."""
    print("\n=== Phase 5: SNOMED Enrichment ===")

    # Create extraction with missing SNOMED codes
    extraction = RadiologyExtraction(
        conditions=[
            ExtractedCondition(condition_name="Venous collaterals", snomed_code=None),
            ExtractedCondition(condition_name="Emphysema", snomed_code="87433001"),  # Already has code
            ExtractedCondition(condition_name="Degenerative changes", snomed_code=None),
        ],
        demographics=DemographicInference(
            estimated_age_min=55, estimated_age_max=85,
            reasoning="Test"
        )
    )

    # Enrich
    enriched = enrich_snomed_codes(extraction)

    # Check results
    results = []
    for cond in enriched.conditions:
        has_code = cond.snomed_code is not None
        results.append((cond.condition_name, has_code, cond.snomed_code))
        status = "PASS" if has_code else "FAIL"
        print(f"  {status}: '{cond.condition_name}' -> {cond.snomed_code}")

    all_have_codes = all(r[1] for r in results)
    print(f"  All conditions enriched: {all_have_codes}")
    return all_have_codes


def test_smoking_observation():
    """Test Phase 2: Smoking status observation creation."""
    print("\n=== Phase 2: Smoking Status Observation ===")

    # Test smoker
    smoker_obs = create_smoking_observation("Patient/123", True, "2026-02-03T10:00:00Z")
    smoker_code = smoker_obs["valueCodeableConcept"]["coding"][0]["code"]

    # Test non-smoker
    non_smoker_obs = create_smoking_observation("Patient/123", False, "2026-02-03T10:00:00Z")
    non_smoker_code = non_smoker_obs["valueCodeableConcept"]["coding"][0]["code"]

    print(f"  Smoker observation code: {smoker_code} (expected 449868002)")
    print(f"  Non-smoker observation code: {non_smoker_code} (expected 266919005)")
    print(f"  Has US Core profile: {'us-core-smokingstatus' in str(smoker_obs['meta'])}")
    print(f"  Has LOINC code 72166-2: {smoker_obs['code']['coding'][0]['code'] == '72166-2'}")

    passed = (
        smoker_code == "449868002" and
        non_smoker_code == "266919005" and
        "us-core-smokingstatus" in str(smoker_obs["meta"])
    )
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_cardiovascular_risk():
    """Test Phase 2: Cardiovascular risk assessment creation."""
    print("\n=== Phase 2: Cardiovascular Risk Assessment ===")

    risk_assessment = create_cardiovascular_risk_assessment(
        "Patient/123",
        "high",
        "2026-02-03T10:00:00Z",
        basis_condition_refs=["Condition/456", "Condition/789"]
    )

    print(f"  Resource type: {risk_assessment['resourceType']}")
    print(f"  Risk level: {risk_assessment['prediction'][0]['qualitativeRisk']['coding'][0]['code']}")
    print(f"  Probability: {risk_assessment['prediction'][0]['probabilityDecimal']}")
    print(f"  Has basis refs: {len(risk_assessment.get('basis', []))} conditions")

    passed = (
        risk_assessment["resourceType"] == "RiskAssessment" and
        risk_assessment["prediction"][0]["qualitativeRisk"]["coding"][0]["code"] == "high" and
        len(risk_assessment.get("basis", [])) == 2
    )
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_temporal_classification():
    """Test Phase 3: Temporal classification of conditions."""
    print("\n=== Phase 3: Temporal Classification ===")

    test_cases = [
        ("Spondylosis", "degenerative"),
        ("Emphysema", "chronic"),
        ("Pneumonia", "acute"),
        ("Pulmonary nodule", "subacute"),
        ("Simple cyst", "incidental"),
        ("Unknown condition XYZ", "chronic"),  # Default
    ]

    passed = 0
    for condition, expected_class in test_cases:
        actual_class = classify_condition_temporality(condition)
        status = "PASS" if actual_class == expected_class else "FAIL"
        print(f"  {status}: '{condition}' -> {actual_class} (expected {expected_class})")
        if status == "PASS":
            passed += 1

    print(f"  Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_onset_calculation():
    """Test Phase 3: Onset date calculation."""
    print("\n=== Phase 3: Onset Date Calculation ===")

    scan_dt = datetime(2026, 2, 3, 10, 0, 0)

    # Test degenerative (5-20 years before)
    onset_deg, note_deg = calculate_onset_date(scan_dt, "degenerative", seed=42)
    years_deg = (scan_dt - onset_deg).days / 365

    # Test acute (0-14 days before)
    onset_acute, note_acute = calculate_onset_date(scan_dt, "acute", seed=42)
    days_acute = (scan_dt - onset_acute).days

    # Test incidental (same as scan)
    onset_inc, note_inc = calculate_onset_date(scan_dt, "incidental", seed=42)

    print(f"  Degenerative: {years_deg:.1f} years before scan (expected 5-20)")
    print(f"  Acute: {days_acute} days before scan (expected 0-14)")
    print(f"  Incidental: same as scan = {onset_inc == scan_dt}")

    # Test determinism (same seed = same result)
    onset_deg2, _ = calculate_onset_date(scan_dt, "degenerative", seed=42)
    deterministic = onset_deg == onset_deg2
    print(f"  Deterministic (same seed): {deterministic}")

    passed = (
        5 <= years_deg <= 20 and
        0 <= days_acute <= 14 and
        onset_inc == scan_dt and
        deterministic
    )
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_synthea_module_validation():
    """Test Phase 1: Synthea module validation."""
    print("\n=== Phase 1: Synthea Module Validation ===")

    test_modules = [
        "copd",  # Valid
        "congestive_heart_failure",  # Valid
        "invalid_module",  # Invalid
        "cardiovascular_disease",  # Invalid but should map
        "COPD",  # Case variation
    ]

    validated = validate_synthea_modules(test_modules)

    print(f"  Input modules: {test_modules}")
    print(f"  Validated modules: {validated}")
    print(f"  Contains 'copd': {'copd' in validated}")
    print(f"  Contains 'congestive_heart_failure': {'congestive_heart_failure' in validated}")
    print(f"  'invalid_module' filtered out: {'invalid_module' not in validated}")

    passed = (
        "copd" in validated and
        "congestive_heart_failure" in validated and
        "invalid_module" not in validated
    )
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_resource_linkages():
    """Test Phase 4: Resource linkages."""
    print("\n=== Phase 4: Resource Linkages ===")

    condition = ExtractedCondition(
        condition_name="Emphysema",
        snomed_code="87433001",
        severity="moderate",
        body_site="lung"
    )

    # Create finding observation
    obs = create_finding_observation(
        "Patient/123",
        condition,
        "2026-02-03T10:00:00Z",
        diagnostic_report_ref="DiagnosticReport/456"
    )

    # Create condition with evidence
    cond = create_condition_resource(
        "Patient/123",
        condition,
        "2020-01-01T00:00:00Z",
        evidence_ref="Observation/789",
        recorded_datetime="2026-02-03T10:00:00Z",
        onset_note="Chronic condition, estimated onset 6 years prior"
    )

    print(f"  Observation has derivedFrom: {'derivedFrom' in obs}")
    print(f"  Observation derivedFrom ref: {obs.get('derivedFrom', [{}])[0].get('reference')}")
    print(f"  Condition has evidence: {'evidence' in cond}")
    print(f"  Condition has recordedDate: {'recordedDate' in cond}")
    print(f"  Condition has note: {'note' in cond}")

    passed = (
        "derivedFrom" in obs and
        obs["derivedFrom"][0]["reference"] == "DiagnosticReport/456" and
        "evidence" in cond and
        "recordedDate" in cond and
        "note" in cond
    )
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_validation_functions():
    """Test Phase 6: Validation functions."""
    print("\n=== Phase 6: Validation Functions ===")

    # Create a test bundle with some issues
    test_bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "patient-1"
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-1",
                    "onsetDateTime": "2026-02-03T10:00:00Z",
                    "recordedDate": "2026-02-03T10:00:00Z",
                    "subject": {"reference": "Patient/patient-1"}
                }
            },
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "report-1",
                    "effectiveDateTime": "2026-02-03T10:00:00Z",
                    "issued": "2026-02-03T10:00:00Z",
                    "subject": {"reference": "Patient/patient-1"},
                    "result": [{"reference": "Observation/obs-1"}]  # Reference to non-existent resource
                }
            },
            {
                "resource": {
                    "resourceType": "ImagingStudy",
                    "id": "study-1",
                    "subject": {"reference": "Patient/patient-1"}
                }
            }
        ]
    }

    # Test temporal validation (should pass)
    temporal_errors = validate_temporal_consistency(test_bundle)
    print(f"  Temporal errors: {len(temporal_errors)}")

    # Test reference validation (should find 1 error for Observation/obs-1)
    ref_errors = validate_reference_integrity(test_bundle)
    print(f"  Reference errors: {len(ref_errors)}")
    for err in ref_errors[:2]:
        print(f"    - {err}")

    # Test full validation
    is_valid, metrics = validate_bundle(test_bundle)
    print(f"  Bundle valid: {is_valid}")
    print(f"  Entry count: {metrics['entry_count']}")
    print(f"  Resource types: {list(metrics['resource_types'].keys())}")

    passed = (
        len(temporal_errors) == 0 and
        len(ref_errors) >= 1 and  # Should find the bad Observation reference
        not is_valid  # Should fail due to reference error
    )
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    """Run all tests."""
    print("=" * 60)
    print("FHIR Generation Improvements - Test Suite")
    print("=" * 60)

    tests = [
        ("SNOMED Lookup", test_snomed_lookup),
        ("SNOMED Enrichment", test_snomed_enrichment),
        ("Smoking Observation", test_smoking_observation),
        ("Cardiovascular Risk", test_cardiovascular_risk),
        ("Temporal Classification", test_temporal_classification),
        ("Onset Calculation", test_onset_calculation),
        ("Synthea Module Validation", test_synthea_module_validation),
        ("Resource Linkages", test_resource_linkages),
        ("Validation Functions", test_validation_functions),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, passed, _ in results if passed)
    for name, passed, error in results:
        status = "PASS" if passed else "FAIL"
        error_msg = f" ({error})" if error else ""
        print(f"  [{status}] {name}{error_msg}")

    print(f"\nTotal: {passed_count}/{len(results)} tests passed")

    return passed_count == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
