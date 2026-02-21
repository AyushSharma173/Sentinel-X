"""Unit tests for ground truth control architecture in synthetic_fhir_pipeline."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add the scripts directory to the path so we can import the pipeline module
sys.path.insert(0, str(Path(__file__).parent.parent / "sentinel_x" / "scripts"))

from synthetic_fhir_pipeline import (
    ALWAYS_ACUTE_CONDITIONS,
    ALWAYS_CHRONIC_CONDITIONS,
    CUSTOM_MODULES_DIR,
    FINDING_TIER_MAP,
    HARDCODED_JAVA_MODULES,
    MODULE_OVERRIDE_PATHS,
    FindingGroundTruth,
    PatientGroundTruth,
    RadiologyExtraction,
    DemographicInference,
    ExtractedCondition,
    classify_finding_tier,
    generate_keep_module,
    generate_module_overrides,
    generate_patient_ground_truth,
    validate_ground_truth_concordance,
    validate_synthea_modules,
    _synthea_bundle_has_condition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extraction(conditions: list[tuple[str, str | None]]) -> RadiologyExtraction:
    """Create a RadiologyExtraction with the given conditions."""
    return RadiologyExtraction(
        conditions=[
            ExtractedCondition(
                condition_name=name,
                snomed_code=code,
                severity="mild",
            )
            for name, code in conditions
        ],
        demographics=DemographicInference(
            estimated_age_min=50,
            estimated_age_max=70,
            gender_hint=None,
            reasoning="test",
        ),
        smoking_history_likely=False,
        cardiovascular_risk="low",
        synthea_modules=[],
    )


def _make_bundle_with_conditions(condition_texts: list[str]) -> dict:
    """Create a minimal FHIR bundle with Synthea-style Condition resources."""
    entries = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "test-patient",
            }
        }
    ]
    for text in condition_texts:
        entries.append({
            "resource": {
                "resourceType": "Condition",
                "id": f"cond-{text.replace(' ', '-').lower()}",
                "code": {
                    "coding": [{"display": text}],
                    "text": text,
                },
                # No encounter-diagnosis category → treated as Synthea-generated
            }
        })
    return {"resourceType": "Bundle", "entry": entries}


# ---------------------------------------------------------------------------
# Phase 0 Tests: Bug Fixes
# ---------------------------------------------------------------------------

class TestValidateSyntheaModules:
    """Test validate_synthea_modules with hard-coded module rejection."""

    def test_cardiovascular_rejected(self):
        """Hard-coded Java module 'cardiovascular_disease' should be rejected."""
        result = validate_synthea_modules(["cardiovascular_disease"])
        assert "cardiovascular_disease" not in result

    def test_hardcoded_modules_all_rejected(self):
        """All HARDCODED_JAVA_MODULES should be rejected."""
        result = validate_synthea_modules(list(HARDCODED_JAVA_MODULES))
        assert result == []

    def test_valid_module_passes(self):
        """Valid JSON modules should pass through."""
        result = validate_synthea_modules(["copd", "hypertension", "lung_cancer"])
        assert result == ["copd", "hypertension", "lung_cancer"]

    def test_duplicates_removed(self):
        """Duplicate modules should be deduplicated."""
        result = validate_synthea_modules(["copd", "copd", "hypertension"])
        assert result == ["copd", "hypertension"]

    def test_normalization(self):
        """Module names should be normalized."""
        result = validate_synthea_modules(["COPD", "Lung-Cancer"])
        assert result == ["copd", "lung_cancer"]


# ---------------------------------------------------------------------------
# Phase 1 Tests: Ground Truth Architecture
# ---------------------------------------------------------------------------

class TestClassifyFindingTier:
    """Test classify_finding_tier for known conditions."""

    def test_tier1_emphysema(self):
        tier, module = classify_finding_tier("emphysema")
        assert tier == 1
        assert module == "copd"

    def test_tier1_cardiomegaly(self):
        tier, module = classify_finding_tier("cardiomegaly")
        assert tier == 1
        assert module == "congestive_heart_failure"

    def test_tier1_case_insensitive(self):
        tier, module = classify_finding_tier("Atherosclerosis")
        assert tier == 1
        assert module == "stable_ischemic_heart_disease"

    def test_tier2_pulmonary_nodule(self):
        tier, module = classify_finding_tier("pulmonary nodule")
        assert tier == 2
        assert module == "sentinel_pulmonary_nodule"

    def test_tier2_pleural_effusion(self):
        tier, module = classify_finding_tier("pleural effusion")
        assert tier == 2
        assert module == "sentinel_pleural_effusion"

    def test_tier2_pulmonary_embolism(self):
        tier, module = classify_finding_tier("pulmonary embolism")
        assert tier == 2
        assert module == "sentinel_pulmonary_embolism"

    def test_tier2_pneumothorax(self):
        tier, module = classify_finding_tier("pneumothorax")
        assert tier == 2
        assert module == "sentinel_pneumothorax"

    def test_tier2_aortic_aneurysm(self):
        tier, module = classify_finding_tier("aortic aneurysm")
        assert tier == 2
        assert module == "sentinel_aortic_aneurysm"

    def test_tier2_pericardial_effusion(self):
        tier, module = classify_finding_tier("pericardial effusion")
        assert tier == 2
        assert module == "sentinel_pericardial_effusion"

    def test_tier3_thyroid_nodule(self):
        tier, module = classify_finding_tier("thyroid nodule")
        assert tier == 3
        assert module is None

    def test_tier3_granuloma(self):
        tier, module = classify_finding_tier("granuloma")
        assert tier == 3
        assert module is None

    def test_unknown_defaults_to_tier3(self):
        tier, module = classify_finding_tier("unusual rare condition xyz")
        assert tier == 3
        assert module is None

    def test_partial_match(self):
        """Should match 'emphysema' within 'centrilobular emphysema'."""
        tier, module = classify_finding_tier("centrilobular emphysema")
        assert tier == 1
        assert module == "copd"


class TestGeneratePatientGroundTruth:
    """Test ground truth generation with deterministic seeding."""

    def test_deterministic_same_seed(self):
        """Same seed should produce identical classifications."""
        extraction = _make_extraction([
            ("emphysema", "87433001"),
            ("pleural effusion", "60046008"),
            ("thyroid nodule", "237495005"),
        ])
        gt1 = generate_patient_ground_truth("p1", extraction, seed=42)
        gt2 = generate_patient_ground_truth("p1", extraction, seed=42)

        for f1, f2 in zip(gt1.findings, gt2.findings):
            assert f1.classification == f2.classification
            assert f1.tier == f2.tier
            assert f1.override_probability == f2.override_probability

    def test_different_seed_may_differ(self):
        """Different seeds may produce different classifications (probabilistic)."""
        extraction = _make_extraction([
            ("emphysema", "87433001"),
            ("pleural effusion", "60046008"),
        ])
        # Run with many different seeds to find at least one difference
        classifications = set()
        for seed in range(100):
            gt = generate_patient_ground_truth("p1", extraction, seed=seed)
            key = tuple(f.classification for f in gt.findings)
            classifications.add(key)
        # With 100 seeds, we should get at least 2 distinct classification patterns
        assert len(classifications) >= 2

    def test_acute_conditions_always_acute(self):
        """Conditions in ALWAYS_ACUTE_CONDITIONS should always be ACUTE_NEW."""
        conditions = [("pulmonary embolism", "59282003"), ("pneumothorax", "36118008")]
        extraction = _make_extraction(conditions)

        for seed in range(20):
            gt = generate_patient_ground_truth("p1", extraction, seed=seed)
            for finding in gt.findings:
                assert finding.classification == "ACUTE_NEW", (
                    f"Expected ACUTE_NEW for '{finding.condition_name}' "
                    f"at seed={seed}, got {finding.classification}"
                )

    def test_degenerative_conditions_always_chronic(self):
        """Conditions in ALWAYS_CHRONIC_CONDITIONS should always be CHRONIC_STABLE."""
        conditions = [("spondylosis", "75320002"), ("osteoarthritis", "396275006")]
        extraction = _make_extraction(conditions)

        for seed in range(20):
            gt = generate_patient_ground_truth("p1", extraction, seed=seed)
            for finding in gt.findings:
                assert finding.classification == "CHRONIC_STABLE", (
                    f"Expected CHRONIC_STABLE for '{finding.condition_name}' "
                    f"at seed={seed}, got {finding.classification}"
                )

    def test_override_probability_mapping(self):
        """CHRONIC_STABLE → 1.0, ACUTE_NEW → 0.0."""
        extraction = _make_extraction([
            ("spondylosis", "75320002"),
            ("pneumothorax", "36118008"),
        ])
        gt = generate_patient_ground_truth("p1", extraction, seed=42)

        for finding in gt.findings:
            if finding.classification == "CHRONIC_STABLE":
                assert finding.override_probability == 1.0
            else:
                assert finding.override_probability == 0.0

    def test_chronic_probability_1_forces_chronic(self):
        """chronic_probability=1.0 should make all non-forced findings CHRONIC_STABLE."""
        extraction = _make_extraction([
            ("emphysema", "87433001"),
            ("pleural effusion", "60046008"),
        ])
        gt = generate_patient_ground_truth(
            "p1", extraction, seed=42, chronic_probability=1.0
        )
        for finding in gt.findings:
            assert finding.classification == "CHRONIC_STABLE"

    def test_chronic_probability_0_forces_acute(self):
        """chronic_probability=0.0 should make non-forced findings ACUTE_NEW."""
        # Use conditions that aren't in ALWAYS_CHRONIC or ALWAYS_ACUTE
        extraction = _make_extraction([
            ("emphysema", "87433001"),  # chronic temporal class
            ("pleural effusion", "60046008"),  # subacute temporal class
        ])
        gt = generate_patient_ground_truth(
            "p1", extraction, seed=42, chronic_probability=0.0
        )
        for finding in gt.findings:
            # Emphysema has 'degenerative' → might still be chronic
            # Pleural effusion is subacute → should be ACUTE_NEW with prob=0
            if finding.condition_name.lower() == "pleural effusion":
                assert finding.classification == "ACUTE_NEW"

    def test_to_dict_structure(self):
        """to_dict() should produce the expected JSON structure."""
        extraction = _make_extraction([("emphysema", "87433001")])
        gt = generate_patient_ground_truth("p1", extraction, seed=42)
        d = gt.to_dict()

        assert d["patient_id"] == "p1"
        assert d["seed"] == 42
        assert isinstance(d["findings"], list)
        assert len(d["findings"]) == 1
        finding = d["findings"][0]
        assert "condition_name" in finding
        assert "classification" in finding
        assert finding["classification"] in ("CHRONIC_STABLE", "ACUTE_NEW")
        assert "tier" in finding
        assert "module" in finding
        assert "override_probability" in finding
        assert "temporal_class" in finding


# ---------------------------------------------------------------------------
# Phase 1 Tests: Module Overrides
# ---------------------------------------------------------------------------

class TestGenerateModuleOverrides:
    """Test the enhanced module override generation with per-module probabilities."""

    def test_force_at_one(self, tmp_path):
        """CHRONIC_STABLE → probability 1.0 should appear in properties file."""
        override_file = generate_module_overrides(
            {"copd": 1.0},
            tmp_path,
        )
        assert override_file is not None
        content = override_file.read_text()
        assert "= 1.0" in content
        assert "copd.json" in content

    def test_suppress_at_zero(self, tmp_path):
        """ACUTE_NEW → probability 0.0 should appear in properties file."""
        override_file = generate_module_overrides(
            {"copd": 0.0},
            tmp_path,
        )
        assert override_file is not None
        content = override_file.read_text()
        assert "= 0.0" in content

    def test_mixed_probabilities(self, tmp_path):
        """Mixed force/suppress should produce correct properties file."""
        override_file = generate_module_overrides(
            {"copd": 1.0, "hypertension": 0.0, "osteoarthritis": 0.5},
            tmp_path,
        )
        assert override_file is not None
        content = override_file.read_text()
        assert "copd" in content
        assert "hypertension" in content
        assert "osteoarthritis" in content
        # Verify both 1.0 and 0.0 appear
        assert "= 1.0" in content
        assert "= 0.0" in content
        assert "= 0.5" in content

    def test_no_overrides_returns_none(self, tmp_path):
        """Empty module dict should return None."""
        result = generate_module_overrides({}, tmp_path)
        assert result is None

    def test_unknown_module_returns_none(self, tmp_path):
        """Module without override paths should not generate a file."""
        result = generate_module_overrides(
            {"nonexistent_module": 1.0},
            tmp_path,
        )
        assert result is None

    def test_cardiovascular_not_in_overrides(self):
        """cardiovascular_disease should NOT have override paths."""
        assert "cardiovascular_disease" not in MODULE_OVERRIDE_PATHS


# ---------------------------------------------------------------------------
# Phase 2 Tests: Custom Modules
# ---------------------------------------------------------------------------

class TestCustomModules:
    """Test custom Synthea module JSON structure."""

    CUSTOM_MODULE_DIR = Path(__file__).parent.parent / "sentinel_x" / "data" / "synthea_custom_modules"

    EXPECTED_MODULES = [
        "sentinel_pulmonary_nodule.json",
        "sentinel_pleural_effusion.json",
        "sentinel_pulmonary_embolism.json",
        "sentinel_pneumothorax.json",
        "sentinel_aortic_aneurysm.json",
        "sentinel_pericardial_effusion.json",
    ]

    @pytest.mark.parametrize("module_file", EXPECTED_MODULES)
    def test_module_exists(self, module_file):
        """Each custom module file should exist."""
        assert (self.CUSTOM_MODULE_DIR / module_file).exists()

    @pytest.mark.parametrize("module_file", EXPECTED_MODULES)
    def test_valid_gmf_structure(self, module_file):
        """Each module should have valid GMF v2 structure."""
        with open(self.CUSTOM_MODULE_DIR / module_file) as f:
            module = json.load(f)

        assert module.get("gmf_version") == 2
        assert "states" in module
        states = module["states"]

        # Must have Initial, Age_Guard, Incidence_Check, Onset, Terminal
        assert "Initial" in states
        assert states["Initial"]["type"] == "Initial"

        assert "Age_Guard" in states
        assert states["Age_Guard"]["type"] == "Guard"
        assert "allow" in states["Age_Guard"]

        assert "Incidence_Check" in states
        assert states["Incidence_Check"]["type"] == "Simple"
        assert "distributed_transition" in states["Incidence_Check"]
        dt = states["Incidence_Check"]["distributed_transition"]
        assert len(dt) == 2
        # Probabilities should sum to 1.0
        total = sum(d["distribution"] for d in dt)
        assert abs(total - 1.0) < 0.001

        assert "Terminal" in states
        assert states["Terminal"]["type"] == "Terminal"

    @pytest.mark.parametrize("module_file", EXPECTED_MODULES)
    def test_has_snomed_code(self, module_file):
        """Each module's Onset state should have a SNOMED-CT code."""
        with open(self.CUSTOM_MODULE_DIR / module_file) as f:
            module = json.load(f)

        onset = module["states"].get("Onset", {})
        assert onset.get("type") == "ConditionOnset"
        codes = onset.get("codes", [])
        assert len(codes) >= 1
        assert codes[0]["system"] == "SNOMED-CT"
        assert codes[0]["code"]  # Non-empty

    @pytest.mark.parametrize("module_file", EXPECTED_MODULES)
    def test_override_path_exists(self, module_file):
        """Each custom module should have an override path in MODULE_OVERRIDE_PATHS."""
        module_name = module_file.replace(".json", "")
        assert module_name in MODULE_OVERRIDE_PATHS


# ---------------------------------------------------------------------------
# Phase 3 Tests: Keep Module
# ---------------------------------------------------------------------------

class TestKeepModule:
    """Test keep module generation."""

    def test_generates_keep_module(self, tmp_path):
        """Should generate a keep module for CHRONIC_STABLE findings."""
        gt = PatientGroundTruth(patient_id="p1", seed=42)
        gt.findings.append(FindingGroundTruth(
            condition_name="emphysema",
            snomed_code="87433001",
            classification="CHRONIC_STABLE",
            tier=1,
            module="copd",
            override_probability=1.0,
            temporal_class="chronic",
        ))
        path = generate_keep_module(gt, tmp_path)
        assert path is not None
        assert path.exists()

        with open(path) as f:
            module = json.load(f)

        assert module["gmf_version"] == 2
        assert "Keep" in module["states"]
        assert module["states"]["Keep"]["type"] == "Terminal"

    def test_skip_when_no_chronic(self, tmp_path):
        """Should return None when there are no CHRONIC_STABLE findings."""
        gt = PatientGroundTruth(patient_id="p1", seed=42)
        gt.findings.append(FindingGroundTruth(
            condition_name="pneumothorax",
            snomed_code="36118008",
            classification="ACUTE_NEW",
            tier=2,
            module="sentinel_pneumothorax",
            override_probability=0.0,
            temporal_class="acute",
        ))
        path = generate_keep_module(gt, tmp_path)
        assert path is None

    def test_skip_tier3_chronic(self, tmp_path):
        """Tier 3 CHRONIC_STABLE findings without SNOMED should not affect keep module."""
        gt = PatientGroundTruth(patient_id="p1", seed=42)
        gt.findings.append(FindingGroundTruth(
            condition_name="unusual finding",
            snomed_code=None,
            classification="CHRONIC_STABLE",
            tier=3,
            module=None,
            override_probability=1.0,
            temporal_class="chronic",
        ))
        path = generate_keep_module(gt, tmp_path)
        assert path is None

    def test_or_condition_with_multiple_chronic(self, tmp_path):
        """Multiple CHRONIC_STABLE findings should create an Or condition."""
        gt = PatientGroundTruth(patient_id="p1", seed=42)
        gt.findings.extend([
            FindingGroundTruth(
                condition_name="emphysema",
                snomed_code="87433001",
                classification="CHRONIC_STABLE",
                tier=1, module="copd",
                override_probability=1.0, temporal_class="chronic",
            ),
            FindingGroundTruth(
                condition_name="cardiomegaly",
                snomed_code="8186001",
                classification="CHRONIC_STABLE",
                tier=1, module="congestive_heart_failure",
                override_probability=1.0, temporal_class="chronic",
            ),
        ])
        path = generate_keep_module(gt, tmp_path)
        assert path is not None

        with open(path) as f:
            module = json.load(f)

        guard = module["states"]["Keep_Guard"]
        assert guard["allow"]["condition_type"] == "Or"
        assert len(guard["allow"]["conditions"]) == 2


# ---------------------------------------------------------------------------
# Phase 6B Tests: Concordance Validation
# ---------------------------------------------------------------------------

class TestGroundTruthConcordance:
    """Test ground truth concordance validation."""

    def test_concordant_chronic_found(self):
        """CHRONIC_STABLE condition present in bundle → concordant."""
        bundle = _make_bundle_with_conditions(["Pulmonary emphysema"])
        gt = PatientGroundTruth(patient_id="p1", seed=42)
        gt.findings.append(FindingGroundTruth(
            condition_name="emphysema",
            snomed_code="87433001",
            classification="CHRONIC_STABLE",
            tier=1, module="copd",
            override_probability=1.0, temporal_class="chronic",
        ))
        result = validate_ground_truth_concordance(bundle, gt)
        assert result["concordant"] == 1
        assert result["concordance_rate"] == 100.0

    def test_concordant_acute_not_found(self):
        """ACUTE_NEW condition absent from bundle → concordant."""
        bundle = _make_bundle_with_conditions([])  # Empty Synthea conditions
        gt = PatientGroundTruth(patient_id="p1", seed=42)
        gt.findings.append(FindingGroundTruth(
            condition_name="pneumothorax",
            snomed_code="36118008",
            classification="ACUTE_NEW",
            tier=2, module="sentinel_pneumothorax",
            override_probability=0.0, temporal_class="acute",
        ))
        result = validate_ground_truth_concordance(bundle, gt)
        assert result["concordant"] == 1
        assert len(result["mismatches"]) == 0

    def test_mismatch_chronic_not_found(self):
        """CHRONIC_STABLE condition missing from bundle → mismatch."""
        bundle = _make_bundle_with_conditions([])
        gt = PatientGroundTruth(patient_id="p1", seed=42)
        gt.findings.append(FindingGroundTruth(
            condition_name="emphysema",
            snomed_code="87433001",
            classification="CHRONIC_STABLE",
            tier=1, module="copd",
            override_probability=1.0, temporal_class="chronic",
        ))
        result = validate_ground_truth_concordance(bundle, gt)
        assert result["concordant"] == 0
        assert len(result["mismatches"]) == 1
        assert result["concordance_rate"] == 0.0

    def test_tier3_skipped(self):
        """Tier 3 findings should not be checked for concordance."""
        bundle = _make_bundle_with_conditions([])
        gt = PatientGroundTruth(patient_id="p1", seed=42)
        gt.findings.append(FindingGroundTruth(
            condition_name="thyroid nodule",
            snomed_code="237495005",
            classification="CHRONIC_STABLE",
            tier=3, module=None,
            override_probability=1.0, temporal_class="incidental",
        ))
        result = validate_ground_truth_concordance(bundle, gt)
        assert result["checked"] == 0
        assert result["concordance_rate"] == 100.0


# ---------------------------------------------------------------------------
# Utility Tests
# ---------------------------------------------------------------------------

class TestSyntheaBundleHasCondition:
    """Test _synthea_bundle_has_condition helper."""

    def test_finds_matching_condition(self):
        bundle = _make_bundle_with_conditions(["Chronic obstructive lung disease"])
        assert _synthea_bundle_has_condition(bundle, "obstructive lung disease")

    def test_no_match(self):
        bundle = _make_bundle_with_conditions(["Hypertension"])
        assert not _synthea_bundle_has_condition(bundle, "emphysema")

    def test_skips_pipeline_conditions(self):
        """Pipeline-created conditions (encounter-diagnosis) should be skipped."""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "pipeline-cond",
                        "code": {"text": "emphysema"},
                        "category": [{
                            "coding": [{"code": "encounter-diagnosis"}]
                        }],
                    }
                }
            ],
        }
        assert not _synthea_bundle_has_condition(bundle, "emphysema")
