# FHIR Generation Improvement Architecture

**Document Version:** 1.0
**Date:** 2026-02-03
**Status:** Proposed
**Author:** Architecture Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Context](#2-project-context)
3. [Current State Analysis](#3-current-state-analysis)
4. [Gap Analysis](#4-gap-analysis)
5. [Synthea Capabilities Reference](#5-synthea-capabilities-reference)
6. [Proposed Improvements](#6-proposed-improvements)
7. [Implementation Specifications](#7-implementation-specifications)
8. [Testing & Validation Strategy](#8-testing--validation-strategy)
9. [Risk Assessment & Mitigation](#9-risk-assessment--mitigation)
10. [Appendices](#10-appendices)

---

## 1. Executive Summary

### Problem Statement

The Sentinel-X synthetic FHIR generation pipeline successfully creates patient records from radiology reports, but several extracted clinical parameters are not utilized, resulting in synthetic data that lacks clinical realism and consistency with actual radiology findings.

### Key Findings

| Gap | Impact | Priority |
|-----|--------|----------|
| `synthea_modules` extracted but not passed to Synthea | Synthea generates generic patients unrelated to radiology findings | High |
| `smoking_history_likely` and `cardiovascular_risk` unused | Missing clinically relevant observations | Medium |
| All conditions have identical onset timestamps | Temporally unrealistic patient histories | Medium |
| No resource linkages (DiagnosticReport ↔ Condition) | Reduced traceability for triage decisions | High |
| 44% of conditions lack SNOMED codes | Reduced semantic interoperability | Medium |

### Proposed Solution

A phased improvement plan addressing each gap through targeted modifications to `synthetic_fhir_pipeline.py`, with comprehensive validation to ensure clinical realism and FHIR compliance.

### Expected Outcomes

- **>90% SNOMED coverage** (currently ~56%)
- **>50% Synthea-radiology condition overlap** (currently ~0%)
- **100% temporal consistency** (realistic onset dates)
- **100% reference integrity** (all FHIR references resolve)

---

## 2. Project Context

### 2.1 Sentinel-X Triage System Overview

Sentinel-X is a CT triage system that combines AI-powered image analysis with clinical context from Electronic Health Records (EHR) to prioritize radiology studies based on urgency. The system uses FHIR R4 (US Core) as the standard for clinical data exchange.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SENTINEL-X TRIAGE SYSTEM                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐     │
│   │  CT Volume   │───▶│  MedGemma    │───▶│  Triage Agent    │     │
│   │  (NIfTI)     │    │  Analysis    │    │  (ReAct)         │     │
│   └──────────────┘    └──────────────┘    └────────┬─────────┘     │
│                                                    │                │
│   ┌──────────────┐                                 │                │
│   │ FHIR Bundle  │─────────────────────────────────┘                │
│   │ (Patient     │    Clinical context for risk assessment         │
│   │  History)    │                                                  │
│   └──────────────┘                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 How FHIR Data is Consumed by the Triage Agent

The triage agent uses FHIR data through two primary mechanisms:

#### 2.2.1 Direct Context Extraction (`fhir_context.py`)

The `parse_fhir_context()` function (lines 299-463) extracts:

- **Demographics**: Age, gender, deceased status
- **Conditions**: All `Condition` resources with SNOMED codes
- **Medications**: Active `MedicationRequest` and `MedicationStatement` resources
- **Risk Factors**: Conditions matching `HIGH_RISK_CONDITIONS` keywords
- **Report Content**: Findings and impressions from `DiagnosticReport`

```python
# fhir_context.py:48-66 - Risk factor identification
HIGH_RISK_CONDITIONS = {
    "cancer", "malignancy", "carcinoma", "tumor", "neoplasm",
    "diabetes", "diabetic",
    "copd", "pulmonary disease",
    "heart disease", "cardiac",
    "hypertension",
    "immunocompromised", "immunosuppressed",
}
```

#### 2.2.2 Dynamic FHIR Query Tools (`tools.py`)

The ReAct agent has access to four FHIR query tools:

| Tool | Purpose | Usage Example |
|------|---------|---------------|
| `get_patient_manifest()` | Overview of available data | First call to understand data scope |
| `search_clinical_history()` | Find relevant conditions | "Does patient have history of DVT?" |
| `get_recent_labs()` | Retrieve lab values by category | "Check D-dimer for PE evaluation" |
| `check_medication_status()` | Check active medications | "Is patient on anticoagulation?" |

**Critical Clinical Pattern**: The agent specifically checks for **treatment failure scenarios**:

```python
# tools.py:339-350 - Treatment failure detection
# - Clot WHILE ON anticoagulation = ANTICOAGULATION FAILURE = CRITICAL!
# - Stroke WHILE ON antiplatelet = ANTIPLATELET FAILURE
# - High HR WHILE ON beta-blocker = RATE CONTROL FAILURE
```

### 2.3 Critical FHIR Resources for Risk Assessment

| Resource Type | Clinical Use | Current Generation |
|---------------|--------------|-------------------|
| `Condition` | Medical history, risk stratification | ✅ Generated from extraction |
| `MedicationRequest` | Treatment failure detection | ✅ From Synthea |
| `Observation` (Labs) | Lab value correlation | ✅ From Synthea |
| `Observation` (Smoking) | Cardiovascular/pulmonary risk | ❌ Not generated |
| `RiskAssessment` | Formal risk scoring | ❌ Not generated |
| `DiagnosticReport` | Imaging findings reference | ✅ Generated |

---

## 3. Current State Analysis

### 3.1 Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SYNTHETIC FHIR GENERATION PIPELINE                        │
│                 sentinel_x/scripts/synthetic_fhir_pipeline.py                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────────────────────────────────────────────┐
│  Radiology   │     │              OPENAI EXTRACTION (gpt-4o)              │
│  Report      │────▶│  Lines 175-210: EXTRACTION_SYSTEM_PROMPT             │
│  (.json)     │     │                                                      │
└──────────────┘     │  Extracts:                                           │
                     │  ├─ conditions[]        (SNOMED codes when known)    │
                     │  ├─ demographics        (age range, gender hint)     │
                     │  ├─ smoking_history_likely  ──────────┐              │
                     │  ├─ cardiovascular_risk     ──────────┼─▶ UNUSED!    │
                     │  └─ synthea_modules[]       ──────────┘              │
                     └──────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYNTHEA GENERATION                                 │
│                     Lines 301-389: run_synthea()                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Command built (lines 320-342):                                             │
│  java -jar synthea.jar                                                      │
│    -p 1                          # Single patient                           │
│    -a {age_min}-{age_max}        # Age from extraction                      │
│    -g {gender}                   # Gender if specified                      │
│    -ps {seed}                    # Deterministic seed                       │
│    --exporter.fhir.use_us_core_ig=true                                     │
│                                                                             │
│  ⚠️  MISSING: -m {module} flags for synthea_modules[]                       │
│                                                                             │
│  Output: Generic patient with unrelated conditions                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESOURCE CREATION & MERGE                             │
│                   Lines 808-880: merge_radiology_resources()                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Creates:                                                                   │
│  ├─ ImagingStudy (lines 401-453)                                           │
│  │   └─ description: "CT Chest - {volume_name}"                            │
│  │                                                                         │
│  ├─ DiagnosticReport (lines 456-535)                                       │
│  │   ├─ category: LOINC 18748-4 "Diagnostic imaging study"                 │
│  │   ├─ code: LOINC 24627-2 "Chest CT"                                     │
│  │   └─ conclusion: Full report text                                       │
│  │                                                                         │
│  └─ Condition[] (lines 538-611)                                            │
│      ├─ code: SNOMED code (if available) or text only                      │
│      ├─ severity: SNOMED 255604002/6736007/24484000                        │
│      ├─ bodySite: text only                                                │
│      └─ onsetDateTime: scan_datetime  ⚠️  ALL SAME!                        │
│                                                                             │
│  ⚠️  NOT CREATED:                                                           │
│  ├─ Observation (Smoking Status)                                           │
│  ├─ RiskAssessment (Cardiovascular)                                        │
│  └─ Resource linkages (DiagnosticReport.result → Observation)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TEMPORAL FILTERING                                   │
│                   Lines 650-747: filter_future_events()                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Removes Synthea resources with dates AFTER scan_datetime                   │
│  Maintains simulation boundary (patient history up to scan time)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────┐
│  fhir.json   │  Final merged FHIR Bundle (US Core R4)
└──────────────┘
```

### 3.2 What Works Well

| Aspect | Implementation | Evidence |
|--------|---------------|----------|
| **Structured Extraction** | Pydantic models ensure type safety | Lines 74-130: Well-defined models |
| **US Core Compliance** | `--exporter.fhir.use_us_core_ig=true` | Synthea generates compliant resources |
| **Deterministic Generation** | SHA256-based seed from report name | Same input → same patient |
| **Temporal Filtering** | Removes future events from Synthea | Lines 650-695: Comprehensive field mapping |
| **SNOMED Integration** | Codes included when available | 56% coverage in extractions |

### 3.3 Data Flow: Radiology Report → FHIR Bundle

**Example: train_2_a_1 (Emphysema case)**

```
INPUT: Radiology Report
├── FINDINGS: "Emphysematous and passive atelectatic changes..."
└── IMPRESSION: "Emphysema, calcific atheroma, cholelithiasis..."

                    │
                    ▼ OpenAI Extraction

EXTRACTION RESULT:
├── conditions:
│   ├── Emphysema (SNOMED 87433001) ✓
│   ├── Atelectasis (SNOMED 46621007) ✓
│   ├── Pulmonary nodule (SNOMED 427359005) ✓
│   ├── Cholelithiasis (SNOMED 235919008) ✓
│   ├── Degenerative changes (NO CODE) ✗
│   └── Calcification (SNOMED 82650004) ✓
├── demographics: age 55-85
├── smoking_history_likely: true  ──────▶ UNUSED
├── cardiovascular_risk: "moderate" ────▶ UNUSED
└── synthea_modules: ["cardiovascular_disease", "copd", "osteoarthritis"]
                                    │
                                    └───▶ UNUSED

                    │
                    ▼ Synthea Generation (WITHOUT modules)

SYNTHEA OUTPUT:
├── Patient (age 55-85, random conditions)
├── Conditions: ~70 unrelated conditions (diabetes, allergies, etc.)
├── Medications: Based on Synthea's random conditions
└── Observations: Labs for Synthea's conditions

                    │
                    ▼ Merge Radiology Resources

FINAL FHIR BUNDLE:
├── Patient (from Synthea)
├── ImagingStudy (CT Chest - train_2_a_1.nii.gz)
├── DiagnosticReport (with full report text)
├── Conditions:
│   ├── ~70 Synthea conditions (unrelated to CT findings)
│   └── 6 Radiology conditions (ALL with onsetDateTime = scan time)
├── Medications (from Synthea, unrelated)
└── Observations (from Synthea, no smoking status)
```

---

## 4. Gap Analysis

### 4.1 Gap #1: Unused `synthea_modules` Parameter

**Evidence Location:** `synthetic_fhir_pipeline.py`

**Extraction (Line 127-129):**
```python
synthea_modules: list[str] = Field(
    default_factory=list,
    description="Synthea module names to use (e.g., 'copd', 'cardiovascular_disease')"
)
```

**Synthea Command (Lines 320-342):**
```python
cmd = [
    "java", "-jar", str(SYNTHEA_JAR),
    "-p", "1",
    "-a", f"{config.age_min}-{config.age_max}",
    # ... other flags ...
]
# ⚠️ NO -m flag for modules!
```

**Evidence from Processing Log:**
```json
// train_2_a_1 extraction
{
  "synthea_modules": ["cardiovascular_disease", "copd", "osteoarthritis"],
  // ↑ Extracted but never used
  "conditions": [
    {"condition_name": "Emphysema", "snomed_code": "87433001"}
  ]
}
```

**Impact:**
- Synthea generates patients with random conditions unrelated to CT findings
- Patient with emphysema on CT may have no COPD in medical history
- Reduces clinical realism and coherence

### 4.2 Gap #2: Unused Clinical Parameters

**Evidence Location:** `synthetic_fhir_pipeline.py:119-126`

```python
smoking_history_likely: bool = Field(
    default=False,
    description="Whether findings suggest smoking history"
)
cardiovascular_risk: Literal["low", "moderate", "high"] = Field(
    default="low",
    description="Assessed cardiovascular risk level"
)
```

**Usage in `merge_radiology_resources()` (Lines 808-880):**
```python
def merge_radiology_resources(
    synthea_bundle: dict,
    extraction: RadiologyExtraction,  # Contains smoking_history_likely, cardiovascular_risk
    report: dict
) -> dict:
    # Creates ImagingStudy, DiagnosticReport, Conditions
    # ⚠️ NEVER uses extraction.smoking_history_likely
    # ⚠️ NEVER uses extraction.cardiovascular_risk
```

**Evidence from Processing Log:**
```json
// train_2_a_1: Emphysema case
{
  "smoking_history_likely": true,
  "cardiovascular_risk": "moderate"
  // Both extracted, neither used in FHIR output
}

// train_3_a_1: Cardiac case
{
  "smoking_history_likely": false,
  "cardiovascular_risk": "high"
  // cardiovascular_risk: "high" not reflected in any resource
}
```

**Impact:**
- Triage agent cannot check smoking status for pulmonary findings
- Cardiovascular risk not formally documented
- Misses important clinical context

### 4.3 Gap #3: Temporal Unrealism

**Evidence Location:** `synthetic_fhir_pipeline.py:538-579`

```python
def create_condition_resource(
    patient_ref: str,
    condition: ExtractedCondition,
    onset_datetime: str  # ← Same for ALL conditions
) -> dict:
    condition_resource = {
        # ...
        "onsetDateTime": onset_datetime  # Line 578
    }
```

**Calling Code (Lines 837-841):**
```python
# Create Condition resources for extracted conditions
conditions = []
for condition in extraction.conditions:
    condition_resource = create_condition_resource(patient_ref, condition, now)
    # ↑ "now" is identical for every condition
    conditions.append(condition_resource)
```

**Example Output (train_2_a_1):**
```json
// ALL conditions have identical onset
{"resourceType": "Condition", "code": {"text": "Emphysema"}, "onsetDateTime": "2026-02-03T16:59:25.179996Z"}
{"resourceType": "Condition", "code": {"text": "Degenerative changes"}, "onsetDateTime": "2026-02-03T16:59:25.179996Z"}
{"resourceType": "Condition", "code": {"text": "Cholelithiasis"}, "onsetDateTime": "2026-02-03T16:59:25.179996Z"}
// ↑ Degenerative changes don't appear suddenly!
```

**Clinical Unrealism:**
- Degenerative spondylosis typically develops over 5-20 years
- Emphysema develops over years/decades
- Cholelithiasis (gallstones) can be present for years before detection
- Acute findings (pneumonia, PE) should have recent onset

### 4.4 Gap #4: Missing Resource Linkages

**Evidence Location:** `synthetic_fhir_pipeline.py:808-862`

```python
def merge_radiology_resources(...):
    # Creates resources but no inter-resource references

    imaging_study = create_imaging_study(patient_ref, volume_name, now)
    imaging_study_ref = f"ImagingStudy/{imaging_study['id']}"

    diagnostic_report = create_diagnostic_report(
        patient_ref, imaging_study_ref, report, extraction, now
    )
    # DiagnosticReport has imagingStudy reference ✓
    # DiagnosticReport has NO result[] references ✗

    for condition in extraction.conditions:
        condition_resource = create_condition_resource(patient_ref, condition, now)
        # Condition has NO evidence[] references ✗
        conditions.append(condition_resource)
```

**Missing FHIR References:**

| Source Resource | Missing Reference | Target Resource |
|-----------------|-------------------|-----------------|
| `DiagnosticReport` | `result[]` | `Observation` (findings) |
| `Condition` | `evidence[].detail` | `Observation` or `DiagnosticReport` |
| `Observation` | `derivedFrom` | `DiagnosticReport` |

**Impact:**
- Cannot trace "how was this condition discovered?"
- Reduced auditability for triage decisions
- Non-compliance with clinical best practices

### 4.5 Gap #5: SNOMED Code Coverage

**Evidence from Processing Log Analysis:**

| Report | Total Conditions | With SNOMED | Without SNOMED | Coverage |
|--------|------------------|-------------|----------------|----------|
| train_1_a_1 | 9 | 3 | 6 | 33% |
| train_1_a_2 | 9 | 3 | 6 | 33% |
| train_2_a_1 | 6 | 5 | 1 | 83% |
| train_2_a_2 | 6 | 5 | 1 | 83% |
| train_3_a_1 | 6 | 6 | 0 | 100% |
| **TOTAL** | **36** | **22** | **14** | **61%** |

**Conditions Missing SNOMED Codes:**
- Venous collaterals
- Collapsed left subclavian vein
- Bronchial wall thickening
- Peribronchial reticulonodular densities
- Peribronchial minimal consolidation
- Infectious process
- Atrophic left kidney
- Degenerative changes

**Current SNOMED Mapping (Lines 183-196):** Only 13 codes defined:
```python
# EXTRACTION_SYSTEM_PROMPT
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
```

**Impact:**
- Reduced semantic interoperability
- Triage agent cannot use coded queries for uncoded conditions
- Harder to map to clinical decision support rules

### 4.6 Gap #6: Synthea-Radiology Condition Disconnect

**Evidence:** Examining actual FHIR bundles shows zero overlap between Synthea-generated conditions and radiology-extracted conditions.

**Example: train_2_a_1**

| Radiology Conditions | Synthea Conditions (Sample) |
|----------------------|----------------------------|
| Emphysema | Viral sinusitis |
| Atelectasis | Acute bronchitis |
| Pulmonary nodule | Childhood asthma |
| Cholelithiasis | Streptococcal sore throat |
| Degenerative changes | Perennial allergic rhinitis |
| Calcification | Laceration of foot |

**Expected Overlap (if modules used):**
- `copd` module → COPD, Emphysema history
- `cardiovascular_disease` module → Atherosclerosis, hypertension history
- `osteoarthritis` module → Degenerative joint disease

**Current Overlap:** ~0%

---

## 5. Synthea Capabilities Reference

### 5.1 Module System

Synthea uses a modular architecture where each module represents a disease pathway or clinical scenario. Modules can be selectively enabled using the `-m` flag.

**Command Syntax:**
```bash
java -jar synthea.jar -m module_name
java -jar synthea.jar -m copd -m cardiovascular_disease  # Multiple modules
```

### 5.2 Relevant Modules for Radiology Findings

| Module Name | Conditions Generated | Radiology Correlation |
|-------------|---------------------|----------------------|
| `copd` | COPD, Emphysema, Chronic bronchitis | Emphysema on CT |
| `cardiovascular_disease` | Atherosclerosis, CAD, MI | Calcific atheromas |
| `congestive_heart_failure` | CHF, Cardiomegaly | Cardiomegaly on CT |
| `chronic_kidney_disease` | CKD stages 1-5 | Atrophic kidney |
| `osteoarthritis` | Degenerative joint disease | Spondylosis, DJD |
| `lung_cancer` | Lung cancer pathway | Pulmonary nodules |
| `colorectal_cancer` | Colorectal cancer | Abdominal findings |
| `diabetes` | Type 2 diabetes | Multi-organ findings |
| `metabolic_syndrome_disease` | Hypertension, obesity | Cardiovascular findings |

### 5.3 Keep Modules for Patient Filtering

Synthea supports "keep modules" that filter generated patients to only include those with specific conditions:

```bash
java -jar synthea.jar -k module_name  # Only generate patients who have this condition
```

**Use Case:** Ensure generated patient actually has the target condition, not just a chance of developing it.

### 5.4 Configuration Options

| Flag | Purpose | Current Use |
|------|---------|-------------|
| `-p N` | Generate N patients | ✅ `-p 1` |
| `-a MIN-MAX` | Age range | ✅ Used |
| `-g M/F` | Gender | ✅ Used if extracted |
| `-m MODULE` | Enable specific module | ❌ Not used |
| `-k MODULE` | Keep only patients with module | ❌ Not used |
| `-ps SEED` | Person seed for reproducibility | ✅ Used |
| `-r PATH` | Custom module directory | ❌ Not used |
| `-c PATH` | Custom properties file | ❌ Not used |

### 5.5 Module Validation

Synthea silently ignores invalid module names. Implementation should validate modules against the known list.

**Known Valid Modules** (partial list):
```
allergies, appendicitis, asthma, atopy, bronchitis,
cardiovascular_disease, chronic_kidney_disease, colorectal_cancer,
congestive_heart_failure, copd, dementia, dermatitis,
diabetes, ear_infections, epilepsy, fibromyalgia, food_allergies,
gallstones, gout, homelessness, hypertension, hypothyroidism,
injuries, lung_cancer, lupus, macular_degeneration,
metabolic_syndrome_disease, opioid_addiction, osteoarthritis,
osteoporosis, pregnancy, rheumatoid_arthritis, self_harm,
sexual_activity, sinusitis, sleep_apnea, sore_throat, stroke,
total_joint_replacement, urinary_tract_infections, wellness_encounters
```

---

## 6. Proposed Improvements

### 6.1 Phase 1: Pass Synthea Modules to Synthea Command

**Problem:** `synthea_modules` extracted but never passed via `-m` flag

**Target File:** `sentinel_x/scripts/synthetic_fhir_pipeline.py`

**Changes Required:**

1. **Add module validation constant** (after line 168):
```python
VALID_SYNTHEA_MODULES = {
    "allergies", "appendicitis", "asthma", "bronchitis",
    "cardiovascular_disease", "chronic_kidney_disease",
    "colorectal_cancer", "congestive_heart_failure", "copd",
    "dementia", "diabetes", "epilepsy", "fibromyalgia",
    "gallstones", "gout", "hypertension", "hypothyroidism",
    "lung_cancer", "lupus", "metabolic_syndrome_disease",
    "opioid_addiction", "osteoarthritis", "osteoporosis",
    "rheumatoid_arthritis", "sleep_apnea", "stroke",
    "urinary_tract_infections"
}
```

2. **Add validation function:**
```python
def validate_synthea_modules(modules: list[str]) -> list[str]:
    """Filter to only valid Synthea module names."""
    valid = []
    for module in modules:
        normalized = module.lower().replace(" ", "_").replace("-", "_")
        if normalized in VALID_SYNTHEA_MODULES:
            valid.append(normalized)
        else:
            logger.warning(f"Unknown Synthea module: {module}")
    return valid
```

3. **Modify `run_synthea()` (line ~340):**
```python
# Add after line 339 (after seed flag):
# Add disease modules if specified
if config.modules:
    validated_modules = validate_synthea_modules(config.modules)
    for module in validated_modules:
        cmd.extend(["-m", module])
    logger.info(f"Using Synthea modules: {validated_modules}")
```

4. **Update `SyntheaConfig` dataclass** (line ~280):
```python
@dataclass
class SyntheaConfig:
    age_min: int
    age_max: int
    gender: Optional[str] = None
    seed: Optional[int] = None
    modules: list[str] = field(default_factory=list)  # Add this
    state: str = "Massachusetts"
    output_dir: Path = SYNTHEA_TEMP_OUTPUT
```

5. **Pass modules in `process_single_report()`:**
```python
config = SyntheaConfig(
    age_min=extraction.demographics.estimated_age_min,
    age_max=extraction.demographics.estimated_age_max,
    gender=extraction.demographics.gender_hint,
    seed=seed,
    modules=extraction.synthea_modules,  # Add this
    output_dir=SYNTHEA_TEMP_OUTPUT
)
```

**Expected Outcome:**
- Patients generated with conditions matching CT findings
- COPD patient with emphysema on CT will have COPD in medical history
- Cardiac findings will correlate with cardiovascular disease history

---

### 6.2 Phase 2: Use Unused Extracted Parameters

**Problem:** `smoking_history_likely` and `cardiovascular_risk` extracted but never used

**Target File:** `sentinel_x/scripts/synthetic_fhir_pipeline.py`

**Changes Required:**

1. **Add `create_smoking_observation()` function:**
```python
def create_smoking_observation(
    patient_ref: str,
    is_smoker: bool,
    effective_datetime: str
) -> dict:
    """Create US Core Smoking Status Observation.

    Profile: http://hl7.org/fhir/us/core/StructureDefinition/us-core-smokingstatus
    """
    # SNOMED codes for smoking status
    if is_smoker:
        snomed_code = "449868002"
        display = "Current every day smoker"
    else:
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
            }]
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
```

2. **Add `create_cardiovascular_risk_assessment()` function:**
```python
def create_cardiovascular_risk_assessment(
    patient_ref: str,
    risk_level: Literal["low", "moderate", "high"],
    effective_datetime: str,
    basis_refs: list[str] = None
) -> dict:
    """Create RiskAssessment resource for cardiovascular risk.

    FHIR RiskAssessment: http://hl7.org/fhir/riskassessment.html
    """
    # Map risk level to probability
    probability_map = {
        "low": 0.1,
        "moderate": 0.3,
        "high": 0.6
    }

    risk_assessment = {
        "resourceType": "RiskAssessment",
        "id": generate_uuid(),
        "status": "final",
        "subject": {"reference": patient_ref},
        "occurrenceDateTime": effective_datetime,
        "code": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": "49601007",
                "display": "Cardiovascular disease risk assessment"
            }],
            "text": f"Cardiovascular Risk Assessment - {risk_level.upper()}"
        },
        "prediction": [{
            "outcome": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "49601007",
                    "display": "Cardiovascular disease"
                }],
                "text": "Cardiovascular disease"
            },
            "qualitativeRisk": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                    "code": risk_level,
                    "display": risk_level.capitalize()
                }]
            },
            "probabilityDecimal": probability_map.get(risk_level, 0.3)
        }]
    }

    # Add basis references if provided
    if basis_refs:
        risk_assessment["basis"] = [{"reference": ref} for ref in basis_refs]

    return risk_assessment
```

3. **Modify `merge_radiology_resources()` (after line 841):**
```python
# Create smoking status observation if extracted
if extraction.smoking_history_likely is not None:
    smoking_obs = create_smoking_observation(
        patient_ref,
        extraction.smoking_history_likely,
        now
    )
    new_entries.append({
        "fullUrl": f"urn:uuid:{smoking_obs['id']}",
        "resource": smoking_obs,
        "request": {"method": "POST", "url": "Observation"}
    })

# Create cardiovascular risk assessment
if extraction.cardiovascular_risk:
    risk_assessment = create_cardiovascular_risk_assessment(
        patient_ref,
        extraction.cardiovascular_risk,
        now
    )
    new_entries.append({
        "fullUrl": f"urn:uuid:{risk_assessment['id']}",
        "resource": risk_assessment,
        "request": {"method": "POST", "url": "RiskAssessment"}
    })
```

**Expected Outcome:**
- FHIR bundle includes US Core-compliant Smoking Status Observation
- RiskAssessment resource documents cardiovascular risk level
- Triage agent can query smoking status and risk assessments

---

### 6.3 Phase 3: Temporal Realism for Conditions

**Problem:** ALL radiology conditions get identical `onsetDateTime` (scan time)

**Target File:** `sentinel_x/scripts/synthetic_fhir_pipeline.py`

**Changes Required:**

1. **Add temporal classification mapping:**
```python
CONDITION_TEMPORAL_CLASS = {
    # Degenerative (5-20 years before scan)
    "degenerative": ["spondylosis", "osteoarthritis", "degenerative",
                     "osteophyte", "disc disease", "stenosis"],

    # Chronic (2-10 years before scan)
    "chronic": ["emphysema", "copd", "fibrosis", "bronchiectasis",
                "cardiomegaly", "atherosclerosis", "calcification",
                "chronic kidney", "diabetes", "hypertension"],

    # Subacute (2 weeks - 6 months before scan)
    "subacute": ["consolidation", "effusion", "nodule",
                 "mass", "lymphadenopathy"],

    # Acute (0-14 days before scan)
    "acute": ["pneumonia", "infection", "pneumothorax",
              "embolism", "infarct", "hemorrhage", "edema",
              "thrombus", "dissection"],

    # Incidental (discovered on scan, onset unknown)
    "incidental": ["cyst", "hemangioma", "lipoma",
                   "granuloma", "calcified granuloma"]
}
```

2. **Add classification function:**
```python
def classify_condition_temporality(condition_name: str) -> str:
    """Classify a condition by its typical temporal pattern."""
    name_lower = condition_name.lower()

    for temporal_class, keywords in CONDITION_TEMPORAL_CLASS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return temporal_class

    # Default to chronic for unknown conditions
    return "chronic"
```

3. **Add onset date calculation:**
```python
import random
from datetime import datetime, timedelta

def calculate_onset_date(
    scan_datetime: datetime,
    temporal_class: str,
    seed: int = None
) -> tuple[datetime, str]:
    """Calculate realistic onset date based on condition type.

    Returns:
        Tuple of (onset_datetime, clinical_note)
    """
    if seed:
        random.seed(seed)

    offsets = {
        "degenerative": (365 * 5, 365 * 20),    # 5-20 years
        "chronic": (365 * 2, 365 * 10),          # 2-10 years
        "subacute": (14, 180),                    # 2 weeks - 6 months
        "acute": (0, 14),                         # 0-14 days
        "incidental": (0, 0),                     # Discovered at scan
    }

    min_days, max_days = offsets.get(temporal_class, (365, 365 * 5))

    if min_days == max_days == 0:
        # Incidental: use scan date
        return scan_datetime, "Incidental finding"

    days_before = random.randint(min_days, max_days)
    onset = scan_datetime - timedelta(days=days_before)

    return onset, f"Estimated onset {days_before} days prior to imaging"
```

4. **Modify `create_condition_resource()` signature:**
```python
def create_condition_resource(
    patient_ref: str,
    condition: ExtractedCondition,
    onset_datetime: str,
    recorded_datetime: str = None,  # Add this
    note: str = None                 # Add this
) -> dict:
    # ... existing code ...

    # Add recordedDate (when condition was documented)
    if recorded_datetime:
        condition_resource["recordedDate"] = recorded_datetime

    # Add note about onset estimation
    if note:
        condition_resource["note"] = [{"text": note}]

    return condition_resource
```

5. **Update condition creation in `merge_radiology_resources()`:**
```python
# Create Condition resources with realistic temporal onset
conditions = []
scan_dt = datetime.fromisoformat(now.replace("Z", "+00:00"))

for i, condition in enumerate(extraction.conditions):
    temporal_class = classify_condition_temporality(condition.condition_name)
    onset_dt, note = calculate_onset_date(
        scan_dt,
        temporal_class,
        seed=hash(f"{volume_name}_{condition.condition_name}") % (2**31)
    )

    condition_resource = create_condition_resource(
        patient_ref,
        condition,
        onset_datetime=onset_dt.isoformat() + "Z",
        recorded_datetime=now,  # Documented at scan time
        note=note
    )
    conditions.append(condition_resource)
```

**Expected Outcome:**
- Degenerative conditions have onset 5-20 years before scan
- Chronic conditions have onset 2-10 years before scan
- Acute conditions have recent onset (0-14 days)
- `recordedDate` always equals scan date (when discovered)

---

### 6.4 Phase 4: Resource Linkages

**Problem:** No references between DiagnosticReport, Observations, and Conditions

**Target File:** `sentinel_x/scripts/synthetic_fhir_pipeline.py`

**Changes Required:**

1. **Add `create_finding_observation()` function:**
```python
def create_finding_observation(
    patient_ref: str,
    diagnostic_report_ref: str,
    condition: ExtractedCondition,
    effective_datetime: str
) -> dict:
    """Create Observation for an imaging finding.

    Links finding to DiagnosticReport and provides discrete representation.
    """
    observation = {
        "resourceType": "Observation",
        "id": generate_uuid(),
        "meta": {
            "profile": [
                "http://hl7.org/fhir/StructureDefinition/observation-imaging"
            ]
        },
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
        "effectiveDateTime": effective_datetime,
        "derivedFrom": [{"reference": diagnostic_report_ref}]
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
            "text": condition.body_site
        }

    # Add severity as component
    if condition.severity != "none":
        severity_codes = {
            "mild": ("255604002", "Mild"),
            "moderate": ("6736007", "Moderate"),
            "severe": ("24484000", "Severe")
        }
        code, display = severity_codes.get(condition.severity, ("255604002", "Mild"))
        observation["component"] = [{
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "246112005",
                    "display": "Severity"
                }]
            },
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": code,
                    "display": display
                }]
            }
        }]

    return observation
```

2. **Modify `create_diagnostic_report()` to accept result references:**
```python
def create_diagnostic_report(
    patient_ref: str,
    imaging_study_ref: str,
    report: dict,
    extraction: RadiologyExtraction,
    effective_datetime: str,
    result_refs: list[str] = None  # Add this parameter
) -> dict:
    # ... existing code ...

    # Add result references to observations
    if result_refs:
        diagnostic_report["result"] = [
            {"reference": ref} for ref in result_refs
        ]

    return diagnostic_report
```

3. **Modify `create_condition_resource()` to accept evidence:**
```python
def create_condition_resource(
    patient_ref: str,
    condition: ExtractedCondition,
    onset_datetime: str,
    recorded_datetime: str = None,
    note: str = None,
    evidence_refs: list[str] = None  # Add this
) -> dict:
    # ... existing code ...

    # Add evidence linking to observations
    if evidence_refs:
        condition_resource["evidence"] = [{
            "detail": [{"reference": ref} for ref in evidence_refs]
        }]

    return condition_resource
```

4. **Update `merge_radiology_resources()` for linked resource chain:**
```python
def merge_radiology_resources(
    synthea_bundle: dict,
    extraction: RadiologyExtraction,
    report: dict
) -> dict:
    patient_ref = get_patient_reference(synthea_bundle)
    now = datetime.utcnow().isoformat() + "Z"
    volume_name = report.get("volume_name", "unknown.nii.gz")

    # Create ImagingStudy
    imaging_study = create_imaging_study(patient_ref, volume_name, now)
    imaging_study_ref = f"ImagingStudy/{imaging_study['id']}"

    # Create Finding Observations FIRST (to get refs)
    finding_observations = []
    finding_obs_refs = []
    for condition in extraction.conditions:
        obs = create_finding_observation(
            patient_ref,
            f"DiagnosticReport/PLACEHOLDER",  # Updated below
            condition,
            now
        )
        finding_observations.append(obs)
        finding_obs_refs.append(f"Observation/{obs['id']}")

    # Create DiagnosticReport with result references
    diagnostic_report = create_diagnostic_report(
        patient_ref, imaging_study_ref, report, extraction, now,
        result_refs=finding_obs_refs
    )
    diagnostic_report_ref = f"DiagnosticReport/{diagnostic_report['id']}"

    # Update Observation.derivedFrom with actual DiagnosticReport reference
    for obs in finding_observations:
        obs["derivedFrom"] = [{"reference": diagnostic_report_ref}]

    # Create Conditions with evidence references
    conditions = []
    for i, condition in enumerate(extraction.conditions):
        evidence_ref = finding_obs_refs[i] if i < len(finding_obs_refs) else None
        condition_resource = create_condition_resource(
            patient_ref, condition, now,
            evidence_refs=[evidence_ref] if evidence_ref else None
        )
        conditions.append(condition_resource)

    # Build entries...
```

**Resource Linkage Diagram:**
```
DiagnosticReport
├── imagingStudy → ImagingStudy
└── result[] → Observation[]
                  │
                  └── derivedFrom → DiagnosticReport

Condition
└── evidence[].detail → Observation
```

**Expected Outcome:**
- Complete traceability from finding → observation → report → condition
- Triage agent can trace "why was this condition added?"
- FHIR-compliant resource relationships

---

### 6.5 Phase 5: SNOMED Code Improvement

**Problem:** 44% of extracted conditions lack SNOMED codes

**Target File:** `sentinel_x/scripts/synthetic_fhir_pipeline.py`

**Changes Required:**

1. **Expand `SNOMED_MAPPING` dictionary:**
```python
SNOMED_MAPPING = {
    # Pulmonary findings
    "emphysema": ("87433001", "Emphysema"),
    "atelectasis": ("46621007", "Atelectasis"),
    "bronchiectasis": ("12295008", "Bronchiectasis"),
    "pulmonary nodule": ("427359005", "Pulmonary nodule"),
    "lung nodule": ("427359005", "Pulmonary nodule"),
    "pulmonary fibrosis": ("51615001", "Pulmonary fibrosis"),
    "pleural effusion": ("60046008", "Pleural effusion"),
    "pneumothorax": ("36118008", "Pneumothorax"),
    "pneumonia": ("233604007", "Pneumonia"),
    "consolidation": ("95436008", "Pulmonary consolidation"),
    "ground glass": ("50196008", "Ground-glass opacity"),
    "bronchial wall thickening": ("26036001", "Bronchial wall thickening"),
    "interstitial disease": ("233703007", "Interstitial lung disease"),
    "copd": ("13645005", "Chronic obstructive pulmonary disease"),

    # Cardiovascular findings
    "atherosclerosis": ("38716007", "Atherosclerosis"),
    "atheroma": ("38716007", "Atherosclerosis"),
    "calcific plaque": ("128305009", "Calcified atherosclerotic plaque"),
    "calcific atheroma": ("128305009", "Calcified atherosclerotic plaque"),
    "atheromatous plaque": ("128305009", "Calcified atherosclerotic plaque"),
    "cardiomegaly": ("8186001", "Cardiomegaly"),
    "enlarged heart": ("8186001", "Cardiomegaly"),
    "pericardial effusion": ("373945007", "Pericardial effusion"),
    "aortic aneurysm": ("67362008", "Aortic aneurysm"),
    "coronary calcification": ("194842008", "Coronary artery calcification"),
    "calcification": ("82650004", "Calcification"),

    # Musculoskeletal findings
    "spondylosis": ("75320002", "Spondylosis"),
    "osteoarthritis": ("396275006", "Osteoarthritis"),
    "degenerative changes": ("396275006", "Osteoarthritis"),
    "degenerative disc": ("77547008", "Degenerative disc disease"),
    "osteophyte": ("88998003", "Osteophyte"),
    "scoliosis": ("298382003", "Scoliosis"),
    "kyphosis": ("414564002", "Kyphosis"),
    "compression fracture": ("207957008", "Compression fracture of vertebra"),
    "vertebral fracture": ("207957008", "Compression fracture of vertebra"),

    # Abdominal findings
    "cholelithiasis": ("235919008", "Cholelithiasis"),
    "gallstone": ("235919008", "Cholelithiasis"),
    "hepatomegaly": ("80515008", "Hepatomegaly"),
    "splenomegaly": ("16294009", "Splenomegaly"),
    "renal cyst": ("36171008", "Renal cyst"),
    "kidney cyst": ("36171008", "Renal cyst"),
    "atrophic kidney": ("16395008", "Renal atrophy"),
    "chronic kidney disease": ("709044004", "Chronic kidney disease"),
    "fatty liver": ("197321007", "Fatty liver"),
    "hepatic steatosis": ("197321007", "Fatty liver"),
    "pancreatic cyst": ("37153006", "Pancreatic cyst"),
    "adrenal nodule": ("126873006", "Adrenal nodule"),
    "adrenal adenoma": ("93911001", "Adrenal adenoma"),

    # Vascular findings
    "venous collateral": ("234042006", "Collateral vein"),
    "collateral vessel": ("234042006", "Collateral vein"),
    "thrombosis": ("64156001", "Thrombosis"),
    "pulmonary embolism": ("59282003", "Pulmonary embolism"),
    "dvt": ("128053003", "Deep vein thrombosis"),
    "aneurysm": ("432119003", "Aneurysm"),

    # Infectious/inflammatory
    "lymphadenopathy": ("30746006", "Lymphadenopathy"),
    "infectious process": ("40733004", "Infectious disease"),
    "abscess": ("128477000", "Abscess"),
    "granuloma": ("45647009", "Granuloma"),

    # Other common findings
    "hiatal hernia": ("84089009", "Hiatal hernia"),
    "thyroid nodule": ("237495005", "Thyroid nodule"),
    "breast mass": ("290078006", "Breast mass"),
    "cyst": ("441457006", "Cyst"),
    "mass": ("4147007", "Mass"),
    "nodule": ("27925004", "Nodule"),
    "lesion": ("52988006", "Lesion"),
}
```

2. **Add lookup function with partial matching:**
```python
def lookup_snomed_code(condition_name: str) -> tuple[str, str] | None:
    """Look up SNOMED code with fuzzy matching.

    Returns:
        Tuple of (code, display) or None if not found
    """
    name_lower = condition_name.lower()

    # Exact match first
    if name_lower in SNOMED_MAPPING:
        return SNOMED_MAPPING[name_lower]

    # Partial match (keyword in condition name)
    for keyword, (code, display) in SNOMED_MAPPING.items():
        if keyword in name_lower:
            return (code, display)

    # No match found
    return None
```

3. **Add post-extraction enrichment:**
```python
def enrich_snomed_codes(extraction: RadiologyExtraction) -> RadiologyExtraction:
    """Enrich extraction with SNOMED codes for conditions missing them."""
    for condition in extraction.conditions:
        if condition.snomed_code is None:
            result = lookup_snomed_code(condition.condition_name)
            if result:
                condition.snomed_code = result[0]
                logger.info(
                    f"Enriched SNOMED: {condition.condition_name} -> {result[0]} ({result[1]})"
                )
    return extraction
```

4. **Update extraction pipeline to use enrichment:**
```python
# In process_single_report(), after extraction:
extraction = await extract_from_report(report_text, client)
extraction = enrich_snomed_codes(extraction)  # Add this line
```

5. **Enhance `EXTRACTION_SYSTEM_PROMPT`:**

Add more SNOMED codes to the prompt to improve extraction accuracy:
```python
EXTRACTION_SYSTEM_PROMPT = """...

Common SNOMED-CT codes (EXPANDED):
- Emphysema: 87433001
- Bronchiectasis: 12295008
- Atelectasis: 46621007
- Pulmonary nodule: 427359005
- Pulmonary fibrosis: 51615001
- Pleural effusion: 60046008
- Consolidation: 95436008

- Atherosclerosis: 38716007
- Calcified plaque: 128305009
- Cardiomegaly: 8186001
- Pericardial effusion: 373945007
- Aortic aneurysm: 67362008

- Spondylosis: 75320002
- Osteoarthritis: 396275006
- Degenerative disc disease: 77547008
- Scoliosis: 298382003

- Cholelithiasis: 235919008
- Hepatomegaly: 80515008
- Renal cyst: 36171008
- Renal atrophy: 16395008
- Chronic kidney disease: 709044004

- Lymphadenopathy: 30746006
- Thrombosis: 64156001
- Pulmonary embolism: 59282003

If the exact SNOMED code is unknown, provide your best match from the list above.
"""
```

**Expected Outcome:**
- SNOMED coverage increases from ~56% to >90%
- Conditions like "Venous collaterals" get mapped to 234042006
- "Degenerative changes" maps to 396275006 (Osteoarthritis)

---

### 6.6 Phase 6: Validation & Consistency Checks

**Problem:** No validation of temporal consistency, reference integrity, or Synthea-radiology overlap

**Target File:** `sentinel_x/scripts/synthetic_fhir_pipeline.py`

**Changes Required:**

1. **Add temporal consistency validator:**
```python
def validate_temporal_consistency(bundle: dict) -> list[str]:
    """Validate temporal ordering of resources.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Get all dates by resource
    dates = {}
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id")

        # Extract dates based on resource type
        if resource_type == "Condition":
            onset = resource.get("onsetDateTime")
            recorded = resource.get("recordedDate")

            if onset and recorded:
                try:
                    onset_dt = datetime.fromisoformat(onset.replace("Z", "+00:00"))
                    recorded_dt = datetime.fromisoformat(recorded.replace("Z", "+00:00"))

                    if onset_dt > recorded_dt:
                        errors.append(
                            f"Condition/{resource_id}: onset ({onset}) after recorded ({recorded})"
                        )
                except ValueError:
                    pass

        elif resource_type == "DiagnosticReport":
            effective = resource.get("effectiveDateTime")
            issued = resource.get("issued")

            if effective and issued:
                try:
                    effective_dt = datetime.fromisoformat(effective.replace("Z", "+00:00"))
                    issued_dt = datetime.fromisoformat(issued.replace("Z", "+00:00"))

                    if effective_dt > issued_dt:
                        errors.append(
                            f"DiagnosticReport/{resource_id}: effective ({effective}) after issued ({issued})"
                        )
                except ValueError:
                    pass

    return errors
```

2. **Add reference integrity validator:**
```python
def validate_reference_integrity(bundle: dict) -> list[str]:
    """Validate all references resolve to existing resources.

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

    # Check all references
    def check_reference(ref_obj, source):
        if isinstance(ref_obj, dict):
            ref = ref_obj.get("reference", "")
            if ref and not ref.startswith("urn:uuid:"):
                if ref not in available_refs:
                    errors.append(f"{source}: unresolved reference '{ref}'")

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "Unknown")
        resource_id = resource.get("id", "?")
        source = f"{resource_type}/{resource_id}"

        # Check common reference fields
        check_reference(resource.get("subject"), source)
        check_reference(resource.get("patient"), source)
        check_reference(resource.get("encounter"), source)

        # Check array references
        for ref in resource.get("result", []):
            check_reference(ref, source)
        for ref in resource.get("derivedFrom", []):
            check_reference(ref, source)
        for evidence in resource.get("evidence", []):
            for detail in evidence.get("detail", []):
                check_reference(detail, source)

    return errors
```

3. **Add Synthea-radiology overlap measurement:**
```python
def validate_synthea_radiology_overlap(
    bundle: dict,
    radiology_conditions: list[str]
) -> dict:
    """Measure overlap between Synthea and radiology conditions.

    Returns:
        Dict with overlap metrics
    """
    # Get Synthea conditions (non-manually created)
    synthea_conditions = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Condition":
            if not is_manually_created(resource, None, None):
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

    # Find overlaps (partial matching)
    overlaps = []
    for rad_cond in radiology_normalized:
        for synth_cond in synthea_conditions:
            # Check if key terms overlap
            rad_words = set(rad_cond.split())
            synth_words = set(synth_cond.split())
            if rad_words & synth_words:
                overlaps.append((rad_cond, synth_cond))

    return {
        "synthea_condition_count": len(synthea_conditions),
        "radiology_condition_count": len(radiology_conditions),
        "overlap_count": len(overlaps),
        "overlap_percentage": (
            len(overlaps) / max(len(radiology_conditions), 1) * 100
        ),
        "overlapping_conditions": overlaps
    }
```

4. **Update `validate_bundle()` to use all validators:**
```python
def validate_bundle(
    bundle: dict,
    radiology_conditions: list[str] = None
) -> tuple[bool, dict]:
    """Perform comprehensive validation on the FHIR bundle.

    Returns:
        Tuple of (is_valid, metrics_dict)
    """
    metrics = {
        "is_valid": True,
        "temporal_errors": [],
        "reference_errors": [],
        "overlap_metrics": None
    }

    # Basic validation
    if not isinstance(bundle, dict):
        metrics["is_valid"] = False
        return False, metrics

    if bundle.get("resourceType") != "Bundle":
        metrics["is_valid"] = False
        return False, metrics

    # Temporal consistency
    metrics["temporal_errors"] = validate_temporal_consistency(bundle)
    if metrics["temporal_errors"]:
        logger.warning(f"Temporal validation errors: {metrics['temporal_errors']}")

    # Reference integrity
    metrics["reference_errors"] = validate_reference_integrity(bundle)
    if metrics["reference_errors"]:
        logger.warning(f"Reference integrity errors: {metrics['reference_errors']}")
        metrics["is_valid"] = False

    # Synthea-radiology overlap
    if radiology_conditions:
        metrics["overlap_metrics"] = validate_synthea_radiology_overlap(
            bundle, radiology_conditions
        )
        logger.info(
            f"Synthea-radiology overlap: {metrics['overlap_metrics']['overlap_percentage']:.1f}%"
        )

    return metrics["is_valid"], metrics
```

5. **Log validation metrics in processing_log.json:**
```python
# In process_single_report(), after bundle creation:
is_valid, validation_metrics = validate_bundle(
    merged_bundle,
    radiology_conditions=[c.condition_name for c in extraction.conditions]
)

result = {
    "report_name": report_name,
    "success": is_valid,
    "extraction": extraction.model_dump(),
    "validation": validation_metrics  # Add this
}
```

**Expected Outcome:**
- Zero temporal consistency violations
- Zero reference integrity errors
- Tracked Synthea-radiology overlap (target: >50%)
- Validation metrics in processing_log.json

---

## 7. Implementation Specifications

### 7.1 Recommended Implementation Order

| Order | Phase | Rationale | Risk Level |
|-------|-------|-----------|------------|
| 1 | Phase 5: SNOMED Codes | Low risk, immediate benefit, no dependencies | Low |
| 2 | Phase 2: Unused Parameters | Adds new resources, no changes to existing | Low |
| 3 | Phase 4: Resource Linkages | High value for triage, requires careful testing | Medium |
| 4 | Phase 3: Temporal Realism | Clinical realism improvement, moderate complexity | Medium |
| 5 | Phase 1: Synthea Modules | Highest impact, requires Synthea integration testing | Medium |
| 6 | Phase 6: Validation | Quality assurance, depends on other phases | Low |

### 7.2 Code Change Summary

| Phase | File | Functions Added | Functions Modified | Lines Changed (Est.) |
|-------|------|-----------------|-------------------|---------------------|
| 1 | synthetic_fhir_pipeline.py | `validate_synthea_modules()` | `run_synthea()`, `SyntheaConfig` | ~30 |
| 2 | synthetic_fhir_pipeline.py | `create_smoking_observation()`, `create_cardiovascular_risk_assessment()` | `merge_radiology_resources()` | ~80 |
| 3 | synthetic_fhir_pipeline.py | `classify_condition_temporality()`, `calculate_onset_date()` | `create_condition_resource()`, `merge_radiology_resources()` | ~60 |
| 4 | synthetic_fhir_pipeline.py | `create_finding_observation()` | `create_diagnostic_report()`, `create_condition_resource()`, `merge_radiology_resources()` | ~100 |
| 5 | synthetic_fhir_pipeline.py | `lookup_snomed_code()`, `enrich_snomed_codes()` | `EXTRACTION_SYSTEM_PROMPT` | ~120 |
| 6 | synthetic_fhir_pipeline.py | `validate_temporal_consistency()`, `validate_reference_integrity()`, `validate_synthea_radiology_overlap()` | `validate_bundle()` | ~100 |

**Total Estimated New Lines:** ~490

### 7.3 Dependency Map

```
Phase 5 (SNOMED) ─────────────────────────────────────┐
                                                      │
Phase 2 (Unused Params) ──────────────────────────────┼─▶ Phase 6 (Validation)
                                                      │
Phase 4 (Linkages) ◀── Phase 3 (Temporal) ◀───────────┤
                                                      │
Phase 1 (Synthea Modules) ────────────────────────────┘
```

- **Phase 5** is independent, can be done first
- **Phase 2** is independent, can be done in parallel with Phase 5
- **Phase 3** benefits from Phase 5 (better condition classification)
- **Phase 4** benefits from Phase 3 (temporal-aware linkages)
- **Phase 1** can be done independently but testing benefits from other phases
- **Phase 6** should be done last as it validates all other phases

---

## 8. Testing & Validation Strategy

### 8.1 Unit Tests

**Phase 1: Synthea Modules**
```python
def test_validate_synthea_modules():
    assert validate_synthea_modules(["copd", "invalid"]) == ["copd"]
    assert validate_synthea_modules(["cardiovascular_disease"]) == ["cardiovascular_disease"]
    assert validate_synthea_modules([]) == []

def test_synthea_command_includes_modules():
    config = SyntheaConfig(age_min=50, age_max=70, modules=["copd", "diabetes"])
    cmd = build_synthea_command(config)
    assert "-m" in cmd
    assert "copd" in cmd
    assert "diabetes" in cmd
```

**Phase 2: Unused Parameters**
```python
def test_create_smoking_observation_smoker():
    obs = create_smoking_observation("Patient/123", True, "2026-01-01T00:00:00Z")
    assert obs["resourceType"] == "Observation"
    assert obs["valueCodeableConcept"]["coding"][0]["code"] == "449868002"

def test_create_cardiovascular_risk_assessment():
    risk = create_cardiovascular_risk_assessment("Patient/123", "high", "2026-01-01T00:00:00Z")
    assert risk["resourceType"] == "RiskAssessment"
    assert risk["prediction"][0]["qualitativeRisk"]["coding"][0]["code"] == "high"
```

**Phase 3: Temporal Realism**
```python
def test_classify_condition_temporality():
    assert classify_condition_temporality("Spondylosis") == "degenerative"
    assert classify_condition_temporality("Emphysema") == "chronic"
    assert classify_condition_temporality("Pneumonia") == "acute"

def test_calculate_onset_date_degenerative():
    scan_dt = datetime(2026, 1, 1)
    onset_dt, _ = calculate_onset_date(scan_dt, "degenerative", seed=42)
    years_before = (scan_dt - onset_dt).days / 365
    assert 5 <= years_before <= 20
```

**Phase 4: Resource Linkages**
```python
def test_diagnostic_report_has_result_refs():
    report = create_diagnostic_report(..., result_refs=["Observation/123"])
    assert "result" in report
    assert report["result"][0]["reference"] == "Observation/123"

def test_condition_has_evidence_refs():
    condition = create_condition_resource(..., evidence_refs=["Observation/456"])
    assert "evidence" in condition
    assert condition["evidence"][0]["detail"][0]["reference"] == "Observation/456"
```

**Phase 5: SNOMED Codes**
```python
def test_lookup_snomed_code_exact():
    result = lookup_snomed_code("emphysema")
    assert result == ("87433001", "Emphysema")

def test_lookup_snomed_code_partial():
    result = lookup_snomed_code("calcific atheromatous plaques")
    assert result[0] == "128305009"

def test_enrich_snomed_codes():
    extraction = RadiologyExtraction(conditions=[
        ExtractedCondition(condition_name="Venous collaterals", snomed_code=None)
    ])
    enriched = enrich_snomed_codes(extraction)
    assert enriched.conditions[0].snomed_code == "234042006"
```

**Phase 6: Validation**
```python
def test_validate_temporal_consistency_valid():
    bundle = create_test_bundle_valid_temporal()
    errors = validate_temporal_consistency(bundle)
    assert errors == []

def test_validate_reference_integrity_invalid():
    bundle = {"resourceType": "Bundle", "entry": [{
        "resource": {
            "resourceType": "Condition",
            "subject": {"reference": "Patient/nonexistent"}
        }
    }]}
    errors = validate_reference_integrity(bundle)
    assert len(errors) == 1
```

### 8.2 Integration Tests

```python
def test_full_pipeline_with_modules():
    """Test end-to-end pipeline produces coherent FHIR bundle."""
    report = load_test_report("emphysema_case.json")
    bundle = process_single_report(report)

    # Check Synthea generated COPD-related conditions
    synthea_conditions = get_synthea_conditions(bundle)
    assert any("copd" in c.lower() or "bronchitis" in c.lower() for c in synthea_conditions)

    # Check smoking status observation exists
    smoking_obs = find_observation_by_code(bundle, "72166-2")
    assert smoking_obs is not None

    # Check all references resolve
    _, metrics = validate_bundle(bundle)
    assert metrics["reference_errors"] == []

def test_temporal_distribution():
    """Test conditions have realistic temporal distribution."""
    report = load_test_report("mixed_findings.json")
    bundle = process_single_report(report)

    conditions = get_radiology_conditions(bundle)

    # Group by temporal class
    degenerative_onsets = [c["onsetDateTime"] for c in conditions if is_degenerative(c)]
    acute_onsets = [c["onsetDateTime"] for c in conditions if is_acute(c)]

    # Degenerative should be older than acute
    if degenerative_onsets and acute_onsets:
        oldest_degenerative = min(degenerative_onsets)
        newest_acute = max(acute_onsets)
        assert oldest_degenerative < newest_acute
```

### 8.3 Validation Metrics

| Metric | Current Baseline | Target | Measurement Method |
|--------|-----------------|--------|-------------------|
| SNOMED coverage | 61% (22/36) | >90% | Count conditions with snomed_code |
| Reference integrity errors | Unknown | 0 | `validate_reference_integrity()` |
| Temporal violations | 0 (but unrealistic) | 0 (realistic) | `validate_temporal_consistency()` |
| Synthea-radiology overlap | ~0% | >50% | `validate_synthea_radiology_overlap()` |
| US Core profile compliance | ~100% | 100% | FHIR validator |

### 8.4 Regression Testing

Before/after comparison for existing test cases:

```python
def test_no_regression_train_2_a_1():
    """Ensure improvements don't break existing functionality."""
    bundle = process_single_report("train_2_a_1.json")

    # Existing assertions still pass
    assert find_imaging_study(bundle) is not None
    assert find_diagnostic_report(bundle) is not None
    assert len(get_radiology_conditions(bundle)) >= 5

    # New improvements are present
    assert find_smoking_observation(bundle) is not None
    assert len(validate_reference_integrity(bundle)) == 0
```

---

## 9. Risk Assessment & Mitigation

### 9.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Synthea module names change between versions | Low | Medium | Pin Synthea version, maintain module list |
| Invalid SNOMED codes | Low | Medium | Validate against SNOMED browser before adding |
| Temporal calculations produce impossible dates | Medium | Low | Add bounds checking, validate against patient birthdate |
| Reference integrity breaks existing consumers | Low | High | Comprehensive integration tests, phased rollout |
| Performance degradation from additional validation | Medium | Low | Profile validation code, make validation optional |
| OpenAI extraction quality degrades with prompt changes | Medium | Medium | A/B test prompt changes, maintain baseline metrics |

### 9.2 Mitigation Strategies

**Synthea Version Pinning:**
```python
SYNTHEA_VERSION = "3.2.0"  # Pin version
VALID_SYNTHEA_MODULES = {...}  # Validated for this version
```

**SNOMED Validation:**
- All SNOMED codes verified against https://browser.ihtsdotools.org/
- Include SNOMED version in mapping comments

**Temporal Bounds Checking:**
```python
def calculate_onset_date(...):
    # Ensure onset is not before patient birth
    patient_birthdate = get_patient_birthdate(bundle)
    if onset < patient_birthdate:
        onset = patient_birthdate + timedelta(days=30)
    return onset
```

**Backwards Compatibility:**
- Keep existing function signatures
- Add new parameters with defaults
- Phase rollout: shadow mode → canary → full deployment

### 9.3 Rollback Plan

Each phase can be independently reverted:

1. **Phase 1**: Remove `-m` flags from Synthea command
2. **Phase 2**: Remove smoking/risk resources from `merge_radiology_resources()`
3. **Phase 3**: Revert to using `now` for all onset dates
4. **Phase 4**: Remove linkage parameters from resource creation
5. **Phase 5**: Remove `enrich_snomed_codes()` call
6. **Phase 6**: Disable validation in `validate_bundle()`

---

## 10. Appendices

### 10.1 SNOMED Code Reference

| Condition Category | Conditions | SNOMED Codes |
|-------------------|------------|--------------|
| Pulmonary | Emphysema, COPD, Bronchiectasis, Atelectasis, Fibrosis, Nodule, Effusion | 87433001, 13645005, 12295008, 46621007, 51615001, 427359005, 60046008 |
| Cardiovascular | Atherosclerosis, Calcified plaque, Cardiomegaly, Aneurysm | 38716007, 128305009, 8186001, 67362008 |
| Musculoskeletal | Spondylosis, Osteoarthritis, Scoliosis, Osteophyte | 75320002, 396275006, 298382003, 88998003 |
| Abdominal | Cholelithiasis, Hepatomegaly, Renal cyst, Fatty liver | 235919008, 80515008, 36171008, 197321007 |
| Vascular | Thrombosis, PE, DVT, Collateral vein | 64156001, 59282003, 128053003, 234042006 |

### 10.2 Synthea Module Reference

| Module | Primary Conditions | Use When |
|--------|-------------------|----------|
| `copd` | COPD, Emphysema, Chronic bronchitis | Emphysema, bronchiectasis on CT |
| `cardiovascular_disease` | Atherosclerosis, MI, Angina | Calcified atheromas, CAD findings |
| `congestive_heart_failure` | CHF, Cardiomegaly | Cardiomegaly on CT |
| `chronic_kidney_disease` | CKD stages 1-5 | Renal atrophy, CKD findings |
| `osteoarthritis` | Degenerative joint disease | Spondylosis, DJD |
| `lung_cancer` | Lung cancer pathway | Suspicious lung nodules |
| `diabetes` | Type 2 diabetes | Multi-system findings |
| `metabolic_syndrome_disease` | Hypertension, obesity | Cardiovascular findings |

### 10.3 US Core Profiles Used

| Profile | URL | Used In |
|---------|-----|---------|
| US Core Condition | `http://hl7.org/fhir/us/core/StructureDefinition/us-core-condition-encounter-diagnosis` | All Condition resources |
| US Core Smoking Status | `http://hl7.org/fhir/us/core/StructureDefinition/us-core-smokingstatus` | Smoking Observation |
| US Core DiagnosticReport | `http://hl7.org/fhir/us/core/StructureDefinition/us-core-diagnosticreport-note` | DiagnosticReport |

### 10.4 File Reference Index

| File | Line Range | Content |
|------|------------|---------|
| `synthetic_fhir_pipeline.py` | 74-130 | Pydantic extraction models |
| `synthetic_fhir_pipeline.py` | 137-168 | CONDITION_TO_MODULE mapping |
| `synthetic_fhir_pipeline.py` | 175-210 | EXTRACTION_SYSTEM_PROMPT |
| `synthetic_fhir_pipeline.py` | 301-389 | run_synthea() function |
| `synthetic_fhir_pipeline.py` | 538-611 | create_condition_resource() |
| `synthetic_fhir_pipeline.py` | 808-880 | merge_radiology_resources() |
| `tools.py` | 66-152 | get_patient_manifest() |
| `tools.py` | 155-218 | search_clinical_history() |
| `tools.py` | 221-330 | get_recent_labs() |
| `tools.py` | 333-499 | check_medication_status() |
| `fhir_context.py` | 48-66 | HIGH_RISK_CONDITIONS |
| `fhir_context.py` | 118-174 | extract_conditions() |
| `fhir_context.py` | 299-463 | parse_fhir_context() |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Architecture Review | Initial document |

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Technical Lead | | | |
| Clinical Advisor | | | |
| QA Lead | | | |
