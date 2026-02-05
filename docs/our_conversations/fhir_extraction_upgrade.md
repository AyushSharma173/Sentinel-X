# FHIR Janitor Module: Dense Clinical Stream Architecture

## Overview

The FHIR Janitor module replaces the previous "Agentic Search" approach with a "Dense Clinical Stream" architecture. It transforms FHIR bundles into condensed, chronological narratives optimized for MedGemma's training format (longitudinal patient narratives).

**Key Benefits:**
- Reduced token usage (~12K vs potential 100K+ for raw FHIR)
- Chronological organization matches clinical thinking
- Automatic noise removal (billing records, organization metadata)
- Extracted hidden diagnoses from claims data
- Structured active medication tracking

## Module Location

```
sentinel_x/triage/fhir_janitor.py
```

## Architecture Components

### 1. GarbageCollector

Removes noise resources and extracts hidden diagnoses from billing records.

**Resources Deleted on Sight:**
- `Provenance` - Audit trail data
- `Organization` - Provider organization details
- `PractitionerRole` - Practitioner assignments
- `Coverage` - Insurance coverage
- `Device` - Medical device references

**Conditional Handling (Claim/ExplanationOfBenefit):**
1. Extract diagnoses from `diagnosis[].diagnosisCodeableConcept`
2. Compare against existing Condition resource codes
3. If code NOT in Conditions → extract as HistoricalDiagnosis
4. Discard the Claim/EOB resource

### 2. NarrativeDecoder

Extracts and parses Base64-encoded content from DiagnosticReports.

**Process:**
1. Check `presentedForm[].data` for Base64 content
2. Decode: `base64.b64decode(data).decode('utf-8')`
3. Parse FINDINGS and IMPRESSION sections using regex
4. Truncate if exceeds `JANITOR_MAX_NARRATIVE_LENGTH` (500 chars)
5. Fallback to `conclusion` field if no Base64 content

### 3. Resource Extractors

Per-resource-type field extraction with safety guardrails.

| Resource | Key Fields Preserved |
|----------|---------------------|
| **Patient** | age, gender, deceased status |
| **Condition** | display, clinicalStatus, onsetDateTime |
| **MedicationRequest** | name, status, dosageInstruction (timing, dose, frequency) |
| **Observation** | name, valueQuantity (value+unit), interpretation (H/HH/L/LL) |
| **Procedure** | display, performedDateTime |
| **Encounter** | type, period.start, reasonCode |

### 4. TimelineSerializer

Formats entries into chronological narrative text.

**Date Handling:**
- Parse dates to datetime objects
- Missing dates → `[Historical/Undated]` label

**Sorting:**
1. Primary: date ascending (undated entries first)
2. Secondary: priority (Encounters > Conditions > Procedures > Meds > Labs)

## Data Structures

```python
@dataclass
class TimelineEntry:
    date: Optional[datetime]
    date_label: str        # "2024-01-15" or "[Historical/Undated]"
    category: str          # "Condition", "Medication", "Lab", etc.
    content: str           # Narrative text
    priority: int          # For same-date sorting (lower = higher)

@dataclass
class ClinicalStream:
    patient_summary: str        # "71-year-old male"
    narrative: str              # Full chronological text
    token_estimate: int         # Approx token count (~4 chars/token)
    extraction_warnings: List[str]
    active_medications: List[str]
```

## Output Format

```
== PATIENT ==
71-year-old male

== CLINICAL TIMELINE ==

[Historical/Undated]
- Historical Diagnosis: Hypertension (from billing records)

2020-03-15
- Encounter: Primary care visit
- Diagnosis: Essential hypertension (active)
- Medication: Lisinopril 10mg 1.0 once daily (active)
- Lab: Creatinine 1.2 mg/dL (Normal)

2024-06-01
- Encounter: Emergency department visit
- Lab: D-dimer 1250 ng/mL (Critical High)
- Narrative (Report):
  FINDINGS: Filling defect in right main pulmonary artery...
  IMPRESSION: Acute pulmonary embolism.

== ACTIVE MEDICATIONS ==
- Lisinopril 10mg once daily (since 2020-03-15)
- Warfarin 5mg once daily (since 2024-06-01)
```

## Configuration Constants

Added to `sentinel_x/triage/config.py`:

```python
# Resource types to discard completely
JANITOR_DISCARD_RESOURCES = {
    "Provenance", "Organization", "PractitionerRole", "Coverage", "Device"
}

# Resource types for conditional processing
JANITOR_CONDITIONAL_RESOURCES = {"Claim", "ExplanationOfBenefit"}

# Label for entries without dates
JANITOR_UNDATED_LABEL = "[Historical/Undated]"

# Max length for narrative sections
JANITOR_MAX_NARRATIVE_LENGTH = 500

# Target maximum tokens for the entire stream
JANITOR_TARGET_MAX_TOKENS = 16000
```

## Integration

### agent.py Changes

```python
# Import
from .fhir_janitor import FHIRJanitor

# In _process_patient_internal():
fhir_bundle = self._load_fhir_bundle(patient_data.report_path)
janitor = FHIRJanitor()
clinical_stream = janitor.process_bundle(fhir_bundle)
context_text = clinical_stream.narrative

# Log warnings
for warning in clinical_stream.extraction_warnings:
    self.logger.warning(f"[{patient_id}] FHIR extraction: {warning}")
```

## Safety Guardrails

### Medications - MUST Preserve:
- `medicationCodeableConcept.coding[].display` (drug name)
- `status` (active/stopped/completed)
- `dosageInstruction[].timing.repeat` (frequency, period, periodUnit)
- `dosageInstruction[].doseAndRate[].doseQuantity` (value, unit)

### Labs - MUST Preserve:
- `code.coding[].display` (test name)
- `valueQuantity.value` + `valueQuantity.unit`
- `interpretation[].coding[].code` → mapped to: H=High, HH=Critical High, L=Low, LL=Critical Low

## Error Handling

| Scenario | Handling |
|----------|----------|
| Missing Patient resource | "Unknown patient demographics" |
| Invalid Base64 | Skip narrative, log warning |
| Missing date | Use `[Historical/Undated]` |
| Malformed dosageInstruction | Extract name only |
| Missing interpretation | Show value+unit only |
| Token count exceeds limit | Add warning to extraction_warnings |

## Testing

### Unit Tests

Located at `tests/test_fhir_janitor.py` (42 tests):

- GarbageCollector tests (6)
- NarrativeDecoder tests (4)
- PatientExtractor tests (3)
- ConditionExtractor tests (3)
- MedicationExtractor tests (3)
- ObservationExtractor tests (4)
- ProcedureExtractor tests (2)
- EncounterExtractor tests (2)
- TimelineSerializer tests (4)
- TimelineEntry tests (3)
- FHIRJanitor integration tests (7)
- Interpretation mapping tests (1)

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run only fhir_janitor tests
python -m pytest tests/test_fhir_janitor.py -v
```

### Integration Test Results

Using `data/raw_ct_rate/combined/train_1_a_1/fhir.json`:

- **Patient Summary:** 71-year-old male
- **Token Estimate:** ~11,885 (under 16K target)
- **Active Medications:** 12 extracted
- **Warnings:** 0

## Files Modified

| File | Changes |
|------|---------|
| `triage/fhir_janitor.py` | New module (created) |
| `triage/config.py` | Added JANITOR_* constants |
| `triage/agent.py` | Updated imports and context extraction |
| `tests/test_fhir_janitor.py` | New test file (42 tests) |

## Files Deprecated (Kept for Fallback)

- `triage/tools.py` - ReAct agent tools (still used for agent loop)
- `triage/agent_loop.py` - Agentic loop (still used for clinical correlation)
- `triage/fhir_context.py` - Legacy extraction (used for conditions list in output)

## Future Improvements

1. **Token Budget Management:** Implement smarter truncation when approaching token limit
2. **Lab Grouping:** Group labs by panel (CBC, BMP, etc.) for cleaner output
3. **Narrative Summarization:** Use LLM to summarize very long findings sections
4. **Duplicate Detection:** Better deduplication of conditions across sources
5. **Reference Resolution:** Resolve FHIR references to include linked resource details
