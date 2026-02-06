# FHIR Story Generation Testing

This directory contains tests for the FHIR extraction pipeline.

## Quick Start

Run all tests:
```bash
cd /workspace/Sentinel-X
python tests/test_fhir_story_generation.py
```

## Usage Examples

### Basic Test (All Patients) - Saves to Files
```bash
python tests/test_fhir_story_generation.py
```
Processes all FHIR bundles and **saves each story to `tests/stories/`** as separate text files:
- `train_1_a_1_story.txt`
- `train_1_a_2_story.txt`
- etc.

### Display to Console Instead of Saving
```bash
python tests/test_fhir_story_generation.py --no-save
```
Print stories to console instead of saving to files.

### Verbose Mode (Shows Resource Statistics)
```bash
python tests/test_fhir_story_generation.py --verbose
```
Includes detailed bundle statistics in the saved story files showing:
- Total number of resources
- Breakdown by resource type (e.g., 1708 Observations, 288 DiagnosticReports)

### Single Patient
```bash
python tests/test_fhir_story_generation.py --patient train_1_a_1
```
Process only a specific patient to focus on one case.

### Custom Output Directory
```bash
python tests/test_fhir_story_generation.py --output-dir /path/to/output
```
Save story files to a custom directory (default is `tests/stories/`).

### Custom Data Directory
```bash
python tests/test_fhir_story_generation.py --data-dir /path/to/fhir/bundles
```

## What the Test Shows

For each FHIR bundle, the test displays:

### 1. Demographics
- Patient summary (e.g., "71-year-old male")
- Age and gender

### 2. Conditions & Risk Factors
- Complete list of all conditions extracted
- High-risk conditions identified (cancer, diabetes, heart disease, etc.)

### 3. Medications
- All medications found
- Active medications with dosage instructions

### 4. Report Content
- Findings from diagnostic reports
- Impressions/conclusions
- Extracted from Base64-encoded report text

### 5. Complete Clinical Narrative
The full chronological story formatted as:
```
== PATIENT ==
71-year-old male

== CLINICAL TIMELINE ==

[Historical/Undated]
- Historical Diagnosis: Hypertension (from billing records)

2020-03-15
- Encounter: Primary care visit
- Diagnosis: Essential hypertension (active)
- Medication: Lisinopril 10mg once daily (active)
- Lab: Creatinine 1.2 mg/dL (Normal)

== ACTIVE MEDICATIONS ==
- Lisinopril 10mg once daily (since 2020-03-15)
```

### 6. Statistics
- Token estimate (for LLM context planning)
- Narrative length in characters
- Count of conditions, medications, risk factors

## Test Results Summary

The current test data includes **4 FHIR bundles**:
- `train_1_a_1` - 71-year-old male, 54 conditions, 26 medications (51.6 KB story)
- `train_1_a_2` - 83-year-old male, 71 conditions, 82 medications (147.0 KB story)
- `train_2_a_1` - 85-year-old female, 35 conditions, 18 medications (57.9 KB story)
- `train_2_a_2` - 61-year-old male, 51 conditions, 30 medications (32.5 KB story)

All bundles process successfully with comprehensive narratives generated.

**Story files are saved to:** `tests/stories/`
- Each patient gets their own text file
- Files are human-readable and easy to review
- Complete with demographics, timeline, and statistics

## What Gets Extracted

The FHIRJanitor extracts and organizes:

### From Patient Resources
- Age (calculated from birthDate)
- Gender
- Deceased status

### From Condition Resources
- Condition display name
- Clinical status (active/resolved)
- Onset date
- Risk factor identification

### From MedicationRequest Resources
- Medication name
- Dosage instructions (frequency, dose, timing)
- Status (active/stopped/completed)
- Prescription date

### From Observation Resources
- Lab test name
- Value with units
- Interpretation flags (High/Low/Critical)
- Effective date

### From DiagnosticReport Resources
- Base64-decoded report text
- FINDINGS section
- IMPRESSION/CONCLUSION section

### From Procedure Resources
- Procedure name and date

### From Encounter Resources
- Encounter type
- Date/time
- Reason for visit

### From Claim/EOB Resources
- Hidden diagnoses (in billing but not in Conditions)
- Extracted as "Historical Diagnosis"

## Verification Checklist

Use this test to verify:
- ✓ All resource types are processed correctly
- ✓ Timeline is chronologically ordered
- ✓ Risk factors are properly identified
- ✓ Active medications include dosage information
- ✓ Base64-encoded reports are decoded
- ✓ Token estimates are reasonable (<16K target)
- ✓ No extraction warnings (or expected warnings only)

## Interpreting Results

### Good Signs
- Conditions list includes both documented conditions and historical diagnoses
- Risk factors subset includes cancer, diabetes, heart disease when present
- Active medications show full dosage instructions
- Narrative is chronological with proper date grouping
- Token estimate is under 16,000 (MedGemma context window)

### Potential Issues
- **Empty conditions list**: Check if Condition resources exist in bundle
- **No risk factors**: Patient may not have high-risk conditions (OK)
- **Missing medication dosage**: Check if dosageInstruction exists in FHIR
- **Token count too high**: May need to truncate or summarize
- **Extraction warnings**: Review warnings in output

## Integration with Main Pipeline

This test verifies the same extraction logic used in:
1. **Triage Agent** (`triage/agent.py`): Provides clinical narrative to MedGemma
2. **API Endpoints** (`api/routes/patients.py`): Returns structured patient data

Changes to `fhir_janitor.py` should be verified with this test before deployment.
