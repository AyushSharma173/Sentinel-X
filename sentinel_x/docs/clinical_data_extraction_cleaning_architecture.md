# Clinical Data Extraction & Cleaning Architecture

**Sentinel-X Triage Pipeline — Technical Specification**
**Date:** 2026-02-09
**Status:** Architecture Report (Pre-Implementation)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Analysis](#2-current-architecture-analysis)
3. [FHIR Data Scale Analysis](#3-fhir-data-scale-analysis)
4. [What Radiologists Need for Chest CT](#4-what-radiologists-need-for-chest-ct)
5. [Synthea Module Mapping](#5-synthea-module-mapping)
6. [Proposed Architecture: Multi-Stage Smart Compression](#6-proposed-architecture-multi-stage-smart-compression)
7. [Implementation Plan](#7-implementation-plan)
8. [Expected Impact](#8-expected-impact)
9. [References](#9-references)

---

## 1. Executive Summary

### The Problem

The Sentinel-X late-fusion pipeline sends clinical history to the 27B reasoning model that is **enormous, unfiltered, and chronologically sorted oldest-first**. This architecture causes critical data loss through cascading truncation:

- The FHIR Janitor produces ~18K tokens of narrative
- Janitor self-truncates to 16K tokens (Gate 2)
- Phase 2 prompt builder truncates to 12K characters / ~3K tokens (Gate 3)
- **58% of the clinical narrative is silently discarded**
- Because entries are sorted oldest-first, the **most recent and clinically important data is what gets cut**

### The train_2 Failure Case

A 66-year-old female (train_2) with a significant medical history including ischemic heart disease, CABG (2024), pulmonary emphysema, pulmonary nodule, and cardiac medications (clopidogrel, metoprolol, nitroglycerin) had a **large right pleural effusion with compressive atelectasis** detected on CT.

The 27B model correctly classified this as **Priority 1 ACUTE_NEW**. But:

1. The janitor produced ~18,172 tokens of narrative, truncated to ~16,000
2. Phase 2 truncated again to 12,000 characters
3. The clinical history fed to the model was consumed by dental caries from 1965, employment status changes, depression screenings, and repeated identical lab panels from 2016–2019
4. The model's JSON response was truncated mid-sentence at 488 tokens (`max_new_tokens=512`)
5. JSON parsing failed → `"No delta entries produced."`
6. Patient classified as **ROUTINE (Priority 3)** instead of **CRITICAL (Priority 1)**

The model performed correctly. The infrastructure sabotaged the result.

### Target Outcome

A smart clinical data compression system that:

| Metric | Current | Target |
|--------|---------|--------|
| Clinical narrative size | ~7,000 tokens (truncated from 18K) | ~500–800 tokens |
| Chest-CT-relevant data retained | ~42% (rest truncated) | 100% |
| Dental/social/admin noise | ~65% of content | 0% |
| Lab representation | Hundreds of repeated individual values | Latest-only with delta flags |
| Sort order | Chronological oldest-first | Structured sections by clinical relevance |
| Compression ratio | 1:0.17 (lossy, critical data lost) | 1:0.04 (lossless for relevant data) |

---

## 2. Current Architecture Analysis

### Data Flow

```
Raw FHIR Bundle (1.9–6.3 MB, ~800K–1.6M tokens)
    │
    ▼
GarbageCollector
    ├── Discard: Provenance, Organization, PractitionerRole, Coverage, Device
    ├── Mine-then-discard: Claim, ExplanationOfBenefit (extract hidden diagnoses)
    └── Pass-through: Patient, Condition, MedicationRequest, Observation,
                      Procedure, Encounter, DiagnosticReport
    │
    ▼
Resource Extractors (6 extractors)
    ├── PatientExtractor → "66-year-old female"
    ├── ConditionExtractor → "Diagnosis: Pulmonary emphysema (active)"
    ├── MedicationExtractor → "Medication: Clopidogrel 75 MG (active)"
    ├── ObservationExtractor → "Lab: Hemoglobin 12.7 g/dL (Normal)"
    ├── ProcedureExtractor → "Procedure: Spirometry"
    └── EncounterExtractor → "Encounter: General examination"
    │
    ▼
TimelineSerializer
    ├── Sort: undated entries first, then ascending by date (oldest → newest)
    ├── Group by date_label with "- " prefix per entry
    └── Append active medications section
    │
    ▼
Gate 2: Janitor Truncation
    ├── Token estimate: len(narrative) // 4
    ├── Threshold: JANITOR_TARGET_MAX_TOKENS = 16,000 (= 64,000 chars)
    └── Action: narrative[:64000] + "\n\n[... narrative truncated ...]"
    │
    ▼
Gate 3: Phase 2 Prompt Truncation
    ├── Threshold: PHASE2_MAX_NARRATIVE_CHARS = 12,000 (~3,000 tokens)
    └── Action: clinical_narrative[:12000] + "\n\n[... clinical history truncated ...]"
    │
    ▼
27B Reasoning Model (Phase 2)
    └── Receives: truncated clinical narrative + full visual narrative
```

### Files Involved

| File | Lines | Role |
|------|-------|------|
| `triage/fhir_janitor.py` | 1,074 | GarbageCollector, NarrativeDecoder, 6 resource extractors, TimelineSerializer, FHIRJanitor orchestrator |
| `triage/config.py` | 128 | `JANITOR_TARGET_MAX_TOKENS=16000`, `JANITOR_DISCARD_RESOURCES`, `JANITOR_CONDITIONAL_RESOURCES`, `HIGH_RISK_CONDITIONS` |
| `triage/prompts.py` | 118 | `PHASE2_MAX_NARRATIVE_CHARS=12000`, `build_phase2_user_prompt()` truncation function |
| `triage/agent.py` | 375 | Pipeline orchestrator: loads FHIR → calls janitor → calls Phase 2 reasoner |
| `triage/medgemma_reasoner.py` | ~200 | `ClinicalReasoner.analyze()` invokes `build_phase2_user_prompt()` internally, `max_new_tokens=512` |

### The Double-Truncation Problem

The clinical narrative passes through **three truncation gates**:

| Gate | Location | Threshold | What Gets Cut |
|------|----------|-----------|---------------|
| Gate 1 | `NarrativeDecoder._parse_report_text()` | 500 chars per section | Individual FINDINGS/IMPRESSION from DiagnosticReports |
| Gate 2 | `FHIRJanitor.process_bundle()` | 16,000 tokens (64,000 chars) | Entire serialized timeline |
| Gate 3 | `build_phase2_user_prompt()` | 12,000 chars (~3,000 tokens) | Clinical narrative in Phase 2 prompt |

Gate 3 is the **binding constraint**. Since 12,000 chars < 64,000 chars, Gate 2 is effectively redundant in all real cases. The actual budget reaching the 27B model is **12,000 characters**.

### Chronological-Oldest-First Sort Problem

`TimelineEntry.__lt__` sorts undated entries first, then ascending by date. This means:

1. Historical/undated diagnoses (from billing record mining) appear first
2. 1960s–1970s dental caries, gingivitis, and cracked teeth follow
3. 1980s social findings (education, employment) come next
4. **2024–2026 IHD, CABG, cardiac medications, and recent labs are at the END**

With a 12,000 character budget consumed by old, irrelevant data, **the most recent and critical clinical information is systematically truncated**.

### Current FHIRJanitor Class Architecture

```
FHIRJanitor (orchestrator)
├── GarbageCollector
│   ├── process(entries, condition_codes) → cleaned_entries, historical_diagnoses
│   └── _extract_hidden_diagnoses(entry) → List[HistoricalDiagnosis]
├── NarrativeDecoder
│   ├── decode_report(entry) → (findings, impression, date)
│   └── _parse_report_text(text) → (findings, impression)
├── PatientExtractor.extract(entry) → (summary, age, gender)
├── ConditionExtractor.extract(entry) → Optional[TimelineEntry]
├── MedicationExtractor.extract(entry) → Optional[(TimelineEntry, active_med, med_name)]
├── ObservationExtractor.extract(entry) → Optional[TimelineEntry]
├── ProcedureExtractor.extract(entry) → Optional[TimelineEntry]
├── EncounterExtractor.extract(entry) → Optional[TimelineEntry]
└── TimelineSerializer
    └── serialize(patient_summary, entries, active_meds) → str
```

Each extractor produces `TimelineEntry` objects with:
- `date`: Optional[datetime] — used for sorting
- `date_label`: str — display string (e.g., "2024-06-16")
- `category`: str — one of: Encounter, Condition, Procedure, Medication, Lab, Narrative
- `content`: str — free-text description (e.g., "Diagnosis: Pulmonary emphysema (active)")
- `priority`: int — category-based sort tiebreaker (Encounter=1 through Narrative=6)

---

## 3. FHIR Data Scale Analysis

### Dataset Overview

| Metric | train_1 | train_2 |
|--------|---------|---------|
| Total FHIR entries | 839 | 1,922 |
| File size | 3.1 MB | 6.3 MB |
| Character count | 3,198,456 | 6,516,012 |
| Estimated raw tokens | ~800K | ~1.6M |
| Janitor output tokens | ~8,000 | ~18,172 |
| Phase 2 effective input | ~3,000 | ~3,000 (truncated) |
| Compression ratio (raw → Phase 2) | 267:1 | 533:1 |

### Resource Type Distribution

#### train_1 (839 entries)

| Resource Type | Count | % | Disposition |
|---------------|-------|---|-------------|
| Observation | 307 | 36.6% | Processed |
| DiagnosticReport | 94 | 11.2% | Processed |
| Procedure | 85 | 10.1% | Processed |
| Claim | 67 | 8.0% | Mine + Discard |
| ExplanationOfBenefit | 67 | 8.0% | Mine + Discard |
| Condition | 51 | 6.1% | Processed |
| Encounter | 45 | 5.4% | Processed |
| DocumentReference | 45 | 5.4% | Ignored (no extractor) |
| MedicationRequest | 22 | 2.6% | Processed |
| SupplyDelivery | 22 | 2.6% | Ignored |
| Immunization | 13 | 1.5% | Ignored |
| CarePlan | 6 | 0.7% | Ignored |
| CareTeam | 6 | 0.7% | Ignored |
| Device | 4 | 0.5% | Discarded |
| ImagingStudy | 2 | 0.2% | Ignored |
| Patient | 1 | 0.1% | Processed |
| Provenance | 1 | 0.1% | Discarded |
| RiskAssessment | 1 | 0.1% | Ignored |

#### train_2 (1,922 entries)

| Resource Type | Count | % | Disposition |
|---------------|-------|---|-------------|
| Observation | 905 | 47.1% | Processed |
| DiagnosticReport | 190 | 9.9% | Processed |
| Procedure | 168 | 8.7% | Processed |
| Claim | 150 | 7.8% | Mine + Discard |
| ExplanationOfBenefit | 150 | 7.8% | Mine + Discard |
| MedicationRequest | 88 | 4.6% | Processed |
| Encounter | 62 | 3.2% | Processed |
| DocumentReference | 62 | 3.2% | Ignored |
| Condition | 52 | 2.7% | Processed |
| SupplyDelivery | 21 | 1.1% | Ignored |
| Immunization | 14 | 0.7% | Ignored |
| ImagingStudy | 11 | 0.6% | Ignored |
| MedicationAdministration | 9 | 0.5% | Ignored |
| Medication | 9 | 0.5% | Ignored |
| AllergyIntolerance | 8 | 0.4% | Ignored |
| CarePlan | 7 | 0.4% | Ignored |
| CareTeam | 7 | 0.4% | Ignored |
| Device | 6 | 0.3% | Discarded |
| Patient | 1 | 0.1% | Processed |
| Provenance | 1 | 0.1% | Discarded |
| RiskAssessment | 1 | 0.1% | Ignored |

### Observation Breakdown (train_2: 905 total)

| Category | Count | % | Relevance to Chest CT |
|----------|-------|---|----------------------|
| Laboratory | 721 | 79.7% | Mixed — many irrelevant panels |
| Vital Signs | 103 | 11.4% | High — most recent set only needed |
| Survey | 46 | 5.1% | Low — PHQ-2, DAST-10, AUDIT-C etc. |
| Procedure | 15 | 1.7% | Medium — spirometry results relevant |
| Social History | 13 | 1.4% | Low — except smoking status |
| Imaging | 7 | 0.8% | High — prior imaging findings |

#### Lab Repetitiveness (train_2)

The 721 laboratory observations span only **67 unique LOINC codes**, yielding an average repetition of **10.8x per test**.

Top repeated labs:

| LOINC | Lab Test | Occurrences | CT Relevant? |
|-------|----------|-------------|--------------|
| 2093-3 | Cholesterol | 18 | Marginal |
| 2571-8 | Triglycerides | 18 | No |
| 18262-6 | LDL Cholesterol | 18 | Marginal |
| 2085-9 | HDL Cholesterol | 18 | No |
| 6690-2 | WBC (Leukocytes) | 17 | Yes |
| 789-8 | RBC (Erythrocytes) | 17 | Marginal |
| 718-7 | Hemoglobin | 17 | Yes |
| 4544-3 | Hematocrit | 17 | Yes |
| 787-2 | MCV | 17 | No |
| 785-6 | MCH | 17 | No |
| 786-4 | MCHC | 17 | No |
| 777-3 | Platelets | 17 | Yes |
| 19123-9 | Magnesium | 16 | No |
| 2744-1 | Arterial pH | 16 | Yes |
| 2019-8 | pCO2 (arterial) | 16 | Yes |
| 2703-7 | pO2 (arterial) | 16 | Yes |
| 1960-4 | Bicarbonate (arterial) | 16 | Yes |
| 788-0 | RDW | 15 | No |
| 2345-7 | Glucose | 15 | Marginal |
| 3094-0 | BUN | 15 | Yes |

**Token waste from lab repetition**: 17 occurrences of each CBC component means 16 redundant entries per test. Across 67 unique codes at ~10.8x average repetition, approximately **90% of lab tokens represent historical duplicates**.

### Condition Analysis (train_2: 52 conditions)

#### By Clinical Relevance to Chest CT

| Category | Count | % | Examples |
|----------|-------|---|---------|
| **Chest-CT-relevant** | 8 | 15% | Pulmonary emphysema, asthma, essential HTN, IHD, CABG history, atelectasis, bronchial wall thickening, pulmonary nodule |
| **Dental/Oral** | 7 | 13% | Dental caries, gingivitis, cracked tooth, necrosis of pulp, tooth loss, periodontal disease |
| **Social/Administrative** | 14 | 27% | Stress (5x), employment status (5x), limited social contact (2x), received higher education, refugee status |
| **Medical (other)** | 10 | 19% | Prediabetes, metabolic syndrome, osteoarthritis, pharyngitis, sinusitis, osteoporosis |
| **Imaging-derived** | 5 | 10% | Emphysema, atelectasis, bronchial wall thickening, pulmonary nodule, calcified atheroma |
| **Other** | 8 | 15% | Chronic pain, anemia, cholelithiasis, degenerative changes, medication review due, tubal ligation history |

**Key finding**: Only ~15% of conditions are directly relevant to chest CT interpretation. A full 40% is dental, social, or administrative noise.

#### Complete Condition List (train_2) — Sorted by Onset

| # | Condition | Code | Status | Onset | CT Relevant? |
|---|-----------|------|--------|-------|-------------|
| 1 | Cracked tooth | ICD:K03.81 | resolved | 1965 | No |
| 2 | Dental caries, unspecified | ICD:K02.9 | resolved | 1965 | No |
| 3 | Gingivitis (disorder) | SNOMED:66383009 | active | 1970 | No |
| 4 | Complete loss of teeth due to trauma | ICD:K08.111 | resolved | 1977 | No |
| 5 | Received higher education | SNOMED:224299000 | active | 1978 | No |
| 6 | **Pulmonary emphysema** | SNOMED:87433001 | active | 1979 | **Yes** |
| 7 | **Essential hypertension** | SNOMED:59621000 | active | 1983 | **Yes** |
| 8 | History of tubal ligation | SNOMED:267020005 | active | 1985 | No |
| 9 | **Asthma** | SNOMED:195967001 | active | 1987 | **Yes** |
| 10 | Neoplasm of uncertain behavior (lip/oral) | ICD:D37.0 | resolved | 1995 | Marginal |
| 11 | Necrosis of pulp | ICD:K04.1 | resolved | 1997 | No |
| 12 | Refugee (person) | SNOMED:446654005 | active | 1998 | No |
| 13 | Disturbances in tooth eruption | ICD:K00.6 | resolved | 1998 | No |
| 14 | Degenerative changes | SNOMED:396275006 | active | 2007 | Marginal |
| 15 | Body mass index 30+ (obesity) | SNOMED:162864005 | active | 2010 | Marginal |
| 16 | Chronic pain | SNOMED:82423001 | active | 2011 | No |
| 17 | Prediabetes | SNOMED:15777000 | active | 2014 | Marginal |
| 18 | Anemia (disorder) | SNOMED:271737000 | active | 2014 | **Yes** |
| 19 | Metabolic syndrome X | SNOMED:237602007 | active | 2015 | No |
| 20 | Calcified atheroma | SNOMED:38716007 | active | 2018 | **Yes** |
| 21 | **Osteoporosis** | SNOMED:64859006 | active | 2020 | Marginal |
| 22 | Malignant neoplasm of tongue | ICD:C02.9 | resolved | 2020 | **Yes** |
| 23 | Osteoarthritis of knee | SNOMED:239873007 | active | 2021 | No |
| 24 | Atelectasis | SNOMED:46621007 | active | 2021 | **Yes** |
| 25 | Cholelithiasis | SNOMED:235919008 | active | 2022 | No |
| 26 | **Ischemic heart disease** | SNOMED:414545008 | active | 2024-06 | **Yes** |
| 27 | Abnormal findings heart/coronary | SNOMED:274531002 | active | 2024-07 | **Yes** |
| 28 | **History of CABG** | SNOMED:399261000 | active | 2024-08 | **Yes** |
| 29 | Bronchial wall thickening | SNOMED:26036001 | active | 2025-12 | **Yes** |
| 30 | **Pulmonary nodule** | SNOMED:427359005 | active | 2025-08 | **Yes** |

*(22 additional conditions omitted: 5x Stress, 5x Employment status, 2x Limited social contact, 2x Medication review due, plus viral pharyngitis, viral sinusitis, severe anxiety, intimate partner abuse, streptococcal sore throat, acute bronchitis, criminal record)*

### Active Medications (train_2)

| Medication | RxNorm | CT Relevant? |
|-----------|--------|-------------|
| **Clopidogrel 75 MG** | 309362 | **Yes** — antiplatelet, post-CABG |
| **Simvastatin 20 MG** | 312961 | **Yes** — cardiac risk management |
| **Metoprolol Succinate XL 100 MG** | 866412 | **Yes** — cardiac, beta-blocker |
| **Nitroglycerin 0.4 MG/ACTUAT spray** | 705129 | **Yes** — cardiac, IHD management |
| **Budesonide 0.25 MG/ML inhaled** | 351109 | **Yes** — pulmonary, asthma/COPD |
| **Albuterol 5 MG/ML inhaled** | 245314 | **Yes** — pulmonary, bronchodilator |
| **Fluticasone/Salmeterol DPI** | 896209 | **Yes** — pulmonary, COPD management |
| **ProAir HFA (albuterol MDI)** | 745752 | **Yes** — pulmonary, rescue inhaler |
| Amlodipine 2.5 MG | 308136 | Marginal — HTN management |
| Naproxen 220 MG | 849574 | No |
| Diphenhydramine 25 MG | 1049630 | No |
| Alendronic acid 10 MG | 904419 | No — osteoporosis |

**8 of 12 active medications are directly relevant to chest CT interpretation** (cardiac + pulmonary). In the current pipeline, these appear at the END of the chronological timeline and are at highest risk of truncation.

### Coding Systems

| System | train_2 Count | Primary Use |
|--------|--------------|-------------|
| LOINC | 1,168 | Observations, labs, diagnostic reports |
| SNOMED-CT | 217 | Conditions, procedures, findings |
| CVX | 14 | Immunizations |
| Ada CDT | 12 | Dental procedures |
| RxNorm | 9 | Medications |
| ICD-10 | 7 | Conditions (alternative to SNOMED) |

### Date Range

| Dataset | Earliest | Latest | Span |
|---------|----------|--------|------|
| train_1 | 1978-07-09 | 2026-02-09 | 48 years |
| train_2 | 1961-06-21 | 2026-02-09 | 65 years |

The vast majority of entries cluster in the 2014–2026 period (annual wellness visits with full lab panels), with sparse historical entries going back to birth.

---

## 4. What Radiologists Need for Chest CT

### Research-Based Clinical Requirements

When interpreting a chest CT, radiologists integrate imaging findings with clinical context. The following categories represent the information hierarchy based on ACR Appropriateness Criteria, radiology workflow literature, and clinical practice:

### Critical Categories (Must Retain)

#### Pulmonary Conditions
- COPD/Emphysema (current severity, FEV1/FVC)
- Asthma (severity, control status)
- Pulmonary fibrosis / ILD
- Prior pneumonia / TB exposure
- Pulmonary embolism history
- Known pulmonary nodules (size, growth trajectory)
- Bronchiectasis
- Smoking history (pack-years, current/former/never)

#### Cardiac History
- Ischemic heart disease / CAD
- Prior CABG / PCI / stenting
- Heart failure (preserved or reduced EF)
- Valvular disease
- Aortic aneurysm
- Pericardial disease
- Congenital heart defects

#### Oncologic History
- Current or prior malignancy (any site)
- Chemotherapy (drug names, current vs. completed)
- Radiation therapy to chest
- Known metastatic disease
- Pulmonary nodule surveillance status

#### Hematologic/Coagulation
- Anticoagulant/antiplatelet use (PE risk, hemorrhage context)
- Known thrombophilia
- Recent DVT/PE
- Anemia (relevant to pleural effusion etiology)

#### Renal Function
- Creatinine / eGFR (contrast safety, fluid status)
- Chronic kidney disease stage
- Dialysis status

#### Key Vital Signs
- Most recent set only: HR, BP, SpO2, temp, BMI
- Respiratory rate (dyspnea assessment)

#### Key Laboratory Values

| Lab | LOINC | Why It Matters |
|-----|-------|---------------|
| WBC | 6690-2 | Infection marker |
| Hemoglobin | 718-7 | Anemia, hemorrhage |
| Hematocrit | 4544-3 | Fluid status |
| Platelets | 777-3 | Bleeding/clotting risk |
| Creatinine | 2160-0 | Renal function, contrast safety |
| BUN | 3094-0 | Renal function |
| D-dimer | 48065-7 | PE risk stratification |
| Troponin | 6598-7 | Acute cardiac injury |
| CRP | 1988-5 | Inflammation marker |
| INR | 6301-6 | Anticoagulation status |
| HbA1c | 4548-4 | Diabetes control |
| Glucose | 2345-7 | Metabolic status |
| Arterial pH | 2744-1 | Acid-base status |
| pCO2 | 2019-8 | Respiratory function |
| pO2 | 2703-7 | Oxygenation |
| Bicarbonate | 1960-4 | Acid-base status |
| FEV1/FVC | 19926-5 | Obstructive disease severity |
| Sodium | 2951-2 | Electrolyte status |
| Potassium | 2823-3 | Electrolyte status |
| Total Cholesterol | 2093-3 | Cardiovascular risk (marginal) |

### Categories to Drop Entirely

| Category | Rationale | train_2 Examples |
|----------|-----------|-----------------|
| **Dental/Oral** | No bearing on thoracic pathology | Dental caries, gingivitis, cracked tooth, necrosis of pulp, tooth eruption, periodontal disease |
| **Social Determinants** | Administrative, not clinical | Employment status, education level, refugee status, criminal record, limited social contact |
| **Psychological Screening** | Survey scores irrelevant to imaging | PHQ-2 (depression), GAD-7 (anxiety), DAST-10 (substance abuse), AUDIT-C (alcohol), HARK (abuse), PRAPARE (social needs) |
| **Administrative** | Process markers, not clinical data | Medication review due, supply deliveries |
| **Immunizations** | No relevance to CT interpretation | Influenza vaccine, Td, pneumococcal |
| **Reproductive History** | Unless currently pregnant | History of tubal ligation |
| **Routine Dental Procedures** | CDT-coded procedures | Dental X-rays, caries risk assessment, restorations |

### Time Relevance Hierarchy

| Window | Label | Weight | What to Keep |
|--------|-------|--------|-------------|
| 0–48 hours | Acute | Highest | All vitals, labs, new symptoms, recent procedures |
| 2–8 weeks | Recent | High | New diagnoses, medication changes, recent imaging |
| 3–12 months | Contextual | Medium | Disease progression, lab trends, procedures |
| 1–5 years | Historical | Low | Major events only (CABG, cancer diagnosis, PE) |
| >5 years | Baseline | Minimal | Chronic condition onset dates only |

### ACR Appropriateness Criteria — Key Chest CT Indications

The American College of Radiology rates imaging appropriateness on a 1–9 scale. Key scenarios where clinical history matters most:

- **Acute chest pain (suspected PE)**: D-dimer, prior VTE, malignancy, recent surgery, immobility
- **Acute chest pain (cardiac)**: Troponin, ECG findings, prior CAD, cardiac risk factors
- **Chronic dyspnea**: PFTs (FEV1/FVC), smoking history, occupational exposures, heart failure status
- **Known/suspected malignancy**: Primary tumor type, staging, prior treatment, metastatic sites
- **Follow-up pulmonary nodule**: Prior nodule size, growth rate, Fleischner criteria application
- **Blunt chest trauma**: Mechanism, associated injuries, hemodynamic stability
- **Immunocompromised infection**: CD4 count, transplant status, immunosuppressant medications

---

## 5. Synthea Module Mapping

### Overview

Synthea generates synthetic patient records using JSON-based state machine modules. Each module models a disease pathway with states for ConditionOnset, Observation, Procedure, MedicationOrder, and Encounter. Understanding which modules generated the data allows code-based filtering by SNOMED/ICD-10 families.

### Disease Modules — Chest CT Relevance Classification

#### KEEP — Directly Relevant to Chest CT

| Module | SNOMED/ICD Families | Relevance |
|--------|-------------------|-----------|
| **COPD** | 13645005, 87433001 (emphysema), 185086009 | Pulmonary disease, air trapping |
| **Asthma** | 195967001, 233678006 | Airway disease, bronchial wall thickening |
| **Lung Cancer** | 254637007, 162573006 | Pulmonary nodules, masses, staging |
| **Coronary Heart Disease** | 53741008, 414545008 (IHD), 399261000 (CABG) | Cardiac silhouette, coronary calcification |
| **Congestive Heart Failure** | 88805009, 42343007 | Cardiomegaly, pleural effusions, pulmonary edema |
| **Atrial Fibrillation** | 49436004 | Cardiac enlargement, PE risk |
| **Pulmonary Embolism** | 59282003, 706870000 | Filling defects, right heart strain |
| **Pneumonia** | 233604007, 10509002 (acute bronchitis) | Consolidation, ground-glass opacity |
| **Cystic Fibrosis** | 190905008 | Bronchiectasis, mucus plugging |
| **Sarcoidosis** | 31541009 | Lymphadenopathy, pulmonary infiltrates |
| **Tuberculosis** | 56717001, 58802007 | Cavitation, lymphadenopathy, calcified granulomas |
| **Breast Cancer** | 254837009 | Chest wall invasion, axillary nodes, lung metastases |
| **Colorectal Cancer** | 363406005, 109838007 | Lung metastases |
| **Lymphoma** | 118600007, 118601006 | Mediastinal lymphadenopathy |
| **Diabetes** | 44054006, 73211009 | Cardiovascular risk factor, metabolic context |
| **Chronic Kidney Disease** | 431855005 | Fluid overload, pleural effusions, contrast contraindication |
| **Hypertension** | 59621000 | Cardiac enlargement, aortic disease |
| **Obesity** | 162864005, 408512008 | Cardiovascular risk, imaging quality |
| **Anemia** | 271737000 | Pleural effusion etiology |

#### DROP — Not Relevant to Chest CT

| Module | SNOMED/ICD Families | Rationale |
|--------|-------------------|-----------|
| **Dental Health** | CDT codes (D0120–D9999), K02.x, K04.x, K05.x, K08.x | Oral pathology |
| **Allergies** | 419199007 (as standalone findings) | Not imaging-relevant |
| **Contraception** | 169553002, 267020005 | Reproductive |
| **Ear Infections** | 65363002, 80602006 | ENT |
| **Dermatitis** | 24079001 | Dermatologic |
| **Epilepsy** | 84757009 | Neurologic |
| **Fibromyalgia** | 95417003 | Musculoskeletal |
| **Gallstones** | 235919008 (cholelithiasis) | Abdominal (unless seen on CT) |
| **GERD** | 235595009 | GI |
| **Gout** | 90560007 | Rheumatologic |
| **Lupus** | 200936003 | Autoimmune (marginal — can cause pleuritis) |
| **Osteoarthritis** | 239873007 | Joint disease |
| **Osteoporosis** | 64859006 | Bone density (marginal — compression fractures visible on CT) |
| **Pregnancy** | Unless currently pregnant | Reproductive |
| **Sinusitis** | 444814009 | ENT |
| **Urinary Tract Infection** | 301011002 | Urologic |

#### CONDITIONAL — Keep Only If Context-Dependent

| Module | Keep If... | Drop If... |
|--------|-----------|-----------|
| **Metabolic Syndrome** | Co-occurs with cardiac/pulmonary disease | Isolated finding |
| **Rheumatoid Arthritis** | ILD/pleural involvement documented | Joint-only manifestation |
| **Lupus (SLE)** | Pleuritis, pericarditis, or pulmonary involvement | Skin/joint-only |
| **Anxiety/Depression** | On medications affecting imaging (benzodiazepines) | Isolated screening scores |
| **Osteoporosis** | Vertebral compression fractures present | DXA-only finding |

### SNOMED/ICD-10 Code Families for Filtering

```
CHEST_CT_RELEVANT_SNOMED_FAMILIES:
  Pulmonary:  13645005, 87433001, 195967001, 233678006, 254637007,
              59282003, 233604007, 190905008, 31541009, 56717001
  Cardiac:    53741008, 414545008, 88805009, 42343007, 49436004,
              399261000, 274531002
  Oncologic:  254837009, 363406005, 109838007, 118600007, 118601006,
              162573006, 427359005
  Metabolic:  44054006, 73211009, 431855005, 59621000
  Hematologic: 271737000

DROP_SNOMED_FAMILIES:
  Dental:     66383009, 80353004, 109573003, 234947003
  Social:     73595000, 160903007, 160904001, 224299000, 423315002,
              446654005, 266948004, 706893006
  Admin:      314529007
  ENT:        65363002, 80602006, 444814009
  Dermatologic: 24079001
  Reproductive: 169553002, 267020005

DROP_ICD10_FAMILIES:
  Dental:     K00.x, K02.x, K03.x, K04.x, K05.x, K08.x
  Social:     Z55-Z65 (social determinants)
```

---

## 6. Proposed Architecture: Multi-Stage Smart Compression

### Overview

Replace the current single-pass chronological serialization with a **5-stage pipeline** that applies increasingly sophisticated filtering:

```
Raw FHIR Bundle
    │
    ▼
Stage 1: Category-Based Hard Filter (rule-based)
    │  Drop dental, social, admin resources by SNOMED/ICD/CDT code
    │
    ▼
Stage 2: Observation Smart Compression (rule-based)
    │  Labs → latest-value-only with delta flags
    │  Drop surveys, compress vitals
    │
    ▼
Stage 3: Time-Decay Weighted Filtering (scoring)
    │  Score each entry by category_weight × time_decay
    │  Drop entries below threshold
    │
    ▼
Stage 4: Structured Output Format (serialization)
    │  Replace chronological timeline with structured sections
    │  Target: 500–800 tokens
    │
    ▼
Stage 5: LLM-Assisted Compression (optional, overflow only)
    │  If output still exceeds budget, use 27B for summarization pre-pass
    │
    ▼
Phase 2 Reasoning Model
```

---

### Stage 1: Category-Based Hard Filter (Rule-Based)

#### Purpose
Eliminate entire categories of clinically irrelevant data before any serialization occurs. This is the highest-impact, lowest-cost filtering stage.

#### Classification Logic

For each FHIR resource, classify by examining `code.coding[].system` and `code.coding[].code`:

```python
class RelevanceCategory(Enum):
    KEEP = "keep"           # Always retain
    DROP = "drop"           # Always discard
    CONDITIONAL = "conditional"  # Keep based on context
```

#### DROP Rules

| Resource Pattern | Detection Method | Example Entries Removed |
|-----------------|-----------------|----------------------|
| Dental conditions | ICD-10: K00–K08.* | Dental caries, gingivitis, cracked tooth |
| Dental procedures | CDT: D0100–D9999 | Dental exams, restorations, extractions |
| Social findings | SNOMED: 73595000 (Stress), 160903007/160904001 (Employment), 224299000 (Education), 423315002 (Social contact), 446654005 (Refugee) | "Stress (finding)", "Full-time employment", "Received higher education" |
| Administrative | SNOMED: 314529007 (Medication review due) | "Medication review due (situation)" |
| Survey observations | Category = "survey" | PHQ-2, GAD-7, DAST-10, AUDIT-C, HARK, PRAPARE |
| Immunizations | Resource type = Immunization | All vaccine records |
| Supply deliveries | Resource type = SupplyDelivery | Diabetic test strips, home medical supplies |
| Reproductive (non-pregnancy) | SNOMED: 267020005 (Tubal ligation) | "History of tubal ligation" |

#### KEEP Rules

| Resource Pattern | Detection Method |
|-----------------|-----------------|
| Pulmonary conditions | SNOMED: 87433001, 195967001, 233678006, 254637007, etc. |
| Cardiac conditions | SNOMED: 414545008, 53741008, 399261000, 88805009, etc. |
| Oncologic conditions | SNOMED: 162573006, 254837009, ICD-10: C*.* |
| Hematologic conditions | SNOMED: 271737000 |
| Renal conditions | SNOMED: 431855005 |
| Active medications | Status = "active" (all retained for context) |
| Diagnostic reports | Resource type = DiagnosticReport (imaging narratives) |
| Imaging observations | Category = "imaging" |

#### CONDITIONAL Rules

| Resource Pattern | Keep If... |
|-----------------|-----------|
| Metabolic conditions (diabetes, metabolic syndrome) | Always keep — cardiovascular risk factor |
| Obesity | Always keep — risk factor |
| Anxiety/depression | Only if on medications that affect imaging (rare) |
| Osteoporosis | Only if vertebral findings on current CT |
| Autoimmune (RA, SLE) | Only if pulmonary manifestations documented |

#### Expected Impact
- **train_2**: Removes ~20 of 52 conditions, all 46 survey observations, 14 immunizations, 21 supply deliveries
- Estimated token reduction: **30–40%** of janitor output

---

### Stage 2: Observation Smart Compression (Rule-Based)

#### Purpose
Transform hundreds of repeated lab observations into a compact, clinically actionable format.

#### Lab Whitelist (LOINC Codes Relevant to Chest CT)

```python
LAB_WHITELIST_LOINC = {
    # Complete Blood Count
    "6690-2":  "WBC",
    "718-7":   "Hgb",
    "4544-3":  "Hct",
    "777-3":   "Plt",

    # Renal Function
    "2160-0":  "Cr",
    "3094-0":  "BUN",
    "33914-3": "eGFR",

    # Coagulation / PE Markers
    "48065-7": "D-dimer",
    "6598-7":  "Troponin",
    "6301-6":  "INR",

    # Inflammatory
    "1988-5":  "CRP",

    # Metabolic
    "4548-4":  "HbA1c",
    "2345-7":  "Glucose",
    "2951-2":  "Na",
    "2823-3":  "K",

    # Arterial Blood Gas
    "2744-1":  "pH",
    "2019-8":  "pCO2",
    "2703-7":  "pO2",
    "1960-4":  "HCO3",

    # Pulmonary Function
    "19926-5": "FEV1/FVC",
    "20150-9": "FEV1",

    # Lipids (marginal — keep for cardiac context)
    "2093-3":  "Chol",
    "18262-6": "LDL",
}
```

#### Compression Algorithm

For each whitelisted LOINC code:
1. Collect all Observation resources matching that code
2. Sort by `effectiveDateTime` descending (most recent first)
3. Keep **only the most recent value**
4. If the previous value differs significantly (>10% for numeric, any change for categorical), add a delta flag: `(prev: X, Nmo ago)`
5. Format as single-line entry: `WBC 5.2, Hgb 12.7, Cr 0.94`

#### Vital Signs Compression
- Keep only the **most recent complete set** of vital signs
- Format: `BP 138/82, HR 72, SpO2 97%, Temp 37.1°C, RR 16`
- Drop all historical vital sign entries

#### Survey Score Handling
- **Drop ALL survey observations**: PHQ-2, GAD-7, DAST-10, AUDIT-C, HARK, PRAPARE
- These are psychological/social screening instruments with no bearing on chest CT interpretation

#### Output Format

```
== RELEVANT LABS (2025-12-28) ==
WBC 5.2, Hgb 12.7 (prev: 11.3, 6mo ago), Hct 38.7%, Plt 258
Cr 0.94, BUN 15, HbA1c 6.2%, Glucose 95
pH 7.41, pCO2 42, pO2 89, HCO3 24
FEV1/FVC 21% (prev: 25%, 12mo ago)
Chol 129, LDL 44

== VITALS (2025-12-28) ==
BP 138/82, HR 72, SpO2 97%, Temp 37.1°C, BMI 30.1
```

#### Expected Impact
- **train_2**: 721 lab observations → ~25 unique latest values → 3–4 lines of text
- Estimated token reduction from labs alone: **~95%** (from ~1800 tokens to ~80 tokens)

---

### Stage 3: Time-Decay Weighted Filtering

#### Purpose
For resources that survive Stages 1–2, apply a scoring function that prioritizes recent and high-relevance data, allowing graceful degradation under token pressure.

#### Scoring Function

```python
relevance_score = category_weight × time_decay_factor
```

#### Category Weights

| Category | Weight | Rationale |
|----------|--------|-----------|
| Active pulmonary conditions | 1.0 | Primary imaging domain |
| Active cardiac conditions | 0.9 | Directly visible on CT |
| Active oncologic conditions | 0.9 | Nodule/mass interpretation |
| Active medications (relevant classes) | 0.8 | Treatment context |
| Recent labs (whitelisted) | 0.7 | Clinical correlation |
| Historical procedures (relevant) | 0.5 | Prior interventions |
| Resolved conditions | 0.3 | Background context |
| Other active conditions | 0.2 | Marginal relevance |

#### Time Decay Function

```python
import math

def time_decay(age_days: float, half_life_days: float) -> float:
    if half_life_days == float('inf'):
        return 1.0
    return math.exp(-0.693 * age_days / half_life_days)
```

#### Category-Specific Half-Lives

| Data Type | Half-Life | Rationale |
|-----------|-----------|-----------|
| Active conditions | infinity | Never decay — always relevant |
| Acute events (recent hospitalization, ER visit) | 90 days | Rapid relevance decay |
| Lab values | 180 days | 6-month clinical significance |
| Procedures | 365 days | 1-year procedural relevance |
| Resolved conditions | 730 days | 2-year historical context |
| Imaging studies | 1095 days | 3-year comparison relevance |

#### Threshold

- Drop entries where `relevance_score < 0.1`
- This effectively removes:
  - Resolved conditions older than ~5 years (0.3 × exp(-0.693 × 1825/730) ≈ 0.056)
  - Old lab values (>2 years): already handled by Stage 2
  - Procedures older than ~8 years (0.5 × exp(-0.693 × 2920/365) ≈ 0.008)

#### Scoring Example (train_2)

| Entry | Category Weight | Age (days) | Half-Life | Decay | Score | Action |
|-------|----------------|------------|-----------|-------|-------|--------|
| IHD (active, 2024-06) | 0.9 | 610 | ∞ | 1.0 | 0.90 | Keep |
| CABG (2024-08) | 0.5 | 555 | 365 | 0.35 | 0.18 | Keep |
| Emphysema (active, 1979) | 1.0 | 17,115 | ∞ | 1.0 | 1.00 | Keep |
| Dental caries (resolved, 1965) | — | — | — | — | — | Already dropped in Stage 1 |
| Creatinine (latest, 2025-12) | 0.7 | 43 | 180 | 0.85 | 0.60 | Keep |
| Cholesterol (2019 value) | 0.7 | 2,227 | 180 | 0.0003 | 0.0002 | Drop (but latest already kept in Stage 2) |

---

### Stage 4: Structured Output Format

#### Purpose
Replace the chronological free-form timeline with a structured, section-based format optimized for LLM consumption. This is the most impactful change for the reasoning model's ability to process clinical context.

#### Output Template

```
== PATIENT ==
{age}{gender}, BMI {bmi}, {smoking_status}

== ACTIVE CONDITIONS ==
{comma-separated list of active, relevant conditions with onset year}

== ACTIVE MEDICATIONS ==
{comma-separated list with dose and frequency}

== RELEVANT LABS ({most_recent_date}) ==
{single-line compressed lab values with delta flags}

== VITALS ({most_recent_date}) ==
{single-line vital signs}

== RECENT IMAGING/PROCEDURES ==
{relevant procedures and prior imaging findings, most recent first}

== ALLERGIES ==
{drug allergies only, or "NKDA" if none}
```

#### Example Output (train_2, target format)

```
== PATIENT ==
66F, BMI 30.1, Never smoker

== ACTIVE CONDITIONS ==
Pulmonary emphysema (since 1979), Asthma (since 1987), Essential HTN (since 1983),
Ischemic heart disease (since 2024-06), Hx CABG (2024-08), Anemia (since 2014),
Prediabetes (since 2014), Obesity (since 2010), Calcified atheroma (since 2018),
Hx malignant neoplasm of tongue (resolved 2020), Pulmonary nodule (since 2025-08),
Bronchial wall thickening (since 2025-12), Atelectasis (since 2021)

== ACTIVE MEDICATIONS ==
Clopidogrel 75mg daily, Simvastatin 20mg daily, Metoprolol XL 100mg daily,
Nitroglycerin 0.4mg spray PRN, Budesonide 0.25mg/mL inhaled PRN,
Albuterol 5mg/mL inhaled PRN, Fluticasone/Salmeterol DPI, ProAir HFA PRN,
Amlodipine 2.5mg daily, Alendronic acid 10mg daily

== RELEVANT LABS (2025-12-28) ==
WBC 5.2, Hgb 12.7, Hct 38.7%, Plt 258, Cr 0.94, BUN 15
HbA1c 6.2%, Glucose 95, Na 140, K 4.2
pH 7.41, pCO2 42, pO2 89, HCO3 24
FEV1/FVC 21% (prev: 25%, 12mo ago)
Chol 129, LDL 44

== VITALS (2025-12-28) ==
BP 138/82, HR 72, SpO2 97%, Temp 37.1°C, RR 16

== RECENT IMAGING/PROCEDURES ==
CABG (2024-08), Coronary angiography (2024-07)
Spirometry: severe obstruction, FEV1/FVC 21% (2025-12)
Prior CT findings: emphysema, atelectasis, bronchial wall thickening,
pulmonary nodule, calcified atheroma

== ALLERGIES ==
Fish, Peanut
```

#### Token Estimate

| Section | Estimated Tokens |
|---------|-----------------|
| Patient | 10 |
| Active Conditions | 80 |
| Active Medications | 60 |
| Relevant Labs | 50 |
| Vitals | 20 |
| Recent Imaging/Procedures | 50 |
| Allergies | 5 |
| Section headers + formatting | 25 |
| **Total** | **~300 tokens** |

This represents a **96% reduction** from the current ~7,000 tokens (pre-truncation ~18,000 tokens) while retaining **100% of chest-CT-relevant clinical data**.

---

### Stage 5: LLM-Assisted Compression (Full Design)

#### Purpose
Provide a fallback for cases where rule-based filtering still produces output exceeding the token budget, or where semantic relationships between entries require LLM understanding.

#### When to Invoke
- Only when Stage 4 output exceeds the target token budget (~800 tokens)
- Expected to be **rarely needed** after Stages 1–4
- Primary use case: patients with complex, multi-system disease where many conditions are genuinely relevant

#### Architecture

```
Stage 4 Output (if > 800 tokens)
    │
    ▼
Chunk into ~2K token segments (by section)
    │
    ▼
For each chunk → 27B Summarization Prompt
    │  "Extract only chest-CT-relevant information from this clinical data"
    │
    ▼
Merge chunk summaries
    │  Deduplicate, priority-rank by recency
    │
    ▼
Compressed Clinical Context (~500 tokens)
    │
    ▼
Phase 2 Delta Analysis (standard prompt)
```

#### Summarization Prompt Design

**System prompt:**
```
You are a clinical data summarizer for a chest CT triage system. Given clinical
data, extract ONLY information relevant to interpreting a chest CT scan. Focus on:
pulmonary conditions, cardiac history, oncologic history, relevant medications,
key lab values, and recent imaging findings. Omit unrelated conditions, normal
values, and administrative data. Output in structured bullet format.
```

**Per-chunk user prompt:**
```
Summarize the following clinical data, keeping only what is relevant for
interpreting a chest CT scan:

{chunk_text}
```

#### Chunking Strategy
- Chunk by section (conditions, medications, labs) rather than by date
- Process most recent data first (if token budget is tight, recent data is preserved)
- Maximum chunk size: 2,000 tokens (fits comfortably in context with system prompt)

#### Merge Strategy
1. Collect all chunk summaries
2. Deduplicate: same condition/medication mentioned in multiple chunks → keep most recent/detailed version
3. Priority-rank: pulmonary > cardiac > oncologic > medications > labs > procedures
4. Truncate from the bottom (lowest priority) if still over budget

#### Latency Analysis

| Step | Estimated Time | Notes |
|------|---------------|-------|
| Chunking | <1s | CPU-only string operations |
| Per-chunk inference | 60–90s | 27B model, ~200 output tokens per chunk |
| Typical chunks | 3–5 | Depends on patient complexity |
| Total latency | 3–7.5 min | Significant overhead |
| Merge | <1s | CPU-only |

#### Optimization: Conditional Invocation
- **Skip Stage 5** if Stage 4 output ≤ 800 tokens (expected for >90% of cases)
- Only invoke when patient has genuinely complex, multi-system relevant disease
- Log a warning when Stage 5 is triggered for monitoring

#### Quality Advantages
- Catches semantic relationships that rule-based filters miss:
  - "chronic cough" + "weight loss" + "smoking history" = combined cancer risk
  - "recent surgery" + "immobility" + "tachycardia" = PE risk constellation
  - "new medication" + "known allergy" = potential adverse reaction context
- Handles edge cases: conditions with ambiguous SNOMED codes, free-text entries without standard codes

#### Trade-offs Table

| Aspect | Rule-Based (Stages 1–4) | LLM-Assisted (Stage 5) | Hybrid (Recommended) |
|--------|------------------------|------------------------|---------------------|
| Latency | <1 second | 3–7.5 minutes | <1s (90%), 3–7.5m (10%) |
| Token cost | 0 | ~2K input + 500 output per chunk | Minimal (rarely invoked) |
| Determinism | 100% reproducible | Stochastic | Mostly deterministic |
| Accuracy | High for code-based data | Higher for ambiguous data | Best of both |
| Maintenance | Code lists need updating | Prompt engineering | Both |
| Edge cases | Misses semantic relationships | Catches complex patterns | Comprehensive |

#### Recommendation
Use the **hybrid approach**: Stages 1–4 for all cases, Stage 5 only when output exceeds budget. This provides sub-second processing for typical patients while maintaining a safety net for complex cases.

---

## 7. Implementation Plan

### Phase A: Refactor FHIRJanitor

#### New Classes to Add to `fhir_janitor.py`

**1. `ChestCTRelevanceFilter`**
```
Purpose: Stage 1 — classify FHIR resources as KEEP/DROP/CONDITIONAL
Input: FHIR resource entry
Output: RelevanceCategory enum

Methods:
  - classify(entry: dict) -> RelevanceCategory
  - _classify_condition(entry: dict) -> RelevanceCategory
  - _classify_observation(entry: dict) -> RelevanceCategory
  - _classify_procedure(entry: dict) -> RelevanceCategory
  - _get_code_families(entry: dict) -> Set[str]

Dependencies:
  - CHEST_CT_RELEVANT_SNOMED_CODES (from config.py)
  - DROP_CATEGORY_CODES (from config.py)
```

**2. `LabCompressor`**
```
Purpose: Stage 2 — compress lab observations to latest-value-only with delta flags
Input: List of Observation resources
Output: Compressed lab summary string

Methods:
  - compress(observations: List[dict]) -> str
  - _get_latest_values(observations: List[dict]) -> Dict[str, LatestLabValue]
  - _compute_delta(current: float, previous: float) -> Optional[str]
  - _format_lab_line(values: Dict[str, LatestLabValue]) -> str

Dependencies:
  - LAB_WHITELIST_LOINC (from config.py)
```

**3. `TimeDecayScorer`**
```
Purpose: Stage 3 — score entries by relevance × time decay
Input: TimelineEntry with date and category
Output: float relevance_score

Methods:
  - score(entry: TimelineEntry) -> float
  - _category_weight(category: str, entry: TimelineEntry) -> float
  - _time_decay(age_days: float, half_life_days: float) -> float

Dependencies:
  - TIME_DECAY_HALF_LIVES (from config.py)
  - RELEVANCE_THRESHOLD (from config.py)
```

#### Modifications to Existing Classes

**`TimelineSerializer.serialize()`**
- Replace chronological timeline output with structured section format
- New method: `serialize_structured()` that produces the Stage 4 output format
- Organize by section (Patient, Active Conditions, Medications, Labs, Vitals, Procedures, Allergies)
- Sort conditions by relevance within each section, not by date

**`FHIRJanitor.process_bundle()`**
- Insert `ChestCTRelevanceFilter` call after `GarbageCollector`
- Insert `LabCompressor` call for Observation resources
- Insert `TimeDecayScorer` call before serialization
- Call `serialize_structured()` instead of `serialize()`

#### New Constants in `config.py`

```python
# Stage 1: Category-based filtering
CHEST_CT_RELEVANT_SNOMED_CODES = {
    # Pulmonary
    "13645005", "87433001", "195967001", "233678006", "254637007",
    "59282003", "233604007", "190905008", "31541009", "56717001",
    "427359005", "46621007", "26036001",
    # Cardiac
    "53741008", "414545008", "88805009", "42343007", "49436004",
    "399261000", "274531002",
    # Oncologic (prefix matching for C*.* ICD-10)
    "162573006", "254837009", "363406005", "109838007",
    "118600007", "118601006",
    # Metabolic/Renal
    "44054006", "73211009", "431855005", "59621000",
    "162864005",  # Obesity
    # Hematologic
    "271737000",
}

DROP_SNOMED_CODES = {
    "73595000",   # Stress
    "160903007",  # Full-time employment
    "160904001",  # Part-time employment
    "224299000",  # Received higher education
    "423315002",  # Limited social contact
    "446654005",  # Refugee
    "266948004",  # Criminal record
    "706893006",  # Intimate partner abuse
    "314529007",  # Medication review due
    "267020005",  # History of tubal ligation
}

DROP_ICD10_PREFIXES = {"K00", "K02", "K03", "K04", "K05", "K08"}

DROP_OBSERVATION_CATEGORIES = {"survey"}

# Stage 2: Lab compression
LAB_WHITELIST_LOINC = {
    "6690-2": "WBC", "718-7": "Hgb", "4544-3": "Hct", "777-3": "Plt",
    "2160-0": "Cr", "3094-0": "BUN", "33914-3": "eGFR",
    "48065-7": "D-dimer", "6598-7": "Troponin", "6301-6": "INR",
    "1988-5": "CRP",
    "4548-4": "HbA1c", "2345-7": "Glucose", "2951-2": "Na", "2823-3": "K",
    "2744-1": "pH", "2019-8": "pCO2", "2703-7": "pO2", "1960-4": "HCO3",
    "19926-5": "FEV1/FVC", "20150-9": "FEV1",
    "2093-3": "Chol", "18262-6": "LDL",
}

LAB_DELTA_THRESHOLD_PERCENT = 10  # Flag if previous differs by >10%

# Stage 3: Time decay
TIME_DECAY_HALF_LIVES = {
    "active_condition": float("inf"),
    "acute_event": 90,
    "lab_value": 180,
    "procedure": 365,
    "resolved_condition": 730,
    "imaging_study": 1095,
}

CATEGORY_WEIGHTS = {
    "pulmonary": 1.0,
    "cardiac": 0.9,
    "oncologic": 0.9,
    "medication": 0.8,
    "lab": 0.7,
    "procedure": 0.5,
    "resolved": 0.3,
    "other": 0.2,
}

RELEVANCE_THRESHOLD = 0.1
TARGET_COMPRESSED_TOKENS = 800
```

### Phase B: Fix the Truncation Chain

#### Changes Required

1. **Remove Gate 3 hard truncation** in `prompts.py`:
   - After smart compression, output will be ~300–800 tokens
   - `PHASE2_MAX_NARRATIVE_CHARS = 12_000` can be raised to 4,000 tokens (~16,000 chars) as a safety net
   - The compressed output should fit well within this

2. **Align Gate 2 with Gate 3**:
   - `JANITOR_TARGET_MAX_TOKENS` should match the Phase 2 budget
   - If Phase 2 accepts 4,000 tokens, janitor should target 4,000 tokens (not 16,000)
   - Eliminate the double-truncation entirely

3. **Add truncation warnings**:
   - Log a warning when any truncation occurs after smart compression
   - This should be a **rare event** after implementation
   - Include metrics: pre-truncation tokens, post-truncation tokens, data lost

#### Suggested Constant Updates

```python
# prompts.py
PHASE2_MAX_NARRATIVE_CHARS = 16_000  # Safety net (4K tokens), rarely hit

# config.py
JANITOR_TARGET_MAX_TOKENS = 4_000  # Aligned with Phase 2 budget
```

### Phase C: Testing & Validation

#### Test Cases

1. **train_2 regression test**: Run the full pipeline and verify:
   - IHD, CABG history, and cardiac medications are **visible** in Phase 2 input
   - Emphysema, pulmonary nodule, bronchial wall thickening are present
   - No dental, social, or administrative entries in output
   - Token count is within 500–800 range
   - Phase 2 model receives complete clinical context

2. **train_1 validation**: Run and compare:
   - Token count before/after (expect ~85–90% reduction)
   - Clinical fidelity check: all pulmonary/cardiac data retained
   - No false drops (conditions incorrectly classified as irrelevant)

3. **Compression metrics**:
   - Measure total tokens at each stage
   - Log the number of entries dropped at each stage
   - Compute precision/recall for relevant data retention

4. **Edge case testing**:
   - Patient with minimal FHIR data (few entries)
   - Patient with no relevant conditions (all social/dental)
   - Patient with complex multi-system disease (many relevant conditions)

#### Validation Criteria

| Criterion | Pass Condition |
|-----------|---------------|
| Compressed token count | ≤ 800 tokens for train_1 and train_2 |
| Chest-CT-relevant conditions retained | 100% (zero false drops) |
| Irrelevant conditions removed | >95% of dental/social/admin |
| Lab compression | Latest values only, delta flags for significant changes |
| Structured format | All sections present, correctly populated |
| train_2 failure case | IHD + CABG + cardiac meds visible to model |
| No truncation warnings | Output fits within budget without truncation |

---

## 8. Expected Impact

### Token Reduction

| Stage | Input Tokens | Output Tokens | Reduction |
|-------|-------------|---------------|-----------|
| Raw FHIR | ~1,600,000 | — | — |
| Current janitor | ~1,600,000 | ~18,000 | 99% (but lossy) |
| Current Phase 2 input | ~18,000 | ~3,000 | 83% (truncation, loses critical data) |
| **Proposed Stage 1** | ~18,000 | ~11,000 | 39% |
| **Proposed Stage 2** | ~11,000 | ~3,000 | 73% |
| **Proposed Stage 3** | ~3,000 | ~1,000 | 67% |
| **Proposed Stage 4** | ~1,000 | ~300–800 | 20–70% |
| **Final output** | ~1,600,000 | **~300–800** | **>99.95%** |

### Clinical Fidelity

| Data Category | Current Pipeline | Proposed Pipeline |
|---------------|-----------------|-------------------|
| Pulmonary conditions | **Truncated** (at end of timeline) | **Retained** (structured section) |
| Cardiac history (IHD, CABG) | **Truncated** (2024 data cut) | **Retained** (Active Conditions) |
| Cardiac medications | **Truncated** (at end of timeline) | **Retained** (Active Medications) |
| Recent labs | **Truncated** (buried under old labs) | **Retained** (latest-only, compressed) |
| Dental conditions | Included (wastes tokens) | **Removed** |
| Social determinants | Included (wastes tokens) | **Removed** |
| Survey scores | Included (wastes tokens) | **Removed** |
| Repeated lab panels | Included (17x each, wastes tokens) | **Compressed** (1x each, latest only) |

### The train_2 Case — Before and After

**Before (current pipeline):**
```
Phase 2 receives: 12,000 chars of chronological data starting from 1961
  ├── 1960s: dental caries, gingivitis, allergies
  ├── 1970s: tooth loss, education status, emphysema onset
  ├── 1980s: tubal ligation, HTN, asthma onset
  ├── 1990s: oral neoplasm, tooth necrosis, refugee status
  ├── 2010s: obesity, chronic pain, repeated lab panels (hundreds of lines)
  │   ├── 2016: Cholesterol 199, Triglycerides 132, ...
  │   ├── 2017: Cholesterol 201, Triglycerides 129, ...
  │   ├── 2018: Cholesterol 195, Triglycerides 135, ...
  │   └── [... truncated at 12,000 chars ...]
  │
  ├── [NOT SEEN BY MODEL]:
  │   ├── 2020: Osteoporosis, tongue cancer (resolved)
  │   ├── 2024: IHD, CABG, coronary angiography
  │   ├── 2024: Clopidogrel, Simvastatin, Metoprolol, Nitroglycerin
  │   ├── 2025: Pulmonary nodule, bronchial wall thickening
  │   └── 2025: Budesonide, Albuterol, Fluticasone/Salmeterol
  │
  └── Result: ROUTINE (Priority 3) — JSON truncation + parsing failure
```

**After (proposed pipeline):**
```
Phase 2 receives: ~300 tokens of structured clinical data
  ├── 66F, BMI 30.1, Never smoker
  ├── Active: emphysema, asthma, HTN, IHD, CABG, anemia, pulmonary nodule,
  │   atelectasis, bronchial wall thickening, calcified atheroma
  ├── Meds: Clopidogrel, Simvastatin, Metoprolol, Nitroglycerin, Budesonide,
  │   Albuterol, Fluticasone/Salmeterol
  ├── Labs: WBC 5.2, Hgb 12.7, Cr 0.94, FEV1/FVC 21%
  ├── Recent: CABG (2024-08), Spirometry severe obstruction (2025)
  └── Allergies: Fish, Peanut

  Result: Model sees IHD + CABG + cardiac meds + emphysema progression
          → Correctly contextualizes large right pleural effusion
          → Priority 1 CRITICAL (with room for complete JSON output)
```

### Downstream Benefits

1. **Shorter Phase 2 input** → fewer input tokens → **smaller KV cache** → more room for output tokens
2. **No truncation** → model generates complete JSON → **no parsing failures**
3. **Structured format** → model can quickly identify relevant history → **better clinical reasoning**
4. **Lower latency** → less text for the model to process → **faster inference**

---

## 9. References

### Clinical Standards
- ACR Appropriateness Criteria: https://acsearch.acr.org/list
- ACR Appropriateness Criteria — Routine Chest Imaging: https://acsearch.acr.org/docs/69451/narrative/
- ACR Appropriateness Criteria — Suspected Pulmonary Embolism (2022 Revision): https://acsearch.acr.org/docs/69404/Narrative/
- Imaging protocols for CT chest (PMC6857267): https://pmc.ncbi.nlm.nih.gov/articles/PMC6857267/
- Tailoring protocols for chest CT applications (PMC5669541): https://pmc.ncbi.nlm.nih.gov/articles/PMC5669541/

### Clinical NLP & Compression
- CLEAR (Clinical Entity Augmented Retrieval): >70% token reduction with improved accuracy. https://pmc.ncbi.nlm.nih.gov/articles/PMC11743751/
- ConTextual: Context-Preserving Token Filtering with Domain-Specific Knowledge Graphs (2024). https://arxiv.org/html/2504.16394
- DistillNote: 79% text compression, 18.2% improvement in AUPRC for heart failure diagnosis. https://arxiv.org/html/2506.16777v1
- CLIN-SUMM: Temporal Summarization of Longitudinal Clinical Notes, up to 85% token reduction. https://www.medrxiv.org/content/10.64898/2025.11.28.25341233v1.full
- CeRTS: Certainty retrieval token search for clinical information extraction. https://www.sciencedirect.com/science/article/pii/S1532046425001297
- Clinical Text Summarization: Adapting LLMs Can Outperform Human Experts (PMC10635391): https://pmc.ncbi.nlm.nih.gov/articles/PMC10635391/

### FHIR & Synthetic Data
- Synthea Module Gallery: https://github.com/synthetichealth/synthea/wiki/Module-Gallery
- Synthea: An approach for generating synthetic patients (PMC7651916): https://pmc.ncbi.nlm.nih.gov/articles/PMC7651916/
- FHIR-Former: Enhancing clinical predictions through FHIR and LLMs: https://academic.oup.com/jamia/article/32/12/1793/8285046
- Synthea FHIR for Research: https://mitre.github.io/fhir-for-research/modules/synthea-overview

### Medical Coding Systems
- LOINC (Logical Observation Identifiers Names and Codes): https://loinc.org/
- SNOMED CT (Systematized Nomenclature of Medicine — Clinical Terms): https://www.snomed.org/
- RxNorm: https://www.nlm.nih.gov/research/umls/rxnorm/
- ICD-10-CM: https://www.cms.gov/medicare/coding-billing/icd-10-codes

### Decaying Relevance in Clinical Data
- Time-dependent contextual models for clinical relevance: https://www.sciencedirect.com/science/article/pii/S138650561730059X

---

*Document generated: 2026-02-09*
*Status: Architecture report — implementation pending*
*Author: Sentinel-X Development Team*
