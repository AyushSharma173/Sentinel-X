# FHIR & MedGemma Technical Analysis

> **Document Version:** 1.0
> **Date:** January 2026
> **Scope:** Analysis of FHIR data utilization and MedGemma capabilities in Sentinel-X

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Implementation Analysis](#2-current-implementation-analysis)
3. [MedGemma Training Deep Dive](#3-medgemma-training-deep-dive)
4. [Gap Analysis](#4-gap-analysis)
5. [What FHIR Data We're Discarding](#5-what-fhir-data-were-discarding)
6. [Recommendations](#6-recommendations)
7. [Appendices](#7-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

This document provides a rigorous technical analysis of how Sentinel-X currently processes FHIR clinical data and 3D CT volumes, compared to the capabilities that MedGemma was trained to leverage. We identify critical gaps where we are underutilizing both the rich FHIR data available and MedGemma's advanced reasoning capabilities.

### 1.2 Key Findings

| Aspect | Current State | MedGemma Capability | Gap Severity |
|--------|--------------|---------------------|--------------|
| **FHIR Temporal Data** | Discarded | Trained via RL on temporal queries | **CRITICAL** |
| **Multi-hop Reasoning** | Not utilized | Core training methodology | **CRITICAL** |
| **Model Variant** | 4B-IT | 27B has full RL training | **HIGH** |
| **3D Spatial Continuity** | Independent slices | Video adapter for 3D | **HIGH** |
| **Observation Values** | Not extracted | Trained on lab/vital data | **MEDIUM** |
| **Encounter Timeline** | Not extracted | Essential for context | **MEDIUM** |

### 1.3 Clinical Significance

The current implementation may miss critical clinical correlations:

1. **Temporal Blind Spots**: Cannot answer "Did symptoms worsen after starting medication X?"
2. **Missing Lab Context**: Ignores elevated troponin, D-dimer, or creatinine values
3. **Treatment Timeline**: No awareness of recent procedures or interventions
4. **Disease Progression**: Cannot track condition evolution over time

### 1.4 High-Level Architecture Comparison

```
CURRENT IMPLEMENTATION:
┌─────────────────────────────────────────────────────────────────────────┐
│                           Sentinel-X Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐   │
│  │ FHIR Bundle │───>│ fhir_context.py  │───>│ Flattened Markdown  │   │
│  │   (926+     │    │ Extracts:        │    │ "67yo male,         │   │
│  │  resources) │    │ - Age/Gender     │    │  diabetes, obesity" │   │
│  │             │    │ - Conditions     │    │                     │   │
│  │  DISCARDS:  │    │ - Medications    │    │  NO TEMPORAL INFO   │   │
│  │  - Dates    │    │ - Risk factors   │    │  NO OBSERVATIONS    │   │
│  │  - Labs     │    │                  │    │  NO ENCOUNTERS      │   │
│  │  - Vitals   │    │                  │    │                     │   │
│  └─────────────┘    └──────────────────┘    └──────────┬──────────┘   │
│                                                         │              │
│  ┌─────────────┐    ┌──────────────────┐               │              │
│  │  NIfTI CT   │───>│ ct_processor.py  │               │              │
│  │   Volume    │    │                  │               │              │
│  │             │    │ - HU windowing   │               ▼              │
│  │  3D DATA    │    │ - Sample 85      │    ┌─────────────────────┐   │
│  │  TREATED AS │    │   slices         │    │ MedGemma 4B-IT      │   │
│  │  2D STACK   │    │ - Convert to     │───>│                     │   │
│  │             │    │   RGB images     │    │ Single inference    │   │
│  │             │    │                  │    │ 85 images + text    │   │
│  └─────────────┘    └──────────────────┘    └─────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

IDEAL IMPLEMENTATION (with MedGemma 27B capabilities):
┌─────────────────────────────────────────────────────────────────────────┐
│                      Enhanced Sentinel-X Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐   │
│  │ FHIR Bundle │───>│ Enhanced FHIR    │───>│ Structured Timeline │   │
│  │             │    │ Processor        │    │                     │   │
│  │  UTILIZES:  │    │                  │    │ Timeline:           │   │
│  │  - Dates    │    │ Builds temporal  │    │ 2023-01: Diabetes Dx│   │
│  │  - Labs     │    │ graph of events  │    │ 2023-06: Started    │   │
│  │  - Vitals   │    │                  │    │          Metformin  │   │
│  │  - Procs    │    │ Extracts obs     │    │ 2024-01: HbA1c 7.2% │   │
│  │  - Encs     │    │ with values      │    │ 2024-03: CT ordered │   │
│  └─────────────┘    └──────────────────┘    └──────────┬──────────┘   │
│                                                         │              │
│  ┌─────────────┐    ┌──────────────────┐               │              │
│  │  NIfTI CT   │───>│ 3D Volume        │               │              │
│  │   Volume    │    │ Processor        │               │              │
│  │             │    │                  │               ▼              │
│  │  TREATED AS │    │ - Video adapter  │    ┌─────────────────────┐   │
│  │  TRUE 3D    │    │   encoding       │    │ MedGemma 27B        │   │
│  │  VOLUME     │    │ - Spatial        │───>│                     │   │
│  │             │    │   continuity     │    │ Multi-hop reasoning │   │
│  │             │    │   preserved      │    │ Temporal queries    │   │
│  └─────────────┘    └──────────────────┘    └─────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Current Implementation Analysis

### 2.1 FHIR Context Processing

**File:** `triage/fhir_context.py`

The FHIR processor extracts clinical context but discards significant temporal and quantitative data.

#### 2.1.1 PatientContext Data Model

```python
# triage/fhir_context.py:14-26

@dataclass
class PatientContext:
    """Extracted patient clinical context."""
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    conditions: List[str] = field(default_factory=list)      # Text only, no dates
    risk_factors: List[str] = field(default_factory=list)    # Derived from conditions
    medications: List[str] = field(default_factory=list)     # Text only, no dates
    allergies: List[str] = field(default_factory=list)       # Currently unused
    findings: str = ""                                        # From DiagnosticReport
    impressions: str = ""                                     # From DiagnosticReport
```

**Critical Omissions:**
- No `onsetDateTime` for conditions
- No `authoredOn` for medications
- No `effectiveDateTime` for observations
- No encounter timeline
- No observation values (labs, vitals)
- No procedure history

#### 2.1.2 Condition Extraction

```python
# triage/fhir_context.py:105-135

def extract_conditions(fhir_bundle: Dict) -> List[str]:
    """Extract conditions from FHIR Condition resources."""
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
                conditions.append(text)  # <-- Only text, no temporal data!

    return conditions
```

**What's discarded from each Condition resource:**

| FHIR Field | Description | Status |
|------------|-------------|--------|
| `onsetDateTime` | When condition started | **DISCARDED** |
| `abatementDateTime` | When condition resolved | **DISCARDED** |
| `clinicalStatus` | active/resolved/remission | **DISCARDED** |
| `verificationStatus` | confirmed/unconfirmed | **DISCARDED** |
| `severity` | Condition severity code | **DISCARDED** |
| `encounter` | Related encounter reference | **DISCARDED** |
| `recordedDate` | When recorded in system | **DISCARDED** |

#### 2.1.3 Medication Extraction

```python
# triage/fhir_context.py:138-168

def extract_medications(fhir_bundle: Dict) -> List[str]:
    """Extract medications from FHIR MedicationStatement/Request resources."""
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
                medications.append(text)  # <-- Only text, no dates!

    return medications
```

**What's discarded from each MedicationRequest:**

| FHIR Field | Description | Status |
|------------|-------------|--------|
| `authoredOn` | When prescribed | **DISCARDED** |
| `status` | active/completed/stopped | **DISCARDED** |
| `intent` | order/proposal/plan | **DISCARDED** |
| `dosageInstruction` | Dosage details | **DISCARDED** |
| `dispenseRequest` | Quantity, duration | **DISCARDED** |
| `requester` | Prescribing physician | **DISCARDED** |

#### 2.1.4 Context Formatting for Prompt

```python
# triage/fhir_context.py:267-306

def format_context_for_prompt(context: PatientContext) -> str:
    """Format patient context for inclusion in MedGemma prompt."""
    lines = ["## EHR Clinical Context"]

    # Demographics
    demo_parts = []
    if context.age:
        demo_parts.append(f"{context.age} year old")
    if context.gender:
        demo_parts.append(context.gender)
    if demo_parts:
        lines.append(f"**Demographics:** {' '.join(demo_parts)}")

    # Conditions - flat list, no temporal info
    if context.conditions:
        lines.append(f"**Medical History:** {', '.join(context.conditions)}")

    # Risk factors
    if context.risk_factors:
        lines.append(f"**High-Risk Factors:** {', '.join(context.risk_factors)}")

    # Medications - flat list, no temporal info
    if context.medications:
        lines.append(f"**Current Medications:** {', '.join(context.medications)}")

    # Report content
    if context.findings:
        lines.append(f"\n## Radiology Report Findings\n{context.findings}")

    if context.impressions:
        lines.append(f"\n## Radiology Report Impressions\n{context.impressions}")

    return "\n".join(lines)
```

**Example Output (Current):**
```markdown
## EHR Clinical Context
**Demographics:** 67 year old female
**Medical History:** Body mass index 30+ - obesity (finding), Prediabetes,
                     Chronic intractable migraine without aura, Chronic pain
**High-Risk Factors:** Prediabetes
**Current Medications:** Clopidogrel 75 MG Oral Tablet

## Radiology Report Findings
[findings text]
```

**What MedGemma could utilize (but doesn't receive):**
```markdown
## Clinical Timeline

### Conditions (Chronological)
| Date | Condition | Status |
|------|-----------|--------|
| 1967-01-30 | Obesity (BMI 30+) | Active |
| 1982-02-15 | Prediabetes | Active |
| 1982-02-15 | Anemia | Active |
| 1990-04-01 | Chronic migraine | Active |
| 2007-02-12 | Coronary artery disease | Active |

### Recent Medications
| Started | Medication | Status |
|---------|------------|--------|
| 2007-02-12 | Clopidogrel 75 MG | Active |
| 2020-05-01 | Metformin 500 MG | Active |

### Recent Lab Values
| Date | Test | Value | Reference |
|------|------|-------|-----------|
| 2024-01-15 | HbA1c | 7.2% | <6.5% |
| 2024-01-15 | Creatinine | 1.4 mg/dL | 0.7-1.3 |
| 2024-01-15 | Troponin | <0.01 | <0.04 |

### Recent Vital Signs
| Date | BP | HR | SpO2 |
|------|----|----|------|
| 2024-01-20 | 142/88 | 82 | 96% |
```

### 2.2 CT Volume Processing

**File:** `triage/ct_processor.py`

#### 2.2.1 Volume Loading and Windowing

```python
# triage/ct_processor.py:100-126

def process_ct_volume(path: Path) -> Tuple[List[Image.Image], List[int], dict]:
    """Full CT processing pipeline."""
    # Load volume
    volume, metadata = load_nifti_volume(path)

    # Apply soft tissue windowing (L=40, W=400)
    windowed = apply_window(volume, CT_WINDOW_CENTER, CT_WINDOW_WIDTH)

    # Sample slice indices
    slice_indices = sample_slices(windowed, CT_NUM_SLICES)  # 85 slices

    # Extract images
    images = []
    for idx in slice_indices:
        img = extract_slice_as_image(windowed, idx)
        images.append(img)

    logger.info(f"Extracted {len(images)} slices from volume")

    return images, slice_indices, metadata
```

**Current Processing:**

```
3D NIfTI Volume (e.g., 512x512x300)
           │
           ▼
┌─────────────────────────────────────┐
│ apply_window(center=40, width=400)  │  <-- Soft tissue window only
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ sample_slices(num_slices=85)        │  <-- Uniform sampling
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ For each slice:                     │
│   - Extract 2D array                │
│   - Rotate 90°                      │
│   - Convert grayscale → RGB         │
│   - Create PIL Image                │
└─────────────────────────────────────┘
           │
           ▼
List[PIL.Image] (85 independent images)
```

#### 2.2.2 Slice Extraction

```python
# triage/ct_processor.py:79-97

def extract_slice_as_image(volume: np.ndarray, slice_idx: int) -> Image.Image:
    """Extract a single axial slice as a PIL Image."""
    slice_data = volume[:, :, slice_idx]

    # Rotate for proper orientation
    slice_data = np.rot90(slice_data)

    # Convert to RGB
    rgb_slice = np.stack([slice_data] * 3, axis=-1)  # <-- Grayscale → RGB

    return Image.fromarray(rgb_slice.astype(np.uint8), mode="RGB")
```

**Issues with current approach:**

1. **Single Window**: Only soft tissue window (L=40, W=400), missing:
   - Lung window (L=-600, W=1500) for pulmonary pathology
   - Bone window (L=400, W=1800) for skeletal findings
   - PE window (L=100, W=700) for pulmonary embolism

2. **Spatial Discontinuity**: 85 slices sent as independent images
   - No inter-slice context
   - Model cannot perceive 3D structures
   - Lesion spanning multiple slices appears fragmented

3. **Fixed Sampling**: Uniform slice distribution regardless of anatomy
   - May miss pathology concentrated in specific regions
   - Chest CT may undersample lung fields
   - Abdomen may undersample liver/kidney regions

### 2.3 MedGemma Model Interface

**File:** `triage/medgemma_analyzer.py`

#### 2.3.1 Model Configuration

```python
# triage/config.py:18
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"  # Using 4B variant
```

**Model Comparison:**

| Aspect | MedGemma 4B-IT | MedGemma 27B |
|--------|----------------|--------------|
| Parameters | 4 billion | 27 billion |
| VRAM Required | ~12 GB | ~65 GB |
| RL Training on FHIR | Limited | **Full Synthea training** |
| Multi-hop Reasoning | Basic | **Advanced** |
| EHR Temporal Understanding | Minimal | **Trained capability** |
| 3D Video Adapter | Not available | **Available in 1.5** |

#### 2.3.2 Message Construction

```python
# triage/medgemma_analyzer.py:63-93

def _build_messages(
    self,
    images: List[Image.Image],
    context_text: str
) -> List[dict]:
    """Build message format for MedGemma."""
    # Build content list with images and text
    content = []

    # Add all images first
    for _ in images:
        content.append({"type": "image"})

    # Add the user prompt
    user_prompt = build_user_prompt(context_text, len(images))
    content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]

    return messages
```

**Current message structure:**
```
[System Message]
├── SYSTEM_PROMPT (triage instructions)

[User Message]
├── [image] × 85 (CT slices as independent images)
└── [text] User prompt with:
    ├── Flattened clinical context
    └── Triage instructions
```

**Issues:**
1. Images sent as independent entities (no video/3D framing)
2. Clinical context lacks temporal structure
3. No multi-hop query formulation
4. No explicit prompting for temporal reasoning

#### 2.3.3 Prompt Templates

```python
# triage/prompts.py:3-55

SYSTEM_PROMPT = """You are an expert radiologist AI assistant performing
triage analysis of chest CT scans. Your task is to analyze CT images
alongside clinical context to assign priority levels for radiologist review.

## Priority Level Definitions

**PRIORITY 1 - CRITICAL**: Acute, life-threatening pathology requiring
immediate attention
- Pulmonary embolism (PE)
- Aortic dissection
...

## Output Format

You MUST provide your analysis in the following structured format:

VISUAL_FINDINGS: [Detailed description of all findings visible in the CT images]
KEY_SLICE: [Integer index 0-84 of the most diagnostically important slice]
PRIORITY_LEVEL: [1, 2, or 3]
PRIORITY_RATIONALE: [Explanation combining visual findings and clinical context]
...
"""

USER_PROMPT_TEMPLATE = """Analyze the following chest CT scan and clinical
information for triage prioritization.

{context}

Please examine all {num_slices} CT slices provided and deliver your
structured analysis following the exact format specified. Consider both
the visual findings and the clinical context when determining priority level.
"""
```

**Missing from prompts:**
1. No temporal reasoning instructions
2. No multi-hop query examples
3. No guidance on lab value interpretation
4. No encounter timeline correlation

---

## 3. MedGemma Training Deep Dive

### 3.1 Model Variants

MedGemma is available in multiple configurations, each with different capabilities:

| Model | Base | Size | Vision | RL Training | EHR Focus |
|-------|------|------|--------|-------------|-----------|
| MedGemma 4B-IT | Gemma 2 | 4B | Yes | Minimal | Limited |
| MedGemma 27B | Gemma 2 | 27B | Yes | **Extensive** | **Synthea-based** |
| MedGemma 1.5 | Gemini 1.5 | Varies | **3D Video** | Yes | Yes |

### 3.2 Reinforcement Learning on FHIR Data

The 27B variant underwent specialized training on Electronic Health Record data using reinforcement learning. This is a critical differentiator.

#### 3.2.1 Training Data: Synthea Synthetic EHRs

**Key Point:** MedGemma was trained on **Synthea-generated FHIR bundles** - the exact same data format Sentinel-X uses!

Synthea generates realistic patient records including:
- Complete FHIR R4 bundles
- Realistic disease progressions
- Temporal relationships between events
- Lab value trajectories
- Medication timelines

**Our Data Structure (from `data/raw_ct_rate/combined/train_1_a_1/fhir.json`):**

```json
{
  "resourceType": "Bundle",
  "entry": [
    // 926 resources including:
    // - 259 Observations (vitals, labs)
    // - 153 Procedures
    // - 112 DiagnosticReports
    // - 80 Claims
    // - 70 Encounters
    // - 50 Conditions
    // - 10 MedicationRequests
    // ... and more
  ]
}
```

#### 3.2.2 Multi-Hop Reasoning Training

MedGemma's RL training specifically targeted **multi-hop reasoning** over patient timelines.

**Training Methodology:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Hop RL Training                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Generate multi-hop question from FHIR timeline:              │
│     "What was the patient's HbA1c value AFTER starting           │
│      Metformin in March 2023?"                                   │
│                                                                  │
│  2. Model must:                                                  │
│     a) Find Metformin start date (MedicationRequest.authoredOn)  │
│     b) Find HbA1c observations AFTER that date                   │
│     c) Return the correct value                                  │
│                                                                  │
│  3. Reward signal:                                               │
│     ✓ Correct answer → Positive reward                           │
│     ✗ Incorrect answer → No reward                               │
│                                                                  │
│  This teaches the model to:                                      │
│  - Navigate temporal relationships                               │
│  - Connect logically distant facts                               │
│  - Understand before/after/during semantics                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Example Multi-Hop Scenarios:**

| Question Type | Example | FHIR Fields Required |
|---------------|---------|---------------------|
| Temporal | "Did blood pressure improve after starting lisinopril?" | MedicationRequest.authoredOn, Observation.effectiveDateTime |
| Causal | "Was the patient diabetic before the cardiac event?" | Condition.onsetDateTime, Encounter.period |
| Sequential | "What medications were added after the MI diagnosis?" | Condition.recordedDate, MedicationRequest.authoredOn |
| Outcome | "Did symptoms resolve after the procedure?" | Procedure.performedDateTime, Observation values |

#### 3.2.3 What Multi-Hop Training Teaches

The RL approach teaches MedGemma to:

1. **Parse FHIR Timestamps**: Understand `onsetDateTime`, `authoredOn`, `effectiveDateTime`
2. **Build Timeline Graphs**: Connect events chronologically
3. **Answer Complex Queries**: "What happened after X but before Y?"
4. **Identify Causal Patterns**: Medication → Lab change → Outcome
5. **Detect Discordance**: Expected vs. actual clinical progression

### 3.3 3D Volume Processing: Video Adapter

MedGemma 1.5 introduces a video adapter that can process 3D medical volumes.

#### 3.3.1 Video Adapter Architecture

```
Traditional Approach (Current):
┌──────────────────────────────────────────────────────────┐
│  CT Volume → [Slice 1] [Slice 2] ... [Slice 85] → Model  │
│                                                          │
│  Each slice processed independently                      │
│  No spatial continuity information                       │
│  Model sees 85 unrelated 2D images                       │
└──────────────────────────────────────────────────────────┘

Video Adapter Approach (MedGemma 1.5):
┌──────────────────────────────────────────────────────────┐
│  CT Volume → [Frame 1→2→3→...→85] → Temporal Encoder     │
│                      │                                    │
│                      ▼                                    │
│         Spatial-Temporal Features                        │
│                      │                                    │
│                      ▼                                    │
│  Model receives 3D context:                              │
│  - Inter-slice relationships                             │
│  - Lesion continuity across slices                       │
│  - Volumetric extent of findings                         │
└──────────────────────────────────────────────────────────┘
```

#### 3.3.2 Advantages for CT Analysis

| Capability | 2D Slice Approach | 3D Video Adapter |
|------------|-------------------|------------------|
| Lesion Size | Per-slice estimate | True 3D volume |
| Lesion Tracking | Manual across slices | Automatic |
| Anatomical Context | Limited | Full |
| PE Detection | Slice-by-slice | Vessel following |
| Nodule Characterization | 2D features | 3D morphology |

### 3.4 EHR Understanding Capabilities

The 27B model's training enables specific EHR reasoning patterns:

#### 3.4.1 Lab Value Interpretation

```
Given: "Troponin: 0.15 ng/mL, Creatinine: 2.1 mg/dL"

Model can reason:
- Elevated troponin suggests myocardial injury
- Elevated creatinine suggests renal dysfunction
- Combined: Possible cardiorenal syndrome
- Priority: Should be HIGH given combined markers
```

#### 3.4.2 Medication-Finding Correlation

```
Given:
- Medications: Warfarin (anticoagulant)
- CT Finding: Hyperdense mass in liver

Model can reason:
- Patient is anticoagulated
- Hyperdense mass could be hemorrhage
- Warfarin increases bleeding risk
- Priority: CRITICAL - possible hemorrhage in anticoagulated patient
```

#### 3.4.3 Disease Progression Tracking

```
Given timeline:
2023-01: Lung nodule 8mm (stable)
2023-06: Lung nodule 8mm (stable)
2024-01: Lung nodule 12mm (GROWING!)

Model can identify:
- Nodule was stable for 6 months
- Recent growth of 50%
- Concerning for malignancy
- Priority: HIGH regardless of current CT
```

---

## 4. Gap Analysis

### 4.1 FHIR Utilization Gaps

#### 4.1.1 Temporal Data Gap

| Data Type | Available in FHIR | Currently Extracted | Gap |
|-----------|-------------------|---------------------|-----|
| Condition onset dates | `onsetDateTime` | NO | **CRITICAL** |
| Condition resolution | `abatementDateTime` | NO | **HIGH** |
| Medication start dates | `authoredOn` | NO | **CRITICAL** |
| Medication status | `status` | NO | **HIGH** |
| Observation timestamps | `effectiveDateTime` | NO | **CRITICAL** |
| Encounter periods | `period.start/end` | NO | **HIGH** |
| Procedure dates | `performedDateTime` | NO | **MEDIUM** |

#### 4.1.2 Observation Data Gap

Our FHIR bundles contain 259 Observations per patient:

```
From train_1_a_1/fhir.json:

Observation: Hemoglobin A1c = 5.88 %          @ 2016-04-18
Observation: Body Height = 171.6 cm           @ 2016-04-18
Observation: Pain severity = 2 {score}        @ 2016-04-18
Observation: Body Weight = 82.3 kg            @ 2016-04-18
Observation: Body mass index = 27.95 kg/m2    @ 2016-04-18
Observation: Heart rate = 74 /min             @ 2016-04-18
Observation: Respiratory rate = 12 /min       @ 2016-04-18
Observation: Glucose = 72.87 mg/dL            @ 2016-04-18
Observation: Urea nitrogen = 8.61 mg/dL       @ 2016-04-18
... (249 more observations)
```

**Currently extracted: 0**

**Clinically relevant observations we're ignoring:**

| Category | Observations | Clinical Significance |
|----------|-------------|----------------------|
| Cardiac Markers | Troponin, BNP, CK-MB | Acute MI, heart failure |
| Coagulation | D-dimer, INR, PT | PE risk, bleeding risk |
| Renal Function | Creatinine, BUN, GFR | Contrast safety, prognosis |
| Metabolic | HbA1c, Glucose | Diabetes control |
| Inflammatory | CRP, ESR, WBC | Infection, inflammation |
| Vital Signs | BP, HR, SpO2, Temp | Hemodynamic status |

#### 4.1.3 Encounter Context Gap

```
From FHIR bundle:

Encounter: Emergency treatment         @ 1990-04-01T13:21:12
Encounter: General examination         @ 1982-02-15T13:21:12
Encounter: Admission to surgical dept  @ 1980-06-09T14:20:51
... (70 encounters total)
```

**What we're missing:**

- Why was the CT ordered? (encounter reason)
- Recent ED visits suggesting acute presentation?
- Pattern of visits (increasing frequency?)
- Previous imaging encounters

#### 4.1.4 Summary Table: FHIR Extraction Gaps

| Resource Type | Count | Currently Used | Fields Extracted | Fields Ignored |
|---------------|-------|----------------|------------------|----------------|
| Patient | 1 | ✓ | age, gender | extensions, address, contact |
| Condition | 50 | Partial | code.text only | onset, abatement, status, severity |
| MedicationRequest | 10 | Partial | medication.text only | authoredOn, status, dosage |
| Observation | 259 | **NO** | None | All values, timestamps |
| Encounter | 70 | **NO** | None | All periods, reasons |
| Procedure | 153 | **NO** | None | All data |
| DiagnosticReport | 112 | Partial | findings text | All structured data |
| ImagingStudy | 1 | **NO** | None | All data |

### 4.2 CT Processing Gaps

#### 4.2.1 Spatial Continuity Gap

```
Current Processing:

Volume: [Z=0] [Z=1] [Z=2] ... [Z=299]
           ↓     ↓     ↓          ↓
Sample:  [i=0] [i=1] [i=2] ... [i=84]
           ↓     ↓     ↓          ↓
Model:   img₀  img₁  img₂  ...  img₈₄  (INDEPENDENT)

Issues:
- Lesion spanning Z=45-48 appears as 2 separate findings
- Vessel continuity lost (critical for PE detection)
- 3D shape information discarded
```

#### 4.2.2 Window Level Gap

| Window Type | Center (HU) | Width (HU) | Use Case | Status |
|-------------|-------------|------------|----------|--------|
| Soft Tissue | 40 | 400 | General | **CURRENT** |
| Lung | -600 | 1500 | Pulmonary nodules, PE | MISSING |
| Bone | 400 | 1800 | Fractures, lesions | MISSING |
| Liver | 60 | 150 | Hepatic lesions | MISSING |
| PE Protocol | 100 | 700 | Pulmonary embolism | MISSING |

#### 4.2.3 Slice Sampling Gap

```
Current: Uniform sampling regardless of region

Volume Height: 300 slices
Thorax: slices 50-150 (100 slices)
Abdomen: slices 150-250 (100 slices)

With uniform 85-slice sampling:
- ~28 slices for thorax
- ~28 slices for abdomen
- May miss small findings
```

### 4.3 Model Variant Gap

#### 4.3.1 Capability Comparison

| Capability | MedGemma 4B-IT (Current) | MedGemma 27B | Gap Impact |
|------------|--------------------------|--------------|------------|
| Multi-hop reasoning | Basic | Advanced | HIGH |
| FHIR temporal queries | Not trained | Trained | CRITICAL |
| Lab value interpretation | Limited | Strong | HIGH |
| Disease progression | Limited | Strong | HIGH |
| 3D volume (video) | No | MedGemma 1.5 only | MEDIUM |

#### 4.3.2 Hardware Requirements

| Model | VRAM | GPU Options | Inference Time |
|-------|------|-------------|----------------|
| 4B-IT | ~12 GB | RTX 4090, A100 | Fast |
| 27B | ~65 GB | A100 80GB, H100 | Moderate |
| 27B (quantized) | ~35 GB | 2x RTX 4090 | Moderate |

### 4.4 Gap Severity Matrix

```
                        Impact on Clinical Accuracy
                        Low         Medium        High
                    ┌───────────┬─────────────┬───────────┐
                    │           │             │           │
         Easy       │           │ Multi-window│ Extract   │
                    │           │ CT display  │ Obs values│
                    │           │             │           │
Effort  ├───────────┼───────────┼─────────────┼───────────┤
to Fix              │           │             │           │
         Medium     │           │ Encounter   │ Temporal  │
                    │           │ context     │ FHIR data │
                    │           │             │           │
        ├───────────┼───────────┼─────────────┼───────────┤
                    │           │             │           │
         Hard       │           │ Video       │ Switch to │
                    │           │ adapter 3D  │ 27B model │
                    │           │             │           │
                    └───────────┴─────────────┴───────────┘
```

---

## 5. What FHIR Data We're Discarding

### 5.1 Complete Resource Inventory

From our sample FHIR bundle (`train_1_a_1/fhir.json`):

| Resource Type | Count | Data Available | Currently Used |
|---------------|-------|----------------|----------------|
| Patient | 1 | Full demographics, extensions, contact | age, gender only |
| Condition | 50 | Code, onset, abatement, status, severity | code.text only |
| MedicationRequest | 10 | Medication, dates, dosage, status | medication.text only |
| Observation | 259 | Code, value, units, timestamp, reference ranges | **NONE** |
| Procedure | 153 | Code, date, outcome, body site | **NONE** |
| Encounter | 70 | Type, period, reason, participants | **NONE** |
| DiagnosticReport | 112 | Code, date, results, conclusion | conclusion only |
| Claim | 80 | Codes, dates, amounts | **NONE** |
| ImagingStudy | 1 | Series, instances, modality | **NONE** |
| CareTeam | 4 | Participants, roles | **NONE** |
| CarePlan | 4 | Activities, goals | **NONE** |
| Immunization | 13 | Vaccine, date, status | **NONE** |
| Device | 2 | Type, status | **NONE** |
| SupplyDelivery | 16 | Item, quantity, dates | **NONE** |
| DocumentReference | 70 | Document type, content | **NONE** |
| Provenance | 1 | Activity, agents | **NONE** |
| ExplanationOfBenefit | 80 | Coverage, adjudication | **NONE** |

### 5.2 Detailed Analysis: Condition Resources

**What a FHIR Condition Contains:**

```json
{
  "resourceType": "Condition",
  "id": "4401b91a-2580-9d32-c405-2d8bf5beadef",
  "clinicalStatus": {
    "coding": [{"code": "active"}]       // ← WE IGNORE
  },
  "verificationStatus": {
    "coding": [{"code": "confirmed"}]    // ← WE IGNORE
  },
  "category": [
    {"coding": [{"display": "Encounter Diagnosis"}]}  // ← WE IGNORE
  ],
  "code": {
    "coding": [{
      "system": "http://snomed.info/sct",
      "code": "5251000175109",
      "display": "Prediabetes"            // ← WE EXTRACT THIS
    }],
    "text": "Prediabetes"                 // ← WE EXTRACT THIS
  },
  "subject": {"reference": "..."},
  "encounter": {"reference": "..."},      // ← WE IGNORE (link to encounter)
  "onsetDateTime": "1982-02-15T13:21:12+00:00",  // ← WE IGNORE (CRITICAL!)
  "recordedDate": "1982-02-15T13:21:12+00:00"    // ← WE IGNORE
}
```

**What we extract:** `"Prediabetes"`
**What we discard:** When it started, if it's active, verification status, related encounter

### 5.3 Detailed Analysis: Observation Resources

**What a FHIR Observation Contains:**

```json
{
  "resourceType": "Observation",
  "status": "final",
  "category": [{"coding": [{"display": "vital-signs"}]}],
  "code": {
    "coding": [{
      "display": "Hemoglobin A1c/Hemoglobin.total in Blood"
    }]
  },
  "subject": {"reference": "..."},
  "effectiveDateTime": "2016-04-18T13:21:12+00:00",
  "valueQuantity": {
    "value": 5.88,
    "unit": "%"
  },
  "referenceRange": [{
    "low": {"value": 4.0, "unit": "%"},
    "high": {"value": 5.6, "unit": "%"}
  }]
}
```

**What we extract:** NOTHING
**What we discard:** Test type, value, units, timestamp, reference ranges

### 5.4 Clinically Critical Observations Available But Unused

```
Category: Cardiac Markers
├── Troponin I/T
├── BNP/NT-proBNP
├── CK-MB
└── [Would indicate acute cardiac event]

Category: Coagulation
├── D-dimer
├── PT/INR
├── aPTT
└── [Would indicate PE risk, bleeding risk]

Category: Inflammatory
├── WBC count
├── CRP
├── ESR
├── Procalcitonin
└── [Would indicate infection/sepsis]

Category: Metabolic
├── HbA1c
├── Glucose
├── Creatinine
├── BUN
├── GFR
└── [Would indicate diabetes control, renal function]

Category: Vital Signs
├── Blood Pressure
├── Heart Rate
├── Respiratory Rate
├── SpO2
├── Temperature
└── [Would indicate hemodynamic status]
```

### 5.5 Impact Examples

#### Example 1: Missing Lab Context

**Scenario:** Patient with chest pain, elevated troponin

**Current extraction:**
```
## EHR Clinical Context
**Demographics:** 55 year old male
**Medical History:** Hypertension, Type 2 Diabetes
**Current Medications:** Metformin, Lisinopril
```

**What's missing (available in FHIR):**
```
**Recent Labs (from last 24 hours):**
- Troponin I: 0.85 ng/mL (CRITICAL - elevated)
- D-dimer: 1.2 μg/mL (elevated)
- Creatinine: 1.8 mg/dL (elevated)
- Glucose: 245 mg/dL (elevated)
```

**Clinical Impact:** Without troponin values, model cannot prioritize potential acute MI.

#### Example 2: Missing Temporal Context

**Scenario:** Worsening chronic condition

**Current extraction:**
```
**Medical History:** COPD, Lung nodule
```

**What's missing:**
```
**Timeline:**
- 2022-06: Lung nodule first detected (6mm)
- 2023-01: Lung nodule stable (6mm)
- 2023-06: Lung nodule stable (7mm)
- 2024-01: Lung nodule GREW (12mm) ← CRITICAL CHANGE
```

**Clinical Impact:** Cannot identify concerning growth pattern.

#### Example 3: Missing Medication Timeline

**Scenario:** Patient on anticoagulation with new finding

**Current extraction:**
```
**Current Medications:** Warfarin
```

**What's missing:**
```
**Medication Timeline:**
- Warfarin started 2023-01-15
- Last INR: 3.8 (supra-therapeutic) on 2024-01-10
- Patient has been on anticoagulation for 1 year
```

**Clinical Impact:** Cannot correlate hemorrhagic risk with imaging findings.

---

## 6. Recommendations

### 6.1 Priority Matrix

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PRIORITY MATRIX                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Priority 1 (Do Now)                    Priority 2 (Next Sprint)       │
│   ───────────────────                    ────────────────────────       │
│   • Extract temporal FHIR data           • Multi-window CT display      │
│   • Add Observation extraction           • Evaluate 27B model           │
│   • Preserve timestamps in context       • Add encounter context        │
│                                                                          │
│   Priority 3 (Future)                    Priority 4 (Research)          │
│   ───────────────────                    ─────────────────────          │
│   • Switch to MedGemma 27B               • Video adapter for 3D CT      │
│   • Multi-hop prompting                  • Multi-modal fusion           │
│   • Lab trend analysis                   • Longitudinal tracking        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Detailed Recommendations

#### 6.2.1 [P1] Enhanced FHIR Condition Extraction

**Current code (`fhir_context.py:105-135`):**
```python
def extract_conditions(fhir_bundle: Dict) -> List[str]:
    conditions = []
    for entry in entries:
        if resource.get("resourceType") == "Condition":
            text = code.get("text")
            if text:
                conditions.append(text)  # Only text!
    return conditions
```

**Recommended enhancement:**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class ConditionRecord:
    """Rich condition data preserving temporal information."""
    code: str
    display: str
    system: str  # SNOMED, ICD-10, etc.
    status: str  # active, resolved, remission
    onset_date: Optional[datetime]
    abatement_date: Optional[datetime]
    severity: Optional[str]

def extract_conditions_rich(fhir_bundle: Dict) -> List[ConditionRecord]:
    """Extract conditions with full temporal and status data."""
    conditions = []

    for entry in fhir_bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue

        # Extract code information
        code_obj = resource.get("code", {})
        coding = code_obj.get("coding", [{}])[0]

        # Parse clinical status
        status_coding = resource.get("clinicalStatus", {}).get("coding", [{}])[0]
        status = status_coding.get("code", "unknown")

        # Parse dates
        onset = None
        if "onsetDateTime" in resource:
            try:
                onset = datetime.fromisoformat(
                    resource["onsetDateTime"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        abatement = None
        if "abatementDateTime" in resource:
            try:
                abatement = datetime.fromisoformat(
                    resource["abatementDateTime"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        # Extract severity if present
        severity = None
        severity_obj = resource.get("severity", {})
        if severity_obj:
            severity = severity_obj.get("text") or \
                       severity_obj.get("coding", [{}])[0].get("display")

        conditions.append(ConditionRecord(
            code=coding.get("code", ""),
            display=coding.get("display", code_obj.get("text", "Unknown")),
            system=coding.get("system", ""),
            status=status,
            onset_date=onset,
            abatement_date=abatement,
            severity=severity
        ))

    # Sort by onset date (most recent first)
    conditions.sort(key=lambda c: c.onset_date or datetime.min, reverse=True)

    return conditions
```

#### 6.2.2 [P1] Add Observation Extraction

**New function to add:**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

@dataclass
class ObservationRecord:
    """Clinical observation with value and timing."""
    code: str
    display: str
    value: Optional[float]
    unit: str
    effective_date: Optional[datetime]
    reference_low: Optional[float]
    reference_high: Optional[float]
    interpretation: Optional[str]  # normal, high, low, critical

def extract_observations(fhir_bundle: Dict) -> List[ObservationRecord]:
    """Extract observation values with reference ranges."""
    observations = []

    for entry in fhir_bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue

        # Get code
        code_obj = resource.get("code", {})
        coding = code_obj.get("coding", [{}])[0]

        # Get value
        value_qty = resource.get("valueQuantity", {})
        value = value_qty.get("value")
        unit = value_qty.get("unit", "")

        # Get effective date
        effective = None
        if "effectiveDateTime" in resource:
            try:
                effective = datetime.fromisoformat(
                    resource["effectiveDateTime"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        # Get reference range
        ref_range = resource.get("referenceRange", [{}])[0]
        ref_low = ref_range.get("low", {}).get("value")
        ref_high = ref_range.get("high", {}).get("value")

        # Get interpretation
        interp_coding = resource.get("interpretation", [{}])[0].get("coding", [{}])[0]
        interpretation = interp_coding.get("code")

        observations.append(ObservationRecord(
            code=coding.get("code", ""),
            display=coding.get("display", code_obj.get("text", "Unknown")),
            value=value,
            unit=unit,
            effective_date=effective,
            reference_low=ref_low,
            reference_high=ref_high,
            interpretation=interpretation
        ))

    return observations

# Clinical filtering for relevant observations
CLINICALLY_RELEVANT_LOINC = {
    # Cardiac markers
    "10839-9": "Troponin I",
    "6598-7": "Troponin T",
    "30934-4": "BNP",
    "33762-6": "NT-proBNP",

    # Coagulation
    "48065-7": "D-dimer",
    "5902-2": "PT",
    "6301-6": "INR",

    # Inflammatory
    "6690-2": "WBC",
    "1988-5": "CRP",
    "30341-2": "ESR",

    # Metabolic
    "4548-4": "HbA1c",
    "2339-0": "Glucose",
    "2160-0": "Creatinine",
    "33914-3": "GFR",

    # Vitals
    "8480-6": "Systolic BP",
    "8462-4": "Diastolic BP",
    "8867-4": "Heart Rate",
    "9279-1": "Respiratory Rate",
    "2708-6": "SpO2",
}

def filter_clinically_relevant(
    observations: List[ObservationRecord]
) -> List[ObservationRecord]:
    """Filter to clinically relevant observations."""
    return [
        obs for obs in observations
        if obs.code in CLINICALLY_RELEVANT_LOINC
    ]
```

#### 6.2.3 [P1] Enhanced Context Formatting with Timeline

**Recommended enhanced formatter:**

```python
def format_context_with_timeline(
    context: PatientContext,
    conditions: List[ConditionRecord],
    observations: List[ObservationRecord],
    medications: List[MedicationRecord]
) -> str:
    """Format context with full temporal information for MedGemma."""

    lines = ["## Clinical Context"]

    # Demographics
    demo_parts = []
    if context.age:
        demo_parts.append(f"{context.age} year old")
    if context.gender:
        demo_parts.append(context.gender)
    if demo_parts:
        lines.append(f"**Patient:** {' '.join(demo_parts)}")

    # Active conditions with timeline
    lines.append("\n### Active Conditions")
    active = [c for c in conditions if c.status == "active"]
    if active:
        lines.append("| Onset | Condition | Duration |")
        lines.append("|-------|-----------|----------|")
        for c in active[:10]:  # Top 10
            onset_str = c.onset_date.strftime("%Y-%m") if c.onset_date else "Unknown"
            if c.onset_date:
                duration = (datetime.now() - c.onset_date).days // 365
                duration_str = f"{duration} years" if duration > 0 else "<1 year"
            else:
                duration_str = "Unknown"
            lines.append(f"| {onset_str} | {c.display} | {duration_str} |")
    else:
        lines.append("No active conditions documented.")

    # Recent observations (last 30 days)
    lines.append("\n### Recent Lab Values")
    recent_cutoff = datetime.now() - timedelta(days=30)
    recent_obs = [
        o for o in observations
        if o.effective_date and o.effective_date > recent_cutoff
    ]
    if recent_obs:
        lines.append("| Date | Test | Value | Reference | Status |")
        lines.append("|------|------|-------|-----------|--------|")
        for o in recent_obs[:15]:  # Top 15
            date_str = o.effective_date.strftime("%Y-%m-%d")
            value_str = f"{o.value:.2f} {o.unit}" if o.value else "N/A"

            ref_str = ""
            if o.reference_low and o.reference_high:
                ref_str = f"{o.reference_low}-{o.reference_high}"

            # Determine if abnormal
            status = "Normal"
            if o.value and o.reference_high and o.value > o.reference_high:
                status = "**HIGH**"
            elif o.value and o.reference_low and o.value < o.reference_low:
                status = "**LOW**"

            lines.append(f"| {date_str} | {o.display} | {value_str} | {ref_str} | {status} |")
    else:
        lines.append("No recent lab values available.")

    # Current medications with start dates
    lines.append("\n### Current Medications")
    active_meds = [m for m in medications if m.status == "active"]
    if active_meds:
        lines.append("| Started | Medication | Dosage |")
        lines.append("|---------|------------|--------|")
        for m in active_meds:
            start_str = m.authored_on.strftime("%Y-%m") if m.authored_on else "Unknown"
            lines.append(f"| {start_str} | {m.display} | {m.dosage or 'N/A'} |")
    else:
        lines.append("No active medications documented.")

    # Report content
    if context.findings:
        lines.append(f"\n### Current Radiology Findings\n{context.findings}")

    if context.impressions:
        lines.append(f"\n### Radiologist Impression\n{context.impressions}")

    return "\n".join(lines)
```

#### 6.2.4 [P2] Multi-Window CT Display

**Enhanced CT processor with multiple windows:**

```python
# Add to config.py
CT_WINDOWS = {
    "soft_tissue": {"center": 40, "width": 400},
    "lung": {"center": -600, "width": 1500},
    "bone": {"center": 400, "width": 1800},
    "liver": {"center": 60, "width": 150},
    "pe_protocol": {"center": 100, "width": 700},
}

# Enhanced slice extraction
def extract_multiwindow_slice(
    volume: np.ndarray,
    slice_idx: int,
    windows: List[str] = ["soft_tissue", "lung", "bone"]
) -> Image.Image:
    """Extract slice with multiple window presets as RGB channels."""

    slice_data = volume[:, :, slice_idx]
    slice_data = np.rot90(slice_data)

    if len(windows) == 3:
        # Use each window for a color channel
        channels = []
        for window_name in windows:
            window = CT_WINDOWS[window_name]
            windowed = apply_window_2d(slice_data, window["center"], window["width"])
            channels.append(windowed)

        rgb_slice = np.stack(channels, axis=-1)
        return Image.fromarray(rgb_slice, mode="RGB")
    else:
        # Standard grayscale to RGB
        windowed = apply_window_2d(
            slice_data,
            CT_WINDOWS["soft_tissue"]["center"],
            CT_WINDOWS["soft_tissue"]["width"]
        )
        rgb_slice = np.stack([windowed] * 3, axis=-1)
        return Image.fromarray(rgb_slice, mode="RGB")
```

#### 6.2.5 [P2] Evaluate MedGemma 27B

**Evaluation script:**

```python
"""Evaluate MedGemma 27B vs 4B on FHIR-aware tasks."""

import json
from pathlib import Path

def create_evaluation_cases():
    """Create test cases requiring multi-hop reasoning."""
    return [
        {
            "id": "multi_hop_1",
            "fhir_context": """
            Conditions:
            - Diabetes (onset: 2020-01-15)
            - Hypertension (onset: 2018-06-01)

            Medications:
            - Metformin (started: 2020-02-01)

            Observations:
            - HbA1c: 8.5% (2020-01-15)
            - HbA1c: 7.2% (2020-06-15)
            - HbA1c: 6.8% (2020-12-15)
            """,
            "question": "Did the patient's diabetes control improve after starting Metformin?",
            "expected_reasoning": "HbA1c decreased from 8.5% to 6.8% after Metformin started",
            "expected_answer": "Yes, HbA1c improved from 8.5% to 6.8%"
        },
        {
            "id": "multi_hop_2",
            "fhir_context": """
            Conditions:
            - Lung nodule (onset: 2022-06-01, note: 8mm)
            - Lung nodule follow-up (onset: 2023-06-01, note: 8mm stable)
            - Lung nodule growth (onset: 2024-01-01, note: 15mm, GROWING)

            Medications:
            - None
            """,
            "question": "What is the concerning trend in the lung nodule?",
            "expected_reasoning": "Stable for 1 year, then rapid growth",
            "expected_answer": "Nodule was stable at 8mm for 1 year, then grew to 15mm"
        },
        {
            "id": "temporal_correlation",
            "fhir_context": """
            Encounter: Emergency visit (2024-01-15)
            Chief Complaint: Chest pain

            Observations (2024-01-15):
            - Troponin I: 0.45 ng/mL (HIGH, ref <0.04)
            - D-dimer: 2.1 ug/mL (HIGH, ref <0.5)

            Conditions:
            - History of DVT (2023-01-01)
            """,
            "question": "What acute conditions should be prioritized based on labs and history?",
            "expected_reasoning": "Elevated troponin suggests MI, elevated D-dimer + DVT history suggests PE",
            "expected_answer": "Both acute MI and PE should be considered given lab values and history"
        }
    ]

def run_comparison(model_4b, model_27b, cases):
    """Run comparison between models."""
    results = []

    for case in cases:
        # Test 4B
        response_4b = model_4b.generate(case["fhir_context"], case["question"])
        score_4b = evaluate_response(response_4b, case["expected_reasoning"])

        # Test 27B
        response_27b = model_27b.generate(case["fhir_context"], case["question"])
        score_27b = evaluate_response(response_27b, case["expected_reasoning"])

        results.append({
            "case_id": case["id"],
            "score_4b": score_4b,
            "score_27b": score_27b,
            "delta": score_27b - score_4b
        })

    return results
```

#### 6.2.6 [P3] Multi-Hop Prompting Strategy

**Enhanced prompt template:**

```python
ENHANCED_SYSTEM_PROMPT = """You are an expert radiologist AI assistant with advanced
EHR reasoning capabilities. You analyze CT scans alongside comprehensive clinical
timelines to provide accurate triage prioritization.

## Your Capabilities

1. **Temporal Reasoning**: You can correlate events across time
   - "After starting medication X, did condition Y improve?"
   - "What changed between the last scan and this one?"

2. **Lab Correlation**: You integrate lab values with imaging
   - Elevated D-dimer + PE finding = increased confidence
   - Elevated troponin + cardiac finding = acute event

3. **Disease Progression**: You track conditions over time
   - Growing nodules vs stable nodules
   - Worsening vs improving disease

## Required Reasoning Steps

Before providing your final assessment, explicitly reason through:

1. **Timeline Analysis**
   - What are the key temporal landmarks?
   - What changed recently?
   - Any concerning trends?

2. **Lab Integration**
   - Are there relevant lab abnormalities?
   - Do labs correlate with imaging findings?
   - Any discordance between labs and imaging?

3. **Risk Factor Correlation**
   - How do conditions affect imaging interpretation?
   - Does medication history change risk assessment?
   - Are there high-risk combinations?

[Rest of prompt with output format...]
"""
```

### 6.3 Implementation Roadmap

```
Week 1-2: Priority 1 - FHIR Temporal Data
├── Day 1-2: Implement ConditionRecord with temporal fields
├── Day 3-4: Implement ObservationRecord extraction
├── Day 5-6: Implement MedicationRecord with dates
├── Day 7-8: Update PatientContext data model
├── Day 9-10: Update format_context_for_prompt with timeline
└── Day 11-14: Testing and validation

Week 3-4: Priority 2 - CT and Model Evaluation
├── Day 15-17: Implement multi-window CT extraction
├── Day 18-20: Set up MedGemma 27B evaluation environment
├── Day 21-23: Run comparative benchmarks
├── Day 24-26: Add encounter context extraction
└── Day 27-28: Documentation and review

Month 2: Priority 3 - Advanced Features
├── Week 1: Migrate to 27B if evaluation positive
├── Week 2: Implement multi-hop prompting
├── Week 3: Add lab trend analysis
└── Week 4: Integration testing

Month 3+: Research Track
├── Evaluate MedGemma 1.5 video adapter
├── Prototype 3D volume processing
└── Longitudinal tracking system
```

### 6.4 Risk Assessment

| Change | Risk Level | Mitigation |
|--------|-----------|------------|
| FHIR extraction changes | LOW | Additive changes, no breaking API |
| Multi-window CT | LOW | New function, existing works |
| 27B model switch | MEDIUM | Higher VRAM, need GPU upgrade |
| Video adapter | HIGH | New architecture, research phase |

---

## 7. Appendices

### Appendix A: Full FHIR Resource Inventory

```
Bundle Type: transaction
Total Entries: 926

Resource Distribution:
─────────────────────────────────
Observation          259  (28.0%)
Procedure            153  (16.5%)
DiagnosticReport     112  (12.1%)
Claim                 80   (8.6%)
ExplanationOfBenefit  80   (8.6%)
Encounter             70   (7.6%)
DocumentReference     70   (7.6%)
Condition             50   (5.4%)
SupplyDelivery        16   (1.7%)
Immunization          13   (1.4%)
MedicationRequest     10   (1.1%)
CareTeam               4   (0.4%)
CarePlan               4   (0.4%)
Device                 2   (0.2%)
Patient                1   (0.1%)
Provenance             1   (0.1%)
ImagingStudy           1   (0.1%)
─────────────────────────────────
```

### Appendix B: Multi-Hop Reasoning Scenarios for CT Triage

| Scenario | Multi-Hop Query | Required FHIR Data |
|----------|-----------------|-------------------|
| Cancer progression | "Given history of lung cancer (2022), is this new nodule concerning?" | Condition.onsetDateTime, previous imaging |
| Anticoagulation risk | "Patient on warfarin with hyperdense mass - hemorrhage risk?" | MedicationRequest.status, Observation.INR |
| Post-procedure | "Any complications after recent CABG?" | Procedure.performedDateTime, current imaging |
| Treatment response | "Did pneumonia improve after antibiotics?" | MedicationRequest.authoredOn, sequential imaging |
| Metabolic correlation | "Diabetic with renal lesion - check creatinine trend" | Condition.code, Observation values over time |

### Appendix C: Critical Files Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `triage/fhir_context.py` | FHIR parsing | `extract_conditions`, `parse_fhir_context`, `format_context_for_prompt` |
| `triage/ct_processor.py` | CT processing | `process_ct_volume`, `apply_window`, `sample_slices` |
| `triage/medgemma_analyzer.py` | Model interface | `_build_messages`, `analyze`, `_parse_response` |
| `triage/prompts.py` | Prompt templates | `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE` |
| `triage/config.py` | Configuration | `MEDGEMMA_MODEL_ID`, `CT_*` constants |

### Appendix D: MedGemma Model Specifications

| Specification | MedGemma 4B-IT | MedGemma 27B | MedGemma 1.5 |
|---------------|----------------|--------------|--------------|
| Base Model | Gemma 2 4B | Gemma 2 27B | Gemini 1.5 |
| Parameters | 4B | 27B | Varies |
| Context Length | 8K | 8K | 1M+ |
| Vision Support | Yes (images) | Yes (images) | Yes (video/3D) |
| FHIR Training | Limited | **Synthea RL** | Yes |
| Multi-hop | Basic | **Advanced** | Advanced |
| VRAM (fp16) | ~12 GB | ~65 GB | Varies |
| VRAM (int8) | ~6 GB | ~35 GB | Varies |
| Inference | Fast | Moderate | Variable |

### Appendix E: LOINC Codes for Clinical Observations

```python
# Essential LOINC codes for CT triage correlation

CARDIAC_MARKERS = {
    "10839-9": "Troponin I.cardiac",
    "6598-7": "Troponin T.cardiac",
    "30934-4": "BNP",
    "33762-6": "NT-proBNP",
    "2157-6": "CK-MB",
}

COAGULATION = {
    "48065-7": "Fibrin D-dimer",
    "5902-2": "Prothrombin time",
    "6301-6": "INR",
    "3173-2": "aPTT",
}

INFLAMMATORY = {
    "6690-2": "WBC count",
    "1988-5": "C-reactive protein",
    "30341-2": "ESR",
    "33959-8": "Procalcitonin",
}

RENAL = {
    "2160-0": "Creatinine",
    "3094-0": "BUN",
    "33914-3": "eGFR",
}

METABOLIC = {
    "4548-4": "Hemoglobin A1c",
    "2339-0": "Glucose",
    "2093-3": "Total cholesterol",
}

VITALS = {
    "8480-6": "Systolic BP",
    "8462-4": "Diastolic BP",
    "8867-4": "Heart rate",
    "9279-1": "Respiratory rate",
    "2708-6": "Oxygen saturation",
    "8310-5": "Body temperature",
}
```

---

## Document Metadata

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Author | Sentinel-X Development Team |
| Last Updated | January 2026 |
| Status | Complete |
| Review Status | Pending Clinical Review |

---

*This document was generated as part of the Sentinel-X AI-powered CT triage system development. For questions or updates, refer to the project repository.*
