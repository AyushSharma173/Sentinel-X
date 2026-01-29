# Sentinel-X AI-Powered CT Triage System: Technical Deep Dive

## Table of Contents

1. [Executive Summary & Value Proposition](#1-executive-summary--value-proposition)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [MedGemma Model Integration](#3-medgemma-model-integration)
4. [3D CT Scan Processing Pipeline](#4-3d-ct-scan-processing-pipeline)
5. [FHIR Clinical Context Integration](#5-fhir-clinical-context-integration)
6. [Prompt Engineering & Analysis Strategy](#6-prompt-engineering--analysis-strategy)
7. [Output Processing & Prioritization Logic](#7-output-processing--prioritization-logic)
8. [Medical Terminology Reference](#8-medical-terminology-reference)
9. [Technical Implementation Details](#9-technical-implementation-details)
10. [Appendices](#10-appendices)

---

## 1. Executive Summary & Value Proposition

### 1.1 The Problem

Radiology departments face a critical bottleneck: **all CT scans enter the same queue**, regardless of clinical urgency. A routine follow-up scan receives the same initial priority as an emergency case with suspected pulmonary embolism. This "first-in, first-out" approach can delay detection of life-threatening conditions.

**Key statistics driving this problem:**
- Emergency departments generate CT scans 24/7
- Radiologists cannot instantly assess every incoming scan
- Critical findings (PE, aortic dissection) require minutes-to-hours response times
- Delayed diagnosis of acute conditions directly impacts patient outcomes

### 1.2 The Solution: Intelligent Pre-Triage

Sentinel-X implements **AI-powered intelligent triage** that:

1. **Monitors** incoming CT scans in real-time
2. **Analyzes** each scan using Google's MedGemma multimodal AI model
3. **Integrates** clinical context from the Electronic Health Record (EHR) via FHIR
4. **Prioritizes** cases into three tiers: CRITICAL, HIGH RISK, and ROUTINE
5. **Presents** a dynamically sorted worklist to radiologists

### 1.3 Value Proposition

| Benefit | Description |
|---------|-------------|
| **Faster Critical Detection** | Life-threatening conditions are surfaced immediately |
| **Context-Aware Prioritization** | A nodule in a cancer patient gets higher priority than the same finding in a healthy patient |
| **Radiologist Efficiency** | Focus human expertise where it matters most |
| **Seamless Integration** | Works with existing PACS/RIS workflows via FHIR standards |
| **Explainable AI** | Every priority decision includes rationale and key diagnostic slice |

### 1.4 Target Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL WORKFLOW                              │
│  CT Scan → FIFO Queue → Wait → Radiologist Review → Report          │
│                          ↑                                           │
│                     Bottleneck                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                  SENTINEL-X WORKFLOW                                 │
│  CT Scan → AI Triage → Priority-Sorted Worklist → Radiologist       │
│               │                                                      │
│               ├── CRITICAL → Immediate Review                       │
│               ├── HIGH RISK → Prompt Review                         │
│               └── ROUTINE → Standard Queue                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                     │
│  ┌────────────────────────┐    ┌────────────────────────┐                │
│  │   CT Volume (NIfTI)    │    │   Clinical Report      │                │
│  │   - .nii.gz format     │    │   - FHIR Bundle        │                │
│  │   - 512x512xN voxels   │    │   - JSON format        │                │
│  │   - Hounsfield Units   │    │   - Demographics       │                │
│  │                        │    │   - Conditions         │                │
│  │                        │    │   - Medications        │                │
│  │                        │    │   - Radiology Report   │                │
│  └───────────┬────────────┘    └───────────┬────────────┘                │
└──────────────┼──────────────────────────────┼────────────────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        PROCESSING LAYER                                   │
│  ┌─────────────────────────┐    ┌─────────────────────────┐              │
│  │   CT Processor          │    │   FHIR Context Parser   │              │
│  │   - Load NIfTI          │    │   - Demographics        │              │
│  │   - HU Windowing        │    │   - Conditions          │              │
│  │   - Slice Sampling      │    │   - Risk Factors        │              │
│  │   - RGB Conversion      │    │   - Medications         │              │
│  │                         │    │   - Report Content      │              │
│  │   Output: 85 PIL Images │    │   Output: Text Context  │              │
│  └───────────┬─────────────┘    └───────────┬─────────────┘              │
└──────────────┼──────────────────────────────┼────────────────────────────┘
               │                              │
               └──────────────┬───────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        AI ANALYSIS LAYER                                  │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    MedGemma 4B Multimodal Model                    │   │
│  │   Input:  85 CT slice images + Clinical context text              │   │
│  │   Output: Structured analysis (findings, priority, rationale)     │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────┐              │
│  │   Triage Result         │    │   Worklist Manager      │              │
│  │   - Priority Level      │    │   - Priority Sorting    │              │
│  │   - Rationale           │    │   - Real-time Updates   │              │
│  │   - Key Slice           │    │   - WebSocket Broadcast │              │
│  │   - Thumbnail           │    │                         │              │
│  └─────────────────────────┘    └─────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Module Structure

| Module | File | Purpose |
|--------|------|---------|
| **Agent** | `triage/agent.py` | Main orchestrator coordinating the pipeline |
| **CT Processor** | `triage/ct_processor.py` | NIfTI loading, windowing, slice extraction |
| **FHIR Context** | `triage/fhir_context.py` | Clinical data extraction from FHIR |
| **MedGemma Analyzer** | `triage/medgemma_analyzer.py` | Model interface and response parsing |
| **Prompts** | `triage/prompts.py` | System and user prompt templates |
| **Output Generator** | `triage/output_generator.py` | Result formatting and thumbnail creation |
| **Worklist** | `triage/worklist.py` | Priority-sorted worklist management |
| **Config** | `triage/config.py` | Constants, paths, and configurations |

---

## 3. MedGemma Model Integration

### 3.1 Model Overview

**MedGemma 4B-IT** is Google's instruction-tuned multimodal model specifically designed for medical applications.

| Property | Value |
|----------|-------|
| **Model ID** | `google/medgemma-4b-it` |
| **Parameters** | 4 billion |
| **Architecture** | Decoder-only Transformer (Gemma 3 base) |
| **Image Encoder** | SigLIP (trained on de-identified medical data) |
| **Context Length** | 128K tokens |
| **Image Token Cost** | 256 tokens per image |
| **Precision** | BF16 (bfloat16) |

**Source:** [MedGemma on HuggingFace](https://huggingface.co/google/medgemma-4b-it), [MedGemma Technical Report](https://arxiv.org/abs/2507.05201)

### 3.2 Model Loading

The model is lazily loaded on first analysis to optimize resource usage:

```python
# File: triage/medgemma_analyzer.py:44-61
def load_model(self) -> None:
    self.processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID)
    self.model = AutoModelForImageTextToText.from_pretrained(
        MEDGEMMA_MODEL_ID,
        torch_dtype=torch.bfloat16,  # Half precision for efficiency
        device_map="auto",           # Automatic GPU/CPU mapping
    )
```

**Key configuration choices:**

1. **`torch_dtype=torch.bfloat16`**: Uses brain floating-point format for 50% memory reduction while maintaining numeric stability for medical inference.

2. **`device_map="auto"`**: Automatically distributes model across available GPUs or falls back to CPU, enabling deployment on various hardware configurations.

### 3.3 Input Construction

MedGemma uses a chat-based message format with interleaved images and text:

```python
# File: triage/medgemma_analyzer.py:63-93
def _build_messages(self, images: List[Image.Image], context_text: str) -> List[dict]:
    content = []

    # Add all 85 images as placeholders
    for _ in images:
        content.append({"type": "image"})

    # Add the user prompt with clinical context
    user_prompt = build_user_prompt(context_text, len(images))
    content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    return messages
```

**Message structure visualization:**

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM MESSAGE                                                   │
│ "You are an expert radiologist AI assistant performing          │
│  triage analysis of chest CT scans..."                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ USER MESSAGE                                                     │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐         ┌─────┐        │
│ │IMG 1│ │IMG 2│ │IMG 3│ │ ... │ │IMG85│  ...    │TEXT │        │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘         └─────┘        │
│                                                                  │
│ TEXT: "Analyze the following chest CT scan and clinical         │
│        information for triage prioritization.                   │
│                                                                  │
│        ## EHR Clinical Context                                  │
│        **Demographics:** 62 year old male                       │
│        **Medical History:** Diabetes, Hypertension, COPD        │
│        ..."                                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Inference Pipeline

```python
# File: triage/medgemma_analyzer.py:155-213
def analyze(self, images: List[Image.Image], context_text: str, max_new_tokens: int = 1024):
    # Step 1: Build message structure
    messages = self._build_messages(images, context_text)

    # Step 2: Apply chat template (converts to model-specific format)
    prompt = self.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Step 3: Tokenize text and encode images
    inputs = self.processor(
        text=prompt,
        images=images,
        return_tensors="pt",
    ).to(self.model.device, dtype=torch.bfloat16)

    # Step 4: Generate response (deterministic, no sampling)
    with torch.inference_mode():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for reproducibility
        )

    # Step 5: Decode and parse response
    response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return self._parse_response(response)
```

**Important inference parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_new_tokens` | 1024 | Sufficient for detailed analysis with all required fields |
| `do_sample` | False | Deterministic output for reproducible medical decisions |
| `dtype` | bfloat16 | Memory efficiency without precision loss for medical imaging |

### 3.5 Token Budget Analysis

With 85 CT slices, the token budget is significant:

```
Images:          85 slices × 256 tokens/image = 21,760 tokens
System Prompt:   ~500 tokens
User Prompt:     ~200-500 tokens (depends on clinical context)
─────────────────────────────────────────────────────────────
Total Input:     ~22,500-23,000 tokens
Output Budget:   1,024 tokens
Context Limit:   128,000 tokens ✓ (well within limit)
```

---

## 4. 3D CT Scan Processing Pipeline

### 4.1 Input Format: NIfTI Files

CT scans are stored in **NIfTI (Neuroimaging Informatics Technology Initiative)** format:

| Property | Description |
|----------|-------------|
| **File Extension** | `.nii` or `.nii.gz` (gzip compressed) |
| **Data Structure** | 3D voxel array + affine transformation matrix |
| **Typical Dimensions** | 512 × 512 × 200-400 voxels |
| **Voxel Values** | Hounsfield Units (HU) |

### 4.2 Hounsfield Units (HU) - Medical Background

**Hounsfield Units** are the standard measurement scale for CT scan density:

| Material | HU Value | Appearance |
|----------|----------|------------|
| Air | -1000 | Black |
| Lung | -500 | Dark gray |
| Fat | -100 to -50 | Gray |
| Water | 0 | Gray |
| Soft Tissue | +40 to +80 | Light gray |
| Bone | +400 to +1000 | White |
| Metal | +1000+ | Bright white |

### 4.3 Windowing Transformation

Raw HU values span -1000 to +3000+, but displays only show 256 gray levels. **Windowing** maps a specific HU range to this display range.

```python
# File: triage/ct_processor.py:16-33
def apply_window(data: np.ndarray, center: int, width: int) -> np.ndarray:
    """Apply HU windowing to CT data."""
    lower = center - width // 2  # 40 - 200 = -160 HU
    upper = center + width // 2  # 40 + 200 = +240 HU

    windowed = np.clip(data, lower, upper)
    normalized = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    return normalized
```

**Configuration (from `config.py:23-24`):**

```python
CT_WINDOW_CENTER = 40   # Soft tissue window center
CT_WINDOW_WIDTH = 400   # Soft tissue window width
```

**What this means:**

| Value | Calculation | Result |
|-------|-------------|--------|
| Lower bound | 40 - 400/2 | **-160 HU** |
| Upper bound | 40 + 400/2 | **+240 HU** |
| Visibility range | -160 to +240 HU | Optimized for soft tissue |

**Window choice rationale:** The soft tissue window (W:400, L:40) is ideal for detecting:
- Pulmonary nodules
- Pleural effusions
- Mediastinal masses
- Vascular abnormalities

### 4.4 Slice Sampling Strategy

A chest CT can have 300-500+ slices. MedGemma processes 85 representative slices:

```python
# File: triage/ct_processor.py:60-76
def sample_slices(volume: np.ndarray, num_slices: int = 85) -> List[int]:
    total_slices = volume.shape[2]  # Axial dimension

    if total_slices <= num_slices:
        return list(range(total_slices))

    # Uniform sampling across entire volume
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    return indices.tolist()
```

**Sampling visualization:**

```
Volume: 400 slices
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Slice 0    Slice 4    Slice 9   ...   Slice 395   Slice 399 │
│    │          │          │                │          │       │
│    ▼          ▼          ▼                ▼          ▼       │
│  Sample    Sample     Sample   ...     Sample     Sample     │
│    1          2          3        85       84        85       │
└─────────────────────────────────────────────────────────────┘

Sampling interval: 400 / 85 ≈ every 4.7 slices
```

### 4.5 Image Extraction and Conversion

```python
# File: triage/ct_processor.py:79-97
def extract_slice_as_image(volume: np.ndarray, slice_idx: int) -> Image.Image:
    # Extract axial slice (Z-axis cross-section)
    slice_data = volume[:, :, slice_idx]

    # Rotate 90° for proper anatomical orientation
    slice_data = np.rot90(slice_data)

    # Convert grayscale to RGB (MedGemma expects RGB)
    rgb_slice = np.stack([slice_data] * 3, axis=-1)

    return Image.fromarray(rgb_slice.astype(np.uint8), mode="RGB")
```

**Why RGB conversion?** MedGemma's SigLIP image encoder expects 3-channel RGB input. Grayscale CT slices are duplicated across R, G, B channels.

### 4.6 Complete Processing Pipeline

```python
# File: triage/ct_processor.py:100-126
def process_ct_volume(path: Path) -> Tuple[List[Image.Image], List[int], dict]:
    # Step 1: Load NIfTI volume
    volume, metadata = load_nifti_volume(path)

    # Step 2: Apply soft tissue windowing
    windowed = apply_window(volume, CT_WINDOW_CENTER, CT_WINDOW_WIDTH)

    # Step 3: Sample 85 slice indices
    slice_indices = sample_slices(windowed, CT_NUM_SLICES)

    # Step 4: Extract as PIL images
    images = [extract_slice_as_image(windowed, idx) for idx in slice_indices]

    return images, slice_indices, metadata
```

---

## 5. FHIR Clinical Context Integration

### 5.1 What is FHIR?

**FHIR (Fast Healthcare Interoperability Resources)** is the modern standard for healthcare data exchange. Sentinel-X extracts clinical context from FHIR-formatted data to enhance triage decisions.

### 5.2 Extracted Data Types

| Resource Type | Extracted Fields | Clinical Relevance |
|---------------|------------------|-------------------|
| **Patient** | Age, Gender | Demographics affect risk profiles |
| **Condition** | Medical history | Cancer history, chronic diseases |
| **MedicationStatement** | Current medications | Immunosuppressants, anticoagulants |
| **DiagnosticReport** | Findings, Impressions | Prior radiology findings |

### 5.3 Data Extraction Functions

#### Demographics Extraction

```python
# File: triage/fhir_context.py:63-102
def extract_patient_demographics(fhir_bundle: Dict) -> tuple:
    for entry in fhir_bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            gender = resource.get("gender")
            birth_date = resource.get("birthDate")  # "1961-03-15"
            age = datetime.now().year - int(birth_date.split("-")[0])
            return age, gender
```

#### Conditions Extraction

```python
# File: triage/fhir_context.py:105-135
def extract_conditions(fhir_bundle: Dict) -> List[str]:
    conditions = []
    for entry in fhir_bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Condition":
            text = resource.get("code", {}).get("text")
            if not text:
                # Fallback to coding display
                text = resource["code"]["coding"][0]["display"]
            conditions.append(text)
    return conditions
```

#### Risk Factor Identification

```python
# File: triage/fhir_context.py:42-60
def identify_risk_factors(conditions: List[str]) -> List[str]:
    risk_factors = []
    for condition in conditions:
        for risk_keyword in HIGH_RISK_CONDITIONS:
            if risk_keyword in condition.lower():
                risk_factors.append(condition)
                break
    return risk_factors
```

**High-risk condition keywords (from `config.py:41-56`):**

```python
HIGH_RISK_CONDITIONS = {
    "cancer", "malignancy", "carcinoma", "tumor", "neoplasm",
    "diabetes", "diabetic",
    "copd", "pulmonary disease",
    "heart disease", "cardiac",
    "hypertension",
    "immunocompromised", "immunosuppressed",
}
```

### 5.4 Context Formatting for Prompt

```python
# File: triage/fhir_context.py:267-306
def format_context_for_prompt(context: PatientContext) -> str:
    lines = ["## EHR Clinical Context"]

    # Demographics
    if context.age and context.gender:
        lines.append(f"**Demographics:** {context.age} year old {context.gender}")

    # Medical History
    if context.conditions:
        lines.append(f"**Medical History:** {', '.join(context.conditions)}")

    # Risk Factors (highlighted for AI attention)
    if context.risk_factors:
        lines.append(f"**High-Risk Factors:** {', '.join(context.risk_factors)}")

    # Medications
    if context.medications:
        lines.append(f"**Current Medications:** {', '.join(context.medications)}")

    # Radiology Report Content
    if context.findings:
        lines.append(f"\n## Radiology Report Findings\n{context.findings}")
    if context.impressions:
        lines.append(f"\n## Radiology Report Impressions\n{context.impressions}")

    return "\n".join(lines)
```

**Example formatted output:**

```markdown
## EHR Clinical Context
**Demographics:** 62 year old male
**Medical History:** Diabetes mellitus type 2, Hypertension, COPD
**High-Risk Factors:** Diabetes mellitus type 2, COPD
**Current Medications:** Metformin, Lisinopril, Albuterol

## Radiology Report Findings
Multiple venous collaterals are present in the anterior left chest wall
and are associated with the anterior jugular vein at the level of the
right sternoclavicular junction...

## Radiology Report Impressions
Multiple venous collaterals in the anterior left chest wall and collapsed
appearance in the left subclavian vein (chronic occlusion?)...
```

---

## 6. Prompt Engineering & Analysis Strategy

### 6.1 System Prompt

The system prompt establishes MedGemma's role and defines the priority framework:

```
# File: triage/prompts.py:3-42

You are an expert radiologist AI assistant performing triage analysis of
chest CT scans. Your task is to analyze CT images alongside clinical context
to assign priority levels for radiologist review.

## Priority Level Definitions

**PRIORITY 1 - CRITICAL**: Acute, life-threatening pathology requiring immediate attention
- Pulmonary embolism (PE)
- Aortic dissection
- Tension pneumothorax
- Active hemorrhage
- Bowel perforation
- Acute aortic rupture

**PRIORITY 2 - HIGH RISK**: Significant findings requiring prompt review,
especially with high-risk clinical context
- Pulmonary nodules in patients with cancer history
- New masses or suspicious lesions
- Significant pleural effusions
- Findings discordant with clinical presentation
- Concerning findings in immunocompromised patients

**PRIORITY 3 - ROUTINE**: Non-urgent findings suitable for standard workflow
- Stable chronic findings
- Minor abnormalities without clinical significance
- Normal or near-normal examinations

## Output Format

You MUST provide your analysis in the following structured format:

VISUAL_FINDINGS: [Detailed description of all findings visible in the CT images]

KEY_SLICE: [Integer index 0-84 of the most diagnostically important slice]

PRIORITY_LEVEL: [1, 2, or 3]

PRIORITY_RATIONALE: [Explanation combining visual findings and clinical context]

FINDINGS_SUMMARY: [Brief 1-2 sentence summary suitable for worklist display]

CONDITIONS_CONSIDERED: [Comma-separated list of differential diagnoses considered]
```

### 6.2 User Prompt Template

```
# File: triage/prompts.py:44-55

Analyze the following chest CT scan and clinical information for triage
prioritization.

{context}

Please examine all {num_slices} CT slices provided and deliver your
structured analysis following the exact format specified. Consider both
the visual findings and the clinical context when determining priority level.

Remember:
- PRIORITY 1 is for acute, life-threatening conditions
- PRIORITY 2 is for significant findings especially with high-risk context
- PRIORITY 3 is for routine findings

Provide your analysis:
```

### 6.3 Priority Decision Matrix

The prompt engineering creates a clinical decision framework:

| Visual Finding | Clinical Context | Priority |
|---------------|------------------|----------|
| Pulmonary embolism | Any | **1 (CRITICAL)** |
| Aortic dissection | Any | **1 (CRITICAL)** |
| Tension pneumothorax | Any | **1 (CRITICAL)** |
| Pulmonary nodule | Cancer history | **2 (HIGH RISK)** |
| New mass | Any | **2 (HIGH RISK)** |
| Findings + immunocompromised | Immunocompromised | **2 (HIGH RISK)** |
| Stable chronic findings | Any | **3 (ROUTINE)** |
| Normal examination | Any | **3 (ROUTINE)** |

### 6.4 Structured Output Parsing

MedGemma's response is parsed using regex patterns:

```python
# File: triage/medgemma_analyzer.py:95-153
def _parse_response(self, response: str) -> AnalysisResult:
    # Parse VISUAL_FINDINGS
    match = re.search(r"VISUAL_FINDINGS:\s*(.+?)(?=\n\w+:|$)", response, re.DOTALL)
    visual_findings = match.group(1).strip() if match else ""

    # Parse KEY_SLICE
    match = re.search(r"KEY_SLICE:\s*(\d+)", response)
    key_slice_index = int(match.group(1)) if match else 0

    # Parse PRIORITY_LEVEL
    match = re.search(r"PRIORITY_LEVEL:\s*(\d+)", response)
    priority_level = int(match.group(1)) if match else 3

    # Parse PRIORITY_RATIONALE
    match = re.search(r"PRIORITY_RATIONALE:\s*(.+?)(?=\n\w+:|$)", response, re.DOTALL)
    priority_rationale = match.group(1).strip() if match else ""

    # Parse FINDINGS_SUMMARY
    match = re.search(r"FINDINGS_SUMMARY:\s*(.+?)(?=\n\w+:|$)", response, re.DOTALL)
    findings_summary = match.group(1).strip() if match else ""

    # Parse CONDITIONS_CONSIDERED
    match = re.search(r"CONDITIONS_CONSIDERED:\s*(.+?)(?=\n\w+:|$)", response, re.DOTALL)
    conditions = [c.strip() for c in match.group(1).split(",")] if match else []
```

---

## 7. Output Processing & Prioritization Logic

### 7.1 AnalysisResult Data Structure

```python
# File: triage/medgemma_analyzer.py:18-27
@dataclass
class AnalysisResult:
    visual_findings: str        # Detailed description of CT findings
    key_slice_index: int        # Index (0-84) of most diagnostic slice
    priority_level: int         # 1 (CRITICAL), 2 (HIGH RISK), 3 (ROUTINE)
    priority_rationale: str     # Explanation for priority assignment
    findings_summary: str       # 1-2 sentence summary for worklist
    conditions_considered: List[str]  # Differential diagnoses
    raw_response: str           # Complete model output for debugging
```

### 7.2 Triage Result Generation

```python
# File: triage/output_generator.py:36-85
def generate_triage_result(
    patient_id: str,
    analysis: AnalysisResult,
    images: List[Image.Image],
    slice_indices: List[int],
    conditions_from_context: List[str],
) -> Dict[str, Any]:

    # Get key slice image
    key_image = images[analysis.key_slice_index]

    # Create thumbnail (128x128) for UI display
    thumbnail = get_thumbnail(key_image)
    thumbnail_base64 = image_to_base64(thumbnail)

    # Map sampled index to original volume index
    original_slice_index = slice_indices[analysis.key_slice_index]

    # Combine rationale with context
    rationale = f"Visual analysis: {analysis.visual_findings}"
    if conditions_from_context:
        rationale += f" EHR Context: Patient has {', '.join(conditions_from_context)}."
    rationale += f" {analysis.priority_rationale}"

    return {
        "patient_id": patient_id,
        "priority_level": analysis.priority_level,
        "rationale": rationale,
        "key_slice_index": original_slice_index,
        "key_slice_thumbnail": thumbnail_base64,
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "conditions_considered": analysis.conditions_considered,
        "findings_summary": analysis.findings_summary,
        "visual_findings": analysis.visual_findings,
    }
```

### 7.3 Priority Definitions

```python
# File: triage/config.py:29-38
PRIORITY_CRITICAL = 1   # Acute pathology (PE, aortic dissection, pneumothorax)
PRIORITY_HIGH_RISK = 2  # Contextual mismatch (nodule + cancer history)
PRIORITY_ROUTINE = 3    # No acute/contextual flags

PRIORITY_NAMES = {
    PRIORITY_CRITICAL: "CRITICAL",
    PRIORITY_HIGH_RISK: "HIGH RISK",
    PRIORITY_ROUTINE: "ROUTINE",
}
```

### 7.4 Worklist Management

The worklist is sorted by priority (lowest number = highest priority):

```python
# File: triage/worklist.py:92-94
def _sort(self) -> None:
    """Sort entries by priority level (1 first), then by time."""
    self._entries.sort(key=lambda e: (e.priority_level, e.processed_at))
```

**Worklist JSON structure:**

```json
{
  "generated_at": "2026-01-29T18:35:40.200249Z",
  "total_entries": 5,
  "entries": [
    {
      "patient_id": "train_1_a_2",
      "priority_level": 2,
      "priority_name": "HIGH RISK",
      "findings_summary": "Multiple venous collaterals and collapsed left subclavian vein...",
      "processed_at": "2026-01-29T18:33:49.058344Z",
      "result_path": "/workspace/sentinel_x/output/triage_results/train_1_a_2/triage_result.json"
    }
  ]
}
```

---

## 8. Medical Terminology Reference

### 8.1 Critical Pathologies (Priority 1)

| Condition | Description | CT Findings |
|-----------|-------------|-------------|
| **Pulmonary Embolism (PE)** | Blood clot in pulmonary arteries | Filling defect in pulmonary artery, Hampton's hump, Westermark sign |
| **Aortic Dissection** | Tear in aortic wall | Intimal flap, double lumen, mediastinal widening |
| **Tension Pneumothorax** | Air in pleural space with mediastinal shift | Collapsed lung, mediastinal shift away from affected side |
| **Active Hemorrhage** | Internal bleeding | High-density fluid, contrast extravasation |
| **Aortic Rupture** | Torn aorta | Mediastinal hematoma, contrast extravasation |

### 8.2 High-Risk Conditions (Priority 2)

| Condition | Why High-Risk | Clinical Context |
|-----------|--------------|------------------|
| **Pulmonary Nodule + Cancer History** | May indicate metastasis | Patient with known malignancy |
| **Pleural Effusion** | May indicate heart failure, infection, malignancy | Large or bilateral |
| **Mediastinal Mass** | May be lymphoma, thymoma, or metastasis | New finding |
| **Consolidation + Immunocompromised** | Opportunistic infection risk | HIV, chemotherapy, transplant |

### 8.3 High-Risk Patient Populations

The system flags patients with these conditions (from `config.py:41-56`):

| Category | Conditions |
|----------|-----------|
| **Oncology** | Cancer, malignancy, carcinoma, tumor, neoplasm |
| **Metabolic** | Diabetes, diabetic |
| **Pulmonary** | COPD, pulmonary disease |
| **Cardiovascular** | Heart disease, cardiac, hypertension |
| **Immunologic** | Immunocompromised, immunosuppressed |

### 8.4 CT Window Settings

| Window | Center (L) | Width (W) | Use Case |
|--------|-----------|-----------|----------|
| **Soft Tissue** (used) | 40 HU | 400 HU | General screening, nodules, masses |
| **Lung** | -600 HU | 1500 HU | Lung parenchyma, emphysema |
| **Bone** | 400 HU | 1800 HU | Skeletal evaluation |
| **Mediastinum** | 50 HU | 350 HU | Mediastinal structures |

---

## 9. Technical Implementation Details

### 9.1 Key File Locations

| Component | File Path | Key Lines |
|-----------|-----------|-----------|
| Main workflow | `triage/agent.py` | 80-126 (`process_patient`) |
| Model loading | `triage/medgemma_analyzer.py` | 44-61 (`load_model`) |
| Message building | `triage/medgemma_analyzer.py` | 63-93 (`_build_messages`) |
| Inference | `triage/medgemma_analyzer.py` | 155-213 (`analyze`) |
| Response parsing | `triage/medgemma_analyzer.py` | 95-153 (`_parse_response`) |
| CT processing | `triage/ct_processor.py` | 100-126 (`process_ct_volume`) |
| HU windowing | `triage/ct_processor.py` | 16-33 (`apply_window`) |
| Slice sampling | `triage/ct_processor.py` | 60-76 (`sample_slices`) |
| FHIR parsing | `triage/fhir_context.py` | 217-264 (`parse_fhir_context`) |
| Context formatting | `triage/fhir_context.py` | 267-306 (`format_context_for_prompt`) |
| Output generation | `triage/output_generator.py` | 36-85 (`generate_triage_result`) |
| System prompt | `triage/prompts.py` | 3-42 (`SYSTEM_PROMPT`) |
| User prompt | `triage/prompts.py` | 44-55 (`USER_PROMPT_TEMPLATE`) |
| Configuration | `triage/config.py` | All constants |

### 9.2 Dependencies

```
transformers>=4.50.0  # For Gemma 3/MedGemma support
torch                 # PyTorch for model inference
nibabel               # NIfTI file loading
numpy                 # Array operations
Pillow                # Image processing
```

### 9.3 Memory Requirements

| Component | Approximate Memory |
|-----------|-------------------|
| MedGemma 4B (BF16) | ~8 GB VRAM |
| 85 CT slice images | ~200-400 MB RAM |
| Input tensors | ~500 MB VRAM |
| **Recommended GPU** | 16 GB+ VRAM |

### 9.4 Performance Characteristics

| Metric | Value |
|--------|-------|
| Model loading time | 30-60 seconds (first run) |
| CT processing time | 5-10 seconds |
| FHIR parsing time | <1 second |
| MedGemma inference time | 30-90 seconds (GPU dependent) |
| Total per-patient time | ~45-120 seconds |

---

## 10. Appendices

### Appendix A: Example Triage Output

```json
{
  "patient_id": "train_1_a_2",
  "priority_level": 2,
  "rationale": "Visual analysis: Multiple venous collaterals in the anterior left chest wall. Collapsed appearance in the left subclavian vein. Thickening of the bronchial wall in both lungs. Peribronchial reticulonodular densities in the lower lobes, minimal consolidations (infection process?). Atelectasis in both lungs. Osteophytes with anterior extension in the thoracic vertebrae. The presence of multiple venous collaterals and a collapsed left subclavian vein raises concern for a potential vascular abnormality, possibly related to chronic venous obstruction or other vascular pathology. The bronchial wall thickening and peribronchial reticulonodular densities in the lower lobes, along with minimal consolidations, are suggestive of a possible infectious or inflammatory process.",
  "key_slice_index": 161,
  "key_slice_thumbnail": "[base64 encoded PNG]",
  "processed_at": "2026-01-29T18:33:49.057640Z",
  "conditions_considered": [
    "Vascular malformation",
    "Chronic venous insufficiency",
    "Bronchiectasis",
    "Pneumonia",
    "Tuberculosis",
    "Lung cancer"
  ],
  "findings_summary": "Multiple venous collaterals and collapsed left subclavian vein raise concern for vascular pathology. Bronchial wall thickening and peribronchial reticulonodular densities in the lower lobes suggest a possible infectious or inflammatory process.",
  "visual_findings": "Multiple venous collaterals in the anterior left chest wall. Collapsed appearance in the left subclavian vein. Thickening of the bronchial wall in both lungs. Peribronchial reticulonodular densities in the lower lobes, minimal consolidations (infection process?). Atelectasis in both lungs. Osteophytes with anterior extension in the thoracic vertebrae."
}
```

### Appendix B: MedGemma Performance Benchmarks

From [MedGemma documentation](https://huggingface.co/google/medgemma-4b-it):

| Task | Benchmark | MedGemma 4B Score |
|------|-----------|-------------------|
| CXR Classification (MIMIC) | Macro F1 | 88.9% |
| Pathology MCQ | Accuracy | 69.8% |
| Dermatology MCQ | Accuracy | 71.8% |
| Radiology VQA (SLAKE) | Tokenized F1 | 72.3% |
| Medical QA (MedQA) | Accuracy | 64.4% |

### Appendix C: FHIR Resource Examples

**Patient Resource:**
```json
{
  "resourceType": "Patient",
  "gender": "male",
  "birthDate": "1961-03-15"
}
```

**Condition Resource:**
```json
{
  "resourceType": "Condition",
  "code": {
    "text": "Diabetes mellitus type 2",
    "coding": [{"system": "ICD-10", "code": "E11", "display": "Type 2 diabetes mellitus"}]
  }
}
```

**DiagnosticReport Resource:**
```json
{
  "resourceType": "DiagnosticReport",
  "status": "final",
  "code": {"text": "CT Chest"},
  "conclusion": "No acute cardiopulmonary findings."
}
```

---

## References

- [MedGemma on Google DeepMind](https://deepmind.google/models/gemma/medgemma/)
- [MedGemma 4B on HuggingFace](https://huggingface.co/google/medgemma-4b-it)
- [MedGemma Technical Report (arXiv:2507.05201)](https://arxiv.org/abs/2507.05201)
- [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [FHIR R4 Specification](https://www.hl7.org/fhir/)
- [NIfTI File Format](https://nifti.nimh.nih.gov/)
