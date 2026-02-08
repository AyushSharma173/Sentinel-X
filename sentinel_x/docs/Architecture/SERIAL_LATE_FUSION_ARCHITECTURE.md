# Serial Late Fusion Architecture

**Date:** 2026-02-08
**Status:** Implemented
**Supersedes:** Single-model pipeline described in `triage_system_technical_deep_dive.md`

---

## Table of Contents

1. [Motivation — Why the Old Architecture Failed](#1-motivation--why-the-old-architecture-failed)
2. [Architecture Overview](#2-architecture-overview)
3. [Hardware Constraints & VRAM Budget](#3-hardware-constraints--vram-budget)
4. [Model Selection Rationale](#4-model-selection-rationale)
5. [Phase 0: Preprocessing](#5-phase-0-preprocessing)
6. [Phase 1: Visual Detection (4B)](#6-phase-1-visual-detection-4b)
7. [VRAM Swap](#7-vram-swap)
8. [Phase 2: Clinical Reasoning (27B)](#8-phase-2-clinical-reasoning-27b)
9. [Phase 3: Output Generation](#9-phase-3-output-generation)
10. [CT Multi-Window Preprocessing](#10-ct-multi-window-preprocessing)
11. [Prompt Engineering](#11-prompt-engineering)
12. [Data Flow & Data Classes](#12-data-flow--data-classes)
13. [API & WebSocket Integration](#13-api--websocket-integration)
14. [Session Logging](#14-session-logging)
15. [File Manifest](#15-file-manifest)
16. [Dependencies](#16-dependencies)
17. [Performance Characteristics](#17-performance-characteristics)

---

## 1. Motivation — Why the Old Architecture Failed

The original Sentinel-X pipeline fed both the FHIR clinical narrative and 85 CT slice images into a **single MedGemma 4B model call**. The model was asked to simultaneously:

1. Detect visual findings in the CT images
2. Synthesize those findings with the patient's clinical history
3. Assign a triage priority level

This caused two fundamental problems:

### 1.1 Cognitive Bias

When the 4B model received the clinical history alongside the images, it "hallucinated" findings that matched the history. For example, a patient with documented hypertension and diabetes would get Priority 2 (HIGH RISK) even when the CT was unremarkable — the model flagged chronic conditions as "new findings" because it lacked the reasoning depth to perform temporal comparison.

The root cause: **the same model that sees the images also sees the clinical history**, so it cannot distinguish between "what I see" and "what I expect to see."

### 1.2 Architecture Mismatch

Per the MedGemma paper, the 4B model has "difficulty following system instructions." Asking it to simultaneously:

- Parse 85 images visually
- Cross-reference against a multi-thousand-token clinical narrative
- Apply a complex priority decision framework
- Output structured text in a specific format

...was simply too much cognitive load for a 4B model. The result was unreliable priority assignments and malformed output.

### 1.3 CT Input Distribution Mismatch

The old pipeline applied a **single soft-tissue window** (center=40, width=400) to all CT data, then duplicated the grayscale result to all 3 RGB channels. But Google's official MedGemma 1.5 CT notebook reveals the model was trained on **3-channel multi-window CT images**:

| Channel | Window | HU Range | Purpose |
|---------|--------|----------|---------|
| R (Red) | Wide | -1024 to 1024 | Full range: air to bone |
| G (Green) | Soft tissue | -135 to 215 | Fat to start of bone |
| B (Blue) | Brain | 0 to 80 | Water to brain density |

Feeding single-window grayscale into a model trained on 3-channel multi-window data is a **training distribution mismatch** — the model was effectively seeing inputs it was never trained on.

### 1.4 The Fix: Serial Late Fusion

Separate **visual detection** (what do I see?) from **clinical reasoning** (what does it mean in context?) using two different models optimized for each task, running serially within the same 24GB VRAM budget.

---

## 2. Architecture Overview

```
Per-Patient Pipeline (Serial Execution):

  ┌─────────────────────────────────────────────────────┐
  │ PHASE 0: PREPROCESSING (No GPU)                     │
  │                                                     │
  │  FHIR Bundle ──→ FHIRJanitor ──→ ClinicalStream    │
  │  CT Volume   ──→ Multi-Window ──→ PIL Images (RGB)  │
  │                   (wide/soft/brain → R/G/B)         │
  └─────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │ PHASE 1: VISUAL DETECTION (~8GB VRAM)               │
  │  Model: google/medgemma-1.5-4b-it (BF16, no quant) │
  │                                                     │
  │  Input:  CT slice images ONLY (no clinical history) │
  │  Prompt: "List all visible anatomical findings"     │
  │  Output: Visual Fact Sheet (structured JSON)        │
  │                                                     │
  │  → UNLOAD model, gc.collect(), empty VRAM cache     │
  └─────────────────────────────────────────────────────┘
                          │
                   Visual Fact Sheet JSON
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │ PHASE 2: CLINICAL REASONING (~13-14GB VRAM)         │
  │  Model: google/medgemma-27b-it (NF4 4-bit quant)   │
  │                                                     │
  │  Input:  FHIR ClinicalStream (text)                 │
  │          + Visual Fact Sheet (JSON from Phase 1)    │
  │  Prompt: "Perform Delta Analysis" (text-only)       │
  │  Output: Delta Analysis + Final Triage Priority     │
  │                                                     │
  │  → UNLOAD model, gc.collect(), empty VRAM cache     │
  └─────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │ PHASE 3: OUTPUT GENERATION (No GPU)                 │
  │                                                     │
  │  Merge Phase 1 + Phase 2 → triage_result.json      │
  │  Update worklist, session log, WebSocket            │
  └─────────────────────────────────────────────────────┘
```

The key insight is that **Phase 1 never sees the clinical history**, which prevents cognitive bias. The 4B model acts as an "unbiased sensor" that reports only what it physically observes. Phase 2 then performs the higher-order reasoning task of comparing those observations against the patient's medical record — a task perfectly suited to a large text-only model.

---

## 3. Hardware Constraints & VRAM Budget

**Target GPU:** NVIDIA RTX 4090, 24GB VRAM

The two models **cannot coexist in memory** — they must be loaded and unloaded serially.

```
Phase 1 (Vision — MedGemma 1.5 4B):
  Model weights (BF16):          ~8.0 GB
  SigLIP image encoding (85x):  ~1.0 GB
  KV cache (~22K tokens):       ~0.5 GB  (85 images × 256 tok + text)
  Activations + overhead:       ~1.5 GB
  ─────────────────────────────────────
  Total:                        ~11.0 GB  ✓ (13GB headroom)

[Swap: unload → gc.collect → empty_cache → synchronize → ~0 GB]

Phase 2 (Reasoning — MedGemma 27B):
  Model weights (NF4 4-bit):    ~13.5 GB (with double quant)
  KV cache (~18K tokens text):  ~1.5 GB
  Activations + overhead:       ~2.0 GB
  ─────────────────────────────────────
  Total:                        ~17.0 GB  ✓ (7GB headroom)
```

---

## 4. Model Selection Rationale

### Phase 1: Vision — `google/medgemma-1.5-4b-it`

| Property | Value | Rationale |
|----------|-------|-----------|
| **Version** | v1.5 (Jan 2026) | v1.5 was explicitly trained on 3D CT volumes. v1.0 docs say "not evaluated for multiple images." |
| **CT Performance** | CT-RATE F1: 27.0 | vs v1.0's 23.5 — 15% improvement on CT classification |
| **Quantization** | BFloat16 (none) | Full precision preserves visual fidelity for medical imaging |
| **VRAM** | ~8 GB | Leaves 16GB headroom on RTX 4090 |
| **Architecture** | Gemma 3 + SigLIP | SigLIP encoder trained on de-identified medical images |

### Phase 2: Reasoning — `google/medgemma-27b-it`

| Property | Value | Rationale |
|----------|-------|-----------|
| **Version** | v1.0 (May 2025) | Best text reasoning available; 27B doesn't have a v1.5 |
| **Medical QA** | MedQA: 87.7% | vs 4B's 64.4% — 23 points better at clinical reasoning |
| **EHR Reasoning** | EHRQA: 93.6% | Excels at multi-hop EHR reasoning — exactly what Delta Analysis needs |
| **Quantization** | NF4 4-bit (BitsAndBytes) | Double quantization compresses 27B params to ~13.5GB |
| **VRAM** | ~13-14 GB | Fits within 24GB budget after Phase 1 unload |

---

## 5. Phase 0: Preprocessing

Phase 0 runs entirely on CPU — no GPU required.

### 5.1 FHIR Extraction

The `FHIRJanitor` processes raw FHIR bundles into a condensed `ClinicalStream`:

```
FHIR Bundle (JSON)
    │
    ├── GarbageCollector: discard Provenance, Organization, etc.
    ├── NarrativeDecoder: extract hidden diagnoses from Claims/EOBs
    ├── Per-resource extractors:
    │   ├── PatientExtractor → demographics, age, gender
    │   ├── ConditionExtractor → medical history with dates
    │   ├── MedicationExtractor → active/historical medications
    │   ├── ObservationExtractor → labs, vitals
    │   ├── ProcedureExtractor → surgical history
    │   └── EncounterExtractor → visit history
    └── TimelineSerializer → chronological narrative
```

The clinical narrative output is passed to **Phase 2 only** — Phase 1 never sees it.

### 5.2 CT Volume Processing

The volume is loaded in raw Hounsfield Units and converted to 3-channel multi-window RGB images. See [Section 10](#10-ct-multi-window-preprocessing) for details.

**File:** `triage/ct_processor.py`

```python
def process_ct_volume(path: Path) -> Tuple[List[Image.Image], List[int], dict]:
    volume_hu, metadata = load_nifti_volume(path)       # Raw HU
    slice_indices = sample_slices(volume_hu, CT_NUM_SLICES)  # 85 indices
    images = [extract_slice_as_multiwindow_image(volume_hu, idx)
              for idx in slice_indices]
    return images, slice_indices, metadata
```

---

## 6. Phase 1: Visual Detection (4B)

**File:** `triage/medgemma_analyzer.py` — class `VisionAnalyzer`

### 6.1 Purpose

Act as an "unbiased sensor" — report ONLY what is physically visible in the CT images, with zero knowledge of the patient's clinical history. This prevents the cognitive bias problem.

### 6.2 Model Loading

```python
self.processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
self.model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

No quantization — BFloat16 preserves visual fidelity.

### 6.3 Message Format

Follows Google's official CT notebook format: a single `user` message (no `system` role) with interleaved images and slice labels:

```
[user message]
  "You are a radiologist's visual detection system..."     ← instruction
  [IMAGE 1]  "SLICE 1"                                     ← interleaved
  [IMAGE 2]  "SLICE 2"
  ...
  [IMAGE 85] "SLICE 85"
  "Examine these 85 chest CT slices. List every..."        ← query
```

Key design decisions:
- **No `system` role** — matching Google's notebook exactly
- **Interleaved images with labels** — helps the model track which slice each finding comes from
- **No clinical context** — prevents bias
- **No priority definitions** — the 4B model doesn't assign priorities

### 6.4 Output: Visual Fact Sheet

The model outputs structured JSON:

```json
{
  "findings": [
    {
      "finding": "nodule",
      "location": "RUL",
      "size": "4mm",
      "slice_index": 42,
      "description": "small round opacity in right upper lobe"
    },
    {
      "finding": "effusion",
      "location": "bilateral",
      "size": "small",
      "slice_index": 71,
      "description": "thin fluid layer in bilateral costophrenic angles"
    }
  ]
}
```

This JSON is parsed into a `VisualFactSheet` dataclass using `json_repair.py` to handle the 4B model's occasional JSON formatting quirks (markdown fences, Python True/False, trailing commas, etc.).

### 6.5 Unloading

After Phase 1, the model is fully unloaded:

```python
self.vision_analyzer.unload()
# Internally calls vram_manager.unload_model() which:
#   1. Moves model to CPU (prevents GPU fragmentation)
#   2. Deletes model and processor references
#   3. gc.collect()
#   4. torch.cuda.empty_cache()
#   5. torch.cuda.synchronize()
```

---

## 7. VRAM Swap

**File:** `triage/vram_manager.py`

The swap between Phase 1 and Phase 2 is the critical moment — we must fully release the 4B model's ~11GB before loading the 27B model's ~17GB.

```python
def unload_model(model, processor) -> None:
    log_vram_status("before unload")

    # Move to CPU first to prevent GPU memory fragmentation
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass  # Quantized models may not support .to()
        del model

    if processor is not None:
        del processor

    gc.collect()                    # Python garbage collection
    torch.cuda.empty_cache()        # Release CUDA memory pool
    torch.cuda.synchronize()        # Wait for all GPU ops to complete

    log_vram_status("after unload")
```

The `log_vram_status()` function records allocated/reserved/free VRAM at each step, making it easy to diagnose memory issues.

---

## 8. Phase 2: Clinical Reasoning (27B)

**File:** `triage/medgemma_reasoner.py` — class `ClinicalReasoner`

### 8.1 Purpose

Perform **Delta Analysis** — compare each visual finding from Phase 1 against the patient's clinical history and classify it as:

| Classification | Priority | Meaning |
|---------------|----------|---------|
| `CHRONIC_STABLE` | 3 | Finding matches a known condition documented >3 months ago |
| `ACUTE_NEW` | 1 or 2 | Finding has NO corresponding entry in clinical history |
| `DISCORDANT` | 2 | Clinical history suggests acute presentation but imaging disagrees, or vice versa |

### 8.2 Model Loading (4-bit Quantization)

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16 for accuracy
    bnb_4bit_use_double_quant=True,        # Nested quantization for extra savings
)

self.model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-27b-it",
    quantization_config=quantization_config,
    device_map="auto",
)
```

**NF4 quantization** compresses 27B parameters from ~54GB (FP16) to ~13.5GB while preserving reasoning quality. **Double quantization** applies a second round of quantization to the quantization constants themselves, saving an additional ~0.4GB.

### 8.3 Message Format

Standard system + user role format. **Text-only** — no images.

```
[system]
  "You are an expert clinical reasoning system performing triage
   Delta Analysis..."

[user]
  "## CLINICAL HISTORY
   [full FHIR narrative from FHIRJanitor]

   ## VISUAL FINDINGS FROM CT SCAN
   [JSON fact sheet from Phase 1]

   Perform Delta Analysis. Compare each visual finding against the
   clinical history. Classify every finding and determine the overall
   triage priority."
```

### 8.4 Output: Delta Analysis Result

```json
{
  "delta_analysis": [
    {
      "finding": "4mm nodule in RUL",
      "classification": "CHRONIC_STABLE",
      "priority": 3,
      "history_match": "Pulmonary nodule (since 2024-01)",
      "reasoning": "Known nodule documented 13 months ago, stable"
    },
    {
      "finding": "bilateral effusion",
      "classification": "ACUTE_NEW",
      "priority": 2,
      "history_match": null,
      "reasoning": "No prior effusion documented in clinical history"
    }
  ],
  "overall_priority": 2,
  "priority_rationale": "New bilateral effusion not present in clinical history warrants prompt review.",
  "findings_summary": "Known stable RUL nodule. New bilateral pleural effusion — recommend urgent follow-up."
}
```

### 8.5 Priority Escalation Rules

The prompt defines strict escalation logic:

1. Any single Priority 1 finding → **Overall Priority 1**
2. Any Priority 2 finding (with no Priority 1) → **Overall Priority 2**
3. All findings `CHRONIC_STABLE` → **Overall Priority 3**
4. Empty visual findings + acute clinical presentation → **Priority 2 (DISCORDANT)**

---

## 9. Phase 3: Output Generation

**File:** `triage/output_generator.py`

Merges Phase 1 and Phase 2 outputs into `triage_result.json`, maintaining backward compatibility with the old format while adding new fields.

### 9.1 Key Slice Selection

The key slice is derived from the highest-priority finding's `slice_index` from the Phase 1 fact sheet. If no findings exist, falls back to the middle slice.

### 9.2 Output JSON Structure

```json
{
  "patient_id": "train_1_a_1",
  "priority_level": 2,
  "rationale": "Visual analysis: nodule (RUL, 4mm)... EHR Context: ...",
  "key_slice_index": 42,
  "key_slice_thumbnail": "base64...",
  "processed_at": "2026-02-08T...",
  "findings_summary": "Known stable RUL nodule. New bilateral effusion.",
  "visual_findings": "nodule (RUL, 4mm): small round opacity; ...",
  "conditions_considered": ["Diabetes mellitus type 2", "Hypertension"],
  "delta_analysis": [
    {
      "finding": "4mm nodule in RUL",
      "classification": "CHRONIC_STABLE",
      "priority": 3,
      "history_match": "Pulmonary nodule (since 2024-01)",
      "reasoning": "Known nodule documented 13 months ago, stable"
    }
  ],
  "phase1_raw": "...",
  "phase2_raw": "..."
}
```

| Field | Source | Backward Compatible? |
|-------|--------|---------------------|
| `patient_id` | Input | Yes |
| `priority_level` | Phase 2 `overall_priority` | Yes |
| `rationale` | Combined Phase 1 + Phase 2 | Yes |
| `key_slice_index` | Phase 1 highest-priority finding | Yes |
| `key_slice_thumbnail` | PIL thumbnail → base64 | Yes |
| `processed_at` | UTC timestamp | Yes |
| `findings_summary` | Phase 2 `findings_summary` | Yes |
| `visual_findings` | Phase 1 findings as text | Yes |
| `conditions_considered` | FHIR conditions | Yes |
| `delta_analysis` | Phase 2 full classification list | **New** |
| `phase1_raw` | Phase 1 raw model output | **New** |
| `phase2_raw` | Phase 2 raw model output | **New** |

---

## 10. CT Multi-Window Preprocessing

**File:** `triage/ct_processor.py`

### 10.1 The Problem with Single-Window

The old pipeline applied a single soft-tissue window (center=40, width=400, mapping -160 to +240 HU) and duplicated it to all 3 RGB channels. This:

1. **Discards information** — air (-1000 to -160 HU) and bone (+240 to +1000 HU) are clipped
2. **Mismatches training** — MedGemma 1.5 was trained on 3-channel multi-window images
3. **Wastes channel capacity** — identical R, G, B channels carry no additional information

### 10.2 Google's Official 3-Channel Windowing

From `high_dimensional_ct_hugging_face.ipynb`:

```python
CT_WINDOW_WIDE  = (-1024, 1024)   # R channel: full HU range (air to bone)
CT_WINDOW_SOFT  = (-135, 215)     # G channel: soft tissue
CT_WINDOW_BRAIN = (0, 80)         # B channel: brain density
```

Each channel independently clips and normalizes the raw HU values to 0-255:

```python
def norm(ct_slice: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    clipped = np.clip(ct_slice, hu_min, hu_max).astype(np.float32)
    clipped -= hu_min
    clipped /= (hu_max - hu_min)
    clipped *= 255.0
    return clipped
```

### 10.3 Visual Comparison

```
Single-Window (old):          Multi-Window (new):
┌───────────────────┐         ┌───────────────────┐
│ R = Soft tissue   │         │ R = Wide (-1024,+1024)
│ G = Soft tissue   │  →      │ G = Soft (-135,+215)
│ B = Soft tissue   │         │ B = Brain (0,+80)
│ (identical)       │         │ (3 distinct views)
└───────────────────┘         └───────────────────┘
```

The multi-window approach gives the model 3x more information per image:
- **R (Wide)**: Shows bone, calcifications, air, and all structures at lower contrast
- **G (Soft tissue)**: Optimized for organs, nodules, effusions
- **B (Brain)**: High-contrast view of water-density structures, subtle density differences

### 10.4 Pipeline Flow

```python
volume_hu, metadata = load_nifti_volume(path)   # Raw HU (NOT pre-windowed)
slice_indices = sample_slices(volume_hu, 85)

for idx in slice_indices:
    raw_slice = volume_hu[:, :, idx]
    raw_slice = np.rot90(raw_slice)

    r = norm(raw_slice, -1024, 1024)   # Wide
    g = norm(raw_slice, -135, 215)     # Soft
    b = norm(raw_slice, 0, 80)         # Brain

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
```

Note: The legacy `apply_window()` and `extract_slice_as_image()` functions are preserved for backward compatibility (e.g., the `/api/patients/{id}/slices/{idx}` endpoint still uses them for serving individual slice images to the frontend viewer).

---

## 11. Prompt Engineering

**File:** `triage/prompts.py`

### 11.1 Phase 1 — "Unbiased Sensor"

**Design principles:**
- **No clinical context** — prevents the model from "seeing what it expects to see"
- **No priority definitions** — the 4B model should not make triage judgments
- **JSON output only** — avoids the 4B model's documented difficulty with complex instructions
- **Factual reporting only** — "Report what you see, not what it means"

The system prompt explicitly instructs:

> "Report ONLY what you physically see in the images. Do NOT infer clinical significance, do NOT suggest diagnoses, do NOT assign urgency."

### 11.2 Phase 2 — "Delta Analyst"

**Design principles:**
- **Text-only** — the 27B model excels at text reasoning (MedQA 87.7%, EHRQA 93.6%)
- **Explicit classification framework** — `CHRONIC_STABLE`, `ACUTE_NEW`, `DISCORDANT`
- **History matching** — each finding must cite its matching history entry or explicitly state "null"
- **Priority escalation rules** — deterministic rules prevent ambiguous priority assignments
- **JSON output** — structured for programmatic consumption

The prompt defines the complete Delta Analysis framework, ensuring the model:
1. Iterates over each visual finding
2. Searches the clinical history for matching entries
3. Classifies based on temporal and clinical comparison
4. Applies deterministic priority escalation rules

---

## 12. Data Flow & Data Classes

### 12.1 Phase 1 Data Classes

**File:** `triage/medgemma_analyzer.py`

```python
@dataclass
class VisualFinding:
    finding: str       # "nodule", "opacity", "effusion"
    location: str      # "RUL", "LLL", "bilateral"
    size: str          # "4mm", "small", "large"
    slice_index: int   # Which slice (1-85) shows this best
    description: str   # Brief factual description

@dataclass
class VisualFactSheet:
    findings: List[VisualFinding]
    raw_response: str
    num_slices_analyzed: int

    def to_dict(self) -> dict:  # For JSON serialization
    def to_json(self) -> str:   # For Phase 2 prompt injection
```

### 12.2 Phase 2 Data Classes

**File:** `triage/medgemma_reasoner.py`

```python
@dataclass
class DeltaEntry:
    finding: str              # Description from fact sheet
    classification: str       # CHRONIC_STABLE | ACUTE_NEW | DISCORDANT
    priority: int             # 1, 2, or 3
    history_match: Optional[str]  # Matching EHR entry or None
    reasoning: str            # Why this classification

@dataclass
class DeltaAnalysisResult:
    delta_analysis: List[DeltaEntry]
    overall_priority: int     # 1, 2, or 3
    priority_rationale: str   # 1-2 sentence explanation
    findings_summary: str     # Worklist-ready summary
    raw_response: str
```

### 12.3 Complete Data Flow

```
FHIR Bundle  ──→  FHIRJanitor  ──→  ClinicalStream.narrative (text)
                                          │
                                          │ (Phase 2 only)
                                          ▼
CT NIfTI  ──→  Multi-Window RGB  ──→  List[PIL.Image]
                                          │
                                          │ (Phase 1)
                                          ▼
                                    VisionAnalyzer.analyze()
                                          │
                                          ▼
                                    VisualFactSheet
                                          │
                                    .to_dict() / .to_json()
                                          │
                                          ▼
                    ClinicalReasoner.analyze(narrative, fact_sheet_dict)
                                          │
                                          ▼
                                    DeltaAnalysisResult
                                          │
                                          ▼
                    generate_triage_result(fact_sheet, delta_result, ...)
                                          │
                                          ▼
                                    triage_result.json
```

---

## 13. API & WebSocket Integration

**Files:** `api/models.py`, `api/services/demo_service.py`, `api/routes/patients.py`

### 13.1 New WebSocket Events

The Serial Late Fusion pipeline adds 5 new WebSocket event types for real-time UI updates:

| Event | Fired When | Data |
|-------|-----------|------|
| `PHASE1_STARTED` | 4B model begins vision analysis | `patient_id`, `model` |
| `PHASE1_COMPLETE` | Phase 1 finishes | `patient_id`, `num_findings` |
| `MODEL_SWAPPING` | Unloading 4B, loading 27B | `from_model`, `to_model` |
| `PHASE2_STARTED` | 27B model begins delta analysis | `patient_id`, `model` |
| `PHASE2_COMPLETE` | Phase 2 finishes | `patient_id`, `priority` |

These are in addition to the existing events (`PROCESSING_STARTED`, `PROCESSING_COMPLETE`, `WORKLIST_UPDATED`, etc.).

### 13.2 Demo Service Changes

The demo service no longer preloads a single model at startup. Models are transient — loaded and unloaded per-patient inside the agent pipeline. The `_model_loaded` flag is set to `True` after the first patient completes successfully.

### 13.3 API Response Model

The `TriageResult` Pydantic model (served at `/api/patients/{id}/triage`) now includes:

```python
class DeltaAnalysisEntry(BaseModel):
    finding: str
    classification: str
    priority: int
    history_match: Optional[str] = None
    reasoning: str

class TriageResult(BaseModel):
    # ... existing fields (backward compatible) ...
    delta_analysis: List[DeltaAnalysisEntry] = []
    phase1_raw: str = ""
    phase2_raw: str = ""
```

---

## 14. Session Logging

**File:** `triage/session_logger.py`

The session logger writes a human-readable `.txt` trace under `logs/sessions/{session_id}/session.txt`. The new pipeline steps are:

| Step | Method | Content |
|------|--------|---------|
| 1 | `log_fhir_extraction()` | Bundle structure, conditions, narrative |
| 2 | `log_ct_processing()` | Volume info, multi-window channel details |
| 3 | `log_phase1_prompt()` | Phase 1 system/user prompt, image count |
| 3 | `log_phase1_response()` | Raw response + parsed findings table |
| — | `log_model_swap()` | VRAM status, swap duration |
| 4 | `log_phase2_prompt()` | Phase 2 system/user prompt (narrative + fact sheet) |
| 4 | `log_phase2_response()` | Raw response |
| 5 | `log_delta_analysis_table()` | Formatted classification table |
| 6 | `log_output_saved()` | Result path, final priority |

Each step includes timestamps and durations for performance profiling.

---

## 15. File Manifest

### New Files

| File | Purpose |
|------|---------|
| `triage/vram_manager.py` | GPU memory management (unload, logging, free check) |
| `triage/medgemma_reasoner.py` | Phase 2 ClinicalReasoner (27B NF4 4-bit) |

### Modified Files

| File | Changes |
|------|---------|
| `triage/config.py` | Added `VISION_MODEL_ID`, `REASONER_MODEL_ID`, quantization config, `CT_WINDOW_WIDE/SOFT/BRAIN` |
| `triage/ct_processor.py` | Added `norm()`, `extract_slice_as_multiwindow_image()`. `process_ct_volume()` now uses multi-window. Legacy functions kept. |
| `triage/prompts.py` | Replaced single prompt with Phase 1 + Phase 2 prompts. Legacy `SYSTEM_PROMPT` and `build_user_prompt()` kept as aliases. |
| `triage/medgemma_analyzer.py` | Renamed to `VisionAnalyzer`, added `VisualFinding`/`VisualFactSheet` dataclasses, JSON output parsing. `MedGemmaAnalyzer` kept as alias. |
| `triage/agent.py` | Rewrote `_process_patient_internal()` for 4-phase serial execution. Removed upfront model preload from `run()`. |
| `triage/output_generator.py` | Accepts `VisualFactSheet` + `DeltaAnalysisResult`. Added `delta_analysis`, `phase1_raw`, `phase2_raw` to output. |
| `triage/session_logger.py` | Added Phase 1/2 logging methods, model swap logging, delta analysis table. |
| `api/models.py` | Added `DeltaAnalysisEntry`, new fields on `TriageResult`, 5 new `WSEventType` values. |
| `api/services/demo_service.py` | Removed model preload block. Models are transient per-patient. |
| `api/routes/patients.py` | Passes `delta_analysis`, `phase1_raw`, `phase2_raw` from result JSON. |
| `requirements-api.txt` | Added `bitsandbytes>=0.43.0`. |

### Unchanged Files

The following files required no changes:
- `triage/fhir_janitor.py` — FHIR extraction is the same
- `triage/worklist.py` — Worklist uses `priority_level` and `findings_summary`, both still provided
- `triage/inbox_watcher.py` — File discovery is unchanged
- `triage/json_repair.py` — Now used by Phase 1 parser (was already designed for this)
- `triage/__init__.py` — Still exports `TriageAgent`
- All frontend files — The frontend reads `priority_level`, `findings_summary`, etc., all backward compatible

---

## 16. Dependencies

### requirements-api.txt

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
websockets>=12.0
pydantic>=2.5.0
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.9.0
pillow>=10.0.0
bitsandbytes>=0.43.0      # NEW: NF4 4-bit quantization for 27B model
```

The `bitsandbytes` library provides the NF4 quantization backend used by `BitsAndBytesConfig` in the `ClinicalReasoner`. It requires a CUDA-capable GPU.

---

## 17. Performance Characteristics

### Per-Patient Timing Estimates

| Stage | Estimated Duration | GPU Required? |
|-------|-------------------|---------------|
| FHIR extraction | <1s | No |
| CT multi-window processing | 5-10s | No |
| Phase 1: Load 4B model | 20-40s (first patient, cached after) | Yes |
| Phase 1: Vision inference | 30-60s | Yes |
| VRAM swap (unload + load) | 30-60s | Yes |
| Phase 2: Reasoning inference | 30-90s | Yes |
| Output generation | <1s | No |
| **Total per patient** | **~2-4 minutes** | |

Note: Model loading times are dominated by disk I/O on the first patient. If HuggingFace cache is warm, subsequent loads are faster. The models are loaded and unloaded for every patient (not cached between patients) to guarantee VRAM cleanliness.

### Memory Usage

| Resource | Approximate Size |
|----------|-----------------|
| Phase 1 peak VRAM | ~11 GB |
| Phase 2 peak VRAM | ~17 GB |
| CT volume in RAM | ~200-400 MB |
| 85 PIL images in RAM | ~100-200 MB |
| Total system RAM | ~4 GB |

### Comparison with Old Architecture

| Metric | Old (Single 4B) | New (Late Fusion) |
|--------|-----------------|-------------------|
| Models | 1 (4B) | 2 (4B + 27B) |
| VRAM peak | ~11 GB | ~17 GB |
| Time per patient | ~45-120s | ~2-4 min |
| Priority accuracy | Low (bias issues) | Higher (unbiased vision + better reasoning) |
| Chronic finding handling | Poor (flagged as new) | Good (Delta Analysis) |
| CT input fidelity | Single-window grayscale | 3-channel multi-window |
| Reasoning quality | 4B (MedQA 64.4%) | 27B (MedQA 87.7%) |

The trade-off is clear: **~2x longer processing time** in exchange for **dramatically more accurate triage** — a worthwhile trade for a system where correctness matters far more than speed.
