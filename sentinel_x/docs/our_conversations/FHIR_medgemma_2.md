# FHIR & MedGemma Deep Dive: Architecture Brainstorming

> **Document Version:** 2.0
> **Date:** January 2026
> **Scope:** Advanced architecture exploration for integrating FHIR data with MedGemma models in Sentinel-X

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Model Capabilities Matrix](#2-model-capabilities-matrix)
3. [How MedGemma Handles FHIR Data](#3-how-medgemma-handles-fhir-data)
4. [How MedGemma Handles 3D CT Volumes](#4-how-medgemma-handles-3d-ct-volumes)
5. [Architecture Options Analysis](#5-architecture-options-analysis)
6. [GPU & Hosting Analysis](#6-gpu--hosting-analysis)
7. [Implementation Approach](#7-implementation-approach)
8. [Critical Questions & Trade-offs](#8-critical-questions--trade-offs)
9. [Appendices](#9-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

This document is a comprehensive technical brainstorming exploration of how to optimally integrate FHIR clinical data with MedGemma models for CT triage. We explore multiple architecture options, analyze the official MedGemma documentation and notebooks, and provide concrete implementation patterns.

### 1.2 Key Research Discoveries

| Discovery | Significance | Impact on Architecture |
|-----------|--------------|----------------------|
| **27B text-only lacks FHIR training** | Only the 27B *multimodal* variant has Synthea RL training | Cannot use 27B text-only for FHIR extraction |
| **MedGemma 1.5 4B supports 3D CT** | Uses video-style processing for volumetric data | Enables true 3D analysis on affordable GPU |
| **85-slice sampling is official** | Google uses same MAX_SLICE=85 as Sentinel-X | Our implementation aligns with official approach |
| **EHR Navigator uses LangGraph** | Agent-based iterative FHIR retrieval | Provides pattern for smart context extraction |
| **EHRQA uses Synthea data** | Same format as Sentinel-X FHIR bundles | Model is trained on compatible data |

### 1.3 Architecture Recommendation Preview

After analyzing all options, we recommend a **phased approach**:

```
Phase 1 (Now): Enhanced FHIR extraction with current MedGemma 4B
Phase 2 (Next): Add EHR Navigator Agent pattern for smart retrieval
Phase 3 (Future): Evaluate MedGemma 1.5 4B for 3D CT support
Phase 4 (If needed): Consider 27B multimodal for complex cases
```

### 1.4 Critical Finding: Model Capability Matrix

| Model | FHIR Training | 3D CT Support | VRAM | EHRQA Score |
|-------|--------------|---------------|------|-------------|
| MedGemma 4B-IT | Limited | No | ~12-16GB | ~85% |
| **MedGemma 27B text-only** | **NO** | No | ~54GB | 86.3% |
| MedGemma 27B multimodal | **YES** (Synthea RL) | Yes | ~54GB | **90.5%** |
| MedGemma 1.5 4B | Some | **YES** | ~12-16GB | 89.6% |

**Critical:** The 27B text-only model does NOT have FHIR training. Only the 27B multimodal variant underwent the Synthea RL training that enables 90.5% EHRQA accuracy.

---

## 2. Model Capabilities Matrix

### 2.1 Detailed Model Comparison

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    MedGemma Model Family Capabilities                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MedGemma 4B-IT (Current Sentinel-X)                                       │
│  ─────────────────────────────────────                                     │
│  ✓ Basic medical imaging                                                    │
│  ✓ General clinical text understanding                                      │
│  ✗ No FHIR-specific RL training                                            │
│  ✗ No 3D volume support                                                     │
│  ✗ Limited temporal reasoning                                               │
│                                                                             │
│  MedGemma 27B Text-Only                                                     │
│  ────────────────────────                                                   │
│  ✓ Larger parameter count                                                   │
│  ✓ Better general reasoning                                                 │
│  ✗ NO FHIR training (86.3% EHRQA only)                                     │
│  ✗ No vision capabilities                                                   │
│  ✗ Cannot process images                                                    │
│                                                                             │
│  MedGemma 27B Multimodal                                                    │
│  ─────────────────────────                                                  │
│  ✓ Full Synthea RL training (90.5% EHRQA)                                  │
│  ✓ Multi-hop temporal reasoning                                             │
│  ✓ Vision + text capabilities                                               │
│  ✓ FHIR bundle understanding                                                │
│  ✗ High VRAM requirement (~54GB)                                           │
│  ✗ Expensive to host                                                        │
│                                                                             │
│  MedGemma 1.5 4B                                                            │
│  ─────────────────                                                          │
│  ✓ 3D volume support (video adapter)                                        │
│  ✓ Good EHRQA performance (89.6%)                                          │
│  ✓ Affordable VRAM (~12-16GB)                                              │
│  ✓ Spatial continuity for CT                                                │
│  ~ Limited compared to 27B multimodal                                       │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 EHRQA Benchmark Performance

The EHRQA (Electronic Health Record Question Answering) benchmark measures multi-hop reasoning over FHIR data:

| Model | EHRQA Score | Training Method |
|-------|-------------|-----------------|
| MedGemma 4B-IT | ~85% | Supervised fine-tuning |
| MedGemma 27B text-only | 86.3% | SFT only |
| MedGemma 27B multimodal | **90.5%** | **Synthea RL** |
| MedGemma 1.5 4B | 89.6% | Enhanced RL |

**Key insight:** The 4.2% improvement from 86.3% to 90.5% comes specifically from reinforcement learning on Synthea FHIR bundles. The text-only variant lacks this training.

### 2.3 Training Data Alignment

MedGemma's EHRQA training used **Synthea synthetic FHIR records** - the same format Sentinel-X uses:

```
Training Data Format                    Sentinel-X Data Format
─────────────────────                   ──────────────────────
Synthea FHIR R4 bundles                 Synthea FHIR R4 bundles
  ├── Patient resources                   ├── Patient resources
  ├── Condition resources                 ├── Condition resources (50+)
  ├── MedicationRequest                   ├── MedicationRequest (10+)
  ├── Observation resources               ├── Observation resources (259+)
  ├── Encounter resources                 ├── Encounter resources (70+)
  └── Procedure resources                 └── Procedure resources (153+)

200 questions/patient                   CT triage queries
10 categories                           Temporal correlation needs
42 question types                       Multi-hop reasoning required
```

**This is excellent news:** MedGemma is trained on data structurally identical to what Sentinel-X has available.

---

## 3. How MedGemma Handles FHIR Data

### 3.1 EHRQA Training Methodology

MedGemma's FHIR capabilities come from a sophisticated training process:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EHRQA Training Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: Synthea FHIR Generation                                            │
│  ───────────────────────────────────                                        │
│  • Generate synthetic patient records                                        │
│  • Full FHIR R4 compliance                                                   │
│  • Realistic disease progressions                                            │
│  • Temporal relationships preserved                                          │
│                                                                              │
│  Step 2: Question Generation (200 per patient)                              │
│  ──────────────────────────────────────────────                             │
│  • 10 categories (demographics, conditions, meds, labs, etc.)               │
│  • 42 question types                                                         │
│  • Multi-hop temporal queries emphasized                                     │
│  • Ground truth answers extracted programmatically                           │
│                                                                              │
│  Step 3: Reinforcement Learning                                             │
│  ────────────────────────────────                                           │
│  • Model attempts to answer questions                                        │
│  • Reward signal based on correctness                                        │
│  • Emphasis on "inter-dependent records" reasoning                           │
│  • Iterative improvement on temporal queries                                 │
│                                                                              │
│  Result: Model learns to:                                                    │
│  • Navigate FHIR resource relationships                                      │
│  • Perform temporal reasoning across records                                 │
│  • Answer complex multi-hop queries                                          │
│  • Correlate medications with outcomes                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 EHRQA Question Categories

The 10 categories and example question types:

| Category | Example Question | FHIR Resources Required |
|----------|-----------------|------------------------|
| Demographics | "What is the patient's age?" | Patient |
| Conditions | "When was diabetes diagnosed?" | Condition.onsetDateTime |
| Medications | "What medications started after 2023?" | MedicationRequest.authoredOn |
| Labs | "What was the HbA1c after starting Metformin?" | Observation, MedicationRequest |
| Vitals | "Did blood pressure improve over time?" | Observation (vital-signs) |
| Encounters | "How many ED visits in the past year?" | Encounter.type, Encounter.period |
| Procedures | "What surgeries has the patient had?" | Procedure.code, Procedure.performedDateTime |
| Allergies | "Is the patient allergic to penicillin?" | AllergyIntolerance |
| Immunizations | "Is COVID vaccination up to date?" | Immunization |
| Care Plans | "What is the current treatment plan?" | CarePlan |

### 3.3 Multi-Hop Reasoning Examples

MedGemma's RL training specifically targets multi-hop temporal queries:

```
Multi-Hop Query Type 1: Temporal Correlation
────────────────────────────────────────────
Question: "What was the pain score AFTER the medication date?"

Required Reasoning:
1. Find MedicationRequest.authoredOn for relevant medication
2. Find Observation resources with code = "pain score"
3. Filter observations where effectiveDateTime > medication date
4. Extract and report the value

Multi-Hop Query Type 2: Causal Chain
────────────────────────────────────
Question: "Did HbA1c improve after starting Metformin?"

Required Reasoning:
1. Find MedicationRequest where medication = "Metformin"
2. Extract authoredOn date
3. Find all HbA1c Observations
4. Compare values before and after medication start
5. Determine if improvement occurred

Multi-Hop Query Type 3: Condition Progression
────────────────────────────────────────────
Question: "What conditions developed after the cardiac event?"

Required Reasoning:
1. Find Condition resource for cardiac event
2. Extract onsetDateTime
3. Find all Condition resources
4. Filter conditions where onsetDateTime > cardiac event date
5. List new diagnoses
```

### 3.4 EHR Navigator Agent Architecture

Google's official approach uses a **LangGraph-based agent** with 4-step workflow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EHR Navigator Agent (from Google notebooks)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                            │
│  │  User Query │                                                            │
│  │  "What was  │                                                            │
│  │  the last   │                                                            │
│  │  HbA1c?"    │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │ Step 1: DISCOVER                                              │           │
│  │ ───────────────                                               │           │
│  │ Tool: get_patient_data_manifest()                            │           │
│  │ Returns: Available FHIR resource types and counts            │           │
│  │          {Observation: 259, Condition: 50, ...}              │           │
│  └──────────────────────────────────────────────────────────────┘           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │ Step 2: IDENTIFY                                              │           │
│  │ ─────────────────                                             │           │
│  │ LLM determines which resources answer the question:           │           │
│  │ "To find HbA1c, I need Observation resources with             │           │
│  │  LOINC code 4548-4"                                          │           │
│  └──────────────────────────────────────────────────────────────┘           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │ Step 3: FETCH (Iterative)                                     │           │
│  │ ─────────────────────────                                     │           │
│  │ Tool: get_resources_by_type(type="Observation",               │           │
│  │                             filter="code=4548-4")            │           │
│  │ Returns: Matching FHIR resources                              │           │
│  │                                                               │           │
│  │ Agent may loop if more context needed                         │           │
│  └──────────────────────────────────────────────────────────────┘           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │ Step 4: PROCESS                                               │           │
│  │ ─────────────────                                             │           │
│  │ LLM synthesizes findings into final answer:                   │           │
│  │ "The last HbA1c was 7.2% on 2024-01-15"                      │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  Key Insight: Full FHIR records exceed context windows, so the              │
│  agent fetches RELEVANT SUBSETS rather than entire bundles                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 EHR Navigator Code Pattern (from Google notebooks)

```python
# Simplified EHR Navigator Agent pattern from official MedGemma notebooks

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    """State maintained across agent steps."""
    query: str
    manifest: Optional[dict]
    identified_resources: List[str]
    fetched_data: List[dict]
    answer: Optional[str]

def discover_resources(state: AgentState) -> AgentState:
    """Step 1: Get available resource types from FHIR bundle."""
    manifest = get_patient_data_manifest(state["patient_id"])
    # Returns: {"Observation": 259, "Condition": 50, "MedicationRequest": 10, ...}
    return {**state, "manifest": manifest}

def identify_relevant(state: AgentState) -> AgentState:
    """Step 2: LLM identifies which resources to fetch."""
    prompt = f"""
    Available FHIR resources: {state['manifest']}
    User query: {state['query']}

    Which resource types do I need to answer this query?
    Return as JSON list.
    """
    identified = llm.generate(prompt)
    # Returns: ["Observation", "MedicationRequest"] for HbA1c query
    return {**state, "identified_resources": identified}

def fetch_resources(state: AgentState) -> AgentState:
    """Step 3: Retrieve relevant FHIR resources."""
    fetched = []
    for resource_type in state["identified_resources"]:
        resources = get_resources_by_type(
            patient_id=state["patient_id"],
            resource_type=resource_type,
            query_context=state["query"]  # For smart filtering
        )
        fetched.extend(resources)
    return {**state, "fetched_data": fetched}

def synthesize_answer(state: AgentState) -> AgentState:
    """Step 4: Generate final answer from fetched data."""
    prompt = f"""
    Query: {state['query']}

    Relevant FHIR data:
    {format_fhir_data(state['fetched_data'])}

    Provide a clear, accurate answer based on this data.
    """
    answer = llm.generate(prompt)
    return {**state, "answer": answer}

# Build the agent graph
workflow = StateGraph(AgentState)
workflow.add_node("discover", discover_resources)
workflow.add_node("identify", identify_relevant)
workflow.add_node("fetch", fetch_resources)
workflow.add_node("synthesize", synthesize_answer)

workflow.add_edge("discover", "identify")
workflow.add_edge("identify", "fetch")
workflow.add_edge("fetch", "synthesize")
workflow.add_edge("synthesize", END)

ehr_navigator = workflow.compile()
```

### 3.6 Why Context Window Management Matters

Full FHIR bundles can be massive:

```
Sentinel-X Sample Patient (train_1_a_1):
────────────────────────────────────────
Total FHIR entries: 926 resources
Estimated token count: ~50,000-100,000 tokens

Breakdown:
├── Observation (259): ~30,000 tokens
├── Procedure (153): ~15,000 tokens
├── DiagnosticReport (112): ~12,000 tokens
├── Encounter (70): ~8,000 tokens
├── Condition (50): ~5,000 tokens
└── Other (282): ~25,000 tokens

Context window limits:
├── MedGemma 4B/27B: 8K tokens
├── MedGemma 1.5: 1M+ tokens (but cost scales)
└── Practical limit: 4-8K for fast inference

Solution: Agent-based retrieval of RELEVANT SUBSETS
```

---

## 4. How MedGemma Handles 3D CT Volumes

### 4.1 MedGemma 1.5 Video-Style Processing

From Google's `high_dimensional_ct_model_garden.ipynb`, MedGemma 1.5 treats CT volumes as video sequences:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    3D CT Volume Processing (MedGemma 1.5)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: 3D NIfTI/DICOM Volume                                               │
│  ────────────────────────────                                               │
│                                                                              │
│  512 x 512 x 300 voxels                                                      │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────┐                                    │
│  │ Slice Sampling (MAX_SLICE = 85)     │                                    │
│  │                                      │                                    │
│  │ if len(slices) > 85:                │                                    │
│  │   # Uniform sampling                 │                                    │
│  │   indices = [int(round(i/85 *       │                                    │
│  │              (len(slices)-1)))       │                                    │
│  │              for i in range(1, 86)]  │                                    │
│  │                                      │                                    │
│  └─────────────────────────────────────┘                                    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────┐                                    │
│  │ Video Adapter Encoding               │                                    │
│  │                                      │                                    │
│  │ 85 slices treated as video frames   │                                    │
│  │ Temporal encoder captures:           │                                    │
│  │ • Inter-slice relationships          │                                    │
│  │ • Spatial continuity                 │                                    │
│  │ • 3D structure information           │                                    │
│  │                                      │                                    │
│  └─────────────────────────────────────┘                                    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────┐                                    │
│  │ Multimodal Fusion                    │                                    │
│  │                                      │                                    │
│  │ 3D visual features + text prompt    │                                    │
│  │ Unified representation for LLM      │                                    │
│  │                                      │                                    │
│  └─────────────────────────────────────┘                                    │
│                                                                              │
│  Output: Spatially-aware analysis                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Official Code Reference (from Google notebook)

```python
# From Google's high_dimensional_ct_model_garden.ipynb

MAX_SLICE = 85  # Maximum slices for MedGemma 1.5

def load_and_sample_ct_volume(dicom_path: str) -> List[np.ndarray]:
    """Load CT volume and sample to MAX_SLICE frames."""

    # Load all DICOM instances
    dicom_instances = load_dicom_series(dicom_path)

    # Sample if exceeds MAX_SLICE
    if len(dicom_instances) > MAX_SLICE:
        # Uniform sampling to get exactly MAX_SLICE slices
        dicom_instances = [
            dicom_instances[int(round(i / MAX_SLICE * (len(dicom_instances) - 1)))]
            for i in range(1, MAX_SLICE + 1)
        ]

    return dicom_instances

def prepare_ct_for_medgemma(volume_slices: List[np.ndarray]) -> dict:
    """Prepare CT volume for MedGemma 1.5 input."""

    # Convert slices to video format
    # MedGemma 1.5 accepts video as sequential frames
    video_frames = []
    for slice_data in volume_slices:
        # Apply windowing
        windowed = apply_ct_window(slice_data, center=40, width=400)

        # Convert to RGB
        rgb_frame = np.stack([windowed] * 3, axis=-1)

        video_frames.append(rgb_frame)

    return {
        "video": video_frames,
        "num_frames": len(video_frames)
    }
```

**Key observation:** Our Sentinel-X implementation already uses `CT_NUM_SLICES = 85`, which matches Google's official `MAX_SLICE = 85`.

### 4.3 Comparison: 2D Stack vs 3D Video Processing

| Aspect | Current (2D Stack) | MedGemma 1.5 (3D Video) |
|--------|-------------------|-------------------------|
| Inter-slice context | None | Preserved |
| Lesion continuity | Lost | Tracked |
| Vessel following | Impossible | Enabled |
| 3D volume estimation | Slice-by-slice | Native |
| PE detection | Fragmented | Continuous |
| Model architecture | Per-image | Temporal encoder |
| VRAM usage | Similar | Similar |

### 4.4 Advantages for CT Triage

```
3D-Aware Analysis Capabilities
─────────────────────────────

1. Pulmonary Embolism Detection
   ┌────────────────────────────────────────────┐
   │ 2D Approach:                               │
   │ Slice 45: "filling defect"                 │
   │ Slice 46: "filling defect"                 │
   │ Slice 47: "filling defect"                 │
   │ → Three separate findings                   │
   │                                            │
   │ 3D Approach:                               │
   │ "Continuous filling defect in pulmonary    │
   │ artery, spanning slices 45-47, consistent  │
   │ with pulmonary embolism"                   │
   └────────────────────────────────────────────┘

2. Nodule Characterization
   ┌────────────────────────────────────────────┐
   │ 2D Approach:                               │
   │ Cannot assess true 3D morphology           │
   │ Size estimation from single slice          │
   │                                            │
   │ 3D Approach:                               │
   │ "Spiculated nodule with 3D volume of       │
   │ 450mm³, irregular margins visible across   │
   │ z-axis"                                    │
   └────────────────────────────────────────────┘

3. Lymphadenopathy Assessment
   ┌────────────────────────────────────────────┐
   │ 2D Approach:                               │
   │ Multiple "enlarged nodes" counted per slice│
   │ May double-count same node                 │
   │                                            │
   │ 3D Approach:                               │
   │ "Three distinct enlarged mediastinal nodes │
   │ identified, largest 2.1cm in short axis"  │
   └────────────────────────────────────────────┘
```

---

## 5. Architecture Options Analysis

### 5.1 Option 1: Two-Model Pipeline (27B → 4B)

**Concept:** Use 27B multimodal for FHIR context extraction, then 4B for imaging analysis.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Option 1: Two-Model Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐                                                         │
│  │ FHIR Bundle    │                                                         │
│  │ (926 resources)│                                                         │
│  └───────┬────────┘                                                         │
│          │                                                                   │
│          ▼                                                                   │
│  ┌────────────────────────────────────────────┐                             │
│  │         MedGemma 27B Multimodal            │                             │
│  │         ────────────────────────           │                             │
│  │  • Full FHIR training (90.5% EHRQA)        │                             │
│  │  • Multi-hop temporal reasoning            │                             │
│  │  • Synthea RL capabilities                 │                             │
│  │                                            │                             │
│  │  Task: Extract relevant clinical context   │                             │
│  │        for CT triage decision              │                             │
│  │                                            │                             │
│  │  VRAM: ~54GB (A100 80GB or H100)          │                             │
│  └─────────────────────┬──────────────────────┘                             │
│                        │                                                     │
│                        ▼                                                     │
│              ┌─────────────────────┐                                        │
│              │ Extracted Context   │                                        │
│              │ (Structured, <4K    │                                        │
│              │  tokens)            │                                        │
│              └──────────┬──────────┘                                        │
│                         │                                                    │
│  ┌────────────────┐     │                                                   │
│  │ CT Volume      │     │                                                   │
│  │ (85 slices)    ├─────┼─────┐                                             │
│  └────────────────┘     │     │                                             │
│                         │     │                                             │
│                         ▼     ▼                                             │
│  ┌────────────────────────────────────────────┐                             │
│  │         MedGemma 1.5 4B / 4B-IT            │                             │
│  │         ──────────────────────────         │                             │
│  │  • Image analysis (+ 3D if 1.5)            │                             │
│  │  • Receives pre-extracted context          │                             │
│  │  • Triage decision making                  │                             │
│  │                                            │                             │
│  │  VRAM: ~12-16GB (RTX 4090)                │                             │
│  └─────────────────────┬──────────────────────┘                             │
│                        │                                                     │
│                        ▼                                                     │
│              ┌─────────────────────┐                                        │
│              │ Priority Assignment │                                        │
│              │ + Rationale         │                                        │
│              └─────────────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Best FHIR understanding (90.5% EHRQA)
- Specialized models for each task
- 4B is cheaper for high-volume imaging

**Disadvantages:**
- Complex orchestration (two models)
- 27B multimodal can do BOTH tasks (redundant?)
- Dual GPU cost (~$1.89 + $0.44/hr on RunPod)
- Additional latency from two inferences

**Cost Analysis:**
```
Per-case inference cost (RunPod):
- 27B on A100 80GB: ~$1.89/hr, ~10 seconds = $0.005/case
- 4B on RTX 4090: ~$0.44/hr, ~5 seconds = $0.0006/case
- Total: ~$0.0056/case

Monthly (1000 cases/day):
- GPU hours: ~4.2hr/day × 30 = 126hr
- Cost: ~$168/month
```

### 5.2 Option 2: Single 27B Multimodal

**Concept:** Use 27B multimodal for both FHIR processing and CT analysis.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Option 2: Single 27B Multimodal                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐     ┌────────────────┐                                  │
│  │ FHIR Bundle    │     │ CT Volume      │                                  │
│  │ (926 resources)│     │ (85 slices)    │                                  │
│  └───────┬────────┘     └───────┬────────┘                                  │
│          │                      │                                            │
│          │    ┌─────────────────┘                                           │
│          │    │                                                              │
│          ▼    ▼                                                              │
│  ┌────────────────────────────────────────────┐                             │
│  │         MedGemma 27B Multimodal            │                             │
│  │         ────────────────────────           │                             │
│  │                                            │                             │
│  │  Capabilities:                             │                             │
│  │  ✓ Full FHIR training (90.5% EHRQA)       │                             │
│  │  ✓ Multi-hop temporal reasoning            │                             │
│  │  ✓ Vision for CT analysis                  │                             │
│  │  ✓ Multimodal fusion                       │                             │
│  │                                            │                             │
│  │  Single unified analysis                   │                             │
│  │                                            │                             │
│  │  VRAM: ~54GB (A100 80GB or H100)          │                             │
│  └─────────────────────┬──────────────────────┘                             │
│                        │                                                     │
│                        ▼                                                     │
│              ┌─────────────────────┐                                        │
│              │ Priority Assignment │                                        │
│              │ + Rationale         │                                        │
│              └─────────────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Simplest architecture
- Best possible FHIR + vision combination
- Single inference, lower latency
- Native multimodal reasoning

**Disadvantages:**
- ~54GB VRAM requirement
- Expensive to host (~$1.89-2.65/hr)
- May be overkill for simple cases
- No fallback if GPU unavailable

**Cost Analysis:**
```
Per-case inference cost (RunPod):
- A100 80GB: ~$1.89/hr, ~15 seconds = $0.008/case
- H100 80GB: ~$2.65/hr, ~10 seconds = $0.007/case

Monthly (1000 cases/day):
- GPU hours: ~4.2hr/day × 30 = 126hr
- Cost: $238-334/month
```

### 5.3 Option 3: EHR Navigator Agent + 4B

**Concept:** Use LangGraph agent for smart FHIR retrieval, then 4B for analysis.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Option 3: EHR Navigator Agent + 4B                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐                                                         │
│  │ FHIR Bundle    │                                                         │
│  │ (926 resources)│                                                         │
│  └───────┬────────┘                                                         │
│          │                                                                   │
│          ▼                                                                   │
│  ┌────────────────────────────────────────────┐                             │
│  │       EHR Navigator Agent (LangGraph)      │                             │
│  │       ─────────────────────────────────    │                             │
│  │                                            │                             │
│  │  Uses MedGemma 4B for agent reasoning:     │                             │
│  │                                            │                             │
│  │  Step 1: DISCOVER                          │                             │
│  │  ┌─────────────────────────────────┐      │                             │
│  │  │ get_patient_data_manifest()     │      │                             │
│  │  │ → {Observation: 259, ...}       │      │                             │
│  │  └─────────────────────────────────┘      │                             │
│  │           │                                │                             │
│  │           ▼                                │                             │
│  │  Step 2: IDENTIFY                          │                             │
│  │  ┌─────────────────────────────────┐      │                             │
│  │  │ "For CT triage, I need recent   │      │                             │
│  │  │  labs, active conditions, and   │      │                             │
│  │  │  current medications"           │      │                             │
│  │  └─────────────────────────────────┘      │                             │
│  │           │                                │                             │
│  │           ▼                                │                             │
│  │  Step 3: FETCH (Iterative)                 │                             │
│  │  ┌─────────────────────────────────┐      │                             │
│  │  │ Retrieves only relevant:        │      │                             │
│  │  │ - Cardiac markers (if chest CT) │      │                             │
│  │  │ - D-dimer (if PE suspected)     │      │                             │
│  │  │ - Active conditions w/ dates    │      │                             │
│  │  └─────────────────────────────────┘      │                             │
│  │           │                                │                             │
│  │  VRAM: ~12GB for agent LLM                │                             │
│  └───────────┬────────────────────────────────┘                             │
│              │                                                               │
│              ▼                                                               │
│    ┌─────────────────────┐                                                  │
│    │ Extracted Context   │                                                  │
│    │ (Smart subset,      │                                                  │
│    │  <2K tokens)        │                                                  │
│    └──────────┬──────────┘                                                  │
│               │                                                              │
│  ┌────────────┴───┐                                                         │
│  │ CT Volume      │                                                         │
│  │ (85 slices)    │                                                         │
│  └───────┬────────┘                                                         │
│          │                                                                   │
│          ▼                                                                   │
│  ┌────────────────────────────────────────────┐                             │
│  │         MedGemma 1.5 4B / 4B-IT            │                             │
│  │         ──────────────────────────         │                             │
│  │  • CT analysis with smart context          │                             │
│  │  • 3D support if using 1.5                 │                             │
│  │  • Triage decision                         │                             │
│  │                                            │                             │
│  │  VRAM: Same 12-16GB (reuses GPU)          │                             │
│  └─────────────────────┬──────────────────────┘                             │
│                        │                                                     │
│                        ▼                                                     │
│              ┌─────────────────────┐                                        │
│              │ Priority Assignment │                                        │
│              │ + Rationale         │                                        │
│              └─────────────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Smart FHIR subset extraction (context window friendly)
- Runs on single affordable GPU
- Can use same model for agent and analysis
- Follows Google's recommended pattern

**Disadvantages:**
- Multiple LLM calls for navigation (2-4 calls)
- 4B has limited FHIR training (89.6% vs 90.5%)
- Agent complexity (LangGraph setup)
- May miss context if agent makes wrong choices

**Cost Analysis:**
```
Per-case inference cost (RunPod RTX 4090):
- Agent calls: 3 × ~2 seconds = 6 seconds
- Analysis call: ~5 seconds
- Total: ~11 seconds @ $0.44/hr = $0.0013/case

Monthly (1000 cases/day):
- GPU hours: ~3hr/day × 30 = 90hr
- Cost: ~$40/month
```

### 5.4 Option 4: Staged 4B with Enhanced Prompting

**Concept:** Use single 4B model for both FHIR extraction and CT analysis in two stages.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Option 4: Staged 4B with Enhanced Prompting               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐                                                         │
│  │ FHIR Bundle    │                                                         │
│  │ (Pre-filtered  │                                                         │
│  │  by code)      │                                                         │
│  └───────┬────────┘                                                         │
│          │                                                                   │
│          ▼                                                                   │
│  ┌────────────────────────────────────────────┐                             │
│  │         Stage 1: Context Extraction        │                             │
│  │         MedGemma 1.5 4B / 4B-IT            │                             │
│  │         ──────────────────────────         │                             │
│  │                                            │                             │
│  │  Prompt: "Extract clinical context         │                             │
│  │  relevant for CT triage from this          │                             │
│  │  FHIR data. Focus on:                      │                             │
│  │  - Recent labs (cardiac, coag, renal)      │                             │
│  │  - Active conditions with dates            │                             │
│  │  - Current medications                     │                             │
│  │  - Recent encounters"                      │                             │
│  │                                            │                             │
│  │  VRAM: ~12-16GB                           │                             │
│  └─────────────────────┬──────────────────────┘                             │
│                        │                                                     │
│                        ▼                                                     │
│              ┌─────────────────────┐                                        │
│              │ Extracted Context   │                                        │
│              │ (Structured text,   │                                        │
│              │  <2K tokens)        │                                        │
│              └──────────┬──────────┘                                        │
│                         │                                                    │
│  ┌────────────────┐     │                                                   │
│  │ CT Volume      │     │                                                   │
│  │ (85 slices)    ├─────┘                                                   │
│  └───────┬────────┘                                                         │
│          │                                                                   │
│          ▼                                                                   │
│  ┌────────────────────────────────────────────┐                             │
│  │         Stage 2: Triage Analysis           │                             │
│  │         MedGemma 1.5 4B / 4B-IT            │                             │
│  │         ──────────────────────────         │                             │
│  │                                            │                             │
│  │  Prompt: Multi-hop triage prompt with:     │                             │
│  │  - Extracted clinical context              │                             │
│  │  - 85 CT slices                            │                             │
│  │  - Temporal reasoning instructions         │                             │
│  │  - Lab correlation guidance                │                             │
│  │                                            │                             │
│  │  VRAM: Same GPU (sequential)              │                             │
│  └─────────────────────┬──────────────────────┘                             │
│                        │                                                     │
│                        ▼                                                     │
│              ┌─────────────────────┐                                        │
│              │ Priority Assignment │                                        │
│              │ + Rationale         │                                        │
│              └─────────────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Simplest implementation
- Single model type
- Affordable GPU (RTX 4090)
- Can incrementally improve prompts

**Disadvantages:**
- 4B has limited FHIR training
- Two sequential inferences
- Relies heavily on prompt engineering
- May miss complex multi-hop patterns

**Cost Analysis:**
```
Per-case inference cost (RunPod RTX 4090):
- Stage 1: ~3 seconds
- Stage 2: ~5 seconds
- Total: ~8 seconds @ $0.44/hr = $0.001/case

Monthly (1000 cases/day):
- GPU hours: ~2.2hr/day × 30 = 66hr
- Cost: ~$29/month
```

### 5.5 Architecture Comparison Matrix

| Aspect | Option 1 (27B→4B) | Option 2 (27B) | Option 3 (Agent+4B) | Option 4 (Staged 4B) |
|--------|-------------------|----------------|---------------------|---------------------|
| FHIR Accuracy | 90.5% | 90.5% | 89.6% | ~85-89% |
| 3D CT Support | If using 1.5 | Yes | If using 1.5 | If using 1.5 |
| Complexity | High | Low | Medium | Low |
| Min VRAM | ~54GB + 16GB | ~54GB | ~16GB | ~16GB |
| Monthly Cost | ~$168 | ~$238-334 | ~$40 | ~$29 |
| Latency | ~15s | ~15s | ~11s | ~8s |
| Implementation | Hard | Medium | Medium | Easy |

### 5.6 Recommendation for Sentinel-X

**Recommended Path: Phased Implementation**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Recommended Phased Approach                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: NOW (Option 4 Enhanced)                                           │
│  ─────────────────────────────────                                          │
│  • Enhance current FHIR extraction in fhir_context.py                       │
│  • Add temporal data (onsetDateTime, authoredOn, effectiveDateTime)         │
│  • Add Observation values (cardiac markers, coagulation, vitals)            │
│  • Update prompts with multi-hop instructions                               │
│  • Continue using MedGemma 4B-IT                                            │
│  • Estimate: 1-2 weeks implementation                                       │
│                                                                              │
│  Phase 2: NEXT (Option 3 Partial)                                           │
│  ────────────────────────────────                                           │
│  • Implement EHR Navigator pattern without full LangGraph                   │
│  • Smart FHIR filtering based on CT type and findings                       │
│  • Pre-filter relevant observations by LOINC code                           │
│  • Evaluate MedGemma 1.5 4B for 3D support                                  │
│  • Estimate: 2-3 weeks implementation                                       │
│                                                                              │
│  Phase 3: EVALUATE (Option 2 Assessment)                                    │
│  ────────────────────────────────────────                                   │
│  • Run comparative benchmarks: 4B vs 27B multimodal                         │
│  • Measure accuracy on multi-hop temporal queries                           │
│  • Calculate cost/benefit for production deployment                         │
│  • Decision point: Is 27B worth 10x cost for ~1-5% accuracy gain?           │
│  • Estimate: 2 weeks evaluation                                             │
│                                                                              │
│  Phase 4: SCALE (Based on Phase 3 results)                                  │
│  ─────────────────────────────────────────                                  │
│  • If 27B needed: Deploy Option 2 for complex cases                         │
│  • If 4B sufficient: Optimize Option 3/4 for production                     │
│  • Consider tiered approach: 4B for simple, 27B for complex                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. GPU & Hosting Analysis

### 6.1 Model VRAM Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VRAM Requirements by Model & Precision                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MedGemma 4B-IT                                                             │
│  ─────────────────                                                          │
│  FP16 (bfloat16): ~8-10GB                                                   │
│  INT8 quantized:  ~4-5GB                                                    │
│  INT4 quantized:  ~2-3GB                                                    │
│  + KV cache:      ~2-4GB (depends on context)                               │
│  + Images:        ~2-4GB (for 85 slices)                                    │
│  ────────────────────────                                                   │
│  Recommended:     12-16GB GPU                                               │
│                                                                              │
│  MedGemma 27B Multimodal                                                    │
│  ────────────────────────                                                   │
│  FP16 (bfloat16): ~54GB                                                     │
│  INT8 quantized:  ~27GB                                                     │
│  INT4 quantized:  ~14GB (quality concerns)                                  │
│  + KV cache:      ~4-8GB                                                    │
│  + Images:        ~2-4GB                                                    │
│  ────────────────────────                                                   │
│  Recommended:     80GB GPU (A100/H100)                                      │
│  Minimum:         40GB with INT8 (A100 40GB)                               │
│                                                                              │
│  MedGemma 1.5 4B                                                            │
│  ─────────────────                                                          │
│  Similar to 4B-IT                                                           │
│  + Video encoder overhead: ~1-2GB                                           │
│  ────────────────────────                                                   │
│  Recommended:     16GB GPU                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 RunPod GPU Options

| GPU | VRAM | Price/hr | Best For |
|-----|------|----------|----------|
| RTX 3090 | 24GB | ~$0.22 | 4B development |
| RTX 4090 | 24GB | ~$0.44 | 4B production |
| A40 | 48GB | ~$0.79 | 27B INT8 (tight) |
| A100 40GB | 40GB | ~$1.29 | 27B INT8 |
| A100 80GB | 80GB | ~$1.89 | 27B FP16 |
| H100 80GB | 80GB | ~$2.65 | 27B fast inference |
| H100 SXM | 80GB | ~$3.99 | Maximum performance |

### 6.3 Cost Projections

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Monthly Cost Projections by Volume                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Assumptions:                                                                │
│  - 4B inference: ~5-8 seconds/case                                          │
│  - 27B inference: ~10-15 seconds/case                                       │
│  - GPU only when processing (serverless)                                    │
│                                                                              │
│  Daily Volume: 100 cases                                                     │
│  ─────────────────────────                                                  │
│  4B on RTX 4090:  ~0.2 hr/day × 30 × $0.44 = $2.64/month                   │
│  27B on A100 80GB: ~0.4 hr/day × 30 × $1.89 = $22.68/month                 │
│                                                                              │
│  Daily Volume: 500 cases                                                     │
│  ─────────────────────────                                                  │
│  4B on RTX 4090:  ~1.1 hr/day × 30 × $0.44 = $14.52/month                  │
│  27B on A100 80GB: ~2.1 hr/day × 30 × $1.89 = $119.07/month                │
│                                                                              │
│  Daily Volume: 1000 cases                                                    │
│  ──────────────────────────                                                 │
│  4B on RTX 4090:  ~2.2 hr/day × 30 × $0.44 = $29.04/month                  │
│  27B on A100 80GB: ~4.2 hr/day × 30 × $1.89 = $238.14/month                │
│                                                                              │
│  Daily Volume: 5000 cases                                                    │
│  ──────────────────────────                                                 │
│  4B on RTX 4090:  ~11 hr/day × 30 × $0.44 = $145.20/month                  │
│  27B on A100 80GB: ~21 hr/day × 30 × $1.89 = $1,190.70/month               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Deployment Architectures

```
Option A: Serverless (RunPod Serverless)
────────────────────────────────────────
┌─────────────────────────────────────────┐
│           RunPod Serverless             │
│  ┌─────────────────────────────────┐   │
│  │  Cold start: ~30-60s            │   │
│  │  Scale to zero when idle        │   │
│  │  Pay per second of compute      │   │
│  │  Good for: Variable load        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘

Pros: No idle cost, auto-scaling
Cons: Cold start latency, less control


Option B: Reserved Pod (RunPod Pods)
────────────────────────────────────
┌─────────────────────────────────────────┐
│           RunPod Reserved Pod           │
│  ┌─────────────────────────────────┐   │
│  │  Always running                 │   │
│  │  Instant response               │   │
│  │  Fixed hourly cost              │   │
│  │  Good for: Consistent load      │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘

Pros: Low latency, predictable performance
Cons: Pay even when idle


Option C: Hybrid (Recommended for Production)
─────────────────────────────────────────────
┌─────────────────────────────────────────────┐
│              Hybrid Architecture            │
│                                             │
│  ┌─────────────┐    ┌─────────────────┐    │
│  │ Reserved 4B │    │ Serverless 27B  │    │
│  │ (always on) │    │ (on-demand)     │    │
│  └──────┬──────┘    └────────┬────────┘    │
│         │                    │              │
│         ▼                    ▼              │
│  ┌──────────────────────────────────────┐  │
│  │         Request Router               │  │
│  │  Simple cases → 4B                   │  │
│  │  Complex cases → 27B                 │  │
│  └──────────────────────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘

Pros: Cost-optimized, best of both worlds
Cons: Routing logic needed
```

### 6.5 Quantization Trade-offs

| Precision | VRAM Reduction | Quality Impact | Recommended For |
|-----------|---------------|----------------|-----------------|
| FP16/BF16 | Baseline | None | Production (if VRAM available) |
| INT8 (dynamic) | ~50% | Minimal (~1%) | Production alternative |
| INT8 (static) | ~50% | Low (~2%) | Cost-sensitive deployment |
| INT4 (GPTQ) | ~75% | Moderate (~5%) | Experimentation only |
| INT4 (AWQ) | ~75% | Low-Moderate (~3%) | Constrained environments |

**Recommendation:** Use INT8 quantization for 27B on A100 40GB to reduce costs while maintaining quality.

---

## 7. Implementation Approach

### 7.1 Phase 1: Enhanced FHIR Context Extraction

**Goal:** Maximize value from current 4B model with better FHIR data.

#### 7.1.1 Enhanced Data Models

```python
# triage/fhir_models.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

@dataclass
class ConditionRecord:
    """Rich condition data preserving temporal information."""
    code: str
    display: str
    system: str  # SNOMED, ICD-10, etc.
    status: str  # active, resolved, remission
    onset_date: Optional[datetime] = None
    abatement_date: Optional[datetime] = None
    severity: Optional[str] = None

    def duration_years(self) -> Optional[float]:
        """Calculate condition duration in years."""
        if not self.onset_date:
            return None
        end = self.abatement_date or datetime.now()
        return (end - self.onset_date).days / 365.25

@dataclass
class ObservationRecord:
    """Clinical observation with value and timing."""
    code: str
    display: str
    value: Optional[float] = None
    unit: str = ""
    effective_date: Optional[datetime] = None
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    interpretation: Optional[str] = None  # normal, high, low, critical

    def is_abnormal(self) -> bool:
        """Check if value is outside reference range."""
        if self.value is None:
            return False
        if self.reference_high and self.value > self.reference_high:
            return True
        if self.reference_low and self.value < self.reference_low:
            return True
        return False

    def abnormality_description(self) -> str:
        """Describe the abnormality if present."""
        if not self.is_abnormal():
            return "normal"
        if self.reference_high and self.value > self.reference_high:
            pct = ((self.value - self.reference_high) / self.reference_high) * 100
            return f"HIGH (+{pct:.0f}%)"
        if self.reference_low and self.value < self.reference_low:
            pct = ((self.reference_low - self.value) / self.reference_low) * 100
            return f"LOW (-{pct:.0f}%)"
        return "abnormal"

@dataclass
class MedicationRecord:
    """Medication with temporal data."""
    code: str
    display: str
    status: str  # active, completed, stopped
    authored_on: Optional[datetime] = None
    dosage: Optional[str] = None
    route: Optional[str] = None

    def is_active(self) -> bool:
        return self.status == "active"

@dataclass
class EncounterRecord:
    """Healthcare encounter with context."""
    encounter_type: str  # emergency, inpatient, outpatient
    reason: Optional[str] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def is_recent(self, days: int = 7) -> bool:
        """Check if encounter is within specified days."""
        if not self.period_start:
            return False
        return (datetime.now() - self.period_start).days <= days

@dataclass
class EnhancedPatientContext:
    """Full patient context with temporal data."""
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None

    conditions: List[ConditionRecord] = field(default_factory=list)
    observations: List[ObservationRecord] = field(default_factory=list)
    medications: List[MedicationRecord] = field(default_factory=list)
    encounters: List[EncounterRecord] = field(default_factory=list)

    # Report content
    findings: str = ""
    impressions: str = ""
```

#### 7.1.2 LOINC Code Registry for CT Triage

```python
# triage/clinical_codes.py

"""LOINC codes for clinically relevant observations in CT triage."""

CARDIAC_MARKERS = {
    "10839-9": "Troponin I.cardiac",
    "6598-7": "Troponin T.cardiac",
    "49563-0": "Troponin I.cardiac [Mass/volume] - High sensitivity",
    "89579-7": "Troponin I.cardiac [Mass/volume] - Ultra sensitive",
    "30934-4": "Natriuretic peptide B",
    "33762-6": "NT-proBNP",
    "2157-6": "Creatine kinase.MB",
}

COAGULATION = {
    "48065-7": "Fibrin D-dimer FEU",
    "48066-5": "Fibrin D-dimer DDU",
    "5902-2": "Prothrombin time (PT)",
    "6301-6": "INR",
    "3173-2": "aPTT",
    "3255-7": "Fibrinogen",
}

INFLAMMATORY = {
    "6690-2": "Leukocytes [#/volume] in Blood",
    "26464-8": "Leukocytes [#/volume] in Blood by Automated count",
    "1988-5": "C reactive protein [Mass/volume]",
    "30341-2": "Erythrocyte sedimentation rate",
    "33959-8": "Procalcitonin [Mass/volume]",
}

RENAL = {
    "2160-0": "Creatinine [Mass/volume] in Serum or Plasma",
    "38483-4": "Creatinine [Mass/volume] in Blood",
    "3094-0": "Urea nitrogen [Mass/volume] in Serum or Plasma",
    "33914-3": "eGFR CKD-EPI",
    "48642-3": "eGFR CKD-EPI (non-Black)",
    "48643-1": "eGFR CKD-EPI (Black)",
}

METABOLIC = {
    "4548-4": "Hemoglobin A1c/Hemoglobin.total in Blood",
    "2339-0": "Glucose [Mass/volume] in Blood",
    "2345-7": "Glucose [Mass/volume] in Serum or Plasma",
    "14749-6": "Glucose [Mass/volume] in Serum or Plasma --fasting",
}

VITALS = {
    "8480-6": "Systolic blood pressure",
    "8462-4": "Diastolic blood pressure",
    "8867-4": "Heart rate",
    "9279-1": "Respiratory rate",
    "2708-6": "Oxygen saturation in Arterial blood",
    "59408-5": "Oxygen saturation in Arterial blood by Pulse oximetry",
    "8310-5": "Body temperature",
}

# Combined for filtering
CT_TRIAGE_RELEVANT_CODES = {
    **CARDIAC_MARKERS,
    **COAGULATION,
    **INFLAMMATORY,
    **RENAL,
    **METABOLIC,
    **VITALS,
}

def get_category(loinc_code: str) -> str:
    """Get observation category from LOINC code."""
    if loinc_code in CARDIAC_MARKERS:
        return "cardiac"
    if loinc_code in COAGULATION:
        return "coagulation"
    if loinc_code in INFLAMMATORY:
        return "inflammatory"
    if loinc_code in RENAL:
        return "renal"
    if loinc_code in METABOLIC:
        return "metabolic"
    if loinc_code in VITALS:
        return "vitals"
    return "other"
```

#### 7.1.3 Enhanced FHIR Extraction

```python
# triage/fhir_context_enhanced.py

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .fhir_models import (
    ConditionRecord, ObservationRecord, MedicationRecord,
    EncounterRecord, EnhancedPatientContext
)
from .clinical_codes import CT_TRIAGE_RELEVANT_CODES, get_category

def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse FHIR datetime string."""
    if not dt_str:
        return None
    try:
        # Handle various FHIR datetime formats
        dt_str = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(dt_str)
    except ValueError:
        return None

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
        onset = parse_datetime(resource.get("onsetDateTime"))
        abatement = parse_datetime(resource.get("abatementDateTime"))

        # Extract severity
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

def extract_observations_relevant(
    fhir_bundle: Dict,
    days_back: int = 30
) -> List[ObservationRecord]:
    """Extract clinically relevant observations from recent timeframe."""
    observations = []
    cutoff = datetime.now() - timedelta(days=days_back)

    for entry in fhir_bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue

        # Get code
        code_obj = resource.get("code", {})
        coding = code_obj.get("coding", [{}])[0]
        loinc_code = coding.get("code", "")

        # Filter to relevant codes
        if loinc_code not in CT_TRIAGE_RELEVANT_CODES:
            continue

        # Get effective date
        effective = parse_datetime(resource.get("effectiveDateTime"))

        # Filter to recent observations
        if effective and effective < cutoff:
            continue

        # Get value
        value_qty = resource.get("valueQuantity", {})
        value = value_qty.get("value")
        unit = value_qty.get("unit", "")

        # Get reference range
        ref_range = resource.get("referenceRange", [{}])[0]
        ref_low = ref_range.get("low", {}).get("value")
        ref_high = ref_range.get("high", {}).get("value")

        # Get interpretation
        interp_coding = resource.get("interpretation", [{}])
        if interp_coding:
            interpretation = interp_coding[0].get("coding", [{}])[0].get("code")
        else:
            interpretation = None

        observations.append(ObservationRecord(
            code=loinc_code,
            display=coding.get("display", CT_TRIAGE_RELEVANT_CODES.get(loinc_code, "Unknown")),
            value=value,
            unit=unit,
            effective_date=effective,
            reference_low=ref_low,
            reference_high=ref_high,
            interpretation=interpretation
        ))

    # Sort by date (most recent first)
    observations.sort(key=lambda o: o.effective_date or datetime.min, reverse=True)
    return observations

def extract_medications_rich(fhir_bundle: Dict) -> List[MedicationRecord]:
    """Extract medications with temporal data."""
    medications = []

    for entry in fhir_bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") not in ("MedicationRequest", "MedicationStatement"):
            continue

        # Get medication info
        med_ref = resource.get("medicationCodeableConcept", {})
        coding = med_ref.get("coding", [{}])[0]

        # Get status
        status = resource.get("status", "unknown")

        # Get authored date
        authored = parse_datetime(resource.get("authoredOn"))

        # Get dosage
        dosage_list = resource.get("dosageInstruction", [])
        dosage = None
        if dosage_list:
            dosage = dosage_list[0].get("text")

        medications.append(MedicationRecord(
            code=coding.get("code", ""),
            display=coding.get("display", med_ref.get("text", "Unknown")),
            status=status,
            authored_on=authored,
            dosage=dosage
        ))

    # Sort by date (most recent first)
    medications.sort(key=lambda m: m.authored_on or datetime.min, reverse=True)
    return medications

def extract_encounters_recent(
    fhir_bundle: Dict,
    days_back: int = 90
) -> List[EncounterRecord]:
    """Extract recent healthcare encounters."""
    encounters = []
    cutoff = datetime.now() - timedelta(days=days_back)

    for entry in fhir_bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Encounter":
            continue

        # Get period
        period = resource.get("period", {})
        start = parse_datetime(period.get("start"))
        end = parse_datetime(period.get("end"))

        # Filter to recent
        if start and start < cutoff:
            continue

        # Get type
        type_list = resource.get("type", [])
        enc_type = "unknown"
        if type_list:
            enc_type = type_list[0].get("text") or \
                       type_list[0].get("coding", [{}])[0].get("display", "unknown")

        # Get reason
        reason_list = resource.get("reasonCode", [])
        reason = None
        if reason_list:
            reason = reason_list[0].get("text") or \
                     reason_list[0].get("coding", [{}])[0].get("display")

        encounters.append(EncounterRecord(
            encounter_type=enc_type,
            reason=reason,
            period_start=start,
            period_end=end
        ))

    # Sort by date (most recent first)
    encounters.sort(key=lambda e: e.period_start or datetime.min, reverse=True)
    return encounters
```

#### 7.1.4 Enhanced Context Formatting

```python
# triage/prompt_formatter.py

from datetime import datetime, timedelta
from typing import List
from .fhir_models import (
    EnhancedPatientContext, ConditionRecord, ObservationRecord,
    MedicationRecord, EncounterRecord
)
from .clinical_codes import get_category

def format_enhanced_context(context: EnhancedPatientContext) -> str:
    """Format enhanced patient context for MedGemma prompt."""
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
    lines.append("\n### Medical History (Chronological)")
    active = [c for c in context.conditions if c.status == "active"]
    if active:
        lines.append("| Onset | Condition | Duration |")
        lines.append("|-------|-----------|----------|")
        for c in active[:15]:  # Top 15
            onset_str = c.onset_date.strftime("%Y-%m") if c.onset_date else "Unknown"
            duration = c.duration_years()
            if duration is not None:
                if duration < 1:
                    duration_str = "<1 year"
                else:
                    duration_str = f"{duration:.1f} years"
            else:
                duration_str = "Unknown"
            lines.append(f"| {onset_str} | {c.display} | {duration_str} |")
    else:
        lines.append("No active conditions documented.")

    # Recent labs grouped by category
    lines.append("\n### Recent Lab Values (Last 30 Days)")
    if context.observations:
        # Group by category
        by_category = {}
        for obs in context.observations:
            cat = get_category(obs.code)
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(obs)

        # Format each category
        category_order = ["cardiac", "coagulation", "inflammatory", "renal", "metabolic", "vitals"]
        for cat in category_order:
            if cat not in by_category:
                continue

            cat_display = cat.replace("_", " ").title()
            lines.append(f"\n**{cat_display} Markers:**")
            lines.append("| Date | Test | Value | Status |")
            lines.append("|------|------|-------|--------|")

            for obs in by_category[cat][:5]:  # Top 5 per category
                date_str = obs.effective_date.strftime("%m/%d") if obs.effective_date else "N/A"
                value_str = f"{obs.value:.2f} {obs.unit}" if obs.value else "N/A"
                status = obs.abnormality_description()
                if "HIGH" in status or "LOW" in status:
                    status = f"**{status}**"
                lines.append(f"| {date_str} | {obs.display[:30]} | {value_str} | {status} |")
    else:
        lines.append("No recent lab values available.")

    # Current medications
    lines.append("\n### Current Medications")
    active_meds = [m for m in context.medications if m.is_active()]
    if active_meds:
        lines.append("| Started | Medication | Dosage |")
        lines.append("|---------|------------|--------|")
        for m in active_meds[:10]:
            start_str = m.authored_on.strftime("%Y-%m") if m.authored_on else "Unknown"
            dosage = m.dosage[:30] if m.dosage else "N/A"
            lines.append(f"| {start_str} | {m.display[:30]} | {dosage} |")
    else:
        lines.append("No active medications documented.")

    # Recent encounters
    recent_encounters = [e for e in context.encounters if e.is_recent(days=30)]
    if recent_encounters:
        lines.append("\n### Recent Healthcare Encounters (Last 30 Days)")
        for enc in recent_encounters[:5]:
            date_str = enc.period_start.strftime("%Y-%m-%d") if enc.period_start else "Unknown"
            reason = f" - {enc.reason}" if enc.reason else ""
            lines.append(f"- {date_str}: {enc.encounter_type}{reason}")

    # Report content
    if context.findings:
        lines.append(f"\n### Current Radiology Findings\n{context.findings}")

    if context.impressions:
        lines.append(f"\n### Radiologist Impression\n{context.impressions}")

    return "\n".join(lines)
```

### 7.2 Phase 2: EHR Navigator Pattern (Simplified)

**Goal:** Implement smart FHIR retrieval without full LangGraph complexity.

```python
# triage/ehr_navigator_simple.py

"""Simplified EHR Navigator pattern for smart FHIR context extraction."""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class FHIRManifest:
    """Summary of available FHIR resources."""
    resource_counts: Dict[str, int]
    observation_codes: List[str]
    condition_codes: List[str]
    medication_codes: List[str]
    date_range: tuple  # (earliest, latest)

def get_fhir_manifest(fhir_bundle: Dict) -> FHIRManifest:
    """
    Step 1: DISCOVER - Get summary of available FHIR data.

    This avoids loading full resources into context.
    """
    resource_counts = {}
    observation_codes = set()
    condition_codes = set()
    medication_codes = set()
    dates = []

    for entry in fhir_bundle.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "Unknown")

        resource_counts[rtype] = resource_counts.get(rtype, 0) + 1

        # Collect codes for filtering
        if rtype == "Observation":
            code = resource.get("code", {}).get("coding", [{}])[0].get("code")
            if code:
                observation_codes.add(code)
            dt = resource.get("effectiveDateTime")
            if dt:
                dates.append(dt)

        elif rtype == "Condition":
            code = resource.get("code", {}).get("coding", [{}])[0].get("code")
            if code:
                condition_codes.add(code)
            dt = resource.get("onsetDateTime")
            if dt:
                dates.append(dt)

        elif rtype in ("MedicationRequest", "MedicationStatement"):
            code = resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("code")
            if code:
                medication_codes.add(code)

    # Date range
    date_range = (min(dates) if dates else None, max(dates) if dates else None)

    return FHIRManifest(
        resource_counts=resource_counts,
        observation_codes=list(observation_codes),
        condition_codes=list(condition_codes),
        medication_codes=list(medication_codes),
        date_range=date_range
    )

def identify_relevant_for_ct_triage(
    manifest: FHIRManifest,
    ct_type: str = "chest"
) -> Dict[str, List[str]]:
    """
    Step 2: IDENTIFY - Determine which resources to fetch.

    Based on CT type and available data, select relevant observations.
    """
    from .clinical_codes import (
        CARDIAC_MARKERS, COAGULATION, INFLAMMATORY, RENAL, VITALS
    )

    # Define relevance by CT type
    ct_relevance = {
        "chest": {
            "high_priority": list(CARDIAC_MARKERS.keys()) + list(COAGULATION.keys()),
            "medium_priority": list(INFLAMMATORY.keys()) + list(RENAL.keys()),
            "always": list(VITALS.keys())
        },
        "abdomen": {
            "high_priority": list(RENAL.keys()) + list(INFLAMMATORY.keys()),
            "medium_priority": list(CARDIAC_MARKERS.keys()),
            "always": list(VITALS.keys())
        },
        "head": {
            "high_priority": list(COAGULATION.keys()),
            "medium_priority": list(INFLAMMATORY.keys()),
            "always": list(VITALS.keys())
        }
    }

    relevance = ct_relevance.get(ct_type, ct_relevance["chest"])

    # Filter to available codes
    to_fetch = {
        "observations": [],
        "conditions": "all_active",  # Always get active conditions
        "medications": "all_active",  # Always get active medications
        "encounters": "recent_30_days"
    }

    # Prioritize available observation codes
    available_codes = set(manifest.observation_codes)
    for code in relevance["high_priority"]:
        if code in available_codes:
            to_fetch["observations"].append(code)
    for code in relevance["medium_priority"]:
        if code in available_codes:
            to_fetch["observations"].append(code)
    for code in relevance["always"]:
        if code in available_codes:
            to_fetch["observations"].append(code)

    return to_fetch

def fetch_relevant_fhir(
    fhir_bundle: Dict,
    fetch_spec: Dict[str, List[str]]
) -> Dict:
    """
    Step 3: FETCH - Retrieve only relevant FHIR resources.
    """
    from datetime import datetime, timedelta

    result = {
        "observations": [],
        "conditions": [],
        "medications": [],
        "encounters": []
    }

    recent_cutoff = datetime.now() - timedelta(days=30)

    for entry in fhir_bundle.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType")

        if rtype == "Observation":
            code = resource.get("code", {}).get("coding", [{}])[0].get("code")
            if code in fetch_spec["observations"]:
                result["observations"].append(resource)

        elif rtype == "Condition":
            status = resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code")
            if status == "active":
                result["conditions"].append(resource)

        elif rtype in ("MedicationRequest", "MedicationStatement"):
            status = resource.get("status")
            if status == "active":
                result["medications"].append(resource)

        elif rtype == "Encounter":
            period = resource.get("period", {})
            start_str = period.get("start")
            if start_str:
                try:
                    start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    if start > recent_cutoff:
                        result["encounters"].append(resource)
                except ValueError:
                    pass

    return result

def navigate_ehr_for_ct_triage(
    fhir_bundle: Dict,
    ct_type: str = "chest"
) -> Dict:
    """
    Main entry point: Navigate EHR to extract triage-relevant context.

    Implements the EHR Navigator pattern in a simplified form.
    """
    # Step 1: Discover
    manifest = get_fhir_manifest(fhir_bundle)

    # Step 2: Identify
    fetch_spec = identify_relevant_for_ct_triage(manifest, ct_type)

    # Step 3: Fetch
    relevant = fetch_relevant_fhir(fhir_bundle, fetch_spec)

    # Step 4: Process (convert to context objects)
    from .fhir_context_enhanced import (
        extract_conditions_rich, extract_observations_relevant,
        extract_medications_rich, extract_encounters_recent
    )

    # Process fetched resources
    # (Using the enhanced extractors on the pre-filtered data)

    return {
        "manifest": manifest,
        "fetch_spec": fetch_spec,
        "relevant_resources": relevant,
        "stats": {
            "total_resources": sum(manifest.resource_counts.values()),
            "fetched_observations": len(relevant["observations"]),
            "active_conditions": len(relevant["conditions"]),
            "active_medications": len(relevant["medications"]),
            "recent_encounters": len(relevant["encounters"])
        }
    }
```

### 7.3 Phase 3: Evaluate MedGemma 1.5 4B for 3D

**Goal:** Test MedGemma 1.5 4B's 3D volume capabilities.

```python
# evaluation/medgemma_1_5_eval.py

"""Evaluation script for MedGemma 1.5 4B 3D CT capabilities."""

from typing import List, Dict
import numpy as np
from PIL import Image

class MedGemma15Evaluator:
    """Evaluate MedGemma 1.5 4B for 3D CT analysis."""

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it"):
        self.model_id = model_id
        self.max_slices = 85  # Official max

    def prepare_ct_as_video(
        self,
        slices: List[np.ndarray]
    ) -> Dict:
        """
        Prepare CT volume in video format for MedGemma 1.5.

        Unlike MedGemma 4B which receives individual images,
        1.5 processes them as a video sequence with temporal encoding.
        """
        # Sample to max slices if needed
        if len(slices) > self.max_slices:
            indices = [
                int(round(i / self.max_slices * (len(slices) - 1)))
                for i in range(1, self.max_slices + 1)
            ]
            slices = [slices[i] for i in indices]

        # Convert to RGB frames
        frames = []
        for slice_data in slices:
            # Normalize to 0-255
            normalized = ((slice_data - slice_data.min()) /
                         (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            # To RGB
            rgb = np.stack([normalized] * 3, axis=-1)
            frames.append(Image.fromarray(rgb))

        return {
            "type": "video",
            "frames": frames,
            "num_frames": len(frames)
        }

    def evaluate_3d_capabilities(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """
        Evaluate model's 3D understanding.

        Test cases should include:
        - Lesion spanning multiple slices
        - Vessel continuity (for PE)
        - Volume estimation tasks
        """
        results = {
            "lesion_tracking": [],
            "vessel_continuity": [],
            "volume_estimation": [],
            "overall_accuracy": 0.0
        }

        for case in test_cases:
            # Run evaluation
            # ... implementation depends on model loading approach
            pass

        return results

    def compare_2d_vs_3d(
        self,
        ct_volume: np.ndarray,
        ground_truth: Dict
    ) -> Dict:
        """
        Compare 2D stack approach vs 3D video approach.

        Measures:
        - Lesion detection accuracy
        - Size estimation accuracy
        - Continuity understanding
        """
        # 2D approach: independent slice analysis
        slices_2d = self._process_as_independent(ct_volume)
        result_2d = self._analyze_2d(slices_2d)

        # 3D approach: video-style processing
        video_input = self.prepare_ct_as_video(ct_volume)
        result_3d = self._analyze_3d(video_input)

        return {
            "2d_approach": result_2d,
            "3d_approach": result_3d,
            "ground_truth": ground_truth,
            "improvement": self._calculate_improvement(result_2d, result_3d, ground_truth)
        }
```

### 7.4 Implementation Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Implementation Timeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Week 1-2: Phase 1 - Enhanced FHIR Extraction                               │
│  ─────────────────────────────────────────────                              │
│  Day 1-2:   Create fhir_models.py with new data classes                     │
│  Day 3-4:   Create clinical_codes.py with LOINC registry                    │
│  Day 5-6:   Implement fhir_context_enhanced.py extractors                   │
│  Day 7-8:   Implement prompt_formatter.py with timeline                     │
│  Day 9-10:  Integration with existing pipeline                              │
│  Day 11-12: Testing with sample FHIR bundles                                │
│  Day 13-14: Documentation and PR                                            │
│                                                                              │
│  Week 3-4: Phase 2 - Smart FHIR Retrieval                                   │
│  ──────────────────────────────────────────                                 │
│  Day 15-17: Implement ehr_navigator_simple.py                               │
│  Day 18-19: Add CT-type specific filtering                                  │
│  Day 20-21: Integrate with enhanced extraction                              │
│  Day 22-24: Testing and optimization                                        │
│  Day 25-28: Documentation and benchmarking                                  │
│                                                                              │
│  Week 5-6: Phase 3 - MedGemma 1.5 Evaluation                                │
│  ────────────────────────────────────────────                               │
│  Day 29-31: Set up MedGemma 1.5 4B environment                              │
│  Day 32-34: Implement video-style CT processing                             │
│  Day 35-37: Create evaluation test cases                                    │
│  Day 38-40: Run comparative benchmarks                                      │
│  Day 41-42: Analysis and decision report                                    │
│                                                                              │
│  Week 7+: Phase 4 - Scale (based on evaluation)                             │
│  ──────────────────────────────────────────────                             │
│  • If 1.5 4B sufficient: Optimize for production                            │
│  • If 27B needed: Plan GPU infrastructure upgrade                           │
│  • Consider tiered routing for complex cases                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Critical Questions & Trade-offs

### 8.1 Key Decision Points

#### 8.1.1 27B Multimodal vs Two-Model Pipeline

**Question:** Is the 27B multimodal's native FHIR+imaging capability worth the cost?

```
Arguments FOR single 27B:
─────────────────────────
• Unified reasoning over FHIR + images
• No context loss between stages
• Simpler architecture
• Best possible accuracy

Arguments AGAINST:
──────────────────
• ~8x higher GPU cost
• May be overkill for simple cases
• Less flexibility
• Single point of failure

Recommendation: Start with 4B, benchmark against 27B,
make data-driven decision based on accuracy delta.
```

#### 8.1.2 EHR Navigator Agent vs Static Extraction

**Question:** Is the agent pattern worth the complexity?

```
Arguments FOR agent pattern:
────────────────────────────
• Smart, query-driven retrieval
• Handles variable FHIR sizes
• Can adapt to different CT types
• Follows Google's recommended approach

Arguments AGAINST:
──────────────────
• Multiple LLM calls = latency
• Agent can make wrong choices
• Implementation complexity
• May not outperform static filtering

Recommendation: Implement simplified version first
(ehr_navigator_simple.py), measure improvement over
static extraction.
```

#### 8.1.3 Context Window Management

**Question:** How to handle large FHIR bundles?

```
Options:
────────

1. Pre-filtering by code
   • Filter observations to CT_TRIAGE_RELEVANT_CODES
   • Always include active conditions/medications
   • Reduces 926 resources → ~50-100 relevant

2. Time-based filtering
   • Only recent observations (30 days)
   • Active conditions regardless of age
   • Recent encounters (90 days)

3. Agent-based retrieval
   • Let model decide what's relevant
   • More flexible, higher latency
   • Risk of missing important data

4. Summarization
   • Pre-summarize FHIR to key points
   • Loses granular data
   • Fast and context-efficient

Recommendation: Combine 1 + 2 as default,
add 3 as enhancement in Phase 2.
```

#### 8.1.4 Latency vs Accuracy Trade-offs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Latency vs Accuracy Trade-off                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Accuracy                                                                    │
│     ▲                                                                        │
│     │                                                                        │
│ 92% │                                          ● 27B Multi (15s)            │
│     │                                                                        │
│ 90% │                              ● 27B→4B Pipeline (15s)                  │
│     │                                                                        │
│ 88% │                  ● Agent+4B (11s)                                     │
│     │                                                                        │
│ 86% │      ● Enhanced 4B (8s)                                               │
│     │                                                                        │
│ 84% │  ● Current 4B (5s)                                                    │
│     │                                                                        │
│     └────────────────────────────────────────────────────────────▶          │
│         5s      8s      11s     14s     17s     20s    Latency              │
│                                                                              │
│  Note: Accuracy numbers are estimates based on EHRQA benchmark              │
│        extrapolation. Actual CT triage accuracy may differ.                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Open Research Questions

1. **How much does 3D understanding improve PE detection?**
   - Need benchmark with PE-positive cases
   - Compare MedGemma 4B-IT vs 1.5 4B

2. **What's the accuracy ceiling for 4B with perfect FHIR context?**
   - If enhanced extraction gets close to 27B, may not need upgrade

3. **How does quantization affect medical reasoning?**
   - INT8 vs FP16 on clinical edge cases
   - Critical for cost optimization

4. **Can prompt engineering close the multi-hop gap?**
   - Explicit chain-of-thought for temporal queries
   - May achieve 27B-like reasoning on 4B

### 8.3 Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| 4B accuracy insufficient | HIGH | Plan 27B evaluation early, have GPU budget ready |
| FHIR extraction misses critical data | HIGH | Comprehensive LOINC code coverage, clinical review |
| Agent makes wrong retrieval choices | MEDIUM | Fallback to full context, logging for analysis |
| 3D CT support doesn't improve accuracy | MEDIUM | Keep 2D as baseline, 3D as enhancement |
| RunPod availability issues | LOW | Multi-provider strategy, on-prem backup option |

---

## 9. Appendices

### Appendix A: MedGemma Official Notebooks Reference

| Notebook | Focus | Key Insights |
|----------|-------|--------------|
| `ehr_navigator_agent.ipynb` | FHIR navigation | LangGraph pattern, iterative retrieval |
| `high_dimensional_ct_model_garden.ipynb` | 3D CT | MAX_SLICE=85, video adapter |
| `medgemma_getting_started.ipynb` | Basic usage | Message format, image handling |
| `medgemma_finetuning.ipynb` | Custom training | LoRA approach, medical domains |

**Source:** https://github.com/google-health/medgemma/tree/main/notebooks

### Appendix B: EHRQA Benchmark Details

From the MedGemma Technical Report:

```
EHRQA Benchmark Structure:
─────────────────────────
• Data Source: Synthea synthetic FHIR records
• Questions per patient: 200
• Categories: 10 (demographics, conditions, medications, labs, etc.)
• Question types: 42
• Training approach: Reinforcement Learning
• Focus: Multi-hop temporal reasoning
• Key metric: Accuracy on "inter-dependent records"

Performance:
───────────
MedGemma 27B text-only: 86.3%
MedGemma 27B multimodal: 90.5% (+4.2%)
MedGemma 1.5 4B: 89.6%

The 4.2% improvement in multimodal comes from RL training
on Synthea FHIR data - specifically the ability to navigate
temporal relationships between records.
```

### Appendix C: RunPod GPU Specifications

| GPU | VRAM | Memory BW | FP16 TFLOPS | Price/hr |
|-----|------|-----------|-------------|----------|
| RTX 3090 | 24GB | 936 GB/s | 35.6 | $0.22 |
| RTX 4090 | 24GB | 1008 GB/s | 82.6 | $0.44 |
| A40 | 48GB | 696 GB/s | 37.4 | $0.79 |
| A100 40GB | 40GB | 1555 GB/s | 77.9 | $1.29 |
| A100 80GB | 80GB | 2039 GB/s | 77.9 | $1.89 |
| H100 80GB | 80GB | 3352 GB/s | 267.6 | $2.65 |
| H100 SXM | 80GB | 3352 GB/s | 267.6 | $3.99 |

**Source:** https://www.runpod.io/pricing (as of January 2026)

### Appendix D: FHIR Resource Quick Reference

Key FHIR resources for CT triage:

```
Patient
├── birthDate → Calculate age
└── gender → Demographics

Condition
├── code.coding[].display → Condition name
├── clinicalStatus.coding[].code → active/resolved
├── onsetDateTime → When diagnosed (CRITICAL)
└── severity.coding[].display → Severity level

Observation
├── code.coding[].code → LOINC code
├── valueQuantity.value → Numeric value
├── valueQuantity.unit → Unit
├── effectiveDateTime → When measured (CRITICAL)
└── referenceRange → Normal range

MedicationRequest
├── medicationCodeableConcept.coding[].display → Drug name
├── status → active/completed/stopped
├── authoredOn → When prescribed (CRITICAL)
└── dosageInstruction[].text → Dosage

Encounter
├── type[].coding[].display → Encounter type
├── period.start → When started
├── period.end → When ended
└── reasonCode[].coding[].display → Chief complaint
```

### Appendix E: Glossary

| Term | Definition |
|------|------------|
| EHRQA | Electronic Health Record Question Answering benchmark |
| FHIR | Fast Healthcare Interoperability Resources |
| LangGraph | Framework for building stateful LLM agents |
| LOINC | Logical Observation Identifiers Names and Codes |
| MAX_SLICE | Maximum CT slices for MedGemma (85) |
| Multi-hop | Reasoning requiring multiple logical steps |
| RL | Reinforcement Learning |
| Synthea | Synthetic patient data generator |
| VRAM | Video RAM (GPU memory) |

---

## Document Metadata

| Field | Value |
|-------|-------|
| Version | 2.0 |
| Author | Sentinel-X Development Team |
| Created | January 2026 |
| Status | Complete |
| Previous Version | FHIR_medgemma.md (Gap Analysis) |
| Next Steps | Begin Phase 1 implementation |

---

## Sources

1. [MedGemma GitHub Notebooks](https://github.com/google-health/medgemma/tree/main/notebooks)
2. [MedGemma 27B HuggingFace](https://huggingface.co/google/medgemma-27b-it)
3. [MedGemma 1.5 4B HuggingFace](https://huggingface.co/google/medgemma-1.5-4b-it)
4. [MedGemma Technical Report](https://arxiv.org/html/2507.05201v1)
5. [Google Research Blog - MedGemma 1.5](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
6. [RunPod GPU Pricing](https://www.runpod.io/pricing)

---

*This document is part of the Sentinel-X AI-powered CT triage system architecture documentation. It builds on the gap analysis in FHIR_medgemma.md and provides actionable implementation guidance.*
