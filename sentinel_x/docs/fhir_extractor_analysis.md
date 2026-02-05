# FHIR Context Retrieval Logs - Deep Analysis

## Executive Summary

This document provides a comprehensive analysis of the Sentinel-X FHIR context retrieval system based on examination of 4 log files in `fhir_context_retreival_logs/`. It covers what the model did right, what it got wrong, bugs discovered, and recommendations for improvement.

---

## Log Files Overview

| Patient | Age | Queries | Success Rate | Critical Issue |
|---------|-----|---------|--------------|----------------|
| train_1_a_1 | 71M | 5 | 100% (5/5) | None - best performing |
| train_1_a_2 | 83M | 5 | 100% (5/5) | Ineffective search strategy |
| train_2_a_1 | 73M | 3 | 33% (1/3) | **BUG: fhir_bundle parameter error** |
| train_2_a_2 | 60M | 2 | 100% (2/2) | Incomplete investigation |

---

## Detailed Analysis by Patient

### 1. train_1_a_1 (Best Performance)

**Visual Findings:** Venous collaterals, collapsed left subclavian vein, bronchial wall thickening, atelectasis

**What the Model Did Right:**
- ✅ Started with `get_patient_manifest()` as instructed
- ✅ Searched for "vascular" (relevant to venous collaterals)
- ✅ Checked anticoagulant status (per PE/clot investigation protocol)
- ✅ Checked antiplatelet status (found active Clopidogrel, Aspirin, Prasugrel)
- ✅ Retrieved Coag labs (INR 1.19, aPTT 33.9 - both normal)

**What Could Be Improved:**
- ⚠️ The model found the patient is on triple antiplatelet therapy (very aggressive cardiac regimen) but didn't investigate WHY
- ⚠️ Should have searched for "cardiac" or "coronary" given the medication profile suggests significant cardiac history
- ⚠️ Didn't correlate the collapsed subclavian vein with the antiplatelet use (possible prior catheterization/stent?)

**Final Assessment Quality:** Log is incomplete - no final assessment shown

---

### 2. train_1_a_2 (100% Success, Poor Strategy)

**Visual Findings:** Same as train_1_a_1 (venous collaterals, subclavian vein collapsed, bronchial thickening, atelectasis)

**What the Model Did Right:**
- ✅ Started with `get_patient_manifest()`
- ✅ All 5 queries executed without errors

**What the Model Got Wrong:**
- ❌ **Overly specific search terms:** Searched for "venous thromboembolism", "chronic venous insufficiency" - too precise, unlikely to match
- ❌ **No partial matching:** Should have used "thrombo" or "venous" instead of full phrases
- ❌ **Missed medication check:** Never checked anticoagulant/antiplatelet status despite imaging showing vascular findings
- ❌ **Never checked labs:** Despite 1,708 observations available, never queried any lab categories

**Query Strategy Failure:**
```
Query 2: "venous thromboembolism"  → 0 matches (too specific)
Query 3: "infection"              → 0 matches (synthetic data rarely has this)
Query 4: "chronic venous insufficiency" → 0 matches (too specific)
Query 5: "malignancy"             → 0 matches (should try "cancer" or "malignan")
```

**The prompt says:** "Use broad, partial terms: 'thrombo' (matches thrombosis, thromboembolism)"
But the model ignored this guidance.

**Final Assessment:** "No documented history... Further investigation needed" - gave up too easily.

---

### 3. train_2_a_1 (CRITICAL BUG)

**Visual Findings:** Emphysematous changes, peribronchial thickening, subpleural nodule, gallbladder stone

**THE BUG:**
```
Query 2: search_clinical_history
Arguments: {"query": "COPD", "fhir_bundle": "69519d53-4e50-1616-e545-701805166688"}

RESULT: error: search_clinical_history() got multiple values for argument 'fhir_bundle'
```

**Root Cause Analysis:**

Looking at `agent_loop.py:164`:
```python
result = tool_func(self.fhir_bundle, **arguments)
```

The code passes `fhir_bundle` as the first positional argument, AND the model included it in the `arguments` dict. This causes Python to receive `fhir_bundle` twice.

**Why did the model include fhir_bundle?**
The model saw the patient_id in the manifest response and incorrectly assumed it should pass it back. The tool docstrings say:
```
Args:
    fhir_bundle: The FHIR Bundle containing patient data
    query: Search term (case-insensitive, partial match)
```

The model interpreted this as needing to provide `fhir_bundle`, but:
1. It used the patient ID instead of the bundle
2. The system already provides fhir_bundle automatically

**Impact:** 2 of 3 queries failed, but the model still produced a final assessment claiming "history of resolved Lung Cancer and resolved COPD" - **this is hallucination** since those queries never succeeded.

**Critical Finding: The model hallucinated clinical history that was never retrieved.**

---

### 4. train_2_a_2 (Incomplete Investigation)

**Visual Findings:** Same as train_2_a_1 (emphysematous changes, nodule, gallbladder stone)

**What the Model Did:**
- ✅ Started with `get_patient_manifest()`
- ✅ Retrieved Renal labs (successful)

**What the Model Got Wrong:**
- ❌ **Wrong lab category:** Visual findings show lung pathology, but model checked "Renal" labs
- ❌ **Only 2 queries:** Given emphysematous changes and a pulmonary nodule, should have searched for:
  - COPD/emphysema history
  - Smoking history
  - Cancer/malignancy (nodule in lung!)
  - Checked CBC for infection markers
- ❌ **No final assessment:** Just says "No assessment provided"

**This is model underperformance** - it failed to investigate appropriately despite having working tools.

---

## Bugs Identified

### Bug 1: fhir_bundle Parameter Collision (CRITICAL)
**File:** `sentinel_x/triage/agent_loop.py:164`
**Issue:** Model sometimes includes `fhir_bundle` in arguments, causing duplicate parameter error
**Fix Options:**
1. Strip `fhir_bundle` from arguments before calling tool
2. Add validation in `_execute_tool()` to remove known implicit args
3. Update prompts to explicitly say "Do NOT include fhir_bundle in arguments"

### Bug 2: Hallucination After Query Failure (CRITICAL)
**File:** `sentinel_x/triage/agent_loop.py` (general logic)
**Issue:** Model generates assessment referencing data from failed queries
**Evidence:** train_2_a_1 claims "history of resolved Lung Cancer and COPD" but those queries failed
**Fix Options:**
1. Track failed queries and inject warning into conversation
2. Add validation that final assessment only references successfully retrieved data
3. Add post-processing check for consistency

### Bug 3: Incomplete Query Strategy (MEDIUM)
**File:** `sentinel_x/triage/prompts.py`
**Issue:** Despite clear guidance on partial matching, model uses full phrases
**Evidence:** "venous thromboembolism" instead of "thrombo" or "venous"
**Fix Options:**
1. Add more one-shot examples showing partial matching
2. Add explicit instruction: "NEVER search for full medical phrases"
3. Implement server-side fuzzy matching

### Bug 4: Irrelevant Lab Category Selection (LOW)
**Evidence:** train_2_a_2 checked Renal labs for lung pathology
**Issue:** Model doesn't correlate visual findings with appropriate lab categories
**Fix Options:**
1. Add mapping table in prompts: "Lung findings → check CBC, Cardiac"
2. Add more specific examples in one-shot

---

## What the Model Got Right

1. **Follows basic protocol:** Always starts with `get_patient_manifest()` as instructed
2. **JSON formatting:** Tool call JSON is generally well-formed
3. **Proper tool selection:** Uses appropriate tools for the task (search for history, check meds, get labs)
4. **Anticoagulation check for vascular findings:** In train_1_a_1, correctly checked anticoagulant status
5. **Structured final assessment format:** When provided, follows FINAL_ASSESSMENT/RISK_ADJUSTMENT/CRITICAL_FINDINGS format

---

## What the Model Got Wrong

1. **Overly specific search terms:** Uses full phrases instead of partial matching
2. **Parameter confusion:** Includes `fhir_bundle` when it shouldn't
3. **Hallucination:** Claims findings from failed queries
4. **Premature termination:** Stops after few queries without thorough investigation
5. **Wrong lab categories:** Checks Renal for lung pathology
6. **No medication investigation:** Often skips medication checks despite vascular/cardiac findings
7. **No correlation:** Doesn't connect imaging findings to appropriate clinical investigations

---

## Root Cause Analysis: Why Model Underperformed

### 1. MedGemma 4B Limitations
- 4B parameter model has limited reasoning capacity
- Struggles with multi-step clinical reasoning
- Tends toward "satisficing" - stops after minimal investigation

### 2. Prompt Design Issues
- One-shot example focuses heavily on PE + anticoagulation
- No examples for non-PE cases (emphysema, nodules, infection)
- Tool descriptions show `fhir_bundle` parameter which confuses the model

### 3. Missing Guardrails
- No validation that model uses partial search terms
- No stripping of `fhir_bundle` from arguments
- No consistency check between queries and final assessment

### 4. Synthetic Data Mismatch
- Search queries may not match Synthea-generated condition names
- Model learned search patterns from real medical records, not synthetic data

---

## Code Improvements Needed

### Priority 1 (Critical)
1. **Fix fhir_bundle parameter bug** in `agent_loop.py:_execute_tool()`
2. **Add query failure tracking** - flag when queries fail
3. **Add hallucination detection** - verify assessment claims against successful queries

### Priority 2 (High)
4. **Improve prompts** with non-PE examples
5. **Hide fhir_bundle from tool signatures** in prompt generation
6. **Add partial matching hint** directly before each search

### Priority 3 (Medium)
7. **Add fuzzy matching** server-side for search_clinical_history
8. **Add lab category recommendation** based on visual findings
9. **Increase minimum queries** before allowing final assessment

---

## Specific Code Locations for Fixes

| Issue | File | Line | Change Needed |
|-------|------|------|---------------|
| fhir_bundle collision | `agent_loop.py` | 164 | Strip fhir_bundle from arguments |
| Hallucination risk | `agent_loop.py` | 284-296 | Add query success validation |
| Prompt improvement | `prompts.py` | 126-173 | Add diverse one-shot examples |
| Tool description | `tools.py` | 45-58 | Hide internal params from descriptions |
| Search strategy | `tools.py` | 155-218 | Add fuzzy/partial matching |

---

## Recommendations

### Immediate (Before Next Run)
1. Fix the fhir_bundle parameter bug
2. Add explicit "Do NOT include fhir_bundle" to prompts

### Short-term
3. Add 2-3 more one-shot examples covering different pathologies
4. Implement query failure tracking with warning injection
5. Add minimum query threshold (at least 3 queries before assessment)

### Long-term
6. Consider upgrading to larger model (8B or Claude)
7. Implement server-side fuzzy matching for clinical history search
8. Add post-processing validation layer for assessment consistency

---

## Part 2: MedGemma Behavior Deep Dive

### Model Configuration

From `config.py`:
```python
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"  # 4B parameter, instruction-tuned
MEDGEMMA_DTYPE = "bfloat16"                   # Memory-efficient precision
AGENT_TOOL_CALL_TEMPERATURE = 0.0             # Deterministic generation
AGENT_MAX_TOKENS_PER_TURN = 512               # Token budget per iteration
AGENT_MAX_ITERATIONS = 5                       # Maximum reasoning steps
```

### Token Generation Analysis

**Observed Timing from Traces:**

| Patient | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Total |
|---------|--------|--------|--------|--------|--------|-------|
| train_1_a_2 | 3.6s | 4.9s | 4.7s | 4.4s | 8.3s | 25.9s |
| train_2_a_1 | 22.7s | 22.6s | 23.0s | 23.0s | 22.1s | 112.4s |
| train_2_a_2 | 22.0s | 25.9s | 25.9s | 25.9s | 25.9s | 125.6s |

**Key Observation:** train_1_a_2 was much faster (avg 5.2s/iter) vs train_2_a_1/a_2 (avg 23s/iter). This suggests the model was doing more reasoning/generation in the slower cases.

### Why the Model Included fhir_bundle in Arguments

The tool docstrings in `tools.py` explicitly show `fhir_bundle` as a parameter:
```python
def search_clinical_history(fhir_bundle: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Search patient's conditions and clinical notes for relevant history.
    ...
    Args:
        fhir_bundle: The FHIR Bundle containing patient data
        query: Search term (case-insensitive, partial match)
    """
```

When `get_tool_descriptions()` generates the prompt, it includes this full docstring. MedGemma 4B, being a smaller model, literally followed the documentation and tried to pass `fhir_bundle`.

**The Problem:** The model saw patient_id "69519d53-4e50-1616-e545-701805166688" in the manifest response and assumed this was the fhir_bundle to pass.

### Repetition Behavior

From the traces, we see a clear pattern of repetition:

**train_2_a_1:**
- Iter 2: `search_clinical_history({"query": "COPD", "fhir_bundle": "69519d53..."})` → ERROR
- Iter 3: Same call detected as duplicate
- Iter 4: `search_clinical_history({"query": "lung", "fhir_bundle": "69519d53..."})` → ERROR
- Iter 5: Same call detected as duplicate

**train_2_a_2:**
- Iter 3: `get_patient_manifest({})` → DUPLICATE (already called in Iter 1)
- Iter 4: `get_recent_labs({"category": "Renal"})` → DUPLICATE
- Iter 5: Same call → DUPLICATE

**Pattern:** When the model encounters an error or doesn't know what to do next, it falls into repetitive loops. The duplicate detection is working correctly but the model isn't learning from the feedback.

### Why Model Hallucinated in train_2_a_1

Looking at the final assessment in the log:
> "The patient has a history of resolved Lung Cancer and resolved COPD."

But the actual query results:
- Query 2 (COPD): ERROR - fhir_bundle collision
- Query 3 (lung): ERROR - fhir_bundle collision

**Root Cause:** The model's training includes general medical reasoning patterns. When it sees:
- 73-year-old male
- Emphysematous changes
- Peribronchial thickening

It likely activated prior knowledge that these findings are commonly associated with COPD and lung cancer history, and stated this as fact even though the queries failed.

This is a classic **confabulation** pattern in smaller language models.

### One-Shot Example Overfitting

The one-shot example in `prompts.py` focuses heavily on:
1. PE detection
2. Anticoagulation check
3. INR lab retrieval
4. Anticoagulation failure assessment

The model appears to have learned this specific pattern well (train_1_a_1 correctly checked antiplatelet and coag labs). But when presented with non-PE cases (emphysema, nodules), it lacks examples to follow and falls into repetitive patterns.

---

## Part 3: End-to-End Flow Analysis

### Complete Pipeline Trace

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA INGESTION                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  inbox/                                                                      │
│  ├── volumes/train_1_a_1.nii.gz   ────────┐                                 │
│  └── reports/train_1_a_1.json     ────────┼──▶ InboxWatcher.watch()         │
│                                            │                                 │
│  PatientData(                              │                                 │
│    patient_id="train_1_a_1",              │                                 │
│    volume_path=Path(...),          ◀──────┘                                 │
│    report_path=Path(...)                                                    │
│  )                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: CONTEXT EXTRACTION (fhir_context.py)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  parse_fhir_context(report_path, patient_id)                                │
│    │                                                                         │
│    ├──▶ Load JSON from report_path                                          │
│    │                                                                         │
│    ├──▶ Check resourceType == "Bundle"                                      │
│    │                                                                         │
│    ├──▶ extract_patient_demographics() ──▶ age, gender, deceased            │
│    │      └── Uses extract_age_from_patient_resource()                      │
│    │                                                                         │
│    ├──▶ extract_conditions() ──▶ List[str] conditions                       │
│    │      └── Iterates over Condition resources                             │
│    │                                                                         │
│    ├──▶ extract_medications() ──▶ List[str] medications                     │
│    │      └── Iterates over MedicationStatement/Request                     │
│    │                                                                         │
│    ├──▶ identify_risk_factors() ──▶ High-risk conditions                    │
│    │      └── Matches against HIGH_RISK_CONDITIONS keywords                 │
│    │                                                                         │
│    ├──▶ Find DiagnosticReports ──▶ findings, impressions                    │
│    │      └── Decodes base64 presentedForm                                  │
│    │                                                                         │
│    └──▶ Return PatientContext                                               │
│                                                                              │
│  format_context_for_prompt(context) ──▶ Markdown string                     │
│    """                                                                       │
│    ## EHR Clinical Context                                                  │
│    **Demographics:** 71 year old male                                       │
│    **Medical History:** COPD, Diabetes, ...                                 │
│    **Current Medications:** Aspirin, Metformin, ...                         │
│    ## Radiology Report Findings                                             │
│    ...                                                                       │
│    """                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: CT PROCESSING (ct_processor.py)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  process_ct_volume(volume_path)                                             │
│    │                                                                         │
│    ├──▶ load_nifti_volume() ──▶ 3D numpy array + metadata                   │
│    │      └── nibabel.load(path).get_fdata()                                │
│    │                                                                         │
│    ├──▶ apply_window(data, center=40, width=400) ──▶ Soft tissue window     │
│    │      └── HU range: [-160, 240] → [0, 255]                              │
│    │                                                                         │
│    ├──▶ sample_slices(volume, num_slices=85) ──▶ Uniform sampling           │
│    │      └── np.linspace(0, total_slices-1, 85)                            │
│    │                                                                         │
│    └──▶ For each slice index:                                               │
│           extract_slice_as_image() ──▶ PIL.Image (RGB)                      │
│             └── Rotate 90°, stack to RGB, convert to uint8                  │
│                                                                              │
│  Output: (List[PIL.Image], List[int] indices, dict metadata)                │
│          85 images, 512x512 RGB each                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: VISUAL ANALYSIS (medgemma_analyzer.py)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MedGemmaAnalyzer.analyze(images, context_text)                             │
│    │                                                                         │
│    ├──▶ _build_messages()                                                   │
│    │      └── System prompt (priority definitions + output format)          │
│    │      └── User content: [85 image tokens] + formatted context          │
│    │                                                                         │
│    ├──▶ processor.apply_chat_template()                                     │
│    │      └── Gemma chat format                                             │
│    │                                                                         │
│    ├──▶ processor(text=prompt, images=images)                               │
│    │      └── Tokenize text + encode images                                 │
│    │      └── ~22,000+ input tokens                                         │
│    │                                                                         │
│    ├──▶ model.generate(max_new_tokens=1024, do_sample=False)                │
│    │      └── Deterministic generation                                      │
│    │                                                                         │
│    └──▶ _parse_response()                                                   │
│           └── Regex extraction:                                             │
│               - VISUAL_FINDINGS: ...                                        │
│               - KEY_SLICE: 42                                               │
│               - PRIORITY_LEVEL: 2                                           │
│               - PRIORITY_RATIONALE: ...                                     │
│               - FINDINGS_SUMMARY: ...                                       │
│               - CONDITIONS_CONSIDERED: ...                                  │
│                                                                              │
│  Output: AnalysisResult                                                     │
│    └── visual_findings: "Multiple venous collaterals..."                    │
│    └── key_slice_index: 42                                                  │
│    └── priority_level: 2                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: REACT AGENT LOOP (agent_loop.py)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ReActAgentLoop.run(visual_findings)                                        │
│    │                                                                         │
│    ├──▶ Initialize state: messages=[], tools_used=[], iteration=0          │
│    │                                                                         │
│    ├──▶ build_agent_user_prompt(visual_findings)                            │
│    │      └── Includes one-shot PE + anticoagulation example               │
│    │                                                                         │
│    └──▶ WHILE iteration < 5 AND not should_stop:                            │
│           │                                                                  │
│           ├──▶ _generate_response(state)                                    │
│           │      └── Build messages (system + history)                      │
│           │      └── model.generate(max_new_tokens=512, temp=0.0)          │
│           │                                                                  │
│           ├──▶ Check for "FINAL_ASSESSMENT:" ──▶ STOP                       │
│           │                                                                  │
│           ├──▶ extract_tool_call(response)                                  │
│           │      └── Regex: TOOL_CALL:\n{...}                              │
│           │      └── parse_json_safely() with repair                       │
│           │                                                                  │
│           ├──▶ Check for duplicate tool call ──▶ SKIP + warn               │
│           │                                                                  │
│           ├──▶ _execute_tool(tool_call)                                     │
│           │      └── tool_func(self.fhir_bundle, **arguments)  ◀── BUG!    │
│           │      └── fhir_bundle passed twice if in arguments              │
│           │                                                                  │
│           └──▶ _format_observation() ──▶ Add to messages                    │
│                                                                              │
│  Output: AgentState                                                         │
│    └── final_assessment: "No treatment failure detected..."                │
│    └── risk_adjustment: "NONE" / "INCREASE" / "DECREASE"                   │
│    └── critical_findings: ["finding1", "finding2"]                         │
│    └── tools_used: ["get_patient_manifest", "check_medication_status", ...]│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: RISK ADJUSTMENT & OUTPUT (agent.py)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Apply risk adjustment:                                                     │
│    original_priority = analysis.priority_level  (from MedGemma)             │
│    adjustment = get_risk_adjustment_value(agent_state.risk_adjustment)     │
│      └── "INCREASE" → -1 (more urgent)                                     │
│      └── "DECREASE" → +1 (less urgent)                                     │
│      └── "NONE" → 0                                                         │
│    final_priority = clamp(original_priority + adjustment, 1, 3)            │
│                                                                              │
│  generate_triage_result():                                                  │
│    └── Create thumbnail of key slice                                       │
│    └── Encode to base64                                                    │
│    └── Package all findings + agent trace                                  │
│                                                                              │
│  save_triage_result() ──▶ output/triage_results/{patient_id}.json          │
│                                                                              │
│  worklist.add_entry():                                                      │
│    └── Update priority-sorted worklist                                     │
│    └── Persist to worklist.json                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Critical Path Issues Identified

1. **FHIR Bundle Parameter Bug** (Phase 5)
   - Location: `agent_loop.py:164`
   - `tool_func(self.fhir_bundle, **arguments)` collides with model-provided fhir_bundle

2. **Tool Docstring Leakage** (Phase 5)
   - Location: `tools.py:45-58`
   - Tool signatures with `fhir_bundle` visible to model in prompt

3. **Confabulation Risk** (Phase 5)
   - No validation between query results and final assessment
   - Model can claim data from failed queries

4. **One-Shot Overfitting** (Phase 5)
   - Location: `prompts.py:126-173`
   - Only PE + anticoagulation example provided

---

## Conclusion

The FHIR context retrieval system shows promise but has critical bugs and model behavior issues:

- **1 of 4 patients** (train_1_a_1) received thorough investigation
- **1 bug** caused 66% query failure rate in one patient
- **Hallucination** occurred when queries failed
- **Search strategy** is ineffective due to overly specific terms
- **Repetition loops** when model encounters errors

The core architecture is sound, but needs:
1. Parameter collision fix in `agent_loop.py`
2. Hidden internal parameters in tool descriptions
3. Query failure tracking with assertion validation
4. More diverse one-shot examples
5. Anti-repetition guardrails beyond duplicate detection
