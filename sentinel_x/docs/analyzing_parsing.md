# Sentinel-X Parsing Pipeline Analysis

> Analysis of data flow, formatting issues, and root causes in the patient detail side panel.
> Based on session log `2026-02-11_19-29-11` (22 patients) and triage results from that run.

---

## 1. End-to-End Data Flow

```
CT Volume (.nii.gz)
  │
  ▼
┌──────────────────────────────────────────────┐
│  Phase 1: VisionAnalyzer (MedGemma 4B BF16)  │
│  Input: 85 CT slice images (no clinical ctx)  │
│  Output: Narrative radiology report (text)    │
└──────────────┬───────────────────────────────┘
               │
               ├── raw text ──► VisualFactSheet.raw_response  ("phase1_raw")
               │
               ├── json_repair.py::parse_json_safely()
               │     └── extract_findings_from_narrative()   ◄── PROBLEM SITE #1
               │           └── regex matches finding keywords in ALL sentences
               │                 (including negative ones like "no evidence of calcification")
               │           └── VisualFactSheet.findings: List[Finding]
               │
               ▼
┌──────────────────────────────────────────────┐
│  Phase 2: ClinicalReasoner (MedGemma 27B NF4)│
│  Input: FHIR narrative + Phase 1 raw report   │
│  Output: Markdown reasoning + TRIAGE SUMMARY  │
└──────────────┬───────────────────────────────┘
               │
               ├── prompts.py::parse_phase2_response()
               │     ├── regex: PRIORITY: (\d)  →  overall_priority
               │     ├── regex: HEADLINE: (.+)  →  headline / findings_summary
               │     └── split on "TRIAGE SUMMARY" → reasoning (text before split)
               │
               ├── DeltaAnalysisResult.raw_response  ("phase2_raw")
               │
               ▼
┌──────────────────────────────────────────────┐
│  output_generator.py::generate_triage_result()│
│  Merges Phase 1 + Phase 2 into JSON           │
└──────────────┬───────────────────────────────┘
               │
               ├── visual_findings: formatted from Phase 1 findings   ◄── PROBLEM SITE #2
               │     Template: "{finding} ({location}, {size}): {description}"
               │     Shows "(unspecified, )" when location/size are empty
               │
               ├── rationale: concatenated string                      ◄── PROBLEM SITE #3
               │     "Visual analysis: {visual_findings} EHR Context: {conditions}. Delta: {headline}"
               │
               ├── phase1_raw / phase2_raw: preserved raw model output
               ├── headline: from Phase 2 parsing
               ├── reasoning: from Phase 2 parsing (markdown text)
               │
               ▼
         triage_result.json
               │
               ▼
┌──────────────────────────────────────────────┐
│  API (FastAPI) → WebSocket/REST → Frontend    │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Frontend: PatientDetail → AIAnalysis.tsx      │
│                                                │
│  AI Analysis Tab:                              │
│    - Visual Findings card  ← visual_findings   │  ◄── displays false positives
│    - Conditions Considered ← conditions_considered │
│    - Priority Rationale    ← rationale         │  ◄── redundant concatenated mess
│    - Clinical Reasoning    ← reasoning         │  ◄── raw markdown (asterisks visible)
│                                                │
│  CT Imaging Tab:                               │
│    - CTViewer (slice navigation)               │
│                                                │
│  Bottom Summary:                               │
│    - headline || findings_summary              │
└────────────────────────────────────────────────┘
```

---

## 2. Where Each UI Field Gets Its Data

| UI Section | Field | Source | Issues |
|---|---|---|---|
| Visual Findings card | `visual_findings` | `output_generator.py:84-87` formats `VisualFactSheet.findings` | False positives from negated sentences; ugly format template |
| Conditions Considered | `conditions_considered` | FHIR extraction → `conditions_from_context` list | Includes non-medical FHIR entries like "Received certificate of high school equivalency (finding)" |
| Priority Rationale | `rationale` | `output_generator.py:90-93` concatenation | Redundant: repeats visual_findings + all conditions + headline |
| Clinical Reasoning | `reasoning` | `parse_phase2_response()` → text before "TRIAGE SUMMARY" split | Good content but rendered as plain text (markdown not parsed) |
| Findings Summary | `headline` / `findings_summary` | `parse_phase2_response()` → HEADLINE regex | Clean, works correctly |

---

## 3. Root Cause Analysis

### Issue 1: False-Positive Findings from Negative Sentences

**Location:** `json_repair.py:182-218` — `extract_findings_from_narrative()`

The regex matches finding keywords (e.g., `calcification`, `thickening`, `consolidation`) in any sentence,
regardless of whether the sentence negates the finding.

**Example from train_20 (Priority 3 — normal scan):**

Phase 1 raw report (clean, correct):
```
1. VASCULAR SYSTEM: The thoracic aorta appears normal in caliber...
   There is no evidence of calcification within the thoracic aorta.
2. LUNGS (AIRWAYS): The airways appear normal in caliber without
   evidence of thickening or dilatation.
3. LUNGS (PARENCHYMA): There is no evidence of consolidation,
   atelectasis, or nodules.
```

Parsed findings (all false positives):
```
#   Finding          Location    Size   Description
1   calcification    unspecified        There is no evidence of calcification wi...
2   thickening       unspecified        LUNGS (AIRWAYS): The airways appear norm...
3   consolidation    unspecified        LUNGS (PARENCHYMA): There is no evidence...
```

All three "findings" are from sentences explicitly stating the finding is ABSENT.

**Root cause:** The sentence loop at line 182 does:
```python
finding_match = re.search(finding_types, sentence_lower)
if not finding_match:
    continue
```
It checks for keyword presence but never checks for negation context.

**Fix:** Before accepting a match, check if the text preceding the keyword contains
a negation pattern like "no evidence of", "no ", "without ", "not ".

### Issue 2: Raw Markdown Displayed as Plain Text

**Location:** `AIAnalysis.tsx:79`

The 27B model outputs well-structured markdown:
```
*   **Clinical History:** Patient has a history of resolved Covid-19...
*   **Visual Findings:** Current CT shows normal lungs...
*   **Delta Analysis:** The current CT findings are essentially normal...
```

But the component renders it as:
```tsx
<p className="text-sm text-muted-foreground whitespace-pre-line">{reasoning}</p>
```

This preserves newlines but doesn't parse markdown — asterisks and `**bold**` markers
are visible as literal characters.

**Fix:** Use `react-markdown` to render the `reasoning` field as proper markdown.

### Issue 3: Priority Rationale is Redundant

**Location:** `output_generator.py:90-93`

The `rationale` field is built by concatenating:
1. `"Visual analysis: "` + all visual_findings text (including false positives)
2. `"EHR Context: Patient has "` + ALL FHIR conditions joined by comma
3. `"Delta: "` + headline

**Example from train_20:**
```
Visual analysis: calcification (unspecified, ): There is no evidence of...
EHR Context: Patient has Tooth eruption disorder, Chronic periodontitis,
Received certificate of high school equivalency (finding), Prediabetes,
Anemia (disorder), Fracture of bone (disorder)...
Delta: No acute abnormality identified on CT scan.
```

Problems:
- Repeats visual_findings (with false positives)
- Dumps ALL FHIR conditions including non-medical entries
- The Clinical Reasoning card already provides much better, LLM-generated analysis

**Fix:** Remove the Priority Rationale card from the frontend entirely. The `rationale`
field remains in triage_result.json for backward compatibility but is not displayed.

### Issue 4: Visual Findings Format Template

**Location:** `output_generator.py:84-87`

```python
visual_findings_text = "; ".join(
    f"{f.finding} ({f.location}, {f.size}): {f.description}"
    for f in visual_fact_sheet.findings
) or "No abnormalities detected"
```

When `location` is `"unspecified"` and `size` is `""`, the output is:
```
calcification (unspecified, ): There is no evidence of...
```

The `(unspecified, )` is ugly and non-informative.

**Fix:** Conditionally include location and size only when they have meaningful values.

---

## 4. Phase 2 Response Parsing Details

**Parser:** `prompts.py:96-127` — `parse_phase2_response()`

The 27B model is prompted to output:
```
[reasoning in bullet points]

---
TRIAGE SUMMARY
PRIORITY: [1, 2, or 3]
HEADLINE: [5-10 word summary]
```

Parsing steps:
1. **Default:** priority=3, headline="Assessment Pending", reasoning=full response
2. **Priority:** `re.findall(r"PRIORITY:\s*(\d)", text)` — takes the LAST match to avoid
   false matches in reasoning text
3. **Headline:** `re.search(r"HEADLINE:\s*(.+)", text)` — takes first match
4. **Reasoning:** Splits on `"TRIAGE SUMMARY"` and keeps text BEFORE the split

The reasoning field retains raw markdown formatting from the model. This is correct behavior —
the frontend should render it as markdown, not strip it.

---

## 5. Phase 1 Findings Extraction Details

**Parser:** `json_repair.py:127-228` — `extract_findings_from_narrative()`

This is a fallback strategy when Phase 1 doesn't produce valid JSON. Since the prompt
asks for a narrative "Visual Inventory" report, this fallback activates for every patient.

Steps:
1. Split text into unique lines, then rejoin as `full_text`
2. Split `full_text` on sentence boundaries `[.;]\s+`
3. For each sentence, regex-search for `finding_types` keywords
4. If found, also search for `locations` and `size_pattern`
5. Build a Finding dict with `{finding, location, size, slice_index, description}`

The fundamental problem: step 3 matches keywords regardless of semantic context.
A sentence like "There is no evidence of calcification" matches `calcification` and
produces a false-positive finding.

---

## 6. Recommendations (Implemented)

1. **Frontend: Use `phase1_raw` for Visual Findings** — The raw Phase 1 report is clean
   professional prose already structured by anatomical region. Display it instead of
   the regex-extracted `visual_findings`.

2. **Frontend: Render markdown** — Use `react-markdown` for both `phase1_raw` and
   `reasoning` fields to properly display bold, bullets, and structure.

3. **Frontend: Remove Priority Rationale card** — The `rationale` field is a concatenated
   mess that adds no value over the Clinical Reasoning card.

4. **Backend: Add negation filtering** — In `extract_findings_from_narrative()`, skip
   sentences where negation words precede the finding keyword. This improves data quality
   in `visual_findings` and `rationale` fields even though the frontend no longer displays them.

5. **Backend: Fix format template** — Clean up the `visual_findings` formatting to handle
   empty size/location gracefully.
