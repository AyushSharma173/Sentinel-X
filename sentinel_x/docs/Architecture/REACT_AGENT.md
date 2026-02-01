# ReAct Agent Architecture for Sentinel-X Triage

## Overview

This document describes the ReAct (Reason+Act) agent architecture implemented in Sentinel-X to enhance CT triage by dynamically investigating clinical context from FHIR patient data.

### The Problem: Static Filtering Misses "Silent Failures"

The original pipeline used static FHIR filtering:
```
CT Finding → Python filter guesses relevant FHIR data → MedGemma analyzes
```

This approach misses **"silent failures"** - situations where clinical context dramatically changes the severity of a finding:

| Finding | Without Agent | With Agent |
|---------|---------------|------------|
| PE detected | Flags as urgent | Checks meds → finds Warfarin → checks INR → **ANTICOAGULATION FAILURE** |
| Stroke found | Flags as urgent | Checks meds → finds Aspirin → **ANTIPLATELET FAILURE** |
| High HR + cardiac | Grabs troponin | Checks meds → finds beta-blocker → **RATE CONTROL FAILURE** |

### The Solution: ReAct Agent Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ReAct Agent Loop                             │
│                                                                     │
│  CT Finding ──► THOUGHT ──► TOOL_CALL ──► OBSERVATION ──► THOUGHT  │
│       │              │            │             │             │     │
│       │              ▼            ▼             ▼             ▼     │
│       │        "Need to     {"tool":        Result from    "Now I  │
│       │         check       "check_meds",   FHIR query     know..."│
│       │         meds..."    "args":{...}}                          │
│       │                                                             │
│       └─────────────────► FINAL_ASSESSMENT ◄────────────────────────┘
│                           + RISK_ADJUSTMENT                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### File Structure

```
sentinel_x/triage/
├── tools.py           # @tool decorated FHIR query functions
├── agent_loop.py      # ReAct while loop with JSON detection
├── json_repair.py     # strip_json_decoration, repair_json
├── state.py           # AgentState TypedDict
├── prompts.py         # Agent system prompt + one-shot example
├── agent.py           # Integration after visual analysis
├── output_generator.py # Includes agent reasoning trace
└── config.py          # Agent configuration constants
```

### Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          TriageAgent.process_patient()                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Parse FHIR Context ──────────────────────────────────────────────┐   │
│     │                                                                │   │
│     ▼                                                                │   │
│  2. Process CT Volume                                                │   │
│     │                                                                │   │
│     ▼                                                                │   │
│  3. MedGemma Visual Analysis ───► visual_findings                    │   │
│     │                                   │                            │   │
│     │                                   ▼                            │   │
│     │                         4. ReAct Agent Loop ◄──── fhir_bundle  │   │
│     │                              │                                 │   │
│     │                              ▼                                 │   │
│     │                         agent_state                            │   │
│     │                         (risk_adjustment)                      │   │
│     │                              │                                 │   │
│     ▼                              ▼                                 │   │
│  5. Apply Risk Adjustment (adjust priority_level)                    │   │
│     │                                                                │   │
│     ▼                                                                │   │
│  6. Generate Output (include agent_reasoning trace)                  │   │
│     │                                                                │   │
│     ▼                                                                │   │
│  7. Save & Update Worklist                                           │   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Module Documentation

### 1. State Management (`state.py`)

Defines TypedDicts for type-safe state management throughout the agent loop.

#### `AgentState`

```python
class AgentState(TypedDict):
    # Core state
    patient_id: str
    visual_findings: str
    fhir_bundle: Dict[str, Any]

    # Loop control
    iteration: int
    should_stop: bool
    max_iterations: int

    # Message history
    messages: List[AgentMessage]

    # Tool tracking
    tools_used: List[str]
    tool_calls: List[ToolCall]
    tool_results: List[Dict[str, Any]]

    # Final outputs
    final_assessment: Optional[str]
    risk_adjustment: Optional[str]  # "INCREASE", "DECREASE", or None
    critical_findings: List[str]

    # Error handling
    errors: List[str]
```

#### Key Functions

- `create_initial_state()`: Factory function to initialize agent state
- `format_agent_trace()`: Converts state to JSON-serializable trace for output

---

### 2. JSON Repair (`json_repair.py`)

Handles common JSON formatting issues from the MedGemma 4B model output.

#### Problem: Model Output Quirks

The 4B model sometimes produces:
- Markdown code fences: ` ```json ... ``` `
- Python literals: `True`, `False`, `None`
- Trailing commas: `{"key": "value",}`
- Single quotes: `{'key': 'value'}`

#### Functions

| Function | Purpose |
|----------|---------|
| `strip_json_decoration(text)` | Remove markdown fences, find JSON boundaries |
| `repair_json(json_str)` | Fix Python→JSON literals, trailing commas |
| `parse_json_safely(text)` | Progressive repair strategies |
| `extract_tool_call(text)` | Parse `TOOL_CALL:` blocks from output |
| `extract_final_assessment(text)` | Parse `FINAL_ASSESSMENT:` blocks |

#### Example

```python
# Input from model
text = '''
THOUGHT: I need to check medications.

TOOL_CALL:
```json
{"tool": "check_medication_status", "arguments": {"medication_name": "warfarin"}}
```
'''

# Extract and parse
tool_call = extract_tool_call(text)
# Returns: {"tool": "check_medication_status", "arguments": {"medication_name": "warfarin"}}
```

---

### 3. FHIR Tools (`tools.py`)

Four tools with extensive docstrings that serve as the model's instruction manual.

#### Tool Registry

```python
TOOL_REGISTRY: Dict[str, Callable] = {}

@tool
def my_tool(fhir_bundle: Dict, ...):
    """Docstring becomes agent's instruction manual."""
    ...
```

#### Available Tools

##### `get_patient_manifest(fhir_bundle)`

**Purpose**: Overview of available clinical data

**When to Use**: Call FIRST to understand what data exists

**Returns**:
```json
{
    "patient_id": "patient-123",
    "demographics": {"age": 67, "gender": "male"},
    "resource_counts": {"Condition": 4, "MedicationRequest": 3, "Observation": 12},
    "available_lab_categories": ["Cardiac", "Coag", "Renal"]
}
```

##### `search_clinical_history(fhir_bundle, query)`

**Purpose**: Search conditions by keyword (partial match)

**When to Use**: Check for relevant history when imaging shows findings

**Search Tips**:
- Use partial terms: `"thrombo"` matches thrombosis, thromboembolism
- Use `"malignan"` to catch malignancy, malignant
- Use `"diabet"` for diabetes, diabetic

**Returns**:
```json
{
    "query": "thrombo",
    "match_count": 2,
    "conditions": [
        {"display": "Deep vein thrombosis", "status": "active", "onset_date": "2024-06-15"}
    ]
}
```

##### `get_recent_labs(fhir_bundle, category)`

**Purpose**: Retrieve recent lab values by category

**Categories**:
| Category | Labs Included |
|----------|---------------|
| Cardiac | Troponin, BNP, NT-proBNP |
| Coag | D-dimer, INR, PTT, aPTT, Fibrinogen |
| Renal | Creatinine, GFR, BUN |
| CBC | Hemoglobin, Hematocrit, WBC, Platelets |
| Metabolic | Glucose, Sodium, Potassium |

**When to Use**:
- PE/clot → check "Coag"
- Cardiac finding → check "Cardiac"
- Contrast planned → check "Renal"

**Returns**:
```json
{
    "category": "Coag",
    "lookback_days": 90,
    "values": [
        {"name": "INR", "value": 2.3, "unit": "", "date": "2025-06-10", "flag": "normal"}
    ]
}
```

##### `check_medication_status(fhir_bundle, medication_name)`

**Purpose**: Check if patient is on specific medication (CRITICAL for treatment failure detection)

**When to Use**:
- PE/DVT → check `"anticoag"`, `"warfarin"`, `"heparin"`
- Stroke → check `"aspirin"`, `"clopidogrel"`, `"antiplatelet"`
- Cardiac → check `"beta"`, `"statin"`

**Auto-Expansion**:
- `"anticoag"` → searches warfarin, heparin, enoxaparin, rivaroxaban, apixaban, etc.
- `"beta"` → searches metoprolol, atenolol, carvedilol, etc.
- `"statin"` → searches atorvastatin, simvastatin, rosuvastatin, etc.

**Returns**:
```json
{
    "query": "anticoag",
    "found": true,
    "medications": [
        {"name": "Warfarin 5mg", "status": "active", "start_date": "2025-01-15", "dosage": "5mg daily"}
    ],
    "is_currently_active": true
}
```

---

### 4. Agent Loop (`agent_loop.py`)

Simple Python while loop implementing the ReAct pattern.

#### `ReActAgentLoop` Class

```python
class ReActAgentLoop:
    TOOL_CALL_TEMPERATURE = 0.0  # Deterministic for JSON output
    MAX_ITERATIONS = 5

    def __init__(self, model, processor, fhir_bundle, patient_id, max_iterations):
        ...

    def run(self, visual_findings: str) -> AgentState:
        """Main loop: THOUGHT → TOOL_CALL → OBSERVATION → repeat"""
        ...
```

#### Loop Algorithm

```python
while not state["should_stop"] and state["iteration"] < MAX_ITERATIONS:
    state["iteration"] += 1

    # 1. Generate model response (Thought + Action)
    response = self._generate_response(state)

    # 2. Check for FINAL_ASSESSMENT (stop condition)
    if "FINAL_ASSESSMENT:" in response:
        self._extract_final_assessment(response, state)
        state["should_stop"] = True
        continue

    # 3. Extract and clean JSON tool call
    tool_call = extract_tool_call(response)
    if tool_call is None:
        # Prompt model to call tool or conclude
        continue

    # 4. Execute tool against FHIR bundle
    result = self._execute_tool(tool_call)

    # 5. Feed observation back to model
    state["messages"].append(format_observation(result))
```

#### Risk Adjustment

```python
def get_risk_adjustment_value(adjustment: Optional[str]) -> int:
    if adjustment == "INCREASE":
        return -1  # Lower number = higher priority
    elif adjustment == "DECREASE":
        return +1  # Higher number = lower priority
    return 0
```

---

### 5. Prompts (`prompts.py`)

#### Agent System Prompt

The system prompt includes:
1. Role definition (clinical reasoning agent)
2. Tool descriptions (injected from docstrings)
3. Response format specification
4. Important guidelines for treatment failure detection

```python
AGENT_SYSTEM_PROMPT = """You are a clinical reasoning agent...

## Available Tools

{tool_descriptions}

## Response Format

### Option 1: Make a tool call
THOUGHT: [reasoning]
TOOL_CALL:
{"tool": "...", "arguments": {...}}

### Option 2: Final assessment
THOUGHT: [final reasoning]
FINAL_ASSESSMENT: [assessment]
RISK_ADJUSTMENT: [INCREASE|DECREASE|NONE]
CRITICAL_FINDINGS: ["finding1", "finding2"]
"""
```

#### One-Shot Example

Critical for 4B model performance. Demonstrates the PE + anticoagulation failure pattern:

1. See PE → check meds
2. Find warfarin active → check INR
3. INR therapeutic but PE occurred → ANTICOAGULATION FAILURE
4. Risk adjustment: INCREASE

---

### 6. Integration (`agent.py`)

The `TriageAgent` class integrates the agent loop after visual analysis.

#### New Parameters

```python
class TriageAgent:
    def __init__(
        self,
        ...,
        use_agent_mode: bool = AGENT_MODE_ENABLED,
    ):
```

#### New Methods

```python
def _load_fhir_bundle(self, report_path: Path) -> Dict[str, Any]:
    """Load raw FHIR bundle for agent tools."""

def _run_agent_loop(self, patient_id, visual_findings, fhir_bundle) -> Optional[AgentState]:
    """Run ReAct agent for clinical correlation."""
```

#### Modified `process_patient()`

```python
# Step 3: MedGemma visual analysis
analysis = self.analyzer.analyze(images, context_text)

# Step 4: NEW - ReAct agent for clinical correlation
agent_state = self._run_agent_loop(patient_id, analysis.visual_findings, fhir_bundle)

# Step 5: NEW - Apply risk adjustment
if agent_state and agent_state.get("risk_adjustment"):
    adjustment = get_risk_adjustment_value(agent_state["risk_adjustment"])
    final_priority = max(1, min(3, analysis.priority_level + adjustment))
```

#### CLI Flag

```bash
# Disable agent mode
python -m sentinel_x.triage.agent --no-agent

# Normal mode (agent enabled by default)
python -m sentinel_x.triage.agent
```

---

### 7. Output Generation (`output_generator.py`)

#### New Parameter

```python
def generate_triage_result(
    ...,
    agent_state: Optional[AgentState] = None,
) -> Dict[str, Any]:
```

#### Output Schema Extension

```json
{
    "patient_id": "train_1_a_1",
    "priority_level": 1,
    "rationale": "Visual analysis: ... Agent Assessment: ...",
    "visual_findings": "...",
    "findings_summary": "...",
    "agent_reasoning": {
        "iterations": 4,
        "tools_used": ["get_patient_manifest", "check_medication_status", "get_recent_labs"],
        "critical_findings": ["Anticoagulation failure", "PE on therapeutic warfarin"],
        "risk_adjustment": "INCREASE",
        "final_assessment": "Acute PE despite therapeutic anticoagulation...",
        "errors": null
    }
}
```

---

### 8. Configuration (`config.py`)

```python
# =============================================================================
# ReAct Agent Configuration
# =============================================================================

AGENT_MODE_ENABLED = True          # Enable/disable agent mode
AGENT_MAX_ITERATIONS = 5           # Max reasoning iterations
AGENT_TOOL_CALL_TEMPERATURE = 0.0  # Deterministic JSON output
AGENT_MAX_TOKENS_PER_TURN = 512    # Max tokens per agent turn
TOOL_LAB_LOOKBACK_DAYS = 90        # Lab value lookback period

# Risk adjustment values
RISK_ADJUSTMENT_INCREASE = -1      # Decrease priority number = increase urgency
RISK_ADJUSTMENT_DECREASE = 1       # Increase priority number = decrease urgency
RISK_ADJUSTMENT_NONE = 0
```

---

## Example: Silent Failure Detection

### Scenario: PE Patient on Warfarin

**Input Visual Finding**:
```
"Filling defect in right pulmonary artery consistent with acute pulmonary embolism"
```

**Agent Investigation**:

```
THOUGHT: I see an acute PE. Need to check anticoagulation status.

TOOL_CALL: {"tool": "get_patient_manifest", "arguments": {}}

OBSERVATION: {"resource_counts": {"MedicationRequest": 3}, "available_lab_categories": ["Coag"]}

THOUGHT: Patient has medications and coag labs. Checking anticoagulation.

TOOL_CALL: {"tool": "check_medication_status", "arguments": {"medication_name": "anticoag"}}

OBSERVATION: {"found": true, "medications": [{"name": "Warfarin 5mg", "status": "active"}], "is_currently_active": true}

THOUGHT: CRITICAL - Patient on active warfarin. Checking INR.

TOOL_CALL: {"tool": "get_recent_labs", "arguments": {"category": "Coag"}}

OBSERVATION: {"values": [{"name": "INR", "value": 2.3, "flag": "normal"}]}

THOUGHT: Therapeutic INR but developed PE = anticoagulation failure.

FINAL_ASSESSMENT: Acute PE despite therapeutic anticoagulation (INR 2.3).
RISK_ADJUSTMENT: INCREASE
CRITICAL_FINDINGS: ["Anticoagulation failure", "PE on therapeutic warfarin"]
```

**Result**:
- Original priority: 2 (HIGH RISK)
- Agent adjustment: INCREASE (-1)
- Final priority: 1 (CRITICAL)

---

## Testing

### Unit Tests

```python
# Test JSON repair
def test_strip_json_decoration():
    text = '```json\n{"key": "value"}\n```'
    assert strip_json_decoration(text) == '{"key": "value"}'

def test_repair_json():
    text = '{"found": True, "value": None,}'
    assert repair_json(text) == '{"found": true, "value": null}'

# Test tool extraction
def test_extract_tool_call():
    text = 'TOOL_CALL:\n{"tool": "get_labs", "arguments": {"category": "Coag"}}'
    result = extract_tool_call(text)
    assert result["tool"] == "get_labs"
```

### Integration Tests

```python
# Test tool execution with sample FHIR
sample_bundle = {
    "resourceType": "Bundle",
    "entry": [
        {"resource": {"resourceType": "Patient", "gender": "male", "birthDate": "1957-01-15"}},
        {"resource": {"resourceType": "MedicationRequest", "medicationCodeableConcept": {"text": "Warfarin 5mg"}, "status": "active"}}
    ]
}

result = check_medication_status(sample_bundle, "warfarin")
assert result["found"] == True
assert result["is_currently_active"] == True
```

---

## Future Enhancements

1. **Additional Tools**:
   - `get_imaging_history()` - prior CT/MRI findings
   - `check_allergies()` - contrast allergy check
   - `get_vital_signs()` - recent vitals

2. **Multi-Agent Collaboration**:
   - Specialist agents for cardiology, pulmonology
   - Consensus mechanism for complex cases

3. **Learning from Feedback**:
   - Track radiologist overrides
   - Adjust tool usage patterns

4. **Performance Optimization**:
   - Cache common FHIR queries
   - Parallel tool execution where possible
