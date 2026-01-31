"""Agent state management for the ReAct triage agent."""

from typing import Any, Dict, List, Optional, TypedDict


class ToolCall(TypedDict):
    """Represents a single tool call from the agent."""
    tool: str
    arguments: Dict[str, Any]


class AgentMessage(TypedDict):
    """A message in the agent conversation history."""
    role: str  # "system", "user", "assistant", "observation"
    content: str


class AgentState(TypedDict):
    """State maintained throughout the ReAct agent loop.

    This TypedDict tracks the complete state of the agent as it reasons
    through clinical context to determine risk adjustments.
    """
    # Core state
    patient_id: str
    visual_findings: str
    fhir_bundle: Dict[str, Any]

    # Loop control
    iteration: int
    should_stop: bool
    max_iterations: int

    # Message history for context window
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


def create_initial_state(
    patient_id: str,
    visual_findings: str,
    fhir_bundle: Dict[str, Any],
    max_iterations: int = 5,
) -> AgentState:
    """Create the initial agent state.

    Args:
        patient_id: Patient identifier
        visual_findings: Visual findings from MedGemma CT analysis
        fhir_bundle: FHIR Bundle containing patient clinical data
        max_iterations: Maximum number of agent iterations

    Returns:
        Initialized AgentState
    """
    return AgentState(
        patient_id=patient_id,
        visual_findings=visual_findings,
        fhir_bundle=fhir_bundle,
        iteration=0,
        should_stop=False,
        max_iterations=max_iterations,
        messages=[],
        tools_used=[],
        tool_calls=[],
        tool_results=[],
        final_assessment=None,
        risk_adjustment=None,
        critical_findings=[],
        errors=[],
    )


def format_agent_trace(state: AgentState) -> Dict[str, Any]:
    """Format the agent state into a trace for output.

    Args:
        state: The final agent state

    Returns:
        Formatted trace dictionary for JSON output
    """
    return {
        "iterations": state["iteration"],
        "tools_used": state["tools_used"],
        "critical_findings": state["critical_findings"],
        "risk_adjustment": state["risk_adjustment"],
        "final_assessment": state["final_assessment"],
        "errors": state["errors"] if state["errors"] else None,
    }
