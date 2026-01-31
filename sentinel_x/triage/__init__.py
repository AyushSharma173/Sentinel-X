"""
Sentinel-X Triage Agent

An intelligent triage system that monitors incoming CT scans, analyzes them
using MedGemma alongside FHIR clinical context, and produces priority-sorted
worklists for radiologist review.

The system includes a ReAct (Reason+Act) agent loop that dynamically
investigates FHIR patient data to detect "silent failures" like treatment
failures (e.g., PE while on anticoagulation).
"""

from .agent import TriageAgent
from .agent_loop import ReActAgentLoop
from .state import AgentState, create_initial_state, format_agent_trace
from .tools import get_all_tools, get_tool, get_tool_descriptions

__all__ = [
    "TriageAgent",
    "ReActAgentLoop",
    "AgentState",
    "create_initial_state",
    "format_agent_trace",
    "get_all_tools",
    "get_tool",
    "get_tool_descriptions",
]
