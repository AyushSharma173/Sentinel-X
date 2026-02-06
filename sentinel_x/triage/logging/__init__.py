"""Logging package for Sentinel-X tracing and debugging.

This package provides comprehensive logging infrastructure for debugging
FHIR context extraction.

Components:
- formatters: JSON and human-readable log formatters
- handlers: Session-based file handlers
- trace_context: Context managers for patient/iteration tracing
- fhir_trace_logger: Specialized logger for FHIR extraction events
- log_analyzer: Post-run analysis utilities
"""

from .formatters import (
    JSONLogFormatter,
    HumanReadableFormatter,
    format_trace_event,
)
from .handlers import (
    SessionManager,
    SessionFileHandler,
    FHIRTraceHandler,
    SummaryHandler,
    setup_session_handlers,
    create_session_id,
    get_sessions_dir,
)
from .trace_context import (
    TraceContextFilter,
    patient_trace_context,
    iteration_trace_context,
    tool_execution_context,
    fhir_extraction_context,
    get_current_context,
    push_context,
    pop_context,
)
from .log_analyzer import (
    LogAnalyzer,
    PatientSummary,
    SessionSummary,
    find_latest_session,
)
from .fhir_trace_logger import (
    FHIRTraceLogger,
    get_fhir_trace_logger,
    initialize_fhir_trace_logger,
)
from .fhir_context_text_logger import (
    FHIRContextTextLogger,
    get_fhir_context_text_logger,
)

__all__ = [
    # Formatters
    "JSONLogFormatter",
    "HumanReadableFormatter",
    "format_trace_event",
    # Handlers
    "SessionManager",
    "SessionFileHandler",
    "FHIRTraceHandler",
    "SummaryHandler",
    "setup_session_handlers",
    "create_session_id",
    "get_sessions_dir",
    # Trace context
    "TraceContextFilter",
    "patient_trace_context",
    "iteration_trace_context",
    "tool_execution_context",
    "fhir_extraction_context",
    "get_current_context",
    "push_context",
    "pop_context",
    # Analysis
    "LogAnalyzer",
    "PatientSummary",
    "SessionSummary",
    "find_latest_session",
    # Specialized loggers
    "FHIRTraceLogger",
    "get_fhir_trace_logger",
    "initialize_fhir_trace_logger",
    # Text-based context logger
    "FHIRContextTextLogger",
    "get_fhir_context_text_logger",
]
