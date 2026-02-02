"""Context managers for patient and iteration tracing.

These context managers provide structured logging context that
automatically attaches relevant metadata to all log records
within their scope.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

# Thread-local storage for trace context
import threading

_trace_context = threading.local()


def get_current_context() -> Dict[str, Any]:
    """Get the current trace context.

    Returns:
        Dictionary with current trace context fields
    """
    if not hasattr(_trace_context, "stack"):
        _trace_context.stack = []

    # Merge all context frames
    result = {}
    for frame in _trace_context.stack:
        result.update(frame)
    return result


def push_context(**kwargs: Any) -> None:
    """Push a new context frame onto the stack.

    Args:
        **kwargs: Context fields to add
    """
    if not hasattr(_trace_context, "stack"):
        _trace_context.stack = []
    _trace_context.stack.append(kwargs)


def pop_context() -> Dict[str, Any]:
    """Pop the top context frame from the stack.

    Returns:
        The popped context frame
    """
    if not hasattr(_trace_context, "stack") or not _trace_context.stack:
        return {}
    return _trace_context.stack.pop()


class TraceContextFilter(logging.Filter):
    """Logging filter that attaches trace context to log records.

    Add this filter to loggers to automatically include
    patient_id, iteration, and other trace context in all records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add trace context to the log record.

        Args:
            record: The log record to augment

        Returns:
            True (always allow the record)
        """
        context = get_current_context()
        for key, value in context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


@contextmanager
def patient_trace_context(
    patient_id: str,
    logger: Optional[logging.Logger] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for patient-level tracing.

    Automatically adds patient_id to all log records within the context
    and logs the start/end of patient processing.

    Args:
        patient_id: The patient identifier
        logger: Optional logger for start/end messages

    Yields:
        Dictionary that can be used to add additional context

    Example:
        with patient_trace_context("CT-001", logger) as ctx:
            # All logs here will have patient_id="CT-001"
            logger.info("Processing patient")
            ctx["custom_field"] = "value"  # Add custom context
    """
    context = {"patient_id": patient_id}
    push_context(**context)

    start_time = time.time()

    if logger:
        logger.info(
            f"Starting patient processing",
            extra={"event_type": "PATIENT_START", "patient_id": patient_id}
        )

    try:
        yield context
    finally:
        duration_ms = int((time.time() - start_time) * 1000)

        if logger:
            logger.info(
                f"Completed patient processing",
                extra={
                    "event_type": "PATIENT_COMPLETE",
                    "patient_id": patient_id,
                    "duration_ms": duration_ms,
                }
            )

        pop_context()


@contextmanager
def iteration_trace_context(
    iteration: int,
    max_iterations: int,
    logger: Optional[logging.Logger] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for agent iteration tracing.

    Automatically adds iteration number to all log records within
    the context and logs iteration start/end.

    Args:
        iteration: Current iteration number
        max_iterations: Maximum iterations allowed
        logger: Optional logger for start/end messages

    Yields:
        Dictionary that can be used to add additional context

    Example:
        with iteration_trace_context(1, 5, logger) as ctx:
            # All logs here will have iteration=1
            logger.info("Running model")
    """
    context = {"iteration": iteration, "max_iterations": max_iterations}
    push_context(**context)

    start_time = time.time()

    if logger:
        logger.info(
            f"Iteration {iteration}/{max_iterations}",
            extra={"event_type": "ITERATION_START", **context}
        )

    try:
        yield context
    finally:
        duration_ms = int((time.time() - start_time) * 1000)
        context["duration_ms"] = duration_ms

        if logger:
            logger.info(
                f"Iteration {iteration} complete",
                extra={"event_type": "ITERATION_END", **context}
            )

        pop_context()


@contextmanager
def tool_execution_context(
    tool_name: str,
    tool_args: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for tool execution tracing.

    Tracks tool execution time and logs results.

    Args:
        tool_name: Name of the tool being executed
        tool_args: Arguments passed to the tool
        logger: Optional logger for execution logs

    Yields:
        Dictionary to store result information

    Example:
        with tool_execution_context("get_conditions", {}, logger) as ctx:
            result = execute_tool()
            ctx["result"] = result
    """
    context = {
        "tool_name": tool_name,
        "tool_args": tool_args,
    }
    push_context(**context)

    start_time = time.time()

    if logger:
        logger.debug(
            f"Executing tool: {tool_name}",
            extra={"event_type": "TOOL_EXECUTION_START", **context}
        )

    try:
        yield context
    finally:
        duration_ms = int((time.time() - start_time) * 1000)
        context["duration_ms"] = duration_ms

        if logger:
            result_preview = str(context.get("result", ""))[:200]
            logger.info(
                f"Tool {tool_name} completed",
                extra={
                    "event_type": "TOOL_EXECUTION",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "duration_ms": duration_ms,
                    "tool_result": context.get("result"),
                }
            )

        pop_context()


@contextmanager
def fhir_extraction_context(
    resource_type: str,
    logger: Optional[logging.Logger] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for FHIR resource extraction tracing.

    Args:
        resource_type: Type of FHIR resource being extracted
        logger: Optional logger for extraction logs

    Yields:
        Dictionary to store extraction results
    """
    context = {"resource_type": resource_type}
    push_context(**context)

    start_time = time.time()

    try:
        yield context
    finally:
        duration_ms = int((time.time() - start_time) * 1000)
        context["duration_ms"] = duration_ms
        pop_context()
