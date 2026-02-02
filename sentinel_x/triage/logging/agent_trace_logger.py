"""Specialized logger for agent loop events.

This module provides a structured logger specifically designed for
tracing the ReAct agent loop, capturing tool calls, model responses,
and decision-making processes.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .formatters import JSONLogFormatter, HumanReadableFormatter
from .handlers import SessionManager, AgentTraceHandler, SummaryHandler
from .trace_context import TraceContextFilter


class AgentTraceLogger:
    """Specialized logger for agent loop tracing.

    Provides methods for logging specific agent events with
    structured data that can be analyzed after runs.
    """

    def __init__(
        self,
        name: str = "sentinel_x.agent_trace",
        enable_summary: bool = True,
    ):
        """Initialize the agent trace logger.

        Args:
            name: Logger name
            enable_summary: Whether to also log to summary file
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Add trace context filter
        self.logger.addFilter(TraceContextFilter())

        # Track current session
        self._session_initialized = False
        self._enable_summary = enable_summary

        # For summary logger
        self.summary_logger: Optional[logging.Logger] = None
        if enable_summary:
            self.summary_logger = logging.getLogger(f"{name}.summary")
            self.summary_logger.setLevel(logging.INFO)
            self.summary_logger.addFilter(TraceContextFilter())

    def initialize_session(self, session_id: Optional[str] = None) -> str:
        """Initialize logging handlers for a session.

        Args:
            session_id: Optional specific session ID

        Returns:
            The session ID
        """
        if self._session_initialized:
            return SessionManager.get_instance().session_id

        # Initialize session manager
        session_manager = SessionManager.get_instance()
        session_id = session_manager.initialize(session_id)

        # Add JSON handler
        json_handler = AgentTraceHandler()
        json_handler.setFormatter(JSONLogFormatter())
        json_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(json_handler)

        # Add summary handler if enabled
        if self._enable_summary and self.summary_logger:
            summary_handler = SummaryHandler()
            summary_handler.setFormatter(HumanReadableFormatter())
            summary_handler.setLevel(logging.INFO)
            self.summary_logger.addHandler(summary_handler)

        self._session_initialized = True
        return session_id

    def _log(
        self,
        level: int,
        message: str,
        event_type: str,
        **kwargs: Any
    ) -> None:
        """Internal logging method.

        Args:
            level: Log level
            message: Log message
            event_type: Type of agent event
            **kwargs: Additional event fields
        """
        extra = {"event_type": event_type, **kwargs}
        self.logger.log(level, message, extra=extra)

        # Also log to summary at INFO level
        if self.summary_logger and level >= logging.INFO:
            self.summary_logger.log(level, message, extra=extra)

    def log_iteration_start(
        self,
        patient_id: str,
        iteration: int,
        max_iterations: int,
        message_count: int,
    ) -> None:
        """Log the start of an agent iteration.

        Args:
            patient_id: Patient being processed
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            message_count: Number of messages in conversation
        """
        self._log(
            logging.INFO,
            f"Starting iteration {iteration}/{max_iterations}",
            event_type="ITERATION_START",
            patient_id=patient_id,
            iteration=iteration,
            max_iterations=max_iterations,
            message_count=message_count,
        )

    def log_prompt_sent(
        self,
        patient_id: str,
        iteration: int,
        prompt: str,
        char_count: Optional[int] = None,
    ) -> None:
        """Log a prompt sent to the model.

        Args:
            patient_id: Patient being processed
            iteration: Current iteration number
            prompt: The full prompt text
            char_count: Character count (computed if not provided)
        """
        char_count = char_count or len(prompt)
        self._log(
            logging.DEBUG,
            f"Sent prompt ({char_count} chars)",
            event_type="PROMPT_SENT",
            patient_id=patient_id,
            iteration=iteration,
            prompt=prompt,
            char_count=char_count,
        )

    def log_response_received(
        self,
        patient_id: str,
        iteration: int,
        response: str,
        duration_ms: int,
        char_count: Optional[int] = None,
    ) -> None:
        """Log a model response received.

        Args:
            patient_id: Patient being processed
            iteration: Current iteration number
            response: The full response text
            duration_ms: Generation time in milliseconds
            char_count: Character count (computed if not provided)
        """
        char_count = char_count or len(response)
        self._log(
            logging.INFO,
            f"Received response ({char_count} chars, {duration_ms}ms)",
            event_type="RESPONSE_RECEIVED",
            patient_id=patient_id,
            iteration=iteration,
            response=response,
            duration_ms=duration_ms,
            char_count=char_count,
        )

    def log_tool_call_extracted(
        self,
        patient_id: str,
        iteration: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        raw_text: Optional[str] = None,
    ) -> None:
        """Log a successfully extracted tool call.

        Args:
            patient_id: Patient being processed
            iteration: Current iteration number
            tool_name: Name of the tool called
            tool_args: Arguments to the tool
            raw_text: Raw text that was parsed
        """
        self._log(
            logging.INFO,
            f"Extracted tool call: {tool_name}",
            event_type="TOOL_CALL_EXTRACTED",
            patient_id=patient_id,
            iteration=iteration,
            tool_name=tool_name,
            tool_args=tool_args,
            raw_text=raw_text,
        )

    def log_tool_call_failed(
        self,
        patient_id: str,
        iteration: int,
        raw_text: str,
        error: Optional[str] = None,
    ) -> None:
        """Log a failed tool call extraction.

        Args:
            patient_id: Patient being processed
            iteration: Current iteration number
            raw_text: Raw text that failed to parse
            error: Error message if available
        """
        self._log(
            logging.WARNING,
            "Failed to extract tool call from response",
            event_type="TOOL_CALL_FAILED",
            patient_id=patient_id,
            iteration=iteration,
            raw_text=raw_text,
            error=error,
        )

    def log_tool_execution(
        self,
        patient_id: str,
        iteration: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
        duration_ms: int,
    ) -> None:
        """Log a tool execution and result.

        Args:
            patient_id: Patient being processed
            iteration: Current iteration number
            tool_name: Name of the tool
            tool_args: Arguments passed
            tool_result: Result from the tool
            duration_ms: Execution time in milliseconds
        """
        self._log(
            logging.INFO,
            f"Executed {tool_name} ({duration_ms}ms)",
            event_type="TOOL_EXECUTION",
            patient_id=patient_id,
            iteration=iteration,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            duration_ms=duration_ms,
        )

    def log_observation_added(
        self,
        patient_id: str,
        iteration: int,
        observation: str,
    ) -> None:
        """Log an observation added to the conversation.

        Args:
            patient_id: Patient being processed
            iteration: Current iteration number
            observation: The observation text
        """
        self._log(
            logging.DEBUG,
            f"Added observation ({len(observation)} chars)",
            event_type="OBSERVATION_ADDED",
            patient_id=patient_id,
            iteration=iteration,
            observation=observation,
            char_count=len(observation),
        )

    def log_final_assessment_extracted(
        self,
        patient_id: str,
        iteration: int,
        assessment: str,
        risk_adjustment: str,
        critical_findings: List[str],
    ) -> None:
        """Log extraction of final assessment.

        Args:
            patient_id: Patient being processed
            iteration: Current iteration number
            assessment: The assessment text
            risk_adjustment: INCREASE/DECREASE/NONE
            critical_findings: List of critical findings
        """
        self._log(
            logging.INFO,
            f"Final assessment: risk_adjustment={risk_adjustment}",
            event_type="FINAL_ASSESSMENT_EXTRACTED",
            patient_id=patient_id,
            iteration=iteration,
            assessment=assessment,
            risk_adjustment=risk_adjustment,
            findings=critical_findings,
        )

    def log_agent_complete(
        self,
        patient_id: str,
        iterations: int,
        tools_used: List[str],
        errors: List[str],
        duration_ms: int,
        risk_adjustment: Optional[str] = None,
    ) -> None:
        """Log completion of agent processing.

        Args:
            patient_id: Patient being processed
            iterations: Total iterations used
            tools_used: List of tools that were called
            errors: List of errors encountered
            duration_ms: Total processing time
            risk_adjustment: Final risk adjustment
        """
        self._log(
            logging.INFO,
            f"Agent complete: {iterations} iterations, {len(tools_used)} tool calls",
            event_type="AGENT_COMPLETE",
            patient_id=patient_id,
            iterations=iterations,
            tools_used=tools_used,
            errors=errors,
            duration_ms=duration_ms,
            risk_adjustment=risk_adjustment,
        )


# Global instance for convenience
_agent_trace_logger: Optional[AgentTraceLogger] = None


def get_agent_trace_logger() -> AgentTraceLogger:
    """Get the global agent trace logger instance.

    Returns:
        The AgentTraceLogger instance
    """
    global _agent_trace_logger
    if _agent_trace_logger is None:
        _agent_trace_logger = AgentTraceLogger()
    return _agent_trace_logger


def initialize_agent_trace_logger(session_id: Optional[str] = None) -> str:
    """Initialize the global agent trace logger.

    Args:
        session_id: Optional specific session ID

    Returns:
        The session ID
    """
    logger = get_agent_trace_logger()
    return logger.initialize_session(session_id)
