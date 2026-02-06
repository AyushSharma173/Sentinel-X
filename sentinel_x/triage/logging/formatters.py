"""JSON and human-readable log formatters for Sentinel-X tracing."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional


class JSONLogFormatter(logging.Formatter):
    """Formatter that outputs structured JSON log lines (JSONL format).

    Each log record is formatted as a single JSON object on one line,
    suitable for machine parsing and analysis tools.
    """

    def __init__(self, include_extra: bool = True):
        """Initialize the JSON formatter.

        Args:
            include_extra: Whether to include extra fields from log records
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON string (single line)
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add standard fields
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        if self.include_extra:
            # Common trace fields
            for field in [
                "event_type", "patient_id", "iteration", "session_id",
                "tool_name", "tool_args", "tool_result", "duration_ms",
                "char_count", "message_count", "resource_type", "entry_count",
                "source_field", "assessment", "findings",
                "error", "raw_text", "prompt", "response", "conditions",
                "medications", "demographics", "age", "gender"
            ]:
                if hasattr(record, field):
                    value = getattr(record, field)
                    if value is not None:
                        log_data[field] = value

        return json.dumps(log_data, default=str, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """Formatter that outputs human-readable summary logs.

    Provides a condensed, readable format suitable for console output
    and quick visual inspection of logs during debugging.
    """

    # Event type symbols for visual scanning
    EVENT_SYMBOLS = {
        "ITERATION_START": "🔄",
        "PROMPT_SENT": "📤",
        "RESPONSE_RECEIVED": "📥",
        "TOOL_CALL_EXTRACTED": "🔧",
        "TOOL_CALL_FAILED": "❌",
        "TOOL_EXECUTION": "⚡",
        "OBSERVATION_ADDED": "👁️",
        "FINAL_ASSESSMENT_EXTRACTED": "✅",
        "AGENT_COMPLETE": "🏁",
        "BUNDLE_RECEIVED": "📦",
        "DEMOGRAPHICS_EXTRACTED": "👤",
        "CONDITION_EXTRACTED": "🏥",
        "CONDITIONS_SUMMARY": "📋",
        "MEDICATION_EXTRACTED": "💊",
        "MEDICATIONS_SUMMARY": "💊",
        "RISK_FACTORS_SUMMARY": "⚠️",
        "REPORT_CONTENT_EXTRACTED": "📄",
        "CONTEXT_COMPLETE": "✨",
    }

    def __init__(self, use_symbols: bool = True):
        """Initialize the human-readable formatter.

        Args:
            use_symbols: Whether to use emoji symbols for event types
        """
        super().__init__()
        self.use_symbols = use_symbols

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record for human reading.

        Args:
            record: The log record to format

        Returns:
            Formatted string
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Get event type if present
        event_type = getattr(record, "event_type", None)
        patient_id = getattr(record, "patient_id", None)

        # Build prefix
        parts = [f"[{timestamp}]"]

        if patient_id:
            parts.append(f"[{patient_id}]")

        if event_type:
            if self.use_symbols and event_type in self.EVENT_SYMBOLS:
                parts.append(f"{self.EVENT_SYMBOLS[event_type]}")
            parts.append(f"{event_type}")

        prefix = " ".join(parts)

        # Build message with key details
        message = record.getMessage()
        details = []

        # Add relevant context
        if hasattr(record, "iteration"):
            details.append(f"iter={record.iteration}")

        if hasattr(record, "tool_name"):
            details.append(f"tool={record.tool_name}")

        if hasattr(record, "duration_ms"):
            details.append(f"duration={record.duration_ms}ms")

        if hasattr(record, "char_count"):
            details.append(f"chars={record.char_count}")

        if hasattr(record, "message_count"):
            details.append(f"messages={record.message_count}")

        # Combine parts
        if details:
            return f"{prefix}: {message} ({', '.join(details)})"
        else:
            return f"{prefix}: {message}"


def format_trace_event(
    event_type: str,
    patient_id: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Create a structured trace event dictionary.

    This helper function creates a consistent structure for trace events
    that can be logged and later analyzed.

    Args:
        event_type: Type of event (e.g., "TOOL_CALL_EXTRACTED")
        patient_id: Optional patient identifier
        **kwargs: Additional event-specific fields

    Returns:
        Dictionary with structured event data
    """
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
    }

    if patient_id:
        event["patient_id"] = patient_id

    event.update(kwargs)
    return event
