"""Simple text-based logger for FHIR context retrieval.

This module provides a human-readable text logger that captures FHIR context
retrieval. Log files are organized per-patient in the
fhir_context_retreival_logs directory.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class FHIRContextTextLogger:
    """Text-based logger for FHIR context retrieval.

    Creates human-readable log files organized per-patient that capture:
    - Visual findings that triggered the investigation
    - Each FHIR query (tool name, arguments, results)
    - Final assessment and risk adjustment
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize the text logger.

        Args:
            log_dir: Directory for log files. Defaults to
                     sentinel_x/logs/fhir_context_retreival_logs/
        """
        if log_dir is None:
            # Default to sentinel_x/logs/fhir_context_retreival_logs/
            module_dir = Path(__file__).parent.parent.parent
            log_dir = module_dir / "logs" / "fhir_context_retreival_logs"

        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current patient state
        self._current_patient_id: Optional[str] = None
        self._current_file: Optional[Path] = None
        self._query_count: int = 0

    def start_patient(self, patient_id: str, visual_findings: str) -> None:
        """Start logging for a new patient.

        Creates/overwrites the patient's log file and writes the header.

        Args:
            patient_id: Patient identifier
            visual_findings: Visual findings from CT analysis that triggered investigation
        """
        self._current_patient_id = patient_id
        self._current_file = self.log_dir / f"{patient_id}.txt"
        self._query_count = 0

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write header (overwrite existing file)
        with open(self._current_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("FHIR CONTEXT RETRIEVAL LOG\n")
            f.write(f"Patient: {patient_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n")
            f.write("\n")
            f.write("VISUAL FINDINGS (Trigger):\n")
            f.write(visual_findings + "\n")
            f.write("\n")

    def log_query(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> None:
        """Log a FHIR query (tool execution).

        Args:
            tool_name: Name of the tool that was called
            tool_args: Arguments passed to the tool
            tool_result: Result returned by the tool
        """
        if self._current_file is None:
            return

        self._query_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        with open(self._current_file, "a") as f:
            f.write("-" * 80 + "\n")
            f.write(f"QUERY {self._query_count} [{timestamp}]\n")
            f.write(f"Tool: {tool_name}\n")
            f.write(f"Arguments: {self._format_args(tool_args)}\n")
            f.write("\n")
            f.write("RESULT:\n")
            f.write(self._format_result(tool_result))
            f.write("\n")

    def end_patient(
        self,
        risk_adjustment: Optional[str],
        critical_findings: List[str],
        assessment: Optional[str],
    ) -> None:
        """Write the final assessment and close the patient log.

        Args:
            risk_adjustment: Risk adjustment (INCREASE, DECREASE, NONE)
            critical_findings: List of critical findings
            assessment: Final assessment text
        """
        if self._current_file is None:
            return

        with open(self._current_file, "a") as f:
            f.write("=" * 80 + "\n")
            f.write("FINAL ASSESSMENT\n")
            f.write("-" * 80 + "\n")
            f.write(f"Risk Adjustment: {risk_adjustment or 'NONE'}\n")

            if critical_findings:
                f.write("Critical Findings:\n")
                for finding in critical_findings:
                    f.write(f"  - {finding}\n")
            else:
                f.write("Critical Findings: None\n")

            f.write("\n")
            f.write("Assessment:\n")
            f.write(assessment or "No assessment provided")
            f.write("\n")
            f.write("=" * 80 + "\n")

        # Reset state
        self._current_patient_id = None
        self._current_file = None
        self._query_count = 0

    def _format_args(self, args: Dict[str, Any]) -> str:
        """Format tool arguments for display.

        Args:
            args: Tool arguments dictionary

        Returns:
            Formatted string representation
        """
        if not args:
            return "{}"
        return json.dumps(args, default=str)

    def _format_result(self, result: Dict[str, Any], indent: int = 2) -> str:
        """Format tool result for human-readable display.

        Args:
            result: Tool result dictionary
            indent: Number of spaces for indentation

        Returns:
            Formatted string representation
        """
        lines = []
        prefix = " " * indent

        for key, value in result.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"{prefix}  {sub_key}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        # Format dict items in list
                        item_str = ", ".join(f"{k}={v}" for k, v in item.items())
                        lines.append(f"{prefix}  - {item_str}")
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")

        return "\n".join(lines) + "\n"


# Global instance for convenience
_fhir_context_text_logger: Optional[FHIRContextTextLogger] = None


def get_fhir_context_text_logger() -> FHIRContextTextLogger:
    """Get the global FHIR context text logger instance.

    Returns:
        The FHIRContextTextLogger instance
    """
    global _fhir_context_text_logger
    if _fhir_context_text_logger is None:
        _fhir_context_text_logger = FHIRContextTextLogger()
    return _fhir_context_text_logger
