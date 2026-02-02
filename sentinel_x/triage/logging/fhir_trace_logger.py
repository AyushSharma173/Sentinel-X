"""Specialized logger for FHIR extraction events.

This module provides a structured logger specifically designed for
tracing FHIR context extraction, capturing resource parsing details,
condition extraction, and data mapping processes.
"""

import logging
from typing import Any, Dict, List, Optional

from .formatters import JSONLogFormatter, HumanReadableFormatter
from .handlers import SessionManager, FHIRTraceHandler, SummaryHandler
from .trace_context import TraceContextFilter


class FHIRTraceLogger:
    """Specialized logger for FHIR extraction tracing.

    Provides methods for logging specific FHIR parsing events with
    structured data that can be analyzed after runs.
    """

    def __init__(
        self,
        name: str = "sentinel_x.fhir_trace",
        enable_summary: bool = True,
    ):
        """Initialize the FHIR trace logger.

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
        json_handler = FHIRTraceHandler()
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
            event_type: Type of FHIR event
            **kwargs: Additional event fields
        """
        extra = {"event_type": event_type, **kwargs}
        self.logger.log(level, message, extra=extra)

        # Also log to summary at INFO level
        if self.summary_logger and level >= logging.INFO:
            self.summary_logger.log(level, message, extra=extra)

    def log_bundle_received(
        self,
        patient_id: str,
        resource_types: Dict[str, int],
        entry_count: int,
    ) -> None:
        """Log receipt and analysis of FHIR bundle structure.

        Args:
            patient_id: Patient being processed
            resource_types: Map of resource type to count
            entry_count: Total number of entries in bundle
        """
        self._log(
            logging.INFO,
            f"Received FHIR bundle: {entry_count} entries",
            event_type="BUNDLE_RECEIVED",
            patient_id=patient_id,
            resource_types=resource_types,
            entry_count=entry_count,
        )

    def log_demographics_extracted(
        self,
        patient_id: str,
        age: Optional[int],
        gender: Optional[str],
        source_field: str,
    ) -> None:
        """Log extraction of patient demographics.

        Args:
            patient_id: Patient being processed
            age: Extracted age
            gender: Extracted gender
            source_field: Field where data was found
        """
        demo_str = []
        if age is not None:
            demo_str.append(f"{age}yo")
        if gender:
            demo_str.append(gender)

        self._log(
            logging.INFO,
            f"Demographics: {' '.join(demo_str) or 'none found'}",
            event_type="DEMOGRAPHICS_EXTRACTED",
            patient_id=patient_id,
            age=age,
            gender=gender,
            source_field=source_field,
        )

    def log_condition_extracted(
        self,
        patient_id: str,
        condition: str,
        source_field: str,
        coding_system: Optional[str] = None,
        coding_code: Optional[str] = None,
    ) -> None:
        """Log extraction of an individual condition.

        Args:
            patient_id: Patient being processed
            condition: The condition text
            source_field: Field where condition was found
            coding_system: Coding system if available
            coding_code: Code if available
        """
        self._log(
            logging.DEBUG,
            f"Condition: {condition}",
            event_type="CONDITION_EXTRACTED",
            patient_id=patient_id,
            condition=condition,
            source_field=source_field,
            coding_system=coding_system,
            coding_code=coding_code,
        )

    def log_conditions_summary(
        self,
        patient_id: str,
        conditions: List[str],
        count: int,
    ) -> None:
        """Log summary of all conditions extracted.

        Args:
            patient_id: Patient being processed
            conditions: List of all conditions
            count: Total count
        """
        self._log(
            logging.INFO,
            f"Extracted {count} conditions",
            event_type="CONDITIONS_SUMMARY",
            patient_id=patient_id,
            conditions=conditions,
            count=count,
        )

    def log_medication_extracted(
        self,
        patient_id: str,
        medication: str,
        source_field: str,
        resource_type: str,
    ) -> None:
        """Log extraction of an individual medication.

        Args:
            patient_id: Patient being processed
            medication: The medication name
            source_field: Field where medication was found
            resource_type: MedicationStatement or MedicationRequest
        """
        self._log(
            logging.DEBUG,
            f"Medication: {medication}",
            event_type="MEDICATION_EXTRACTED",
            patient_id=patient_id,
            medication=medication,
            source_field=source_field,
            resource_type=resource_type,
        )

    def log_medications_summary(
        self,
        patient_id: str,
        medications: List[str],
        count: int,
    ) -> None:
        """Log summary of all medications extracted.

        Args:
            patient_id: Patient being processed
            medications: List of all medications
            count: Total count
        """
        self._log(
            logging.INFO,
            f"Extracted {count} medications",
            event_type="MEDICATIONS_SUMMARY",
            patient_id=patient_id,
            medications=medications,
            count=count,
        )

    def log_risk_factors_summary(
        self,
        patient_id: str,
        risk_factors: List[str],
        count: int,
    ) -> None:
        """Log summary of identified risk factors.

        Args:
            patient_id: Patient being processed
            risk_factors: List of risk factors
            count: Total count
        """
        self._log(
            logging.INFO,
            f"Identified {count} risk factors",
            event_type="RISK_FACTORS_SUMMARY",
            patient_id=patient_id,
            risk_factors=risk_factors,
            count=count,
        )

    def log_report_content_extracted(
        self,
        patient_id: str,
        findings: str,
        impressions: str,
        source_field: str,
    ) -> None:
        """Log extraction of report content (findings/impressions).

        Args:
            patient_id: Patient being processed
            findings: Extracted findings text
            impressions: Extracted impressions text
            source_field: Field where content was found
        """
        findings_len = len(findings) if findings else 0
        impressions_len = len(impressions) if impressions else 0

        self._log(
            logging.INFO,
            f"Report content: findings={findings_len} chars, impressions={impressions_len} chars",
            event_type="REPORT_CONTENT_EXTRACTED",
            patient_id=patient_id,
            findings=findings,
            impressions=impressions,
            source_field=source_field,
            findings_length=findings_len,
            impressions_length=impressions_len,
        )

    def log_context_complete(
        self,
        patient_id: str,
        age: Optional[int],
        gender: Optional[str],
        conditions_count: int,
        medications_count: int,
        risk_factors_count: int,
        has_findings: bool,
        has_impressions: bool,
    ) -> None:
        """Log completion of context extraction.

        Args:
            patient_id: Patient being processed
            age: Patient age
            gender: Patient gender
            conditions_count: Number of conditions
            medications_count: Number of medications
            risk_factors_count: Number of risk factors
            has_findings: Whether findings were extracted
            has_impressions: Whether impressions were extracted
        """
        summary_parts = []
        if age:
            summary_parts.append(f"{age}yo")
        if gender:
            summary_parts.append(gender)
        summary_parts.append(f"{conditions_count} conditions")
        summary_parts.append(f"{medications_count} meds")
        summary_parts.append(f"{risk_factors_count} risk factors")

        self._log(
            logging.INFO,
            f"Context complete: {', '.join(summary_parts)}",
            event_type="CONTEXT_COMPLETE",
            patient_id=patient_id,
            age=age,
            gender=gender,
            conditions_count=conditions_count,
            medications_count=medications_count,
            risk_factors_count=risk_factors_count,
            has_findings=has_findings,
            has_impressions=has_impressions,
        )

    def log_extraction_failed(
        self,
        patient_id: str,
        resource_type: str,
        error: str,
    ) -> None:
        """Log a failed extraction attempt.

        Args:
            patient_id: Patient being processed
            resource_type: Type of resource being extracted
            error: Error message
        """
        self._log(
            logging.WARNING,
            f"Extraction failed for {resource_type}: {error}",
            event_type="EXTRACTION_FAILED",
            patient_id=patient_id,
            resource_type=resource_type,
            error=error,
        )

    def log_parse_error(
        self,
        patient_id: str,
        error: str,
        source: Optional[str] = None,
    ) -> None:
        """Log a parsing error.

        Args:
            patient_id: Patient being processed
            error: Error message
            source: Source file or data being parsed
        """
        self._log(
            logging.ERROR,
            f"Parse error: {error}",
            event_type="PARSE_ERROR",
            patient_id=patient_id,
            error=error,
            source=source,
        )


# Global instance for convenience
_fhir_trace_logger: Optional[FHIRTraceLogger] = None


def get_fhir_trace_logger() -> FHIRTraceLogger:
    """Get the global FHIR trace logger instance.

    Returns:
        The FHIRTraceLogger instance
    """
    global _fhir_trace_logger
    if _fhir_trace_logger is None:
        _fhir_trace_logger = FHIRTraceLogger()
    return _fhir_trace_logger


def initialize_fhir_trace_logger(session_id: Optional[str] = None) -> str:
    """Initialize the global FHIR trace logger.

    Args:
        session_id: Optional specific session ID

    Returns:
        The session ID
    """
    logger = get_fhir_trace_logger()
    return logger.initialize_session(session_id)
