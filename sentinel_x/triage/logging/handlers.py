"""Session-based file handlers for Sentinel-X logging.

This module provides log handlers that organize logs into per-session
directories, making it easy to analyze individual demo runs.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import LOG_DIR


def get_sessions_dir() -> Path:
    """Get the sessions directory path.

    Returns:
        Path to the sessions directory
    """
    sessions_dir = LOG_DIR / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def create_session_id() -> str:
    """Create a unique session ID based on timestamp.

    Returns:
        Session ID in format YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class SessionManager:
    """Manages logging session directories and handlers.

    A session represents a single demo run or processing batch.
    Each session gets its own directory with separate log files
    for different trace types.
    """

    _instance: Optional["SessionManager"] = None
    _session_id: Optional[str] = None
    _session_dir: Optional[Path] = None
    _initialized: bool = False

    def __new__(cls) -> "SessionManager":
        """Singleton pattern for session manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "SessionManager":
        """Get the singleton instance.

        Returns:
            The SessionManager instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
        cls._session_id = None
        cls._session_dir = None
        cls._initialized = False

    def initialize(self, session_id: Optional[str] = None) -> str:
        """Initialize a new logging session.

        Args:
            session_id: Optional specific session ID (auto-generated if None)

        Returns:
            The session ID
        """
        if self._initialized and self._session_id:
            return self._session_id

        self._session_id = session_id or create_session_id()
        self._session_dir = get_sessions_dir() / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

        return self._session_id

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id

    @property
    def session_dir(self) -> Optional[Path]:
        """Get the current session directory."""
        return self._session_dir

    def get_log_path(self, log_name: str) -> Path:
        """Get the path for a specific log file in the current session.

        Args:
            log_name: Name of the log file (e.g., "fhir_trace.jsonl")

        Returns:
            Full path to the log file

        Raises:
            RuntimeError: If session not initialized
        """
        if not self._initialized or not self._session_dir:
            raise RuntimeError("Session not initialized. Call initialize() first.")
        return self._session_dir / log_name


class SessionFileHandler(logging.FileHandler):
    """File handler that writes to session-specific log files.

    This handler automatically creates the session directory structure
    and writes logs to the appropriate file within the session.
    """

    def __init__(
        self,
        log_name: str,
        mode: str = "a",
        encoding: str = "utf-8",
        delay: bool = False,
    ):
        """Initialize the session file handler.

        Args:
            log_name: Name of the log file (e.g., "fhir_trace.jsonl")
            mode: File open mode
            encoding: File encoding
            delay: Whether to delay file opening
        """
        self.log_name = log_name
        self._session_manager = SessionManager.get_instance()

        # Ensure session is initialized
        self._session_manager.initialize()

        # Get the log path
        log_path = self._session_manager.get_log_path(log_name)

        super().__init__(
            filename=str(log_path),
            mode=mode,
            encoding=encoding,
            delay=delay,
        )


class FHIRTraceHandler(SessionFileHandler):
    """Handler specifically for FHIR extraction trace events."""

    def __init__(self):
        """Initialize the FHIR trace handler."""
        super().__init__(log_name="fhir_trace.jsonl")


class SummaryHandler(SessionFileHandler):
    """Handler for human-readable summary logs."""

    def __init__(self):
        """Initialize the summary handler."""
        super().__init__(log_name="summary.log")


def setup_session_handlers(
    fhir_logger: logging.Logger,
    summary_logger: Optional[logging.Logger] = None,
    session_id: Optional[str] = None,
) -> str:
    """Set up session-based handlers for the trace loggers.

    Args:
        fhir_logger: Logger for FHIR trace events
        summary_logger: Optional logger for human-readable summary
        session_id: Optional specific session ID

    Returns:
        The session ID
    """
    from .formatters import JSONLogFormatter, HumanReadableFormatter

    # Initialize session
    session_manager = SessionManager.get_instance()
    session_id = session_manager.initialize(session_id)

    # Set up FHIR trace handler
    fhir_handler = FHIRTraceHandler()
    fhir_handler.setFormatter(JSONLogFormatter())
    fhir_handler.setLevel(logging.DEBUG)
    fhir_logger.addHandler(fhir_handler)
    fhir_logger.setLevel(logging.DEBUG)

    # Set up summary handler if provided
    if summary_logger:
        summary_handler = SummaryHandler()
        summary_handler.setFormatter(HumanReadableFormatter())
        summary_handler.setLevel(logging.INFO)
        summary_logger.addHandler(summary_handler)
        summary_logger.setLevel(logging.INFO)

    return session_id
