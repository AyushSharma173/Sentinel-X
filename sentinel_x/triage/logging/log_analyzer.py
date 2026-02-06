"""Post-run analysis utilities for Sentinel-X logs.

This module provides tools for analyzing logged sessions after
demo runs, including summary generation, filtering, and reporting.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set


@dataclass
class PatientSummary:
    """Summary of processing for a single patient."""
    patient_id: str
    conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class SessionSummary:
    """Summary of an entire logging session."""
    session_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    patient_count: int = 0
    patients: Dict[str, PatientSummary] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)


class LogAnalyzer:
    """Analyzer for Sentinel-X log sessions.

    Provides methods to parse, filter, and summarize log data
    from completed demo runs.
    """

    def __init__(self, session_dir: str | Path):
        """Initialize the analyzer with a session directory.

        Args:
            session_dir: Path to the session directory containing log files
        """
        self.session_dir = Path(session_dir)
        self.session_id = self.session_dir.name

        # Log file paths
        self.fhir_trace_path = self.session_dir / "fhir_trace.jsonl"
        self.summary_path = self.session_dir / "summary.log"

        # Cached data
        self._fhir_events: Optional[List[Dict[str, Any]]] = None

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load events from a JSONL file.

        Args:
            path: Path to the JSONL file

        Returns:
            List of event dictionaries
        """
        events = []
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return events

    @property
    def fhir_events(self) -> List[Dict[str, Any]]:
        """Get all FHIR trace events (cached)."""
        if self._fhir_events is None:
            self._fhir_events = self._load_jsonl(self.fhir_trace_path)
        return self._fhir_events

    def get_patients(self) -> Set[str]:
        """Get set of all patient IDs in the session.

        Returns:
            Set of patient IDs
        """
        patients = set()
        for event in self.fhir_events:
            if "patient_id" in event:
                patients.add(event["patient_id"])
        return patients

    def filter_by_patient(
        self,
        patient_id: str,
    ) -> List[Dict[str, Any]]:
        """Filter events for a specific patient.

        Args:
            patient_id: The patient ID to filter for

        Returns:
            List of events for the patient
        """
        events = [
            e for e in self.fhir_events
            if e.get("patient_id") == patient_id
        ]

        # Sort by timestamp
        events.sort(key=lambda e: e.get("timestamp", ""))
        return events

    def filter_by_event_type(
        self,
        event_types: List[str],
    ) -> List[Dict[str, Any]]:
        """Filter events by event type.

        Args:
            event_types: List of event types to include

        Returns:
            List of matching events
        """
        events = [
            e for e in self.fhir_events
            if e.get("event_type") in event_types
        ]

        events.sort(key=lambda e: e.get("timestamp", ""))
        return events

    def get_failures(self) -> List[Dict[str, Any]]:
        """Get all failure/error events.

        Returns:
            List of error events
        """
        failure_types = [
            "EXTRACTION_FAILED",
            "PARSE_ERROR",
        ]

        failures = []
        for event in self.fhir_events:
            if (
                event.get("event_type") in failure_types
                or "error" in event
                or event.get("level") == "ERROR"
            ):
                failures.append(event)

        return failures

    def get_patient_summary(self, patient_id: str) -> PatientSummary:
        """Generate a summary for a specific patient.

        Args:
            patient_id: The patient ID

        Returns:
            PatientSummary dataclass
        """
        summary = PatientSummary(patient_id=patient_id)
        events = self.filter_by_patient(patient_id)

        for event in events:
            event_type = event.get("event_type", "")

            if event_type == "CONDITIONS_SUMMARY":
                summary.conditions = event.get("conditions", [])

            elif event_type == "MEDICATIONS_SUMMARY":
                summary.medications = event.get("medications", [])

            if "error" in event:
                summary.errors.append(event["error"])

        return summary

    def generate_summary_report(self) -> SessionSummary:
        """Generate a summary report for the entire session.

        Returns:
            SessionSummary dataclass
        """
        summary = SessionSummary(session_id=self.session_id)

        # Get all patients
        patients = self.get_patients()
        summary.patient_count = len(patients)

        # Analyze each patient
        for patient_id in patients:
            patient_summary = self.get_patient_summary(patient_id)
            summary.patients[patient_id] = patient_summary

        # Get errors
        summary.errors = self.get_failures()

        # Extract timestamps
        all_events = self.fhir_events
        if all_events:
            timestamps = [
                e.get("timestamp", "")
                for e in all_events
                if e.get("timestamp")
            ]
            if timestamps:
                timestamps.sort()
                try:
                    summary.start_time = datetime.fromisoformat(
                        timestamps[0].replace("Z", "+00:00")
                    )
                    summary.end_time = datetime.fromisoformat(
                        timestamps[-1].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

        return summary

    def print_patient_summary(self, patient_id: str) -> None:
        """Print a formatted summary for a patient.

        Args:
            patient_id: The patient ID
        """
        summary = self.get_patient_summary(patient_id)

        print(f"\n{'='*60}")
        print(f"Patient: {summary.patient_id}")
        print(f"{'='*60}")

        if summary.conditions:
            print(f"\nConditions:")
            for condition in summary.conditions:
                print(f"  - {condition}")

        if summary.errors:
            print(f"\nErrors:")
            for error in summary.errors:
                print(f"  - {error}")

        print()

    def print_session_summary(self) -> None:
        """Print a formatted summary for the entire session."""
        summary = self.generate_summary_report()

        print(f"\n{'='*60}")
        print(f"Session Summary: {summary.session_id}")
        print(f"{'='*60}")

        if summary.start_time:
            print(f"Start: {summary.start_time.isoformat()}")
        if summary.end_time:
            print(f"End: {summary.end_time.isoformat()}")

        print(f"Patients processed: {summary.patient_count}")
        print(f"Errors: {len(summary.errors)}")

        print(f"\nPatient summaries:")
        for patient_id, patient_summary in summary.patients.items():
            errors = len(patient_summary.errors)
            print(
                f"  {patient_id}"
                + (f", {errors} errors" if errors else "")
            )

        if summary.errors:
            print(f"\nErrors encountered:")
            for error in summary.errors[:5]:  # Show first 5
                patient = error.get("patient_id", "unknown")
                event_type = error.get("event_type", "ERROR")
                msg = error.get("error", error.get("message", "Unknown error"))
                print(f"  [{patient}] {event_type}: {msg}")
            if len(summary.errors) > 5:
                print(f"  ... and {len(summary.errors) - 5} more")

        print()


def find_latest_session(log_dir: Path) -> Optional[Path]:
    """Find the most recent session directory.

    Args:
        log_dir: Base log directory containing sessions/

    Returns:
        Path to the latest session directory, or None
    """
    sessions_dir = log_dir / "sessions"
    if not sessions_dir.exists():
        return None

    sessions = list(sessions_dir.iterdir())
    if not sessions:
        return None

    # Sort by name (timestamp format ensures chronological order)
    sessions.sort(key=lambda p: p.name, reverse=True)
    return sessions[0]
