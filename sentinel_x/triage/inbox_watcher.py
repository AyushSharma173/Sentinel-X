"""Inbox watcher for monitoring incoming CT scans and reports."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from .config import INBOX_POLL_INTERVAL, INBOX_REPORTS_DIR, INBOX_VOLUMES_DIR

logger = logging.getLogger(__name__)


@dataclass
class PatientData:
    """Paired patient data from inbox."""
    patient_id: str
    volume_path: Path
    report_path: Path


class InboxWatcher:
    """Monitors inbox directories for new CT scans and reports."""

    def __init__(
        self,
        volumes_dir: Path = INBOX_VOLUMES_DIR,
        reports_dir: Path = INBOX_REPORTS_DIR,
        poll_interval: float = INBOX_POLL_INTERVAL,
    ):
        """Initialize the inbox watcher.

        Args:
            volumes_dir: Directory containing CT volumes (.nii.gz)
            reports_dir: Directory containing reports (.json)
            poll_interval: Seconds between directory scans
        """
        self.volumes_dir = volumes_dir
        self.reports_dir = reports_dir
        self.poll_interval = poll_interval
        self._processed: Set[str] = set()
        self._running = False

    def _extract_patient_id(self, filename: str) -> str:
        """Extract patient ID from filename.

        Handles formats like:
        - train_1_a_1.nii.gz -> train_1_a_1
        - train_1_a_1_report.json -> train_1_a_1

        Args:
            filename: Filename to parse

        Returns:
            Patient ID string
        """
        # Remove extension(s)
        patient_id = filename
        for ext in [".nii.gz", ".nii", ".json", "_report"]:
            patient_id = patient_id.replace(ext, "")

        return patient_id

    def scan_inbox(self) -> List[PatientData]:
        """Scan inbox for complete patient data pairs.

        Returns:
            List of PatientData for patients with both volume and report
        """
        complete_pairs = []

        # Ensure directories exist
        if not self.volumes_dir.exists():
            logger.warning(f"Volumes directory does not exist: {self.volumes_dir}")
            return complete_pairs
        if not self.reports_dir.exists():
            logger.warning(f"Reports directory does not exist: {self.reports_dir}")
            return complete_pairs

        # Find all volumes
        volumes: Dict[str, Path] = {}
        for vol_path in self.volumes_dir.glob("*.nii.gz"):
            patient_id = self._extract_patient_id(vol_path.name)
            volumes[patient_id] = vol_path

        # Find all reports
        reports: Dict[str, Path] = {}
        for report_path in self.reports_dir.glob("*.json"):
            patient_id = self._extract_patient_id(report_path.name)
            reports[patient_id] = report_path

        # Find complete pairs not yet processed
        for patient_id in volumes:
            if patient_id in self._processed:
                continue

            if patient_id in reports:
                complete_pairs.append(PatientData(
                    patient_id=patient_id,
                    volume_path=volumes[patient_id],
                    report_path=reports[patient_id],
                ))

        logger.debug(f"Found {len(complete_pairs)} new complete pairs")
        return complete_pairs

    def mark_processed(self, patient_id: str) -> None:
        """Mark a patient as processed.

        Args:
            patient_id: Patient ID to mark
        """
        self._processed.add(patient_id)
        logger.debug(f"Marked patient as processed: {patient_id}")

    def get_processed_count(self) -> int:
        """Get number of processed patients."""
        return len(self._processed)

    def watch(
        self,
        callback: Callable[[PatientData], None],
        max_iterations: Optional[int] = None,
    ) -> None:
        """Start watching inbox and process new patients.

        Args:
            callback: Function to call for each new patient
            max_iterations: Maximum number of poll iterations (None = infinite)
        """
        logger.info(f"Starting inbox watch (poll interval: {self.poll_interval}s)")
        self._running = True
        iterations = 0

        while self._running:
            # Check for new patients
            new_patients = self.scan_inbox()

            for patient_data in new_patients:
                try:
                    logger.info(f"Processing patient: {patient_data.patient_id}")
                    callback(patient_data)
                    self.mark_processed(patient_data.patient_id)
                except Exception as e:
                    logger.error(f"Error processing {patient_data.patient_id}: {e}")

            iterations += 1
            if max_iterations and iterations >= max_iterations:
                logger.info("Reached maximum iterations, stopping")
                break

            # Wait before next poll
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        """Stop the inbox watcher."""
        logger.info("Stopping inbox watcher")
        self._running = False

    def process_single(self, patient_id: str) -> Optional[PatientData]:
        """Get data for a single patient by ID.

        Args:
            patient_id: Patient ID to look up

        Returns:
            PatientData if found, None otherwise
        """
        # Look for volume
        volume_path = None
        for pattern in [f"{patient_id}.nii.gz", f"{patient_id}.nii"]:
            path = self.volumes_dir / pattern
            if path.exists():
                volume_path = path
                break

        if not volume_path:
            logger.error(f"Volume not found for patient: {patient_id}")
            return None

        # Look for report
        report_path = None
        for pattern in [f"{patient_id}.json", f"{patient_id}_report.json"]:
            path = self.reports_dir / pattern
            if path.exists():
                report_path = path
                break

        if not report_path:
            logger.error(f"Report not found for patient: {patient_id}")
            return None

        return PatientData(
            patient_id=patient_id,
            volume_path=volume_path,
            report_path=report_path,
        )
