"""Priority-sorted worklist management for triage results."""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import OUTPUT_DIR, PRIORITY_NAMES

logger = logging.getLogger(__name__)


@dataclass
class WorklistEntry:
    """Single entry in the triage worklist."""
    patient_id: str
    priority_level: int
    findings_summary: str
    processed_at: str
    result_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "priority_level": self.priority_level,
            "priority_name": PRIORITY_NAMES.get(self.priority_level, "UNKNOWN"),
            "findings_summary": self.findings_summary,
            "processed_at": self.processed_at,
            "result_path": self.result_path,
        }


class Worklist:
    """Priority-sorted worklist of triage results."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        """Initialize the worklist.

        Args:
            output_dir: Directory for worklist file
        """
        self.output_dir = output_dir
        self.worklist_path = output_dir / "worklist.json"
        self._entries: List[WorklistEntry] = []
        self._lock = threading.Lock()

        # Load existing worklist if present
        self._load()

    def _load(self) -> None:
        """Load existing worklist from disk."""
        if not self.worklist_path.exists():
            return

        try:
            with open(self.worklist_path, "r") as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                entry = WorklistEntry(
                    patient_id=entry_data["patient_id"],
                    priority_level=entry_data["priority_level"],
                    findings_summary=entry_data["findings_summary"],
                    processed_at=entry_data["processed_at"],
                    result_path=entry_data["result_path"],
                )
                self._entries.append(entry)

            logger.info(f"Loaded {len(self._entries)} entries from worklist")
        except Exception as e:
            logger.error(f"Failed to load worklist: {e}")

    def _save(self) -> None:
        """Save worklist to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_entries": len(self._entries),
            "entries": [entry.to_dict() for entry in self._entries],
        }

        with open(self.worklist_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved worklist with {len(self._entries)} entries")

    def _sort(self) -> None:
        """Sort entries by priority (1=CRITICAL first), then by time."""
        self._entries.sort(key=lambda e: (e.priority_level, e.processed_at))

    def add_entry(
        self,
        patient_id: str,
        priority_level: int,
        findings_summary: str,
        result_path: str,
    ) -> None:
        """Add a new entry to the worklist.

        Args:
            patient_id: Patient identifier
            priority_level: Priority level (1-3)
            findings_summary: Brief summary of findings
            result_path: Path to full result JSON
        """
        with self._lock:
            # Remove existing entry for same patient if present
            self._entries = [e for e in self._entries if e.patient_id != patient_id]

            entry = WorklistEntry(
                patient_id=patient_id,
                priority_level=priority_level,
                findings_summary=findings_summary,
                processed_at=datetime.utcnow().isoformat() + "Z",
                result_path=result_path,
            )

            self._entries.append(entry)
            self._sort()
            self._save()

            logger.info(f"Added {patient_id} to worklist (priority {priority_level})")

    def get_entries(
        self,
        priority_filter: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[WorklistEntry]:
        """Get worklist entries.

        Args:
            priority_filter: Filter by priority level (None = all)
            limit: Maximum entries to return (None = all)

        Returns:
            List of WorklistEntry objects
        """
        with self._lock:
            entries = self._entries.copy()

        if priority_filter is not None:
            entries = [e for e in entries if e.priority_level == priority_filter]

        if limit is not None:
            entries = entries[:limit]

        return entries

    def get_patient_entry(self, patient_id: str) -> Optional[WorklistEntry]:
        """Get entry for a specific patient.

        Args:
            patient_id: Patient ID to look up

        Returns:
            WorklistEntry if found, None otherwise
        """
        with self._lock:
            for entry in self._entries:
                if entry.patient_id == patient_id:
                    return entry
        return None

    def remove_entry(self, patient_id: str) -> bool:
        """Remove an entry from the worklist.

        Args:
            patient_id: Patient ID to remove

        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            original_len = len(self._entries)
            self._entries = [e for e in self._entries if e.patient_id != patient_id]

            if len(self._entries) < original_len:
                self._save()
                logger.info(f"Removed {patient_id} from worklist")
                return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get worklist statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total = len(self._entries)
            by_priority = {}

            for entry in self._entries:
                level = entry.priority_level
                by_priority[level] = by_priority.get(level, 0) + 1

        return {
            "total": total,
            "by_priority": by_priority,
            "priority_names": PRIORITY_NAMES,
        }

    def clear(self) -> None:
        """Clear all entries from the worklist."""
        with self._lock:
            self._entries.clear()
            self._save()
            logger.info("Cleared worklist")
