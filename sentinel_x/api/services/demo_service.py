"""Demo orchestration service for controlling simulator and triage agent."""

import asyncio
import logging
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Ensure the project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from triage.config import (
    BASE_DIR,
    COMBINED_DIR,
    COMBINED_MANIFEST,
    INBOX_DIR,
    INBOX_REPORTS_DIR,
    INBOX_VOLUMES_DIR,
    OUTPUT_DIR,
)
from triage.worklist import Worklist
from api.models import DemoStatus, SystemStatus, WSEventType
from api.services.ws_manager import ws_manager

logger = logging.getLogger(__name__)


class DemoService:
    """Orchestrates the demo by controlling simulator and triage agent."""

    def __init__(self):
        self._status = DemoStatus.STOPPED
        self._simulator_thread: Optional[threading.Thread] = None
        self._agent_thread: Optional[threading.Thread] = None
        self._simulator_running = False
        self._agent_running = False
        self._model_loaded = False
        self._patients_processed = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Source data - using combined folder (unified structure)
        self._source_combined = COMBINED_DIR
        self._source_manifest = COMBINED_MANIFEST

        # Legacy source directories (fallback if combined folder doesn't exist)
        self._source_volumes = BASE_DIR / "data" / "raw_ct_rate" / "volumes"
        self._source_reports = BASE_DIR / "data" / "raw_ct_rate" / "reports"

        # Worklist instance
        self._worklist = Worklist(output_dir=OUTPUT_DIR)

        # Processing callbacks
        self._on_patient_arrived: Optional[Callable] = None
        self._on_processing_started: Optional[Callable] = None
        self._on_processing_progress: Optional[Callable] = None
        self._on_processing_complete: Optional[Callable] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for async operations."""
        self._loop = loop

    def _run_async(self, coro):
        """Run an async coroutine from a sync context."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    def get_status(self) -> SystemStatus:
        """Get current system status."""
        # Count patients in inbox queue
        queue_count = 0
        if INBOX_VOLUMES_DIR.exists():
            queue_count = len(list(INBOX_VOLUMES_DIR.glob("*.nii.gz")))

        return SystemStatus(
            demo_status=self._status,
            simulator_running=self._simulator_running,
            agent_running=self._agent_running,
            model_loaded=self._model_loaded,
            patients_in_queue=queue_count,
            patients_processed=self._patients_processed,
        )

    def get_worklist(self) -> Worklist:
        """Get the worklist instance."""
        return self._worklist

    async def start_demo(self) -> bool:
        """Start the demo (simulator + agent)."""
        if self._status != DemoStatus.STOPPED:
            logger.warning("Demo already running or starting")
            return False

        self._status = DemoStatus.STARTING
        logger.info("Starting demo...")

        try:
            # Ensure inbox directories exist
            INBOX_VOLUMES_DIR.mkdir(parents=True, exist_ok=True)
            INBOX_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Start simulator in background thread
            self._simulator_running = True
            self._simulator_thread = threading.Thread(
                target=self._run_simulator,
                daemon=True
            )
            self._simulator_thread.start()

            # Start triage agent in background thread
            self._agent_running = True
            self._agent_thread = threading.Thread(
                target=self._run_agent,
                daemon=True
            )
            self._agent_thread.start()

            self._status = DemoStatus.RUNNING
            logger.info("Demo started successfully")

            await ws_manager.send_event(WSEventType.DEMO_STARTED, {
                "message": "Demo started",
                "status": self.get_status().model_dump()
            })

            return True

        except Exception as e:
            logger.error(f"Failed to start demo: {e}")
            self._status = DemoStatus.STOPPED
            return False

    async def stop_demo(self) -> bool:
        """Stop the demo."""
        if self._status != DemoStatus.RUNNING:
            logger.warning("Demo not running")
            return False

        self._status = DemoStatus.STOPPING
        logger.info("Stopping demo...")

        # Signal threads to stop
        self._simulator_running = False
        self._agent_running = False

        # Wait for threads to finish (with timeout)
        if self._simulator_thread and self._simulator_thread.is_alive():
            self._simulator_thread.join(timeout=5)

        if self._agent_thread and self._agent_thread.is_alive():
            self._agent_thread.join(timeout=5)

        self._status = DemoStatus.STOPPED
        logger.info("Demo stopped")

        await ws_manager.send_event(WSEventType.DEMO_STOPPED, {
            "message": "Demo stopped",
            "status": self.get_status().model_dump()
        })

        return True

    async def reset_demo(self) -> bool:
        """Reset the demo (clear inbox and worklist)."""
        # Stop if running
        if self._status == DemoStatus.RUNNING:
            await self.stop_demo()

        logger.info("Resetting demo...")

        # Clear inbox directories
        if INBOX_VOLUMES_DIR.exists():
            for f in INBOX_VOLUMES_DIR.glob("*"):
                f.unlink()

        if INBOX_REPORTS_DIR.exists():
            for f in INBOX_REPORTS_DIR.glob("*"):
                f.unlink()

        # Clear worklist
        self._worklist.clear()

        # Clear output directory (keep structure)
        if OUTPUT_DIR.exists():
            for patient_dir in OUTPUT_DIR.iterdir():
                if patient_dir.is_dir():
                    shutil.rmtree(patient_dir)

        # Reset counters
        self._patients_processed = 0

        logger.info("Demo reset complete")
        return True

    def _run_simulator(self) -> None:
        """Run the simulator in a background thread."""
        import json
        import random
        import time

        logger.info("Simulator thread started")

        # Discover patients from combined folder
        patient_folders = self._discover_patients()

        if not patient_folders:
            logger.error("No patient data found in combined folder")
            self._simulator_running = False
            return

        remaining = patient_folders.copy()
        random.shuffle(remaining)

        while self._simulator_running and remaining:
            patient_folder = remaining.pop()
            patient_id = patient_folder.name

            # Copy all patient files to inbox
            self._copy_patient_to_inbox(patient_folder, patient_id)

            logger.info(f"Simulator: queued {patient_id}")

            self._run_async(ws_manager.send_event(
                WSEventType.PATIENT_ARRIVED,
                {"patient_id": patient_id, "remaining": len(remaining)}
            ))

            if remaining and self._simulator_running:
                time.sleep(10)

        self._simulator_running = False
        logger.info("Simulator thread finished")

    def _discover_patients(self) -> list:
        """Discover patient folders from combined directory."""
        import json

        patient_folders = []

        # Try manifest first (more reliable)
        if self._source_manifest.exists():
            try:
                with open(self._source_manifest) as f:
                    manifest = json.load(f)
                for patient in manifest.get("patients", []):
                    folder = self._source_combined / patient["folder"]
                    if folder.exists() and (folder / "fhir.json").exists():
                        patient_folders.append(folder)
                logger.info(f"Loaded {len(patient_folders)} patients from manifest")
                return patient_folders
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")

        # Fallback: scan combined directory
        if self._source_combined.exists():
            for folder in self._source_combined.iterdir():
                if folder.is_dir() and (folder / "fhir.json").exists():
                    patient_folders.append(folder)
            logger.info(f"Discovered {len(patient_folders)} patients by scanning")
            return patient_folders

        # Last resort: use legacy directories
        logger.warning("Combined folder not found, falling back to legacy directories")
        if self._source_volumes.exists():
            for volume in self._source_volumes.glob("*.nii.gz"):
                # Create a pseudo-folder path for legacy handling
                patient_folders.append(volume)
            logger.info(f"Found {len(patient_folders)} volumes in legacy directory")

        return patient_folders

    def _copy_patient_to_inbox(self, patient_folder: Path, patient_id: str) -> None:
        """Copy patient files from combined folder to inbox."""

        # Handle legacy mode (direct volume file instead of folder)
        if patient_folder.is_file() and patient_folder.suffix == ".gz":
            # Legacy mode: copying from volumes directory
            shutil.copy2(patient_folder, INBOX_VOLUMES_DIR / f"{patient_id}.nii.gz")
            # Try to find matching report
            base_name = patient_folder.stem.replace(".nii", "")
            report_json = self._source_reports / f"{base_name}.json"
            report_txt = self._source_reports / f"{base_name}.txt"
            if report_json.exists():
                shutil.copy2(report_json, INBOX_REPORTS_DIR / f"{base_name}.json")
            if report_txt.exists():
                shutil.copy2(report_txt, INBOX_REPORTS_DIR / f"{base_name}.txt")
            return

        # Combined folder mode
        # Copy volume (resolve symlink if needed)
        volume_src = patient_folder / "volume.nii.gz"
        if volume_src.exists() or volume_src.is_symlink():
            actual = volume_src.resolve() if volume_src.is_symlink() else volume_src
            shutil.copy2(actual, INBOX_VOLUMES_DIR / f"{patient_id}.nii.gz")

        # Copy FHIR bundle as main report file
        fhir_src = patient_folder / "fhir.json"
        if fhir_src.exists():
            shutil.copy2(fhir_src, INBOX_REPORTS_DIR / f"{patient_id}.json")

        # Copy report.txt for debugging/display (optional)
        report_txt = patient_folder / "report.txt"
        if report_txt.exists():
            shutil.copy2(report_txt, INBOX_REPORTS_DIR / f"{patient_id}.txt")

    def _run_agent(self) -> None:
        """Run the triage agent in a background thread.

        Serial Late Fusion: models are loaded/unloaded per-patient inside
        the agent pipeline — no upfront model preload needed.
        """
        import time

        from triage.agent import TriageAgent
        from triage.inbox_watcher import PatientData

        logger.info("Agent thread started (Serial Late Fusion — no model preload)")

        # Create agent (models loaded on-demand per-phase)
        agent = TriageAgent()

        # Set up callback wrappers
        original_process = agent.process_patient

        def process_with_callbacks(patient_data: PatientData) -> None:
            patient_id = patient_data.patient_id

            # Notify processing started
            self._run_async(ws_manager.send_event(
                WSEventType.PROCESSING_STARTED,
                {"patient_id": patient_id}
            ))

            # Notify Phase 1 starting
            self._run_async(ws_manager.send_event(
                WSEventType.PHASE1_STARTED,
                {"patient_id": patient_id, "model": "google/medgemma-1.5-4b-it"}
            ))

            try:
                # Run actual processing (both phases happen inside)
                original_process(patient_data)

                # Reload worklist from disk to get the entry added by agent
                self._worklist.reload()

                # Get result
                entry = self._worklist.get_patient_entry(patient_id)
                if entry:
                    # Notify processing complete
                    self._run_async(ws_manager.send_event(
                        WSEventType.PROCESSING_COMPLETE,
                        {
                            "patient_id": patient_id,
                            "priority_level": entry.priority_level,
                            "findings_summary": entry.findings_summary,
                        }
                    ))

                    # Notify worklist updated
                    self._run_async(ws_manager.send_event(
                        WSEventType.WORKLIST_UPDATED,
                        {"total": len(self._worklist.get_entries())}
                    ))

                self._patients_processed += 1
                # Models are transient — loaded/unloaded per patient
                self._model_loaded = True

            except Exception as e:
                logger.error(f"Agent processing failed for {patient_id}: {e}")
                self._run_async(ws_manager.send_event(
                    WSEventType.ERROR,
                    {"patient_id": patient_id, "error": str(e)}
                ))

        agent.process_patient = process_with_callbacks

        # Watch inbox (no model preload — models load per-phase inside pipeline)
        while self._agent_running:
            new_patients = agent.inbox_watcher.scan_inbox()

            for patient_data in new_patients:
                if not self._agent_running:
                    break

                try:
                    process_with_callbacks(patient_data)
                    agent.inbox_watcher.mark_processed(patient_data.patient_id)
                except Exception as e:
                    logger.error(f"Error processing {patient_data.patient_id}: {e}")

            time.sleep(agent.inbox_watcher.poll_interval)

        self._agent_running = False
        logger.info("Agent thread finished")


# Global demo service instance
demo_service = DemoService()
