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

        # Source data directories
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
        import random
        import time

        logger.info("Simulator thread started")

        # Get available scans
        if not self._source_volumes.exists():
            logger.error(f"Source volumes directory not found: {self._source_volumes}")
            self._simulator_running = False
            return

        all_scans = list(self._source_volumes.glob("*.nii.gz"))
        if not all_scans:
            logger.error("No CT scans found in source directory")
            self._simulator_running = False
            return

        remaining_scans = all_scans.copy()
        random.shuffle(remaining_scans)

        while self._simulator_running and remaining_scans:
            scan = remaining_scans.pop()
            base_name = scan.stem.replace(".nii", "")

            # Copy volume
            dest_volume = INBOX_VOLUMES_DIR / scan.name
            shutil.copy2(scan, dest_volume)

            # Copy report
            report_json = self._source_reports / f"{base_name}.json"
            report_txt = self._source_reports / f"{base_name}.txt"

            if report_json.exists():
                shutil.copy2(report_json, INBOX_REPORTS_DIR / report_json.name)
            if report_txt.exists():
                shutil.copy2(report_txt, INBOX_REPORTS_DIR / report_txt.name)

            logger.info(f"Simulator: copied {scan.name}")

            # Broadcast patient arrived event
            self._run_async(ws_manager.send_event(
                WSEventType.PATIENT_ARRIVED,
                {"patient_id": base_name, "remaining": len(remaining_scans)}
            ))

            # Wait before next scan
            if remaining_scans and self._simulator_running:
                time.sleep(10)

        self._simulator_running = False
        logger.info("Simulator thread finished")

    def _run_agent(self) -> None:
        """Run the triage agent in a background thread."""
        import time

        from triage.agent import TriageAgent
        from triage.inbox_watcher import PatientData

        logger.info("Agent thread started")

        # Create agent with callbacks
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

            try:
                # Run actual processing
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

            except Exception as e:
                logger.error(f"Agent processing failed for {patient_id}: {e}")
                self._run_async(ws_manager.send_event(
                    WSEventType.ERROR,
                    {"patient_id": patient_id, "error": str(e)}
                ))

        agent.process_patient = process_with_callbacks

        # Load model
        try:
            logger.info("Loading MedGemma model...")
            agent.analyzer.load_model()
            self._model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._agent_running = False
            return

        # Watch inbox
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
