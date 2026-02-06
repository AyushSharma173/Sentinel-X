"""Main triage agent orchestrating the CT analysis pipeline."""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import (
    INBOX_POLL_INTERVAL,
    INBOX_REPORTS_DIR,
    INBOX_VOLUMES_DIR,
    LOG_DIR,
    LOG_FILE,
    LOG_FORMAT,
    OUTPUT_DIR,
)
from .ct_processor import process_ct_volume
from .fhir_janitor import FHIRJanitor
from .inbox_watcher import InboxWatcher, PatientData
from .medgemma_analyzer import MedGemmaAnalyzer
from .output_generator import generate_triage_result, save_triage_result
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .session_logger import SessionLogger
from .worklist import Worklist


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the agent.

    Args:
        verbose: Enable verbose debug logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Reduce noise from transformers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)


class TriageAgent:
    """Main triage agent that orchestrates the processing pipeline."""

    def __init__(
        self,
        volumes_dir: Path = INBOX_VOLUMES_DIR,
        reports_dir: Path = INBOX_REPORTS_DIR,
        output_dir: Path = OUTPUT_DIR,
        poll_interval: float = INBOX_POLL_INTERVAL,
    ):
        """Initialize the triage agent.

        Args:
            volumes_dir: Directory containing CT volumes
            reports_dir: Directory containing reports
            output_dir: Directory for output results
            poll_interval: Seconds between inbox scans
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.inbox_watcher = InboxWatcher(
            volumes_dir=volumes_dir,
            reports_dir=reports_dir,
            poll_interval=poll_interval,
        )
        self.analyzer = MedGemmaAnalyzer()
        self.worklist = Worklist(output_dir=output_dir)
        self.output_dir = output_dir

        self._running = False
        self._patient_count = 0

        # Session trace logger
        self.session_logger = SessionLogger(LOG_DIR)

    def _load_fhir_bundle(self, report_path: Path) -> Dict[str, Any]:
        """Load the raw FHIR bundle from report path.

        Args:
            report_path: Path to the report JSON file

        Returns:
            FHIR Bundle dictionary
        """
        try:
            with open(report_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load FHIR bundle: {e}")
            return {}

    def process_patient(self, patient_data: PatientData) -> None:
        """Process a single patient through the triage pipeline.

        Args:
            patient_data: Patient volume and report paths
        """
        self._process_patient_internal(patient_data)

    def _process_patient_internal(self, patient_data: PatientData) -> None:
        """Internal patient processing implementation.

        Args:
            patient_data: Patient volume and report paths
        """
        patient_id = patient_data.patient_id
        self.logger.info(f"Starting triage for patient: {patient_id}")

        self._patient_count += 1
        self.session_logger.log_patient_start(self._patient_count, patient_id)

        try:
            # Step 1: Parse FHIR context using FHIRJanitor (Dense Clinical Stream)
            self.logger.info(f"[{patient_id}] Parsing clinical context with FHIRJanitor")
            t0 = time.time()
            fhir_bundle = self._load_fhir_bundle(patient_data.report_path)
            janitor = FHIRJanitor()
            clinical_stream = janitor.process_bundle(fhir_bundle)
            context_text = clinical_stream.narrative
            fhir_duration = time.time() - t0

            for warning in clinical_stream.extraction_warnings:
                self.logger.warning(f"[{patient_id}] FHIR extraction: {warning}")

            self.session_logger.log_fhir_extraction(
                report_path=patient_data.report_path,
                fhir_bundle=fhir_bundle,
                clinical_stream=clinical_stream,
                duration_secs=fhir_duration,
            )

            # Step 2: Process CT volume
            self.logger.info(f"[{patient_id}] Processing CT volume")
            t0 = time.time()
            images, slice_indices, metadata = process_ct_volume(patient_data.volume_path)
            ct_duration = time.time() - t0

            self.session_logger.log_ct_processing(
                volume_path=patient_data.volume_path,
                num_slices=len(images),
                slice_indices=slice_indices,
                image_size=images[0].size if images else (0, 0),
                duration_secs=ct_duration,
            )

            # Step 3: Run MedGemma analysis
            self.logger.info(f"[{patient_id}] Running MedGemma analysis")

            # Log the prompt before calling the model
            user_prompt = build_user_prompt(context_text, len(images))
            self.session_logger.log_medgemma_prompt(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                num_images=len(images),
                volume_path=patient_data.volume_path,
            )

            t0 = time.time()
            analysis = self.analyzer.analyze(images, context_text)
            analysis_duration = time.time() - t0

            self.session_logger.log_medgemma_response(
                raw_response=analysis.raw_response,
                duration_secs=analysis_duration,
            )

            # Step 4: Log parsed results
            self.session_logger.log_parsed_results(analysis, slice_indices)

            # Step 5: Generate and save output
            self.logger.info(f"[{patient_id}] Generating triage output")
            result = generate_triage_result(
                patient_id=patient_id,
                analysis=analysis,
                images=images,
                slice_indices=slice_indices,
                conditions_from_context=clinical_stream.conditions,
            )

            result_path = save_triage_result(patient_id, result, self.output_dir)

            self.session_logger.log_output_saved(result_path, analysis.priority_level)

            # Step 6: Update worklist
            self.worklist.add_entry(
                patient_id=patient_id,
                priority_level=analysis.priority_level,
                findings_summary=analysis.findings_summary,
                result_path=str(result_path),
            )

            self.logger.info(
                f"[{patient_id}] Triage complete - Priority {analysis.priority_level}"
            )

            self.session_logger.log_patient_end(self._patient_count, patient_id)

        except Exception as e:
            self.logger.error(f"[{patient_id}] Triage failed: {e}", exc_info=True)
            self.session_logger.log_error(patient_id, str(e))
            self.session_logger.log_patient_end(self._patient_count, patient_id)
            raise

    def run(self, max_patients: int = None) -> None:
        """Start the triage agent in watch mode.

        Args:
            max_patients: Maximum number of patients to process (None = infinite)
        """
        self.logger.info("Starting Sentinel-X Triage Agent")
        self.logger.info(f"Watching: {self.inbox_watcher.volumes_dir}")
        self.logger.info(f"Output: {self.output_dir}")

        # Load model upfront
        self.logger.info("Loading MedGemma model...")
        self.analyzer.load_model()
        self.logger.info("Model loaded, starting inbox watch")

        self._running = True

        # Set up signal handlers
        def signal_handler(sig, frame):
            self.logger.info("Received shutdown signal")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start watching inbox
        try:
            self.inbox_watcher.watch(
                callback=self.process_patient,
                max_iterations=max_patients,
            )
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the triage agent."""
        self.logger.info("Stopping triage agent")
        self._running = False
        self.inbox_watcher.stop()

        # Print statistics
        stats = self.worklist.get_statistics()
        self.logger.info(f"Processed {stats['total']} patients")
        for level, count in stats.get("by_priority", {}).items():
            name = stats["priority_names"].get(level, "UNKNOWN")
            self.logger.info(f"  Priority {level} ({name}): {count}")

    def process_single_patient(self, patient_id: str) -> bool:
        """Process a single patient by ID.

        Args:
            patient_id: Patient ID to process

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Processing single patient: {patient_id}")

        # Load model
        self.analyzer.load_model()

        # Find patient data
        patient_data = self.inbox_watcher.process_single(patient_id)
        if not patient_data:
            self.logger.error(f"Patient not found: {patient_id}")
            return False

        try:
            self.process_patient(patient_data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to process patient: {e}")
            return False


def main():
    """Main entry point for the triage agent."""
    parser = argparse.ArgumentParser(
        description="Sentinel-X Triage Agent - CT scan prioritization using MedGemma"
    )
    parser.add_argument(
        "--single",
        type=str,
        help="Process a single patient by ID instead of watching inbox",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Maximum number of patients to process before stopping",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=INBOX_POLL_INTERVAL,
        help=f"Seconds between inbox scans (default: {INBOX_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(verbose=args.verbose)

    # Create agent
    agent = TriageAgent(
        poll_interval=args.poll_interval,
    )

    if args.single:
        # Process single patient
        success = agent.process_single_patient(args.single)
        sys.exit(0 if success else 1)
    else:
        # Run in watch mode
        agent.run(max_patients=args.max_patients)


if __name__ == "__main__":
    main()
