"""Main triage agent orchestrating the CT analysis pipeline."""

import argparse
import logging
import signal
import sys
from pathlib import Path

from .config import (
    INBOX_POLL_INTERVAL,
    INBOX_REPORTS_DIR,
    INBOX_VOLUMES_DIR,
    LOG_FILE,
    LOG_FORMAT,
    OUTPUT_DIR,
)
from .ct_processor import process_ct_volume
from .fhir_context import format_context_for_prompt, parse_fhir_context
from .inbox_watcher import InboxWatcher, PatientData
from .medgemma_analyzer import MedGemmaAnalyzer
from .output_generator import generate_triage_result, save_triage_result
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

    def process_patient(self, patient_data: PatientData) -> None:
        """Process a single patient through the triage pipeline.

        Args:
            patient_data: Patient volume and report paths
        """
        patient_id = patient_data.patient_id
        self.logger.info(f"Starting triage for patient: {patient_id}")

        try:
            # Step 1: Parse FHIR context
            self.logger.info(f"[{patient_id}] Parsing clinical context")
            context = parse_fhir_context(patient_data.report_path, patient_id)
            context_text = format_context_for_prompt(context)

            # Step 2: Process CT volume
            self.logger.info(f"[{patient_id}] Processing CT volume")
            images, slice_indices, metadata = process_ct_volume(patient_data.volume_path)

            # Step 3: Run MedGemma analysis
            self.logger.info(f"[{patient_id}] Running MedGemma analysis")
            analysis = self.analyzer.analyze(images, context_text)

            # Step 4: Generate output
            self.logger.info(f"[{patient_id}] Generating triage output")
            result = generate_triage_result(
                patient_id=patient_id,
                analysis=analysis,
                images=images,
                slice_indices=slice_indices,
                conditions_from_context=context.conditions,
            )

            # Step 5: Save result
            result_path = save_triage_result(patient_id, result, self.output_dir)

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

        except Exception as e:
            self.logger.error(f"[{patient_id}] Triage failed: {e}", exc_info=True)
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
    agent = TriageAgent(poll_interval=args.poll_interval)

    if args.single:
        # Process single patient
        success = agent.process_single_patient(args.single)
        sys.exit(0 if success else 1)
    else:
        # Run in watch mode
        agent.run(max_patients=args.max_patients)


if __name__ == "__main__":
    main()
