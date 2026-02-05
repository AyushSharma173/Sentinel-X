"""Main triage agent orchestrating the CT analysis pipeline."""

import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .agent_loop import ReActAgentLoop, get_risk_adjustment_value
from .config import (
    AGENT_MAX_ITERATIONS,
    AGENT_MODE_ENABLED,
    INBOX_POLL_INTERVAL,
    INBOX_REPORTS_DIR,
    INBOX_VOLUMES_DIR,
    LOG_FILE,
    LOG_FORMAT,
    LOG_JSON_ENABLED,
    OUTPUT_DIR,
    PRIORITY_CRITICAL,
    PRIORITY_ROUTINE,
)
from .logging import (
    SessionManager,
    get_agent_trace_logger,
    get_fhir_trace_logger,
    patient_trace_context,
)
from .ct_processor import process_ct_volume
from .fhir_janitor import FHIRJanitor
from .inbox_watcher import InboxWatcher, PatientData
from .medgemma_analyzer import MedGemmaAnalyzer
from .output_generator import generate_triage_result, save_triage_result
from .state import AgentState
from .worklist import Worklist


def setup_logging(verbose: bool = False, session_id: str = None) -> str:
    """Configure logging for the agent.

    Args:
        verbose: Enable verbose debug logging
        session_id: Optional specific session ID for trace logs

    Returns:
        The session ID being used for trace logs
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

    # Initialize session-based trace logging
    return initialize_trace_logging(session_id)


def initialize_trace_logging(session_id: str = None) -> str:
    """Initialize session-based trace logging.

    This can be called independently of setup_logging() when trace logging
    needs to be initialized without reconfiguring the root logger (e.g., when
    running through an API server that has its own logging setup).

    Args:
        session_id: Optional specific session ID for trace logs

    Returns:
        The session ID being used for trace logs, or None if disabled
    """
    if not LOG_JSON_ENABLED:
        return None

    session_manager = SessionManager.get_instance()
    session_id = session_manager.initialize(session_id)

    # Initialize specialized trace loggers
    agent_trace = get_agent_trace_logger()
    agent_trace.initialize_session(session_id)

    fhir_trace = get_fhir_trace_logger()
    fhir_trace.initialize_session(session_id)

    logging.getLogger(__name__).info(
        f"Trace logging initialized: session={session_id}"
    )

    return session_id


class TriageAgent:
    """Main triage agent that orchestrates the processing pipeline."""

    def __init__(
        self,
        volumes_dir: Path = INBOX_VOLUMES_DIR,
        reports_dir: Path = INBOX_REPORTS_DIR,
        output_dir: Path = OUTPUT_DIR,
        poll_interval: float = INBOX_POLL_INTERVAL,
        use_agent_mode: bool = AGENT_MODE_ENABLED,
    ):
        """Initialize the triage agent.

        Args:
            volumes_dir: Directory containing CT volumes
            reports_dir: Directory containing reports
            output_dir: Directory for output results
            poll_interval: Seconds between inbox scans
            use_agent_mode: Enable ReAct agent for clinical correlation
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
        self.use_agent_mode = use_agent_mode

        self._running = False

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

    def _run_agent_loop(
        self,
        patient_id: str,
        visual_findings: str,
        fhir_bundle: Dict[str, Any],
    ) -> Optional[AgentState]:
        """Run the ReAct agent loop for clinical correlation.

        Args:
            patient_id: Patient identifier
            visual_findings: Visual findings from MedGemma
            fhir_bundle: FHIR Bundle for tool queries

        Returns:
            Agent state with findings, or None if agent mode disabled/failed
        """
        if not self.use_agent_mode:
            return None

        if not fhir_bundle or fhir_bundle.get("resourceType") != "Bundle":
            self.logger.warning(
                f"[{patient_id}] Skipping agent mode - no valid FHIR Bundle"
            )
            return None

        try:
            self.logger.info(f"[{patient_id}] Running ReAct agent for clinical correlation")

            agent_loop = ReActAgentLoop(
                model=self.analyzer.model,
                processor=self.analyzer.processor,
                fhir_bundle=fhir_bundle,
                patient_id=patient_id,
                max_iterations=AGENT_MAX_ITERATIONS,
            )

            agent_state = agent_loop.run(visual_findings)

            self.logger.info(
                f"[{patient_id}] Agent complete: "
                f"{agent_state['iteration']} iterations, "
                f"risk_adjustment={agent_state.get('risk_adjustment', 'NONE')}"
            )

            return agent_state

        except Exception as e:
            self.logger.error(f"[{patient_id}] Agent loop failed: {e}", exc_info=True)
            return None

    def process_patient(self, patient_data: PatientData) -> None:
        """Process a single patient through the triage pipeline.

        Args:
            patient_data: Patient volume and report paths
        """
        patient_id = patient_data.patient_id

        # Wrap processing in patient trace context for structured logging
        with patient_trace_context(patient_id, self.logger):
            self._process_patient_internal(patient_data)

    def _process_patient_internal(self, patient_data: PatientData) -> None:
        """Internal patient processing implementation.

        Args:
            patient_data: Patient volume and report paths
        """
        patient_id = patient_data.patient_id
        self.logger.info(f"Starting triage for patient: {patient_id}")

        try:
            # Step 1: Parse FHIR context using FHIRJanitor (Dense Clinical Stream)
            self.logger.info(f"[{patient_id}] Parsing clinical context with FHIRJanitor")
            fhir_bundle = self._load_fhir_bundle(patient_data.report_path)
            janitor = FHIRJanitor()
            clinical_stream = janitor.process_bundle(fhir_bundle)
            context_text = clinical_stream.narrative

            # Log any extraction warnings
            for warning in clinical_stream.extraction_warnings:
                self.logger.warning(f"[{patient_id}] FHIR extraction: {warning}")

            # Step 2: Process CT volume
            self.logger.info(f"[{patient_id}] Processing CT volume")
            images, slice_indices, metadata = process_ct_volume(patient_data.volume_path)

            # Step 3: Run MedGemma analysis
            self.logger.info(f"[{patient_id}] Running MedGemma analysis")
            analysis = self.analyzer.analyze(images, context_text)

            # Step 4: Run ReAct agent for clinical correlation
            agent_state = self._run_agent_loop(
                patient_id=patient_id,
                visual_findings=analysis.visual_findings,
                fhir_bundle=fhir_bundle,
            )

            # Step 5: Apply risk adjustment from agent
            final_priority = analysis.priority_level
            if agent_state:
                adjustment = get_risk_adjustment_value(
                    agent_state.get("risk_adjustment")
                )
                if adjustment != 0:
                    original_priority = final_priority
                    final_priority = max(
                        PRIORITY_CRITICAL,
                        min(PRIORITY_ROUTINE, final_priority + adjustment),
                    )
                    self.logger.info(
                        f"[{patient_id}] Priority adjusted: "
                        f"{original_priority} -> {final_priority} "
                        f"(agent: {agent_state.get('risk_adjustment')})"
                    )

            # Step 6: Generate output with agent trace
            self.logger.info(f"[{patient_id}] Generating triage output")
            result = generate_triage_result(
                patient_id=patient_id,
                analysis=analysis,
                images=images,
                slice_indices=slice_indices,
                conditions_from_context=clinical_stream.conditions,
                agent_state=agent_state,
            )

            # Override priority level with adjusted value
            result["priority_level"] = final_priority

            # Step 7: Save result
            result_path = save_triage_result(patient_id, result, self.output_dir)

            # Step 8: Update worklist with adjusted priority
            self.worklist.add_entry(
                patient_id=patient_id,
                priority_level=final_priority,
                findings_summary=analysis.findings_summary,
                result_path=str(result_path),
            )

            self.logger.info(
                f"[{patient_id}] Triage complete - Priority {final_priority}"
                + (f" (agent adjusted)" if agent_state and agent_state.get("risk_adjustment") else "")
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
        "--no-agent",
        action="store_true",
        help="Disable ReAct agent mode for clinical correlation",
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
        use_agent_mode=not args.no_agent,
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
