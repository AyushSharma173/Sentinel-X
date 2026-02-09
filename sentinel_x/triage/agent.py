"""Main triage agent orchestrating the Serial Late Fusion pipeline.

Per-patient execution:
  Phase 0: FHIR extraction + CT multi-window preprocessing (no GPU)
  Phase 1: Load 4B vision model -> visual detection -> unload (8GB VRAM)
  Phase 2: Load 27B reasoner -> delta analysis -> unload (13-14GB VRAM)
  Phase 3: Merge outputs -> save results -> update worklist (no GPU)
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Reduce CUDA memory fragmentation (must be set before any torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Skip HuggingFace Hub HTTP checks when weights are already cached locally.
# Saves ~5s per model load (10-20 HEAD requests avoided).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .config import (
    INBOX_POLL_INTERVAL,
    INBOX_REPORTS_DIR,
    INBOX_VOLUMES_DIR,
    LOG_DIR,
    LOG_FILE,
    LOG_FORMAT,
    OUTPUT_DIR,
    VISION_MODEL_ID,
    REASONER_MODEL_ID,
)
from .ct_processor import process_ct_volume
from .fhir_janitor import FHIRJanitor
from .inbox_watcher import InboxWatcher, PatientData
from .medgemma_analyzer import VisionAnalyzer
from .medgemma_reasoner import ClinicalReasoner
from .output_generator import generate_triage_result, save_triage_result
from .prompts import (
    PHASE1_SYSTEM_PROMPT,
    PHASE2_SYSTEM_PROMPT,
    build_phase1_user_prompt,
    build_phase2_user_prompt,
)
from .session_logger import SessionLogger
from .vram_manager import get_vram_free_mb, get_vram_total_mb, log_vram_status, verify_clean_state
from .worklist import Worklist

# Minimum total GPU memory (MB) to run the full pipeline.
# Phase 2 (27B NF4) peaks at ~17GB — need at least ~22GB total for safety.
MIN_GPU_TOTAL_MB = 22_000


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)


class TriageAgent:
    """Main triage agent — Serial Late Fusion architecture.

    Models are loaded on-demand per-phase and unloaded immediately after,
    so they never coexist in VRAM (24GB budget).
    """

    def __init__(
        self,
        volumes_dir=INBOX_VOLUMES_DIR,
        reports_dir=INBOX_REPORTS_DIR,
        output_dir=OUTPUT_DIR,
        poll_interval=INBOX_POLL_INTERVAL,
    ):
        self.logger = logging.getLogger(__name__)
        self.inbox_watcher = InboxWatcher(
            volumes_dir=volumes_dir,
            reports_dir=reports_dir,
            poll_interval=poll_interval,
        )
        # Phase 1 and Phase 2 components (loaded on-demand, not at init)
        self.vision_analyzer = VisionAnalyzer()
        self.reasoner = ClinicalReasoner()
        # Legacy alias so existing references (e.g. demo_service) still work
        self.analyzer = self.vision_analyzer

        self.worklist = Worklist(output_dir=output_dir)
        self.output_dir = output_dir
        self._running = False
        self._patient_count = 0
        self.session_logger = SessionLogger(LOG_DIR)

    def _load_fhir_bundle(self, report_path: Path) -> Dict[str, Any]:
        try:
            with open(report_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load FHIR bundle: {e}")
            return {}

    def process_patient(self, patient_data: PatientData) -> None:
        self._process_patient_internal(patient_data)

    def _preflight_gpu_check(self) -> None:
        """Verify GPU has enough total memory and nothing else is using it."""
        import torch
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available — pipeline will fail at model load")
            return

        total = get_vram_total_mb()
        free = get_vram_free_mb()

        if total < MIN_GPU_TOTAL_MB:
            raise RuntimeError(
                f"GPU has {total:.0f}MB total VRAM — need at least {MIN_GPU_TOTAL_MB}MB "
                f"for the Serial Late Fusion pipeline (Phase 2 peaks at ~17GB). "
                f"An RTX 4090 (24GB) or better is required."
            )

        # Warn if something else is using significant VRAM
        used = total - free
        if used > 1000:
            self.logger.warning(
                f"GPU has {used:.0f}MB already in use before pipeline start. "
                f"Free: {free:.0f}MB / {total:.0f}MB. "
                f"Other GPU processes may cause OOM — consider stopping them."
            )
        else:
            self.logger.info(
                f"GPU pre-flight OK: {free:.0f}MB free / {total:.0f}MB total"
            )

    def _process_patient_internal(self, patient_data: PatientData) -> None:
        patient_id = patient_data.patient_id
        self._patient_count += 1
        self.logger.info(
            f"{'='*60}\n"
            f"  Patient {self._patient_count}: {patient_id}\n"
            f"{'='*60}"
        )
        log_vram_status(f"patient {self._patient_count} start")
        self.session_logger.log_patient_start(self._patient_count, patient_id)

        try:
            # GPU pre-flight check (first patient only, or if problems detected)
            if self._patient_count == 1:
                self._preflight_gpu_check()

            # ==============================================================
            # PHASE 0: Preprocessing (no GPU)
            # ==============================================================

            # Step 0a: Parse FHIR context using FHIRJanitor
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

            # Step 0b: Process CT volume (multi-window RGB)
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

            # ==============================================================
            # PHASE 1: Visual Detection (4B, ~8GB VRAM)
            # ==============================================================

            phase1_user_prompt = build_phase1_user_prompt(len(images))
            self.session_logger.log_phase1_prompt(
                system_prompt=PHASE1_SYSTEM_PROMPT,
                user_prompt=phase1_user_prompt,
                num_images=len(images),
                volume_path=patient_data.volume_path,
            )

            t0 = time.time()
            self.vision_analyzer.load_model()
            visual_fact_sheet = self.vision_analyzer.analyze(images)
            phase1_duration = time.time() - t0

            self.session_logger.log_phase1_response(
                raw_response=visual_fact_sheet.raw_response,
                fact_sheet=visual_fact_sheet,
                duration_secs=phase1_duration,
            )

            # Unload 4B model before loading 27B
            t_swap = time.time()
            self.vision_analyzer.unload()
            verify_clean_state("post-Phase1-unload")

            # ==============================================================
            # PHASE 2: Clinical Reasoning (27B, ~13-14GB VRAM)
            # ==============================================================

            self.reasoner.load_model()
            swap_duration = time.time() - t_swap

            self.session_logger.log_model_swap(
                from_model=VISION_MODEL_ID,
                to_model=REASONER_MODEL_ID,
                duration_secs=swap_duration,
            )

            phase2_user_prompt = build_phase2_user_prompt(
                context_text, visual_fact_sheet.to_narrative()
            )
            self.session_logger.log_phase2_prompt(
                system_prompt=PHASE2_SYSTEM_PROMPT,
                user_prompt=phase2_user_prompt,
            )

            t0 = time.time()
            delta_result = self.reasoner.analyze(
                clinical_narrative=context_text,
                visual_narrative=visual_fact_sheet.to_narrative(),
            )
            phase2_duration = time.time() - t0

            self.session_logger.log_phase2_response(
                raw_response=delta_result.raw_response,
                delta_result=delta_result,
                duration_secs=phase2_duration,
            )

            # Unload 27B model
            self.reasoner.unload()
            verify_clean_state("post-Phase2-unload")

            # ==============================================================
            # PHASE 3: Output Generation (no GPU)
            # ==============================================================

            result = generate_triage_result(
                patient_id=patient_id,
                visual_fact_sheet=visual_fact_sheet,
                delta_result=delta_result,
                images=images,
                slice_indices=slice_indices,
                conditions_from_context=clinical_stream.conditions,
            )
            result_path = save_triage_result(patient_id, result, self.output_dir)
            self.session_logger.log_output_saved(result_path, delta_result.overall_priority)

            # Update worklist
            self.worklist.add_entry(
                patient_id=patient_id,
                priority_level=delta_result.overall_priority,
                findings_summary=delta_result.findings_summary,
                result_path=str(result_path),
            )

            self.logger.info(
                f"[{patient_id}] Triage complete - Priority {delta_result.overall_priority} "
                f"(Phase1: {phase1_duration:.1f}s, Swap: {swap_duration:.1f}s, "
                f"Phase2: {phase2_duration:.1f}s)"
            )
            log_vram_status(f"patient {self._patient_count} end")
            self.session_logger.log_patient_end(self._patient_count, patient_id)

        except Exception as e:
            self.logger.error(f"[{patient_id}] Triage failed: {e}", exc_info=True)
            self.session_logger.log_error(patient_id, str(e))
            self.session_logger.log_patient_end(self._patient_count, patient_id)
            # Ensure models are unloaded on failure
            try:
                self.vision_analyzer.unload()
            except Exception:
                pass
            try:
                self.reasoner.unload()
            except Exception:
                pass
            raise

    def run(self, max_patients=None):
        """Start the triage agent in watch mode.

        Models are loaded/unloaded per-patient inside _process_patient_internal,
        NOT preloaded at startup.
        """
        self.logger.info("Starting Sentinel-X Triage Agent (Serial Late Fusion)")
        self._running = True

        def signal_handler(sig, frame):
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            self.inbox_watcher.watch(
                callback=self.process_patient, max_iterations=max_patients
            )
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self.inbox_watcher.stop()
        # Ensure models are unloaded
        try:
            self.vision_analyzer.unload()
        except Exception:
            pass
        try:
            self.reasoner.unload()
        except Exception:
            pass
        stats = self.worklist.get_statistics()
        self.logger.info(f"Processed {stats['total']} patients")

    def process_single_patient(self, patient_id: str) -> bool:
        """Process a single patient by ID (models loaded on-demand per-phase)."""
        patient_data = self.inbox_watcher.process_single(patient_id)
        if not patient_data:
            return False
        try:
            self.process_patient(patient_data)
            return True
        except Exception:
            return False


def main():
    parser = argparse.ArgumentParser(description="Sentinel-X Triage Agent")
    parser.add_argument("--single", type=str)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=INBOX_POLL_INTERVAL)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    agent = TriageAgent(poll_interval=args.poll_interval)
    if args.single:
        success = agent.process_single_patient(args.single)
        sys.exit(0 if success else 1)
    else:
        agent.run(max_patients=args.max_patients)


if __name__ == "__main__":
    main()
