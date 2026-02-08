"""Human-readable session trace logger for Sentinel-X pipeline.

Writes a single .txt log file per session that traces the full pipeline
for every patient across both phases of the Serial Late Fusion architecture:
  Phase 1 (Vision): 4B model visual detection
  Phase 2 (Reasoning): 27B model delta analysis
"""

import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .config import PRIORITY_NAMES


class SessionLogger:
    """Human-readable session trace logger for Sentinel-X pipeline.

    Creates a single .txt file per session under log_dir/sessions/{session_id}/
    with heavy visual demarcation between patients and pipeline steps.
    """

    def __init__(self, log_dir: Path):
        now = datetime.now()
        self._session_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        self._session_dir = log_dir / "sessions" / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._session_dir / "session.txt"
        self._patient_start_time: Optional[float] = None

        # Write session header
        self._write(
            "=" * 80 + "\n"
            + " " * 15 + "SENTINEL-X TRIAGE SESSION LOG (Late Fusion)\n"
            + "=" * 80 + "\n"
            + f"Session:  {self._session_id}\n"
            + f"Started:  {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            + "=" * 80 + "\n"
            + "\n\n"
        )

        # Print path to console
        print(f"Session log: {self._log_path}")

    def _write(self, text: str) -> None:
        """Append text to the session log file."""
        with open(self._log_path, "a") as f:
            f.write(text)

    @staticmethod
    def _ts() -> str:
        """Current timestamp as HH:MM:SS."""
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _priority_name(level: int) -> str:
        return PRIORITY_NAMES.get(level, "UNKNOWN")

    # ------------------------------------------------------------------
    # Patient lifecycle
    # ------------------------------------------------------------------

    def log_patient_start(self, patient_number: int, patient_id: str) -> None:
        ts = self._ts()
        self._patient_start_time = time.time()
        pad_id = f"PATIENT {patient_number}: {patient_id}"
        inner_width = 76
        lines = [
            "#" * 80,
            "##" + " " * inner_width + "##",
            "##    " + pad_id.ljust(inner_width - 4) + "##",
            "##    " + f"Started: {ts}".ljust(inner_width - 4) + "##",
            "##" + " " * inner_width + "##",
            "#" * 80,
        ]
        self._write("\n".join(lines) + "\n\n")

    def log_patient_end(self, patient_number: int, patient_id: str) -> None:
        ts = self._ts()
        elapsed = ""
        if self._patient_start_time is not None:
            elapsed = f" | Total: {time.time() - self._patient_start_time:.2f}s"
        pad_id = f"END PATIENT {patient_number}: {patient_id}"
        inner_width = 76
        lines = [
            "#" * 80,
            "##" + " " * inner_width + "##",
            "##    " + pad_id.ljust(inner_width - 4) + "##",
            "##    " + f"Finished: {ts}{elapsed}".ljust(inner_width - 4) + "##",
            "##" + " " * inner_width + "##",
            "#" * 80,
        ]
        self._write("\n" + "\n".join(lines) + "\n\n\n")
        self._patient_start_time = None

    # ------------------------------------------------------------------
    # Step 1: FHIR extraction
    # ------------------------------------------------------------------

    def log_fhir_extraction(
        self,
        report_path: Path,
        fhir_bundle: dict,
        clinical_stream,  # ClinicalStream
        duration_secs: float,
    ) -> None:
        ts = self._ts()
        header = f" STEP 1: FHIR CLINICAL CONTEXT EXTRACTION                [{ts} | {duration_secs:.2f}s]"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        self._write(f"FHIR Bundle: {report_path}\n\n")

        # Bundle structure summary
        entries = fhir_bundle.get("entry", [])
        resource_counts: Counter = Counter()
        for entry in entries:
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType", "Unknown")
            resource_counts[rtype] += 1

        self._write("--- BUNDLE STRUCTURE ---\n")
        self._write(f"Total Entries: {len(entries)}\n")
        for rtype, count in sorted(resource_counts.items(), key=lambda x: -x[1]):
            self._write(f"  {rtype}: {count}\n")
        self._write("--- END BUNDLE STRUCTURE ---\n\n")

        # Clinical stream details
        self._write(f"Patient Summary: {clinical_stream.patient_summary}\n")
        self._write(f"Conditions: {', '.join(clinical_stream.conditions) if clinical_stream.conditions else '[none]'}\n")
        self._write(f"Medications: {', '.join(clinical_stream.medications) if clinical_stream.medications else '[none]'}\n")
        self._write(f"Active Medications: {', '.join(clinical_stream.active_medications) if clinical_stream.active_medications else '[none]'}\n")
        self._write(f"Risk Factors: {', '.join(clinical_stream.risk_factors) if clinical_stream.risk_factors else '[none]'}\n")
        warnings_text = ", ".join(clinical_stream.extraction_warnings) if clinical_stream.extraction_warnings else "[none]"
        self._write(f"Extraction Warnings: {warnings_text}\n\n")

        # Full narrative
        self._write("--- CLINICAL NARRATIVE (sent to Phase 2 reasoner) ---\n\n")
        self._write(clinical_stream.narrative + "\n\n")
        self._write("--- END CLINICAL NARRATIVE ---\n\n")

    # ------------------------------------------------------------------
    # Step 2: CT processing
    # ------------------------------------------------------------------

    def log_ct_processing(
        self,
        volume_path: Path,
        num_slices: int,
        slice_indices: list,
        image_size: Tuple[int, int],
        duration_secs: float,
    ) -> None:
        ts = self._ts()
        header = f" STEP 2: CT VOLUME PROCESSING (Multi-Window)              [{ts} | {duration_secs:.2f}s]"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        self._write(f"Volume File: {volume_path}\n")
        self._write(f"Total Slices Sampled: {num_slices}\n")
        indices_str = str(slice_indices[:15])
        if len(slice_indices) > 15:
            indices_str = indices_str[:-1] + ", ...]"
        self._write(f"Slice Indices: {indices_str}\n")
        self._write(f"Image Dimensions: {image_size[0]} x {image_size[1]} (RGB Multi-Window)\n")
        self._write(f"Channels: R=Wide(-1024,1024) G=Soft(-135,215) B=Brain(0,80)\n\n")

    # ------------------------------------------------------------------
    # Step 3: Phase 1 — Vision Analysis (4B)
    # ------------------------------------------------------------------

    def log_phase1_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        num_images: int,
        volume_path: Path,
    ) -> None:
        ts = self._ts()
        header = f" STEP 3: PHASE 1 — VISUAL DETECTION (4B)                  [{ts}]"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        self._write(f"Model: google/medgemma-1.5-4b-it (BF16, no quant)\n")
        self._write(f"Mode: Vision-only (NO clinical context)\n\n")

        self._write("--- PHASE 1 SYSTEM PROMPT ---\n\n")
        self._write(system_prompt + "\n\n")
        self._write("--- END PHASE 1 SYSTEM PROMPT ---\n\n")

        self._write("--- PHASE 1 USER PROMPT ---\n\n")
        self._write(f"[{num_images} CT slice images from {volume_path.name}]\n\n")
        self._write(user_prompt + "\n\n")
        self._write("--- END PHASE 1 USER PROMPT ---\n\n")

    def log_phase1_response(
        self, raw_response: str, fact_sheet, duration_secs: float
    ) -> None:
        ts = self._ts()
        self._write(f"Phase 1 Duration: {duration_secs:.2f}s (completed {ts})\n\n")

        self._write("--- PHASE 1 RAW RESPONSE ---\n\n")
        self._write(raw_response + "\n\n")
        self._write("--- END PHASE 1 RAW RESPONSE ---\n\n")

        # Parsed findings table
        self._write(f"Parsed Findings: {len(fact_sheet.findings)}\n\n")
        if fact_sheet.findings:
            self._write(f"{'#':<4}{'Finding':<20}{'Location':<12}{'Size':<10}{'Slice':<8}Description\n")
            self._write(f"{'—'*3:<4}{'—'*18:<20}{'—'*10:<12}{'—'*8:<10}{'—'*5:<8}{'—'*30}\n")
            for i, f in enumerate(fact_sheet.findings, 1):
                self._write(
                    f"{i:<4}{f.finding[:18]:<20}{f.location[:10]:<12}"
                    f"{f.size[:8]:<10}{f.slice_index:<8}{f.description[:40]}\n"
                )
        self._write("\n")

    # ------------------------------------------------------------------
    # Model Swap
    # ------------------------------------------------------------------

    def log_model_swap(self, from_model: str, to_model: str, duration_secs: float) -> None:
        ts = self._ts()
        header = f" MODEL SWAP                                               [{ts} | {duration_secs:.2f}s]"
        self._write("-" * 80 + "\n" + header + "\n" + "-" * 80 + "\n\n")
        self._write(f"Unloaded: {from_model}\n")
        self._write(f"Loading:  {to_model}\n\n")

    # ------------------------------------------------------------------
    # Step 4: Phase 2 — Clinical Reasoning (27B)
    # ------------------------------------------------------------------

    def log_phase2_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        ts = self._ts()
        header = f" STEP 4: PHASE 2 — CLINICAL REASONING (27B)               [{ts}]"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        self._write(f"Model: google/medgemma-27b-it (NF4 4-bit, double quant)\n")
        self._write(f"Mode: Text-only (Clinical narrative + Visual fact sheet)\n\n")

        self._write("--- PHASE 2 SYSTEM PROMPT ---\n\n")
        self._write(system_prompt + "\n\n")
        self._write("--- END PHASE 2 SYSTEM PROMPT ---\n\n")

        self._write("--- PHASE 2 USER PROMPT ---\n\n")
        self._write(user_prompt + "\n\n")
        self._write("--- END PHASE 2 USER PROMPT ---\n\n")

    def log_phase2_response(
        self, raw_response: str, delta_result, duration_secs: float
    ) -> None:
        ts = self._ts()
        self._write(f"Phase 2 Duration: {duration_secs:.2f}s (completed {ts})\n\n")

        self._write("--- PHASE 2 RAW RESPONSE ---\n\n")
        self._write(raw_response + "\n\n")
        self._write("--- END PHASE 2 RAW RESPONSE ---\n\n")

        # Delta analysis table
        self.log_delta_analysis_table(delta_result)

    def log_delta_analysis_table(self, delta_result) -> None:
        """Log formatted table of finding classifications."""
        header = " STEP 5: DELTA ANALYSIS RESULTS"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        if delta_result.delta_analysis:
            self._write(
                f"{'#':<4}{'Finding':<25}{'Classification':<18}{'Pri':<5}"
                f"{'History Match':<25}Reasoning\n"
            )
            self._write(
                f"{'—'*3:<4}{'—'*23:<25}{'—'*16:<18}{'—'*3:<5}"
                f"{'—'*23:<25}{'—'*30}\n"
            )
            for i, de in enumerate(delta_result.delta_analysis, 1):
                hm = str(de.history_match or "—")[:23]
                self._write(
                    f"{i:<4}{de.finding[:23]:<25}{de.classification:<18}"
                    f"{de.priority:<5}{hm:<25}{de.reasoning[:40]}\n"
                )
        else:
            self._write("No delta entries produced.\n")

        self._write(f"\nOverall Priority: {delta_result.overall_priority} "
                     f"({self._priority_name(delta_result.overall_priority)})\n")
        self._write(f"Rationale: {delta_result.priority_rationale}\n")
        self._write(f"Summary: {delta_result.findings_summary}\n\n")

    # ------------------------------------------------------------------
    # Step 6: Output saved
    # ------------------------------------------------------------------

    def log_output_saved(self, result_path: Path, priority_level: int) -> None:
        header = " STEP 6: OUTPUT SAVED"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        self._write(f"Result File: {result_path}\n")
        self._write(f"Priority: {priority_level} ({self._priority_name(priority_level)})\n\n")

    # ------------------------------------------------------------------
    # Legacy methods (kept for backward compatibility)
    # ------------------------------------------------------------------

    def log_medgemma_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        num_images: int,
        volume_path: Path,
    ) -> None:
        """Legacy — delegates to log_phase1_prompt."""
        self.log_phase1_prompt(system_prompt, user_prompt, num_images, volume_path)

    def log_medgemma_response(self, raw_response: str, duration_secs: float) -> None:
        """Legacy — logs raw response without fact sheet parsing."""
        ts = self._ts()
        self._write(f"Analysis Duration: {duration_secs:.2f}s (completed {ts})\n\n")
        self._write("--- MEDGEMMA RAW RESPONSE ---\n\n")
        self._write(raw_response + "\n\n")
        self._write("--- END MEDGEMMA RAW RESPONSE ---\n\n")

    def log_parsed_results(self, analysis, slice_indices: list) -> None:
        """Legacy — logs old-style parsed results."""
        header = " PARSED RESULTS (Legacy)"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        sampled_idx = analysis.key_slice_index
        original_idx = slice_indices[sampled_idx] if sampled_idx < len(slice_indices) else sampled_idx
        priority_label = f"{analysis.priority_level} ({self._priority_name(analysis.priority_level)})"

        fields = [
            ("VISUAL_FINDINGS", analysis.visual_findings),
            ("KEY_SLICE_INDEX", f"{sampled_idx} (sampled) -> {original_idx} (original volume index)"),
            ("PRIORITY_LEVEL", priority_label),
            ("PRIORITY_RATIONALE", analysis.priority_rationale),
            ("FINDINGS_SUMMARY", analysis.findings_summary),
            ("CONDITIONS_CONSIDERED", ", ".join(analysis.conditions_considered)),
        ]

        self._write(f"{'Field':<23}Extracted Value\n")
        self._write(f"{'-----':<23}---------------\n")
        for name, value in fields:
            self._write(f"{name + ':':<23}{value}\n")
        self._write("\n")

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def log_error(self, patient_id: str, error: str) -> None:
        ts = self._ts()
        self._write(
            "!" * 80 + "\n"
            + f" ERROR processing {patient_id} at {ts}\n"
            + "!" * 80 + "\n\n"
            + error + "\n\n"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def log_path(self) -> Path:
        return self._log_path
