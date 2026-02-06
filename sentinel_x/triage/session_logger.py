"""Human-readable session trace logger for Sentinel-X pipeline.

Writes a single .txt log file per session that traces the full pipeline
for every patient: FHIR extraction, MedGemma prompt/response, and parsed results.
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
            + " " * 22 + "SENTINEL-X TRIAGE SESSION LOG\n"
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
        self._write("--- CLINICAL NARRATIVE (sent to MedGemma as context) ---\n\n")
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
        header = f" STEP 2: CT VOLUME PROCESSING                            [{ts} | {duration_secs:.2f}s]"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        self._write(f"Volume File: {volume_path}\n")
        self._write(f"Total Slices Sampled: {num_slices}\n")
        # Show first ~15 indices for readability
        indices_str = str(slice_indices[:15])
        if len(slice_indices) > 15:
            indices_str = indices_str[:-1] + ", ...]"
        self._write(f"Slice Indices: {indices_str}\n")
        self._write(f"Image Dimensions: {image_size[0]} x {image_size[1]} (RGB)\n\n")

    # ------------------------------------------------------------------
    # Step 3: MedGemma analysis (prompt + response)
    # ------------------------------------------------------------------

    def log_medgemma_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        num_images: int,
        volume_path: Path,
    ) -> None:
        ts = self._ts()
        # Write step header (duration will be filled in by response half)
        header = f" STEP 3: MEDGEMMA ANALYSIS                               [{ts}]"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        self._write(f"Model: google/medgemma-4b-it\n\n")

        self._write("--- SYSTEM PROMPT ---\n\n")
        self._write(system_prompt + "\n\n")
        self._write("--- END SYSTEM PROMPT ---\n\n")

        self._write("--- USER PROMPT ---\n\n")
        self._write(f"[{num_images} CT slice images from {volume_path.name}]\n\n")
        self._write(user_prompt + "\n\n")
        self._write("--- END USER PROMPT ---\n\n")

    def log_medgemma_response(self, raw_response: str, duration_secs: float) -> None:
        ts = self._ts()
        self._write(f"Analysis Duration: {duration_secs:.2f}s (completed {ts})\n\n")

        self._write("--- MEDGEMMA RAW RESPONSE ---\n\n")
        self._write(raw_response + "\n\n")
        self._write("--- END MEDGEMMA RAW RESPONSE ---\n\n")

    # ------------------------------------------------------------------
    # Step 4: Parsed results
    # ------------------------------------------------------------------

    def log_parsed_results(self, analysis, slice_indices: list) -> None:
        header = " STEP 4: PARSED RESULTS"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        # Calculate original volume index for key slice
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
            # Truncate long values for table display but show full text
            self._write(f"{name + ':':<23}{value}\n")
        self._write("\n")

    # ------------------------------------------------------------------
    # Step 5: Output saved
    # ------------------------------------------------------------------

    def log_output_saved(self, result_path: Path, priority_level: int) -> None:
        header = " STEP 5: OUTPUT SAVED"
        self._write("=" * 80 + "\n" + header + "\n" + "=" * 80 + "\n\n")

        self._write(f"Result File: {result_path}\n")
        self._write(f"Priority: {priority_level} ({self._priority_name(priority_level)})\n\n")

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
