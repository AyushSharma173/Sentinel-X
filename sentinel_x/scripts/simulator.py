#!/usr/bin/env python3
"""
CT Scan Simulator

Simulates incoming CT scans by copying patient data from the combined/
directory to an inbox folder every 10 seconds.

For each patient folder, copies volume_1.nii.gz (first reconstruction)
and fhir.json to the inbox using patient-level IDs (e.g., train_1).

This is a standalone utility script. The demo service (demo_service.py)
has built-in simulation capabilities, so this script is optional.
"""

import random
import shutil
import time
from pathlib import Path


# Use relative paths based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent  # sentinel_x/


def main():
    # Define paths relative to project directory
    combined_dir = PROJECT_DIR / "data" / "raw_ct_rate" / "combined"
    inbox_dir = PROJECT_DIR / "inbox"

    # Create inbox subdirectories
    inbox_volumes = inbox_dir / "volumes"
    inbox_reports = inbox_dir / "reports"
    inbox_volumes.mkdir(parents=True, exist_ok=True)
    inbox_reports.mkdir(parents=True, exist_ok=True)

    # Discover patient folders from combined directory
    if not combined_dir.exists():
        print(f"Combined directory not found: {combined_dir}")
        print("Run synthetic_fhir_pipeline.py first to generate patient data.")
        return

    patient_folders = [
        d for d in sorted(combined_dir.iterdir())
        if d.is_dir() and (d / "fhir.json").exists()
    ]

    if not patient_folders:
        print(f"No patient folders found in {combined_dir}")
        return

    print(f"Found {len(patient_folders)} patients in {combined_dir}")
    print(f"Inbox directory: {inbox_dir}")
    print("-" * 50)

    # Track which patients haven't been copied yet
    remaining = patient_folders.copy()
    random.shuffle(remaining)

    while remaining:
        patient_folder = remaining.pop()
        patient_id = patient_folder.name  # e.g., "train_1"

        # Copy volume_1.nii.gz to inbox (first reconstruction only for triage)
        volume_src = patient_folder / "volume_1.nii.gz"
        if volume_src.exists() or volume_src.is_symlink():
            actual = volume_src.resolve() if volume_src.is_symlink() else volume_src
            shutil.copy2(actual, inbox_volumes / f"{patient_id}.nii.gz")
        else:
            print(f"[WARN] No volume_1.nii.gz in {patient_folder.name}, skipping")
            continue

        # Copy FHIR bundle as report file
        fhir_src = patient_folder / "fhir.json"
        if fhir_src.exists():
            shutil.copy2(fhir_src, inbox_reports / f"{patient_id}.json")

        # Copy report.txt for debugging/display (optional)
        report_txt = patient_folder / "report.txt"
        if report_txt.exists():
            shutil.copy2(report_txt, inbox_reports / f"{patient_id}.txt")

        print(f"[{time.strftime('%H:%M:%S')}] Copied: {patient_id} (volume_1 + fhir)")
        print(f"  Remaining: {len(remaining)} patients")

        # If there are more patients, wait 10 seconds
        if remaining:
            print(f"  Next patient in 10 seconds...")
            time.sleep(10)

    print("-" * 50)
    print("All patients have been copied to inbox. Simulator complete.")


if __name__ == "__main__":
    main()
