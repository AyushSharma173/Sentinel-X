#!/usr/bin/env python3
"""
CT Scan Simulator

Simulates incoming CT scans by copying random .nii.gz files
from the volumes directory to an inbox folder every 10 seconds.
Also copies the corresponding radiology report if available.

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
    volumes_dir = PROJECT_DIR / "data" / "raw_ct_rate" / "volumes"
    reports_dir = PROJECT_DIR / "data" / "raw_ct_rate" / "reports"
    inbox_dir = PROJECT_DIR / "inbox"

    # Create inbox subdirectories
    inbox_volumes = inbox_dir / "volumes"
    inbox_reports = inbox_dir / "reports"
    inbox_volumes.mkdir(parents=True, exist_ok=True)
    inbox_reports.mkdir(parents=True, exist_ok=True)

    # Get all .nii.gz files from volumes directory
    all_scans = list(volumes_dir.glob("*.nii.gz"))

    if not all_scans:
        print(f"No .nii.gz files found in {volumes_dir}")
        return

    print(f"Found {len(all_scans)} CT scans in {volumes_dir}")
    print(f"Inbox directory: {inbox_dir}")
    print("-" * 50)

    # Track which scans haven't been copied yet
    remaining_scans = all_scans.copy()
    random.shuffle(remaining_scans)

    while remaining_scans:
        # Pick and remove a random scan from remaining
        scan = remaining_scans.pop()
        base_name = scan.stem.replace(".nii", "")  # e.g., "train_1_a_1"

        # Copy volume to inbox
        dest_volume = inbox_volumes / scan.name
        shutil.copy2(scan, dest_volume)

        # Copy corresponding report files if they exist
        report_json = reports_dir / f"{base_name}.json"
        report_txt = reports_dir / f"{base_name}.txt"

        copied_report = False
        if report_json.exists():
            shutil.copy2(report_json, inbox_reports / report_json.name)
            copied_report = True
        if report_txt.exists():
            shutil.copy2(report_txt, inbox_reports / report_txt.name)
            copied_report = True

        report_status = "+ report" if copied_report else "(no report)"
        print(f"[{time.strftime('%H:%M:%S')}] Copied: {scan.name} {report_status}")
        print(f"  Remaining: {len(remaining_scans)} scans")

        # If there are more scans, wait 10 seconds
        if remaining_scans:
            print(f"  Next scan in 10 seconds...")
            time.sleep(10)

    print("-" * 50)
    print("All CT scans have been copied to inbox. Simulator complete.")


if __name__ == "__main__":
    main()
