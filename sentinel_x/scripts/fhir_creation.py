#!/usr/bin/env python3
"""
Script to download Synthea and generate synthetic FHIR patient data.

Usage:
    python sentinel_x/scripts/fhir_creation.py
"""

import subprocess
import sys
import os
import urllib.request
from pathlib import Path

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent  # sentinel_x/

SYNTHEA_VERSION = "v3.3.0"
SYNTHEA_JAR_NAME = "synthea-with-dependencies.jar"
SYNTHEA_JAR = PROJECT_DIR / "lib" / SYNTHEA_JAR_NAME
SYNTHEA_URL = f"https://github.com/synthetichealth/synthea/releases/download/{SYNTHEA_VERSION}/{SYNTHEA_JAR_NAME}"
OUTPUT_DIR = PROJECT_DIR / "data" / "synthea_output"


def check_java():
    """Check if Java is installed and available."""
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Java is available.")
            return True
    except FileNotFoundError:
        pass

    print("Error: Java is not installed or not in PATH.")
    print("Please install Java 11 or newer (JDK).")
    sys.exit(1)


def download_synthea():
    """Download Synthea JAR if not present."""
    if SYNTHEA_JAR.exists():
        print(f"Synthea JAR already exists: {SYNTHEA_JAR}")
        return

    # Ensure lib directory exists
    SYNTHEA_JAR.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Synthea from {SYNTHEA_URL}...")
    try:
        urllib.request.urlretrieve(SYNTHEA_URL, SYNTHEA_JAR)
        print(f"Download complete: {SYNTHEA_JAR}")
    except Exception as e:
        print(f"Error downloading Synthea: {e}")
        sys.exit(1)


def generate_fhir_data(num_patients=10, state="Massachusetts"):
    """Generate synthetic FHIR patient data using Synthea."""
    print(f"Generating {num_patients} synthetic patients in {state}...")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java", "-jar", str(SYNTHEA_JAR),
        "-p", str(num_patients),
        f"--exporter.baseDirectory={OUTPUT_DIR}",
        "--exporter.fhir.export=true",
        "--exporter.ccda.export=false",
        "--exporter.csv.export=false",
        state
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\nFHIR data generation complete!")
        print(f"Output location: {OUTPUT_DIR}")

        # List generated files
        fhir_dir = OUTPUT_DIR / "fhir"
        if fhir_dir.exists():
            fhir_files = list(fhir_dir.glob("*.json"))
            print(f"Generated {len(fhir_files)} FHIR bundle(s).")

    except subprocess.CalledProcessError as e:
        print(f"Error running Synthea: {e}")
        sys.exit(1)


def main():
    print("=" * 50)
    print("Synthea FHIR Data Generator")
    print("=" * 50)

    check_java()
    download_synthea()
    generate_fhir_data(num_patients=10)

    print("\nDone!")


if __name__ == "__main__":
    main()
