# Unified FHIR + CT Volume Pipeline

This document describes the `synthetic_fhir_pipeline.py` script which generates synthetic FHIR patient records from radiology reports and organizes them alongside CT volumes in a unified folder structure.

## Overview

The pipeline:
1. Extracts structured clinical data from radiology reports using OpenAI (gpt-4o)
2. Configures Synthea to generate matching patient profiles
3. Merges radiology DiagnosticReport into the FHIR bundle (US Core R4 compliant)
4. Outputs a unified folder structure with FHIR records and CT volumes together

## Project Structure

```
sentinel_x/
├── scripts/
│   ├── data_loading.py           # Downloads CT-RATE dataset
│   ├── fhir_creation.py          # Downloads Synthea JAR
│   ├── synthetic_fhir_pipeline.py # Main FHIR pipeline
│   └── simulator.py              # Standalone simulator (optional)
├── lib/
│   └── synthea-with-dependencies.jar
├── docs/
│   └── unified_fhir_pipeline.md  # This file
└── data/
    └── raw_ct_rate/
        ├── volumes/              # CT volumes (.nii.gz)
        ├── reports/              # Radiology reports (.json)
        └── combined/             # Pipeline output
```

## Prerequisites

- Python 3.10+
- **Java 11+** (required for Synthea)
  ```bash
  # Ubuntu/Debian (with sudo)
  sudo apt update && sudo apt install -y openjdk-11-jdk

  # Ubuntu/Debian in Docker (as root, no sudo)
  apt update && apt install -y openjdk-11-jdk

  # macOS (with Homebrew)
  brew install openjdk@11

  # Verify installation
  java -version
  ```
- Synthea JAR file (run `python sentinel_x/scripts/fhir_creation.py` to download)
- OpenAI API key set as `OPENAI_API_KEY` environment variable
- Required Python packages: `openai`, `pydantic`, `python-dotenv`

## Input Structure

The pipeline expects a data directory with the following structure:

```
data-dir/
├── volumes/           # CT scan volumes
│   ├── train_1_a_1.nii.gz
│   ├── train_1_a_2.nii.gz
│   └── ...
└── reports/           # Radiology reports (JSON)
    ├── train_1_a_1.json
    ├── train_1_a_2.json
    └── ...
```

Report files should be JSON with fields: `clinical_information`, `technique`, `findings`, `impressions`.

## Output Structure

The pipeline creates a `combined/` directory with per-patient folders:

```
data-dir/combined/
├── manifest.json          # Index of all data points
├── .checkpoint.json       # Processing checkpoint (for resume)
├── processing_log.json    # Detailed processing log
├── train_1_a_1/
│   ├── fhir.json          # FHIR bundle (US Core R4)
│   ├── volume.nii.gz      # Symlink or copy of CT scan
│   ├── report.json        # Original radiology report
│   └── report.txt         # Human-readable report
└── ...
```

### manifest.json

The manifest provides an index for easy iteration:

```json
{
  "created": "2026-01-29T12:00:00Z",
  "total_patients": 5,
  "patients": [
    {
      "id": "train_1_a_1",
      "folder": "train_1_a_1",
      "fhir_path": "train_1_a_1/fhir.json",
      "volume_path": "train_1_a_1/volume.nii.gz",
      "report_json_path": "train_1_a_1/report.json",
      "report_txt_path": "train_1_a_1/report.txt",
      "patient_fhir_id": "uuid-from-fhir",
      "conditions_count": 8
    }
  ]
}
```

## Usage

All commands should be run from the workspace root directory.

### Basic Usage

Process all reports in the default data directory:

```bash
python sentinel_x/scripts/synthetic_fhir_pipeline.py
```

### Specify Data Directory

```bash
python sentinel_x/scripts/synthetic_fhir_pipeline.py --data-dir /path/to/data
```

### Custom Output Directory

By default, output goes to `{data-dir}/combined/`. Override with:

```bash
python sentinel_x/scripts/synthetic_fhir_pipeline.py --output-dir /path/to/output
```

### Copy Volumes Instead of Symlinking

By default, volumes are symlinked to save disk space. To copy instead:

```bash
python sentinel_x/scripts/synthetic_fhir_pipeline.py --copy-volumes
```

### Process a Single Report

```bash
python sentinel_x/scripts/synthetic_fhir_pipeline.py --report train_1_a_1.json
```

### Control Concurrency

Limit concurrent OpenAI API requests (default: 3):

```bash
python sentinel_x/scripts/synthetic_fhir_pipeline.py --max-concurrent 5
```

## CLI Reference

```
usage: synthetic_fhir_pipeline.py [-h] [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
                                  [--report REPORT] [--copy-volumes] [--max-concurrent MAX_CONCURRENT]

Generate synthetic FHIR patient records from radiology reports

options:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Input directory containing volumes/ and reports/
                        (default: sentinel_x/data/raw_ct_rate)
  --output-dir OUTPUT_DIR
                        Output directory for combined data
                        (default: {data-dir}/combined)
  --report REPORT       Process a single report (filename, e.g., train_1_a_1.json)
  --copy-volumes        Copy volumes instead of symlinking (uses more disk space)
  --max-concurrent MAX_CONCURRENT
                        Maximum concurrent OpenAI requests (default: 3)
```

## Iterating Over Combined Data

### Python Example

```python
import json
from pathlib import Path

data_dir = Path("sentinel_x/data/raw_ct_rate/combined")

with open(data_dir / "manifest.json") as f:
    manifest = json.load(f)

print(f"Total patients: {manifest['total_patients']}")

for patient in manifest["patients"]:
    fhir_path = data_dir / patient["fhir_path"]
    volume_path = data_dir / patient["volume_path"]
    report_json_path = data_dir / patient.get("report_json_path", "")
    report_txt_path = data_dir / patient.get("report_txt_path", "")

    # Load FHIR bundle
    with open(fhir_path) as f:
        fhir_bundle = json.load(f)

    # Check if volume exists (symlink or file)
    if volume_path.exists():
        print(f"Processing {patient['id']}: {patient['conditions_count']} conditions")
        # Load volume with nibabel, etc.
```

### Bash Example

```bash
# List all patient folders
ls -la sentinel_x/data/raw_ct_rate/combined/

# View manifest
cat sentinel_x/data/raw_ct_rate/combined/manifest.json | jq .

# Check a specific patient's files
ls -la sentinel_x/data/raw_ct_rate/combined/train_1_a_1/

# Verify symlinks work
file sentinel_x/data/raw_ct_rate/combined/train_1_a_1/volume.nii.gz
```

## Resume Support

The pipeline saves progress to `.checkpoint.json` after each report. If interrupted, simply re-run the same command to resume from where it left off.

To force reprocessing, delete the checkpoint file:

```bash
rm sentinel_x/data/raw_ct_rate/combined/.checkpoint.json
```

## FHIR Bundle Contents

Each `fhir.json` contains a US Core R4 compliant FHIR Bundle with:

- **Patient**: Synthetic patient demographics from Synthea
- **ImagingStudy**: Links to the CT volume file
- **DiagnosticReport**: Contains the radiology report text and conclusion codes
- **Condition**: One resource per extracted medical condition (with SNOMED codes)
- Additional Synthea-generated resources (encounters, observations, etc.)

## Troubleshooting

### "Java is not installed or not in PATH"
Install Java 11 or newer:
```bash
# Ubuntu/Debian (with sudo)
sudo apt update && sudo apt install -y openjdk-11-jdk

# Ubuntu/Debian in Docker (as root, no sudo)
apt update && apt install -y openjdk-11-jdk

# macOS (with Homebrew)
brew install openjdk@11

# Verify installation
java -version
```

### "Synthea JAR not found"
Run the fhir_creation script to download the Synthea JAR file:
```bash
python sentinel_x/scripts/fhir_creation.py
```

### "OPENAI_API_KEY environment variable not set"
Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-key-here
```
Or add it to a `.env` file in the workspace root.

### "Volumes directory not found"
The pipeline will proceed without volume linking. Volumes can be added later manually.

### Broken symlinks
If you moved the data directory, symlinks may break. Re-run with `--copy-volumes` or recreate symlinks manually.
