# Sentinel-X

Medical imaging analysis pipeline that processes CT scans with synthetic FHIR patient records.

## Project Structure

```
sentinel_x/
├── README.md                      # This file
├── docs/                          # Documentation
│   └── unified_fhir_pipeline.md   # FHIR pipeline docs
├── scripts/                       # Pipeline scripts
│   ├── fhir_creation.py           # Downloads Synthea JAR
│   └── synthetic_fhir_pipeline.py # Main FHIR generation pipeline
├── lib/                           # External binaries
│   └── synthea-with-dependencies.jar
├── data/                          # Data directory
│   └── raw_ct_rate/
│       ├── volumes/               # CT volumes (.nii.gz)
│       ├── reports/               # Radiology reports (.json, .txt)
│       └── combined/              # Unified FHIR + volume output
├── data_loading.py                # Downloads CT-RATE dataset from HuggingFace
├── simulator.py                   # Simulates incoming CT scans
└── inbox/                         # Simulator output directory
```

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install openai pydantic python-dotenv datasets huggingface_hub pandas tqdm

# Set OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Download Data

Download CT-RATE dataset (5 samples by default):

```bash
python sentinel_x/data_loading.py
```

### 3. Download Synthea

```bash
python sentinel_x/scripts/fhir_creation.py
```

### 4. Generate FHIR Records

```bash
python sentinel_x/scripts/synthetic_fhir_pipeline.py
```

Output will be in `sentinel_x/data/raw_ct_rate/combined/` with each patient having their own folder containing `fhir.json` and `volume.nii.gz`.

## Scripts

| Script | Purpose |
|--------|---------|
| `data_loading.py` | Downloads CT volumes and radiology reports from HuggingFace CT-RATE dataset |
| `scripts/fhir_creation.py` | Downloads Synthea JAR for synthetic patient generation |
| `scripts/synthetic_fhir_pipeline.py` | Main pipeline: extracts conditions from reports, generates FHIR bundles, links volumes |
| `simulator.py` | Simulates incoming CT scans by copying files to inbox every 10 seconds |

## Documentation

- [Unified FHIR Pipeline](docs/unified_fhir_pipeline.md) - Detailed documentation for the FHIR generation pipeline

## Data Flow

```
CT-RATE Dataset (HuggingFace)
        │
        ▼ data_loading.py
   raw_ct_rate/
   ├── volumes/
   └── reports/
        │
        ▼ synthetic_fhir_pipeline.py
   raw_ct_rate/combined/
   ├── manifest.json
   ├── train_1_a_1/
   │   ├── fhir.json      (US Core R4 FHIR Bundle)
   │   └── volume.nii.gz  (symlink to CT volume)
   └── ...
```

## Configuration

Environment variables (set in `.env`):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for condition extraction |

## Requirements

- Python 3.10+
- Java 11+ (for Synthea)
- ~500MB disk space for Synthea JAR
- CT volumes are large (~100MB each)
