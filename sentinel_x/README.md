# Sentinel-X

Medical imaging analysis pipeline that processes CT scans with synthetic FHIR patient records.

## Project Structure

```
sentinel_x/
├── README.md                      # This file
├── run_demo.py                    # Demo entry point
├── requirements-api.txt
├── scripts/                       # All data/utility scripts
│   ├── data_loading.py            # Downloads CT-RATE dataset
│   ├── fhir_creation.py           # Downloads Synthea JAR
│   ├── synthetic_fhir_pipeline.py # Generates FHIR bundles
│   └── simulator.py               # Standalone simulator (optional)
├── api/                           # REST API backend
├── triage/                        # AI triage module
├── frontend/                      # React UI
├── lib/                           # External binaries
│   └── synthea-with-dependencies.jar
├── docs/                          # Documentation
│   └── unified_fhir_pipeline.md
├── data/                          # Data directory
│   └── raw_ct_rate/
│       ├── volumes/               # Raw CT volumes
│       ├── reports/               # Raw radiology reports
│       └── combined/              # Unified output (FHIR + volumes + reports)
└── inbox/                         # Simulator output directory
```

## Quick Start

### 1. Install Prerequisites

**Java 11+** is required for Synthea (FHIR patient generation):

```bash
# For me this worked:
apt update && apt install -y openjdk-11-jdk

# Ubuntu/Debian (with sudo)
sudo apt update && sudo apt install -y openjdk-11-jdk

# Ubuntu/Debian in Docker (as root, no sudo)
apt update && apt install -y openjdk-11-jdk

# macOS (with Homebrew)
brew install openjdk@11

# Verify installation
java -version
```

### 2. Setup Python Environment

```bash
cd Sentinel-X
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r sentinel_x/requirements-api.txt
pip install openai pydantic python-dotenv datasets huggingface_hub pandas tqdm
```

### 3. Download Data (3 steps)

```bash
# Step 1: Download CT-RATE dataset from HuggingFace
python sentinel_x/scripts/data_loading.py

# Step 2: Download Synthea JAR (for synthetic patient generation)
# Requires Java 11+ to be installed (see step 1)
python sentinel_x/scripts/fhir_creation.py

# Step 3: Generate FHIR records (requires OpenAI API key)
export OPENAI_API_KEY="sk-..."
python sentinel_x/scripts/synthetic_fhir_pipeline.py
```

### 4. Run the Demo

```bash
cd sentinel_x
python run_demo.py --reload
# Open http://localhost:8000 in browser
```

The API server will start at `http://localhost:8000`. You can view the API docs at `http://localhost:8000/docs`.

### 5. Start the Frontend (optional)

In a second terminal:

```bash
cd sentinel_x/frontend
npm install   # first time only
npm run dev -- --host
```

The frontend will start at `http://localhost:5173`.

### Access the Application

- **Frontend UI:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/data_loading.py` | Downloads CT volumes and reports from HuggingFace CT-RATE |
| `scripts/fhir_creation.py` | Downloads Synthea JAR for patient generation |
| `scripts/synthetic_fhir_pipeline.py` | Generates FHIR bundles and creates combined folder |
| `scripts/simulator.py` | Standalone simulator (optional, demo has built-in) |

## Documentation

- [Unified FHIR Pipeline](docs/unified_fhir_pipeline.md) - Detailed documentation for the FHIR generation pipeline

## Data Flow

```
CT-RATE Dataset (HuggingFace)
        │
        ▼ scripts/data_loading.py
   raw_ct_rate/
   ├── volumes/     (.nii.gz files)
   └── reports/     (.json + .txt files)
        │
        ▼ scripts/synthetic_fhir_pipeline.py
   raw_ct_rate/combined/
   ├── manifest.json
   └── train_1_a_1/
       ├── fhir.json       (US Core R4 FHIR Bundle)
       ├── volume.nii.gz   (symlink to CT volume)
       ├── report.json     (original radiology report)
       └── report.txt      (human-readable report)
```

## Configuration

Environment variables (set in `.env`):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for condition extraction |

## Requirements

- Python 3.10+
- Node.js 18+ (for frontend)
- Java 11+ (for Synthea)
- ~500MB disk space for Synthea JAR
- CT volumes are large (~100MB each)

### Python Dependencies

For the demo API:

```bash
pip install -r requirements-api.txt
```

For the data pipeline:

```bash
pip install openai pydantic python-dotenv datasets huggingface_hub pandas tqdm
```
