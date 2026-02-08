# Sentinel-X

AI-powered CT triage system using a **Serial Late Fusion** architecture: a 4B vision model detects findings, then a 27B reasoning model performs clinical triage — running sequentially on a single 24GB GPU.

## Architecture

```
CT Volume  ──→  Multi-Window RGB  ──→  4B Vision (Phase 1)  ──→  Visual Fact Sheet
FHIR Bundle ──→  FHIRJanitor  ──→  Clinical Narrative  ─────────────────────┐
                                                                             ▼
                                              27B Reasoner (Phase 2)  ←──  Delta Analysis
                                                        │
                                                        ▼
                                              triage_result.json + Worklist
```

- **Phase 1** (`google/medgemma-1.5-4b-it`, BF16): Vision-only analysis of CT slices — no clinical context, preventing cognitive bias
- **Phase 2** (`unsloth/medgemma-27b-text-it-unsloth-bnb-4bit`, pre-quantized NF4): Text-only Delta Analysis comparing visual findings against EHR history
- Models are loaded/unloaded serially — they never coexist in VRAM

See [Architecture Documentation](docs/Architecture/SERIAL_LATE_FUSION_ARCHITECTURE.md) for full details.

## GPU Requirements

| Requirement | Value |
|-------------|-------|
| **Minimum GPU** | NVIDIA RTX 4090 (24GB VRAM) or equivalent |
| **Phase 1 peak** | ~11 GB |
| **Phase 2 peak** | ~17 GB |
| **CUDA** | Required (11.8+) |
| **Other GPU processes** | Must be stopped — the pipeline needs the full 24GB |

**Before running the demo**, verify your GPU is clean:

```bash
# Check GPU memory — "Used" should be near 0 MiB
nvidia-smi

# Kill anything else using the GPU (if needed)
# nvidia-smi shows PIDs of processes using GPU memory
```

## Project Structure

```
sentinel_x/
├── README.md
├── run_demo.py                    # Demo entry point
├── requirements-api.txt
├── scripts/                       # Data pipeline scripts
│   ├── data_loading.py            # Downloads CT-RATE dataset
│   ├── fhir_creation.py           # Downloads Synthea JAR
│   ├── synthetic_fhir_pipeline.py # Generates FHIR bundles
│   └── simulator.py               # Standalone simulator (optional)
├── triage/                        # AI triage pipeline
│   ├── agent.py                   # Main pipeline orchestrator
│   ├── medgemma_analyzer.py       # Phase 1: VisionAnalyzer (4B)
│   ├── medgemma_reasoner.py       # Phase 2: ClinicalReasoner (27B)
│   ├── vram_manager.py            # GPU memory management
│   ├── ct_processor.py            # Multi-window CT preprocessing
│   ├── prompts.py                 # Phase 1 + Phase 2 prompt templates
│   ├── output_generator.py        # Merges both phases → JSON
│   ├── fhir_janitor.py            # FHIR clinical context extraction
│   ├── session_logger.py          # Human-readable pipeline trace
│   ├── config.py                  # All constants and paths
│   ├── worklist.py                # Priority-sorted worklist
│   ├── inbox_watcher.py           # File watcher for incoming patients
│   └── json_repair.py             # Handles 4B model JSON quirks
├── api/                           # FastAPI backend
│   ├── main.py
│   ├── models.py                  # Pydantic models + WebSocket events
│   ├── routes/
│   │   ├── demo.py
│   │   ├── patients.py
│   │   ├── websocket.py
│   │   └── worklist.py
│   └── services/
│       ├── demo_service.py        # Demo orchestration
│       └── ws_manager.py
├── frontend/                      # React + TypeScript UI
├── docs/
│   └── Architecture/
│       └── SERIAL_LATE_FUSION_ARCHITECTURE.md
├── data/raw_ct_rate/              # CT volumes + FHIR records
│   └── combined/                  # Unified patient folders
└── inbox/                         # Runtime inbox for simulator
```

## Quick Start

### 1. Install Prerequisites

**Java 11+** (for Synthea FHIR generation):

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y openjdk-11-jdk

# macOS
brew install openjdk@11

java -version
```

### 2. Setup Python Environment

```bash
cd Sentinel-X
python -m venv venv
source venv/bin/activate

# Core dependencies (API + ML pipeline)
pip install -r sentinel_x/requirements-api.txt

# Data pipeline dependencies
pip install openai pydantic python-dotenv datasets huggingface_hub pandas tqdm nibabel numpy
```

**Key dependency: `bitsandbytes`** — required for the 27B model's NF4 quantization. It's included in `requirements-api.txt` and requires CUDA to be installed.

### 3. Authenticate with HuggingFace

The MedGemma models are gated. You must:

1. Accept the license at [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
2. Login:

```bash
huggingface-cli login
# Enter your access token when prompted
```

### 4. Download Data

```bash
# Step 1: Download CT-RATE dataset from HuggingFace
python sentinel_x/scripts/data_loading.py

# Step 2: Download Synthea JAR
python sentinel_x/scripts/fhir_creation.py

# Step 3: Generate FHIR records (requires OpenAI API key)
export OPENAI_API_KEY="sk-..."
python sentinel_x/scripts/synthetic_fhir_pipeline.py
```

### 5. Pre-Download Models (Recommended)

Download both models before running the demo to avoid timeouts during the first patient:

```bash
python -c "
from huggingface_hub import snapshot_download
print('Downloading MedGemma 1.5 4B (Phase 1 vision)...')
snapshot_download('google/medgemma-1.5-4b-it')
print('Downloading MedGemma 27B pre-quantized (Phase 2 reasoning)...')
snapshot_download('unsloth/medgemma-27b-text-it-unsloth-bnb-4bit')
print('Done — both models cached locally.')
"
```

This downloads ~21GB total to the HuggingFace cache. Subsequent loads will be from disk.

> **Storage note:** The Phase 2 model uses Unsloth's pre-quantized BnB NF4 version (16.6GB)
> instead of the full-precision 27B weights (54GB). If your disk is tight, consider
> symlinking `~/.cache/huggingface` to a larger volume before downloading.

### 6. Verify GPU (Important)

```bash
# Confirm you have an RTX 4090 or equivalent with 24GB+
nvidia-smi

# Verify CUDA is accessible from Python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"

# Verify bitsandbytes works (needed for 27B quantization)
python -c "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__} OK')"
```

Expected output:
```
CUDA: True, GPU: NVIDIA GeForce RTX 4090, VRAM: 24.0GB
bitsandbytes 0.43.x OK
```

### 7. Run the Demo

**Terminal 1 — Backend:**

```bash
cd sentinel_x
python run_demo.py
# API starts at http://localhost:8000
```

**Terminal 2 — Frontend:**

```bash
cd sentinel_x/frontend
npm install   # first time only
npm run dev -- --host
# UI starts at http://localhost:5173
```

Then open `http://localhost:5173` and click **Start Demo**.

### Access Points

| Service | URL |
|---------|-----|
| Frontend UI | http://localhost:5173 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/api/health |

## What Happens During the Demo

1. **Simulator** copies patient data (CT volume + FHIR bundle) to the inbox every 10 seconds
2. **Agent** picks up each patient and runs the Serial Late Fusion pipeline:
   - FHIR extraction + CT multi-window preprocessing (~5-10s, CPU only)
   - Phase 1: Load 4B → vision detection → unload (~30-60s)
   - Model swap: unload 4B, load 27B (~30-60s)
   - Phase 2: Delta analysis → unload (~30-90s)
   - Output generation + worklist update (<1s)
3. **WebSocket** pushes real-time updates to the frontend (phase progress, results)
4. **Worklist** displays priority-sorted results as they complete

Total per-patient time: **~2-4 minutes** (first patient may be slower due to model download/cache warming).

## Troubleshooting

### OOM / GPU Freeze

If the GPU runs out of memory:

```bash
# Check what's using GPU memory
nvidia-smi

# Kill stuck Python processes
pkill -9 -f "python.*run_demo"

# Reset GPU state (if nvidia-smi shows memory still allocated)
nvidia-smi --gpu-reset  # requires root, use only if stuck

# Then restart the demo
```

The pipeline has built-in safety guards:
- **Pre-flight check**: Verifies GPU has 22GB+ total and warns if other processes are using VRAM
- **Per-phase check**: Aborts with a clear error if insufficient free VRAM before loading each model
- **max_memory cap**: Phase 2 is capped at 20GB GPU allocation to leave headroom
- **Automatic cleanup**: Models are unloaded even on errors (try/finally in the agent)

### Model Download Issues

```bash
# If a model fails to download mid-pipeline, pre-cache it:
huggingface-cli download google/medgemma-1.5-4b-it
huggingface-cli download google/medgemma-27b-it
```

### bitsandbytes Issues

```bash
# If bitsandbytes fails to import, reinstall with CUDA support:
pip install bitsandbytes --force-reinstall

# Check CUDA version compatibility:
python -c "import torch; print(torch.version.cuda)"
```

## Configuration

Environment variables (set in `.env`):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for FHIR condition extraction |
| `HF_TOKEN` | HuggingFace token (alternative to `huggingface-cli login`) |

Key constants in `triage/config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `VISION_MODEL_ID` | `google/medgemma-1.5-4b-it` | Phase 1 model |
| `REASONER_MODEL_ID` | `unsloth/medgemma-27b-text-it-unsloth-bnb-4bit` | Phase 2 model (pre-quantized) |
| `CT_NUM_SLICES` | `85` | Slices sampled per volume |
| `CT_WINDOW_WIDE` | `(-1024, 1024)` | R channel HU range |
| `CT_WINDOW_SOFT` | `(-135, 215)` | G channel HU range |
| `CT_WINDOW_BRAIN` | `(0, 80)` | B channel HU range |

## Session Logs

Each demo run creates a human-readable trace at `logs/sessions/{timestamp}/session.txt` with:
- FHIR extraction details
- CT processing info (multi-window channels)
- Phase 1 prompt, raw response, and parsed findings table
- Model swap VRAM status
- Phase 2 prompt, raw response, and delta analysis table
- Final priority and timing breakdown

## Requirements

- Python 3.10+
- Node.js 18+ (for frontend)
- Java 11+ (for Synthea)
- NVIDIA GPU with 24GB+ VRAM (RTX 4090 or better)
- CUDA 11.8+
- ~21GB disk for model cache (4.8GB Phase 1 + 16.6GB Phase 2 pre-quantized)
- ~500MB for Synthea JAR
- CT volumes are ~100MB each

### Python Dependencies

```bash
# Core (API + ML pipeline)
pip install -r requirements-api.txt

# Data pipeline
pip install openai pydantic python-dotenv datasets huggingface_hub pandas tqdm nibabel numpy
```

## Documentation

- [Serial Late Fusion Architecture](docs/Architecture/SERIAL_LATE_FUSION_ARCHITECTURE.md) — Full architecture deep dive
- [FHIR Generation Improvements](docs/Architecture/FHIR_GENERATION_IMPROVEMENTS.md) — Proposed FHIR pipeline enhancements
- [Triage System Deep Dive](docs/triage_system_technical_deep_dive.md) — Original single-model architecture (historical)

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
