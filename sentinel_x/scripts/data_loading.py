import os
import re
import json
import shutil
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_ID = "ibrahimhamamci/CT-RATE"
NUM_SAMPLES = 50  # How many patients to download

# Use relative paths based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent  # sentinel_x/

# Default to /runpod-volume when available (root disk is small).
# Override with SENTINEL_DATA_DIR env var.
_RUNPOD_DATA = Path("/runpod-volume/sentinel_x_data/raw_ct_rate")
_DEFAULT_OUTPUT = _RUNPOD_DATA if _RUNPOD_DATA.parent.exists() else PROJECT_DIR / "data" / "raw_ct_rate"
OUTPUT_DIR = Path(os.getenv("SENTINEL_DATA_DIR", str(_DEFAULT_OUTPUT)))

# Minimum free disk space (GB) required before starting downloads
MIN_FREE_DISK_GB = float(os.getenv("SENTINEL_MIN_FREE_GB", "2"))

def check_disk_space(path: Path, min_free_gb: float = MIN_FREE_DISK_GB) -> None:
    """Raise if insufficient disk space at the target path."""
    target = path if path.exists() else path.parent
    stat = shutil.disk_usage(str(target))
    free_gb = stat.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient disk space at {target}: {free_gb:.1f}GB free, "
            f"need at least {min_free_gb}GB. Free up space or set "
            f"SENTINEL_DATA_DIR to a volume with more room."
        )
    print(f"   Disk check OK: {free_gb:.1f}GB free at {target}")


def setup_directories():
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_disk_space(OUTPUT_DIR)
    print(f"Output directory: {OUTPUT_DIR}")

def parse_volume_name(volume_name: str) -> dict:
    """Parse volume name to extract IDs."""
    # Pattern: {split}_{patient_id}_{scan_id}_{reconstruction_id}.nii.gz
    pattern = r"^(train|valid)_(\d+)_([a-z])_(\d+)\.nii\.gz$"
    match = re.match(pattern, volume_name)
    if match:
        return {
            "split": match.group(1),
            "patient_id": match.group(2),
            "scan_id": match.group(3),
            "reconstruction_id": match.group(4),
            "scan_unique_id": f"{match.group(1)}_{match.group(2)}_{match.group(3)}" # e.g. train_1_a
        }
    return None

def get_patient_id(volume_name: str) -> str | None:
    """Extract patient-level ID from a volume name.

    Examples:
        train_1_a_1.nii.gz → train_1
        train_2_a_2.nii.gz → train_2
    """
    parsed = parse_volume_name(volume_name)
    if not parsed:
        return None
    return f"{parsed['split']}_{parsed['patient_id']}"


def get_volume_repo_path(volume_name: str) -> str:
    """Construct the HF repo path for the file."""
    parsed = parse_volume_name(volume_name)
    if not parsed: return None

    # Path: dataset/train/train_1/train_1_a/train_1_a_1.nii.gz
    folder = f"{parsed['split']}_{parsed['patient_id']}_{parsed['scan_id']}"
    return f"dataset/{parsed['split']}/{parsed['split']}_{parsed['patient_id']}/{folder}/{volume_name}"

def load_metadata(num_samples=NUM_SAMPLES):
    """
    Loads BOTH Labels and Reports and merges them by VolumeName.

    Reports dataset columns:
    - VolumeName: matches the .nii.gz filename
    - ClinicalInformation_EN: clinical context
    - Technique_EN: scanning technique
    - Findings_EN: radiologist findings
    - Impressions_EN: radiologist impressions
    """
    print(f"\nFetching metadata for top {num_samples} samples...")

    # 1. Load Labels (Contains Volume Names and abnormality labels)
    ds_labels = load_dataset(DATASET_ID, name="labels", split=f"train[:{num_samples}]")
    df_labels = ds_labels.to_pandas()

    # 2. Load Reports (Contains radiology text)
    print("   Fetching radiology reports...")
    ds_reports = load_dataset(DATASET_ID, name="reports", split="train", streaming=True)

    # Get the volume names we need reports for
    target_volumes = set(df_labels['VolumeName'].tolist())

    # Stream through reports and collect matching ones
    matching_reports = []
    for report in ds_reports:
        if report['VolumeName'] in target_volumes:
            matching_reports.append(report)
            if len(matching_reports) >= num_samples:
                break

    df_reports = pd.DataFrame(matching_reports)

    print(f"   Labels columns: {df_labels.columns.tolist()}")
    print(f"   Reports columns: {df_reports.columns.tolist()}")

    # 3. Merge on VolumeName
    merged_df = pd.merge(df_labels, df_reports, on="VolumeName", how="left")

    # Check merge success
    reports_found = merged_df['Findings_EN'].notna().sum()
    print(f"   Matched {reports_found}/{len(merged_df)} reports to volumes")

    return merged_df

def download_volume_and_report(row, output_dir: Path):
    """Downloads the NIfTI and saves the Report Text side-by-side."""
    vol_name = row['VolumeName']

    # Create subdirectories
    volumes_dir = output_dir / "volumes"
    reports_dir = output_dir / "reports"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download Image
    repo_path = get_volume_repo_path(vol_name)
    if not repo_path:
        return

    out_path = volumes_dir / vol_name
    if not out_path.exists():
        try:
            # Pre-flight: check disk space before each download
            check_disk_space(output_dir, min_free_gb=1.0)

            hf_hub_download(
                repo_id=DATASET_ID,
                filename=repo_path,
                repo_type="dataset",
                local_dir=output_dir
            )
            # Fix nested folder structure from HF download
            nested_path = output_dir / repo_path
            if nested_path.exists() and nested_path != out_path:
                shutil.move(str(nested_path), str(out_path))
                shutil.rmtree(output_dir / "dataset", ignore_errors=True)

            # Clean up HF download cache to reclaim space
            hf_cache = output_dir / ".cache"
            if hf_cache.exists():
                shutil.rmtree(hf_cache, ignore_errors=True)

        except Exception as e:
            print(f"Failed to download {vol_name}: {e}")
            return

    # 2. Save Report as structured JSON
    report_data = {
        "volume_name": vol_name,
        "clinical_information": row.get('ClinicalInformation_EN', ''),
        "technique": row.get('Technique_EN', ''),
        "findings": row.get('Findings_EN', ''),
        "impressions": row.get('Impressions_EN', ''),
    }

    # Check if we have actual report content
    has_report = bool(report_data['findings'] or report_data['impressions'])

    json_path = reports_dir / vol_name.replace(".nii.gz", ".json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Also save a plain text version for easy reading
    txt_path = reports_dir / vol_name.replace(".nii.gz", ".txt")
    with open(txt_path, "w") as f:
        f.write(f"Volume: {vol_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"CLINICAL INFORMATION:\n{report_data['clinical_information'] or 'N/A'}\n\n")
        f.write(f"TECHNIQUE:\n{report_data['technique'] or 'N/A'}\n\n")
        f.write(f"FINDINGS:\n{report_data['findings'] or 'N/A'}\n\n")
        f.write(f"IMPRESSIONS:\n{report_data['impressions'] or 'N/A'}\n")

    status = "OK" if has_report else "WARN"
    print(f"[{status}] Saved: {vol_name} + Report")

def main():
    print("SENTINEL-X DATA LOADER")
    setup_directories()

    # 1. Get the combined data
    df = load_metadata(NUM_SAMPLES)
    print(f"   Prepared {len(df)} samples.")

    # 2. Download loop
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        download_volume_and_report(row, OUTPUT_DIR)

    print(f"\nDone! Check {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
