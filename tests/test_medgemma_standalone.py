#!/usr/bin/env python3
"""Standalone test: load MedGemma 4B with SMART SAMPLING.

IMPROVEMENTS:
1. Validates HU range (Fixes "White Circle" bug).
2. Uses "Smart Sampling" to prioritize lung/vascular regions.
3. Uses "Chain of Thought" prompt for deeper reasoning.
4. [NEW] VISUAL DEBUG: Saves a GIF of the input data.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
import numpy as np 

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def weighted_sample_slices(volume_shape, num_slices):
    """
    Smart Sampling Strategy:
    Prioritize middle 60% (lungs/heart).
    """
    total_slices = volume_shape[2]
    if total_slices <= num_slices:
        return list(range(total_slices))

    p1 = int(total_slices * 0.15)
    p2 = int(total_slices * 0.85)
    
    n_top = int(num_slices * 0.15)
    n_mid = int(num_slices * 0.70)
    n_bot = num_slices - n_top - n_mid 
    
    idx_top = np.linspace(0, p1, n_top, endpoint=False, dtype=int)
    idx_mid = np.linspace(p1, p2, n_mid, endpoint=False, dtype=int)
    idx_bot = np.linspace(p2, total_slices-1, n_bot, dtype=int)
    
    indices = np.concatenate([idx_top, idx_mid, idx_bot])
    indices = np.sort(np.unique(indices))
    
    return indices.tolist()


def main():
    parser = argparse.ArgumentParser(description="Standalone MedGemma 4B test on a CT volume")
    parser.add_argument("--patient", type=str, default=None,
                        help="Patient folder name (e.g. train_1_a_1). Random if omitted.")
    parser.add_argument("--num-slices", type=int, default=85, 
                        help="Number of slices to sample (default: 100)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max new tokens for generation")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt")
    args = parser.parse_args()

    # ---- Find a CT volume ----
    data_dir = PROJECT_ROOT / "sentinel_x" / "data" / "raw_ct_rate" / "combined"
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir() and (d / "volume.nii.gz").exists()]

    if not patient_dirs:
        print(f"ERROR: No patient folders with volume.nii.gz found in {data_dir}")
        sys.exit(1)

    if args.patient:
        chosen = data_dir / args.patient
        if not (chosen / "volume.nii.gz").exists():
            print(f"ERROR: {chosen / 'volume.nii.gz'} not found")
            sys.exit(1)
    else:
        chosen = random.choice(patient_dirs)

    volume_path = chosen / "volume.nii.gz"
    print(f"Patient: {chosen.name}")
    print(f"Volume:  {volume_path}")
    print()

    # ---- Load and process CT volume ----
    import nibabel as nib
    from sentinel_x.triage.ct_processor import (
        extract_slice_as_multiwindow_image,
        load_nifti_volume,
    )

    print("Loading NIfTI volume...")
    t0 = time.time()
    volume_hu, _ = load_nifti_volume(Path(volume_path))
    print(f"  Shape: {volume_hu.shape}  (loaded in {time.time() - t0:.1f}s)")

    # === [START FIX: HU CLAMPING] ===
    volume_hu[volume_hu < -1000] = -1000
    volume_hu[volume_hu > 3000] = 3000
    # === [END FIX] ===

    print(f"  HU range: [{volume_hu.min():.0f}, {volume_hu.max():.0f}]")

    # USE SMART SAMPLING
    slice_indices = weighted_sample_slices(volume_hu.shape, args.num_slices)
    print(f"  Sampling {len(slice_indices)} slices (Weighted Distribution)")

    images = [extract_slice_as_multiwindow_image(volume_hu, idx) for idx in slice_indices]
    print(f"  Extracted {len(images)} multi-window RGB images ({images[0].size})")

    # ==================================================================================
    # [NEW] VISUALIZATION BLOCK - SAVE DEBUG IMAGES
    # ==================================================================================
    debug_dir = Path("debug_input_viz")
    debug_dir.mkdir(exist_ok=True)
    print(f"\n[DEBUG] Saving visualization to: {debug_dir.absolute()}")

    # 1. Save animated GIF (Great for seeing the "flow" and orientation)
    print("  -> Generating volume_scan.gif...")
    images[0].save(
        debug_dir / "volume_scan.gif",
        save_all=True,
        append_images=images[1:],
        duration=50,  # 50ms per frame = 20fps
        loop=0
    )

    # 2. Save sample PNGs (Every 10th slice) for detailed inspection
    print("  -> Saving sample PNG slices...")
    for i, img in enumerate(images):
        if i % 10 == 0:
            img.save(debug_dir / f"slice_{i:03d}.png")
            
    print("[DEBUG] Visualization complete. Check the folder before trusting the model.\n")
    # ==================================================================================


    # ---- Load MedGemma 4B ----
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = "google/medgemma-1.5-4b-it" 
    print(f"Loading model: {model_id}")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print()

    # ---- Build "Interrogator" Prompt ----
    
    system_text = (
        "You are an expert radiologist analyzing a volumetric Chest CT. "
        "Your goal is to detect ALL abnormalities, no matter how subtle. "
        "Assume the scan contains pathology until proven otherwise."
    )

    if args.prompt:
            user_text = args.prompt
    else:
        user_text = args.prompt or (
            f"Please review these {len(images)} axial chest CT slices (ordered Top to Bottom).\n\n"
            "Perform a systematic 'Visual Inventory' by objectively describing the appearance of these anatomical regions. "
            "Do not assume pathology; report only what is visible.\n\n"
            "1. VASCULAR SYSTEM: SYSTEMATICALLY TRACE the course of the aorta, pulmonary arteries, and subclavian veins. Describe their caliber, patency, and any contrast defects or collateral vessels.\n"
            "2. AIRWAYS: EXAMINE the trachea and bronchial tree. Describe the bronchial wall thickness and lumen patency. Are there any endoluminal obstructions?\n"
            "3. LUNGS (PARENCHYMA): SCAN both lungs from apex to base. Characterize any opacities (ground glass, consolidation, nodules, reticulation) and their distribution. Describe the pleura.\n"
            "4. UPPER ABDOMEN: ASSESS the visible liver, spleen, adrenals, and kidneys. Describe organ size, contour, and homogeneity.\n"
            "5. BONES & SOFT TISSUE: REVIEW the thoracic cage, spine, and chest wall soft tissues. Note any fractures, lesions, or degenerative changes.\n\n"
            "6. Give a FINAL IMPRESSION."
        )

    content = []
    content.append({"type": "text", "text": system_text})
    for i, image in enumerate(images, 1):
        content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": f" SLICE {i}"}) 
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]

    print("=" * 70)
    print(f"SYSTEM PROMPT: {system_text}")
    print("-" * 30)
    print(f"USER PROMPT:\n{user_text}")
    print("=" * 70)
    print()

    # ---- Run inference ----
    print("Running inference...")
    t0 = time.time()

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    print(f"  Input tokens: {input_len}")
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False, # Deterministic for testing
            repetition_penalty=1.05,
        )

    generated_tokens = outputs[0][input_len:]
    response = processor.decode(generated_tokens, skip_special_tokens=True)
    duration = time.time() - t0

    print(f"  Generated {len(generated_tokens)} tokens in {duration:.1f}s "
          f"({len(generated_tokens) / duration:.1f} tok/s)")
    print()

    # ---- Print output ----
    print("=" * 70)
    print("MODEL OUTPUT")
    print("=" * 70)
    print(response)
    print("=" * 70)


if __name__ == "__main__":
    main()