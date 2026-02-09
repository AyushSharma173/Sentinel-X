"""CT scan processing: NIfTI loading, windowing, and slice extraction.

Supports both legacy single-window and the new 3-channel multi-window
encoding that matches MedGemma 1.5's training distribution:
  R = Wide window  (-1024 to 1024 HU)
  G = Soft tissue  (-135 to 215 HU)
  B = Brain window  (0 to 80 HU)
"""

import logging
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

from .config import (
    CT_NUM_SLICES,
    CT_WINDOW_BRAIN,
    CT_WINDOW_CENTER,
    CT_WINDOW_SOFT,
    CT_WINDOW_WIDE,
    CT_WINDOW_WIDTH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy single-window (kept for backward compatibility)
# ---------------------------------------------------------------------------

def apply_window(data: np.ndarray, center: int, width: int) -> np.ndarray:
    """Apply HU windowing to CT data (legacy single-window).

    Args:
        data: CT volume in Hounsfield Units
        center: Window center in HU
        width: Window width in HU

    Returns:
        Normalized array with values in [0, 255]
    """
    lower = center - width // 2
    upper = center + width // 2
    windowed = np.clip(data, lower, upper)
    normalized = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    return normalized


def extract_slice_as_image(volume: np.ndarray, slice_idx: int) -> Image.Image:
    """Extract a single axial slice as a PIL Image (legacy single-window).

    Args:
        volume: 3D windowed volume (values in 0-255)
        slice_idx: Index of the slice to extract

    Returns:
        PIL Image in RGB format (grayscale duplicated to 3 channels)
    """
    slice_data = volume[:, :, slice_idx]
    slice_data = np.rot90(slice_data)
    slice_data = np.fliplr(slice_data)  # Standard radiology: patient R on viewer L
    rgb_slice = np.stack([slice_data] * 3, axis=-1)
    return Image.fromarray(rgb_slice.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Multi-window 3-channel (matches MedGemma 1.5 CT training distribution)
# ---------------------------------------------------------------------------

def norm(ct_slice: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Window and normalize CT HU values to 0-255.

    Matches Google's exact implementation from the official MedGemma 1.5
    CT notebook (high_dimensional_ct_hugging_face.ipynb).

    Args:
        ct_slice: 2D array of raw HU values
        hu_min: Lower bound of the HU window
        hu_max: Upper bound of the HU window

    Returns:
        Normalized array with values in [0, 255] as float32
    """
    clipped = np.clip(ct_slice, hu_min, hu_max).astype(np.float32)
    clipped -= hu_min
    clipped /= (hu_max - hu_min)
    clipped *= 255.0
    return clipped


def extract_slice_as_multiwindow_image(
    volume_hu: np.ndarray, slice_idx: int
) -> Image.Image:
    """Extract axial slice with CHEST-optimized windowing.
    
    R = Lung Window     (W:1500 L:-600)  -> Critical for pneumonia/atelectasis
    G = Soft Tissue     (W:400  L:50)    -> Standard mediastinal/vascular view
    B = Bone Window     (W:2000 L:300)   -> For spine/rib abnormalities
    """
    raw_slice = volume_hu[:, :, slice_idx]
    raw_slice = np.rot90(raw_slice)
    raw_slice = np.fliplr(raw_slice)  # Standard radiology: patient R on viewer L

    # Define normalization function
    def normalize(data, center, width):
        lower = center - (width / 2)
        upper = center + (width / 2)
        # Clip and scale to 0-255
        img = np.clip(data, lower, upper)
        img = (img - lower) / (upper - lower) * 255
        return img

    # 4. Use CHEST windows (Not Brain!)
    # Red: Lung Window (Best for the infection/atelectasis you missed)
    r_ch = normalize(raw_slice, center=-600, width=1500)
    
    # Green: Soft Tissue (Best for the heart/vessels)
    g_ch = normalize(raw_slice, center=50, width=400)
    
    # Blue: Bone Window (Best for the osteophytes)
    b_ch = normalize(raw_slice, center=300, width=2000)

    # Stack and return
    rgb = np.stack([r_ch, g_ch, b_ch], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")

# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------
def load_nifti_volume(path: Path) -> Tuple[np.ndarray, dict]:
    """Load a NIfTI volume and return data in Hounsfield Units.

    CT-RATE NIfTI files store raw DICOM pixel values — the RescaleSlope
    and RescaleIntercept were NOT baked in during DICOM→NPZ→NIfTI conversion
    (the NIfTI headers have scl_slope/scl_inter = NaN).

    We apply: HU = stored_value * RescaleSlope + RescaleIntercept
    The per-volume values come from the CT-RATE metadata config on HuggingFace.
    All current volumes use Slope=1, Intercept=-8192 (Siemens extended CT scale).

    Args:
        path: Path to .nii or .nii.gz file

    Returns:
        Tuple of (volume data in Hounsfield Units, metadata dict)
    """
    logger.info(f"Loading NIfTI volume: {path}")
    img = nib.load(str(path))
    data = img.get_fdata()

    # Apply DICOM RescaleIntercept to convert stored pixel values → HU.
    # CT-RATE metadata: RescaleSlope=1, RescaleIntercept=-8192 for all
    # current volumes (Siemens SOMATOM go.All extended CT scale).
    slope = float(img.header["scl_slope"]) if not np.isnan(img.header["scl_slope"]) else 1.0
    intercept = float(img.header["scl_inter"]) if not np.isnan(img.header["scl_inter"]) else 0.0

    if slope == 1.0 and intercept == 0.0:
        # NIfTI header has no scaling info — apply CT-RATE default
        rescale_intercept = -8192
        data = data + rescale_intercept
        logger.info(
            f"Applied CT-RATE RescaleIntercept={rescale_intercept} "
            f"(NIfTI header had no scaling)"
        )
    else:
        data = data * slope + intercept
        logger.info(f"Applied NIfTI header scaling: slope={slope}, intercept={intercept}")

    # === [CRITICAL FIX] ===
    # Clamp Hounsfield Units to standard medical range.
    # 1. Fix "White Circle Blindness": The -8192 fill value destroys contrast 
    #    normalization if not clamped to Air (-1000).
    data[data < -1000] = -1000
    
    # 2. Fix "Metal Artifacts": High-density objects (pacemakers, bullets, etc.) 
    #    can spike to +10,000 HU, compressing the lung window range.
    data[data > 3000] = 3000
    # ======================

    metadata = {
        "shape": data.shape,
        "affine": img.affine.tolist(),
        "header": dict(img.header),
    }
    logger.info(f"Loaded volume with shape: {data.shape}, HU range: [{data.min():.0f}, {data.max():.0f}]")
    return data, metadata
    


def sample_slices(volume: np.ndarray, num_slices: int = CT_NUM_SLICES) -> List[int]:
    """Get uniformly distributed slice indices from a volume.

    Args:
        volume: 3D numpy array
        num_slices: Number of slices to sample

    Returns:
        List of slice indices
    """
    total_slices = volume.shape[2]
    if total_slices <= num_slices:
        return list(range(total_slices))
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    return indices.tolist()


def process_ct_volume(path: Path) -> Tuple[List[Image.Image], List[int], dict]:
    """Full CT processing pipeline using 3-channel multi-window encoding.

    Loads the NIfTI volume in raw HU, samples slices, and converts each
    to a 3-channel RGB image using Wide/Soft/Brain windowing that matches
    the MedGemma 1.5 training distribution.

    Args:
        path: Path to NIfTI file

    Returns:
        Tuple of (list of PIL images, slice indices, metadata)
    """
    # Load volume in raw HU (not pre-windowed)
    volume_hu, metadata = load_nifti_volume(path)

    if volume_hu.min() > -1000:
        logger.warning(
            "HU RANGE SUSPICIOUS: min value is %.0f but air should be ~ -1000. "
            "Data may be missing the -8192 offset fix — model will likely see "
            "a solid white image and fail to diagnose.",
            volume_hu.min(),
        )

    # Sample slice indices from the raw volume
    slice_indices = sample_slices(volume_hu, CT_NUM_SLICES)
    logger.info(
        "Sampling %d slices from volume: [%d...%d]",
        len(slice_indices), slice_indices[0], slice_indices[-1],
    )

    # Extract images with multi-window encoding
    images = []
    for idx in slice_indices:
        img = extract_slice_as_multiwindow_image(volume_hu, idx)
        images.append(img)

    logger.info(
        "Extracted %d multi-window slices (%s) from volume",
        len(images), f"{images[0].size[0]}x{images[0].size[1]}" if images else "none",
    )
    return images, slice_indices, metadata


def get_thumbnail(image: Image.Image, size: Tuple[int, int] = (128, 128)) -> Image.Image:
    """Create a thumbnail from an image.

    Args:
        image: PIL Image
        size: Thumbnail dimensions

    Returns:
        Resized thumbnail image
    """
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    return thumb
