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
    """Extract axial slice with 3-channel windowing matching MedGemma 1.5 training.

    R = Wide window   (-1024 to 1024 HU) — full range, air to bone
    G = Soft tissue    (-135 to 215 HU)  — fat to start of bone
    B = Brain window   (0 to 80 HU)      — water to brain density

    Args:
        volume_hu: 3D volume in raw Hounsfield Units (NOT pre-windowed)
        slice_idx: Index of the axial slice to extract

    Returns:
        PIL Image in RGB format with multi-window encoding
    """
    raw_slice = volume_hu[:, :, slice_idx]
    raw_slice = np.rot90(raw_slice)

    window_clips = [CT_WINDOW_WIDE, CT_WINDOW_SOFT, CT_WINDOW_BRAIN]
    channels = [norm(raw_slice, clip[0], clip[1]) for clip in window_clips]
    rgb = np.round(np.stack(channels, axis=-1), 0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def load_nifti_volume(path: Path) -> Tuple[np.ndarray, dict]:
    """Load a NIfTI volume and return data with metadata.

    Args:
        path: Path to .nii or .nii.gz file

    Returns:
        Tuple of (volume data in raw HU, metadata dict)
    """
    logger.info(f"Loading NIfTI volume: {path}")
    img = nib.load(str(path))
    data = img.get_fdata()
    metadata = {
        "shape": data.shape,
        "affine": img.affine.tolist(),
        "header": dict(img.header),
    }
    logger.info(f"Loaded volume with shape: {data.shape}")
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

    # Sample slice indices from the raw volume
    slice_indices = sample_slices(volume_hu, CT_NUM_SLICES)

    # Extract images with multi-window encoding
    images = []
    for idx in slice_indices:
        img = extract_slice_as_multiwindow_image(volume_hu, idx)
        images.append(img)

    logger.info(f"Extracted {len(images)} multi-window slices from volume")
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
