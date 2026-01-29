"""CT scan processing: NIfTI loading, windowing, and slice extraction."""

import logging
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

from .config import CT_NUM_SLICES, CT_WINDOW_CENTER, CT_WINDOW_WIDTH

logger = logging.getLogger(__name__)


def apply_window(data: np.ndarray, center: int, width: int) -> np.ndarray:
    """Apply HU windowing to CT data.

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


def load_nifti_volume(path: Path) -> Tuple[np.ndarray, dict]:
    """Load a NIfTI volume and return data with metadata.

    Args:
        path: Path to .nii or .nii.gz file

    Returns:
        Tuple of (volume data, metadata dict)
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


def extract_slice_as_image(volume: np.ndarray, slice_idx: int) -> Image.Image:
    """Extract a single axial slice as a PIL Image.

    Args:
        volume: 3D windowed volume (values in 0-255)
        slice_idx: Index of the slice to extract

    Returns:
        PIL Image in RGB format
    """
    slice_data = volume[:, :, slice_idx]

    # Rotate for proper orientation (may need adjustment based on data)
    slice_data = np.rot90(slice_data)

    # Convert to RGB
    rgb_slice = np.stack([slice_data] * 3, axis=-1)

    return Image.fromarray(rgb_slice.astype(np.uint8), mode="RGB")


def process_ct_volume(path: Path) -> Tuple[List[Image.Image], List[int], dict]:
    """Full CT processing pipeline.

    Args:
        path: Path to NIfTI file

    Returns:
        Tuple of (list of PIL images, slice indices, metadata)
    """
    # Load volume
    volume, metadata = load_nifti_volume(path)

    # Apply soft tissue windowing
    windowed = apply_window(volume, CT_WINDOW_CENTER, CT_WINDOW_WIDTH)

    # Sample slice indices
    slice_indices = sample_slices(windowed, CT_NUM_SLICES)

    # Extract images
    images = []
    for idx in slice_indices:
        img = extract_slice_as_image(windowed, idx)
        images.append(img)

    logger.info(f"Extracted {len(images)} slices from volume")

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
