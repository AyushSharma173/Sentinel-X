"""GPU VRAM management utilities for serial model swapping.

Provides centralized functions to unload models, clear GPU memory,
and monitor VRAM usage — enabling the 4B→27B serial swap within
24GB VRAM budget.
"""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def log_vram_status(label: str) -> None:
    """Log current VRAM allocated/reserved/free."""
    if not torch.cuda.is_available():
        logger.info(f"[VRAM {label}] CUDA not available")
        return

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_mem / 1024**2
    free = total - allocated

    logger.info(
        f"[VRAM {label}] "
        f"Allocated: {allocated:.0f}MB | "
        f"Reserved: {reserved:.0f}MB | "
        f"Free: {free:.0f}MB | "
        f"Total: {total:.0f}MB"
    )


def get_vram_free_mb() -> float:
    """Return free VRAM in MB."""
    if not torch.cuda.is_available():
        return 0.0
    allocated = torch.cuda.memory_allocated() / 1024**2
    total = torch.cuda.get_device_properties(0).total_mem / 1024**2
    return total - allocated


def unload_model(model, processor) -> None:
    """Delete model/processor refs, run gc.collect(), empty CUDA cache.

    Steps:
    1. Move model to CPU first (prevents GPU memory fragmentation)
    2. Delete model and processor references
    3. gc.collect() to trigger Python garbage collection
    4. torch.cuda.empty_cache() to release CUDA memory pool
    5. torch.cuda.synchronize() to ensure all ops complete
    6. Log VRAM usage before/after
    """
    log_vram_status("before unload")

    # Move model to CPU to free GPU memory cleanly
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            # Quantized models may not support .to() — just delete
            pass
        del model

    if processor is not None:
        del processor

    # Force Python garbage collection
    gc.collect()

    # Release CUDA memory pool
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    log_vram_status("after unload")
