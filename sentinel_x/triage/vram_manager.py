"""GPU VRAM management utilities for serial model swapping.

Provides centralized functions to unload models, clear GPU memory,
monitor VRAM usage, and run pre-flight safety checks — enabling the
4B→27B serial swap within 24GB VRAM budget without OOM freezes.
"""

import gc
import logging

import torch

logger = logging.getLogger(__name__)

# Minimum free VRAM (MB) required before loading a model.
# If free VRAM is below this after cleanup, we abort rather than OOM.
VRAM_MIN_FREE_PHASE1_MB = 10_000   # 4B BF16 needs ~8GB + overhead
VRAM_MIN_FREE_PHASE2_MB = 15_000   # 27B NF4 needs ~13.5GB + overhead


def log_vram_status(label: str) -> None:
    """Log current VRAM allocated/reserved/free."""
    if not torch.cuda.is_available():
        logger.info(f"[VRAM {label}] CUDA not available")
        return

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
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
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return total - allocated


def get_vram_total_mb() -> float:
    """Return total VRAM in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**2


def assert_vram_available(required_mb: float, phase_label: str) -> None:
    """Pre-flight check: abort early if not enough VRAM is free.

    Call this BEFORE loading a model. Raises RuntimeError with a clear
    message instead of letting PyTorch OOM-kill or freeze the GPU.
    """
    if not torch.cuda.is_available():
        logger.warning(f"[{phase_label}] CUDA not available — skipping VRAM check")
        return

    free = get_vram_free_mb()
    if free < required_mb:
        msg = (
            f"[{phase_label}] INSUFFICIENT VRAM: {free:.0f}MB free, "
            f"need {required_mb:.0f}MB. "
            f"Something is still holding GPU memory. "
            f"Try stopping other GPU processes or reducing batch size."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(
        f"[{phase_label}] VRAM pre-flight OK: {free:.0f}MB free >= {required_mb:.0f}MB required"
    )


def verify_clean_state(label: str, warn_threshold_mb: float = 500) -> bool:
    """Check if VRAM is clean after unload. Warn if allocated > threshold.

    Returns True if clean, False if leaked memory detected.
    """
    if not torch.cuda.is_available():
        return True

    allocated = torch.cuda.memory_allocated() / 1024**2
    if allocated > warn_threshold_mb:
        logger.warning(
            f"[VRAM LEAK {label}] {allocated:.0f}MB still allocated after cleanup "
            f"(threshold: {warn_threshold_mb:.0f}MB) — VRAM leak detected"
        )
        return False

    logger.info(f"[VRAM {label}] Clean state verified: {allocated:.0f}MB allocated")
    return True


def unload_model(model, processor) -> None:
    """Delete model/processor refs, run gc.collect(), empty CUDA cache.

    Steps:
    1. Move model to CPU first (prevents GPU memory fragmentation)
    2. Delete model and processor references
    3. Two rounds of gc.collect() + empty_cache for stubborn BnB models
    4. Reset peak memory stats to help the allocator
    5. Verify clean state and warn on leaks
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

    # First pass: standard cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Second pass: catch stragglers from BnB quantized models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    log_vram_status("after unload")
    verify_clean_state("post-unload")
