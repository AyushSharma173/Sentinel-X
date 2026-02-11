"""Disk space monitoring utilities for Sentinel-X.

Provides pre-flight checks and continuous monitoring to prevent
'No space left on device' failures during CT downloads, FHIR generation,
and triage processing.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default mount points to monitor on RunPod
DEFAULT_MOUNTS = ["/", "/runpod-volume"]

# Thresholds
CRITICAL_PERCENT = 95
WARNING_PERCENT = 85
DEFAULT_MIN_FREE_GB = 2.0


def get_disk_usage(path: str = "/") -> dict:
    """Return disk usage stats for the given path."""
    stat = shutil.disk_usage(path)
    return {
        "path": path,
        "total_gb": stat.total / (1024 ** 3),
        "used_gb": stat.used / (1024 ** 3),
        "free_gb": stat.free / (1024 ** 3),
        "percent_used": (stat.used / stat.total) * 100,
    }


def check_disk_space(
    path: str = "/",
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    raise_on_fail: bool = True,
) -> bool:
    """Check if enough free disk space is available.

    Args:
        path: Filesystem path to check.
        min_free_gb: Minimum free space in GB required.
        raise_on_fail: If True, raise RuntimeError on insufficient space.

    Returns:
        True if sufficient space, False otherwise.
    """
    usage = get_disk_usage(path)
    free = usage["free_gb"]
    pct = usage["percent_used"]

    if free < min_free_gb:
        msg = (
            f"Insufficient disk space at {path}: {free:.1f}GB free "
            f"({pct:.0f}% used), need at least {min_free_gb}GB. "
            f"Run cleanup or set SENTINEL_DATA_ROOT to a larger volume."
        )
        logger.error(msg)
        if raise_on_fail:
            raise RuntimeError(msg)
        return False

    if pct > WARNING_PERCENT:
        logger.warning(
            f"Disk {path} at {pct:.0f}% ({free:.1f}GB free) — approaching capacity"
        )
    else:
        logger.debug(f"Disk {path} OK: {free:.1f}GB free ({pct:.0f}% used)")

    return True


def log_disk_status(label: str = "", mounts: Optional[list] = None) -> None:
    """Log disk usage for all monitored mount points."""
    if mounts is None:
        mounts = [m for m in DEFAULT_MOUNTS if Path(m).exists()]

    for mount in mounts:
        usage = get_disk_usage(mount)
        pct = usage["percent_used"]
        free = usage["free_gb"]
        total = usage["total_gb"]
        prefix = f"[{label}] " if label else ""

        if pct >= CRITICAL_PERCENT:
            logger.critical(f"{prefix}{mount}: {pct:.0f}% used — {free:.1f}GB free / {total:.0f}GB")
        elif pct >= WARNING_PERCENT:
            logger.warning(f"{prefix}{mount}: {pct:.0f}% used — {free:.1f}GB free / {total:.0f}GB")
        else:
            logger.info(f"{prefix}{mount}: {pct:.0f}% used — {free:.1f}GB free / {total:.0f}GB")


def estimate_required_space_gb(num_patients: int, avg_volume_mb: float = 250.0) -> float:
    """Estimate disk space needed for a batch of patients.

    Each patient typically has 2 reconstructions (~250MB each avg) plus
    reports (~10KB) and FHIR bundles (~500KB).
    """
    volume_gb = (num_patients * 2 * avg_volume_mb) / 1024
    overhead_gb = 0.5  # FHIR + reports + metadata
    return volume_gb + overhead_gb
