#!/usr/bin/env python3
"""Automated cleanup service for Sentinel-X.

Removes temporary files, caches, and old data to prevent disk exhaustion.
Can be run standalone (CLI) or imported as a module.

Usage:
    # Run all cleanup tasks
    python -m sentinel_x.utils.cleanup_service

    # Dry run (show what would be cleaned, don't delete)
    python -m sentinel_x.utils.cleanup_service --dry-run

    # Only clean specific targets
    python -m sentinel_x.utils.cleanup_service --pycache --pip-cache
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve project root
_THIS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = _THIS_DIR.parent.parent  # /workspace/Sentinel-X


class CleanupService:
    """Manages cleanup of temporary files, caches, and old data."""

    def __init__(self, project_root: Path = PROJECT_ROOT, dry_run: bool = False):
        self.project_root = project_root
        self.sentinel_x = project_root / "sentinel_x"
        self.dry_run = dry_run
        self._freed_bytes = 0

    def _get_size(self, path: Path) -> int:
        """Get total size of a file or directory in bytes."""
        if path.is_file():
            return path.stat().st_size
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    def _format_size(self, size_bytes: int) -> str:
        """Format byte count as human-readable string."""
        for unit in ("B", "KB", "MB", "GB"):
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"

    def _remove(self, path: Path, label: str) -> int:
        """Remove a file or directory and return bytes freed."""
        if not path.exists():
            return 0
        size = self._get_size(path)
        if self.dry_run:
            logger.info(f"[DRY RUN] Would remove {label}: {path} ({self._format_size(size)})")
        else:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            logger.info(f"Removed {label}: {path} ({self._format_size(size)})")
        self._freed_bytes += size
        return size

    def cleanup_pycache(self) -> int:
        """Remove all __pycache__ directories and .pyc files."""
        freed = 0
        for pycache in self.project_root.rglob("__pycache__"):
            freed += self._remove(pycache, "__pycache__")
        for pyc in self.project_root.rglob("*.pyc"):
            freed += self._remove(pyc, ".pyc file")
        return freed

    def cleanup_pip_cache(self) -> int:
        """Clear pip download cache."""
        pip_cache = Path.home() / ".cache" / "pip"
        return self._remove(pip_cache, "pip cache")

    def cleanup_hf_download_cache(self) -> int:
        """Remove HuggingFace download cache inside data directories.

        This is the .cache/ dir created by hf_hub_download inside the
        output directory — NOT the main HF model cache.
        """
        freed = 0
        data_dir = self.sentinel_x / "data"
        if data_dir.exists():
            for cache_dir in data_dir.rglob(".cache"):
                freed += self._remove(cache_dir, "HF download cache")

        # Also check /runpod-volume path
        runpod_data = Path("/runpod-volume/sentinel_x_data")
        if runpod_data.exists():
            for cache_dir in runpod_data.rglob(".cache"):
                freed += self._remove(cache_dir, "HF download cache (runpod)")
        return freed

    def cleanup_old_session_logs(self, max_age_days: int = 7) -> int:
        """Remove session log files older than max_age_days."""
        freed = 0
        cutoff = time.time() - (max_age_days * 86400)

        for log_dir in [
            self.sentinel_x / "logs" / "sessions",
            Path("/runpod-volume/sentinel_x_data/logs/sessions"),
        ]:
            if not log_dir.exists():
                continue
            for log_file in log_dir.iterdir():
                if log_file.is_file() and log_file.stat().st_mtime < cutoff:
                    freed += self._remove(log_file, f"old session log ({max_age_days}d+)")
        return freed

    def cleanup_synthea_temp(self) -> int:
        """Remove Synthea temporary output directory."""
        freed = 0
        for temp_dir in [
            self.sentinel_x / "data" / ".synthea_temp",
            Path("/runpod-volume/sentinel_x_data/.synthea_temp"),
        ]:
            freed += self._remove(temp_dir, "Synthea temp")
        return freed

    def cleanup_all(self) -> int:
        """Run all cleanup tasks. Returns total bytes freed."""
        self._freed_bytes = 0
        logger.info("=" * 50)
        logger.info("Sentinel-X Cleanup Service")
        logger.info("=" * 50)

        self.cleanup_pycache()
        self.cleanup_pip_cache()
        self.cleanup_hf_download_cache()
        self.cleanup_old_session_logs()
        self.cleanup_synthea_temp()

        logger.info("-" * 50)
        logger.info(f"Total freed: {self._format_size(self._freed_bytes)}")

        # Report disk status after cleanup
        from .disk_monitor import log_disk_status
        log_disk_status("post-cleanup")

        return self._freed_bytes


def main():
    parser = argparse.ArgumentParser(description="Sentinel-X Cleanup Service")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--pycache", action="store_true", help="Clean __pycache__ directories")
    parser.add_argument("--pip-cache", action="store_true", help="Clean pip cache")
    parser.add_argument("--hf-cache", action="store_true", help="Clean HF download caches in data dirs")
    parser.add_argument("--logs", action="store_true", help="Clean old session logs")
    parser.add_argument("--synthea", action="store_true", help="Clean Synthea temp files")
    parser.add_argument("--log-age-days", type=int, default=7, help="Max age for session logs (default: 7)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    service = CleanupService(dry_run=args.dry_run)

    # If no specific target selected, run all
    run_all = not any([args.pycache, args.pip_cache, args.hf_cache, args.logs, args.synthea])

    if run_all:
        service.cleanup_all()
    else:
        if args.pycache:
            service.cleanup_pycache()
        if args.pip_cache:
            service.cleanup_pip_cache()
        if args.hf_cache:
            service.cleanup_hf_download_cache()
        if args.logs:
            service.cleanup_old_session_logs(max_age_days=args.log_age_days)
        if args.synthea:
            service.cleanup_synthea_temp()

        logger.info(f"Total freed: {service._format_size(service._freed_bytes)}")


if __name__ == "__main__":
    main()
