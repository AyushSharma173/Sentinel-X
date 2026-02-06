#!/usr/bin/env python3
"""Command-line tool to analyze Sentinel-X log sessions.

Usage:
    # Summary report for a session
    python -m sentinel_x.scripts.analyze_logs logs/sessions/<session-id>

    # Specific patient trace
    python -m sentinel_x.scripts.analyze_logs logs/sessions/<session-id> -p CT-001

    # Find all failures
    python -m sentinel_x.scripts.analyze_logs logs/sessions/<session-id> --failures

    # Analyze latest session
    python -m sentinel_x.scripts.analyze_logs --latest
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import LogAnalyzer directly to avoid transformer dependencies
# We load the module by spec to avoid triggering the triage __init__.py
import importlib.util

def _load_module(name: str, path: str):
    """Load a module by file path without triggering parent imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get the base path
_base_path = Path(__file__).parent.parent

# Load config directly
_config = _load_module('config', str(_base_path / 'triage' / 'config.py'))
LOG_DIR = _config.LOG_DIR

# Load log_analyzer directly
_log_analyzer = _load_module('log_analyzer', str(_base_path / 'triage' / 'logging' / 'log_analyzer.py'))
LogAnalyzer = _log_analyzer.LogAnalyzer
find_latest_session = _log_analyzer.find_latest_session


def print_json(data: any, indent: int = 2) -> None:
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=indent, default=str))


def main():
    """Main entry point for the log analyzer CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze Sentinel-X log sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "session_path",
        nargs="?",
        type=str,
        help="Path to session directory (e.g., logs/sessions/2025-06-15_10-30-45)",
    )

    parser.add_argument(
        "--latest",
        action="store_true",
        help="Analyze the most recent session",
    )

    parser.add_argument(
        "-p", "--patient",
        type=str,
        help="Show summary for a specific patient ID",
    )

    parser.add_argument(
        "--failures",
        action="store_true",
        help="Show all failure/error events",
    )

    parser.add_argument(
        "--events",
        type=str,
        nargs="+",
        metavar="EVENT_TYPE",
        help="Filter events by type (e.g., CONDITIONS_SUMMARY MEDICATIONS_SUMMARY)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()

    # Determine session path
    session_path: Optional[Path] = None

    if args.latest:
        session_path = find_latest_session(LOG_DIR)
        if session_path is None:
            print("Error: No sessions found in", LOG_DIR / "sessions")
            sys.exit(1)
        print(f"Using latest session: {session_path.name}")
    elif args.session_path:
        session_path = Path(args.session_path)
    else:
        parser.print_help()
        sys.exit(1)

    # Validate session path
    if not session_path.exists():
        print(f"Error: Session directory not found: {session_path}")
        sys.exit(1)

    # Create analyzer
    analyzer = LogAnalyzer(session_path)

    # Handle different output modes
    if args.patient:
        # Show patient summary
        summary = analyzer.get_patient_summary(args.patient)
        if not summary.conditions and not summary.medications and not summary.errors:
            print(f"No data found for patient: {args.patient}")
            sys.exit(1)

        if args.json:
            print_json({
                "patient_id": summary.patient_id,
                "conditions": summary.conditions,
                "medications": summary.medications,
                "errors": summary.errors,
                "duration_ms": summary.duration_ms,
            })
        else:
            analyzer.print_patient_summary(args.patient)

    elif args.failures:
        # Show all failures
        failures = analyzer.get_failures()
        if args.json:
            print_json(failures)
        else:
            print(f"\nFailures found: {len(failures)}")
            for i, failure in enumerate(failures):
                print(f"\n--- Failure {i+1} ---")
                patient = failure.get("patient_id", "unknown")
                event_type = failure.get("event_type", "ERROR")
                error = failure.get("error", failure.get("message", "Unknown"))
                timestamp = failure.get("timestamp", "")
                print(f"Patient: {patient}")
                print(f"Type: {event_type}")
                print(f"Time: {timestamp}")
                print(f"Error: {error}")

    elif args.events:
        # Filter by event types
        events = analyzer.filter_by_event_type(args.events)
        if args.json:
            print_json(events)
        else:
            print(f"\nEvents matching {args.events}: {len(events)}")
            for event in events:
                patient = event.get("patient_id", "")
                event_type = event.get("event_type", "")
                timestamp = event.get("timestamp", "")
                message = event.get("message", "")

                prefix = f"[{timestamp}]" if timestamp else ""
                if patient:
                    prefix += f" [{patient}]"

                print(f"{prefix} {event_type}: {message}")

                if args.verbose:
                    # Show additional fields
                    for key in ["tool_name", "tool_args", "tool_result",
                               "duration_ms", "char_count", "error"]:
                        if key in event:
                            print(f"    {key}: {event[key]}")

    else:
        # Default: show session summary
        if args.json:
            summary = analyzer.generate_summary_report()
            print_json({
                "session_id": summary.session_id,
                "start_time": summary.start_time.isoformat() if summary.start_time else None,
                "end_time": summary.end_time.isoformat() if summary.end_time else None,
                "patient_count": summary.patient_count,
                "error_count": len(summary.errors),
                "patients": {
                    pid: {
                        "conditions": p.conditions,
                        "medications": p.medications,
                        "error_count": len(p.errors),
                    }
                    for pid, p in summary.patients.items()
                },
            })
        else:
            analyzer.print_session_summary()


if __name__ == "__main__":
    main()
