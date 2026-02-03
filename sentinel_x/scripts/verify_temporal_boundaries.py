#!/usr/bin/env python3
"""
Verify Temporal Boundaries in FHIR Bundles

This script analyzes generated FHIR bundles to identify temporal boundary violations.
It checks if any Synthea-generated resources have dates after the scan acquisition date.

Usage:
    python sentinel_x/scripts/verify_temporal_boundaries.py
    python sentinel_x/scripts/verify_temporal_boundaries.py --data-dir /path/to/combined
    python sentinel_x/scripts/verify_temporal_boundaries.py --bundle train_1_a_1
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "raw_ct_rate" / "combined"

# Temporal fields to check per resource type
TEMPORAL_FIELDS_MAP = {
    "Condition": ["onsetDateTime", "abatementDateTime", "recordedDate"],
    "Encounter": ["period.start", "period.end"],
    "Observation": ["effectiveDateTime", "issued"],
    "MedicationRequest": ["authoredOn"],
    "Procedure": ["performedDateTime", "performedPeriod.start", "performedPeriod.end"],
    "DiagnosticReport": ["effectiveDateTime", "issued"],
    "ImagingStudy": ["started"],
    "Immunization": ["occurrenceDateTime"],
    "AllergyIntolerance": ["recordedDate"],
    "CarePlan": ["period.start", "period.end"],
    "Claim": ["created"],
    "ExplanationOfBenefit": ["created"],
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_scan_date_from_bundle(bundle: dict, volume_name: Optional[str] = None) -> Optional[str]:
    """
    Find the scan date from the manually created ImagingStudy resource.

    This is the ImagingStudy with description matching "CT Chest - {volume_name}".
    """
    for entry in bundle.get('entry', []):
        resource = entry.get('resource', {})
        if resource.get('resourceType') == 'ImagingStudy':
            desc = resource.get('description', '')
            # Check if this is our manually created ImagingStudy
            if volume_name and desc == f"CT Chest - {volume_name}":
                return resource.get('started')
            # Fallback: if no volume_name, look for "CT Chest - " pattern
            elif desc.startswith("CT Chest - ") and desc.endswith(".nii.gz"):
                return resource.get('started')
    return None


def extract_temporal_value(resource: dict, field_path: str) -> Optional[str]:
    """
    Extract a temporal value from a resource given a field path (e.g., 'period.start').
    """
    parts = field_path.split('.')
    value = resource
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value if isinstance(value, str) else None


def get_all_temporal_values(resource: dict) -> list[tuple[str, str]]:
    """
    Extract all temporal values from a resource.

    Returns:
        List of (field_name, datetime_value) tuples
    """
    resource_type = resource.get('resourceType')
    temporal_fields = TEMPORAL_FIELDS_MAP.get(resource_type, [])

    values = []
    for field in temporal_fields:
        value = extract_temporal_value(resource, field)
        if value:
            values.append((field, value))

    return values


def is_manually_created_resource(resource: dict, scan_date: str, volume_name: Optional[str] = None) -> bool:
    """
    Check if a resource was manually created (not from Synthea).

    Logic:
    - ImagingStudy with description "CT Chest - {volume_name}" is manually created
    - DiagnosticReport/Condition with temporal field exactly matching scan_date is likely manually created
    """
    resource_type = resource.get('resourceType')

    # Check ImagingStudy by description
    if resource_type == 'ImagingStudy':
        desc = resource.get('description', '')
        if volume_name and desc == f"CT Chest - {volume_name}":
            return True
        elif desc.startswith("CT Chest - ") and desc.endswith(".nii.gz"):
            return True

    # Check if temporal fields match scan_date exactly
    if resource_type in ['DiagnosticReport', 'Condition']:
        temporal_values = get_all_temporal_values(resource)
        for _, value in temporal_values:
            if value == scan_date:
                return True

    return False


def analyze_bundle(bundle_path: Path) -> dict:
    """
    Analyze a FHIR bundle for temporal violations.

    Returns:
        Dictionary with analysis results
    """
    with open(bundle_path, 'r') as f:
        bundle = json.load(f)

    # Extract volume name from bundle path (folder name)
    volume_name = bundle_path.parent.name + ".nii.gz"

    # Find scan date
    scan_date = get_scan_date_from_bundle(bundle, volume_name)

    if not scan_date:
        return {
            "error": "Could not find scan date in bundle",
            "bundle_path": str(bundle_path)
        }

    # Analyze resources
    total_resources = 0
    manually_created = 0
    synthea_resources = 0
    future_violations = 0
    violation_details = []
    resource_type_counts = defaultdict(int)
    violation_by_type = defaultdict(int)

    earliest_date = None
    latest_date = None

    for entry in bundle.get('entry', []):
        resource = entry.get('resource', {})
        resource_type = resource.get('resourceType')
        resource_id = resource.get('id', 'unknown')

        total_resources += 1
        resource_type_counts[resource_type] += 1

        # Skip non-temporal resources
        if resource_type in ['Patient', 'Practitioner', 'Organization', 'Location']:
            continue

        # Check if manually created
        is_manual = is_manually_created_resource(resource, scan_date, volume_name)
        if is_manual:
            manually_created += 1
            continue

        synthea_resources += 1

        # Check all temporal fields
        temporal_values = get_all_temporal_values(resource)
        has_violation = False

        for field_name, date_value in temporal_values:
            # Track earliest and latest dates
            if earliest_date is None or date_value < earliest_date:
                earliest_date = date_value
            if latest_date is None or date_value > latest_date:
                latest_date = date_value

            # Check for future violation
            if date_value > scan_date:
                has_violation = True
                violation_details.append({
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "field": field_name,
                    "date": date_value,
                    "scan_date": scan_date
                })

        if has_violation:
            future_violations += 1
            violation_by_type[resource_type] += 1

    return {
        "bundle_path": str(bundle_path),
        "bundle_name": bundle_path.parent.name,
        "scan_date": scan_date,
        "total_resources": total_resources,
        "manually_created_resources": manually_created,
        "synthea_resources": synthea_resources,
        "future_violations": future_violations,
        "violation_percentage": (future_violations / synthea_resources * 100) if synthea_resources > 0 else 0,
        "earliest_date": earliest_date,
        "latest_date": latest_date,
        "resource_type_counts": dict(resource_type_counts),
        "violation_by_type": dict(violation_by_type),
        "violation_details": violation_details[:10]  # Limit to first 10 for readability
    }


def print_summary_table(results: list[dict]):
    """Print a summary table of all analyzed bundles."""
    print("\n" + "=" * 120)
    print("TEMPORAL BOUNDARY VERIFICATION SUMMARY")
    print("=" * 120)
    print()

    # Header
    header = f"{'Bundle':<20} {'Scan Date':<20} {'Total':<8} {'Manual':<8} {'Synthea':<8} {'Violations':<12} {'%':<8}"
    print(header)
    print("-" * 120)

    # Rows
    total_synthea = 0
    total_violations = 0

    for result in results:
        if "error" in result:
            print(f"{result['bundle_name']:<20} ERROR: {result['error']}")
            continue

        bundle_name = result['bundle_name']
        scan_date = result['scan_date'][:10]  # Just the date part
        total = result['total_resources']
        manual = result['manually_created_resources']
        synthea = result['synthea_resources']
        violations = result['future_violations']
        pct = result['violation_percentage']

        total_synthea += synthea
        total_violations += violations

        row = f"{bundle_name:<20} {scan_date:<20} {total:<8} {manual:<8} {synthea:<8} {violations:<12} {pct:<8.1f}"
        print(row)

    print("-" * 120)
    print(f"{'TOTAL':<20} {'':<20} {'':<8} {'':<8} {total_synthea:<8} {total_violations:<12} {(total_violations/total_synthea*100 if total_synthea > 0 else 0):<8.1f}")
    print("=" * 120)


def print_detailed_analysis(result: dict):
    """Print detailed analysis for a single bundle."""
    print("\n" + "=" * 80)
    print(f"DETAILED ANALYSIS: {result['bundle_name']}")
    print("=" * 80)
    print()

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"Bundle Path: {result['bundle_path']}")
    print(f"Scan Date: {result['scan_date']}")
    print()

    print("Resource Summary:")
    print(f"  Total Resources: {result['total_resources']}")
    print(f"  Manually Created: {result['manually_created_resources']}")
    print(f"  Synthea Resources: {result['synthea_resources']}")
    print(f"  Future Violations: {result['future_violations']} ({result['violation_percentage']:.1f}%)")
    print()

    print("Date Range:")
    print(f"  Earliest: {result['earliest_date']}")
    print(f"  Latest: {result['latest_date']}")
    print(f"  Scan: {result['scan_date']}")
    print()

    print("Resource Type Counts:")
    for rt, count in sorted(result['resource_type_counts'].items()):
        violations = result['violation_by_type'].get(rt, 0)
        if violations > 0:
            print(f"  {rt:<30} {count:>5}  (⚠️  {violations} violations)")
        else:
            print(f"  {rt:<30} {count:>5}")
    print()

    if result['violation_details']:
        print("Sample Violations (first 10):")
        for v in result['violation_details']:
            print(f"  ❌ {v['resource_type']:<20} {v['field']:<25} {v['date']} (scan: {v['scan_date']})")
    print()


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Verify temporal boundaries in FHIR bundles"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Directory containing bundle folders (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--bundle",
        type=str,
        help="Analyze a single bundle (folder name, e.g., train_1_a_1)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis for each bundle"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Find bundles to analyze
    if args.bundle:
        bundle_path = data_dir / args.bundle / "fhir.json"
        if not bundle_path.exists():
            print(f"Error: Bundle not found: {bundle_path}")
            sys.exit(1)
        bundle_paths = [bundle_path]
    else:
        bundle_paths = sorted(data_dir.glob("*/fhir.json"))

    if not bundle_paths:
        print("Error: No FHIR bundles found")
        sys.exit(1)

    print(f"Analyzing {len(bundle_paths)} bundle(s)...")

    # Analyze bundles
    results = []
    for bundle_path in bundle_paths:
        result = analyze_bundle(bundle_path)
        results.append(result)
        if args.detailed:
            print_detailed_analysis(result)

    # Print summary
    print_summary_table(results)

    # Overall assessment
    total_violations = sum(r.get('future_violations', 0) for r in results)
    if total_violations > 0:
        print("\n⚠️  TEMPORAL BOUNDARY VIOLATIONS DETECTED")
        print(f"Found {total_violations} resources with dates after the scan date.")
        print("These resources represent 'future' data relative to the scan and should be filtered.")
    else:
        print("\n✅ NO TEMPORAL BOUNDARY VIOLATIONS")
        print("All Synthea-generated resources have dates on or before the scan date.")


if __name__ == "__main__":
    main()
