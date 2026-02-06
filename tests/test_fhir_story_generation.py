#!/usr/bin/env python3
"""Test FHIR story generation pipeline.

This script processes all FHIR bundles in the data directory and displays
the generated clinical narratives to verify the extraction is working correctly.

Usage:
    python tests/test_fhir_story_generation.py

    # Or with verbose output:
    python tests/test_fhir_story_generation.py --verbose
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent / "sentinel_x"
sys.path.insert(0, str(PROJECT_ROOT))

from triage.fhir_janitor import FHIRJanitor


def print_section_header(text: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print()
    print(char * 80)
    print(f"  {text}")
    print(char * 80)


def print_subsection(text: str) -> None:
    """Print a formatted subsection header."""
    print()
    print(f"--- {text} ---")


def print_field(label: str, value, max_length: int = None) -> None:
    """Print a labeled field with optional truncation."""
    if isinstance(value, list):
        if value:
            print(f"{label}: {len(value)} items")
            for item in value:
                print(f"  • {item}")
        else:
            print(f"{label}: (none)")
    elif isinstance(value, str):
        if max_length and len(value) > max_length:
            print(f"{label}: {value[:max_length]}... (truncated, {len(value)} chars total)")
        else:
            print(f"{label}: {value if value else '(empty)'}")
    else:
        print(f"{label}: {value if value is not None else '(none)'}")


def find_fhir_bundles(data_dir: Path) -> list[Path]:
    """Find all fhir.json files in the data directory."""
    fhir_files = sorted(data_dir.glob("**/fhir.json"))
    return fhir_files


def process_fhir_bundle(fhir_path: Path, verbose: bool = False) -> dict:
    """Process a single FHIR bundle and return results."""
    # Load the FHIR bundle
    with open(fhir_path, "r") as f:
        fhir_bundle = json.load(f)

    # Process with FHIRJanitor
    janitor = FHIRJanitor()
    clinical_stream = janitor.process_bundle(fhir_bundle)

    # Analyze bundle structure if verbose
    bundle_stats = None
    if verbose:
        entries = fhir_bundle.get("entry", [])
        resource_types = {}
        for entry in entries:
            resource = entry.get("resource", {})
            rt = resource.get("resourceType", "Unknown")
            resource_types[rt] = resource_types.get(rt, 0) + 1
        bundle_stats = {
            "total_entries": len(entries),
            "resource_types": resource_types
        }

    return {
        "path": fhir_path,
        "clinical_stream": clinical_stream,
        "bundle_stats": bundle_stats
    }


def save_story_to_file(result: dict, output_dir: Path, verbose: bool = False) -> Path:
    """Save the clinical story to a text file.

    Args:
        result: Processing result dictionary
        output_dir: Directory to save story files
        verbose: Include verbose statistics

    Returns:
        Path to the saved story file
    """
    path = result["path"]
    stream = result["clinical_stream"]
    stats = result["bundle_stats"]
    patient_id = path.parent.name

    # Create output file
    output_file = output_dir / f"{patient_id}_story.txt"

    with open(output_file, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"  PATIENT: {patient_id}\n")
        f.write("=" * 80 + "\n\n")

        # Bundle statistics (verbose mode)
        if verbose and stats:
            f.write("--- Bundle Statistics ---\n")
            f.write(f"Total resources: {stats['total_entries']}\n")
            f.write("Resource breakdown:\n")
            for rt, count in sorted(stats["resource_types"].items(), key=lambda x: -x[1]):
                f.write(f"  {rt:25s}: {count:3d}\n")
            f.write("\n")

        # Demographics
        f.write("--- Demographics ---\n")
        f.write(f"Summary: {stream.patient_summary}\n")
        f.write(f"Age: {stream.age if stream.age is not None else '(none)'}\n")
        f.write(f"Gender: {stream.gender if stream.gender else '(none)'}\n\n")

        # Conditions
        f.write("--- Conditions & Risk Factors ---\n")
        if stream.conditions:
            f.write(f"All Conditions ({len(stream.conditions)} items):\n")
            for cond in stream.conditions:
                f.write(f"  • {cond}\n")
        else:
            f.write("All Conditions: (none)\n")

        if stream.risk_factors:
            f.write(f"\nRisk Factors ({len(stream.risk_factors)} items):\n")
            for rf in stream.risk_factors:
                f.write(f"  • {rf}\n")
        else:
            f.write("\nRisk Factors: (none)\n")
        f.write("\n")

        # Medications
        f.write("--- Medications ---\n")
        if stream.medications:
            f.write(f"All Medications ({len(stream.medications)} items):\n")
            for med in stream.medications:
                f.write(f"  • {med}\n")
        else:
            f.write("All Medications: (none)\n")

        if stream.active_medications:
            f.write(f"\nActive Medications ({len(stream.active_medications)} items):\n")
            for med in stream.active_medications:
                f.write(f"  • {med}\n")
        else:
            f.write("\nActive Medications: (none)\n")
        f.write("\n")

        # Report Content
        if stream.findings or stream.impressions:
            f.write("--- Report Content ---\n")
            if stream.findings:
                f.write(f"Findings ({len(stream.findings)} chars):\n")
                f.write(stream.findings + "\n\n")
            if stream.impressions:
                f.write(f"Impressions ({len(stream.impressions)} chars):\n")
                f.write(stream.impressions + "\n\n")

        # Warnings
        if stream.extraction_warnings:
            f.write("--- Extraction Warnings ---\n")
            for warning in stream.extraction_warnings:
                f.write(f"  ⚠️  {warning}\n")
            f.write("\n")

        # Full Clinical Narrative
        f.write("--- Complete Clinical Narrative ---\n\n")
        f.write(stream.narrative)
        f.write("\n\n")

        # Statistics
        f.write("--- Statistics ---\n")
        f.write(f"Token estimate: ~{stream.token_estimate:,} tokens\n")
        f.write(f"Narrative length: {len(stream.narrative):,} characters\n")
        f.write(f"Conditions found: {len(stream.conditions)}\n")
        f.write(f"Medications found: {len(stream.medications)}\n")
        f.write(f"Risk factors identified: {len(stream.risk_factors)}\n")

        # Source file
        f.write(f"\nSource: {path}\n")

    return output_file


def display_results(result: dict, verbose: bool = False) -> None:
    """Display the results of processing a FHIR bundle."""
    path = result["path"]
    stream = result["clinical_stream"]
    stats = result["bundle_stats"]

    # Header with patient ID
    patient_id = path.parent.name
    print_section_header(f"PATIENT: {patient_id}")

    # Bundle statistics (verbose mode)
    if verbose and stats:
        print_subsection("Bundle Statistics")
        print(f"Total resources: {stats['total_entries']}")
        print("Resource breakdown:")
        for rt, count in sorted(stats["resource_types"].items(), key=lambda x: -x[1]):
            print(f"  {rt:25s}: {count:3d}")

    # Demographics
    print_subsection("Demographics")
    print_field("Summary", stream.patient_summary)
    print_field("Age", stream.age)
    print_field("Gender", stream.gender)

    # Conditions
    print_subsection("Conditions & Risk Factors")
    print_field("All Conditions", stream.conditions)
    print_field("Risk Factors", stream.risk_factors)

    # Medications
    print_subsection("Medications")
    print_field("All Medications", stream.medications)
    print_field("Active Medications", stream.active_medications)

    # Report Content
    if stream.findings or stream.impressions:
        print_subsection("Report Content")
        if stream.findings:
            print_field("Findings", stream.findings, max_length=200)
        if stream.impressions:
            print_field("Impressions", stream.impressions, max_length=200)

    # Warnings
    if stream.extraction_warnings:
        print_subsection("Extraction Warnings")
        for warning in stream.extraction_warnings:
            print(f"  ⚠️  {warning}")

    # Full Clinical Narrative
    print_subsection("Complete Clinical Narrative")
    print()
    print(stream.narrative)
    print()

    # Statistics
    print_subsection("Statistics")
    print(f"Token estimate: ~{stream.token_estimate:,} tokens")
    print(f"Narrative length: {len(stream.narrative):,} characters")
    print(f"Conditions found: {len(stream.conditions)}")
    print(f"Medications found: {len(stream.medications)}")
    print(f"Risk factors identified: {len(stream.risk_factors)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test FHIR story generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_fhir_story_generation.py
  python tests/test_fhir_story_generation.py --verbose
  python tests/test_fhir_story_generation.py --patient train_1_a_1
        """
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed bundle statistics and resource counts"
    )
    parser.add_argument(
        "--patient", "-p",
        type=str,
        help="Process only a specific patient (e.g., 'train_1_a_1')"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "sentinel_x" / "data" / "raw_ct_rate" / "combined",
        help="Path to data directory containing FHIR bundles"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "stories",
        help="Directory to save story text files (default: tests/stories)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save stories to files, just display to console"
    )

    args = parser.parse_args()

    # Find FHIR bundles
    print_section_header("FHIR Story Generation Pipeline Test", "=")
    print(f"Searching for FHIR bundles in: {args.data_dir}")

    fhir_files = find_fhir_bundles(args.data_dir)

    if not fhir_files:
        print("❌ No FHIR bundles found!")
        return 1

    # Filter by patient if specified
    if args.patient:
        fhir_files = [f for f in fhir_files if args.patient in str(f)]
        if not fhir_files:
            print(f"❌ No FHIR bundle found for patient: {args.patient}")
            return 1

    print(f"Found {len(fhir_files)} FHIR bundle(s) to process")

    # Process each bundle
    results = []
    errors = []

    for fhir_path in fhir_files:
        patient_id = fhir_path.parent.name
        print(f"\nProcessing: {patient_id}...", end=" ")

        try:
            result = process_fhir_bundle(fhir_path, verbose=args.verbose)
            results.append(result)
            print("✓")
        except Exception as e:
            error_msg = f"Failed to process {patient_id}: {e}"
            errors.append(error_msg)
            print(f"❌\n  Error: {e}")

    # Save and/or display results
    print()
    saved_files = []

    if not args.no_save:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving stories to: {args.output_dir}\n")

        for result in results:
            patient_id = result["path"].parent.name
            output_file = save_story_to_file(result, args.output_dir, verbose=args.verbose)
            saved_files.append(output_file)
            print(f"  ✓ Saved: {output_file.name}")
    else:
        # Display to console
        for result in results:
            display_results(result, verbose=args.verbose)

    # Summary
    print()
    print_section_header("Summary", "=")
    print(f"✓ Successfully processed: {len(results)}/{len(fhir_files)} bundles")

    if saved_files:
        print(f"✓ Story files saved: {len(saved_files)}")
        print(f"  Location: {args.output_dir}")
        print(f"\nGenerated files:")
        for file in saved_files:
            size_kb = file.stat().st_size / 1024
            print(f"  • {file.name} ({size_kb:.1f} KB)")

    if errors:
        print(f"\n❌ Errors encountered:")
        for error in errors:
            print(f"  • {error}")
        return 1

    print("\n✅ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
