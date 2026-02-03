"""Age calculation utilities for FHIR Patient resources.

This module provides centralized age calculation logic with:
- Month/day precision (accounts for whether birthday has occurred)
- Deceased patient handling (age at death, not current age)
- Robust date parsing and error handling
- FHIR extension fallback support
"""

from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def calculate_age(
    birth_date: str,
    deceased_date: Optional[str] = None,
    reference_date: Optional[datetime] = None
) -> Optional[int]:
    """
    Calculate age with month/day precision.

    Algorithm:
        1. Parse birth_date to (birth_year, birth_month, birth_day)
        2. Use reference_date (or today if None) as (ref_year, ref_month, ref_day)
        3. If deceased_date provided, use that as reference instead
        4. Calculate: age = ref_year - birth_year
        5. If (ref_month, ref_day) < (birth_month, birth_day): age -= 1

    Args:
        birth_date: ISO 8601 date string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS...)
        deceased_date: Optional ISO 8601 datetime string
        reference_date: Optional reference date (defaults to today)

    Returns:
        Age in years, or None if calculation fails

    Examples:
        >>> calculate_age("1942-08-04", reference_date=datetime(2026, 2, 2))
        83  # Birthday hasn't occurred yet in 2026
        >>> calculate_age("1942-08-04", reference_date=datetime(2026, 9, 1))
        84  # Birthday has passed
        >>> calculate_age("1942-08-04", deceased_date="2008-09-28T06:00:40+00:00")
        66  # Age at death
    """
    if not birth_date:
        logger.warning("calculate_age called with empty birth_date")
        return None

    try:
        # Parse birth date
        # Handle both date-only (YYYY-MM-DD) and datetime formats
        birth_date_str = birth_date.split("T")[0]  # Strip time component if present
        birth_parts = birth_date_str.split("-")

        if len(birth_parts) < 1:
            logger.warning(f"Invalid birth_date format: {birth_date}")
            return None

        birth_year = int(birth_parts[0])
        birth_month = int(birth_parts[1]) if len(birth_parts) >= 2 else 1
        birth_day = int(birth_parts[2]) if len(birth_parts) >= 3 else 1

        # Determine reference date
        if deceased_date:
            # Use death date as reference
            deceased_date_str = deceased_date.split("T")[0]
            ref_parts = deceased_date_str.split("-")
            ref_year = int(ref_parts[0])
            ref_month = int(ref_parts[1]) if len(ref_parts) >= 2 else 1
            ref_day = int(ref_parts[2]) if len(ref_parts) >= 3 else 1
        elif reference_date:
            ref_year = reference_date.year
            ref_month = reference_date.month
            ref_day = reference_date.day
        else:
            now = datetime.now()
            ref_year = now.year
            ref_month = now.month
            ref_day = now.day

        # Calculate age with month/day precision
        age = ref_year - birth_year

        # Subtract 1 if birthday hasn't occurred yet this year
        if (ref_month, ref_day) < (birth_month, birth_day):
            age -= 1

        # Sanity checks
        if age < 0:
            logger.warning(
                f"Negative age calculated: birth={birth_date}, "
                f"reference={deceased_date or reference_date or 'now'}"
            )
            return None

        if age > 150:
            logger.warning(f"Suspiciously high age: {age} years")

        return age

    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to calculate age from birth_date={birth_date}: {e}")
        return None


def extract_age_from_patient_resource(
    patient_resource: Dict[str, Any]
) -> Tuple[Optional[int], bool, str]:
    """
    Extract age from FHIR Patient resource.

    Handles:
    - birthDate with month/day precision
    - deceasedDateTime (calculates age at death)
    - deceasedBoolean flag
    - FHIR extensions with age values
    - Invalid/missing dates

    Args:
        patient_resource: FHIR Patient resource (dict)

    Returns:
        Tuple of (age, is_deceased, calculation_method):
            - age: Calculated age in years, or None if unavailable
            - is_deceased: True if patient is deceased
            - calculation_method: String describing how age was calculated
                ("birthDate", "birthDate_at_death", "extension", etc.)

    Examples:
        >>> patient = {
        ...     "birthDate": "1942-08-04",
        ...     "deceasedDateTime": "2008-09-28T06:00:40+00:00"
        ... }
        >>> age, is_deceased, method = extract_age_from_patient_resource(patient)
        >>> (age, is_deceased, method)
        (66, True, 'birthDate_at_death')
    """
    if not patient_resource:
        logger.warning("extract_age_from_patient_resource called with empty resource")
        return None, False, "missing_resource"

    # Check deceased status
    deceased_datetime = patient_resource.get("deceasedDateTime")
    deceased_boolean = patient_resource.get("deceasedBoolean", False)
    is_deceased = bool(deceased_datetime or deceased_boolean)

    # Try to calculate from birthDate
    birth_date = patient_resource.get("birthDate")

    if birth_date:
        if deceased_datetime:
            # Calculate age at death
            age = calculate_age(birth_date, deceased_date=deceased_datetime)
            method = "birthDate_at_death"
        else:
            # Calculate current age
            age = calculate_age(birth_date)
            method = "birthDate"

        if age is not None:
            return age, is_deceased, method
        else:
            logger.warning(
                f"Failed to calculate age from birthDate={birth_date}, "
                f"deceasedDateTime={deceased_datetime}"
            )

    # Fallback: Try FHIR extensions
    extensions = patient_resource.get("extension", [])
    for ext in extensions:
        if "age" in ext.get("url", "").lower():
            # Try to extract age value from extension
            if "valueInteger" in ext:
                age = ext["valueInteger"]
                return age, is_deceased, "extension_valueInteger"
            elif "valueQuantity" in ext:
                value_qty = ext["valueQuantity"]
                if value_qty.get("unit") == "years" or value_qty.get("code") == "a":
                    age = int(value_qty.get("value", 0))
                    return age, is_deceased, "extension_valueQuantity"

    # No age could be determined
    logger.info(f"No age could be extracted from Patient resource (id={patient_resource.get('id')})")
    return None, is_deceased, "unavailable"
