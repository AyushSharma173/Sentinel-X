"""Unit tests for age calculation utilities."""

import pytest
from datetime import datetime

from sentinel_x.triage.age_utils import (
    calculate_age,
    extract_age_from_patient_resource,
)


class TestCalculateAge:
    """Test cases for calculate_age function."""

    def test_age_before_birthday(self):
        """Test age calculation when birthday hasn't occurred yet this year.

        Born Aug 4, 1942, tested on Feb 2, 2026 → age should be 83
        (Birthday hasn't occurred yet in 2026)
        """
        age = calculate_age("1942-08-04", reference_date=datetime(2026, 2, 2))
        assert age == 83

    def test_age_after_birthday(self):
        """Test age calculation when birthday has already passed this year.

        Born Aug 4, 1942, tested on Sep 1, 2026 → age should be 84
        (Birthday has passed in 2026)
        """
        age = calculate_age("1942-08-04", reference_date=datetime(2026, 9, 1))
        assert age == 84

    def test_age_on_birthday(self):
        """Test age calculation on the exact birthday.

        Born Aug 4, 1942, tested on Aug 4, 2026 → age should be 84
        """
        age = calculate_age("1942-08-04", reference_date=datetime(2026, 8, 4))
        assert age == 84

    def test_deceased_patient_after_birthday(self):
        """Test age calculation for deceased patient (died after birthday).

        Born Aug 4, 1942, died Sep 28, 2008 → age should be 66
        (Birthday had occurred in 2008 before death)
        """
        age = calculate_age("1942-08-04", deceased_date="2008-09-28T06:00:40+00:00")
        assert age == 66

    def test_deceased_patient_before_birthday(self):
        """Test age calculation for deceased patient (died before birthday).

        Born Aug 4, 1942, died Jul 15, 2008 → age should be 65
        (Birthday had not yet occurred in 2008)
        """
        age = calculate_age("1942-08-04", deceased_date="2008-07-15")
        assert age == 65

    def test_deceased_on_birthday(self):
        """Test age calculation for patient who died on their birthday.

        Born Aug 4, 1942, died Aug 4, 2008 → age should be 66
        """
        age = calculate_age("1942-08-04", deceased_date="2008-08-04")
        assert age == 66

    def test_datetime_format_with_timezone(self):
        """Test handling of full ISO datetime format with timezone."""
        age = calculate_age(
            "1942-08-04T00:00:00-05:00",
            deceased_date="2008-09-28T06:00:40+00:00"
        )
        assert age == 66

    def test_year_only_birthdate(self):
        """Test fallback to year-only calculation when month/day missing."""
        age = calculate_age("1942", reference_date=datetime(2026, 2, 2))
        # Should calculate as: 2026 - 1942 = 84
        # Month/day comparison (2, 2) vs (1, 1) means birthday has passed
        assert age == 84

    def test_year_month_only_birthdate(self):
        """Test year-month birthdate with default day."""
        age = calculate_age("1942-08", reference_date=datetime(2026, 2, 2))
        # Birthday is August (month 8), reference is February (month 2)
        # Birthday hasn't occurred yet
        assert age == 83

    def test_empty_birthdate(self):
        """Test handling of empty birthdate string."""
        age = calculate_age("")
        assert age is None

    def test_none_birthdate(self):
        """Test handling of None birthdate."""
        age = calculate_age(None)
        assert age is None

    def test_invalid_date_format(self):
        """Test handling of invalid date format."""
        age = calculate_age("not-a-date")
        assert age is None

    def test_future_birthdate(self):
        """Test handling of future birthdate (should give negative age)."""
        age = calculate_age("2030-01-01", reference_date=datetime(2026, 2, 2))
        # Function returns None for negative ages
        assert age is None

    def test_very_old_age(self):
        """Test calculation for very old age (150+ years)."""
        age = calculate_age("1800-01-01", reference_date=datetime(2026, 2, 2))
        # Should calculate but log warning
        assert age == 226

    def test_default_reference_date(self):
        """Test that default reference date is current date."""
        # Can't assert exact value since it depends on when test runs
        # Just verify it returns a reasonable value
        age = calculate_age("1980-01-01")
        assert age is not None
        assert 40 <= age <= 50  # Reasonable range for 1980 birth

    def test_leap_year_birthday(self):
        """Test age calculation for leap year birthday."""
        # Born Feb 29, 2000
        age = calculate_age("2000-02-29", reference_date=datetime(2026, 3, 1))
        assert age == 26  # Birthday has passed

        age = calculate_age("2000-02-29", reference_date=datetime(2026, 2, 28))
        assert age == 25  # Birthday hasn't occurred yet


class TestExtractAgeFromPatientResource:
    """Test cases for extract_age_from_patient_resource function."""

    def test_living_patient_before_birthday(self):
        """Test extraction for living patient before birthday."""
        patient = {
            "resourceType": "Patient",
            "id": "test-123",
            "birthDate": "1942-08-04",
            "gender": "male"
        }
        age, is_deceased, method = extract_age_from_patient_resource(patient)

        # Age depends on current date, but should not be deceased
        assert age is not None
        assert is_deceased is False
        assert method == "birthDate"

    def test_deceased_patient_with_datetime(self):
        """Test extraction for deceased patient with deceasedDateTime."""
        patient = {
            "resourceType": "Patient",
            "id": "train_1_a_2",
            "birthDate": "1942-08-04",
            "deceasedDateTime": "2008-09-28T06:00:40+00:00",
            "gender": "male"
        }
        age, is_deceased, method = extract_age_from_patient_resource(patient)

        assert age == 66
        assert is_deceased is True
        assert method == "birthDate_at_death"

    def test_deceased_patient_with_boolean(self):
        """Test extraction for deceased patient with deceasedBoolean only."""
        patient = {
            "resourceType": "Patient",
            "id": "test-456",
            "birthDate": "1950-05-15",
            "deceasedBoolean": True,
            "gender": "female"
        }
        age, is_deceased, method = extract_age_from_patient_resource(patient)

        # Should calculate current age since no death date
        assert age is not None
        assert is_deceased is True
        assert method == "birthDate"

    def test_patient_without_birthdate(self):
        """Test extraction when birthDate is missing."""
        patient = {
            "resourceType": "Patient",
            "id": "test-789",
            "gender": "male"
        }
        age, is_deceased, method = extract_age_from_patient_resource(patient)

        assert age is None
        assert is_deceased is False
        assert method == "unavailable"

    def test_patient_with_age_extension_integer(self):
        """Test extraction from FHIR extension with valueInteger."""
        patient = {
            "resourceType": "Patient",
            "id": "test-ext",
            "gender": "male",
            "extension": [
                {
                    "url": "http://example.org/fhir/StructureDefinition/patient-age",
                    "valueInteger": 75
                }
            ]
        }
        age, is_deceased, method = extract_age_from_patient_resource(patient)

        assert age == 75
        assert is_deceased is False
        assert method == "extension_valueInteger"

    def test_patient_with_age_extension_quantity(self):
        """Test extraction from FHIR extension with valueQuantity."""
        patient = {
            "resourceType": "Patient",
            "id": "test-ext-qty",
            "gender": "female",
            "extension": [
                {
                    "url": "http://hl7.org/fhir/StructureDefinition/patient-age",
                    "valueQuantity": {
                        "value": 82,
                        "unit": "years",
                        "code": "a"
                    }
                }
            ]
        }
        age, is_deceased, method = extract_age_from_patient_resource(patient)

        assert age == 82
        assert is_deceased is False
        assert method == "extension_valueQuantity"

    def test_birthdate_priority_over_extension(self):
        """Test that birthDate takes priority over extension."""
        patient = {
            "resourceType": "Patient",
            "id": "test-priority",
            "birthDate": "1942-08-04",
            "deceasedDateTime": "2008-09-28T06:00:40+00:00",
            "extension": [
                {
                    "url": "http://example.org/fhir/StructureDefinition/patient-age",
                    "valueInteger": 999  # Should be ignored
                }
            ]
        }
        age, is_deceased, method = extract_age_from_patient_resource(patient)

        # Should use birthDate calculation, not extension
        assert age == 66
        assert is_deceased is True
        assert method == "birthDate_at_death"

    def test_empty_patient_resource(self):
        """Test handling of empty patient resource."""
        age, is_deceased, method = extract_age_from_patient_resource({})

        assert age is None
        assert is_deceased is False
        assert method == "missing_resource"

    def test_none_patient_resource(self):
        """Test handling of None patient resource."""
        age, is_deceased, method = extract_age_from_patient_resource(None)

        assert age is None
        assert is_deceased is False
        assert method == "missing_resource"

    def test_invalid_birthdate_with_extension_fallback(self):
        """Test fallback to extension when birthDate is invalid."""
        patient = {
            "resourceType": "Patient",
            "id": "test-fallback",
            "birthDate": "invalid-date",
            "extension": [
                {
                    "url": "http://example.org/fhir/StructureDefinition/patient-age",
                    "valueInteger": 68
                }
            ]
        }
        age, is_deceased, method = extract_age_from_patient_resource(patient)

        assert age == 68
        assert is_deceased is False
        assert method == "extension_valueInteger"

    def test_deceased_date_without_birthdate(self):
        """Test deceased patient without birthDate (can't calculate age at death)."""
        patient = {
            "resourceType": "Patient",
            "id": "test-no-birth",
            "deceasedDateTime": "2008-09-28T06:00:40+00:00",
            "gender": "male"
        }
        age, is_deceased, method = extract_age_from_patient_resource(patient)

        assert age is None
        assert is_deceased is True
        assert method == "unavailable"


class TestRealWorldScenarios:
    """Test cases based on real patient data from Sentinel-X."""

    def test_train_1_a_2_patient(self):
        """Test the actual train_1_a_2 patient that had the bug.

        This patient:
        - Born: 1942-08-04
        - Died: 2008-09-28 (should be age 66 at death)
        - Was incorrectly showing as 84yo (current age)
        """
        patient = {
            "resourceType": "Patient",
            "id": "train_1_a_2",
            "birthDate": "1942-08-04",
            "deceasedDateTime": "2008-09-28T06:00:40+00:00",
            "gender": "male",
            "address": [
                {
                    "line": ["123 Test St"],
                    "city": "Boston",
                    "state": "MA",
                    "postalCode": "02134"
                }
            ]
        }

        age, is_deceased, method = extract_age_from_patient_resource(patient)

        # Critical assertions
        assert age == 66, f"Expected age 66 at death, got {age}"
        assert is_deceased is True, "Patient should be marked as deceased"
        assert method == "birthDate_at_death", f"Expected birthDate_at_death method, got {method}"

    def test_february_2026_reference_date(self):
        """Test with the actual date when the bug was discovered (Feb 2, 2026)."""
        # Living patient born Aug 4, 1942
        age = calculate_age("1942-08-04", reference_date=datetime(2026, 2, 2))

        # Birthday hasn't occurred yet in 2026
        assert age == 83, f"Expected age 83 in Feb 2026, got {age}"

        # Same patient if still alive in Sep 2026
        age = calculate_age("1942-08-04", reference_date=datetime(2026, 9, 1))
        assert age == 84, f"Expected age 84 in Sep 2026, got {age}"
