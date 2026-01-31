"""FHIR query tools for the ReAct triage agent.

These tools provide structured access to FHIR Bundle data, allowing the agent
to dynamically investigate clinical context based on imaging findings.

Each tool has extensive docstrings that serve as the model's instruction manual.
The agent uses these docstrings to understand WHEN and HOW to use each tool.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from .config import TOOL_LAB_LOOKBACK_DAYS

logger = logging.getLogger(__name__)


# Tool registry for dynamic lookup
TOOL_REGISTRY: Dict[str, Callable] = {}


def tool(func: Callable) -> Callable:
    """Decorator to register a function as an agent tool.

    The decorated function's docstring becomes its instruction manual
    for the agent. Tools are registered globally for lookup.
    """
    TOOL_REGISTRY[func.__name__] = func
    return func


def get_tool(name: str) -> Optional[Callable]:
    """Get a tool by name from the registry."""
    return TOOL_REGISTRY.get(name)


def get_all_tools() -> Dict[str, Callable]:
    """Get all registered tools."""
    return TOOL_REGISTRY.copy()


def get_tool_descriptions() -> str:
    """Generate tool descriptions for the system prompt.

    Extracts docstrings from all registered tools and formats them
    for inclusion in the agent's system prompt.

    Returns:
        Formatted string describing all available tools
    """
    descriptions = []
    for name, func in TOOL_REGISTRY.items():
        doc = func.__doc__ or "No description available."
        descriptions.append(f"### {name}\n{doc}")
    return "\n\n".join(descriptions)


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================


@tool
def get_patient_manifest(fhir_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Get overview of available clinical data for this patient.

    WHEN TO USE: Call this FIRST to understand what data exists before
    making specific queries. This helps you plan your investigation.

    Args:
        fhir_bundle: The FHIR Bundle containing patient data

    Returns:
        {
            "patient_id": str,
            "demographics": {"age": int or null, "gender": str or null},
            "resource_counts": {"Condition": int, "MedicationRequest": int, ...},
            "available_lab_categories": ["Cardiac", "Coag", "Renal", ...]
        }
    """
    result = {
        "patient_id": None,
        "demographics": {"age": None, "gender": None},
        "resource_counts": {},
        "available_lab_categories": [],
    }

    entries = fhir_bundle.get("entry", [])

    # Count resources by type
    for entry in entries:
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "Unknown")
        result["resource_counts"][resource_type] = (
            result["resource_counts"].get(resource_type, 0) + 1
        )

        # Extract patient info
        if resource_type == "Patient":
            result["demographics"]["gender"] = resource.get("gender")
            result["patient_id"] = resource.get("id")

            # Calculate age from birthDate
            birth_date = resource.get("birthDate")
            if birth_date:
                try:
                    birth_year = int(birth_date.split("-")[0])
                    result["demographics"]["age"] = datetime.now().year - birth_year
                except (ValueError, IndexError):
                    pass

    # Determine available lab categories from Observation resources
    lab_categories = set()
    lab_category_mapping = {
        "troponin": "Cardiac",
        "bnp": "Cardiac",
        "nt-probnp": "Cardiac",
        "d-dimer": "Coag",
        "inr": "Coag",
        "ptt": "Coag",
        "aptt": "Coag",
        "fibrinogen": "Coag",
        "creatinine": "Renal",
        "gfr": "Renal",
        "bun": "Renal",
        "hemoglobin": "CBC",
        "hematocrit": "CBC",
        "wbc": "CBC",
        "platelet": "CBC",
        "glucose": "Metabolic",
        "sodium": "Metabolic",
        "potassium": "Metabolic",
    }

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Observation":
            code = resource.get("code", {})
            display = code.get("text", "").lower()
            if not display:
                for coding in code.get("coding", []):
                    display = coding.get("display", "").lower()
                    if display:
                        break

            for keyword, category in lab_category_mapping.items():
                if keyword in display:
                    lab_categories.add(category)
                    break

    result["available_lab_categories"] = sorted(lab_categories)

    return result


@tool
def search_clinical_history(fhir_bundle: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Search patient's conditions and clinical notes for relevant history.

    WHEN TO USE: When imaging shows a finding, check for relevant history.
    This helps identify if a finding is new vs. known, and provides context.

    SEARCH TIPS:
    - Use broad, partial terms: "thrombo" (matches thrombosis, thromboembolism)
    - Use "malignan" to catch malignancy, malignant
    - Use "diabet" to catch diabetes, diabetic
    - Use "pulmonary" for lung-related conditions
    - Use "cardiac" or "heart" for cardiovascular conditions

    Args:
        fhir_bundle: The FHIR Bundle containing patient data
        query: Search term (case-insensitive, partial match)

    Returns:
        {
            "query": str,
            "match_count": int,
            "conditions": [
                {
                    "display": str,
                    "status": str (active, resolved, etc.),
                    "onset_date": str or null
                }
            ]
        }
    """
    query_lower = query.lower()
    matches = []

    entries = fhir_bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Condition":
            # Get condition text
            code = resource.get("code", {})
            display = code.get("text", "")
            if not display:
                for coding in code.get("coding", []):
                    if coding.get("display"):
                        display = coding["display"]
                        break

            # Check for match
            if display and query_lower in display.lower():
                condition_info = {
                    "display": display,
                    "status": resource.get("clinicalStatus", {})
                    .get("coding", [{}])[0]
                    .get("code", "unknown"),
                    "onset_date": resource.get("onsetDateTime"),
                }
                matches.append(condition_info)

    return {
        "query": query,
        "match_count": len(matches),
        "conditions": matches,
    }


@tool
def get_recent_labs(fhir_bundle: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Retrieve recent laboratory values for a specific category.

    WHEN TO USE:
    - See PE/clot -> check "Coag" for D-dimer, INR, PTT
    - See cardiac finding -> check "Cardiac" for Troponin, BNP
    - See contrast use planned -> check "Renal" for Creatinine, GFR
    - See infection signs -> check "CBC" for WBC, differential
    - General workup -> check "Metabolic" for electrolytes

    AVAILABLE CATEGORIES:
    - "Cardiac": Troponin, BNP, NT-proBNP
    - "Coag": D-dimer, INR, PTT, aPTT, Fibrinogen
    - "Renal": Creatinine, GFR, BUN
    - "CBC": Hemoglobin, Hematocrit, WBC, Platelets
    - "Metabolic": Glucose, Sodium, Potassium

    Args:
        fhir_bundle: The FHIR Bundle containing patient data
        category: Lab category (Cardiac, Coag, Renal, CBC, Metabolic)

    Returns:
        {
            "category": str,
            "lookback_days": int,
            "values": [
                {
                    "name": str,
                    "value": float or str,
                    "unit": str,
                    "date": str,
                    "flag": str (normal, high, low, critical)
                }
            ]
        }
    """
    category_keywords = {
        "Cardiac": ["troponin", "bnp", "nt-probnp", "pro-bnp"],
        "Coag": ["d-dimer", "inr", "ptt", "aptt", "fibrinogen", "coag"],
        "Renal": ["creatinine", "gfr", "bun", "urea"],
        "CBC": ["hemoglobin", "hematocrit", "wbc", "platelet", "rbc"],
        "Metabolic": ["glucose", "sodium", "potassium", "chloride", "bicarbonate"],
    }

    keywords = category_keywords.get(category, [])
    lookback_days = TOOL_LAB_LOOKBACK_DAYS
    cutoff_date = datetime.now() - timedelta(days=lookback_days)

    values = []
    entries = fhir_bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Observation":
            # Get observation name
            code = resource.get("code", {})
            display = code.get("text", "").lower()
            if not display:
                for coding in code.get("coding", []):
                    display = coding.get("display", "").lower()
                    if display:
                        break

            # Check if this observation matches the category
            if not any(kw in display for kw in keywords):
                continue

            # Check date if available
            effective_date = resource.get("effectiveDateTime")
            if effective_date:
                try:
                    obs_date = datetime.fromisoformat(
                        effective_date.replace("Z", "+00:00")
                    )
                    if obs_date.replace(tzinfo=None) < cutoff_date:
                        continue
                except (ValueError, TypeError):
                    pass

            # Extract value
            value_quantity = resource.get("valueQuantity", {})
            value = value_quantity.get("value")
            unit = value_quantity.get("unit", "")

            # Determine flag from interpretation or reference range
            flag = "normal"
            interpretation = resource.get("interpretation", [])
            if interpretation:
                interp_code = interpretation[0].get("coding", [{}])[0].get("code", "")
                if interp_code in ("H", "HH"):
                    flag = "high" if interp_code == "H" else "critical_high"
                elif interp_code in ("L", "LL"):
                    flag = "low" if interp_code == "L" else "critical_low"

            values.append(
                {
                    "name": display.title(),
                    "value": value,
                    "unit": unit,
                    "date": effective_date,
                    "flag": flag,
                }
            )

    return {
        "category": category,
        "lookback_days": lookback_days,
        "values": values,
    }


@tool
def check_medication_status(
    fhir_bundle: Dict[str, Any], medication_name: str
) -> Dict[str, Any]:
    """Check if patient is on a specific medication (active or historical).

    WHEN TO USE - CRITICAL FOR DETECTING TREATMENT FAILURES:
    - See PE/DVT/clot -> check "anticoag", "warfarin", "heparin", "rivaroxaban",
      "apixaban", "enoxaparin"
    - See stroke/TIA -> check "aspirin", "clopidogrel", "antiplatelet"
    - See cardiac event -> check "beta", "metoprolol", "statin"
    - See infection -> check "antibiotic"

    CLINICAL INSIGHT - TREATMENT FAILURE PATTERNS:
    - Clot WHILE ON anticoagulation = ANTICOAGULATION FAILURE = CRITICAL!
    - Stroke WHILE ON antiplatelet = ANTIPLATELET FAILURE = needs escalation
    - High HR WHILE ON beta-blocker = RATE CONTROL FAILURE
    - Elevated LDL WHILE ON statin = LIPID THERAPY FAILURE

    SEARCH TIPS:
    - Use partial terms: "anticoag" matches anticoagulant, anticoagulation
    - Use drug classes: "statin", "beta" (for beta-blockers)
    - Use generic names: "warfarin", "heparin", "aspirin"

    Args:
        fhir_bundle: The FHIR Bundle containing patient data
        medication_name: Medication name or partial match (case-insensitive)

    Returns:
        {
            "query": str,
            "found": bool,
            "medications": [
                {
                    "name": str,
                    "status": str (active, stopped, completed),
                    "start_date": str or null,
                    "dosage": str or null
                }
            ],
            "is_currently_active": bool
        }
    """
    query_lower = medication_name.lower()

    # Expand common abbreviations and classes
    search_terms = [query_lower]

    anticoagulant_terms = [
        "warfarin",
        "coumadin",
        "heparin",
        "enoxaparin",
        "lovenox",
        "rivaroxaban",
        "xarelto",
        "apixaban",
        "eliquis",
        "dabigatran",
        "pradaxa",
        "edoxaban",
        "fondaparinux",
    ]
    if "anticoag" in query_lower:
        search_terms.extend(anticoagulant_terms)

    antiplatelet_terms = [
        "aspirin",
        "clopidogrel",
        "plavix",
        "prasugrel",
        "ticagrelor",
        "brilinta",
    ]
    if "antiplatelet" in query_lower or "platelet" in query_lower:
        search_terms.extend(antiplatelet_terms)

    beta_blocker_terms = [
        "metoprolol",
        "atenolol",
        "carvedilol",
        "propranolol",
        "bisoprolol",
        "labetalol",
    ]
    if "beta" in query_lower:
        search_terms.extend(beta_blocker_terms)

    statin_terms = [
        "atorvastatin",
        "lipitor",
        "simvastatin",
        "zocor",
        "rosuvastatin",
        "crestor",
        "pravastatin",
    ]
    if "statin" in query_lower:
        search_terms.extend(statin_terms)

    medications = []
    is_active = False

    entries = fhir_bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "")

        if resource_type not in ("MedicationStatement", "MedicationRequest"):
            continue

        # Get medication name
        med_concept = resource.get("medicationCodeableConcept", {})
        med_name = med_concept.get("text", "")
        if not med_name:
            for coding in med_concept.get("coding", []):
                med_name = coding.get("display", "")
                if med_name:
                    break

        # Check for match
        med_name_lower = med_name.lower()
        if not any(term in med_name_lower for term in search_terms):
            continue

        # Get status
        status = resource.get("status", "unknown")
        if resource_type == "MedicationStatement":
            status = resource.get("status", "unknown")
        elif resource_type == "MedicationRequest":
            status = resource.get("status", "unknown")

        # Determine if active
        active_statuses = ["active", "intended", "in-progress"]
        if status in active_statuses:
            is_active = True

        # Get dates
        start_date = None
        if resource_type == "MedicationStatement":
            effective = resource.get("effectivePeriod", {})
            start_date = effective.get("start")
        elif resource_type == "MedicationRequest":
            start_date = resource.get("authoredOn")

        # Get dosage
        dosage = None
        dosage_info = resource.get("dosageInstruction", resource.get("dosage", []))
        if dosage_info and isinstance(dosage_info, list) and len(dosage_info) > 0:
            dosage = dosage_info[0].get("text", "")

        medications.append(
            {
                "name": med_name,
                "status": status,
                "start_date": start_date,
                "dosage": dosage,
            }
        )

    return {
        "query": medication_name,
        "found": len(medications) > 0,
        "medications": medications,
        "is_currently_active": is_active,
    }
