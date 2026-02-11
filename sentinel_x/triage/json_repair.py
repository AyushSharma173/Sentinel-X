"""JSON repair utilities for handling 4B model output quirks.

The MedGemma 4B model sometimes produces JSON with common formatting issues:
- Markdown code fences (```json ... ```)
- Python literals (True/False/None instead of true/false/null)
- Trailing commas
- Single quotes instead of double quotes

This module provides utilities to clean and repair these outputs.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def strip_json_decoration(text: str) -> str:
    """Remove markdown code fences and extract JSON content.

    Handles various formats:
    - ```json\\n{...}\\n```
    - ```\\n{...}\\n```
    - Leading/trailing whitespace
    - Trailing text after JSON

    Args:
        text: Raw model output that may contain JSON

    Returns:
        Cleaned text with code fences removed
    """
    # Remove ```json or ``` fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)

    # Find the first { and first [
    first_brace = text.find("{")
    first_bracket = text.find("[")

    # If [ comes before { (or { not found), it's a bare array — extract it
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        depth = 0
        last_bracket = -1
        for i, char in enumerate(text[first_bracket:], first_bracket):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    last_bracket = i
                    break
        if last_bracket != -1:
            return text[first_bracket : last_bracket + 1]

    # Find the JSON object boundaries
    # Look for the first { and last matching }
    if first_brace == -1:
        return text.strip()

    # Count braces to find matching close
    depth = 0
    last_brace = -1
    for i, char in enumerate(text[first_brace:], first_brace):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                last_brace = i
                break

    if last_brace != -1:
        return text[first_brace : last_brace + 1]

    return text.strip()


def repair_json(json_str: str) -> str:
    """Fix common JSON errors from 4B model output.

    Repairs:
    - Python True/False/None -> JSON true/false/null
    - Trailing commas before } or ]
    - Single quotes -> double quotes (careful with apostrophes)

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Repaired JSON string
    """
    # Replace Python booleans/None with JSON equivalents
    # Use word boundaries to avoid replacing within strings
    json_str = re.sub(r"\bTrue\b", "true", json_str)
    json_str = re.sub(r"\bFalse\b", "false", json_str)
    json_str = re.sub(r"\bNone\b", "null", json_str)

    # Remove trailing commas before } or ]
    # This regex finds comma followed by optional whitespace and closing bracket
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

    # Handle single quotes - this is tricky
    # Only convert single quotes that appear to be JSON string delimiters
    # Pattern: Look for 'key': or : 'value' patterns
    # First, temporarily protect apostrophes in English words
    json_str = re.sub(r"(\w)'(\w)", r"\1__APOSTROPHE__\2", json_str)

    # Now convert remaining single quotes to double quotes
    # Be conservative - only convert if it looks like a JSON string delimiter
    def convert_quotes(match):
        content = match.group(1)
        # Escape any existing unescaped double quotes in the content
        content = content.replace('"', '\\"')
        return f'"{content}"'

    json_str = re.sub(r"'([^']*)'", convert_quotes, json_str)

    # Restore apostrophes
    json_str = json_str.replace("__APOSTROPHE__", "'")

    return json_str


def extract_findings_from_narrative(text: str) -> Optional[Dict[str, Any]]:
    """Extract findings from narrative/bullet-point radiology text.

    Handles cases where the 4B model produces a radiology report instead of JSON.
    Looks for anatomical findings described in natural language and converts them
    to the expected structured format.

    Args:
        text: Raw narrative text from model output

    Returns:
        Dict with "findings" list, or None if no findings detected
    """
    findings = []
    seen_descriptions = set()

    # Anatomical locations for matching
    locations = (
        r"(?:RUL|RML|RLL|LUL|LLL|bilateral|left|right|upper|middle|lower|"
        r"mediastin\w*|hilum|hilar|perihilar|lingula|pleural|pericardial|"
        r"subcarinal|paratracheal|apic\w*|basal|basilar)"
    )

    # Finding types to look for
    finding_types = (
        r"(?:nodule|nodular|opacity|opacities|effusion|consolidation|"
        r"emphysema|emphysematous|atelectasis|atelectatic|mass|"
        r"lymphadenopathy|adenopathy|thickening|bronchial\s+wall\s+thickening|"
        r"ground.?glass|fibrosis|fibrotic|calcification|calcified|"
        r"pneumothorax|pleural\s+effusion|cardiomegaly|edema|infiltrate|"
        r"cavit\w+|bronchiectasis|tree.?in.?bud|cyst|bulla|bullae)"
    )

    # Pattern 1: "SIZE finding in/of LOCATION" or "LOCATION: SIZE finding"
    # e.g., "1.1 cm nodule in RUL" or "RUL: 1.1 cm nodule"
    size_pattern = r"(\d+(?:\.\d+)?\s*(?:mm|cm|millimeter|centimeter)s?)"

    # Split into lines and deduplicate
    lines = text.split("\n")
    unique_lines = []
    for line in lines:
        stripped = line.strip().lstrip("- *•")
        if stripped and stripped not in seen_descriptions:
            seen_descriptions.add(stripped)
            unique_lines.append(stripped)

    full_text = " ".join(unique_lines)

    # Strategy A: Look for sentences/phrases containing both a finding type and location
    # Split on sentence boundaries AND treat each original line as a separate candidate
    sentences = re.split(r"[.;]\s+", full_text)
    # Also add individual deduplicated lines as sentence candidates
    for line in unique_lines:
        if line not in sentences:
            sentences.append(line)
    for sentence in sentences:
        sentence_lower = sentence.lower()

        finding_match = re.search(finding_types, sentence_lower)
        if not finding_match:
            continue

        # Skip negative findings — the keyword appears but is negated
        negation_patterns = [
            "no evidence of", "no ", "without evidence of", "without ",
            "not ", "absent", "negative for", "rules out", "ruled out",
        ]
        finding_pos = finding_match.start()
        prefix = sentence_lower[:finding_pos]
        if any(neg in prefix for neg in negation_patterns):
            continue

        finding_type = finding_match.group(0).strip()

        # Normalize finding type
        type_map = {
            "nodular": "nodule", "opacities": "opacity",
            "emphysematous": "emphysema", "atelectatic": "atelectasis",
            "fibrotic": "fibrosis", "calcified": "calcification",
            "adenopathy": "lymphadenopathy", "bullae": "bulla",
        }
        finding_type = type_map.get(finding_type, finding_type)

        # Extract location
        loc_match = re.search(locations, sentence, re.IGNORECASE)
        location = loc_match.group(0) if loc_match else "unspecified"

        # Extract size
        size_match = re.search(size_pattern, sentence, re.IGNORECASE)
        size = size_match.group(0) if size_match else ""

        # Deduplicate by (finding_type, location)
        dedup_key = (finding_type.lower(), location.lower())
        if dedup_key in {(f["finding"].lower(), f["location"].lower()) for f in findings}:
            continue

        findings.append({
            "finding": finding_type,
            "location": location,
            "size": size,
            "slice_index": 0,
            "description": sentence.strip()[:200],
        })

    if findings:
        logger.info(
            f"Extracted {len(findings)} findings from narrative text "
            f"(fallback strategy)"
        )
        return {"findings": findings}

    return None


def _deduplicate_lines(text: str) -> str:
    """Collapse consecutive identical lines to reduce repetition noise.

    The 4B model often loops, producing 80+ copies of the same line.
    This collapses them before any parsing to give cleaner input.
    """
    lines = text.split("\n")
    deduped = []
    prev = None
    for line in lines:
        if line != prev:
            deduped.append(line)
            prev = line
    return "\n".join(deduped)


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON with progressive repair strategies.

    Args:
        text: Raw text that should contain JSON

    Returns:
        Parsed JSON dict, or None if parsing failed
    """
    # Upfront: collapse consecutive identical lines (model repetition loops)
    text = _deduplicate_lines(text)

    # Strategy 1: Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return {"findings": result}
        return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip decoration and try again
    cleaned = strip_json_decoration(text)
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return {"findings": result}
        return result
    except json.JSONDecodeError:
        pass

    # Strategy 3: Apply repairs
    repaired = repair_json(cleaned)
    try:
        result = json.loads(repaired)
        if isinstance(result, list):
            return {"findings": result}
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON after repairs: {e}")
        logger.debug(f"Attempted to parse: {repaired[:200]}...")

    # Strategy 4: Extract findings from narrative/bullet-point text
    narrative_result = extract_findings_from_narrative(text)
    if narrative_result is not None:
        return narrative_result

    return None


