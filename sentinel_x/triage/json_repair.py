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

    # Find the JSON object boundaries
    # Look for the first { and last matching }
    first_brace = text.find("{")
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


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON with progressive repair strategies.

    Args:
        text: Raw text that should contain JSON

    Returns:
        Parsed JSON dict, or None if parsing failed
    """
    # Strategy 1: Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip decoration and try again
    cleaned = strip_json_decoration(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Apply repairs
    repaired = repair_json(cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON after repairs: {e}")
        logger.debug(f"Attempted to parse: {repaired[:200]}...")
        return None


def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract a tool call JSON from model output.

    Looks for the pattern:
    TOOL_CALL:
    {"tool": "...", "arguments": {...}}

    Args:
        text: Model response text

    Returns:
        Parsed tool call dict with 'tool' and 'arguments' keys, or None
    """
    # Look for TOOL_CALL: marker
    match = re.search(r"TOOL_CALL:\s*\n?\s*(\{.*)", text, re.DOTALL)
    if not match:
        return None

    json_text = match.group(1)
    result = parse_json_safely(json_text)

    if result and "tool" in result:
        # Normalize: ensure 'arguments' key exists
        if "arguments" not in result:
            result["arguments"] = result.get("args", {})
        return result

    return None


def extract_final_assessment(text: str) -> Optional[Dict[str, str]]:
    """Extract final assessment components from model output.

    Looks for:
    FINAL_ASSESSMENT: <text>
    RISK_ADJUSTMENT: INCREASE|DECREASE|NONE
    CRITICAL_FINDINGS: [list]

    Args:
        text: Model response text

    Returns:
        Dict with 'assessment', 'risk_adjustment', 'critical_findings'
    """
    result = {}

    # Extract FINAL_ASSESSMENT
    match = re.search(
        r"FINAL_ASSESSMENT:\s*(.+?)(?=\n(?:RISK_ADJUSTMENT|CRITICAL_FINDINGS)|$)",
        text,
        re.DOTALL,
    )
    if match:
        result["assessment"] = match.group(1).strip()

    # Extract RISK_ADJUSTMENT
    match = re.search(r"RISK_ADJUSTMENT:\s*(INCREASE|DECREASE|NONE)", text, re.IGNORECASE)
    if match:
        result["risk_adjustment"] = match.group(1).upper()

    # Extract CRITICAL_FINDINGS
    match = re.search(r"CRITICAL_FINDINGS:\s*\[([^\]]*)\]", text)
    if match:
        findings_text = match.group(1)
        # Parse the list items - handle quoted strings
        findings = []
        for item in re.findall(r'"([^"]+)"', findings_text):
            findings.append(item)
        result["critical_findings"] = findings
    else:
        result["critical_findings"] = []

    return result if result else None
