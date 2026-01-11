"""
llm_utils.py

Helpers for LLM JSON validation, cleanup, and safety checks.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def cleanup_llm_response(text: str) -> str:
    """
    Attempt to extract and clean JSON from LLM response.
    Removes extra text, code fences, etc.
    """
    import re

    # Remove code fences and non-JSON pre/post text
    json_start = text.find("{")
    json_end = text.rfind("}")
    if json_start == -1 or json_end == -1:
        logger.warning("No JSON object found in LLM response.")
        return ""
    cleaned = text[json_start:json_end + 1]
    # Remove markdown code fences if present
    cleaned = re.sub(r"^```json|^```python|^```", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    return cleaned


def validate_llm_json(text: str) -> Optional[dict]:
    """
    Try to parse and validate JSON from LLM output.
    """
    try:
        data = json.loads(text)
        # Additional schema validation can be added here
        return data
    except Exception as e:
        logger.error(f"Failed to parse LLM JSON: {e}")
        return None


def safety_check(data: dict) -> bool:
    """
    Placeholder for safety checks on LLM output.
    Returns True if safe, False otherwise.
    """
    # TODO: Implement real safety checks
    return True