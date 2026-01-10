"""
schema.py

Defines the standard JSON schema and validation logic.
"""

STANDARD_SCHEMA = {
    "source_file": "",
    "language": "",
    "segments": [
        {
            "id": "",
            "original_text": "",
            "generalized_text": "",
            "segment_type": "",
            "tags": [],
            "safety": "",
            "metadata": {
                "confidence": "",
                "notes": ""
            }
        }
    ]
}


def validate_schema(data: dict) -> bool:
    """
    Validate that data matches the STANDARD_SCHEMA structure.
    Checks required fields and types.
    """
    try:
        if not isinstance(data, dict):
            return False
        for key in ["source_file", "language", "segments"]:
            if key not in data:
                return False
        if not isinstance(data["segments"], list):
            return False
        for seg in data["segments"]:
            for field in ["id", "original_text", "generalized_text", "segment_type", "tags", "safety", "metadata"]:
                if field not in seg:
                    return False
            if not isinstance(seg["tags"], list):
                return False
            if not isinstance(seg["metadata"], dict):
                return False
            for meta_field in ["confidence", "notes"]:
                if meta_field not in seg["metadata"]:
                    return False
        return True
    except Exception:
        return False