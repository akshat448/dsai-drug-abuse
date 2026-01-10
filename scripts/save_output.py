"""
save_output.py

Validates and saves LLM JSON output to output/processed/<filename>.json.
Pretty-prints JSON and checks required fields.
"""

import os
import json
import logging
from utils.schema import validate_schema
from utils.file_utils import get_output_path, setup_logging

logger = setup_logging(__name__)


def save_output(data: dict, filename: str) -> None:
    """
    Validate and save JSON output.
    """
    if not validate_schema(data):
        logger.error("Output does not match required schema. Not saving.")
        raise ValueError("Invalid output schema.")

    output_path = get_output_path(filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Output saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save LLM output JSON.")
    parser.add_argument("--input_file", required=True, help="Path to input JSON file")
    parser.add_argument("--output_file", required=True, help="Output filename")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    save_output(data, args.output_file)