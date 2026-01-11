"""
process_document.py

High-level orchestrator for the preprocessing pipeline.
Extracts text, builds prompt, calls Gemini, validates, and saves output.
"""
import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root BEFORE importing other modules
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(str(DOTENV_PATH))

from utils.file_utils import get_filename_no_ext, setup_logging
from extract_text import extract_text
from build_prompt import build_prompt
from call_gemini import call_gemini
from save_output import save_output

logger = setup_logging(__name__)

def process_document(filepath: str, language: str = "English") -> None:
    """
    Full pipeline for a single document.
    """
    logger.info(f"Processing document: {filepath}")

    # Step 1: Extract text
    text = extract_text(filepath)
    if not text:
        logger.error("Text extraction failed.")
        return

    # Step 2: Build prompt
    source_file = os.path.basename(filepath)
    prompt = build_prompt(text, source_file, language=language)

    # Step 3: Call Gemini
    llm_output = call_gemini(prompt)
    if "error" in llm_output:
        logger.error("LLM processing failed.")
        return

    # Step 4: Save output
    output_filename = get_filename_no_ext(source_file) + ".json"
    save_output(llm_output, output_filename)
    logger.info(f"Processing complete: {output_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process a document through the full pipeline.")
    parser.add_argument("--file", required=True, help="Path to input document")
    parser.add_argument("--language", default="English", help="Document language")
    # TODO: Add batch processing support
    args = parser.parse_args()

    process_document(args.file, language=args.language)