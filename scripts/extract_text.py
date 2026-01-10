"""
extract_text.py

Extracts clean raw text from .docx and .pdf files.
Handles corrupted files, weird formatting, and missing paragraphs.
Fallback to OCR is a placeholder (TODO).
"""

import os
import sys
import logging
from typing import Optional

import fitz  # PyMuPDF
import docx

from utils.file_utils import detect_file_type, setup_logging

logger = setup_logging(__name__)


def extract_text_docx(filepath: str) -> str:
    """Extract text from a .docx file."""
    try:
        doc = docx.Document(filepath)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        logger.error(f"Failed to extract DOCX: {e}")
        raise


def extract_text_pdf(filepath: str) -> str:
    """Extract text from a .pdf file using PyMuPDF."""
    try:
        doc = fitz.open(filepath)
        text = []
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}")
        raise


def extract_text_ocr(filepath: str) -> str:
    """Placeholder for OCR extraction (TODO)."""
    logger.warning("OCR extraction not implemented.")
    return ""


def extract_text(filepath: str) -> Optional[str]:
    """Main extraction function. Returns clean text or None."""
    if not os.path.isfile(filepath):
        logger.error(f"File not found: {filepath}")
        return None

    file_type = detect_file_type(filepath)
    try:
        if file_type == "docx":
            return extract_text_docx(filepath)
        elif file_type == "pdf":
            return extract_text_pdf(filepath)
        else:
            logger.warning(f"Unsupported file type: {file_type}. Trying OCR fallback.")
            return extract_text_ocr(filepath)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from document.")
    parser.add_argument("--file", required=True, help="Path to input file")
    args = parser.parse_args()

    text = extract_text(args.file)
    if text:
        print(text)
    else:
        logger.error("No text extracted.")