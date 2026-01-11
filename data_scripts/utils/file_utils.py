"""
file_utils.py

File path helpers, file type detection, and logging setup.
"""

import os
import logging


def detect_file_type(filepath: str) -> str:
    """Detect file type by extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".docx":
        return "docx"
    elif ext == ".pdf":
        return "pdf"
    else:
        return "unknown"


def get_filename_no_ext(filepath: str) -> str:
    """Get filename without extension."""
    return os.path.splitext(os.path.basename(filepath))[0]


def get_output_path(filename: str) -> str:
    """Get output path for processed JSON."""
    return os.path.join("output", "processed", filename)


def setup_logging(name: str) -> logging.Logger:
    """Set up and return a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger