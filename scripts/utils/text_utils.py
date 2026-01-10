"""
text_utils.py

Text normalization and preprocessing helpers.
"""

import re


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and normalize newlines."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\n\s*){2,}", "\n\n", text)
    return text.strip()


def normalize_bullets(text: str) -> str:
    """Convert various bullet styles to a standard one."""
    text = re.sub(r"^[\*\-\u2022]\s+", "- ", text, flags=re.MULTILINE)
    return text


def join_paragraphs(paragraphs: list) -> str:
    """Join paragraphs with double newlines."""
    return "\n\n".join(p.strip() for p in paragraphs if p.strip())


def pre_tokenize(text: str) -> list:
    """Simple whitespace tokenization."""
    return text.split()