"""
Configuration settings for the RAG pipeline.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_JSON_FOLDER = BASE_DIR / "output" / "processed"
RAG_OUTPUT_DIR = BASE_DIR / "rag_pipeline" / "output"
CHUNK_OUTPUT_JSONL = RAG_OUTPUT_DIR / "rag_dataset.jsonl"
CHROMA_DB_PATH = RAG_OUTPUT_DIR / "chroma_db"

# Collection settings
COLLECTION_NAME = "rag_chunks"

# Embedding model
EMBED_MODEL = "BAAI/bge-m3"

# Chunking parameters
CHUNK_TOKEN_TARGET = 180
CHUNK_TOKEN_MAX = 220
CHUNK_TOKEN_MIN = 120

# Ensure output directory exists
RAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)