# """
# Configuration settings for the RAG pipeline.
# """

# import os
# from pathlib import Path

# # Base directories
# BASE_DIR = Path(__file__).parent.parent.parent
# PROCESSED_JSON_FOLDER = BASE_DIR / "output" / "processed"
# RAG_OUTPUT_DIR = BASE_DIR / "rag_pipeline" / "output"
# CHUNK_OUTPUT_JSONL = RAG_OUTPUT_DIR / "rag_dataset.jsonl"
# CHROMA_DB_PATH = RAG_OUTPUT_DIR / "chroma_db"

# # Collection settings
# COLLECTION_NAME = "rag_chunks"

# # Embedding model
# EMBED_MODEL = "BAAI/bge-m3"

# # Chunking parameters
# CHUNK_TOKEN_TARGET = 180
# CHUNK_TOKEN_MAX = 220
# CHUNK_TOKEN_MIN = 120

# # Ensure output directory exists
# RAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Retrieval settings
# TOP_K = 5

# # LLM settings
# GEMINI_MODEL = "gemini-2.5-flash"
# GEMINI_TEMPERATURE = 0.3
# GEMINI_MAX_TOKENS = 1024

# # Safety keywords
# CRISIS_KEYWORDS = [
#     "kill myself", "suicide", "end my life", "want to die", 
#     "overdose on purpose", "self-harm", "hurt myself badly"
# ]

# UNSAFE_KEYWORDS = [
#     "how much should i take", "safe dose", "how to get high",
#     "where to buy", "how to inject", "best way to use"
# ]

# CRAVING_KEYWORDS = [
#     "craving", "urge", "want to use", "tempted", "trigger",
#     "relapse", "slip", "struggling with urges"
# ]

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

# Retrieval settings
TOP_K = 5

# LLM settings (Qwen)
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_REPETITION_PENALTY = 1.1

# Safety keywords
CRISIS_KEYWORDS = [
    "kill myself", "suicide", "end my life", "want to die", 
    "overdose on purpose", "self-harm", "hurt myself badly",
    "no reason to live", "better off dead"
]

UNSAFE_KEYWORDS = [
    "how much should i take", "safe dose", "how to get high",
    "where to buy", "how to inject", "best way to use",
    "cutting corners", "mixing substances"
]

CRAVING_KEYWORDS = [
    "craving", "urge", "want to use", "tempted", "trigger",
    "relapse", "slip", "struggling with urges"
]