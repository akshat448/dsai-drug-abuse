"""
Chunk JSON segments from processed interview files.
Reads standardized JSON files, chunks original_text, and creates rag_dataset.jsonl.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import uuid

from .utils_tokenizer import chunk_text_by_tokens, count_tokens
from ..config.settings import (
    PROCESSED_JSON_FOLDER,
    CHUNK_OUTPUT_JSONL,
    CHUNK_TOKEN_TARGET,
    CHUNK_TOKEN_MAX,
    CHUNK_TOKEN_MIN,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_files(folder: Path) -> List[Dict[str, Any]]:
    """
    Load all JSON files from the processed folder.
    
    Args:
        folder: Path to folder containing JSON files
        
    Returns:
        List of parsed JSON documents
    """
    documents = []
    json_files = list(folder.glob("*.json"))
    
    logger.info(f"Found {len(json_files)} JSON files in {folder}")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents.append(data)
            logger.debug(f"Loaded: {json_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")
    
    return documents


def create_chunks_from_segment(
    segment: Dict[str, Any],
    source_file: str,
    language: str
) -> List[Dict[str, Any]]:
    """
    Create chunks from a single segment's original_text.
    
    Args:
        segment: Segment dictionary with original_text, tags, etc.
        source_file: Source filename
        language: Document language
        
    Returns:
        List of chunk dictionaries
    """
    original_text = segment.get("original_text", "")
    
    if not original_text.strip():
        logger.warning(f"Empty original_text in segment {segment.get('id', 'unknown')}")
        return []
    
    # Chunk the original text
    text_chunks = chunk_text_by_tokens(
        original_text,
        target_tokens=CHUNK_TOKEN_TARGET,
        max_tokens=CHUNK_TOKEN_MAX,
        min_tokens=CHUNK_TOKEN_MIN
    )
    
    chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": i,
            "chunk_text": chunk_text,
            "tokens": count_tokens(chunk_text),
            "parent_segment_id": segment.get("id", "unknown"),
            "source_file": source_file,
            "language": language,
            "generalized_summary": segment.get("generalized_text", ""),
            "tags": segment.get("tags", []),
            "segment_type": segment.get("segment_type", ""),
            "safety": segment.get("safety", ""),
            "metadata": segment.get("metadata", {}),
        }
        chunks.append(chunk)
    
    return chunks


def process_all_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all documents and create chunks.
    
    Args:
        documents: List of JSON documents
        
    Returns:
        List of all chunks
    """
    all_chunks = []
    total_segments = 0
    
    for doc in documents:
        source_file = doc.get("source_file", "unknown")
        language = doc.get("language", "en")
        segments = doc.get("segments", [])
        
        total_segments += len(segments)
        
        for segment in segments:
            chunks = create_chunks_from_segment(segment, source_file, language)
            all_chunks.extend(chunks)
    
    logger.info(f"Processed {len(documents)} documents")
    logger.info(f"Processed {total_segments} segments")
    logger.info(f"Created {len(all_chunks)} chunks")
    
    return all_chunks


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Save chunks to JSONL file.
    
    Args:
        chunks: List of chunk dictionaries
        output_file: Path to output JSONL file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(chunks)} chunks to {output_file}")


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Starting RAG Dataset Creation")
    logger.info("=" * 60)
    
    # Load all JSON files
    documents = load_json_files(PROCESSED_JSON_FOLDER)
    
    if not documents:
        logger.error("No documents found. Exiting.")
        return
    
    # Create chunks
    all_chunks = process_all_documents(documents)
    
    # Save to JSONL
    save_chunks_to_jsonl(all_chunks, CHUNK_OUTPUT_JSONL)
    
    # Statistics
    logger.info("=" * 60)
    logger.info("Statistics:")
    logger.info(f"  Total chunks: {len(all_chunks)}")
    
    if all_chunks:
        token_counts = [c["tokens"] for c in all_chunks]
        logger.info(f"  Avg tokens/chunk: {sum(token_counts) / len(token_counts):.1f}")
        logger.info(f"  Min tokens: {min(token_counts)}")
        logger.info(f"  Max tokens: {max(token_counts)}")
        
        safety_dist = {}
        for chunk in all_chunks:
            safety = chunk.get("safety", "unknown")
            safety_dist[safety] = safety_dist.get(safety, 0) + 1
        logger.info(f"  Safety distribution: {safety_dist}")
    
    logger.info("=" * 60)
    logger.info("✓ RAG dataset creation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()