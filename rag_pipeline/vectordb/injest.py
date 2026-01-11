"""
Ingest RAG chunks into ChromaDB with embeddings.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import uuid

from .chroma_setup import get_collection
from ..embeddings.embedder_bge_m3 import batch_embed
from ..config.settings import CHUNK_OUTPUT_JSONL, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_chunks_from_jsonl(jsonl_file: Path) -> List[Dict[str, Any]]:
    """
    Load chunks from JSONL file.
    
    Args:
        jsonl_file: Path to JSONL file
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    logger.info(f"Loaded {len(chunks)} chunks from {jsonl_file}")
    return chunks


def prepare_batch_data(
    chunks: List[Dict[str, Any]]
) -> tuple[List[str], List[str], List[Dict], List[str]]:
    """
    Prepare data for batch insertion into ChromaDB.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Tuple of (ids, documents, metadatas, texts_for_embedding)
    """
    ids = []
    documents = []
    metadatas = []
    texts = []
    
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", str(uuid.uuid4()))
        chunk_text = chunk.get("chunk_text", "")
        
        if not chunk_text.strip():
            continue
        
        ids.append(chunk_id)
        documents.append(chunk_text)
        texts.append(chunk_text)
        
        metadata = {
            "parent_segment_id": chunk.get("parent_segment_id", ""),
            "source_file": chunk.get("source_file", ""),
            "language": chunk.get("language", "en"),
            "segment_type": chunk.get("segment_type", ""),
            "safety": chunk.get("safety", ""),
            "tags": ",".join(chunk.get("tags", [])),  # ChromaDB metadata must be strings
            "tokens": chunk.get("tokens", 0),
            "chunk_index": chunk.get("chunk_index", 0),
        }
        
        metadatas.append(metadata)
    
    return ids, documents, metadatas, texts


def ingest_chunks(
    chunks: List[Dict[str, Any]],
    batch_size: int = 100,
    reset_collection: bool = False
) -> None:
    """
    Ingest chunks into ChromaDB with embeddings.
    
    Args:
        chunks: List of chunk dictionaries
        batch_size: Batch size for embedding and insertion
        reset_collection: Whether to reset collection before ingestion
    """
    logger.info("=" * 60)
    logger.info("Starting ChromaDB Ingestion")
    logger.info("=" * 60)
    
    # Get or create collection
    collection = get_collection(reset=reset_collection)
    
    # Prepare data
    ids, documents, metadatas, texts = prepare_batch_data(chunks)
    
    logger.info(f"Prepared {len(ids)} chunks for ingestion")
    
    # Process in batches
    total_batches = (len(ids) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(ids), batch_size), desc="Ingesting batches"):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]
        
        # Generate embeddings
        embeddings = batch_embed(batch_texts, batch_size=32, show_progress=False)
        
        # Insert into ChromaDB
        try:
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_metas
            )
        except Exception as e:
            logger.error(f"Failed to insert batch {i // batch_size + 1}: {e}")
            continue
    
    final_count = collection.count()
    logger.info("=" * 60)
    logger.info(f"✓ Ingestion complete! Total documents: {final_count}")
    logger.info("=" * 60)


def main(reset: bool = False):
    """Main execution function."""
    # Load chunks
    chunks = load_chunks_from_jsonl(CHUNK_OUTPUT_JSONL)
    
    if not chunks:
        logger.error("No chunks found. Run chunk_json_segments.py first.")
        return
    
    # Ingest into ChromaDB
    ingest_chunks(chunks, batch_size=100, reset_collection=reset)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest RAG chunks into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Reset collection before ingestion")
    args = parser.parse_args()
    
    main(reset=args.reset)