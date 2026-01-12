"""
Search module for retrieving relevant chunks from ChromaDB.
"""

import logging
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config.settings import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
_embedder = None
_chroma_client = None
_collection = None


def get_embedder() -> SentenceTransformer:
    """Load and cache the embedding model."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
        logger.info("✓ Embedder loaded")
    return _embedder


def get_chroma_collection():
    """Get ChromaDB collection."""
    global _chroma_client, _collection
    
    if _collection is None:
        logger.info(f"Connecting to ChromaDB at: {CHROMA_PATH}")
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
        logger.info(f"✓ Connected to collection '{COLLECTION_NAME}' ({_collection.count()} docs)")
    
    return _collection


def embed_query(text: str) -> List[float]:
    """
    Embed query text using BGE-M3 model.
    
    Args:
        text: Query text to embed
        
    Returns:
        Embedding vector as list of floats
    """
    embedder = get_embedder()
    embedding = embedder.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return embedding.tolist()


def search_chroma(query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search ChromaDB for relevant chunks.
    
    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return
        
    Returns:
        List of result dictionaries with chunk text, metadata, and scores
    """
    collection = get_chroma_collection()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    formatted_results = []
    
    if results["ids"] and len(results["ids"][0]) > 0:
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            chunk_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Convert distance to similarity score
            similarity = 1 - distance
            
            result = {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "score": similarity,
                "source_file": metadata.get("source_file", ""),
                "segment_type": metadata.get("segment_type", ""),
                "tags": metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                "safety": metadata.get("safety", ""),
                "language": metadata.get("language", "en"),
                "tokens": metadata.get("tokens", 0),
            }
            
            formatted_results.append(result)
    
    logger.debug(f"Retrieved {len(formatted_results)} chunks")
    return formatted_results