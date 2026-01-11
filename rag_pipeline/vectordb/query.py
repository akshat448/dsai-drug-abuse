"""
Query ChromaDB for relevant chunks.
"""

import logging
from typing import List, Dict, Any, Optional

from .chroma_setup import get_collection
from ..embeddings.embedder_bge_m3 import embed
from ..config.settings import COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def query(
    q_text: str,
    top_k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
    exclude_unsafe: bool = True
) -> List[Dict[str, Any]]:
    """
    Query ChromaDB for relevant chunks.
    
    Args:
        q_text: Query text
        top_k: Number of results to return
        filter_metadata: Optional metadata filters (e.g., {"segment_type": "coping"})
        exclude_unsafe: Whether to exclude chunks with safety="unsafe" or "red_flag"
        
    Returns:
        List of result dictionaries with chunk_text, score, metadata
    """
    logger.info(f"Query: '{q_text}' (top_k={top_k})")
    
    # Get collection
    collection = get_collection()
    
    # Generate query embedding
    query_embedding = embed(q_text)
    
    # Build where clause for filtering
    where_clause = {}
    if filter_metadata:
        where_clause.update(filter_metadata)
    
    if exclude_unsafe:
        # ChromaDB doesn't support NOT IN, so we'll filter post-query
        pass
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2 if exclude_unsafe else top_k,  # Get extra if filtering
        where=where_clause if where_clause else None,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    formatted_results = []
    
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        chunk_text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        
        # Convert distance to similarity score (cosine distance -> similarity)
        # ChromaDB returns squared L2 distance for normalized vectors (cosine distance)
        similarity = 1 - distance
        
        # Filter unsafe if requested
        if exclude_unsafe and metadata.get("safety") in ["unsafe", "red_flag"]:
            continue
        
        result = {
            "chunk_id": doc_id,
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
        
        # Stop once we have enough results
        if len(formatted_results) >= top_k:
            break
    
    logger.info(f"Retrieved {len(formatted_results)} results")
    
    return formatted_results


def query_with_filters(
    q_text: str,
    top_k: int = 5,
    segment_types: Optional[List[str]] = None,
    safety_levels: Optional[List[str]] = None,
    min_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Query with advanced filtering options.
    
    Args:
        q_text: Query text
        top_k: Number of results
        segment_types: Filter by segment types (e.g., ["coping", "triggers"])
        safety_levels: Filter by safety levels (e.g., ["safe", "sensitive"])
        min_score: Minimum similarity score threshold
        
    Returns:
        Filtered results
    """
    # For now, we'll do post-filtering since ChromaDB's where clause is limited
    results = query(q_text, top_k=top_k * 3, exclude_unsafe=False)
    
    filtered_results = []
    
    for result in results:
        # Filter by segment type
        if segment_types and result["segment_type"] not in segment_types:
            continue
        
        # Filter by safety level
        if safety_levels and result["safety"] not in safety_levels:
            continue
        
        # Filter by minimum score
        if result["score"] < min_score:
            continue
        
        filtered_results.append(result)
        
        if len(filtered_results) >= top_k:
            break
    
    return filtered_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query ChromaDB for relevant chunks")
    parser.add_argument("--q", required=True, help="Query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--segment_type", help="Filter by segment type")
    parser.add_argument("--include_unsafe", action="store_true", help="Include unsafe chunks")
    args = parser.parse_args()
    
    results = query(
        q_text=args.q,
        top_k=args.top_k,
        exclude_unsafe=not args.include_unsafe
    )
    
    print("\n" + "=" * 60)
    print(f"Query: '{args.q}'")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [Score: {result['score']:.4f}] {result['source_file']}")
        print(f"   Type: {result['segment_type']} | Safety: {result['safety']}")
        print(f"   Tags: {', '.join(result['tags'])}")
        print(f"   Text: {result['chunk_text'][:200]}...")
    
    print("\n" + "=" * 60)