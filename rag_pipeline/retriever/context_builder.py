"""
Context builder for combining retrieved chunks.
"""

import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_context(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build context from retrieved chunks.
    
    Args:
        results: List of chunk dictionaries from search
        
    Returns:
        Dictionary containing:
            - combined_text: Concatenated chunk texts
            - chunks: List of chunk texts
            - metadata: List of metadata dicts
            - safety_flags: List of safety levels from chunks
    """
    if not results:
        logger.warning("No chunks retrieved for context building")
        return {
            "combined_text": "",
            "chunks": [],
            "metadata": [],
            "safety_flags": []
        }
    
    # Extract components
    chunks = []
    metadata_list = []
    safety_flags = []
    
    for result in results:
        chunk_text = result.get("chunk_text", "")
        chunks.append(chunk_text)
        
        # Collect metadata
        meta = {
            "source_file": result.get("source_file", ""),
            "segment_type": result.get("segment_type", ""),
            "tags": result.get("tags", []),
            "score": result.get("score", 0.0),
            "safety": result.get("safety", "")
        }
        metadata_list.append(meta)
        
        # Collect safety flags
        safety_level = result.get("safety", "")
        if safety_level:
            safety_flags.append(safety_level)
    
    # Combine chunk texts with separators
    combined_text = "\n---\n".join(chunks)
    
    logger.info(f"Built context from {len(chunks)} chunks")
    logger.debug(f"Safety flags detected: {set(safety_flags)}")
    
    return {
        "combined_text": combined_text,
        "chunks": chunks,
        "metadata": metadata_list,
        "safety_flags": safety_flags
    }