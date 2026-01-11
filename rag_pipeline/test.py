"""
Test script for RAG retrieval pipeline.
Loads embedder and ChromaDB, runs queries, and prints results.
"""

import argparse
import logging
from pathlib import Path

from rag_pipeline.vectordb.chroma_query import query, query_with_filters
from rag_pipeline.embeddings.embedder_bge_m3 import load_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def print_results(results, query_text):
    """Pretty print query results."""
    print("\n" + "=" * 80)
    print(f"Query: '{query_text}'")
    print(f"Found {len(results)} results")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f} | Source: {result['source_file']}")
        print(f"   Type: {result['segment_type']} | Safety: {result['safety']} | Tokens: {result['tokens']}")
        
        tags = result.get('tags', [])
        if tags and tags != ['']:
            print(f"   Tags: {', '.join(tags)}")
        
        # Print chunk text (truncated)
        chunk_text = result['chunk_text']
        if len(chunk_text) > 300:
            chunk_text = chunk_text[:300] + "..."
        
        print(f"\n   {chunk_text}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Main test retrieval function."""
    parser = argparse.ArgumentParser(description="Test RAG retrieval pipeline")
    parser.add_argument("--q", required=True, help="Query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--segment_type", help="Filter by segment type")
    parser.add_argument("--include_unsafe", action="store_true", help="Include unsafe content")
    parser.add_argument("--min_score", type=float, default=0.0, help="Minimum similarity score")
    args = parser.parse_args()
    
    logger.info("Initializing RAG Retrieval System")
    logger.info("=" * 60)
    
    # Load embedding model
    logger.info("Loading BGE-M3 embedding model...")
    load_model()
    logger.info("✓ Model loaded")
    
    # Run query
    logger.info(f"Running query: '{args.q}'")
    
    if args.segment_type or args.min_score > 0:
        # Use advanced filtering
        results = query_with_filters(
            q_text=args.q,
            top_k=args.top_k,
            segment_types=[args.segment_type] if args.segment_type else None,
            safety_levels=None,
            min_score=args.min_score
        )
    else:
        # Use simple query
        results = query(
            q_text=args.q,
            top_k=args.top_k,
            exclude_unsafe=not args.include_unsafe
        )
    
    # Print results
    print_results(results, args.q)
    
    # Show statistics
    if results:
        avg_score = sum(r['score'] for r in results) / len(results)
        logger.info(f"Average similarity score: {avg_score:.4f}")
        
        segment_types = {}
        for r in results:
            st = r['segment_type']
            segment_types[st] = segment_types.get(st, 0) + 1
        logger.info(f"Segment types: {segment_types}")
    else:
        logger.warning("No results found")


if __name__ == "__main__":
    main()