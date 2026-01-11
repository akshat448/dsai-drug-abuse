"""
Run the complete RAG pipeline: chunk → embed → ingest → test.
"""

import logging
import argparse
from pathlib import Path

# Import pipeline steps
from rag_pipeline.chunker.chunk_json_segments import main as create_chunks
from rag_pipeline.vectordb.chroma_ingest import main as ingest_chunks
from rag_pipeline.vectordb.chroma_query import query

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(reset_db: bool = False, test_query: str = None):
    """
    Run the complete RAG pipeline.
    
    Args:
        reset_db: Whether to reset ChromaDB before ingestion
        test_query: Optional test query to run after ingestion
    """
    logger.info("=" * 80)
    logger.info("STARTING FULL RAG PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Create chunks
    logger.info("\n[STEP 1/3] Creating chunks from JSON files...")
    try:
        create_chunks()
        logger.info("✓ Chunking complete\n")
    except Exception as e:
        logger.error(f"✗ Chunking failed: {e}")
        return
    
    # Step 2: Ingest into ChromaDB
    logger.info("[STEP 2/3] Ingesting chunks into ChromaDB...")
    try:
        ingest_chunks(reset=reset_db)
        logger.info("✓ Ingestion complete\n")
    except Exception as e:
        logger.error(f"✗ Ingestion failed: {e}")
        return
    
    # Step 3: Test retrieval (optional)
    if test_query:
        logger.info(f"[STEP 3/3] Testing retrieval with query: '{test_query}'")
        try:
            results = query(test_query, top_k=3)
            
            print("\n" + "=" * 80)
            print("TEST QUERY RESULTS")
            print("=" * 80)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [Score: {result['score']:.4f}] {result['source_file']}")
                print(f"   Type: {result['segment_type']} | Safety: {result['safety']}")
                print(f"   {result['chunk_text'][:150]}...")
            
            print("\n" + "=" * 80)
            logger.info("✓ Test query complete\n")
        except Exception as e:
            logger.error(f"✗ Test query failed: {e}")
    else:
        logger.info("[STEP 3/3] Skipping test query (use --test_query to enable)\n")
    
    logger.info("=" * 80)
    logger.info("✓ FULL PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  - Run test queries: python rag_pipeline/main_test_retrieval.py --q 'your query'")
    logger.info("  - Check stats: ls -lh rag_pipeline/output/")
    logger.info("  - View chunks: head -n 1 rag_pipeline/output/rag_dataset.jsonl | jq")


def main():
    parser = argparse.ArgumentParser(description="Run full RAG pipeline")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB before ingestion")
    parser.add_argument("--test_query", help="Test query to run after ingestion")
    parser.add_argument("--skip_chunks", action="store_true", help="Skip chunking step")
    args = parser.parse_args()
    
    run_pipeline(
        reset_db=args.reset,
        test_query=args.test_query
    )


if __name__ == "__main__":
    main()