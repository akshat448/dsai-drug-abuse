"""
ChromaDB setup and client management.
"""

import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings

from ..config.settings import CHROMA_DB_PATH, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client instance
_client = None


def get_chroma_client(persist_dir: Path = CHROMA_DB_PATH) -> chromadb.Client:
    """
    Get or create ChromaDB persistent client.
    
    Args:
        persist_dir: Directory to persist ChromaDB data
        
    Returns:
        ChromaDB client instance
    """
    global _client
    
    if _client is None:
        logger.info(f"Initializing ChromaDB client at: {persist_dir}")
        
        # Ensure directory exists
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        _client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info("✓ ChromaDB client initialized")
    
    return _client


def get_collection(
    name: str = COLLECTION_NAME,
    reset: bool = False
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection.
    
    Args:
        name: Collection name
        reset: Whether to delete and recreate collection
        
    Returns:
        ChromaDB collection
    """
    client = get_chroma_client()
    
    if reset:
        logger.warning(f"Resetting collection: {name}")
        try:
            client.delete_collection(name=name)
        except Exception as e:
            logger.debug(f"Collection didn't exist: {e}")
    
    collection = client.get_or_create_collection(
        name=name,
        metadata={"description": "RAG chunks from addiction recovery interviews"}
    )
    
    count = collection.count()
    logger.info(f"Collection '{name}' has {count} documents")
    
    return collection


def list_collections() -> list:
    """
    List all collections in ChromaDB.
    
    Returns:
        List of collection names
    """
    client = get_chroma_client()
    collections = client.list_collections()
    return [c.name for c in collections]


if __name__ == "__main__":
    # Test ChromaDB setup
    logger.info("Testing ChromaDB setup")
    
    client = get_chroma_client()
    logger.info(f"Client: {client}")
    
    collections = list_collections()
    logger.info(f"Existing collections: {collections}")
    
    collection = get_collection()
    logger.info(f"Collection: {collection.name}, Count: {collection.count()}")
    
    logger.info("✓ ChromaDB setup test complete")