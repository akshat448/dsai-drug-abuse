"""
BGE-M3 embedder using HuggingFace sentence-transformers.
Provides text embedding functionality for RAG pipeline.
"""

import logging
from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer

from ..config.settings import EMBED_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
_model = None


def load_model(model_name: str = EMBED_MODEL, device: str = None) -> SentenceTransformer:
    """
    Load BGE-M3 model using sentence-transformers.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        
    Returns:
        Loaded SentenceTransformer model
    """
    global _model
    
    if _model is None:
        logger.info(f"Loading embedding model: {model_name}")
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {device}")
        
        _model = SentenceTransformer(model_name, device=device)
        logger.info("✓ Model loaded successfully")
    
    return _model


def embed(text: str, normalize: bool = True) -> List[float]:
    """
    Embed a single text string.
    
    Args:
        text: Input text to embed
        normalize: Whether to normalize embeddings (recommended for cosine similarity)
        
    Returns:
        Embedding vector as list of floats
    """
    model = load_model()
    
    # Encode returns numpy array by default
    embedding = model.encode(
        text,
        normalize_embeddings=normalize,
        show_progress_bar=False
    )
    
    return embedding.tolist()


def batch_embed(
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True
) -> List[List[float]]:
    """
    Embed multiple texts in batches.
    
    Args:
        texts: List of input texts
        batch_size: Batch size for encoding
        normalize: Whether to normalize embeddings
        show_progress: Show progress bar
        
    Returns:
        List of embedding vectors
    """
    model = load_model()
    
    logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    # Convert to list of lists
    return [emb.tolist() for emb in embeddings]


def get_embedding_dim() -> int:
    """
    Get the dimensionality of embeddings.
    
    Returns:
        Embedding dimension
    """
    model = load_model()
    return model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    # Test embeddings
    logger.info("Testing BGE-M3 embedder")
    
    test_texts = [
        "How do I manage cravings at night?",
        "Family support is important for recovery.",
        "Triggers include stress and peer pressure."
    ]
    
    # Test single embedding
    single_emb = embed(test_texts[0])
    logger.info(f"Single embedding dim: {len(single_emb)}")
    logger.info(f"First 5 values: {single_emb[:5]}")
    
    # Test batch embedding
    batch_embs = batch_embed(test_texts)
    logger.info(f"Batch embeddings: {len(batch_embs)} x {len(batch_embs[0])}")
    
    # Test similarity
    from numpy import dot
    from numpy.linalg import norm
    
    similarity = dot(batch_embs[0], batch_embs[1]) / (norm(batch_embs[0]) * norm(batch_embs[1]))
    logger.info(f"Cosine similarity (text 0 vs 1): {similarity:.4f}")
    
    logger.info("✓ Embedder test complete")