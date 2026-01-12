"""
Main chat pipeline orchestrating retrieval, safety, and generation.
"""

import logging
from typing import Dict, Any

from ..retriever.search import embed_query, search_chroma
from ..retriever.context_builder import build_context
from ..safety.safety_router import detect_user_risk, route_safety
from ..generator.rag_prompt import build_rag_prompt
from ..generator.llm_generator import generate_answer
from ..config.settings import TOP_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def answer_user_query(query: str) -> Dict[str, Any]:
    """
    Complete pipeline for answering user queries with safety checks.
    
    Pipeline steps:
    1. Pre-retrieval safety check
    2. Retrieve relevant chunks
    3. Build context
    4. Post-retrieval safety routing
    5. Generate RAG response (if safe)
    
    Args:
        query: User's input question
        
    Returns:
        Dictionary containing:
            - answer: Generated response text
            - chunks_used: List of chunks used for context
            - safety_status: Safety assessment result
            - metadata: Additional context metadata
    """
    logger.info("="*60)
    logger.info(f"Processing query: '{query[:50]}...'")
    logger.info("="*60)
    
    # Step 1: Pre-retrieval safety check
    logger.info("[1/5] Pre-retrieval safety check...")
    user_risk = detect_user_risk(query)
    
    # Step 2: Retrieve relevant chunks
    logger.info("[2/5] Retrieving relevant chunks...")
    query_embedding = embed_query(query)
    search_results = search_chroma(query_embedding, top_k=TOP_K)
    
    # Step 3: Build context
    logger.info("[3/5] Building context...")
    context = build_context(search_results)
    
    # Step 4: Post-retrieval safety routing
    logger.info("[4/5] Post-retrieval safety routing...")
    safety_response = route_safety(query, context)
    
    if safety_response:
        # Safety intervention triggered - return template
        logger.warning("⚠️ Safety template returned")
        return {
            "answer": safety_response,
            "chunks_used": context["chunks"],
            "safety_status": user_risk if user_risk != "none" else "context_unsafe",
            "metadata": context["metadata"],
            "intervention": True
        }
    
    # Step 5: Generate RAG response
    logger.info("[5/5] Generating response...")
    
    if not context["combined_text"]:
        logger.warning("No context available - returning generic support")
        from ..safety.crisis_templates import GENERIC_SUPPORT_TEMPLATE
        return {
            "answer": GENERIC_SUPPORT_TEMPLATE,
            "chunks_used": [],
            "safety_status": "no_context",
            "metadata": [],
            "intervention": False
        }
    
    # Build prompt and generate answer
    prompt = build_rag_prompt(query, context)
    answer = generate_answer(prompt)
    
    logger.info("✓ Response generated successfully")
    
    return {
        "answer": answer,
        "chunks_used": context["chunks"],
        "safety_status": "safe",
        "metadata": context["metadata"],
        "intervention": False
    }