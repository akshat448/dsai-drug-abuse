# """
# Interactive RAG chatbot for addiction recovery support.
# Retrieves relevant context and generates supportive responses.
# """

# import sys
# import argparse
# import logging
# import os
# import time
# import requests
# import json
# from pathlib import Path

# # Add parent to path for imports
# PROJECT_ROOT = Path(__file__).parent.parent
# sys.path.insert(0, str(PROJECT_ROOT))

# from rag_pipeline.vectordb.query import query
# from rag_pipeline.generator.rag_prompt import build_rag_prompt
# from rag_pipeline.embeddings.embeddings import load_model

# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(levelname)s] %(message)s"
# )
# logger = logging.getLogger(__name__)

# # Import Gemini settings directly
# from dotenv import load_dotenv
# load_dotenv(PROJECT_ROOT / ".env")

# GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY_1") or os.environ.get("GEMINI_API_KEY")


# def call_gemini_simple(prompt: str) -> dict:
#     """
#     Simple Gemini API call for chat responses.
    
#     Args:
#         prompt: The prompt text
        
#     Returns:
#         dict with response text or error
#     """
#     if not GEMINI_API_KEY:
#         return {"error": "No Gemini API key found"}
    
#     headers = {"Content-Type": "application/json"}
#     payload = {
#         "contents": [{"parts": [{"text": prompt}]}],
#         "generationConfig": {
#             "temperature": 0.7,  # Higher for more natural chat
#             "maxOutputTokens": 1024,
#         }
#     }
    
#     try:
#         response = requests.post(
#             f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
#             headers=headers,
#             json=payload,
#             timeout=30,
#         )
#         response.raise_for_status()
        
#         result = response.json()
#         text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
#         if text:
#             return {"response": text}
#         else:
#             return {"error": "Empty response from Gemini"}
            
#     except requests.exceptions.HTTPError as e:
#         logger.error(f"HTTP error: {e.response.status_code}")
#         return {"error": f"API error: {e.response.status_code}"}
#     except Exception as e:
#         logger.error(f"Error calling Gemini: {e}")
#         return {"error": str(e)}


# def format_context(results):
#     """Format retrieval results into context text."""
#     if not results:
#         return {"combined_text": "[No relevant context found]", "sources": []}
    
#     chunks = []
#     sources = []
    
#     for i, result in enumerate(results, 1):
#         chunk_text = result['chunk_text']
#         source = result['source_file']
#         segment_type = result['segment_type']
        
#         chunks.append(f"[Chunk {i} - {segment_type}]\n{chunk_text}")
#         sources.append(f"{source} ({segment_type})")
    
#     return {
#         "combined_text": "\n\n".join(chunks),
#         "sources": sources
#     }


# def chat(user_query: str, top_k: int = 3, exclude_unsafe: bool = True):
#     """
#     Run a single chat interaction.
    
#     Args:
#         user_query: User's question
#         top_k: Number of chunks to retrieve
#         exclude_unsafe: Whether to exclude unsafe content
#     """
#     logger.info("=" * 80)
#     logger.info(f"User Query: '{user_query}'")
#     logger.info("=" * 80)
    
#     # Step 1: Retrieve relevant context
#     logger.info(f"\n[STEP 1/2] Retrieving relevant context (top_k={top_k})...")
#     results = query(user_query, top_k=top_k, exclude_unsafe=exclude_unsafe)
    
#     if not results:
#         logger.warning("No relevant context found")
#         print("\n💬 Response:")
#         print("I don't have specific information about that in my knowledge base.")
#         print("However, I encourage you to reach out to a counselor or trusted support person.")
#         return
    
#     logger.info(f"✓ Retrieved {len(results)} relevant chunks\n")
    
#     # Show retrieved context
#     print("\n📚 Retrieved Context:")
#     print("-" * 80)
#     for i, result in enumerate(results, 1):
#         print(f"{i}. [Score: {result['score']:.3f}] {result['source_file']}")
#         print(f"   Type: {result['segment_type']} | Safety: {result['safety']}")
#         print(f"   {result['chunk_text'][:150]}...\n")
    
#     # Step 2: Format context and build prompt
#     context = format_context(results)
#     prompt = build_rag_prompt(user_query, context)
    
#     # Step 3: Generate response
#     logger.info("[STEP 2/2] Generating supportive response...")
    
#     # Call Gemini
#     llm_output = call_gemini_simple(prompt)
    
#     if "error" in llm_output:
#         logger.error(f"LLM generation failed: {llm_output['error']}")
#         print("\n💬 Response:")
#         print("I'm having trouble generating a response right now. Please try again.")
#         return
    
#     response_text = llm_output.get("response", "")
    
#     logger.info("✓ Response generated\n")
    
#     # Display response
#     print("=" * 80)
#     print("💬 Response:")
#     print("=" * 80)
#     print(response_text)
#     print("\n" + "=" * 80)
    
#     # Show sources
#     print("\n📖 Sources:")
#     for source in context["sources"]:
#         print(f"  - {source}")
#     print()


# def interactive_mode():
#     """Run in interactive loop mode."""
#     logger.info("Starting Interactive Chat Mode")
#     logger.info("Type 'quit' or 'exit' to stop\n")
    
#     # Load model once
#     logger.info("Loading embedding model...")
#     load_model()
#     logger.info("✓ Model loaded\n")
    
#     while True:
#         try:
#             user_input = input("\n💭 You: ").strip()
            
#             if user_input.lower() in ['quit', 'exit', 'q']:
#                 print("\n👋 Take care! Remember, support is always available.\n")
#                 break
            
#             if not user_input:
#                 continue
            
#             chat(user_input, top_k=3, exclude_unsafe=True)
            
#         except KeyboardInterrupt:
#             print("\n\n👋 Take care! Remember, support is always available.\n")
#             break
#         except Exception as e:
#             logger.error(f"Error: {e}")
#             print(f"\n❌ Error: {e}\n")


# def main():
#     parser = argparse.ArgumentParser(description="RAG-based addiction support chatbot")
#     parser.add_argument("--q", help="Single query (non-interactive mode)")
#     parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
#     parser.add_argument("--include_unsafe", action="store_true", help="Include unsafe content")
#     parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
#     args = parser.parse_args()
    
#     # Load embedding model
#     logger.info("Loading embedding model...")
#     load_model()
#     logger.info("✓ Model loaded\n")
    
#     if args.interactive:
#         interactive_mode()
#     elif args.q:
#         chat(args.q, top_k=args.top_k, exclude_unsafe=not args.include_unsafe)
#     else:
#         # Default to interactive if no query provided
#         interactive_mode()


# if __name__ == "__main__":
#     main()

"""
Interactive RAG chatbot for addiction recovery support.
Uses Qwen2.5-14B-Instruct for local inference.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.vectordb.query import query
from rag_pipeline.generator.rag_prompt import build_rag_prompt, build_crisis_response_prompt
from rag_pipeline.embeddings.embeddings import load_model
from rag_pipeline.llm.qwen_inference import generate_response, load_qwen_model
from rag_pipeline.config.settings import CRISIS_KEYWORDS, UNSAFE_KEYWORDS

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def detect_crisis(query: str) -> bool:
    """
    Detect if query contains crisis keywords.
    
    Args:
        query: User query
        
    Returns:
        True if crisis keywords detected
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in CRISIS_KEYWORDS)


def detect_unsafe(query: str) -> bool:
    """
    Detect if query contains unsafe content.
    
    Args:
        query: User query
        
    Returns:
        True if unsafe keywords detected
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in UNSAFE_KEYWORDS)


def format_context(results):
    """Format retrieval results into context text."""
    if not results:
        return {"combined_text": "[No relevant context found]", "sources": []}
    
    chunks = []
    sources = []
    
    for i, result in enumerate(results, 1):
        chunk_text = result['chunk_text']
        source = result['source_file']
        segment_type = result['segment_type']
        
        chunks.append(f"[Chunk {i} - {segment_type}]\n{chunk_text}")
        sources.append(f"{source} ({segment_type})")
    
    return {
        "combined_text": "\n\n".join(chunks),
        "sources": sources
    }


def chat(
    user_query: str,
    top_k: int = 3,
    exclude_unsafe: bool = True,
    max_tokens: int = 512,
    temperature: float = 0.7
):
    """
    Run a single chat interaction.
    
    Args:
        user_query: User's question
        top_k: Number of chunks to retrieve
        exclude_unsafe: Whether to exclude unsafe content
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    logger.info("=" * 80)
    logger.info(f"User Query: '{user_query}'")
    logger.info("=" * 80)
    
    # Check for crisis content
    if detect_crisis(user_query):
        logger.warning("⚠ Crisis keywords detected")
        print("\n" + "=" * 80)
        print("⚠️  CRISIS SUPPORT NEEDED")
        print("=" * 80)
        print("\n🆘 If you're in crisis, please reach out immediately:")
        print("   • National Suicide Prevention Lifeline: 988")
        print("   • Crisis Text Line: Text HOME to 741741")
        print("   • Emergency Services: 911")
        print("\nYour life matters. Professional help is available 24/7.")
        print("\n" + "=" * 80 + "\n")
        
        # Generate supportive crisis response
        crisis_prompt = build_crisis_response_prompt(user_query)
        response = generate_response(
            crisis_prompt,
            max_new_tokens=300,
            temperature=0.5
        )
        print("💬 Additional Support:")
        print(response)
        return
    
    # Check for unsafe content
    if detect_unsafe(user_query):
        logger.warning("⚠ Unsafe keywords detected")
        print("\n💬 Response:")
        print("I can't provide information about that topic as it could be harmful.")
        print("I'm here to support your recovery journey in safe, healthy ways.")
        print("Please reach out to your counselor or treatment provider for guidance.")
        return
    
    # Step 1: Retrieve relevant context
    logger.info(f"\n[STEP 1/2] Retrieving relevant context (top_k={top_k})...")
    results = query(user_query, top_k=top_k, exclude_unsafe=exclude_unsafe)
    
    if not results:
        logger.warning("No relevant context found")
        print("\n💬 Response:")
        print("I don't have specific information about that in my knowledge base.")
        print("However, I encourage you to reach out to a counselor or trusted support person.")
        print("\nFor general support:")
        print("  • SAMHSA National Helpline: 1-800-662-4357")
        print("  • Local support groups and recovery communities")
        return
    
    logger.info(f"✓ Retrieved {len(results)} relevant chunks\n")
    
    # Show retrieved context
    print("\n📚 Retrieved Context:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. [Score: {result['score']:.3f}] {result['source_file']}")
        print(f"   Type: {result['segment_type']} | Safety: {result['safety']}")
        print(f"   {result['chunk_text'][:150]}...\n")
    
    # Step 2: Format context and build prompt
    context = format_context(results)
    prompt = build_rag_prompt(user_query, context)
    
    # Step 3: Generate response
    logger.info("[STEP 2/2] Generating supportive response...")
    
    try:
        response_text = generate_response(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        logger.info("✓ Response generated\n")
        
        # Display response
        print("=" * 80)
        print("💬 Response:")
        print("=" * 80)
        print(response_text)
        print("\n" + "=" * 80)
        
        # Show sources
        print("\n📖 Sources:")
        for source in context["sources"]:
            print(f"  - {source}")
        print()
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        print("\n💬 Response:")
        print("I'm having trouble generating a response right now.")
        print("Please try rephrasing your question or reach out to a support person.")


def interactive_mode(
    top_k: int = 3,
    max_tokens: int = 512,
    temperature: float = 0.7,
    load_in_4bit: bool = False
):
    """
    Run in interactive loop mode.
    
    Args:
        top_k: Number of chunks to retrieve
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        load_in_4bit: Load LLM in 4-bit precision
    """
    logger.info("Starting Interactive Chat Mode")
    logger.info("Type 'quit' or 'exit' to stop\n")
    
    # Load models
    logger.info("Loading embedding model...")
    load_model()
    logger.info("✓ Embedding model loaded")
    
    logger.info("Loading Qwen LLM...")
    load_qwen_model(load_in_4bit=load_in_4bit)
    logger.info("✓ LLM loaded\n")
    
    print("=" * 80)
    print("🤖 Addiction Recovery Support Chatbot")
    print("=" * 80)
    print("Ask me questions about recovery, coping strategies, or managing challenges.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("💭 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Take care! Remember, support is always available.\n")
                break
            
            if not user_input:
                continue
            
            chat(user_input, top_k=top_k, max_tokens=max_tokens, temperature=temperature)
            
        except KeyboardInterrupt:
            print("\n\n👋 Take care! Remember, support is always available.\n")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG-based addiction support chatbot with Qwen")
    parser.add_argument("--q", help="Single query (non-interactive mode)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--include_unsafe", action="store_true", help="Include unsafe content in retrieval")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load LLM in 4-bit precision (saves VRAM)")
    args = parser.parse_args()
    
    if args.interactive or not args.q:
        interactive_mode(
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            load_in_4bit=args.load_in_4bit
        )
    else:
        # Load models for single query
        logger.info("Loading models...")
        load_model()
        load_qwen_model(load_in_4bit=args.load_in_4bit)
        logger.info("✓ Models loaded\n")
        
        chat(
            args.q,
            top_k=args.top_k,
            exclude_unsafe=not args.include_unsafe,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )


if __name__ == "__main__":
    main()