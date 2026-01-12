"""
LLM generator using Google Gemini API.
"""

import os
import logging
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from ..config.settings import GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
_model = None


def get_model():
    """Load and cache Gemini model."""
    global _model
    
    if _model is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY_1")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        logger.info(f"Loading Gemini model: {GEMINI_MODEL}")
        _model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={
                "temperature": GEMINI_TEMPERATURE,
                "max_output_tokens": GEMINI_MAX_TOKENS,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"  # Allow harm reduction discussions
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        )
        logger.info("✓ Gemini model loaded")
    
    return _model


def generate_answer(prompt: str) -> str:
    """
    Generate answer using Gemini API.
    
    Args:
        prompt: Complete prompt with system instructions and context
        
    Returns:
        Generated text response
    """
    model = get_model()
    
    try:
        logger.debug("Calling Gemini API...")
        response = model.generate_content(prompt)
        
        if not response.text:
            logger.error("Empty response from Gemini")
            return "I apologize, but I wasn't able to generate a response. Please try rephrasing your question or contact SAMHSA at 1-800-662-4357 for support."
        
        logger.info(f"✓ Generated response ({len(response.text)} chars)")
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I encountered an error while processing your request. Please try again or contact SAMHSA at 1-800-662-4357 for immediate support."