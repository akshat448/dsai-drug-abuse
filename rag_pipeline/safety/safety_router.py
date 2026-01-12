"""
Safety routing and risk detection for user queries.
"""

import logging
from typing import Optional, Dict, Any

from .crisis_templates import (
    CRISIS_TEMPLATE,
    UNSAFE_BEHAVIOR_TEMPLATE,
    CRAVING_TEMPLATE,
    GENERIC_SUPPORT_TEMPLATE
)
from ..config.settings import CRISIS_KEYWORDS, UNSAFE_KEYWORDS, CRAVING_KEYWORDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_user_risk(user_query: str) -> str:
    """
    Detect risk level in user query.
    
    Args:
        user_query: User's input text
        
    Returns:
        Risk level: "crisis", "unsafe", "craving", or "none"
    """
    query_lower = user_query.lower()
    
    # Check for crisis indicators (highest priority)
    for keyword in CRISIS_KEYWORDS:
        if keyword in query_lower:
            logger.warning(f"Crisis keyword detected: '{keyword}'")
            return "crisis"
    
    # Check for unsafe behavior requests
    for keyword in UNSAFE_KEYWORDS:
        if keyword in query_lower:
            logger.warning(f"Unsafe keyword detected: '{keyword}'")
            return "unsafe"
    
    # Check for craving/urge mentions
    for keyword in CRAVING_KEYWORDS:
        if keyword in query_lower:
            logger.info(f"Craving keyword detected: '{keyword}'")
            return "craving"
    
    return "none"


def route_safety(user_query: str, context: Dict[str, Any]) -> Optional[str]:
    """
    Route to safety template if needed based on query and context.
    
    Args:
        user_query: User's input text
        context: Context dictionary from retriever
        
    Returns:
        Safety template string if intervention needed, None otherwise
    """
    # Check user query risk level
    user_risk = detect_user_risk(user_query)
    
    # Check chunk-level safety flags
    safety_flags = context.get("safety_flags", [])
    has_red_flag = "red_flag" in safety_flags
    has_unsafe = "unsafe" in safety_flags
    
    # Route based on risk level
    if user_risk == "crisis" or has_red_flag:
        logger.warning("🚨 CRISIS INTERVENTION triggered")
        return CRISIS_TEMPLATE
    
    if user_risk == "unsafe" or has_unsafe:
        logger.warning("⚠️ UNSAFE BEHAVIOR template triggered")
        return UNSAFE_BEHAVIOR_TEMPLATE
    
    if user_risk == "craving":
        logger.info("💙 CRAVING SUPPORT template triggered")
        return CRAVING_TEMPLATE
    
    # No safety intervention needed
    logger.debug("No safety intervention required")
    return None