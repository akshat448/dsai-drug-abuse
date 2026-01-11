"""
call_gemini.py

Calls Gemini API with prompt and input text.
Handles retries, rate limits, and JSON validation.
Supports multiple API keys for increased throughput (20 RPD per key).
"""

import os
import time
import logging
import requests
import json
from threading import Lock
from datetime import datetime, timedelta
from utils.llm_utils import validate_llm_json, cleanup_llm_response
from utils.schema import validate_schema

# Add dotenv support
from dotenv import load_dotenv

# Load .env from the project root (DSAI/.env)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(DOTENV_PATH)

logger = logging.getLogger(__name__)

# Use gemini-2.5-flash (5 RPM, 20 RPD - best balance)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Rate limit settings (per model limits)
MAX_RPM = 5  # 5 requests per minute per key
MAX_RPD = 20  # 20 requests per day per key 

# Conservative settings to stay under limits
REQUESTS_PER_MINUTE = 4  # Stay under 5 RPM limit
MIN_DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # 15 seconds

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds (reduced from 10)

# Load all available API keys
API_KEYS = []
for i in range(1, 10):  # Support up to 9 keys (GEMINI_API_KEY_1 through GEMINI_API_KEY_9)
    key = os.environ.get(f"GEMINI_API_KEY_{i}")
    if key:
        API_KEYS.append(key)
        logger.info(f"✓ Loaded API key #{i}")

# Fallback to single key if no numbered keys found
if not API_KEYS:
    single_key = os.environ.get("GEMINI_API_KEY")
    if single_key:
        API_KEYS.append(single_key)
        logger.info("✓ Using single GEMINI_API_KEY")

if not API_KEYS:
    logger.error("✗ No API keys found. Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. in .env")

logger.info(f"Initialized with {len(API_KEYS)} API key(s)")
logger.info(f"Model: {GEMINI_API_URL}")
logger.info(f"Limits: {MAX_RPM} RPM, {MAX_RPD} RPD per key")

# Global state for key rotation and rate limiting
_state_lock = Lock()
_current_key_index = 0
_key_stats = {i: {"requests_today": 0, "last_request_time": None, "requests_this_minute": 0, "minute_start_time": None} 
              for i in range(len(API_KEYS))}


def _get_next_api_key():
    """
    Get the next available API key using round-robin with rate limit awareness.
    Returns (api_key, key_index) or (None, None) if all keys exhausted.
    """
    global _current_key_index
    
    with _state_lock:
        # Try all keys in rotation
        for _ in range(len(API_KEYS)):
            idx = _current_key_index
            stats = _key_stats[idx]
            
            # Check if this key has quota left
            if stats["requests_today"] < MAX_RPD:
                # Check per-minute rate limit
                now = datetime.now()
                
                # Reset minute counter if needed
                if stats["minute_start_time"] is None or (now - stats["minute_start_time"]) >= timedelta(minutes=1):
                    stats["minute_start_time"] = now
                    stats["requests_this_minute"] = 0
                
                # If under minute limit, use this key
                if stats["requests_this_minute"] < REQUESTS_PER_MINUTE:
                    # Enforce minimum delay between requests for this key
                    if stats["last_request_time"] is not None:
                        elapsed = (now - stats["last_request_time"]).total_seconds()
                        if elapsed < MIN_DELAY_BETWEEN_REQUESTS:
                            wait_time = MIN_DELAY_BETWEEN_REQUESTS - elapsed
                            logger.debug(f"Key #{idx+1}: Spacing requests, waiting {wait_time:.1f}s")
                            time.sleep(wait_time)
                    
                    # Update stats
                    stats["last_request_time"] = datetime.now()
                    stats["requests_this_minute"] += 1
                    stats["requests_today"] += 1
                    
                    # Move to next key for next request
                    _current_key_index = (idx + 1) % len(API_KEYS)
                    
                    logger.debug(f"Using API key #{idx+1} (today: {stats['requests_today']}/{MAX_RPD})")
                    return API_KEYS[idx], idx
                else:
                    # This key hit per-minute limit, wait and retry
                    sleep_time = 60 - (now - stats["minute_start_time"]).total_seconds()
                    if sleep_time > 0:
                        logger.info(f"Key #{idx+1}: Rate limit (RPM), waiting {sleep_time:.1f}s")
                        time.sleep(sleep_time)
                        stats["minute_start_time"] = datetime.now()
                        stats["requests_this_minute"] = 0
                        return _get_next_api_key()  # Retry
            
            # Try next key
            _current_key_index = (idx + 1) % len(API_KEYS)
        
        # All keys exhausted
        logger.error("✗ All API keys have reached their daily quota (20 RPD each)")
        return None, None


def call_gemini(prompt: str, request_id: str = None) -> dict:
    """
    Call Gemini API with the given prompt using available API keys.
    Automatically rotates between keys and handles rate limits.
    
    Args:
        prompt: The prompt text to send to Gemini
        request_id: Optional identifier for logging (e.g., filename)
    
    Returns:
        dict: Valid schema-compliant JSON or error dict
    """
    if not API_KEYS:
        logger.error("✗ No API keys configured.")
        raise RuntimeError("Missing Gemini API keys. Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.")

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,  # Very low for consistent JSON structure
            "topP": 0.1,  # Restrict token diversity
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json"  # Force JSON output
        }
    }

    log_prefix = f"[{request_id}] " if request_id else ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Get next available API key
            api_key, key_idx = _get_next_api_key()
            if api_key is None:
                logger.error(f"{log_prefix}✗ All API keys exhausted.")
                return {"error": "All API keys have reached their daily quota."}
            
            logger.debug(f"{log_prefix}Making API call (attempt {attempt}/{MAX_RETRIES}) with key #{key_idx+1}...")
            
            response = requests.post(
                f"{GEMINI_API_URL}?key={api_key}",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            raw = response.json()
            
            # Extract model output
            try:
                model_text = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not model_text:
                    logger.warning(f"{log_prefix}⚠ Empty response from Gemini (attempt {attempt})")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    continue
                
                # Clean JSON
                cleaned = cleanup_llm_response(model_text)
                if not cleaned:
                    logger.warning(f"{log_prefix}⚠ Failed to extract JSON from response (attempt {attempt})")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    continue
                
                # Parse JSON
                json_data = validate_llm_json(cleaned)
                if not json_data:
                    logger.warning(f"{log_prefix}⚠ Failed to parse JSON (attempt {attempt})")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    continue
                
                # Validate schema
                if not validate_schema(json_data):
                    logger.error(f"{log_prefix}✗ JSON does not match schema")
                    logger.error(f"  Expected keys: source_file, language, segments")
                    logger.error(f"  Got keys: {list(json_data.keys())}")
                    
                    if "segments" in json_data and isinstance(json_data["segments"], list):
                        if json_data["segments"]:
                            logger.error(f"  First segment keys: {list(json_data['segments'][0].keys())}")
                    
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    continue
                
                logger.info(f"{log_prefix}✓ Valid schema-compliant JSON received (key #{key_idx+1})")
                logger.info(f"{log_prefix}  - Segments: {len(json_data.get('segments', []))}")
                return json_data
                
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"{log_prefix}✗ Error parsing API response: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue
                
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            
            if status_code == 429:
                # Rate limit hit
                wait_time = RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(f"{log_prefix}⚠ Rate limit (429). Waiting {wait_time}s before retry {attempt}/{MAX_RETRIES}")
                time.sleep(wait_time)
                
            elif status_code == 503:
                logger.warning(f"{log_prefix}⚠ Service unavailable (503). Waiting {RETRY_DELAY}s before retry {attempt}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
                
            elif status_code == 404:
                logger.error(f"{log_prefix}✗ Model endpoint not found (404)")
                logger.error(f"  URL: {GEMINI_API_URL}")
                return {"error": f"API endpoint not found: {GEMINI_API_URL}"}
                
            elif status_code == 401:
                logger.error(f"{log_prefix}✗ Authentication failed (401). Check API key #{key_idx+1}")
                return {"error": "Invalid API key"}
                
            else:
                logger.error(f"{log_prefix}✗ HTTP error {status_code} (attempt {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    
        except Exception as e:
            logger.error(f"{log_prefix}✗ API call failed (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    logger.error(f"{log_prefix}✗ All API attempts failed. Cannot process.")
    return {"error": "Failed to get valid schema-compliant JSON from Gemini after all retries."}


def get_api_key_stats():
    """Return current usage statistics for all API keys."""
    with _state_lock:
        return {f"key_{i+1}": {
            "requests_today": stats["requests_today"],
            "quota_remaining": MAX_RPD - stats["requests_today"],
            "requests_this_minute": stats["requests_this_minute"]
        } for i, stats in _key_stats.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Call Gemini API with prompt.")
    parser.add_argument("--prompt_file", required=True, help="Path to prompt file")
    parser.add_argument("--stats", action="store_true", help="Show API key stats")
    args = parser.parse_args()

    if args.stats:
        print("\n" + "="*60)
        print("API Key Usage Statistics")
        print("="*60)
        stats = get_api_key_stats()
        for key, info in stats.items():
            print(f"{key}:")
            print(f"  Used today: {info['requests_today']}/20")
            print(f"  Remaining: {info['quota_remaining']}")
            print(f"  This minute: {info['requests_this_minute']}/4")
        print("="*60 + "\n")
    else:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()

        result = call_gemini(prompt, request_id=args.prompt_file)
        if "error" not in result:
            print("\n" + "="*60)
            print("Success! Output:")
            print("="*60)
            print(json.dumps(result, indent=2))
        else:
            print(f"\n✗ Error: {result['error']}")
        
        print("\n" + "="*60)
        print("Current Stats:")
        print("="*60)
        stats = get_api_key_stats()
        for key, info in stats.items():
            print(f"{key}: {info['requests_today']}/20 used, {info['quota_remaining']} remaining")
        print("="*60 + "\n")