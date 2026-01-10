"""
call_gemini.py

Calls Gemini API with prompt and input text.
Handles retries, rate limits, and JSON validation.
"""

import os
import time
import logging
import requests
from utils.llm_utils import validate_llm_json, cleanup_llm_response

# Add dotenv support
from dotenv import load_dotenv

# Load .env from the project root (DSAI/.env)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(DOTENV_PATH)

logger = logging.getLogger(__name__)

# GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def call_gemini(prompt: str) -> dict:
    """
    Call Gemini API with the given prompt.
    Returns parsed JSON or fallback message.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set in environment.")
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY env variable.")

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={api_key}",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json()
            # Extract model output (assuming text in 'candidates[0].content.parts[0].text')
            model_text = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            cleaned = cleanup_llm_response(model_text)
            json_data = validate_llm_json(cleaned)
            if json_data:
                logger.info("Valid JSON received from Gemini.")
                return json_data
            else:
                logger.warning("Invalid JSON from Gemini. Attempting cleanup.")
        except Exception as e:
            logger.error(f"Gemini API call failed (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                break

    logger.error("All Gemini API attempts failed or returned invalid JSON.")
    return {"error": "Failed to get valid JSON from Gemini."}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Call Gemini API with prompt.")
    parser.add_argument("--prompt_file", required=True, help="Path to prompt file")
    args = parser.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    result = call_gemini(prompt)
    print(result)