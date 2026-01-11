"""
process_batch.py

Batch processor for multiple documents.
Supports multiple API keys for increased throughput (20 RPD per key).
Includes checkpoint system to avoid reprocessing files.
"""

import os
import sys
import logging
import glob
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Load .env from the project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(str(DOTENV_PATH))

from utils.file_utils import get_filename_no_ext, setup_logging
from extract_text import extract_text
from build_prompt import build_prompt
from call_gemini import call_gemini, get_api_key_stats
from save_output import save_output

logger = setup_logging(__name__)

# Checkpoint file to track processed files
CHECKPOINT_FILE = PROJECT_ROOT / "output" / "batch_checkpoint.json"


def load_checkpoint():
    """Load processing checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return {"processed_files": [], "failed_files": [], "last_run": None}


def save_checkpoint(checkpoint):
    """Save processing checkpoint."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.debug(f"Checkpoint saved: {len(checkpoint['processed_files'])} files processed")


def process_single_file(filepath: str, language: str = "English") -> dict:
    """Process a single document and return result."""
    filename = os.path.basename(filepath)
    logger.info(f"Processing: {filename}")
    
    try:
        # Step 1: Extract text
        text = extract_text(filepath)
        if not text:
            logger.error(f"{filename}: Text extraction failed.")
            return {"file": filename, "status": "failed", "error": "Text extraction failed"}

        # Step 2: Build prompt
        prompt = build_prompt(text, filename, language=language)

        # Step 3: Call Gemini (rate limiting and key rotation handled internally)
        llm_output = call_gemini(prompt, request_id=filename)
        if "error" in llm_output:
            logger.error(f"{filename}: LLM processing failed.")
            return {"file": filename, "status": "failed", "error": "LLM processing failed"}

        # Step 4: Save output
        output_filename = get_filename_no_ext(filename) + ".json"
        save_output(llm_output, output_filename)
        logger.info(f"{filename}: ✓ Processing complete → {output_filename}")
        return {"file": filename, "status": "success", "output": output_filename}
        
    except Exception as e:
        logger.error(f"{filename}: Error - {e}")
        return {"file": filename, "status": "failed", "error": str(e)}


def process_batch(data_dir: str, pattern: str = "*.docx", language: str = "English", max_files: int = None, skip_processed: bool = True) -> None:
    """
    Process all matching files in directory sequentially.
    With 2 API keys: 40 files per day (20 RPD per key).
    
    Args:
        data_dir: Directory containing input files
        pattern: File pattern to match
        language: Document language
        max_files: Max files to process in this run
        skip_processed: If True, skip already processed files
    """
    files = glob.glob(os.path.join(data_dir, pattern))
    
    if not files:
        logger.warning(f"No files found matching pattern: {pattern}")
        return
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_files = set(checkpoint["processed_files"])
    
    # Filter files
    if skip_processed:
        files_to_process = [f for f in files if os.path.basename(f) not in processed_files]
        logger.info(f"Found {len(files)} total files | {len(processed_files)} already processed | {len(files_to_process)} remaining")
    else:
        files_to_process = files
        logger.info(f"Found {len(files)} total files (reprocessing enabled)")
    
    if not files_to_process:
        logger.info("✓ All files already processed! Use --skip_processed False to reprocess.")
        return
    
    # Calculate max capacity based on number of API keys
    num_keys = len([k for k in os.environ.keys() if k.startswith("GEMINI_API_KEY_")])
    if num_keys == 0:
        num_keys = 1 if os.environ.get("GEMINI_API_KEY") else 0
    
    max_capacity = num_keys * 20  # 20 RPD per key
    
    # Limit to max_files or total capacity
    if max_files:
        batch_files = files_to_process[:max_files]
        logger.info(f"Processing {len(batch_files)} files (limited by --max_files {max_files})")
    elif len(files_to_process) > max_capacity:
        batch_files = files_to_process[:max_capacity]
        logger.warning(f"Found {len(files_to_process)} remaining files, but max capacity is {max_capacity} ({num_keys} keys × 20 RPD).")
        logger.warning(f"Processing {len(batch_files)} files now. Run again to process remaining {len(files_to_process) - len(batch_files)}.")
    else:
        batch_files = files_to_process
        logger.info(f"Processing all {len(batch_files)} remaining files")
    
    logger.info(f"Capacity: {max_capacity} files/day ({num_keys} API key(s) × 20 RPD)")
    logger.info("")
    
    results = []
    for i, filepath in enumerate(batch_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"File {i}/{len(batch_files)} (Total: {len(processed_files) + i}/{len(files)})")
        logger.info(f"{'='*60}")
        
        result = process_single_file(filepath, language)
        results.append(result)
        
        # Save checkpoint after each successful file
        if result["status"] == "success":
            checkpoint["processed_files"].append(result["file"])
            checkpoint["last_run"] = datetime.now().isoformat()
            save_checkpoint(checkpoint)
        
        # Show API key stats every 5 files
        if i % 5 == 0:
            stats = get_api_key_stats()
            logger.info(f"\n📊 API Key Usage:")
            for key_name, key_stats in stats.items():
                logger.info(f"   {key_name}: {key_stats['requests_today']}/20 used, {key_stats['quota_remaining']} remaining")
    
    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"=== Batch Processing Complete ===")
    logger.info(f"This session: {len(results)} | ✓ Success: {success} | ✗ Failed: {failed}")
    logger.info(f"Total processed: {len(checkpoint['processed_files'])}/{len(files)}")
    logger.info(f"{'='*60}")
    
    # Final API key stats
    final_stats = get_api_key_stats()
    logger.info(f"\n📊 Final API Key Usage:")
    for key_name, key_stats in final_stats.items():
        logger.info(f"   {key_name}: {key_stats['requests_today']}/20 used, {key_stats['quota_remaining']} remaining")
    
    if failed > 0:
        logger.warning("\n⚠ Failed files:")
        for r in results:
            if r["status"] == "failed":
                logger.warning(f"  - {r['file']}: {r.get('error', 'Unknown error')}")
                checkpoint["failed_files"].append({"file": r["file"], "error": r.get("error")})
        save_checkpoint(checkpoint)
    
    # Summary for next run
    remaining = len(files) - len(checkpoint["processed_files"])
    if remaining > 0:
        logger.info(f"\n📋 Next run: {remaining} files remaining")
        if remaining > max_capacity:
            days_needed = (remaining + max_capacity - 1) // max_capacity
            logger.info(f"   Estimate: Run script {days_needed} more time(s) to complete all files")
    else:
        logger.info(f"\n✓ All {len(files)} files processed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process documents with checkpoint resumption.")
    parser.add_argument("--data_dir", default="data", help="Directory containing input files")
    parser.add_argument("--pattern", default="*.docx", help="File pattern (e.g., *.docx, *.pdf, *.*)")
    parser.add_argument("--language", default="English", help="Document language")
    parser.add_argument("--max_files", type=int, help="Max files to process in this run")
    parser.add_argument("--skip_processed", type=bool, default=True, help="Skip already processed files (default: True)")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and reprocess all files")
    args = parser.parse_args()
    
    # Handle reset
    if args.reset:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("✓ Checkpoint reset. Reprocessing all files.")
        args.skip_processed = False

    process_batch(
        args.data_dir,
        pattern=args.pattern,
        language=args.language,
        max_files=args.max_files,
        skip_processed=args.skip_processed
    )