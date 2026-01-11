"""
Tokenization utilities for chunking text.
Uses HuggingFace tokenizers for consistent token counting.
"""

from typing import List
import re
from transformers import AutoTokenizer

# Default tokenizer (aligned with embedding model)
_tokenizer = None


def get_tokenizer():
    """Lazy load the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        # Use same tokenizer as BGE-M3 for consistency
        _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Count tokens in text using HuggingFace tokenizer.
    
    Args:
        text: Input text string
        
    Returns:
        Number of tokens
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def tokenize(text: str) -> List[int]:
    """
    Tokenize text into token IDs.
    
    Args:
        text: Input text string
        
    Returns:
        List of token IDs
    """
    tokenizer = get_tokenizer()
    return tokenizer.encode(text, add_special_tokens=False)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex-based sentence boundary detection.
    
    Args:
        text: Input text string
        
    Returns:
        List of sentences
    """
    # Simple sentence splitter (handles ., !, ?, etc.)
    # Handles common abbreviations and decimal numbers
    sentence_endings = re.compile(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
    )
    
    sentences = sentence_endings.split(text.strip())
    
    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def chunk_text_by_tokens(
    text: str,
    target_tokens: int = 180,
    max_tokens: int = 220,
    min_tokens: int = 120
) -> List[str]:
    """
    Chunk text into segments based on token count.
    Accumulates sentences until target token count is reached.
    
    Args:
        text: Input text to chunk
        target_tokens: Target token count per chunk
        max_tokens: Maximum allowed tokens per chunk
        min_tokens: Minimum tokens for a valid chunk
        
    Returns:
        List of text chunks
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If single sentence exceeds max_tokens, split it further
        if sentence_tokens > max_tokens:
            # Save current chunk if exists
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by whitespace
            words = sentence.split()
            temp_chunk = []
            temp_tokens = 0
            
            for word in words:
                word_tokens = count_tokens(word)
                if temp_tokens + word_tokens > max_tokens:
                    if temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                    temp_chunk = [word]
                    temp_tokens = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_tokens += word_tokens
            
            if temp_chunk:
                chunks.append(" ".join(temp_chunk))
            continue
        
        # Check if adding this sentence exceeds max_tokens
        if current_tokens + sentence_tokens > max_tokens:
            # Save current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            
            # If we've reached target, save chunk
            if current_tokens >= target_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
    
    # Add remaining chunk if it meets minimum threshold
    if current_chunk and current_tokens >= min_tokens:
        chunks.append(" ".join(current_chunk))
    elif current_chunk and chunks:
        # Merge small remaining chunk with last chunk
        chunks[-1] = chunks[-1] + " " + " ".join(current_chunk)
    elif current_chunk:
        # If it's the only chunk, add it anyway
        chunks.append(" ".join(current_chunk))
    
    return chunks


if __name__ == "__main__":
    # Test tokenization
    test_text = """
    This is the first sentence. This is the second sentence.
    This is a longer third sentence that contains more information.
    Fourth sentence here. And a fifth one too.
    """
    
    print(f"Token count: {count_tokens(test_text)}")
    print(f"\nSentences: {split_into_sentences(test_text)}")
    print(f"\nChunks:")
    for i, chunk in enumerate(chunk_text_by_tokens(test_text, target_tokens=20)):
        print(f"{i+1}. [{count_tokens(chunk)} tokens] {chunk}")