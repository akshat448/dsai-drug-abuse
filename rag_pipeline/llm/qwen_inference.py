"""
Qwen2.5-14B-Instruct inference module.
Handles local LLM generation with proper chat formatting.
"""

import logging
import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and tokenizer instances
_model = None
_tokenizer = None


def load_qwen_model(
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    device: str = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
):
    """
    Load Qwen2.5-14B-Instruct model.
    
    Args:
        model_name: HuggingFace model name
        device: Device to use ('cuda', 'cpu', or None for auto)
        load_in_8bit: Load model in 8-bit precision (saves memory)
        load_in_4bit: Load model in 4-bit precision (saves more memory)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    logger.info(f"Loading Qwen model: {model_name}")
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Configure quantization if requested
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if device == "cuda" else None,
    }
    
    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        logger.info("Loading in 4-bit precision")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif load_in_8bit and device == "cuda":
        logger.info("Loading in 8-bit precision")
        model_kwargs["load_in_8bit"] = True
    else:
        logger.info("Loading in full precision")
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
    
    # Load model
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    if device == "cpu":
        _model = _model.to(device)
    
    logger.info("✓ Qwen model loaded successfully")
    logger.info(f"  Model dtype: {_model.dtype}")
    logger.info(f"  Device: {next(_model.parameters()).device}")
    
    return _model, _tokenizer


def generate_response(
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate response using Qwen model.
    
    Args:
        prompt: User prompt/query
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repetition
        system_prompt: Optional system prompt
        
    Returns:
        Generated text response
    """
    model, tokenizer = load_qwen_model()
    
    # Format chat messages
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096  # Qwen context window
    ).to(model.device)
    
    # Generate
    logger.debug(f"Generating with max_new_tokens={max_new_tokens}, temp={temperature}")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (exclude input)
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return response.strip()


def batch_generate(
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int = 4,
    system_prompt: Optional[str] = None
) -> list[str]:
    """
    Generate responses for multiple prompts in batches.
    
    Args:
        prompts: List of prompts
        max_new_tokens: Maximum tokens per response
        temperature: Sampling temperature
        batch_size: Batch size for generation
        system_prompt: Optional system prompt
        
    Returns:
        List of generated responses
    """
    model, tokenizer = load_qwen_model()
    
    responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Format all messages
        batch_messages = []
        for prompt in batch_prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            batch_messages.append(messages)
        
        # Apply chat template to batch
        batch_texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in batch_messages
        ]
        
        # Tokenize batch
        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(model.device)
        
        # Generate batch
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode batch
        for j, gen_ids in enumerate(generated_ids):
            # Remove input tokens
            gen_ids = gen_ids[model_inputs.input_ids[j].shape[-1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(response.strip())
    
    return responses


if __name__ == "__main__":
    # Test inference
    logger.info("Testing Qwen inference")
    
    test_prompt = "Explain in one sentence what coping strategies are in addiction recovery."
    
    response = generate_response(
        test_prompt,
        max_new_tokens=100,
        temperature=0.7
    )
    
    logger.info(f"\nPrompt: {test_prompt}")
    logger.info(f"\nResponse: {response}")
    logger.info("\n✓ Inference test complete")