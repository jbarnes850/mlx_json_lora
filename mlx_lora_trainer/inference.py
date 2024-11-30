"""MLX LoRA Trainer Inference Module."""

import argparse
import json
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import mlx.core as mx
from transformers import AutoTokenizer, PreTrainedTokenizer

from .model import ModelRegistry
from .utils import load_config


@dataclass
class Cache:
    """Cache for key/value pairs in attention."""

    def __init__(self, max_seq_length: int, num_layers: int, num_heads: int, head_dim: int):
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.keys = [None] * num_layers
        self.values = [None] * num_layers
        self.offset = 0

    def update_and_fetch(self, layer_idx: int, key: mx.array, value: mx.array):
        """Update the cache with new key/value pairs and return the concatenated values."""
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            # Concatenate along sequence length dimension (dim 2)
            self.keys[layer_idx] = mx.concatenate([self.keys[layer_idx], key], axis=2)
            self.values[layer_idx] = mx.concatenate([self.values[layer_idx], value], axis=2)

        return self.keys[layer_idx], self.values[layer_idx]


def sample_top_p(logits: mx.array, p: float = 0.9, temp: float = 1.0) -> mx.array:
    """Sample from the logits with temperature and nucleus sampling."""
    logits = logits / temp
    
    # Sort logits in descending order
    sorted_indices = mx.argsort(-logits, axis=-1)  # Negative for descending order
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    
    # Compute cumulative probabilities
    probs = mx.softmax(sorted_logits, axis=-1)
    cumulative_probs = mx.cumsum(probs, axis=-1)
    
    # Create nucleus sampling mask
    mask = cumulative_probs > p
    sorted_logits = mx.where(mask, -float('inf'), sorted_logits)
    
    # Sample from the filtered distribution
    probs = mx.softmax(sorted_logits, axis=-1)
    item = mx.random.categorical(probs)
    return mx.take(sorted_indices[0], item)


def generate_text(
    prompt: str,
    model_path: str,
    max_tokens: int = 100,
    temp: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text using a fine-tuned model.
    
    Args:
        prompt: Input text prompt
        model_path: Path to the model directory containing the adapter
        max_tokens: Maximum number of tokens to generate
        temp: Temperature for sampling
        top_p: Top-p sampling parameter
    
    Returns:
        Generated text response
    """
    # Load model config and get model name
    config = load_config(model_path)
    model_name = config["model"]["name"]
    
    # Load model and tokenizer
    model_class = ModelRegistry.get_model_class(model_name)
    model = model_class.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load adapter weights if they exist
    adapter_path = Path(model_path) / "adapter.safetensors"
    if adapter_path.exists():
        model.load_weights(str(adapter_path))
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].numpy()
    input_tokens = mx.array(inputs)
    
    # Initialize cache
    max_seq_length = config["model"].get("max_seq_length", 4096)
    cache = Cache(
        max_seq_length=max_seq_length,
        num_layers=model.num_hidden_layers,
        num_heads=model.num_attention_heads,
        head_dim=model.head_dim,
    )
    
    # Generate tokens
    generated = []
    for i in range(max_tokens):
        logits = model(input_tokens, cache)
        next_token = sample_top_p(logits[:, -1, :], p=top_p, temp=temp)
        generated.append(next_token.item())
        input_tokens = mx.array([[next_token.item()]])
        cache.offset += 1
        
        # Stop if we hit the EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode and return
    return tokenizer.decode(generated, skip_special_tokens=True)


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Generate text with a LoRA fine-tuned model")
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model configuration YAML file",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the LoRA adapter weights",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, required=True,
                      help="Path to model configuration file")
    parser.add_argument("--adapter-path", type=str, required=True,
                      help="Path to adapter weights")
    parser.add_argument("--temp", type=float, default=0.7,
                      help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                      help="Top-p sampling parameter")
    parser.add_argument("--max-tokens", type=int, default=512,
                      help="Maximum number of tokens to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        # Verify numpy version
        import numpy as np
        np_version = np.__version__
        if np_version.startswith("2."):
            print("[WARNING] NumPy 2.x detected. Downgrading to 1.x for compatibility...")
            import subprocess
            subprocess.check_call(["pip", "install", "--quiet", "numpy<2.0.0"])
            import importlib
            importlib.reload(np)
        
        # Load and validate config
        with open(args.model_config) as f:
            config = yaml.safe_load(f)
            
        model_name = config["model"]["name"]
        print(f"[INFO] Loading model {model_name}...")
        
        try:
            # Load base model
            model_class = ModelRegistry.get_model_class(model_name)
            model = model_class.from_pretrained(model_name)
            
            # Load and map adapter weights
            if args.adapter_path:
                print(f"[INFO] Loading adapter weights from {args.adapter_path}")
                adapter_weights = np.load(args.adapter_path)
                
                # Map weight names based on model architecture
                weight_mapping = {
                    "microsoft/Phi-3.5-mini-instruct": {
                        "q_proj.lora.weight": "attention.q_proj.lora.weight",
                        "k_proj.lora.weight": "attention.k_proj.lora.weight",
                        "v_proj.lora.weight": "attention.v_proj.lora.weight"
                    },
                    "google/gemma-2-2b": {
                        "q_proj.lora.weight": "self_attn.q_proj.lora.weight",
                        "k_proj.lora.weight": "self_attn.k_proj.lora.weight",
                        "v_proj.lora.weight": "self_attn.v_proj.lora.weight"
                    }
                }
                
                # Get mapping for current model
                if model_name in weight_mapping:
                    mapped_weights = {}
                    for old_name, new_name in weight_mapping[model_name].items():
                        if old_name in adapter_weights:
                            mapped_weights[new_name] = adapter_weights[old_name]
                    
                    # Load mapped weights
                    model.load_weights(mapped_weights)
                else:
                    print(f"[WARNING] No weight mapping found for {model_name}")
                    model.load_weights(adapter_weights)
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Interactive chat loop...
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize model: {str(e)}")
            raise
            
    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    main()
