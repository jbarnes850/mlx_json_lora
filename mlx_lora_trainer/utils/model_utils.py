"""Model utility functions."""

import psutil
from typing import Dict

def get_model_size(model_name: str) -> float:
    """Get model size in GB."""
    model_sizes = {
        "microsoft/Phi-3.5-mini-instruct": 3.8,
        "google/gemma-2-2b": 4.5,
        "Qwen/Qwen2.5-7B-Instruct": 14.0
    }
    return model_sizes.get(model_name, 0.0)

def calculate_model_size(config: Dict) -> float:
    """Calculate model size based on configuration."""
    base_size = get_model_size(config["model"]["name"])
    lora_size = (config["lora"]["r"] * config["model"]["num_layers"] * 0.001)
    return base_size + lora_size

def get_available_memory() -> float:
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3) 