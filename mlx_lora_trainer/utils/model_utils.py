"""Model utility functions for MLX LoRA."""

import os
import sys
import platform
import psutil
from typing import Dict, Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

def get_model_size(model: nn.Module) -> float:
    """Get model size in GB."""
    total_params = 0
    for p in model.parameters():
        total_params += p.size
    return (total_params * 4) / (1024 ** 3)  # Convert bytes to GB

def calculate_model_size(config: Dict[str, Any]) -> float:
    """Calculate model size from config."""
    if "model" not in config:
        return 0.0
        
    model_config = config["model"]
    if "size" in model_config:
        return float(model_config["size"])
        
    # Estimate from architecture parameters
    hidden_size = model_config.get("hidden_size", 768)
    num_layers = model_config.get("num_layers", 12)
    vocab_size = model_config.get("vocab_size", 32000)
    
    # Basic size estimation formula
    params = (
        # Embeddings
        hidden_size * vocab_size +
        # Transformer layers
        num_layers * (
            # Self attention
            4 * hidden_size * hidden_size +
            # Feed forward
            8 * hidden_size * hidden_size
        )
    )
    
    return (params * 4) / (1024 ** 3)  # Convert to GB

def get_available_memory() -> float:
    """Get available system memory in GB."""
    if hasattr(mx.metal, "get_device_memory_info"):
        # MLX Metal backend
        mem_info = mx.metal.get_device_memory_info()
        return mem_info["free"] / (1024 ** 3)
    else:
        # Fallback to system memory
        return psutil.virtual_memory().available / (1024 ** 3)

def detect_hardware() -> Dict[str, Any]:
    """Detect and validate hardware configuration.
    
    Returns:
        Dict containing hardware information and compatibility:
        {
            "is_compatible": bool,
            "message": str,
            "device": str,
            "memory": float,
            "processor": str,
            "machine": str
        }
    """
    is_apple_silicon = platform.processor() == 'arm'
    total_memory = psutil.virtual_memory().total / (1024**3)
    
    # Check for Apple Silicon and minimum requirements
    if not is_apple_silicon:
        return {
            "is_compatible": False,
            "message": "MLX requires Apple Silicon (M1/M2/M3) hardware",
            "device": "cpu",
            "memory": total_memory
        }
    
    # Check minimum memory requirement (8GB)
    if total_memory < 8:
        return {
            "is_compatible": False,
            "message": f"Insufficient memory: {total_memory:.1f}GB (minimum 8GB required)",
            "device": "mps",
            "memory": total_memory
        }
    
    # Check MLX Metal support
    try:
        if not mx.metal.is_available():
            return {
                "is_compatible": False,
                "message": "Metal not available on this system",
                "device": "cpu",
                "memory": total_memory
            }
    except Exception:
        return {
            "is_compatible": False,
            "message": "MLX Metal support not available",
            "device": "cpu",
            "memory": total_memory
        }
    
    # All checks passed
    return {
        "is_compatible": True,
        "message": "Hardware compatible",
        "device": "mps",
        "memory": total_memory,
        "processor": platform.processor(),
        "machine": platform.machine()
    }

def validate_system_resources(
    required_memory: float,
    model_name: str
) -> Tuple[bool, Optional[str]]:
    """Validate system has required resources.
    
    Args:
        required_memory: Required memory in GB
        model_name: Name of the model
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    hw_info = detect_hardware()
    if not hw_info["is_compatible"]:
        return False, hw_info["message"]
    
    available_memory = get_available_memory()
    if available_memory < required_memory:
        return False, (
            f"Insufficient memory for {model_name}. "
            f"Need {required_memory:.1f}GB, have {available_memory:.1f}GB"
        )
    
    return True, None 