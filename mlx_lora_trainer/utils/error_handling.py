"""Error handling utilities for MLX LoRA."""

import psutil
from typing import Dict, Tuple, Optional
import sys

from .model_utils import get_model_size, get_available_memory

class MLXLoRAError(Exception):
    """Base exception class for MLX LoRA Trainer."""
    pass

class ModelConfigError(MLXLoRAError):
    """Raised when model configuration is invalid."""
    pass

class ResourceError(MLXLoRAError):
    """Raised when system resources are insufficient."""
    pass

def validate_model_config(config: Dict) -> Tuple[bool, Optional[str]]:
    """Validate model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check required sections
        required_sections = ["model", "lora", "training"]
        for section in required_sections:
            if section not in config:
                return False, f"Missing required section: {section}"
        
        # Validate model section
        model_required = ["name", "path", "batch_size", "max_seq_length"]
        for field in model_required:
            if field not in config["model"]:
                return False, f"Missing required model field: {field}"
        
        # Validate LoRA section
        lora_required = ["r", "alpha", "dropout", "target_modules"]
        for field in lora_required:
            if field not in config["lora"]:
                return False, f"Missing required LoRA field: {field}"
        
        # Validate training section
        training_required = ["seed", "iters", "steps_per_eval"]
        for field in training_required:
            if field not in config["training"]:
                return False, f"Missing required training field: {field}"
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_system_resources(config: Dict) -> Tuple[bool, Optional[str]]:
    """Validate system has required resources for training.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check Metal support on macOS
        if sys.platform == "darwin":
            try:
                import mlx.core as mx
                if not mx.metal.is_available():
                    return False, "Metal not available on this system"
            except ImportError:
                return False, "MLX not installed or Metal not supported"
        
        # Check memory requirements
        model_name = config["model"]["name"]
        model_size = get_model_size(model_name)
        available_memory = get_available_memory()
        
        required_memory = model_size * 1.5  # 1.5x safety margin
        if available_memory < required_memory:
            return False, (
                f"Insufficient memory for {model_name}. "
                f"Need {required_memory:.1f}GB, have {available_memory:.1f}GB"
            )
        
        return True, None
        
    except Exception as e:
        return False, f"Resource validation error: {str(e)}"