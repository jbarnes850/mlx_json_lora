import psutil
from typing import Dict
from mlx_lora_trainer.utils import get_model_size

class MLXLoRAError(Exception):
    """Base exception class for MLX LoRA Trainer."""
    pass

class ModelConfigError(MLXLoRAError):
    """Raised for model configuration issues."""
    pass

class ResourceError(MLXLoRAError):
    """Raised for resource-related issues."""
    pass

def validate_system_resources(config: Dict):
    """Validate system resources before training."""
    memory_gb = psutil.virtual_memory().total / 1024**3
    model_size = get_model_size(config["model"]["name"])
    
    if model_size * 1.5 > memory_gb:  # 1.5x for safety margin
        raise ResourceError(
            f"Insufficient memory for model. Need {model_size * 1.5:.1f}GB, "
            f"have {memory_gb:.1f}GB"
        ) 