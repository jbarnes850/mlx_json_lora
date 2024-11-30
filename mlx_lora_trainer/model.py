"""Model implementation."""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Any, Type, Optional, List
from transformers import AutoConfig, PretrainedConfig, AutoModelForCausalLM
from .models.phi import PhiModel
from .models.gemma import GemmaModel
from .models.qwen import QwenModel

class ModelRegistry:
    """Model registry for managing model classes."""
    
    _registry: Dict[str, Type] = {}
    _model_paths = {
        "microsoft/Phi-3.5-mini-instruct": PhiModel,
        "google/gemma-2-2b": GemmaModel,
        "Qwen/Qwen2.5-7B-Instruct": QwenModel,
    }

    @classmethod
    def register(cls, name: str, model_class: Type[nn.Module]) -> None:
        """Register a model class."""
        if not issubclass(model_class, nn.Module):
            raise TypeError(f"Model class must inherit from nn.Module")
        if name in cls._registry:
            raise ValueError(f"Model {name} already registered")
        cls._registry[name] = model_class
        
    @classmethod
    def get_model_class(cls, name: str) -> Type:
        """Get model class by name or path."""
        # Try direct lookup first
        if name in cls._registry:
            return cls._registry[name]
        
        # Try looking up by model path
        if name in cls._model_paths:
            return cls._model_paths[name]
            
        raise KeyError(f"Model {name} not found in registry. Available models: {list(cls._registry.keys())}")

# Register models
ModelRegistry.register("phi-3", PhiModel)
ModelRegistry.register("gemma", GemmaModel)
ModelRegistry.register("qwen", QwenModel)
