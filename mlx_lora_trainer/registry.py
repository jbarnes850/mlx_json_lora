"""Model registry for MLX LoRA."""

from typing import Dict, Type, Set


class ModelRegistry:
    """Registry for model architectures."""
    
    _instance = None
    _registry: Dict[str, Type["BaseModel"]] = {}
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, model_name: str):
        """Register a model class."""
        def decorator(model_cls):
            cls._registry[model_name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def get_model_class(cls, model_name: str) -> Type["BaseModel"]:
        """Get model class by name."""
        if model_name not in cls._registry:
            raise KeyError(f"Model {model_name} not found in registry")
        return cls._registry[model_name]
    
    @classmethod
    def get_all_models(cls) -> Set[Type["BaseModel"]]:
        """Get all registered model classes."""
        return set(cls._registry.values())
    
    @classmethod
    def clear_registry(cls):
        """Clear the registry (for testing)."""
        cls._registry.clear()
