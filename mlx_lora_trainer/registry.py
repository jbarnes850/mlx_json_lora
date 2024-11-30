"""Model registry for MLX LoRA."""

from typing import Dict, Type, Set, Optional, Any, Tuple
from pathlib import Path
import yaml
from rich.console import Console

from mlx_lora_trainer.model import BaseModel
from mlx_lora_trainer.utils.model_utils import (
    get_available_memory,
    detect_hardware,
)

class ModelRegistry:
    """Registry for model architectures."""
    
    _instance = None
    _registry: Dict[str, Type[BaseModel]] = {}
    _console = Console()
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, model_name: str):
        """Register a model class."""
        def decorator(model_cls):
            if not issubclass(model_cls, BaseModel):
                raise TypeError(f"Model class must inherit from BaseModel")
            cls._registry[model_name] = model_cls
            cls._console.print(f"[green]Registered model: {model_name}[/green]")
            return model_cls
        return decorator
    
    @classmethod
    def get_model_class(cls, model_name: str) -> Type[BaseModel]:
        """Get model class by name."""
        if model_name not in cls._registry:
            raise KeyError(f"Model {model_name} not found in registry")
        return cls._registry[model_name]
    
    @classmethod
    def get_all_models(cls) -> Set[str]:
        """Get all registered model names."""
        return set(cls._registry.keys())
        
    @classmethod
    def select_model(cls) -> Dict[str, Any]:
        """Select appropriate model based on hardware."""
        hw_info = detect_hardware()
        if not hw_info["is_compatible"]:
            cls._console.print(f"[red]Error: {hw_info['message']}[/red]")
            return {}
            
        config_path = Path(__file__).parent.parent / "configs" / "models" / "model_selection.yaml"
        try:
            with open(config_path) as f:
                selection_config = yaml.safe_load(f)
        except Exception as e:
            cls._console.print(f"[red]Error loading model selection config: {str(e)}[/red]")
            return {}
            
        available_memory = get_available_memory()
        
        for model_name, model_info in selection_config["models"].items():
            if model_info["min_memory"] <= available_memory:
                return {
                    "name": model_info["name"],
                    "description": model_info["description"],
                    "memory_required": model_info["min_memory"],
                    "max_batch_size": model_info["max_batch_size"]
                }
        
        cls._console.print("[red]Error: No suitable model found for your hardware[/red]")
        return {}
        
    @classmethod
    def validate_environment(cls) -> Tuple[bool, Optional[str]]:
        """Validate environment for training."""
        hw_info = detect_hardware()
        if not hw_info["is_compatible"]:
            return False, hw_info["message"]
        return True, None
