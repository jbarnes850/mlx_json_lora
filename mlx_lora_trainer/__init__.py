"""MLX LoRA Trainer package."""

from .inference import generate_text
from .model import ModelRegistry
from .trainer import LoraTrainer, TrainingArgs
from .utils import setup_logging, ensure_dir, load_config

__version__ = "0.1.0"
__all__ = [
    "generate_text",
    "ModelRegistry",
    "setup_logging",
    "ensure_dir",
    "load_config",
    "LoraTrainer",
    "TrainingArgs",
]
