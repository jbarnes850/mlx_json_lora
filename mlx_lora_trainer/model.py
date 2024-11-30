"""Model implementation."""

import os
import sys
from pathlib import Path
import yaml
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Any, Optional, Type, Tuple, List
from rich.console import Console
import transformers

from mlx_lora_trainer.utils.model_utils import (
    get_available_memory,
    detect_hardware,
)

class BaseModel(nn.Module):
    """Base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass of the model."""
        raise NotImplementedError
        
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "BaseModel":
        """Load a pretrained model."""
        raise NotImplementedError
        
    def save_pretrained(self, save_path: str) -> None:
        """Save model weights."""
        raise NotImplementedError
        
    def prepare_for_training(self) -> None:
        """Prepare model for training."""
        pass
