"""Inference utilities for MLX LoRA models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator, Tuple, Dict, Any
import json

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizer

from mlx_lora_trainer.registry import ModelRegistry
from mlx_lora_trainer.utils.utils import (
    load_config,
    generate,
    sample_top_p,
)

@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    model_name: str
    model_path: str
    adapter_path: Optional[str] = None
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

class InferenceEngine:
    """Engine for model inference."""
    
    def __init__(self, config: InferenceConfig):
        """Initialize inference engine."""
        self.config = config
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
    
    def _load_model(self) -> nn.Module:
        """Load model with optional LoRA weights."""
        model_cls = ModelRegistry.get_model_class(self.config.model_name)
        model = model_cls.from_pretrained(self.config.model_path)
        
        if self.config.adapter_path:
            weights = mx.load(self.config.adapter_path)
            model.load_adapter_weights(weights)
        
        return model
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer."""
        return AutoTokenizer.from_pretrained(self.config.model_path)
    
    def generate(self, prompt: str) -> Generator[Tuple[str, mx.array], None, None]:
        """Generate text from prompt."""
        return generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
