"""Model implementation."""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Any, Type, Optional, List
from transformers import AutoConfig, PretrainedConfig, AutoModelForCausalLM
import torch
from .lora import LoRALinear

class ModelRegistry:
    """Model registry for managing model classes."""
    
    _instance = None
    _models: Dict[str, Type["BaseModel"]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str):
        """Register a model class."""
        def decorator(model_cls: Type["BaseModel"]) -> Type["BaseModel"]:
            cls._models[name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def get_model_class(cls, name: str) -> Type["BaseModel"]:
        """Get a model class by name."""
        if name not in cls._models:
            raise KeyError(f"Model class {name} not found in registry.")
        return cls._models[name]
    
    @classmethod
    def get_all_models(cls) -> List[Type["BaseModel"]]:
        """Get all registered model classes."""
        return list(cls._models.values())

class BaseModel(nn.Module):
    """Base model class."""
    
    def __init__(self, config: PretrainedConfig):
        """Initialize model."""
        super().__init__()
        self.config = config
    
    def state_dict(self) -> Dict[str, Any]:
        """Get model state dict."""
        state = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (nn.Module, mx.array)):
                state[key] = value
            elif isinstance(value, dict):
                state[key] = {k: v for k, v in value.items() if isinstance(v, (nn.Module, mx.array))}
        return state
    
    def update(self, state_dict: Dict[str, Any]):
        """Update model from state dict."""
        for key, value in state_dict.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    target = getattr(self, key)
                    if isinstance(target, dict):
                        for k, v in value.items():
                            if k in target and hasattr(target[k], 'update'):
                                target[k].update(v)
                            else:
                                target[k] = v
                elif hasattr(value, 'update'):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
    
    def add_lora_layers(self, rank: int = 8, alpha: float = 16.0) -> List[str]:
        """Add LoRA layers to the model."""
        raise NotImplementedError("Subclasses must implement add_lora_layers")

@ModelRegistry.register("phi")
class PhiModel(BaseModel):
    """Phi model implementation."""
    
    def __init__(self, config: Optional[PretrainedConfig] = None):
        """Initialize Phi model."""
        if config is None:
            config = AutoConfig.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        super().__init__(config)
        
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Initialize layers dict
        self.layers = {}
        for i in range(config.num_hidden_layers):
            layer = {
                "ln_1": nn.LayerNorm(config.hidden_size),
                "ln_2": nn.LayerNorm(config.hidden_size),
                "self_attn": {
                    "qkv_proj": nn.Linear(config.hidden_size, 3 * config.hidden_size),
                    "o_proj": nn.Linear(config.hidden_size, config.hidden_size),
                },
                "mlp": {
                    "gate_up_proj": nn.Linear(config.hidden_size, 2 * config.intermediate_size),
                    "down_proj": nn.Linear(config.intermediate_size, config.hidden_size),
                },
            }
            self.layers[f"layers.{i}"] = layer
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size)
    
    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass."""
        B, T = input_ids.shape
        
        # Get embeddings
        h = self.wte(input_ids)
        pos = mx.arange(T, dtype=mx.int32)
        h = h + self.wpe(pos)
        
        # Transformer layers
        for i in range(self.num_hidden_layers):
            layer = self.layers[f"layers.{i}"]
            
            # Self attention
            ln1_out = layer["ln_1"](h)
            
            # QKV projection
            qkv = layer["self_attn"]["qkv_proj"](ln1_out)
            q, k, v = mx.split(qkv, 3, axis=-1)
            
            # Reshape for attention
            q = q.reshape(B, T, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
            
            # Attention
            scores = (q @ k.transpose(0, 1, 3, 2)) / mx.sqrt(float(q.shape[-1]))
            scores = mx.softmax(scores, axis=-1)
            
            # Combine and project
            attn_out = (scores @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
            attn_out = layer["self_attn"]["o_proj"](attn_out)
            
            # First residual
            h = h + attn_out
            
            # MLP
            ln2_out = layer["ln_2"](h)
            gate_up = layer["mlp"]["gate_up_proj"](ln2_out)
            gate, up = mx.split(gate_up, 2, axis=-1)
            
            # SwiGLU activation
            gate = mx.sigmoid(gate)
            h = h + layer["mlp"]["down_proj"](gate * up)
        
        # Final layer norm
        h = self.ln_f(h)
        
        # Project to vocab
        return h @ self.wte.weight.T
    
    def add_lora_layers(self, rank: int = 8, alpha: float = 16.0) -> List[str]:
        """Add LoRA layers to the model."""
        modified_layers = []
        
        for i in range(self.num_hidden_layers):
            layer = self.layers[f"layers.{i}"]
            
            # Add LoRA to attention
            qkv = layer["self_attn"]["qkv_proj"]
            o_proj = layer["self_attn"]["o_proj"]
            
            # Replace with LoRA layers
            layer["self_attn"]["qkv_proj"] = LoRALinear(
                qkv.weight.shape[1],
                qkv.weight.shape[0],
                rank=rank,
                alpha=alpha
            )
            layer["self_attn"]["o_proj"] = LoRALinear(
                o_proj.weight.shape[1],
                o_proj.weight.shape[0],
                rank=rank,
                alpha=alpha
            )
            
            # Add LoRA to MLP
            gate_up = layer["mlp"]["gate_up_proj"]
            down = layer["mlp"]["down_proj"]
            
            layer["mlp"]["gate_up_proj"] = LoRALinear(
                gate_up.weight.shape[1],
                gate_up.weight.shape[0],
                rank=rank,
                alpha=alpha
            )
            layer["mlp"]["down_proj"] = LoRALinear(
                down.weight.shape[1],
                down.weight.shape[0],
                rank=rank,
                alpha=alpha
            )
            
            modified_layers.extend([
                f"layers.{i}.self_attn.qkv_proj",
                f"layers.{i}.self_attn.o_proj",
                f"layers.{i}.mlp.gate_up_proj",
                f"layers.{i}.mlp.down_proj"
            ])
        
        return modified_layers
    
    @classmethod
    def from_pretrained(cls, model_name: str) -> "PhiModel":
        """Load pretrained model."""
        config = AutoConfig.from_pretrained(model_name)
        model = cls(config)
        
        print(f"\nLoading model: {model_name}")
        torch_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Convert weights to MLX format
        weights = {}
        for name, param in torch_model.state_dict().items():
            weights[name] = mx.array(param.detach().numpy())
        
        model.update(weights)
        return model

@ModelRegistry.register("gemma")
class GemmaModel(BaseModel):
    """Gemma model implementation."""
    
    def __init__(self, config: Optional[PretrainedConfig] = None):
        """Initialize Gemma model."""
        if config is None:
            config = AutoConfig.from_pretrained("google/gemma-2b")
        super().__init__(config)
        # TODO: Implement Gemma model
        raise NotImplementedError("Gemma model not yet implemented")

@ModelRegistry.register("qwen")
class QwenModel(BaseModel):
    """Qwen model implementation."""
    
    def __init__(self, config: Optional[PretrainedConfig] = None):
        """Initialize Qwen model."""
        if config is None:
            config = AutoConfig.from_pretrained("Qwen/Qwen-1_8B")
        super().__init__(config)
        # TODO: Implement Qwen model
        raise NotImplementedError("Qwen model not yet implemented")
