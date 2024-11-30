"""LoRA layer implementation for MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any

class LoRALinear(nn.Module):
    """LoRA adaptation layer that inherits from MLX.
    
    This layer implements Low-Rank Adaptation (LoRA) as described in:
    "LoRA: Low-Rank Adaptation of Large Language Models" (https://arxiv.org/abs/2106.09685)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
        device: Optional[str] = None,
        dtype: Any = None,
    ):
        """Initialize LoRA layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Rank of LoRA adaptation matrices (if 0, no LoRA is applied)
            alpha: Alpha scaling factor
            bias: Whether to include bias term
            device: Device to place tensors on
            dtype: Data type of tensors
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.merged = False
        
        # Initialize base weights like nn.Linear
        k = 1.0 / (in_features ** 0.5)
        self.weight = mx.random.uniform(
            low=-k,
            high=k,
            shape=(out_features, in_features)
        )
        
        self.bias = None
        if bias:
            self.bias = mx.zeros((out_features,))
        
        # Only create LoRA matrices if rank > 0
        if rank > 0:
            # Initialize LoRA matrices following paper
            # A initialized with normal distribution
            self.lora_A = mx.random.normal(
                shape=(rank, in_features)
            ) * (1.0 / rank)
            
            # B initialized with zeros
            self.lora_B = mx.random.normal(
                shape=(out_features, rank)
            ) * (1.0 / rank)
        else:
            self.lora_A = None
            self.lora_B = None
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass combining base Linear layer with LoRA adaptation.
        
        Args:
            x: Input tensor of shape (*, in_features)
            
        Returns:
            Output tensor of shape (*, out_features)
        """
        # Base linear transformation
        out = x @ self.weight.T
        
        # Add LoRA contribution if enabled and not merged
        if self.rank > 0 and not self.merged and self.lora_A is not None and self.lora_B is not None:
            # Compute LoRA path: x -> A -> B with scaling
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
            out = out + lora_out
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def merge_weights(self):
        """Merge LoRA weights into the base weights for inference."""
        if self.rank > 0 and not self.merged and self.lora_A is not None and self.lora_B is not None:
            # Compute LoRA contribution: B @ A
            delta = (self.lora_B @ self.lora_A) * self.scaling
            # Add to base weights
            self.weight = mx.array(self.weight + delta)
            self.merged = True
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from base weights to enable training."""
        if self.rank > 0 and self.merged and self.lora_A is not None and self.lora_B is not None:
            # Compute LoRA contribution: B @ A
            delta = (self.lora_B @ self.lora_A) * self.scaling
            # Subtract from base weights
            self.weight = mx.array(self.weight - delta)
            self.merged = False
    
    def parameters(self):
        """Get trainable parameters."""
        params = {"weight": self.weight}
        if self.bias is not None:
            params["bias"] = self.bias
        if self.rank > 0:
            if self.lora_A is not None:
                params["lora_A"] = self.lora_A
            if self.lora_B is not None:
                params["lora_B"] = self.lora_B
        return params
    
    def state_dict(self) -> Dict[str, Any]:
        """Get layer state for serialization."""
        state = self.parameters()
        state["merged"] = self.merged
        state["scaling"] = self.scaling
        return state
    
    def update(self, state_dict: Dict[str, Any]):
        """Update layer state from state dictionary."""
        if "weight" in state_dict:
            self.weight = state_dict["weight"]
        if "bias" in state_dict and self.bias is not None:
            self.bias = state_dict["bias"]
        if self.rank > 0:
            if "lora_A" in state_dict and "lora_B" in state_dict:
                self.lora_A = state_dict["lora_A"]
                self.lora_B = state_dict["lora_B"]
        if "merged" in state_dict:
            self.merged = state_dict["merged"]
        if "scaling" in state_dict:
            self.scaling = state_dict["scaling"]
