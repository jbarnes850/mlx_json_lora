"""Training metrics utilities."""

import mlx.core as mx
import mlx.nn as nn
from typing import Union, Dict

def compute_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm for the model parameters.
    
    Args:
        model: The neural network model
        
    Returns:
        Gradient norm as a float
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = mx.sum(p.grad * p.grad)
            total_norm += param_norm
            
    return float(mx.sqrt(total_norm)) 