"""Test LoRA layer implementation."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx_lora_trainer.lora import LoRALinear

def test_lora_linear_init():
    """Test LoRA layer initialization."""
    in_features = 4
    out_features = 3
    rank = 2
    alpha = 16.0
    
    layer = LoRALinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        alpha=alpha
    )
    
    # Check dimensions and parameters
    assert layer.weight.shape == (out_features, in_features)
    assert layer.rank == rank
    assert layer.alpha == alpha
    assert layer.scaling == alpha / rank
    assert not layer.merged
    
    # Check LoRA matrices
    assert layer.lora_A.shape == (rank, in_features)
    assert layer.lora_B.shape == (out_features, rank)
    
    # Check initialization scale
    assert mx.mean(mx.abs(layer.lora_A)) < 1.0
    assert mx.mean(mx.abs(layer.lora_B)) < 1.0

def test_lora_linear_forward():
    """Test LoRA layer forward pass."""
    in_features = 4
    out_features = 3
    rank = 2
    alpha = 16.0
    
    layer = LoRALinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        alpha=alpha
    )
    
    # Test forward pass
    x = mx.random.normal((2, in_features))
    y = layer(x)
    
    # Check output shape
    assert y.shape == (2, out_features)
    
    # Verify output is different from base layer
    base_out = x @ layer.weight.T
    assert not mx.allclose(y, base_out)

def test_lora_linear_merge_and_unmerge():
    """Test LoRA weight merging and unmerging."""
    in_features = 4
    out_features = 3
    rank = 2
    
    layer = LoRALinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank
    )
    
    # Save original weights
    original_weight = mx.array(layer.weight)
    
    # Test merging
    layer.merge_weights()
    assert layer.merged
    assert not mx.array_equal(layer.weight, original_weight)
    
    # Test unmerging
    layer.unmerge_weights()
    assert not layer.merged
    assert mx.allclose(layer.weight, original_weight)

def test_lora_linear_zero_rank():
    """Test LoRA layer with rank=0 (no LoRA)."""
    in_features = 4
    out_features = 3
    rank = 0
    
    # Create LoRA layer with rank=0
    mx.random.seed(42)
    layer = LoRALinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank
    )
    
    # Check that LoRA matrices are None
    assert layer.lora_A is None
    assert layer.lora_B is None
    
    # Forward pass should be identical to base layer
    x = mx.random.normal((2, in_features))
    y1 = layer(x)
    y2 = x @ layer.weight.T
    if layer.bias is not None:
        y2 = y2 + layer.bias
    
    assert mx.allclose(y1, y2)

def test_lora_linear_state():
    """Test LoRA layer state management."""
    in_features = 4
    out_features = 3
    rank = 2
    
    layer = LoRALinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank
    )
    
    # Get state dict
    state = layer.state_dict()
    
    # Check state contents
    assert "weight" in state
    assert "merged" in state
    assert "lora_A" in state
    assert "lora_B" in state
    assert "scaling" in state
    
    # Create new layer and update with state
    new_layer = LoRALinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank
    )
    new_layer.update(state)
    
    # Check if states match
    assert mx.allclose(layer.weight, new_layer.weight)
    assert mx.allclose(layer.lora_A, new_layer.lora_A)
    assert mx.allclose(layer.lora_B, new_layer.lora_B)
    assert layer.merged == new_layer.merged
    assert layer.scaling == new_layer.scaling
