"""Tests for the model module."""

import pytest
import mlx.core as mx
import torch.nn as nn
from mlx_lora_trainer.model import ModelRegistry, BaseModel, PhiModel, GemmaModel, QwenModel
from mlx_lora_trainer.lora import LoRALinear

def test_model_registry_singleton():
    """Test that ModelRegistry is a singleton."""
    registry1 = ModelRegistry()
    registry2 = ModelRegistry()
    assert registry1 is registry2, "ModelRegistry should be a singleton"

def test_model_registration():
    """Test model registration."""
    # Clear registry for testing
    ModelRegistry._models = {}
    
    # Register test models
    @ModelRegistry.register("test-model")
    class TestModel(BaseModel):
        pass
    
    # Check registration
    models = ModelRegistry.get_all_models()
    assert TestModel in models, "TestModel not found in registry"
    
    # Check model name mapping
    assert ModelRegistry.get_model_class("test-model") == TestModel

def test_builtin_models():
    """Test that built-in models are registered."""
    models = {
        "phi": PhiModel,
        "gemma": GemmaModel,
        "qwen": QwenModel
    }
    
    for name, model_cls in models.items():
        registered_cls = ModelRegistry.get_model_class(name)
        assert registered_cls == model_cls, f"{name} model not properly registered"
        assert issubclass(registered_cls, BaseModel), f"{name} model not subclass of BaseModel"

def test_invalid_model():
    """Test handling of invalid model names."""
    with pytest.raises(KeyError):
        ModelRegistry.get_model_class("nonexistent-model")

@pytest.fixture
def mock_config():
    """Create a mock model config."""
    class Config:
        def __init__(self):
            self.hidden_size = 32
            self.num_hidden_layers = 2
            self.intermediate_size = 64
    return Config()

def test_model_layer_setup(mock_config):
    """Test model layer setup."""
    from mlx_lora_trainer.model import Model
    
    model = Model(mock_config)
    assert len(model.layers) == mock_config.num_hidden_layers
    
    for layer in model.layers:
        # Check attention components
        assert 'self_attn' in layer
        assert isinstance(layer['self_attn']['qkv_proj'], nn.Linear)
        assert isinstance(layer['self_attn']['o_proj'], nn.Linear)
        
        # Check MLP components
        assert 'mlp' in layer
        assert isinstance(layer['mlp']['gate_up_proj'], nn.Linear)
        assert isinstance(layer['mlp']['down_proj'], nn.Linear)

def test_model_forward_pass(mock_config):
    """Test model forward pass."""
    from mlx_lora_trainer.model import Model
    
    model = Model(mock_config)
    batch_size = 2
    seq_length = 16
    
    # Create random input
    x = mx.random.normal((batch_size, seq_length, mock_config.hidden_size))
    
    # Forward pass
    output = model(x)
    assert output.shape == (batch_size, seq_length, mock_config.hidden_size)

def test_phi_model_lora_integration():
    """Test PhiModel LoRA layer integration."""
    model = PhiModel()
    
    # Mock the model initialization
    model.model = Model(mock_config())
    
    # Add LoRA layers
    rank = 8
    alpha = 16.0
    modified_layers = model.add_lora_layers(rank=rank, alpha=alpha)
    
    # Check that layers were modified
    assert len(modified_layers) > 0
    
    # Verify each modified layer
    for layer_path in modified_layers:
        parts = layer_path.split('.')
        current = model.model.layers[int(parts[2])]
        for part in parts[3:]:
            current = current[part]
        assert isinstance(current, LoRALinear)
        assert current.rank == rank
        assert current.alpha == alpha
