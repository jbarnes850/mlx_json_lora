"""Tests for the inference module."""

import os
import pytest
import mlx.core as mx

from mlx_lora_trainer.model import ModelRegistry, BaseModel
from mlx_lora_trainer.inference import generate_text, Cache, sample_top_p
from mlx_lora_trainer.utils import load_config


def test_model_registry():
    """Test that the model registry works correctly."""
    # Get all registered models
    models = ModelRegistry.get_all_models()
    assert len(models) > 0, "No models registered"
    
    # Check that each model is properly registered
    for model_cls in models:
        assert issubclass(model_cls, BaseModel), f"{model_cls.__name__} is not a subclass of BaseModel"


def test_config_loading():
    """Test that model configs can be loaded correctly."""
    # Test with Phi-3 config
    config_path = os.path.join("configs", "models", "phi3.yaml")
    config = load_config(config_path)
    
    # Check required sections
    assert "model" in config, "Model section missing from config"
    assert "training" in config, "Training section missing from config"
    assert "lora" in config, "LoRA section missing from config"
    
    # Check model section
    assert "name" in config["model"], "Model name missing"
    assert "path" in config["model"], "Model path missing"


def test_cache_initialization():
    """Test cache initialization and update."""
    cache = Cache(
        max_seq_length=512,
        num_layers=4,
        num_heads=8,
        head_dim=64
    )
    
    assert cache.max_seq_length == 512
    assert cache.num_layers == 4
    assert cache.num_heads == 8
    assert cache.head_dim == 64
    assert len(cache.keys) == 4
    assert len(cache.values) == 4
    assert cache.offset == 0


def test_cache_update():
    """Test cache update functionality."""
    cache = Cache(
        max_seq_length=512,
        num_layers=4,
        num_heads=8,
        head_dim=64
    )
    
    # Create test data - shape: (batch, heads, seq_len, head_dim)
    key = mx.random.normal((1, 8, 1, 64))
    value = mx.random.normal((1, 8, 1, 64))
    
    # Test initial update
    k, v = cache.update_and_fetch(0, key, value)
    assert mx.array_equal(k, key)
    assert mx.array_equal(v, value)
    
    # Test concatenation - should append along sequence length dimension
    k2, v2 = cache.update_and_fetch(0, key, value)
    assert k2.shape == (1, 8, 2, 64)
    assert v2.shape == (1, 8, 2, 64)


def test_sampling():
    """Test top-p sampling."""
    logits = mx.random.normal((1, 100))
    sampled = sample_top_p(logits, p=0.9, temp=0.7)
    assert isinstance(sampled, mx.array)
    assert sampled.shape == ()  # Single token


@pytest.mark.skip(reason="Requires model weights")
def test_text_generation():
    """Test text generation with a model."""
    response = generate_text(
        "What is 2+2?",
        model_path="outputs/test_model",
        max_tokens=50
    )
    assert isinstance(response, str)
    assert len(response) > 0
