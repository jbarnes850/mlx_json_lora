"""End-to-end tests for MLX LoRA Trainer."""

import os
import json
import pytest
import mlx.core as mx
from pathlib import Path
from click.testing import CliRunner

from mlx_lora_trainer.scripts.python.cli import cli
from mlx_lora_trainer.model import ModelRegistry
from mlx_lora_trainer.inference import generate_text
from mlx_lora_trainer.utils import load_config


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create a tiny instruction dataset."""
    data = [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "instruction": "Write a haiku about spring.",
            "response": "Cherry blossoms bloom\nPetals dance in gentle breeze\nSpring awakens now"
        },
        {
            "instruction": "Explain what is 2+2.",
            "response": "2+2 equals 4. This is a basic addition problem where we combine two groups of two to get four."
        }
    ]
    
    # Create train/valid/test splits
    splits = ["train", "valid", "test"]
    for split in splits:
        data_path = tmp_path / f"{split}.jsonl"
        with open(data_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    
    return str(tmp_path)


@pytest.fixture
def model_configs(tmp_path):
    """Create test configurations for different models."""
    base_config = {
        "training": {
            "batch_size": 1,
            "learning_rate": 2.0e-4,
            "num_epochs": 1,
            "warmup_steps": 2,
            "grad_checkpoint": True,
            "max_seq_length": 512,
            "weight_decay": 0.01
        },
        "lora": {
            "r": 8,
            "alpha": 32,
            "dropout": 0.1
        }
    }
    
    # Phi-3 config
    phi_config = base_config.copy()
    phi_config["model"] = {
        "name": "microsoft/phi-3-mini",
        "class": "phi"
    }
    phi_config["lora"]["target_modules"] = [
        "model.layers.*.self_attn.qkv_proj",
        "model.layers.*.self_attn.o_proj",
        "model.layers.*.mlp.gate_up_proj",
        "model.layers.*.mlp.down_proj"
    ]
    
    # Gemma config
    gemma_config = base_config.copy()
    gemma_config["model"] = {
        "name": "google/gemma-2b",
        "class": "gemma"
    }
    gemma_config["lora"]["target_modules"] = [
        "model.layers.*.self_attn.q_proj",
        "model.layers.*.self_attn.k_proj",
        "model.layers.*.self_attn.v_proj",
        "model.layers.*.self_attn.o_proj",
        "model.layers.*.mlp.gate_proj",
        "model.layers.*.mlp.up_proj",
        "model.layers.*.mlp.down_proj"
    ]
    
    # Qwen config
    qwen_config = base_config.copy()
    qwen_config["model"] = {
        "name": "Qwen/Qwen2-1_8B",
        "class": "qwen"
    }
    qwen_config["lora"]["target_modules"] = [
        "model.layers.*.self_attn.q_proj",
        "model.layers.*.self_attn.k_proj",
        "model.layers.*.self_attn.v_proj",
        "model.layers.*.self_attn.o_proj",
        "model.layers.*.mlp.gate_proj",
        "model.layers.*.mlp.up_proj",
        "model.layers.*.mlp.down_proj"
    ]
    
    # Save configs
    models = {"phi": phi_config, "gemma": gemma_config, "qwen": qwen_config}
    config_paths = {}
    
    for name, config in models.items():
        path = tmp_path / f"{name}_config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        config_paths[name] = str(path)
    
    return config_paths


@pytest.mark.parametrize("model_type", ["phi", "gemma", "qwen"])
def test_e2e_training(runner, tiny_dataset, model_configs, model_type, tmp_path):
    """Test end-to-end training workflow for different models."""
    config_path = model_configs[model_type]
    output_dir = tmp_path / f"{model_type}_output"
    
    # Run training
    result = runner.invoke(cli, [
        "train",
        "--model-config", config_path,
        "--train-data", tiny_dataset,
        "--output-dir", str(output_dir),
        "--num-epochs", "1",
        "--batch-size", "1"
    ])
    
    assert result.exit_code == 0
    
    # Check output files
    assert (output_dir / "adapter_config.json").exists()
    assert (output_dir / "adapter_model.safetensors").exists()
    assert (output_dir / "training_args.json").exists()


@pytest.mark.parametrize("model_type", ["phi", "gemma", "qwen"])
def test_e2e_inference(runner, model_configs, model_type, tmp_path):
    """Test end-to-end inference workflow for different models."""
    # First train the model
    test_e2e_training(runner, tiny_dataset, model_configs, model_type, tmp_path)
    
    model_path = tmp_path / f"{model_type}_output"
    
    # Test generation
    result = runner.invoke(cli, [
        "generate",
        "--model-path", str(model_path),
        "--prompt", "What is the capital of Spain?",
        "--max-tokens", "50"
    ])
    
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0


def test_hf_dataset_training(runner, model_configs, tmp_path):
    """Test training with a Hugging Face dataset."""
    config_path = model_configs["phi"]  # Use Phi-3 for this test
    output_dir = tmp_path / "hf_output"
    
    # Run training with a small HF dataset
    result = runner.invoke(cli, [
        "train",
        "--model-config", config_path,
        "--hf-dataset", "mlx-community/tiny-shakespeare",
        "--output-dir", str(output_dir),
        "--num-epochs", "1",
        "--batch-size", "1"
    ])
    
    assert result.exit_code == 0
    assert (output_dir / "adapter_model.safetensors").exists()


def test_custom_dataset_training(runner, model_configs, tmp_path):
    """Test training with a custom dataset format."""
    # Create a custom dataset
    data = [
        {"text": "This is a custom training example."},
        {"text": "Another example for testing."}
    ]
    
    data_dir = tmp_path / "custom_data"
    data_dir.mkdir()
    
    for split in ["train", "valid", "test"]:
        with open(data_dir / f"{split}.jsonl", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    
    config_path = model_configs["phi"]
    output_dir = tmp_path / "custom_output"
    
    # Run training
    result = runner.invoke(cli, [
        "train",
        "--model-config", config_path,
        "--train-data", str(data_dir),
        "--output-dir", str(output_dir),
        "--num-epochs", "1",
        "--batch-size", "1"
    ])
    
    assert result.exit_code == 0
    assert (output_dir / "adapter_model.safetensors").exists()


def test_model_export_and_merge(runner, tiny_dataset, model_configs, tmp_path):
    """Test model export and weight merging functionality."""
    # First train a model
    test_e2e_training(runner, tiny_dataset, model_configs, "phi", tmp_path)
    
    model_path = tmp_path / "phi_output"
    merged_path = tmp_path / "merged_model"
    
    # Export and merge weights
    result = runner.invoke(cli, [
        "export",
        "--model-path", str(model_path),
        "--output-dir", str(merged_path),
        "--merge-weights"
    ])
    
    assert result.exit_code == 0
    assert (merged_path / "model.safetensors").exists()
    
    # Test generation with merged model
    result = runner.invoke(cli, [
        "generate",
        "--model-path", str(merged_path),
        "--prompt", "Test prompt",
        "--max-tokens", "20"
    ])
    
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0
