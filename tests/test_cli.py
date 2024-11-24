"""Tests for the CLI interface."""

import json
import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from mlx_lora_trainer.scripts.python.cli import cli


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def test_data(tmp_path):
    """Create a tiny test dataset."""
    data = [
        {"instruction": "Say hello", "response": "Hello! How can I help you today?"},
        {"instruction": "Count to 3", "response": "1, 2, 3"},
    ]
    
    data_path = tmp_path / "test_data.jsonl"
    with open(data_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    return str(data_path)


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration."""
    config = {
        "model": {
            "name": "microsoft/phi-3-mini-4k-instruct",
            "class": "phi"
        },
        "training": {
            "batch_size": 1,
            "learning_rate": 2.0e-4,
            "num_epochs": 1,
            "warmup_steps": 100,
            "grad_checkpoint": True
        },
        "lora": {
            "r": 8,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "model.layers.*.self_attn.qkv_proj",
                "model.layers.*.self_attn.o_proj",
                "model.layers.*.mlp.gate_up_proj",
                "model.layers.*.mlp.down_proj"
            ]
        }
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return str(config_path)


def test_cli_help(runner):
    """Test the CLI help command."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "MLX LoRA Fine-Tuning Toolkit CLI" in result.output
    assert "train" in result.output
    assert "generate" in result.output


def test_train_command_help(runner):
    """Test the train command help."""
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--model-config" in result.output
    assert "--train-data" in result.output
    assert "--output-dir" in result.output


def test_generate_command_help(runner):
    """Test the generate command help."""
    result = runner.invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--model-path" in result.output
    assert "--prompt" in result.output


def test_train_command_validation(runner):
    """Test validation of training command arguments."""
    result = runner.invoke(cli, ["train"])
    assert result.exit_code != 0
    assert "Missing option" in result.output
    
    result = runner.invoke(cli, ["train", "--model-config", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert "Missing option '--train-data'" in result.output


@pytest.mark.slow
def test_train_command(runner, test_data, test_config, tmp_path):
    """Test the training command with minimal data."""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)
    
    result = runner.invoke(cli, [
        "train",
        "--model-config", test_config,
        "--train-data", test_data,
        "--output-dir", str(output_dir)
    ])
    
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(output_dir, "adapter.safetensors"))
    assert os.path.exists(os.path.join(output_dir, "config.yaml"))


def test_generate_command_validation(runner):
    """Test validation of generation command arguments."""
    result = runner.invoke(cli, ["generate"])
    assert result.exit_code != 0
    assert "Missing option" in result.output
    
    result = runner.invoke(cli, ["generate", "--model-path", "nonexistent"])
    assert result.exit_code != 0
    assert "Missing option '--prompt'" in result.output


@pytest.mark.slow
def test_generate_command(runner, tmp_path):
    """Test the generate command with a mock model."""
    # Create mock model directory with required files
    model_dir = tmp_path / "model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create mock adapter file
    with open(model_dir / "adapter.safetensors", "w") as f:
        f.write("{}")  # Empty safetensors file for testing
    
    config = {
        "model": {"name": "microsoft/phi-2"},
        "lora": {"r": 8, "alpha": 16}
    }
    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    result = runner.invoke(cli, [
        "generate",
        "--model-path", str(model_dir),
        "--prompt", "Hello",
        "--max-tokens", "10",
        "--temp", "0.7",
        "--top-p", "0.9"
    ])
    
    # We expect this to fail since we don't have a real model,
    # but it should fail with a model-related error
    assert result.exit_code != 0
    assert "model" in str(result.exception).lower()


def test_shell_script_compatibility(runner, test_config, test_data, tmp_path):
    """Test compatibility with shell scripts."""
    # Test train_lora.sh compatibility
    output_dir = tmp_path / "train_output"
    os.makedirs(output_dir, exist_ok=True)
    
    train_result = runner.invoke(cli, [
        "train",
        "--model-config", test_config,
        "--train-data", test_data,
        "--output-dir", str(output_dir)
    ])
    
    # Expect training to succeed
    assert train_result.exit_code == 0
    assert os.path.exists(os.path.join(output_dir, "adapter.safetensors"))
    
    # Test run_inference.sh compatibility
    generate_result = runner.invoke(cli, [
        "generate",
        "--model-path", str(output_dir),
        "--prompt", "Test prompt",
        "--max-tokens", "10"
    ])
    
    # Should fail with model-related error since we don't have a real model
    assert generate_result.exit_code != 0
    assert "model" in str(generate_result.exception).lower()
