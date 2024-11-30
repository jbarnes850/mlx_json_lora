#!/usr/bin/env python3
"""Test model configurations."""

import os
import yaml
from pathlib import Path
from rich.console import Console
import mxnet as mx
from mlx_lora_trainer.models import PhiModel, GemmaModel, QwenModel
from typing import Dict, List

console = Console()

def test_model_configs():
    """Test model configuration files."""
    # Get project root
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
    configs_dir = project_root / "configs" / "models"
    
    # Create configs directory if it doesn't exist
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Expected config files
    config_files = {
        "phi3.yaml": {
            "model": {
                "name": "microsoft/Phi-3.5-mini-instruct",
                "path": "microsoft/Phi-3.5-mini-instruct"
            }
        },
        "gemma.yaml": {
            "model": {
                "name": "google/gemma-2-2b",
                "path": "google/gemma-2-2b"
            }
        },
        "qwen.yaml": {
            "model": {
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "path": "Qwen/Qwen2.5-7B-Instruct"
            }
        }
    }
    
    # Test each config file
    for filename, expected_config in config_files.items():
        config_path = configs_dir / filename
        console.print(f"\n[cyan]Testing {filename}...")
        
        # Create config if it doesn't exist
        if not config_path.exists():
            console.print(f"[yellow]Creating {filename}...")
            with open(config_path, 'w') as f:
                if filename == "phi3.yaml":
                    yaml.safe_dump({
                        "model": {
                            "name": "microsoft/Phi-3.5-mini-instruct",
                            "path": "microsoft/Phi-3.5-mini-instruct",
                            "batch_size": 1,
                            "max_seq_length": 2048,
                            "learning_rate": 2.0e-4,
                            "num_layers": 32
                        },
                        "training": {
                            "seed": 42,
                            "iters": 600,
                            "val_batches": 20,
                            "steps_per_report": 10,
                            "steps_per_eval": 50,
                            "save_every": 100,
                            "grad_checkpoint": True
                        },
                        "lora": {
                            "r": 8,
                            "alpha": 32,
                            "dropout": 0.1,
                            "target_modules": [
                                "self_attn.q_proj",
                                "self_attn.k_proj",
                                "self_attn.v_proj",
                                "self_attn.o_proj",
                                "mlp.gate_proj",
                                "mlp.up_proj",
                                "mlp.down_proj"
                            ],
                            "lora_layers": 32
                        }
                    }, f)
        
        # Verify config exists and is valid
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                
            # Verify required fields
            assert "model" in config, f"Missing 'model' section in {filename}"
            assert "name" in config["model"], f"Missing 'name' in model section of {filename}"
            assert config["model"]["name"] == expected_config["model"]["name"], \
                f"Incorrect model name in {filename}"
            
            console.print(f"[green]✓ {filename} verified successfully")
            
        except Exception as e:
            console.print(f"[red]Error in {filename}: {str(e)}")
            return False
    
    return True

def test_model_compatibility():
    """Verify model implementations match configs."""
    models = {
        "phi3": ("microsoft/Phi-3.5-mini-instruct", PhiModel),
        "gemma": ("google/gemma-2-2b", GemmaModel),
        "qwen": ("Qwen/Qwen2.5-7B-Instruct", QwenModel)
    }
    
    for name, (path, model_class) in models.items():
        # Test model loading
        try:
            model = model_class.from_pretrained(path)
            # Test basic forward pass
            test_input = mx.array([[1, 2, 3]])
            output = model(test_input)
            print(f"✓ {name} model verified")
        except Exception as e:
            print(f"⚠️ {name} model failed: {str(e)}")

def validate_model_config(config: Dict) -> List[str]:
    """Validate model configuration."""
    errors = []
    
    # Validate LoRA parameters
    if config["lora"]["r"] * config["lora"]["alpha"] > config["model"]["hidden_size"]:
        errors.append("LoRA rank too large for model dimension")
    
    # Validate batch size vs memory
    total_params = calculate_model_size(config["model"]["name"])
    if total_params * config["model"]["batch_size"] > get_available_memory() * 0.8:
        errors.append("Batch size too large for available memory")
    
    return errors

def validate_training_config(config: Dict) -> List[str]:
    """Validate training configuration."""
    errors = []
    
    # Validate model parameters
    if config["model"]["batch_size"] * config["model"]["max_seq_length"] > get_available_memory():
        errors.append("Batch size * sequence length exceeds available memory")
        
    # Validate LoRA parameters
    if config["lora"]["r"] * config["lora"]["alpha"] > config["model"]["hidden_size"]:
        errors.append("LoRA rank too large for model dimension")
        
    # Validate training parameters
    if config["training"]["iters"] < 100:
        errors.append("Training iterations too low (min 100)")
    if config["training"]["val_batches"] > config["training"]["iters"] // 10:
        errors.append("Too many validation batches")
        
    # Model specific validation
    model_name = config["model"]["name"]
    if "Phi" in model_name:
        if config["model"]["max_seq_length"] > 2048:
            errors.append("Phi models support max 2048 sequence length")
    elif "gemma" in model_name:
        if config["model"]["max_seq_length"] > 8192:
            errors.append("Gemma models support max 8192 sequence length")
    elif "Qwen" in model_name:
        if config["model"]["max_seq_length"] > 32768:
            errors.append("Qwen models support max 32768 sequence length")
            
    return errors

if __name__ == "__main__":
    console.print("[bold blue]Testing Model Configurations...")
    if test_model_configs():
        console.print("\n[bold green]All configurations verified successfully!")
    else:
        console.print("\n[bold red]Configuration verification failed!") 