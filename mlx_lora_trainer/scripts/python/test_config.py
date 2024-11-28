#!/usr/bin/env python3
"""Test model configurations."""

import os
import yaml
from pathlib import Path
from rich.console import Console

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

if __name__ == "__main__":
    console.print("[bold blue]Testing Model Configurations...")
    if test_model_configs():
        console.print("\n[bold green]All configurations verified successfully!")
    else:
        console.print("\n[bold red]Configuration verification failed!") 