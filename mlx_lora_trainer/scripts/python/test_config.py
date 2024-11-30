#!/usr/bin/env python3
"""Test configuration utilities."""

import argparse
from pathlib import Path
import yaml
from rich.console import Console

from mlx_lora_trainer.models.phi import PhiModel
from mlx_lora_trainer.models.gemma import GemmaModel
from mlx_lora_trainer.models.qwen import QwenModel
from typing import Dict, List
from mlx_lora_trainer.utils.model_utils import (
    calculate_model_size,
    get_available_memory,
)

console = Console()

def validate_config(config_path: str) -> bool:
    """Validate model configuration file."""
    try:
        # Load and parse config
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check memory requirements
        model_size = calculate_model_size(config)
        available_memory = get_available_memory()
        
        if model_size * 1.5 > available_memory:  # 1.5x safety margin
            console.print(
                f"[red]Warning: Model requires {model_size * 1.5:.1f}GB, "
                f"but only {available_memory:.1f}GB available[/red]"
            )
            return False
        
        console.print(f"[green]Config validation passed: {config_path}[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Error validating config: {str(e)}[/red]")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config file")
    args = parser.parse_args()
    
    if not validate_config(args.config_path):
        exit(1)

if __name__ == "__main__":
    main() 