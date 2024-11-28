#!/usr/bin/env python3
"""Script to download models from Hugging Face."""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
import os

console = Console()

def download_model(model_name: str, show_progress: bool = True) -> None:
    """Download model from Hugging Face."""
    try:
        # Setup cache directory
        cache_dir = Path(".cache") / model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model-specific configurations and correct paths
        model_configs = {
            "microsoft/Phi-3.5-mini-instruct": {
                "trust_remote_code": True,
                "revision": "main",
                "use_safetensors": True
            },
            "google/gemma-2-2b": {
                "trust_remote_code": True,
                "revision": "main",
                "use_safetensors": True
            },
            "Qwen/Qwen2.5-7B-Instruct": {
                "trust_remote_code": True,
                "revision": "main",
                "use_safetensors": True
            }
        }
        
        # Get model-specific config
        model_config = model_configs.get(model_name, {
            "trust_remote_code": True,
            "revision": "main",
            "use_safetensors": True
        })
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(f"[cyan]Downloading {model_name}...", total=None)
                
                try:
                    # Simplified download configuration with only supported parameters
                    download_config = {
                        "max_workers": 16,          # Increased for more parallelism
                        "resume_download": True,     # Resume interrupted downloads
                        "force_download": False,
                        "local_files_only": False,
                        "local_dir": str(cache_dir),
                        "ignore_patterns": ["*.md", "*.txt", "*.ipynb"],
                        "token": os.getenv("HF_TOKEN"),  # Use token if available
                    }
                    
                    # First try to download config to verify model exists
                    progress.update(task, description=f"[cyan]Verifying model...")
                    config = AutoConfig.from_pretrained(
                        model_name,
                        **model_config,
                        local_files_only=False
                    )
                    
                    # Download tokenizer with caching
                    progress.update(task, description=f"[cyan]Downloading tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=str(cache_dir),
                        **model_config,
                        local_files_only=False,
                        use_fast=True  # Use fast tokenizer implementation
                    )
                    
                    # Download full model with optimized settings
                    progress.update(task, description=f"[cyan]Downloading model files...")
                    snapshot_download(
                        repo_id=model_name,
                        **download_config,
                        **{k: v for k, v in model_config.items() 
                           if k not in ["trust_remote_code", "use_safetensors"]}
                    )
                    
                    progress.update(task, description="[green]Download complete!")
                
                except Exception as e:
                    console.print(f"[red]Error during download: {str(e)}")
                    raise
        else:
            # Download without progress bar but with same optimizations
            download_config = {
                "max_workers": 16,
                "resume_download": True,
                "force_download": False,
                "local_files_only": False,
                "local_dir": str(cache_dir),
                "ignore_patterns": ["*.md", "*.txt", "*.ipynb"]
            }
            
            AutoConfig.from_pretrained(model_name, **model_config)
            AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=str(cache_dir),
                **model_config,
                use_fast=True
            )
            snapshot_download(
                repo_id=model_name,
                **download_config,
                **{k: v for k, v in model_config.items() 
                   if k not in ["trust_remote_code", "use_safetensors"]}
            )
        
        # Verify downloaded files
        if not (cache_dir / "config.json").exists():
            raise RuntimeError(f"Model configuration not found in {cache_dir}")
            
        console.print(f"[green]✓ Model downloaded successfully to {cache_dir}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model {model_name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Download model from Hugging Face")
    parser.add_argument("--model", type=str, required=True, help="Model name on Hugging Face")
    parser.add_argument("--show-progress", action="store_true", help="Show download progress")
    args = parser.parse_args()
    
    try:
        download_model(args.model, args.show_progress)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 