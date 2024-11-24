#!/usr/bin/env python3
"""
Training logger for MLX LoRA fine-tuning.
Handles metrics tracking and visualization.
"""

import matplotlib.pyplot as plt
import json
from pathlib import Path
import yaml
from rich.console import Console
from rich.table import Table
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.console = Console()
        
        # Load model config
        self.load_model_config()
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'throughput': []
        }
        
        # Create log file
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def load_model_config(self):
        """Load model configuration."""
        with open("model_selection.yaml", "r") as f:
            model_selection = yaml.safe_load(f)
        
        with open("lora_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        self.model_key = model_selection["model_key"]
        self.model_config = config["models"][self.model_key]
        self.training_config = config
    
    def log_metrics(self, step, metrics):
        """Log training metrics."""
        # Update metrics
        self.metrics['train_loss'].append(metrics.get('train_loss', 0))
        self.metrics['val_loss'].append(metrics.get('val_loss', 0))
        self.metrics['learning_rate'].append(metrics.get('learning_rate', 0))
        self.metrics['throughput'].append(metrics.get('throughput', 0))
        
        # Create rich table
        table = Table(title=f"Training Metrics (Step {step})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
        
        # Print to console
        self.console.print(table)
        
        # Save to log file
        with open(self.log_file, "a") as f:
            metrics['step'] = step
            f.write(json.dumps(metrics) + "\n")
    
    def plot_metrics(self):
        """Plot training metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot training loss
        ax1.plot(self.metrics['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        
        # Plot validation loss
        ax2.plot(self.metrics['val_loss'])
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        
        # Plot learning rate
        ax3.plot(self.metrics['learning_rate'])
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('LR')
        
        # Plot throughput
        ax4.plot(self.metrics['throughput'])
        ax4.set_title('Throughput (tokens/sec)')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Tokens/sec')
        
        # Add model info
        plt.suptitle(f"Training Metrics for {self.model_config['name']}")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.log_dir / "training_metrics.png")
        plt.close()
    
    def save_final_metrics(self):
        """Save final training metrics."""
        final_metrics = {
            'model_name': self.model_config['name'],
            'batch_size': self.model_config['batch_size'],
            'final_train_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
            'final_val_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None,
            'average_throughput': sum(self.metrics['throughput']) / len(self.metrics['throughput']) if self.metrics['throughput'] else 0
        }
        
        with open(self.log_dir / "final_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        # Print final summary
        self.console.print("\n[bold green]Final Training Metrics:[/bold green]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in final_metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
        
        self.console.print(table)
