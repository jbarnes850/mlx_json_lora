"""MLX LoRA Trainer implementation."""

import os
import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Callable
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten
from transformers import AutoTokenizer
from rich.console import Console
from rich.progress import Progress
from .utils import setup_logging
from .model import ModelRegistry

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class TrainingArgs:
    """Training arguments for LoRA fine-tuning."""
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    num_epochs: int = field(default=3, metadata={"help": "Number of epochs to train for."})
    val_batches: int = field(default=25, metadata={"help": "Number of validation batches."})
    steps_per_report: int = field(default=10, metadata={"help": "Steps between loss reports."})
    steps_per_eval: int = field(default=200, metadata={"help": "Steps between validations."})
    steps_per_save: int = field(default=100, metadata={"help": "Steps between model saves."})
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})
    adapter_file: str = field(default="adapters.safetensors", metadata={"help": "Adapter weights path."})
    grad_checkpoint: bool = field(default=False, metadata={"help": "Use gradient checkpointing."})
    learning_rate: float = field(default=2e-4, metadata={"help": "Learning rate."})
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank."})
    lora_alpha: float = field(default=16.0, metadata={"help": "LoRA alpha."})
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"], 
                               metadata={"help": "Target modules for LoRA."})
    model_name: str = field(default="mistral-7b-instruct", metadata={"help": "Model name."})
    model_class: str = field(default="phi", metadata={"help": "Model class name."})

class LoraTrainer:
    """LoRA (Low-Rank Adaptation) trainer for MLX models."""
    
    def __init__(self, args: TrainingArgs):
        """Initialize the LoRA trainer."""
        self.args = args
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
    def add_lora_layers(self, rank: int = 8, alpha: float = 16.0) -> list:
        """Add LoRA layers to the model."""
        print("\n" + "="*80)
        print("Adding LoRA layers")
        print("="*80)
        print("\nTarget modules:", self.args.target_modules)
        print("-"*40)
        
        modified_layers = []
        
        # Directly call the model's add_lora_layers method
        modified_layers = self.model.add_lora_layers(rank=rank, alpha=alpha)
        
        print("\nModified layers:", modified_layers)
        print("-"*40 + "\n")
        
        if not modified_layers:
            raise ValueError("Failed to add any LoRA layers to the model")
        
        return modified_layers

    def _setup_model(self):
        """Set up the model with LoRA layers."""
        model_name = self.args.model_name
        console.print(f"\n[bold cyan]Loading model: {model_name}[/bold cyan]")
        
        # Load model and tokenizer
        model_cls = ModelRegistry.get_model_class(self.args.model_class)
        self.model = model_cls.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add LoRA layers
        self.add_lora_layers(rank=self.args.lora_rank, alpha=self.args.lora_alpha)
        
    def _setup_optimizer(self):
        """Set up the optimizer."""
        self.optimizer = optim.Adam(learning_rate=self.args.learning_rate)
    
    def load_adapter(self, adapter_path: str):
        """Load LoRA adapter weights."""
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter file not found: {adapter_path}")
        weights = mx.load(adapter_path)
        self.model.load_state_dict(weights)
    
    def save_adapter(self, adapter_path: str):
        """Save LoRA adapter weights."""
        os.makedirs(os.path.dirname(adapter_path), exist_ok=True)
        weights = self.model.state_dict()
        mx.save(adapter_path, weights)
    
    def train(self, train_data: str):
        """Train the model."""
        if self.model is None:
            self._setup_model()
        if self.optimizer is None:
            self._setup_optimizer()
            
        # Load training data
        with open(train_data) as f:
            dataset = [json.loads(line) for line in f]
            
        # Training loop
        for epoch in range(self.args.num_epochs):
            console.print(f"\n[bold]Epoch {epoch + 1}/{self.args.num_epochs}[/bold]")
            
            with Progress() as progress:
                task = progress.add_task("Training...", total=len(dataset))
                
                for batch in self._batch_data(dataset, self.args.batch_size):
                    loss = self._train_step(batch)
                    progress.update(task, advance=len(batch))
                    progress.print(f"Loss: {loss:.4f}")
    
    def _train_step(self, batch):
        """Perform a single training step."""
        def loss_fn(model):
            total_loss = 0
            for item in batch:
                inputs = self.tokenizer(item["prompt"], return_tensors="np")["input_ids"]
                outputs = self.tokenizer(item["completion"], return_tensors="np")["input_ids"]
                
                logits = model(mx.array(inputs))
                loss = nn.losses.cross_entropy(logits[:, :-1], mx.array(outputs[:, 1:]))
                total_loss += loss
            return total_loss / len(batch)
        
        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
        self.optimizer.update(self.model, grads)
        return loss.item()
    
    def _batch_data(self, data, batch_size):
        """Create batches from data."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
