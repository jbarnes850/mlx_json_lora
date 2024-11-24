#!/usr/bin/env python3
"""
MLX LoRA Fine-Tuning Toolkit CLI
A comprehensive command-line interface for the entire fine-tuning workflow.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import yaml
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.progress import Progress
from rich.tree import Tree
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from datetime import datetime
import psutil
from datasets import load_dataset
from transformers import AutoTokenizer
from mlx_lora_trainer import LoraTrainer, TrainingArgs, generate_text
import mlx.core as mx
import csv

console = Console()

def load_config(config_path: str = "configs/models/lora_config.yaml"):
    """Load model and training configuration."""
    try:
        # Convert to absolute path if relative
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_root, config_path)
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        if "model_key" not in config:
            raise ValueError("model_key not found in config")
        if "models" not in config:
            raise ValueError("models section not found in config")
        if config["model_key"] not in config["models"]:
            raise ValueError(f"Model {config['model_key']} not found in models section")
            
        return config
    except Exception as e:
        console.print(f"[bold red]Error loading config: {str(e)}[/bold red]")
        sys.exit(1)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["models"][config["model_key"]]["name"])
    
    def process_squad(self):
        """Process SQuAD dataset."""
        console.print("[bold green]Loading SQuAD dataset...[/bold green]")
        dataset = load_dataset("squad_v2")
        
        def create_completion(context, question, answer):
            answer_text = answer["text"][0] if len(answer["text"]) > 0 else "I Don't Know"
            return f"Given this context: {context}\n\nAnswer this question: {question}\n\nAnswer: {answer_text}"
        
        def process_split(split_data):
            processed = []
            with Progress() as progress:
                task = progress.add_task(f"Processing {split_data.split} split...", total=len(split_data))
                for sample in split_data:
                    completion = create_completion(sample['context'], sample['question'], sample['answers'])
                    processed.append({"prompt": sample['question'], "completion": completion})
                    progress.advance(task)
            return processed
        
        # Process splits
        train_data = process_split(dataset['train'])
        valid_data = process_split(dataset['validation'])
        
        # Save data
        os.makedirs("data", exist_ok=True)
        for split, data in [("train", train_data), ("valid", valid_data)]:
            path = Path("data") / f"{split}.jsonl"
            with open(path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            console.print(f"[green]Saved {len(data)} samples to {path}[/green]")
    
    def process_custom(self, input_file: str):
        """Process custom dataset."""
        console.print(f"[bold green]Processing custom dataset from {input_file}...[/bold green]")
        
        # Determine input format
        if input_file.endswith('.jsonl'):
            with open(input_file) as f:
                data = [json.loads(line) for line in f]
        else:
            with open(input_file) as f:
                data = json.load(f)
        
        # Process and validate
        processed = []
        with Progress() as progress:
            task = progress.add_task("Processing data...", total=len(data))
            for item in data:
                if isinstance(item, dict) and "prompt" in item and "completion" in item:
                    processed.append(item)
                elif isinstance(item, dict) and "messages" in item:
                    # Handle chat format
                    messages = item["messages"]
                    if len(messages) >= 2:
                        prompt = messages[0]["content"]
                        completion = messages[1]["content"]
                        processed.append({"prompt": prompt, "completion": completion})
                progress.advance(task)
        
        # Save processed data
        os.makedirs("data", exist_ok=True)
        output_path = Path("data") / "train.jsonl"
        with open(output_path, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")
        console.print(f"[green]Saved {len(processed)} samples to {output_path}[/green]")

def load_hf_dataset(dataset_name: str, split: str = "train") -> list:
    """Load and format a dataset from Hugging Face.
    
    Example datasets:
    - mlx-community/wikisql (SQL generation)
    - mlx-community/helpful-base (General instruction following)
    - mlx-community/code-help (Code assistance)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]Please install datasets: pip install datasets[/red]")
        return []
    
    console.print(f"[cyan]Loading {dataset_name} ({split} split)...[/cyan]")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        
        # Auto-detect format based on column names
        columns = dataset.column_names
        
        if "messages" in columns:
            # Chat format
            return [{"messages": msg} for msg in dataset["messages"]]
            
        elif "prompt" in columns and "completion" in columns:
            # Completion format
            return [{"prompt": p, "completion": c} 
                   for p, c in zip(dataset["prompt"], dataset["completion"])]
            
        elif "question" in columns and "answer" in columns:
            # QA format -> convert to completion format
            return [{"prompt": q, "completion": a} 
                   for q, a in zip(dataset["question"], dataset["answer"])]
            
        elif "text" in columns:
            # Text format
            return [{"text": t} for t in dataset["text"]]
            
        else:
            console.print(f"[yellow]Warning: Unknown dataset format. Columns: {columns}[/yellow]")
            console.print("[yellow]Please specify format mapping in config.yaml:[/yellow]")
            console.print("""
hf_dataset:
  name: "{dataset_name}"
  prompt_feature: "column_for_prompt"
  completion_feature: "column_for_completion"
            """)
            return []
            
    except Exception as e:
        console.print(f"[red]Error loading dataset: {str(e)}[/red]")
        return []

def prepare_data(args):
    """Prepare training data from various sources."""
    if args.dataset and args.dataset.startswith(("mlx-community/", "huggingface/")):
        # Load from Hugging Face
        console.print(f"[bold green]Loading dataset from Hugging Face: {args.dataset}[/bold green]")
        
        train_data = load_hf_dataset(args.dataset, "train")
        valid_data = load_hf_dataset(args.dataset, "validation")
        
        if not train_data:
            console.print("[red]Failed to load training data[/red]")
            return False
            
        # Save datasets
        os.makedirs("data", exist_ok=True)
        
        with open("data/train.jsonl", "w") as f:
            for item in train_data:
                json.dump(item, f)
                f.write("\n")
                
        if valid_data:
            with open("data/valid.jsonl", "w") as f:
                for item in valid_data:
                    json.dump(item, f)
                    f.write("\n")
        
        console.print(f"[green]✓ Saved {len(train_data)} training examples[/green]")
        if valid_data:
            console.print(f"[green]✓ Saved {len(valid_data)} validation examples[/green]")
            
        return True
        
    else:
        # Handle local data preparation
        return format_data(args.input, args.output, args.type, args.split_ratio)

def format_data(input_file: str, output_file: str, format_type: str, split_ratio: float = 0.2):
    """Format input data to JSONL format for MLX fine-tuning.
    
    Supports formats:
    - chat: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    - tools: Function calling format with tool definitions
    - completions: {"prompt": "...", "completion": "..."}
    - text: {"text": "..."}
    - qa: Q&A format
    - csv: CSV file
    """
    
    formats = {
        "chat": format_chat_data,
        "tools": format_tools_data,
        "completions": format_completions_data,
        "text": format_text_data,
        "qa": format_qa_data,
        "csv": format_csv_data,
    }
    
    if format_type not in formats:
        raise ValueError(f"Unsupported format: {format_type}. Must be one of {list(formats.keys())}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Format the data
    data = formats[format_type](input_file)
    
    # Split into train/valid if requested
    if split_ratio > 0:
        train_size = int(len(data) * (1 - split_ratio))
        train_data = data[:train_size]
        valid_data = data[train_size:]
        
        # Save train data
        with open(output_file, 'w') as f:
            for item in train_data:
                json.dump(item, f)
                f.write('\n')
        
        # Save validation data
        valid_file = output_file.replace('train.jsonl', 'valid.jsonl')
        with open(valid_file, 'w') as f:
            for item in valid_data:
                json.dump(item, f)
                f.write('\n')
    else:
        # Save all data to output file
        with open(output_file, 'w') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')

def format_chat_data(input_file: str) -> list:
    """Format chat data with system, user, and assistant messages."""
    data = []
    current_messages = []
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_messages:
                    data.append({"messages": current_messages})
                    current_messages = []
                continue
            
            if line.startswith("system:"):
                current_messages.append({
                    "role": "system",
                    "content": line[7:].strip()
                })
            elif line.startswith("user:"):
                current_messages.append({
                    "role": "user",
                    "content": line[5:].strip()
                })
            elif line.startswith("assistant:"):
                current_messages.append({
                    "role": "assistant",
                    "content": line[10:].strip()
                })
    
    if current_messages:
        data.append({"messages": current_messages})
    
    return data

def format_tools_data(input_file: str) -> list:
    """Format function calling data with tool definitions."""
    data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    continue
                    
                messages = []
                if "user_input" in entry:
                    messages.append({
                        "role": "user",
                        "content": entry["user_input"]
                    })
                
                if "tool_calls" in entry:
                    messages.append({
                        "role": "assistant",
                        "tool_calls": entry["tool_calls"]
                    })
                
                if messages and "tools" in entry:
                    data.append({
                        "messages": messages,
                        "tools": entry["tools"]
                    })
            except json.JSONDecodeError:
                continue
    
    return data

def format_completions_data(input_file: str) -> list:
    """Format completion data."""
    data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    continue
                    
                if "prompt" in entry and "completion" in entry:
                    data.append({
                        "prompt": entry["prompt"],
                        "completion": entry["completion"]
                    })
            except json.JSONDecodeError:
                continue
    
    return data

def format_text_data(input_file: str) -> list:
    """Format text data."""
    data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    continue
                    
                if "text" in entry:
                    data.append({
                        "text": entry["text"]
                    })
            except json.JSONDecodeError:
                continue
    
    return data

def format_qa_data(input_file: str) -> list:
    """Format Q&A data."""
    data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    continue
                    
                if "question" in entry and "answer" in entry:
                    data.append({
                        "prompt": entry["question"],
                        "completion": entry["answer"]
                    })
            except json.JSONDecodeError:
                continue
    
    return data

def format_csv_data(input_file: str) -> list:
    """Format CSV data."""
    data = []
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        for row in reader:
            if len(row) == len(headers):
                entry = dict(zip(headers, row))
                if "prompt" in entry and "completion" in entry:
                    data.append({
                        "prompt": entry["prompt"],
                        "completion": entry["completion"]
                    })
    
    return data

class ModelHandler:
    def __init__(self, config):
        self.config = config
        model_name = config["models"][config["model_key"]]["name"]
        console.print(f"[bold green]Loading {model_name}...[/bold green]")
        self.model, self.tokenizer = None, None
    
    def load_for_inference(self, model_path: str, adapter_path: str = None):
        """Load model for inference."""
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
    
    def generate_response(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate response for a prompt."""
        return generate(
            prompt=prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            max_tokens=max_tokens
        )
    
    def chat_session(self):
        """Run interactive chat session."""
        console.print("[bold green]Starting chat session (Ctrl+C to exit)...[/bold green]")
        history = []
        
        try:
            while True:
                prompt = console.input("\n[bold cyan]You:[/bold cyan] ")
                response = self.generate_response(prompt)
                
                console.print("\n[bold cyan]Model:[/bold cyan]", style="green")
                console.print(Markdown(response))
                
                history.append({"user": prompt, "model": response})
        except KeyboardInterrupt:
            console.print("\n[bold green]Chat session ended.[/bold green]")
            
            # Save history
            history_file = "chat_history.json"
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)
            console.print(f"[green]Chat history saved to {history_file}[/green]")
    
    def process_file(self, input_file: str, output_file: str):
        """Process prompts from file."""
        # Load prompts
        with open(input_file) as f:
            if input_file.endswith('.jsonl'):
                prompts = [json.loads(line)["prompt"] for line in f]
            else:
                prompts = json.load(f)
        
        # Generate responses
        responses = []
        with Progress() as progress:
            task = progress.add_task("Generating responses...", total=len(prompts))
            for prompt in prompts:
                response = self.generate_response(prompt)
                responses.append({"prompt": prompt, "response": response})
                progress.advance(task)
        
        # Save results
        with open(output_file, "w") as f:
            json.dump(responses, f, indent=2)
        console.print(f"[green]Results saved to {output_file}[/green]")

class TrainingDashboard:
    def __init__(self):
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header"),
            Layout(name="main"),
            Layout(name="footer")
        )
        self.layout["main"].split_row(
            Layout(name="stats"),
            Layout(name="chart")
        )
        self.start_time = datetime.now()
        
    def generate_header(self, model_name, batch_size):
        return Panel(f"🚀 Training {model_name} | Batch Size: {batch_size}", style="bold blue")
        
    def generate_stats(self, metrics):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        for k, v in metrics.items():
            table.add_row(k, str(v))
        return Panel(table, title="📊 Training Stats", border_style="green")
        
    def generate_chart(self, loss_history):
        # ASCII chart of loss history
        if not loss_history:
            return Panel("No data yet", title="📈 Loss Chart")
        max_width = 40
        max_height = 10
        chart = self._create_ascii_chart(loss_history, max_width, max_height)
        return Panel(chart, title="📈 Loss History")
        
    def _create_ascii_chart(self, data, width, height):
        if not data:
            return ""
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val or 1
        
        points = []
        step = len(data) / width if len(data) > width else 1
        for i in range(width):
            idx = int(i * step)
            if idx < len(data):
                normalized = (data[idx] - min_val) / range_val
                y = int((height - 1) * (1 - normalized))
                points.append((i, y))
        
        chart = []
        for y in range(height):
            line = ""
            for x in range(width):
                if any(p[0] == x and p[1] == y for p in points):
                    line += "⬤"
                else:
                    line += "·"
            chart.append(line)
        return "\n".join(chart)
        
    def update(self, metrics, loss_history):
        self.layout["header"].update(self.generate_header(
            metrics.get("model_name", "Unknown"),
            metrics.get("batch_size", 0)
        ))
        self.layout["stats"].update(self.generate_stats(metrics))
        self.layout["chart"].update(self.generate_chart(loss_history))
        self.layout["footer"].update(Panel(
            f"⏱️ Elapsed: {datetime.now() - self.start_time} | 💾 Memory: {metrics.get('memory_used', '0')}GB",
            style="bold cyan"
        ))
        
    def __rich__(self):
        return self.layout

def train_with_dashboard(model, config):
    dashboard = TrainingDashboard()
    metrics = {}
    loss_history = []
    
    with Live(dashboard, refresh_per_second=4):
        for step, batch in enumerate(training_loop()):
            # Update metrics
            metrics.update({
                "model_name": config["model"],
                "batch_size": config["batch_size"],
                "step": step,
                "loss": batch["loss"],
                "learning_rate": batch["lr"],
                "memory_used": mx.metal.get_peak_memory() / 1e9,
                "tokens/sec": batch["tokens_per_second"]
            })
            loss_history.append(batch["loss"])
            
            # Update dashboard
            dashboard.update(metrics, loss_history)
            
            # Save checkpoints
            if step % config["save_every"] == 0:
                save_checkpoint(model, step)

class ConfigWizard:
    def __init__(self):
        self.console = Console()
        
    def create_tree(self, data, tree=None):
        if tree is None:
            tree = Tree("📋 Configuration Overview")
        for key, value in data.items():
            if isinstance(value, dict):
                branch = tree.add(f"[bold cyan]{key}[/]")
                self.create_tree(value, branch)
            else:
                tree.add(f"[cyan]{key}:[/] [green]{value}[/]")
        return tree
        
    def run(self):
        self.console.print("\n[bold blue]🧙‍♂️ Configuration Wizard[/]\n")
        
        # Get hardware info
        with open('.hardware_config', 'r') as f:
            hardware = json.load(f)
        
        config = {
            "model": {
                "name": Prompt.ask(
                    "Select model",
                    choices=["phi-3-mini", "gemma-2b", "qwen2.5-4b"],
                    default=hardware["recommended_model"]
                ),
                "batch_size": int(Prompt.ask(
                    "Batch size",
                    default=str(hardware["batch_size"])
                ))
            },
            "training": {
                "epochs": int(Prompt.ask("Number of epochs", default="3")),
                "learning_rate": float(Prompt.ask(
                    "Learning rate",
                    default="2e-4"
                ))
            },
            "memory": {
                "grad_checkpoint": Confirm.ask(
                    "Enable gradient checkpointing?",
                    default=True
                ),
                "adaptive_batch": Confirm.ask(
                    "Enable adaptive batch sizing?",
                    default=True
                )
            }
        }
        
        # Show configuration summary
        self.console.print("\n[bold green]Configuration Summary:[/]")
        self.console.print(self.create_tree(config))
        
        # Confirm configuration
        if Confirm.ask("\nProceed with this configuration?", default=True):
            return config
        else:
            return self.run()  # Restart wizard
            
    def validate_config(self, config):
        """Validate and optimize configuration based on hardware."""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        warnings = []
        optimizations = []
        
        # Check memory requirements
        if config["model"]["name"] == "qwen2.5-4b" and memory_gb < 16:
            warnings.append("⚠️ Qwen2.5-4B requires 16GB+ RAM")
            optimizations.append("Enabling gradient checkpointing")
            config["memory"]["grad_checkpoint"] = True
            
        # Adjust batch size if needed
        if config["model"]["batch_size"] * get_model_size(config["model"]["name"]) > memory_gb * 0.7:
            old_batch = config["model"]["batch_size"]
            config["model"]["batch_size"] = max(1, old_batch // 2)
            warnings.append(f"⚠️ Reduced batch size from {old_batch} to {config['model']['batch_size']}")
            
        return config, warnings, optimizations

def setup_with_wizard():
    wizard = ConfigWizard()
    config = wizard.run()
    config, warnings, optimizations = wizard.validate_config(config)
    
    # Show warnings and optimizations
    if warnings:
        console.print("\n[yellow]Warnings:[/]")
        for warning in warnings:
            console.print(f"[yellow]• {warning}[/]")
            
    if optimizations:
        console.print("\n[green]Optimizations Applied:[/]")
        for opt in optimizations:
            console.print(f"[green]• {opt}[/]")
            
    return config

@click.group()
def cli():
    """MLX LoRA Fine-Tuning Toolkit CLI."""
    pass

@cli.command()
@click.option("--model-config", required=True, help="Path to model configuration file")
@click.option("--train-data", required=True, help="Path to training data")
@click.option("--output-dir", required=True, help="Directory to save model outputs")
def train(model_config: str, train_data: str, output_dir: str):
    """Train a model with LoRA."""
    # Load config
    with open(model_config) as f:
        config = yaml.safe_load(f)
    
    # Create training args
    args = TrainingArgs(
        model_name=config["model"]["name"],
        model_class=config["model"]["class"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        num_epochs=config["training"]["num_epochs"],
        lora_rank=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
    )
    
    # Initialize trainer
    trainer = LoraTrainer(args)
    
    # Train model
    trainer.train(train_data)
    
    # Save adapter
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_adapter(os.path.join(output_dir, "adapter.safetensors"))
    
    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

@cli.command()
@click.option("--model-path", required=True, help="Path to model directory")
@click.option("--prompt", required=True, help="Input prompt")
@click.option("--max-tokens", default=100, help="Maximum tokens to generate")
@click.option("--temp", default=0.7, help="Temperature for sampling")
@click.option("--top-p", default=0.9, help="Top-p sampling parameter")
def generate(model_path: str, prompt: str, max_tokens: int, temp: float, top_p: float):
    """Generate text with a fine-tuned model."""
    response = generate_text(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p
    )
    console.print(response)

if __name__ == "__main__":
    cli()
