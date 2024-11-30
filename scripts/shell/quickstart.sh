#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Version and constants
VERSION="0.1.0-beta"
REPO_URL="https://github.com/jbarnes850/mlx-lora-trainer"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${CYAN}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

check_version() {
    print_info "Checking for updates..."
    if ! command -v curl &> /dev/null; then
        print_info "curl not found, skipping version check"
        return
    fi
    
    latest=$(curl -s https://api.github.com/repos/jbarnes850/mlx-lora-trainer/releases/latest | grep tag_name | cut -d '"' -f 4)
    if [[ -n "$latest" && "$VERSION" != "$latest" ]]; then
        echo -e "${YELLOW}⚠️ New version available: $latest${NC}"
        echo -e "Update with: git pull origin main"
        echo
    fi
}

show_beta_notice() {
    echo -e "\n${YELLOW}⚠️ Beta Release${NC}"
    echo -e "• Report issues: ${REPO_URL}/issues"
    echo -e "• Join MLX Community: discord.gg/mlx-community"
    echo
}

print_logo() {
    echo -e "${BLUE}"
    echo "     __  __ _    __  __    _     ___  ____      _     "
    echo "    |  \/  | |   \ \/ /   | |   / _ \|  _ \    / \    "
    echo "    | |\/| | |    >  <    | |  | | | | |_) |  / _ \   "
    echo "    | |  | | |___ /  \    | |__| |_| |  _ <  / ___ \  "
    echo "    |_|  |_|_____/_/\_\   |_____\___/|_| \_\/_/   \_\ "
    echo -e "${NC}"
    echo -e "${CYAN}Version: ${VERSION}${NC}"
}

# Function to prepare quickstart dataset
prepare_dataset() {
    print_header "Preparing Dataset"
    
    # Create data directory
    mkdir -p "${PROJECT_ROOT}/data/quickstart"
    
    # Download small curated dataset based on model type
    python3 - << EOF
import json
from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress

console = Console()

def prepare_quickstart_dataset():
    # Load a small subset of the alpaca dataset
    console.print("[cyan]Downloading small training dataset...[/cyan]")
    
    with Progress() as progress:
        task = progress.add_task("Downloading...", total=1000)
        
        # Load a small subset for quick testing
        dataset = load_dataset(
            "mlx-community/alpaca-cleaned", 
            split="train[:1000]"
        )
        
        # Convert to our format
        formatted_data = []
        for item in dataset:
            formatted_data.append({
                "prompt": item["instruction"] + "\n" + (item["input"] or ""),
                "completion": item["output"]
            })
            progress.update(task, advance=1)
    
        # Save training data
        with open("${PROJECT_ROOT}/data/quickstart/train.jsonl", "w") as f:
            for item in formatted_data[:800]:  # 800 for training
                f.write(json.dumps(item) + "\n")
                
        # Save validation data
        with open("${PROJECT_ROOT}/data/quickstart/valid.jsonl", "w") as f:
            for item in formatted_data[800:]:  # 200 for validation
                f.write(json.dumps(item) + "\n")
    
    console.print("[green]✓ Dataset prepared successfully![/green]")
    return True

if not prepare_quickstart_dataset():
    exit(1)
EOF
}

# Run quickstart configuration
python3 - << EOF
from mlx_lora_trainer.model import ModelRegistry
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
import time

console = Console()

def show_welcome():
    """Display welcome message and framework purpose"""
    layout = Layout()
    layout.split_column(
        Layout(Panel.fit(
            "[bold cyan]🚀 MLX LoRA Trainer Quickstart[/bold cyan]\n"
            "The fastest way to fine-tune LLMs on Apple Silicon",
            border_style="cyan"
        )),
        Layout(Panel.fit(
            "This quickstart will help you:\n"
            "• Fine-tune a model in ~15-30 minutes\n"
            "• Learn the basics of local LLM training\n"
            "• Create your own custom AI assistant",
            border_style="blue"
        ))
    )
    console.print(layout)

def show_model_preview(config):
    """Show a quick example of the model's capabilities"""
    console.print(Panel.fit(
        "Let's see what the base model can do before fine-tuning",
        title="Model Preview",
        border_style="cyan"
    ))
    
    example_prompt = "Explain what machine learning is in one sentence."
    console.print(f"\n[bold]Example Prompt:[/bold] {example_prompt}")
    
    # Simulate quick inference (replace with actual inference when ready)
    console.print("[cyan]Running inference...[/cyan]")
    time.sleep(2)  # Simulate inference time
    console.print("[bold]Base Model Response:[/bold]\n" + 
                 "Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed.")

def show_training_preview():
    """Show what to expect during training"""
    table = Table(title="Training Process Overview", border_style="cyan")
    table.add_column("Step", style="cyan")
    table.add_column("Duration", style="green")
    table.add_column("Description", style="white")
    
    table.add_row("Setup", "1-2 min", "Preparing model and dataset")
    table.add_row("Training", "15-30 min", "Fine-tuning with live metrics")
    table.add_row("Validation", "2-3 min", "Testing model performance")
    table.add_row("Export", "1 min", "Saving trained model")
    
    console.print(table)

def show_next_steps():
    """Show what users can do after quickstart"""
    console.print(Panel.fit(
        "[bold]After training, you can:[/bold]\n"
        "1. Chat with your model using [cyan]./scripts/shell/chat.sh[/cyan]\n"
        "2. Export for Ollama using [cyan]./scripts/shell/export_ollama.sh[/cyan]\n"
        "3. Try the full tutorial: [cyan]./scripts/shell/run_tutorial.sh[/cyan]\n"
        "4. Join our community: [link]https://github.com/mlx-community[/link]",
        title="Next Steps",
        border_style="green"
    ))

def run_quickstart():
    show_welcome()
    
    console.print("\n[cyan]Analyzing your hardware...[/cyan]")
    
    # Get optimized configuration
    config = ModelRegistry.quickstart()
    if config is None:
        return False
        
    # Display configuration
    console.print("\n[bold green]✓ Optimal Configuration Found![/bold green]")
    
    # Show model info
    console.print(Panel.fit(
        f"[bold]Selected Model:[/bold] {config['model_name']}\n"
        f"[bold]Description:[/bold] {config['description']}\n"
        f"[bold]Batch Size:[/bold] {config['batch_size']}",
        title="Model Configuration",
        border_style="green"
    ))
    
    # Show hardware info
    hw = config['hardware']
    console.print(Panel.fit(
        f"• Device: {hw['device']}\n"
        f"• Memory: {hw['memory']:.1f}GB\n"
        f"• Processor: {hw['processor']}",
        title="Hardware Information",
        border_style="cyan"
    ))
    
    # Show model preview
    show_model_preview(config)
    
    # Show training preview
    show_training_preview()
    
    # Update config with dataset paths
    config["config"]["data"] = {
        "train_path": "data/quickstart/train.jsonl",
        "valid_path": "data/quickstart/valid.jsonl"
    }
    
    # Save configuration
    import yaml
    with open("quickstart_config.yaml", "w") as f:
        yaml.dump(config["config"], f)
    
    console.print("\n[bold green]Configuration saved to quickstart_config.yaml[/bold green]")
    
    # Show next steps
    show_next_steps()
    return True

if not run_quickstart():
    exit(1)
EOF

# Main execution
main() {
    clear
    print_logo
    show_beta_notice
    check_version
    
    print_header "MLX LoRA Trainer Quickstart"
    
    # Step 1: Prepare environment
    print_info "Setting up quickstart environment..."
    source "${PROJECT_ROOT}/scripts/shell/setup_test_env.sh" || {
        print_error "Environment setup failed"
        exit 1
    }
    
    # Step 2: Prepare dataset
    prepare_dataset || {
        print_error "Dataset preparation failed"
        exit 1
    }
    
    # Step 3: Generate configuration
    print_info "Generating optimal configuration..."
    
    # Check if quickstart was successful
    if [ $? -eq 0 ]; then
        print_success "Quickstart setup complete!"
        echo
        print_info "Ready to start training!"
        echo -e "Run this command to begin:"
        echo -e "${CYAN}./scripts/shell/train_lora.sh --config quickstart_config.yaml${NC}"
    else
        print_error "Quickstart setup failed"
        exit 1
    fi
}

main 