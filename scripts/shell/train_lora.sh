#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Define standard paths
DATA_DIR="${PROJECT_ROOT}/data"
CONFIG_DIR="${PROJECT_ROOT}/configs/models"
ADAPTER_DIR="${PROJECT_ROOT}/adapters"
LOG_DIR="${PROJECT_ROOT}/logs"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'
DIM='\033[2m'

# Print functions with enhanced formatting
print_header() {
    echo -e "\n${BOLD}${BLUE}==> $1${NC}"
}

print_step() {
    echo -e "\n${BOLD}${CYAN}==> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 could not be found. Please install it first."
        exit 1
    fi
}

# Function to show progress spinner
show_spinner() {
    local pid=$1
    local message=$2
    local spin='-\|/'
    local i=0
    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) %4 ))
        printf "\r${CYAN}%s... %s${NC}" "$message" "${spin:$i:1}"
        sleep .1
    done
    printf "\r"
}

# Function to check dependencies
check_dependencies() {
    local missing_deps=()
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    else
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if (( $(echo "$python_version < 3.9" | bc -l) )); then
            print_error "Python 3.9 or later is required (found $python_version)"
            return 1
        fi
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("pip3")
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    # Report missing dependencies
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        echo -e "${CYAN}Please install the missing dependencies and try again${NC}"
        return 1
    fi
    
    return 0
}

# Function to setup virtual environment
setup_venv() {
    print_header "Setting up Python Environment"
    print_info "Creating a virtual environment for isolated package management..."
    
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        if [ $? -ne 0 ]; then
            print_error "Failed to create virtual environment"
            exit 1
        fi
        print_success "Created virtual environment"
    else
        print_info "Using existing virtual environment"
    fi
    
    source .venv/bin/activate
    if [ $? -ne 0 ]; then
        print_error "Failed to activate virtual environment"
        exit 1
    fi
    print_success "Activated virtual environment"
}

# Function to install dependencies
install_dependencies() {
    print_header "Installing Required Packages"
    print_info "This may take a few minutes..."
    
    # List of required packages
    local packages=(
        "torch"
        "transformers"
        "sentencepiece"
        "datasets"
        "rich"
        "pyyaml"
        "mlx-lm"
    )
    
    # Upgrade pip first
    python -m pip install --upgrade pip
    
    # Install packages
    for package in "${packages[@]}"; do
        python -m pip install "$package"
    done
    
    # Install local package in development mode
    print_info "Installing MLX LoRA Trainer..."
    python -m pip install -e .
    
    print_success "All packages installed successfully"
}

# Function to select model
select_model() {
    print_header "Model Selection"
    echo -e "${CYAN}Choose a model to fine-tune:${NC}"
    echo
    echo -e "${BOLD}1) Phi-3-mini (1.3B parameters)${NC}"
    echo -e "   ${CYAN}• Fastest training speed"
    echo -e "   ${CYAN}• Good for testing and small datasets"
    echo -e "   ${CYAN}• Requires ~8GB memory${NC}"
    echo
    echo -e "${BOLD}2) Gemma-2B${NC}"
    echo -e "   ${CYAN}• Balanced performance"
    echo -e "   ${CYAN}• Good for medium-sized datasets"
    echo -e "   ${CYAN}• Requires ~12GB memory${NC}"
    echo
    echo -e "${BOLD}3) Qwen2.5-4B${NC}"
    echo -e "   ${CYAN}• Best quality results"
    echo -e "   ${CYAN}• Good for production use"
    echo -e "   ${CYAN}• Requires ~16GB memory${NC}"
    echo
    
    read -p "Enter your choice (1-3): " choice
    echo
    
    case $choice in
        1|2|3)
            generate_config $choice
            ;;
        *)
            print_warning "Invalid choice. Using default (Phi-3-mini)"
            generate_config 1
            ;;
    esac
}

# Function to prepare data
prepare_data() {
    print_header "Preparing Dataset"
    
    # Create data directory if it doesn't exist
    mkdir -p "${DATA_DIR}"
    
    # Check if dataset exists
    if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
        print_info "Downloading dataset..."
        (
            python3 - << EOF
import json
import requests
from tqdm import tqdm

url = "https://raw.githubusercontent.com/example/dataset/main/train.jsonl"
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open("${DATA_DIR}/train.jsonl", "wb") as f:
    for data in tqdm(response.iter_content(chunk_size=1024), 
                    total=total_size//1024, 
                    unit='KB',
                    desc="Downloading dataset"):
        f.write(data)
EOF
        ) &
        show_spinner $! "Downloading dataset"
        
        if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
            print_error "Failed to download dataset"
            return 1
        fi
        print_success "Dataset downloaded successfully"
    else
        print_info "Using existing dataset"
    fi
}

# Function to generate configuration file
generate_config() {
    local choice=$1
    local model=${models[$choice]}
    local config_file="${CONFIG_DIR}/lora_config.yaml"
    
    print_step "Generating configuration for $model..."
    
    # Get model-specific settings
    local model_settings=${model_configs[$choice]}
    
    cat > "$config_file" << EOL
# Model and training configuration
model: "$model"
train: true
data: "${DATA_DIR}"
adapter_path: "${ADAPTER_DIR}"

# Memory optimization settings
memory:
  grad_checkpoint: true      # Enable MLX gradient checkpointing (trades computation for memory)
  checkpoint_layers: "all"   # Apply checkpointing to all transformer layers
  optimize_memory: true      # Enable MLX memory optimizations
  min_batch_size: 1         # Minimum batch size for adaptive batching
  max_batch_size: 8         # Maximum batch size for adaptive batching
  adaptive_batch_size: true  # Automatically adjust batch size based on memory

# Training configuration
training:
  batch_size: ${batch_size}  # Will be replaced with model-specific value
  max_seq_length: ${max_seq_length}
  learning_rate: ${learning_rate}
  num_layers: ${num_layers}

# LoRA configuration
lora:
  r: 8                    # LoRA rank
  alpha: 32               # LoRA alpha scaling
  dropout: 0.1           # LoRA dropout
  target_modules:        # Layers to apply LoRA
    - "q_proj"           # Query projection
    - "k_proj"           # Key projection
    - "v_proj"           # Value projection
    - "o_proj"           # Output projection
EOL

    if [ $? -ne 0 ]; then
        print_error "Failed to generate configuration file"
        exit 1
    fi
    print_success "Generated configuration file with memory optimizations"
    
    # Print memory optimization tips
    print_info "Memory Optimization Tips:"
    print_info "• Gradient checkpointing enabled (trades speed for memory)"
    print_info "• Memory efficient attention activated"
    print_info "• Adaptive batch sizing based on available memory"
    print_info "• Optimized number of LoRA layers for your hardware"
}

# Function to run training
run_training() {
    print_header "Starting Model Training"
    
    # Validate model configuration
    if [ ! -f "${CONFIG_DIR}/config.yaml" ]; then
        print_error "Model configuration not found at: ${CONFIG_DIR}/config.yaml"
        return 1
    fi
    
    # Validate dataset
    if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
        print_error "Training data not found at: ${DATA_DIR}/train.jsonl"
        return 1
    fi
    
    # Create output directories
    mkdir -p "${ADAPTER_DIR}" "${LOG_DIR}"
    
    print_info "Training configuration:"
    echo -e "• Model: $(get_model_description $MODEL_CHOICE)"
    echo -e "• Config: ${CONFIG_DIR}/config.yaml"
    echo -e "• Dataset: ${DATA_DIR}/train.jsonl"
    echo -e "• Output: ${ADAPTER_DIR}/lora_weights.npz"
    echo
    
    # Start training
    print_info "Starting training process..."
    echo -e "${CYAN}Training can be safely interrupted with Ctrl+C${NC}"
    echo -e "${CYAN}Progress will be saved automatically${NC}"
    echo
    
    CUDA_VISIBLE_DEVICES="" python3 "${PROJECT_ROOT}/mlx_lora_trainer/scripts/python/cli.py" train \
        --model-config "${CONFIG_DIR}/config.yaml" \
        --train-data "${DATA_DIR}/train.jsonl" \
        --output-dir "${ADAPTER_DIR}" \
        --log-dir "${LOG_DIR}" \
        --save-every 100 2>&1 | tee "${LOG_DIR}/training.log" || {
            print_error "Training failed. Check ${LOG_DIR}/training.log for details"
            return 1
        }
    
    print_success "Training completed successfully!"
    print_info "Model weights saved to: ${ADAPTER_DIR}/lora_weights.npz"
    print_info "Training log saved to: ${LOG_DIR}/training.log"
    return 0
}

# Function to verify environment
verify_environment() {
    print_header "Verifying Environment"
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python version: $python_version"
    
    # Check available memory
    if [[ "$OSTYPE" == "darwin"* ]]; then
        memory=$(sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}')
        print_success "Available memory: $memory"
    fi
    
    # Check MLX installation
    pip show mlx > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        mlx_version=$(pip show mlx | grep Version | awk '{print $2}')
        print_success "MLX version: $mlx_version"
    else
        print_error "MLX not found"
        exit 1
    fi
}

# Main execution
main() {
    # Print welcome message
    clear
    print_header "MLX LoRA Fine-tuning"
    echo -e "${CYAN}Fine-tune large language models efficiently on Apple Silicon${NC}"
    echo
    
    # Verify environment
    verify_environment || exit 1
    
    # Check dependencies
    check_dependencies || exit 1
    
    # Select model
    select_model || exit 1
    
    # Prepare data
    prepare_data || exit 1
    
    # Run training
    run_training || exit 1
    
    print_success "Process completed successfully!"
    echo
    echo -e "${CYAN}Next steps:${NC}"
    echo -e "1. Try your model: ./scripts/shell/chat.sh"
    echo -e "2. Export model:   ./scripts/shell/export.sh"
    echo
    
    return 0
}

# Run main function
main
