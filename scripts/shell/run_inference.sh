#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Print functions
print_header() {
    echo -e "\n${BOLD}${BLUE}==> $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to check if adapter exists
check_adapter() {
    if [ ! -f "$1" ]; then
        print_error "Adapter weights not found at: $1"
        print_info "Please ensure you have trained a model first using train_lora.sh"
        exit 1
    fi
}

# Main execution
main() {
    print_header "MLX LoRA Inference"
    echo -e "${CYAN}Starting chat session with your fine-tuned model...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to exit the chat session${NC}\n"

    # Change to project root
    cd "${PROJECT_ROOT}"

    # Check for adapter weights
    ADAPTER_PATH="adapters/lora_weights.npz"
    check_adapter "$ADAPTER_PATH"

    # Load model config
    MODEL_CONFIG="configs/models/model_selection.yaml"
    if [ ! -f "$MODEL_CONFIG" ]; then
        print_error "Model configuration not found at: $MODEL_CONFIG"
        exit 1
    fi

    # Run inference
    python -m mlx_lora_trainer.inference \
        --model-config "$MODEL_CONFIG" \
        --adapter-path "$ADAPTER_PATH" \
        --temp 0.7 \
        --top-p 0.9 \
        --max-tokens 512 \
        --chat

    if [ $? -eq 0 ]; then
        print_success "Chat session ended successfully"
    else
        print_error "Chat session ended with an error"
        exit 1
    fi
}

# Run main function
main
