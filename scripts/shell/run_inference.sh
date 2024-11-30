#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Colors for pretty output
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Print functions
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Main execution
main() {
    echo -e "${GREEN}Starting MLX LoRA Inference...${NC}"
    
    # Verify config directory exists
    if [ ! -d "${PROJECT_ROOT}/configs/models" ]; then
        print_error "Config directory not found: ${PROJECT_ROOT}/configs/models"
        print_info "Please ensure the configs directory exists"
        exit 1
    fi

    # Check for adapter weights
    ADAPTER_PATH="adapters/lora_weights.npz"
    if [ ! -f "$ADAPTER_PATH" ]; then
        print_error "Adapter weights not found at: $ADAPTER_PATH"
        print_info "Please ensure you have trained a model first"
        exit 1
    fi

    # Load and verify model config
    MODEL_CONFIG="configs/models/phi3.yaml"
    if [ ! -f "$MODEL_CONFIG" ]; then
        print_error "Model configuration not found at: $MODEL_CONFIG"
        print_info "Please ensure the model configuration exists"
        exit 1
    fi

    # Verify config file is readable and valid YAML
    if ! python3 -c "import yaml; yaml.safe_load(open('${MODEL_CONFIG}'))" 2>/dev/null; then
        print_error "Invalid YAML in model configuration: $MODEL_CONFIG"
        exit 1
    fi

    print_success "Configuration verified"
    print_info "Starting inference..."

    # Run inference
    python -m mlx_lora_trainer.inference \
        --model-config "$MODEL_CONFIG" \
        --adapter-path "$ADAPTER_PATH" \
        --temp 0.7 \
        --top-p 0.9 \
        --max-tokens 512 || {
            print_error "Chat session ended with an error"
            exit 1
        }
}

# Run main function
main
