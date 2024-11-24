#!/bin/bash

# Colors and formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
NC="\033[0m"

# Default paths
ADAPTER_PATH="adapters"
OUTPUT_DIR="exported_model"
CONFIG_FILE="adapters/config.json"

print_usage() {
    echo -e "${BOLD}Usage:${NC}"
    echo -e "  ./export.sh [options]"
    echo
    echo -e "${BOLD}Options:${NC}"
    echo -e "  --merge         Merge LoRA weights with base model"
    echo -e "  --quantize      Quantize the model to 4-bit precision"
    echo -e "  --output DIR    Output directory (default: exported_model)"
    echo -e "  --help          Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --merge)
            MERGE=true
            shift
            ;;
        --quantize)
            QUANTIZE=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Export the model
if [ "$MERGE" = true ]; then
    echo -e "${BOLD}Merging LoRA weights with base model...${NC}"
    python -m mlx_lora_trainer.scripts.python.cli export \
        --adapter-path "$ADAPTER_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --merge-weights
    
    if [ "$QUANTIZE" = true ]; then
        echo -e "${BOLD}Quantizing merged model...${NC}"
        python -m mlx_lora_trainer.scripts.python.cli quantize \
            --model-path "$OUTPUT_DIR" \
            --output-dir "${OUTPUT_DIR}_4bit"
    fi
else
    # Copy adapter and config for standalone deployment
    cp "$ADAPTER_PATH/lora_weights.safetensors" "$OUTPUT_DIR/"
    cp "$CONFIG_FILE" "$OUTPUT_DIR/"
fi

echo -e "${GREEN}Export complete!${NC}"
echo -e "Model exported to: ${BOLD}$OUTPUT_DIR${NC}"
echo
echo -e "${BOLD}To use your model:${NC}"
echo -e "1. For local use:"
echo -e "   ./scripts/shell/chat.sh"
echo
echo -e "2. For API deployment:"
echo -e "   mlx_lm.server --model $OUTPUT_DIR --port 8080"
echo
echo -e "3. For integration:"
echo -e "   See documentation for MLX LM Python API usage"
