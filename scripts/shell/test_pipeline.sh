#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test dataset
TEST_DATA="
Question: What is 2+2?
Answer: Let me solve this step by step:
1) 2+2 is a basic addition problem
2) The sum of 2 and 2 is 4
Therefore, 2+2 = 4

Question: Write a Python function to add two numbers.
Answer: Here's a simple Python function to add two numbers:
def add_numbers(a, b):
    return a + b

Question: Explain what is machine learning?
Answer: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.
"

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
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

create_test_dataset() {
    local dataset_dir="$PROJECT_ROOT/data/test_dataset"
    mkdir -p "$dataset_dir"
    echo "$TEST_DATA" > "$dataset_dir/test_data.txt"
    print_success "Created test dataset at $dataset_dir"
}

test_model_config() {
    local model_config="$1"
    local model_name="$2"
    
    print_header "Testing $model_name Configuration"
    
    # Export model config
    export MLX_MODEL_CONFIG="configs/models/$model_config"
    
    # Test configuration loading
    python3 -c "
import yaml
with open('$PROJECT_ROOT/configs/models/$model_config', 'r') as f:
    config = yaml.safe_load(f)
assert 'model' in config, 'Model section missing'
assert 'training' in config, 'Training section missing'
assert 'lora' in config, 'LoRA section missing'
print('✓ Configuration validation passed')
"
    if [ $? -eq 0 ]; then
        print_success "Configuration test passed"
        return 0
    else
        print_error "Configuration test failed"
        return 1
    fi
}

test_training() {
    local model_config="$1"
    local model_name="$2"
    local model_name_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]' | tr -d '-')
    local output_dir="$PROJECT_ROOT/outputs/${model_name_lower}_test"
    
    print_header "Testing Training Pipeline for $model_name"
    
    # Set a very small number of iterations for testing
    export TRAIN_ITERS=2
    export EVAL_STEPS=1
    export MLX_MODEL_CONFIG="configs/models/$model_config"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Copy config for inference
    cp "$PROJECT_ROOT/configs/models/$model_config" "$output_dir/config.yaml"
    
    # Run training with test dataset
    bash "$PROJECT_ROOT/scripts/shell/train_lora.sh" \
        --data_path "$PROJECT_ROOT/data/test_dataset" \
        --output_dir "$output_dir" \
        --num_train_epochs 1
    
    if [ $? -eq 0 ]; then
        print_success "Training test passed"
        return 0
    else
        print_error "Training test failed"
        return 1
    fi
}

test_inference() {
    local model_name="$1"
    local model_name_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]' | tr -d '-')
    local output_dir="$PROJECT_ROOT/outputs/${model_name_lower}_test"
    
    print_header "Testing Inference for $model_name"
    
    # Test inference with a simple prompt
    echo "Testing simple inference..."
    python3 -c "
from mlx_lora_trainer.inference import generate_text
response = generate_text(
    'What is 2+2?',
    model_path='$output_dir',
    max_tokens=50
)
print(f'Response: {response}')
assert len(response) > 0, 'Empty response from model'
"
    
    if [ $? -eq 0 ]; then
        print_success "Inference test passed"
        return 0
    else
        print_error "Inference test failed"
        return 1
    fi
}

run_all_tests() {
    local failed=0
    
    print_header "Starting End-to-End Pipeline Tests"
    
    # Create test dataset
    create_test_dataset
    
    # Test models sequentially
    local models=(
        "phi3.yaml:Phi-3"
        "gemma.yaml:Gemma"
        "qwen.yaml:Qwen"
    )
    
    # Test each model
    for model_pair in "${models[@]}"; do
        IFS=: read -r config model_name <<< "$model_pair"
        
        print_header "Testing $model_name Pipeline"
        
        # Test configuration
        test_model_config "$config" "$model_name"
        if [ $? -ne 0 ]; then
            ((failed++))
            continue
        fi
        
        # Test training
        test_training "$config" "$model_name"
        if [ $? -ne 0 ]; then
            ((failed++))
            continue
        fi
        
        # Test inference
        test_inference "$model_name"
        if [ $? -ne 0 ]; then
            ((failed++))
        fi
    done
    
    print_header "Test Summary"
    if [ $failed -eq 0 ]; then
        print_success "All tests passed successfully!"
    else
        print_error "$failed test(s) failed"
    fi
    
    return $failed
}

# Main execution
run_all_tests
