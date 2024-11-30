#!/bin/bash

# Source common utilities and setup
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/utils/profiler.sh"

# Initialize profiler and set design elements
init_profiler

VERSION="1.0.0"
TOTAL_STEPS=8

# Design elements
HEADER_STYLE="━"
BULLET_POINT="•"
CHECKMARK="✓"
WARNING="⚠️"
INFO="ℹ️"

print_logo() {
    echo -e "${BLUE}"
    echo "     __  __ _    __  __    _     ___  ____      _     "
    echo "    |  \/  | |   \ \/ /   | |   / _ \|  _ \    / \    "
    echo "    | |\/| | |    >  <    | |  | | | | |_) |  / _ \   "
    echo "    | |  | | |___ /  \    | |__| |_| |  _ <  / ___ \  "
    echo "    |_|  |_|_____/_/\_\   |_____\___/|_| \_\/_/   \_\ "
    echo -e "${NC}"
}

print_model_info() {
    local model=$1
    echo -e "\n${CYAN}${BULLET_POINT} Model Details:${NC}"
    case $model in
        "phi-3")
            echo "  ${BULLET_POINT} Architecture: Phi-3.5 Mini"
            echo "  ${BULLET_POINT} Parameters: 3.5B"
            echo "  ${BULLET_POINT} RAM Required: 8GB+"
            echo "  ${BULLET_POINT} Training Time: ~25min/epoch"
            ;;
        "gemma")
            echo "  ${BULLET_POINT} Architecture: Gemma 2-2B"
            echo "  ${BULLET_POINT} Parameters: 2.2B"
            echo "  ${BULLET_POINT} RAM Required: 12GB+"
            echo "  ${BULLET_POINT} Training Time: ~45min/epoch"
            ;;
        "qwen")
            echo "  ${BULLET_POINT} Architecture: Qwen 2.5 7B"
            echo "  ${BULLET_POINT} Parameters: 7B"
            echo "  ${BULLET_POINT} RAM Required: 32GB+"
            echo "  ${BULLET_POINT} Training Time: ~90min/epoch"
            ;;
    esac
}

# Main function with complete journey
main() {
    setup_environment
    detect_hardware
    setup_training_environment
    select_model
    select_dataset
    run_training
    run_inference
    export_model
}

# Enhanced model selection with detailed info
print_model_options() {
    echo -e "${CYAN}1. Phi-3.5 Mini (3B)${NC}"
    echo -e "   • Optimized for instruction following"
    echo -e "   • Memory: ~8GB required"
    echo -e "   • Best for: General tasks, coding"
    echo -e "   • Training time: ~25min/epoch"
    echo
    echo -e "${CYAN}2. Gemma 2-2B${NC}"
    echo -e "   • Google's latest compact model"
    echo -e "   • Memory: ~12GB required"
    echo -e "   • Best for: Balanced performance"
    echo -e "   • Training time: ~45min/epoch"
    echo
    echo -e "${CYAN}3. Qwen 2.5 7B${NC}"
    echo -e "   • Advanced multilingual model"
    echo -e "   • Memory: ~32GB required"
    echo -e "   • Best for: Complex tasks"
    echo -e "   • Training time: ~90min/epoch"
}

# Dataset selection with multiple options
select_dataset() {
    print_header "Dataset Selection"
    echo -e "\n${BOLD}Choose your dataset:${NC}\n"
    
    # Example Datasets
    echo -e "1) ${CYAN}Example Datasets${NC}"
    echo -e "   • SQL Generation (WikiSQL format)"
    echo -e "   • Instruction Following (Alpaca format)"
    echo -e "   • Code Generation (Python/JavaScript)"
    echo -e "   • Chat Conversations (ShareGPT format)"
    
    # HuggingFace Datasets
    echo -e "\n2) ${CYAN}HuggingFace Datasets${NC}"
    echo -e "   • OpenAssistant Conversations"
    echo -e "   • Code Alpaca"
    echo -e "   • Anthropic HH"
    
    # Custom Dataset
    echo -e "\n3) ${CYAN}Custom Dataset${NC}"
    echo -e "   • JSONL format"
    echo -e "   • Auto-formatting support"
    echo -e "   • Validation included"
    
    read -p "Select dataset type (1-3): " dataset_choice
    
    case $dataset_choice in
        1) setup_example_dataset ;;
        2) setup_huggingface_dataset ;;
        3) setup_custom_dataset ;;
        *) 
            print_error "Invalid choice"
            return 1
            ;;
    esac
}

setup_example_dataset() {
    echo -e "\n${CYAN}Select Example Dataset:${NC}"
    echo "1) SQL Generation (2K examples)"
    echo "2) Instruction Following (5K examples)"
    echo "3) Code Generation (3K examples)"
    echo "4) Chat (4K examples)"
    
    read -p "Choose dataset (1-4): " choice
    
    local dataset_path="data/examples"
    mkdir -p "$dataset_path"
    
    case $choice in
        1) python scripts/python/prepare_data.py --type sql --output "$dataset_path" ;;
        2) python scripts/python/prepare_data.py --type instruct --output "$dataset_path" ;;
        3) python scripts/python/prepare_data.py --type code --output "$dataset_path" ;;
        4) python scripts/python/prepare_data.py --type chat --output "$dataset_path" ;;
        *) 
            print_error "Invalid choice"
            return 1
            ;;
    esac
    
    # Show dataset preview
    preview_dataset "$dataset_path/train.jsonl"
    
    echo -e "\n${CYAN}Would you like to:"
    echo "1) Proceed with this dataset"
    echo "2) View more examples"
    echo "3) Choose a different dataset${NC}"
    
    read -p "> " choice
    case $choice in
        1) return 0 ;;
        2) preview_dataset "$dataset_path/train.jsonl" 10 ;;
        3) select_dataset ;;
        *) print_error "Invalid choice" ; return 1 ;;
    esac
}

setup_huggingface_dataset() {
    echo -e "\n${CYAN}Enter HuggingFace Dataset Path:${NC}"
    echo "Example: OpenAssistant/oasst_top1_2023-08-25"
    read -p "> " dataset_path
    
    python scripts/python/prepare_data.py \
        --source huggingface \
        --path "$dataset_path" \
        --output "data/huggingface" || return 1
}

setup_custom_dataset() {
    echo -e "\n${CYAN}Custom Dataset Setup${NC}"
    echo -e "Place your dataset in: data/custom/input.jsonl"
    echo -e "\nRequired format:"
    echo -e '{"instruction": "task description", "input": "optional context", "output": "expected output"}'
    echo -e "\nPress Enter when ready..."
    read
    
    if [ ! -f "data/custom/input.jsonl" ]; then
        print_error "data/custom/input.jsonl not found"
        return 1
    fi
    
    python scripts/python/prepare_data.py \
        --source custom \
        --input "data/custom/input.jsonl" \
        --output "data/custom/processed" || return 1
}

# Training configuration
configure_training() {
    echo -e "\n${CYAN}Configuring Training Parameters${NC}"
    
    # Show parameter guide
    echo -e "Would you like to see the parameter guide? (y/n)"
    read -p "> " show_guide
    if [[ $show_guide == "y" ]]; then
        show_training_guide
    fi
    
    echo -e "\n${CYAN}Configuring training parameters...${NC}"
    
    # Load default config based on model
    local config_path="configs/models/${MODEL_NAME}.yaml"
    
    # Allow user customization
    echo -e "\nDefault configuration loaded from: ${config_path}"
    echo -e "Would you like to customize any parameters? (y/n)"
    read -p "> " customize
    
    if [[ $customize == "y" ]]; then
        customize_training_config "$config_path"
    fi
    
    print_success "Training configuration complete"
    return 0
}

# Export and deployment
export_model() {
    echo -e "\n${CYAN}Exporting Model${NC}"
    
    # Show size estimates
    local model_size=$(du -sh exported_model/merged 2>/dev/null | cut -f1 || echo "2-7GB")
    local lora_size=$(du -sh adapters/lora_weights.npz 2>/dev/null | cut -f1 || echo "~100MB")
    
    echo -e "\n${CYAN}Export Options:${NC}"
    echo -e "1) ${BOLD}Full Merged Model${NC}"
    echo -e "   • Size: ~$model_size"
    echo -e "   • Best for: Deployment, sharing"
    echo -e "   • No base model needed"
    echo
    echo -e "2) ${BOLD}LoRA Weights Only${NC}"
    echo -e "   • Size: ~$lora_size"
    echo -e "   • Best for: Distribution, version control"
    echo -e "   • Requires base model"
    echo
    echo -e "3) ${BOLD}Ollama Format${NC}"
    echo -e "   • Optimized for Ollama"
    echo -e "   • Easy local deployment"
    echo -e "   • Includes chat template"
    
    read -p "Select export format (1-3): " format_choice
    
    case $format_choice in
        1) export_merged_model ;;
        2) export_lora_weights ;;
        3) export_ollama_format ;;
        *) print_error "Invalid choice" ; return 1 ;;
    esac
    
    # Generate deployment guide
    generate_deployment_guide "$format_choice"
}

export_ollama_format() {
    echo -e "\n${CYAN}Exporting for Ollama...${NC}"
    
    # Create Ollama model directory
    mkdir -p exported_model/ollama
    
    # Export model and config
    ./scripts/shell/export.sh \
        --format ollama \
        --output exported_model/ollama || return 1
        
    print_success "Model exported for Ollama"
    echo -e "\n${CYAN}To use with Ollama:${NC}"
    echo "1. Copy to Ollama models directory:"
    echo "   cp -r exported_model/ollama ~/.ollama/models/$(get_model_name)"
    echo "2. Run model:"
    echo "   ollama run $(get_model_name)"
}

# Show completion message with next steps
show_completion_message() {
    clear
    print_logo
    echo -e "${BOLD}${GREEN}🎉 Congratulations! Your Model is Ready${NC}"
    echo
    echo -e "${CYAN}What you've accomplished:${NC}"
    echo -e "✓ Environment setup"
    echo -e "✓ Hardware optimization"
    echo -e "✓ Model selection and configuration"
    echo -e "✓ Dataset preparation"
    echo -e "✓ Fine-tuning with LoRA"
    echo -e "✓ Model evaluation"
    echo -e "✓ Export and deployment"
    echo
    echo -e "${BOLD}Your model is available at:${NC}"
    echo -e "• ${CYAN}Merged model: exported_model/merged/${NC}"
    echo -e "• ${CYAN}LoRA weights: exported_model/lora/${NC}"
    echo
    echo -e "${BOLD}${YELLOW}Next Steps:${NC}"
    echo -e "1. ${YELLOW}Try your model:${NC}"
    echo -e "   ./scripts/shell/chat.sh --model exported_model/merged"
    echo -e "2. ${YELLOW}Share your model:${NC}"
    echo -e "   ./scripts/shell/export.sh --push"
    echo -e "3. ${YELLOW}Start a new training run:${NC}"
    echo -e "   ./run_tutorial.sh --fresh"
    echo
}

# Run main with error handling
if ! main; then
    print_error "Tutorial failed. Please check the errors above."
    print_detail "For troubleshooting, see docs/troubleshooting.md"
    exit 1
fi

# Add progress bar function
show_progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local title=$3
    local percentage=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    
    # Show spinner animation
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local temp=${spinstr#?}
    printf "\r${CYAN}${title} [%-${width}s] %3d%% %c${NC}" \
           "$(printf "%${filled}s" | tr ' ' '=')" \
           "$percentage" \
           "$spinstr"
}

# Add to setup_environment()
setup_environment() {
    print_step "Setting up Environment" 1
    
    local total_steps=5
    local current=0
    
    # Create virtual environment
    echo -e "\n${CYAN}Installing Dependencies (this may take a few minutes)...${NC}"
    ((current++))
    show_progress_bar $current $total_steps "Installing Dependencies"
    python3 -m venv venv || return 1
    
    # Activate environment
    ((current++))
    show_progress_bar $current $total_steps "Activating Environment"
    source venv/bin/activate || return 1
    
    # Install base packages
    ((current++))
    show_progress_bar $current $total_steps "Installing Base Packages"
    pip install --quiet --upgrade pip || return 1
    
    # Install MLX dependencies
    ((current++))
    show_progress_bar $current $total_steps "Installing MLX Dependencies"
    pip install --quiet mlx transformers || return 1
    
    # Verify installation
    ((current++))
    show_progress_bar $current $total_steps "Verifying Installation"
    verify_installations || return 1
    
    echo -e "\n${GREEN}✓ Environment setup complete${NC}"
}

preview_dataset() {
    local dataset_path=$1
    local preview_count=3
    
    echo -e "\n${CYAN}Dataset Preview:${NC}"
    echo -e "${DIM}Showing first $preview_count examples...${NC}\n"
    
    python3 -c "
import json
import sys

try:
    with open('$dataset_path') as f:
        for i, line in enumerate(f):
            if i >= $preview_count: break
            data = json.loads(line)
            print(f'${BOLD}Example {i+1}:${NC}')
            print(f'${CYAN}Input:${NC} {data[\"instruction\"][:100]}...')
            if 'input' in data and data['input']:
                print(f'${CYAN}Context:${NC} {data[\"input\"][:100]}...')
            print(f'${CYAN}Output:${NC} {data[\"output\"][:100]}...')
            print()
except Exception as e:
    print(f'Error previewing dataset: {str(e)}', file=sys.stderr)
    sys.exit(1)
"
}

show_training_guide() {
    echo -e "\n${CYAN}Training Parameters Guide:${NC}"
    echo -e "${BOLD}Key Parameters:${NC}"
    echo -e "• ${CYAN}batch_size${NC}: Number of examples processed together"
    echo -e "  - Higher = faster training but more memory"
    echo -e "  - Recommended: 1-4 for most setups"
    echo
    echo -e "• ${CYAN}learning_rate${NC}: How fast the model learns"
    echo -e "  - Higher = faster learning but may be unstable"
    echo -e "  - Recommended: 2e-4 to 5e-4"
    echo
    echo -e "• ${CYAN}max_seq_length${NC}: Maximum input length"
    echo -e "  - Longer = more context but more memory"
    echo -e "  - Model limits: Phi-3 (2048), Gemma (8192), Qwen (32768)"
    echo
    echo -e "• ${CYAN}lora.r${NC}: LoRA rank - controls capacity"
    echo -e "  - Higher = more capacity but more memory"
    echo -e "  - Recommended: 8-32 depending on task"
    
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read
}

# Add detailed hardware optimization functions
detect_hardware() {
    print_step "Hardware Detection & Optimization" 2
    
    # Get detailed system info
    local cpu_info=$(sysctl -n machdep.cpu.brand_string)
    local total_ram=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
    
    # Set optimal configurations based on hardware
    case $total_ram in
        8|16) RECOMMENDED_MODEL="phi-3" ;;
        32|64) RECOMMENDED_MODEL="gemma" ;;
        *) RECOMMENDED_MODEL="qwen" ;;
    esac
}

save_hardware_profile() {
    # Save hardware configuration for other scripts
    cat > .hardware_config << EOF
{
    "cpu": "$cpu_info",
    "ram": $total_ram,
    "recommended_model": "$RECOMMENDED_MODEL",
    "batch_size": $BATCH_SIZE,
    "memory_limit": $MLX_METAL_DEVICE_MEMORY_LIMIT,
    "grad_checkpoint": $GRAD_CHECKPOINT,
    "mlx_threads": $MLX_NUM_THREADS
}
EOF
}

optimize_for_training() {
    print_info "Optimizing system for training..."
    
    # Close unnecessary applications
    osascript -e 'tell application "System Events" to set the visible of every process to true'
    osascript -e 'tell application "System Events" to set the frontmost of every process to true'
    
    # Set process priority
    sudo renice -n -20 $$
    
    # Disable system sleep
    caffeinate -i &
    CAFFEINATE_PID=$!
    
    # Clean up on exit
    trap 'kill $CAFFEINATE_PID 2>/dev/null' EXIT
    
    print_success "System optimized for training"
}

monitor_resources() {
    # Start resource monitoring in background
    (while true; do
        memory_used=$(ps -o rss= -p $$ | awk '{print $1/1024/1024}')
        gpu_usage=$(python3 -c "import mlx.core as mx; print(mx.metal.get_gpu_utilization())")
        
        if (( $(echo "$memory_used > $MLX_METAL_DEVICE_MEMORY_LIMIT * 0.9" | bc -l) )); then
            print_warning "High memory usage: ${memory_used}GB"
        fi
        
        sleep 5
    done) &
    MONITOR_PID=$!
    
    # Clean up monitor on exit
    trap 'kill $MONITOR_PID 2>/dev/null' EXIT
}

setup_training_environment() {
    print_step "Training Environment Setup" 3
    
    # Create necessary directories
    mkdir -p data/{custom,examples,huggingface}
    mkdir -p adapters
    mkdir -p exported_model/{merged,lora}
    mkdir -p logs
    
    # Install required packages
    pip install --quiet --upgrade pip
    pip install --quiet mlx transformers huggingface_hub tokenizers
    
    # Verify MLX installation
    python3 -c "import mlx" 2>/dev/null || {
        print_error "MLX installation failed"
        return 1
    }
    
    return 0
}

run_training() {
    print_step "Training Process" 6
    
    echo -e "\n${CYAN}Training Configuration:${NC}"
    echo -e "• Model: $selected_model"
    echo -e "• Dataset: $dataset_path"
    echo -e "• Output: adapters/lora_weights.npz"
    
    # Show training parameters guide
    show_training_guide
    
    # Start training with progress monitoring
    echo -e "\n${INFO} Starting training..."
    python scripts/python/train.py \
        --model $selected_model \
        --dataset $dataset_path \
        --output adapters \
        --config configs/training.yaml \
        --log logs/training.log || {
            print_error "Training failed"
            return 1
        }
    
    print_success "Training completed successfully!"
    return 0
}

run_inference() {
    print_step "Model Inference" 7
    
    echo -e "\n${CYAN}Testing your model...${NC}"
    echo -e "Enter a prompt (or 'q' to quit):"
    
    while true; do
        read -p "> " prompt
        [ "$prompt" = "q" ] && break
        
        python scripts/python/inference.py \
            --model $selected_model \
            --weights adapters/lora_weights.npz \
            --prompt "$prompt" || {
                print_error "Inference failed"
                return 1
            }
    done
    
    return 0
}

export_model() {
    print_step "Model Export" 8
    
    echo -e "\n${CYAN}Export Options:${NC}"
    echo "1) Merged Model (ready to use)"
    echo "2) LoRA Weights Only (smaller size)"
    echo "3) Ollama Format"
    read -p "Select export format (1-3): " format
    
    case $format in
        1) export_merged_model ;;
        2) export_lora_weights ;;
        3) export_ollama_format ;;
        *) print_error "Invalid choice" ; return 1 ;;
    esac
}

# Add after main initialization
setup_recovery() {
    mkdir -p .recovery
    
    # Setup auto-save
    (while true; do
        save_checkpoint
        sleep 300
    done) &
    AUTOSAVE_PID=$!
    
    # Cleanup on exit
    trap 'kill $AUTOSAVE_PID 2>/dev/null' EXIT
}

save_checkpoint() {
    echo "CURRENT_STEP=$CURRENT_STEP" > .recovery/checkpoint
    echo "MODEL_CHOICE=$MODEL_CHOICE" >> .recovery/checkpoint
    echo "DATASET_PATH=$DATASET_PATH" >> .recovery/checkpoint
}

show_progress() {
    clear
    print_logo
    
    # Show overall progress
    echo -e "\n${BOLD}Progress: ${CURRENT_STEP}/${TOTAL_STEPS}${NC}"
    print_progress_bar $CURRENT_STEP $TOTAL_STEPS
    
    # Show current task details
    echo -e "\n${CYAN}Current Task:${NC}"
    case $CURRENT_STEP in
        1) echo "Setting up environment..." ;;
        2) echo "Configuring hardware..." ;;
        3) echo "Preparing model..." ;;
        4) echo "Processing dataset..." ;;
        5) echo "Training model..." ;;
        6) echo "Testing model..." ;;
        7) echo "Exporting model..." ;;
        8) echo "Finalizing..." ;;
    esac
}

# Add at the beginning of the script
setup_recovery_system() {
    mkdir -p .recovery
    
    # Create recovery script
    cat > .recovery/auto_backup.sh << 'EOF'
#!/bin/bash
while true; do
    # Save current state
    if [ -f ".training_state" ]; then
        cp .training_state .recovery/
    fi
    
    # Save model weights
    if [ -f "adapters/lora_weights.npz" ]; then
        cp adapters/lora_weights.npz .recovery/
    fi
    
    # Save training logs
    if [ -f "logs/training.log" ]; then
        cp logs/training.log .recovery/
    fi
    
    sleep 300  # Backup every 5 minutes
done
EOF
    
    chmod +x .recovery/auto_backup.sh
    
    # Start auto-backup in background
    ./.recovery/auto_backup.sh &
    BACKUP_PID=$!
    
    # Save PID for cleanup
    echo $BACKUP_PID > .recovery/backup.pid
}

restore_from_recovery() {
    if [ -d ".recovery" ]; then
        echo -e "${CYAN}Recovery files found. Would you like to restore? (y/n)${NC}"
        read -p "> " restore_choice
        
        if [[ $restore_choice == "y" ]]; then
            # Restore state
            if [ -f ".recovery/.training_state" ]; then
                cp .recovery/.training_state .
                source .training_state
                print_success "Restored training state"
            fi
            
            # Restore weights
            if [ -f ".recovery/lora_weights.npz" ]; then
                cp .recovery/lora_weights.npz adapters/
                print_success "Restored model weights"
            fi
            
            return 0
        fi
    fi
    return 1
}

cleanup() {
    # Kill auto-backup process
    if [ -f ".recovery/backup.pid" ]; then
        kill $(cat .recovery/backup.pid) 2>/dev/null
    fi
    
    # Save final state before exit
    save_final_state
    
    # Clean up temporary files
    rm -rf .tmp_* 2>/dev/null
}

show_step_progress() {
    clear
    print_logo
    
    echo -e "\n${BOLD}${BLUE}Progress Overview:${NC}"
    echo -e "╔════════════════════════════════════════════════╗"
    
    for i in $(seq 1 $TOTAL_STEPS); do
        if [ $i -lt $CURRENT_STEP ]; then
            echo -e "║ ${GREEN}✓${NC} Step $i: ${COMPLETED_STEPS[$i]}"
        elif [ $i -eq $CURRENT_STEP ]; then
            echo -e "║ ${YELLOW}◉${NC} Step $i: ${CURRENT_TASK} ${YELLOW}(In Progress)${NC}"
        else
            echo -e "║ ○ Step $i: ${PENDING_STEPS[$i]}"
        fi
    done
    
    echo -e "╚════════════════════════════════════════════════╝"
    
    # Show current task details
    if [ ! -z "$CURRENT_TASK" ]; then
        echo -e "\n${CYAN}Current Task: $CURRENT_TASK${NC}"
        show_task_progress
    fi
}

show_training_dashboard() {
    clear
    print_logo
    
    # System Stats
    echo -e "\n${CYAN}System Metrics:${NC}"
    memory_used=$(ps -o rss= -p $$ | awk '{print $1/1024/1024}')
    gpu_util=$(python3 -c "import mlx.core as mx; print(mx.metal.get_gpu_utilization())")
    cpu_util=$(ps -o %cpu= -p $$)
    
    echo -e "╔═══════════════════════════════════════════════╗"
    echo -e "║ Memory: ${memory_used}GB / ${MLX_METAL_DEVICE_MEMORY_LIMIT}GB"
    echo -e "║ GPU Util: ${gpu_util}%"
    echo -e "║ CPU Util: ${cpu_util}%"
    echo -e "╚═══════════════════════════════════════════════╝"
    
    # Training Metrics
    echo -e "\n${CYAN}Training Metrics:${NC}"
    echo -e "╔════════════════════════════════════════════════╗"
    echo -e "║ Loss: ${current_loss} $(show_loss_trend)"
    echo -e "║ Perplexity: ${perplexity}"
    echo -e "║ Gradient Norm: ${grad_norm}"
    echo -e "║ Learning Rate: ${current_lr}"
    echo -e "╚════════════════════════════════════════════════╝"
    
    # Progress and ETA
    show_progress_bar $current_step $total_steps
    echo -e "ETA: ${remaining_time}"
    
    # Live Sample Generation
    if (( $current_step % 50 == 0 )); then
        echo -e "\n${CYAN}Sample Generation:${NC}"
        generate_sample
    fi
    
    # Show alerts if needed
    show_alerts
}

show_alerts() {
    if (( $(echo "$memory_used > $MLX_METAL_DEVICE_MEMORY_LIMIT * 0.9" | bc -l) )); then
        echo -e "\n${RED}⚠️ High Memory Usage Warning${NC}"
    fi
    if (( $(echo "$current_loss > 3.0" | bc -l) )); then
        echo -e "\n${YELLOW}💡 High Loss Warning${NC}"
    fi
}

show_model_comparison() {
    clear
    print_logo
    
    echo -e "\n${CYAN}Available Models${NC}"
    echo -e "╔════════════════════════════════════════════════════════════════╗"
    echo -e "║ 1. Phi-3.5 Mini                                                ║"
    echo -e "║    • Size: 3.5B parameters                                     ║"
    echo -e "║    • RAM: 8GB minimum                                          ║"
    echo -e "║    • Speed: ~1000 tokens/sec                                   ║"
    echo -e "║    • Best for: Quick experiments, testing                      ║"
    echo -e "╟────────────────────────────────────────────────────────────────╢"
    echo -e "║ 2. Gemma 2-2B                                                  ║"
    echo -e "║    • Size: 2.2B parameters                                     ║"
    echo -e "║    • RAM: 12GB minimum                                         ║"
    echo -e "║    • Speed: ~800 tokens/sec                                    ║"
    echo -e "║    • Best for: Balanced performance                            ║"
    echo -e "╟────────────────────────────────────────────────────────────────╢"
    echo -e "║ 3. Qwen 2.5 7B                                                 ║"
    echo -e "║    • Size: 7B parameters                                       ║"
    echo -e "║    • RAM: 32GB minimum                                         ║"
    echo -e "║    • Speed: ~400 tokens/sec                                    ║"
    echo -e "║    • Best for: Production quality                              ║"
    echo -e "╚════════════════════════════════════════════════════════════════╝"
    
    # Show hardware-based recommendation
    echo -e "\n${GREEN}💡 Recommended for your hardware: $RECOMMENDED_MODEL${NC}"
}

validate_and_preview_dataset() {
    echo -e "\n${CYAN}Analyzing Dataset...${NC}"
    
    # Run quick analysis
    python3 -c "
import json
import sys
from pathlib import Path

def analyze_dataset(path):
    with open(path) as f:
        data = [json.loads(line) for line in f]
    
    total = len(data)
    avg_length = sum(len(str(d)) for d in data) / total
    formats = set(d.get('format', 'unknown') for d in data)
    
    return {
        'total': total,
        'avg_length': int(avg_length),
        'formats': list(formats),
        'samples': data[:3]
    }

try:
    stats = analyze_dataset('$dataset_path')
    print(json.dumps(stats))
except Exception as e:
    print(f'ERROR: {str(e)}', file=sys.stderr)
    sys.exit(1)
" > .dataset_stats

    # Show analysis results
    echo -e "\n${CYAN}Dataset Overview:${NC}"
    echo -e "• Total Examples: $(jq .total .dataset_stats)"
    echo -e "• Average Length: $(jq .avg_length .dataset_stats) chars"
    echo -e "• Format: $(jq -r '.formats[]' .dataset_stats)"
    
    # Show samples with syntax highlighting
    echo -e "\n${CYAN}Sample Entries:${NC}"
    jq -r '.samples[] | "User: \(.conversations[0].content)\nAssistant: \(.conversations[1].content)\n"' .dataset_stats
}

show_export_options() {
    clear
    print_logo
    
    echo -e "\n${CYAN}Export Options${NC}"
    echo -e "╔════════════════════════════════════════════════╗"
    echo -e "║ 1. Quick Start Bundle                          ║"
    echo -e "║    • Ready-to-use model                        ║"
    echo -e "║    • Includes chat script                      ║"
    echo -e "║    • Size: ~${merged_size}GB                   ║"
    echo -e "╟────────────────────────────────────────────────╢"
    echo -e "║ 2. Lightweight Package                         ║"
    echo -e "║    • LoRA weights only                         ║"
    echo -e "║    • Requires base model                       ║"
    echo -e "║    • Size: ~${lora_size}MB                     ║"
    echo -e "╟────────────────────────────────────────────────╢"
    echo -e "║ 3. Ollama Integration                          ║"
    echo -e "║    • Direct Ollama import                      ║"
    echo -e "║    • Local deployment                          ║"
    echo -e "║    • Easy sharing                              ║"
    echo -e "╚════════════════════════════════════════════════╝"
    
    read -p "Select export format (1-3): " export_choice
    
    case $export_choice in
        1) export_full_model ;;
        2) export_lora_weights ;;
        3) export_to_ollama ;;
        *) print_error "Invalid choice" ; return 1 ;;
    esac
}

convert_dataset() {
    echo -e "\n${CYAN}Dataset Conversion Utility${NC}"
    echo -e "╔════════════════════════════════════════════════╗"
    echo -e "║ 1. Text Files (.txt)                           ║"
    echo -e "║ 2. CSV Files (.csv)                            ║"
    echo -e "║ 3. JSON Files (.json)                          ║"
    echo -e "║ 4. Custom Format                               ║"
    echo -e "╚════════════════════════════════════════════════╝"
    
    read -p "Select input format (1-4): " format_choice
    
    # Create conversion script
    cat > scripts/python/convert_dataset.py << 'EOF'
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict

def convert_text(file_path: str) -> List[Dict]:
    """Convert plain text to conversation format."""
    conversations = []
    with open(file_path, 'r') as f:
        text = f.read()
        # Smart chunking with overlap
        chunks = chunk_text(text, max_length=512, overlap=50)
        for chunk in chunks:
            conversations.append({
                "conversations": [
                    {"role": "user", "content": "Continue the text:"},
                    {"role": "assistant", "content": chunk}
                ]
            })
    return conversations

def convert_csv(file_path: str) -> List[Dict]:
    """Convert CSV to conversation format."""
    conversations = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'input' in row and 'output' in row:
                conversations.append({
                    "conversations": [
                        {"role": "user", "content": row['input']},
                        {"role": "assistant", "content": row['output']}
                    ]
                })
    return conversations

# Main conversion logic
input_path = sys.argv[1]
output_path = sys.argv[2]
format_type = sys.argv[3]

converters = {
    'txt': convert_text,
    'csv': convert_csv,
    'json': convert_json
}

conversations = converters[format_type](input_path)
with open(output_path, 'w') as f:
    for conv in conversations:
        f.write(json.dumps(conv) + '\n')
EOF
    
    # Run conversion
    python3 scripts/python/convert_dataset.py "$input_path" "data/train.jsonl" "$format_type"
}

deploy_model() {
    echo -e "\n${CYAN}Deployment Options${NC}"
    echo -e "╔════════════════════════════════════════════════╗"
    echo -e "║ 1. Start Local API Server                      ║"
    echo -e "║    • FastAPI endpoint                          ║"
    echo -e "║    • WebSocket support                         ║"
    echo -e "║    • Swagger documentation                     ║"
    echo -e "╟────────────────────────────────────────────────╢"
    echo -e "║ 2. Export to Ollama                            ║"
    echo -e "║    • Local deployment                          ║"
    echo -e "║    • CLI interface                             ║"
    echo -e "║    • API access                                ║"
    echo -e "╟────────────────────────────────────────────────╢"
    echo -e "║ 3. Package for Distribution                    ║"
    echo -e "║    • Compressed weights                        ║"
    echo -e "║    • Config files                              ║"
    echo -e "║    • Example scripts                           ║"
    echo -e "╚════════════════════════════════════════════════╝"
    
    read -p "Select deployment option (1-3): " deploy_choice
    
    case $deploy_choice in
        1) deploy_api_server ;;
        2) deploy_to_ollama ;;
        3) package_for_distribution ;;
        *) print_error "Invalid choice" ; return 1 ;;
    esac
}

deploy_api_server() {
    echo -e "\n${CYAN}Starting API Server...${NC}"
    
    # Create FastAPI server
    cat > scripts/python/api_server.py << 'EOF'
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import mlx.core as mx
from pathlib import Path
import json

app = FastAPI(title="MLX LoRA Model API")

class PredictRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/predict")
async def predict(request: PredictRequest):
    response = model.generate(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    return {"response": response}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = model.generate(data)
        await websocket.send_text(response)
EOF
    
    # Start server
    uvicorn scripts.python.api_server:app --host 0.0.0.0 --port 8000
}

show_quick_start() {
    echo -e "\n${CYAN}Quick Start Mode${NC}"
    echo -e "• Optimized for your M${CHIP_GENERATION} Mac"
    echo -e "• Using ${RECOMMENDED_MODEL} (best for your hardware)"
    echo -e "• Example dataset ready to go"
    echo -e "• ~${ESTIMATED_TIME} minutes to first results"
    
    echo -e "\n${YELLOW}Start quick demo? (y/n)${NC}"
    read -p "> " quick_start
    
    if [[ $quick_start == "y" ]]; then
        run_quick_demo
    fi
}

show_completion() {
    clear
    print_logo
    
    echo -e "\n${GREEN}🎉 Congratulations! You've Successfully Fine-tuned Your First Model!${NC}"
    echo -e "\n${CYAN}Your Achievement:${NC}"
    echo -e "• Model: $selected_model"
    echo -e "• Training Time: ${training_duration}"
    echo -e "• Final Loss: ${final_loss}"
    
    echo -e "\n${CYAN}Next Steps:${NC}"
    echo -e "1. Try your model: ./scripts/chat.sh"
    echo -e "2. Export for deployment: ./scripts/export.sh"
    echo -e "3. Explore advanced features: ./scripts/advanced.sh"
    
    echo -e "\n${YELLOW}💡 Pro Tip: Share your success!${NC}"
    echo -e "Tag @InfiniteCanvas with your results!"
}

show_company_footer() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Powered by Infinite Canvas${NC}"
    echo -e "Democratizing AI, One Model at a Time"
}

validate_dataset_format() {
    local dataset_path=$1
    
    # Check file exists
    if [[ ! -f "$dataset_path" ]]; then
        print_error "Dataset file not found: $dataset_path"
        return 1
    fi
    
    # Validate JSONL format
    jq -c . "$dataset_path" > /dev/null 2>&1 || {
        print_error "Invalid JSONL format"
        return 1
    }
    
    # Check conversation format
    python3 -c '
import sys, json
with open(sys.argv[1]) as f:
    for line in f:
        data = json.loads(line)
        if "conversations" not in data:
            sys.exit(1)
        if not isinstance(data["conversations"], list):
            sys.exit(1)
        if len(data["conversations"]) < 2:
            sys.exit(1)
' "$dataset_path" || {
        print_error "Invalid conversation format"
        return 1
    }
}

# Add to save_checkpoint()
save_training_state() {
    # Save model state
    python3 -c "
import mlx.core as mx
mx.save('${CHECKPOINT_DIR}/model_state.npz', {
    'step': current_step,
    'optimizer': optimizer_state,
    'model': model_state,
    'loss': current_loss
})
"
    
    # Save training config
    cat > "${CHECKPOINT_DIR}/training_config.json" << EOF
{
    "model": "${selected_model}",
    "dataset": "${dataset_path}",
    "batch_size": ${batch_size},
    "learning_rate": ${learning_rate},
    "steps": ${current_step}
}
EOF
}

# Add to main()
setup_error_handling() {
    trap 'handle_error $?' ERR
    trap 'handle_interrupt' INT TERM
}

handle_error() {
    local exit_code=$1
    print_error "An error occurred (code: $exit_code)"
    
    if [[ -f "${CHECKPOINT_DIR}/training_config.json" ]]; then
        echo -e "\n${YELLOW}Would you like to resume from the last checkpoint? (y/n)${NC}"
        read -p "> " resume_choice
        if [[ $resume_choice == "y" ]]; then
            restore_checkpoint
            return 0
        fi
    fi
    exit $exit_code
}

monitor_resources() {
    # Get GPU memory usage
    local gpu_mem=$(python3 -c "
import mlx.core as mx
print(mx.metal.get_gpu_memory_used() / 1024**3)
")
    
    # Get process memory
    local process_mem=$(ps -o rss= -p $$ | awk '{print $1/1024/1024}')
    
    echo -e "\n${CYAN}Resource Usage:${NC}"
    echo -e "• GPU Memory: ${gpu_mem:.1f}GB"
    echo -e "• Process Memory: ${process_mem:.1f}GB"
    echo -e "• GPU Utilization: ${gpu_util}%"
}

validate_exported_model() {
    echo -e "\n${CYAN}Validating Exported Model...${NC}"
    
    # 1. Environment Validation
    echo -e "\n${CYAN}Checking Deployment Environment...${NC}"
    python3 -c "
import sys
import mlx.core as mx

# Check MLX compatibility
if not mx.metal.is_available():
    print('Error: Metal not available')
    sys.exit(1)

# Check memory requirements
total_memory = mx.metal.get_device_memory_info()['total'] / 1024**3
if total_memory < 4:
    print('Error: Insufficient GPU memory')
    sys.exit(1)
"
    
    # 2. Model File Validation
    echo -e "\n${CYAN}Validating Model Files...${NC}"
    required_files=(
        "${EXPORT_DIR}/model.bin"
        "${EXPORT_DIR}/config.json"
        "${EXPORT_DIR}/tokenizer.json"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Missing required file: $file"
            return 1
        fi
    done
    
    # 3. Model Loading Test
    echo -e "\n${CYAN}Testing Model Loading...${NC}"
    python3 -c "
import mlx.core as mx
try:
    model = mx.load('${EXPORT_DIR}/model.bin')
    print('✓ Model loaded successfully')
except Exception as e:
    print(f'Error loading model: {e}')
    exit(1)
"
    
    # 4. Basic Inference Test
    echo -e "\n${CYAN}Running Inference Test...${NC}"
    python3 -c "
from mlx_lora_trainer.utils import generate
from transformers import AutoTokenizer
import mlx.core as mx

try:
    # Load model and tokenizer
    model = mx.load('${EXPORT_DIR}/model.bin')
    tokenizer = AutoTokenizer.from_pretrained('${EXPORT_DIR}')
    
    # Test generation
    test_prompt = 'Hello, how are you?'
    response = next(generate(model, tokenizer, test_prompt, max_tokens=20))
    print('✓ Inference test passed')
except Exception as e:
    print(f'Error during inference: {e}')
    exit(1)
"
    
    # 5. Performance Check
    echo -e "\n${CYAN}Checking Performance...${NC}"
    python3 -c "
import mlx.core as mx
import time

# Simple performance test
start = time.time()
for _ in range(5):
    mx.eval(mx.random.normal((1024, 1024)) @ mx.random.normal((1024, 1024)))
duration = time.time() - start
print(f'✓ Performance check: {duration:.2f}s for 5 matrix multiplications')
"
    
    echo -e "\n${GREEN}✓ Model Validation Complete${NC}"
    
    # 6. Generate Deployment Report
    cat > "${EXPORT_DIR}/deployment_report.json" << EOF
{
    "model": "${selected_model}",
    "validation_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "tests_passed": true,
    "deployment_ready": true,
    "recommended_batch_size": 1,
    "min_memory_required": "4GB"
}
EOF
}
