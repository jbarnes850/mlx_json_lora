#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Change to project root
cd "${PROJECT_ROOT}"

# Handle command line arguments first
if [ "$1" == "--fresh" ] || [ "$1" == "-f" ]; then
    rm -rf .recovery .tutorial_progress 2>/dev/null
    FRESH_START=1
elif [ "$1" == "--troubleshoot" ]; then
    print_troubleshooting
    exit 0
fi

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

VERSION="1.0.0"

# Initialize step tracking
CURRENT_STEP=0
TOTAL_STEPS=6

# Define models using simple arrays
MODEL_NAMES=(
    "phi-3.5-mini"
    "gemma-2-2b"
    "qwen2.5-7b"
)

MODEL_PATHS=(
    "microsoft/Phi-3.5-mini-instruct"
    "google/gemma-2-2b"
    "Qwen/Qwen2.5-7B-Instruct"
)

MODEL_RAM_REQ=(
    8  # phi-3.5-mini
    12 # gemma-2-2b
    32 # qwen2.5-7b
)

get_model_path() {
    local index=$(($1 - 1))
    echo "${MODEL_PATHS[$index]}"
}

get_model_name() {
    local index=$(($1 - 1))
    echo "${MODEL_NAMES[$index]}"
}

get_model_ram() {
    local index=$(($1 - 1))
    echo "${MODEL_RAM_REQ[$index]}"
}

# Progress tracking functions
save_progress() {
    if [ "$FRESH_START" != "1" ]; then
        mkdir -p .recovery
        echo "CURRENT_STEP=$CURRENT_STEP" > .recovery/checkpoint.sh
        echo "MODEL_CHOICE=$MODEL_CHOICE" >> .recovery/checkpoint.sh
    fi
}

resume_progress() {
    if [ -f .recovery/checkpoint.sh ] && [ "$FRESH_START" != "1" ]; then
        source .recovery/checkpoint.sh
        if [ $CURRENT_STEP -gt 0 ]; then
            print_info "Resuming from step $CURRENT_STEP"
        fi
    else
        CURRENT_STEP=0
    fi
}

# Simple ASCII logo
print_logo() {
    echo -e "${BLUE}"
    echo "     __  __ _    __  __    _     ___  ____      _     "
    echo "    |  \/  | |   \ \/ /   | |   / _ \|  _ \    / \    "
    echo "    | |\/| | |    >  <    | |  | | | | |_) |  / _ \   "
    echo "    | |  | | |___ /  \    | |__| |_| |  _ <  / ___ \  "
    echo "    |_|  |_|_____/_/\_\   |_____\___/|_| \_\/_/   \_\ "
    echo -e "${NC}"
}

# Welcome message
print_welcome() {
    clear
    print_logo
    echo
    echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════╗"
    echo -e "║                                                            ║"
    echo -e "║          MLX LoRA Fine-Tuning Framework Tutorial           ║"
    echo -e "║                                                            ║"
    echo -e "║                                           v${VERSION}           ║"
    echo -e "╚════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${BOLD}Welcome to the MLX LoRA Framework - the fastest way to fine-tune${NC}"
    echo -e "${BOLD}large language models on Apple Silicon.${NC}"
    echo
    echo -e "${BOLD}What you'll achieve:${NC}"
    echo -e "${CYAN}✓ Create a custom AI model tailored to your needs"
    echo -e "✓ Learn MLX and LoRA fine-tuning best practices"
    echo -e "✓ Deploy production-ready models on Apple Silicon"
    echo -e "✓ Join a community of MLX developers${NC}"
    echo
    echo -e "${BOLD}This tutorial will guide you through:${NC}"
    echo -e "${CYAN}1. Setting up your environment"
    echo -e "2. Preparing your training data"
    echo -e "3. Selecting and configuring your model"
    echo -e "4. Training with LoRA"
    echo -e "5. Evaluating results"
    echo -e "6. Testing your model${NC}"
    echo
    echo -e "${BOLD}${YELLOW}System Requirements:${NC}"
    echo -e "${YELLOW}• macOS with Apple Silicon (M1/M2/M3/M4)"
    echo -e "• Python 3.9+"
    echo -e "• At least 8GB of available memory"
    echo -e "• Internet connection for model download${NC}"
    echo
    echo -e "${BOLD}${CYAN}Time & Resource Estimates:${NC}"
    echo -e "${CYAN}• Environment setup: 2-3 minutes"
    echo -e "• Data preparation: 3-5 minutes"
    echo -e "• Model training: 15-30 minutes"
    echo -e "• Evaluation: 5-10 minutes"
    echo -e "• ${BOLD}Total time: ~30-45 minutes${NC}"
    echo
    echo -e "${DIM}${PURPLE}Need help? Run './run_tutorial.sh --troubleshoot' for assistance${NC}"
    echo
    echo -e "${BOLD}Press Enter to begin the tutorial...${NC}"
    read
}

# Status indicators for steps
print_status() {
    local step=$1
    local status=$2
    case $status in
        "pending")  echo -e "${DIM}○${NC} $step" ;;
        "active")   echo -e "${YELLOW}◉${NC} ${BOLD}$step${NC}" ;;
        "success")  echo -e "${GREEN}✓${NC} $step" ;;
        "error")    echo -e "${RED}✗${NC} $step" ;;
    esac
}

# Show current progress
show_step_progress() {
    clear
    print_logo
    echo
    echo -e "${BOLD}Progress:${NC}"
    
    local status
    for i in {1..6}; do
        if [ $CURRENT_STEP -gt $i ]; then
            status="success"
        elif [ $CURRENT_STEP -eq $i ]; then
            status="active"
        else
            status="pending"
        fi
        
        case $i in
            1) step="Environment Setup" ;;
            2) step="Data Preparation" ;;
            3) step="Model Selection" ;;
            4) step="Model Fine-tuning" ;;
            5) step="Evaluation" ;;
            6) step="Testing" ;;
        esac
        
        print_status "$step" "$status"
    done
    echo
}

# Function to print progress
print_progress() {
    echo -e "\nProgress:"
    
    local steps=(
        "Environment Setup"
        "Data Preparation"
        "Model Selection"
        "Model Fine-tuning"
        "Evaluation"
        "Testing"
    )
    
    for i in "${!steps[@]}"; do
        local step_num=$((i + 1))
        local step="${steps[$i]}"
        
        if [ $step_num -lt $CURRENT_STEP ]; then
            echo -e "✓ $step"
        elif [ $step_num -eq $CURRENT_STEP ]; then
            echo -e "◉ $step"
        else
            echo -e "○ $step"
        fi
    done
    echo
}

# Print functions with enhanced formatting
print_header() {
    echo -e "\n${BOLD}${BLUE}==> $1${NC}"
}

print_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo -e "\n${BOLD}${CYAN}Step $CURRENT_STEP/$TOTAL_STEPS: $1${NC}"
    save_progress
    show_step_progress
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

# Progress tracking functions
save_progress() {
    if [ "$FRESH_START" != "1" ]; then
        mkdir -p .recovery
        echo "CURRENT_STEP=$CURRENT_STEP" > .recovery/checkpoint.sh
        echo "MODEL_CHOICE=$MODEL_CHOICE" >> .recovery/checkpoint.sh
    fi
}

resume_progress() {
    if [ -f .recovery/checkpoint.sh ] && [ "$FRESH_START" != "1" ]; then
        source .recovery/checkpoint.sh
        if [ $CURRENT_STEP -gt 0 ]; then
            print_info "Resuming from step $CURRENT_STEP"
        fi
    else
        CURRENT_STEP=0
    fi
}

# Hardware detection and system requirements check
check_hardware() {
    # Check for Apple Silicon
    if [[ $(uname -m) != "arm64" ]]; then
        print_error "Apple Silicon (M1/M2/M3/M4) required"
        return 1
    fi
    print_success "Apple Silicon detected: $(sysctl -n machdep.cpu.brand_string)"

    # Check available RAM
    total_ram=$(sysctl hw.memsize | awk '{print $2}')
    ram_gb=$((total_ram / 1024 / 1024 / 1024))
    if [ "$ram_gb" -lt 8 ]; then
        print_error "Minimum 8GB RAM required (found ${ram_gb}GB)"
        return 1
    fi
    print_success "Memory: ${ram_gb}GB RAM available"

    # Check available disk space
    available_space=$(df -h . | awk 'NR==2 {print $4}' | sed 's/[A-Za-z]//g')
    if [ -n "$available_space" ] && [ "$(echo "$available_space < 10" | bc 2>/dev/null)" = "1" ]; then
        print_warning "Less than 10GB free space available (${available_space}GB). Some models may require more space."
    else
        print_success "Storage: ${available_space}GB available"
    fi

    # Check for Metal support
    if ! system_profiler SPDisplaysDataType | grep -q "Metal"; then
        print_error "Metal GPU acceleration not detected"
        return 1
    fi
    gpu_info=$(system_profiler SPDisplaysDataType | grep "Metal:")
    print_success "Metal GPU acceleration available"

    return 0
}

# System checks
check_system() {
    print_header "System Check"
    
    # Check Python version
    check_python_version || {
        return 1
    }

    # Check hardware requirements
    check_hardware || {
        print_error "Hardware requirements not met"
        return 1
    }
    
    echo
    print_success "System check completed"
    echo
    echo -e "${BOLD}${GREEN}System requirements met! Press Enter to continue...${NC}"
    read
}

# Function to install Python if needed
install_python() {
    print_info "Python 3.9+ required. Installing latest Python version..."
    
    # Check if Homebrew is installed
    if ! command -v brew >/dev/null 2>&1; then
        print_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
            print_error "Failed to install Homebrew"
            return 1
        }
        print_success "Homebrew installed"
    fi
    
    # Install pyenv using Homebrew
    if ! command -v pyenv >/dev/null 2>&1; then
        print_info "Installing pyenv..."
        brew install pyenv || {
            print_error "Failed to install pyenv"
            return 1
        }
        print_success "pyenv installed"
        
        # Add pyenv to shell
        echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
        echo 'eval "$(pyenv init -)"' >> ~/.zshrc
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    fi
    
    # Install latest Python version
    print_info "Installing Python 3.12..."
    pyenv install 3.12 || {
        print_error "Failed to install Python 3.12"
        return 1
    }
    pyenv global 3.12
    
    # Verify installation
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$python_version >= 3.9" | bc -l) -eq 1 ]]; then
        print_success "Python $python_version installed successfully"
        return 0
    else
        print_error "Python installation failed"
        return 1
    fi
}

check_python_version() {
    local python_cmd=""
    
    # Check for python3 command first
    if command -v python3 >/dev/null 2>&1; then
        python_cmd="python3"
    # Fall back to python command if python3 not found
    elif command -v python >/dev/null 2>&1; then
        python_cmd="python"
    else
        print_warning "Python not found. Would you like to install it? (y/n)"
        read -p "> " install_choice
        case $install_choice in
            [Yy]*)
                install_python || {
                    print_error "Failed to install Python. Please install Python 3.9+ manually."
                    return 1
                }
                return 0
                ;;
            *)
                print_error "Python 3.9+ is required to continue."
                return 1
                ;;
        esac
    fi

    # Check Python version
    local python_version=$($python_cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    local major_version=$(echo $python_version | cut -d. -f1)
    local minor_version=$(echo $python_version | cut -d. -f2)

    if [ "$major_version" -eq 3 ] && [ "$minor_version" -ge 9 ]; then
        print_success "Python 3.9+ available (found $python_version)"
        return 0
    else
        print_warning "Python 3.9+ required (found $python_version). Would you like to install the latest version? (y/n)"
        read -p "> " install_choice
        case $install_choice in
            [Yy]*)
                install_python || {
                    print_error "Failed to install Python. Please install Python 3.9+ manually."
                    return 1
                }
                ;;
            *)
                print_error "Python 3.9+ is required to continue."
                return 1
                ;;
        esac
    fi
}

# Progress spinner animation
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        echo -en "\b\b\b\b\b\b"
        sleep $delay
    done
    printf "    \b\b\b\b"
}

# Progress bar
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((width * current / total))
    local remaining=$((width - completed))
    
    printf "\r${CYAN}["
    printf "%${completed}s" | tr " " "="
    printf "%${remaining}s" | tr " " " "
    printf "] %d%%${NC}" $percentage
}

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 could not be found. Please install it first."
        exit 1
    fi
}

# Function to setup environment
setup_environment() {
    print_step "Setting up Environment"
    CURRENT_STEP=1
    save_progress
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv || {
            print_error "Failed to create virtual environment"
            return 1
        }
    fi

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate || {
        print_error "Failed to activate virtual environment"
        return 1
    }

    # Install required packages
    print_info "Installing required packages..."
    pip install --upgrade pip >/dev/null 2>&1
    pip install mlx transformers rich psutil >/dev/null 2>&1 || {
        print_error "Failed to install required packages"
        return 1
    }

    print_success "Environment setup completed"
    echo
    echo -e "${BOLD}${GREEN}Environment is ready! Press Enter to continue...${NC}"
    read
    return 0
}

# Add these functions BEFORE select_model()
create_phi_config() {
    local config_path=$1
    mkdir -p "$(dirname "$config_path")"
    cat > "$config_path" << 'EOF'
# Phi-3.5 Mini specific configuration
model:
  name: "microsoft/Phi-3.5-mini-instruct"
  path: "microsoft/Phi-3.5-mini-instruct"
  batch_size: 1
  max_seq_length: 2048
  learning_rate: 2.0e-4
  num_layers: 32

# Training parameters
training:
  seed: 42
  iters: 600
  val_batches: 20
  steps_per_report: 10
  steps_per_eval: 50
  save_every: 100
  grad_checkpoint: true

# LoRA configuration for Phi-3.5
lora:
  r: 8
  alpha: 32
  dropout: 0.1
  target_modules:
    - "self_attn.q_proj"
    - "self_attn.k_proj"
    - "self_attn.v_proj"
    - "self_attn.o_proj"
    - "mlp.gate_proj"
    - "mlp.up_proj"
    - "mlp.down_proj"
  lora_layers: 32
EOF
    print_success "Created Phi-3.5 configuration"
}

verify_or_create_config() {
    local model_choice=$1
    local config_path=$2
    
    # Create configs directory if it doesn't exist
    mkdir -p "$(dirname "$config_path")"
    
    # If config doesn't exist, create it
    if [ ! -f "$config_path" ]; then
        case $model_choice in
            1)
                create_phi_config "$config_path"
                ;;
            2)
                create_gemma_config "$config_path"
                ;;
            3)
                create_qwen_config "$config_path"
                ;;
            *)
                print_error "Invalid model choice: $model_choice"
                return 1
                ;;
        esac
    fi
    
    # Verify config was created
    if [ ! -f "$config_path" ]; then
        print_error "Failed to create config file: $config_path"
        return 1
    fi
    
    print_success "Model configuration verified: $config_path"
    return 0
}

# Then update select_model to use absolute paths
select_model() {
    print_step "Model Selection"
    CURRENT_STEP=3
    save_progress
    
    echo -e "\nAvailable Models:"
    echo -e "${CYAN}1. Phi-3.5 Mini (3B)${NC}"
    echo -e "   - Optimized for instruction following"
    echo -e "   - Memory: ~8GB required"
    echo -e "   - Best for: General tasks, coding, instruction following"
    echo
    echo -e "${CYAN}2. Gemma 2-2B${NC}"
    echo -e "   - Google's latest compact model"
    echo -e "   - Memory: ~12GB required"
    echo -e "   - Best for: Balanced performance, efficiency"
    echo
    echo -e "${CYAN}3. Qwen 2.5 7B${NC}"
    echo -e "   - Advanced multilingual model"
    echo -e "   - Memory: ~32GB required"
    echo -e "   - Best for: Complex tasks, multilingual support"
    echo
    
    while true; do
        read -p "Select a model (1-3): " MODEL_CHOICE
        case $MODEL_CHOICE in
            1)
                MODEL_CONFIG="configs/models/phi3.yaml"
                break
                ;;
            2)
                MODEL_CONFIG="configs/models/gemma.yaml"
                break
                ;;
            3)
                MODEL_CONFIG="configs/models/qwen.yaml"
                break
                ;;
            *)
                print_error "Invalid selection. Please choose 1, 2, or 3."
                ;;
        esac
    done
    
    # Verify the config exists
    if [ ! -f "$MODEL_CONFIG" ]; then
        print_error "Model configuration not found: $MODEL_CONFIG"
        print_info "Please ensure the model configuration files exist in configs/models/"
        return 1
    fi
    
    print_success "Model selected: $(get_model_name $MODEL_CHOICE)"
    export MLX_MODEL_CONFIG="$MODEL_CONFIG"
    return 0
}

# Function to prepare data
prepare_data() {
    print_step "Data Preparation"
    CURRENT_STEP=2
    save_progress
    
    echo -e "\nChoose your dataset:\n"
    
    echo -e "${BOLD}1) Example Datasets${NC}"
    echo -e "   ${CYAN}• Pre-formatted training data"
    echo -e "   • Instant start, no preparation needed"
    echo -e "   • Great for learning${NC}"
    echo
    
    echo -e "${BOLD}2) Custom Dataset${NC}"
    echo -e "   ${CYAN}• Your own training data"
    echo -e "   • We'll help with formatting"
    echo -e "   • Perfect for specific use cases and private data${NC}"
    echo

    read -p "Enter your choice (1-2): " data_choice
    case $data_choice in
        1)
            select_example_dataset
            ;;
        2)
            setup_custom_dataset
            ;;
        *)
            print_warning "Invalid choice. Using example dataset."
            select_example_dataset
            ;;
    esac
    
    save_progress
}

# Function to select example dataset
select_example_dataset() {
    echo -e "\n${BOLD}Choose an example dataset:${NC}\n"
    echo -e "${CYAN}1. SQL Generation${NC}"
    echo -e "   • Generate SQL queries from natural language"
    echo -e "   • Based on WikiSQL dataset"
    echo -e "   • Great for database applications"
    echo
    echo -e "${CYAN}2. General Instructions${NC}"
    echo -e "   • Follow complex instructions"
    echo -e "   • Helpful and safe responses"
    echo -e "   • Ideal for chatbots"
    echo
    echo -e "${CYAN}3. Code Generation${NC}"
    echo -e "   • Programming assistance"
    echo -e "   • Multiple languages"
    echo -e "   • Perfect for coding tools"
    echo

    read -p "Select dataset (1-3): " dataset_number
    case $dataset_number in
        1|2|3)
            print_success "Selected example dataset $dataset_number"
            ;;
        *)
            print_warning "Invalid choice. Using SQL Generation dataset."
            dataset_number=1
            ;;
    esac
}

# Function to setup custom dataset
setup_custom_dataset() {
    echo -e "\n${BOLD}Custom Dataset Setup${NC}\n"
    echo -e "${CYAN}Choose your data format:${NC}\n"
    echo -e "1) Chat Conversations"
    echo -e "   • Multiple roles (system/user/assistant)"
    echo -e "   • Perfect for chatbot training"
    echo
    echo -e "2) Tool Interactions"
    echo -e "   • Function calling examples"
    echo -e "   • Great for AI assistants"
    echo
    echo -e "3) Simple Completions"
    echo -e "   • Input/output pairs"
    echo -e "   • Ideal for straightforward tasks"
    echo
    echo -e "4) Raw Text"
    echo -e "   • Custom processing"
    echo -e "   • Maximum flexibility"
    echo

    read -p "Select format (1-4): " format_choice
    case $format_choice in
        1) setup_chat_format ;;
        2) setup_tool_format ;;
        3) setup_completion_format ;;
        4) setup_raw_format ;;
        *)
            print_warning "Invalid choice. Using chat format."
            setup_chat_format
            ;;
    esac
}

# Function to validate datasets
validate_datasets() {
    local required_files=("train.jsonl" "test.jsonl" "valid.jsonl")
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "data/$file" ] || [ ! -s "data/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        print_warning "Missing or empty dataset files: ${missing_files[*]}"
        print_info "Downloading example datasets..."
        python "${PROJECT_ROOT}/mlx_lora_trainer/scripts/python/prepare_data.py" \
            --dataset "mlx-community/code-instructions" \
            --output-dir "data" \
            --split "all" || {
                print_error "Failed to download datasets"
                return 1
            }
    fi
    return 0
}

# Function to run training
run_training() {
    print_step "Model Fine-tuning"
    CURRENT_STEP=4
    save_progress
    
    print_info "Preparing for training..."
    echo
    print_info "Configuration:"
    echo -e "${CYAN}• Model: $(get_model_name $MODEL_CHOICE)"
    echo -e "• Dataset: $([ $data_choice -eq 1 ] && echo "Example Dataset $dataset_number" || echo "Custom Dataset")"
    echo -e "• Device: $(sysctl -n machdep.cpu.brand_string)"
    echo -e "• Memory: ${ram_gb}GB RAM${NC}"
    echo

    # Check RAM before proceeding
    check_available_ram || {
        print_error "RAM check failed. Please choose a smaller model or free up memory."
        return 1
    }
    
    # Create necessary directories with proper permissions
    for dir in "data" "logs" "adapters" ".cache" "exported_model"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            chmod 755 "$dir"
        fi
    done
    
    # Download model if needed
    download_model || return 1
    
    # Validate datasets
    validate_datasets || {
        print_error "Dataset validation failed"
        return 1
    }
    
    # Validate model configuration
    if [ ! -f "${PROJECT_ROOT}/${MODEL_CONFIG}" ]; then
        print_error "Model configuration not found: ${MODEL_CONFIG}"
        return 1
    fi
    
    # Check available disk space
    available_space=$(df -k . | awk 'NR==2 {print $4}')
    required_space=$((10 * 1024 * 1024))  # 10GB in KB
    if [ $available_space -lt $required_space ]; then
        print_error "Insufficient disk space. At least 10GB required."
        return 1
    fi
    
    print_info "Starting LoRA fine-tuning process..."
    echo -e "${CYAN}Training can be safely interrupted with Ctrl+C"
    echo -e "Progress and weights are automatically saved${NC}"
    echo
    
    # Show estimated time and resource usage
    ram_required=$(get_model_ram $MODEL_CHOICE)
    print_info "Estimated resource usage:"
    echo -e "• RAM: ${ram_required}GB"
    echo -e "• Training time: ~30-60 minutes"
    echo -e "• Disk space: ~5-10GB"
    echo
    
    read -p "Press Enter to begin training (or Ctrl+C to cancel)..."
    echo
    
    # Start training with automatic recovery
    (
        # Improved error handling with exit code
        trap 'handle_training_interrupt $?' INT TERM
        
        # Use train_lora.sh instead of direct Python call
        "${PROJECT_ROOT}/scripts/shell/train_lora.sh" \
            --model-config "${MODEL_CONFIG}" \
            --train-data "data/train.jsonl" \
            --valid-data "data/valid.jsonl" || {
                print_error "Training failed"
                return 1
            }
        
        training_status=$?
        
        if [ $training_status -eq 0 ]; then
            # After successful training, automatically export
            print_info "Exporting model..."
            "${PROJECT_ROOT}/scripts/shell/export.sh" \
                --model-config "${MODEL_CONFIG}" \
                --adapter-path "adapters/lora_weights.npz" \
                --output-dir "exported_model" || {
                    print_warning "Export failed, but training was successful"
                }
            
            # Start chat session automatically
            print_info "Starting chat session with your fine-tuned model..."
            "${PROJECT_ROOT}/scripts/shell/chat.sh" \
                --model-config "${MODEL_CONFIG}" \
                --adapter-path "adapters/lora_weights.npz"
            
            handle_training_success
            return 0
        else
            print_error "Training failed with status: $training_status"
            print_info "Check logs/training.log for details"
            print_info "You can resume training by running: ./scripts/shell/run_tutorial.sh --resume"
            return 1
        fi
    )
}

handle_training_interrupt() {
    echo
    print_warning "Training interrupted. Progress has been saved."
    print_info "Resume training with: ./scripts/shell/run_tutorial.sh --resume"
    exit 1
}

# Function to test model
test_model() {
    print_step "Testing Your Fine-tuned Model"
    CURRENT_STEP=6
    save_progress
    
    # Validate required files exist
    if [ ! -f "adapters/lora_weights.npz" ]; then
        print_error "LoRA weights not found. Please complete training first."
        return 1
    fi
    
    print_info "Starting interactive chat session..."
    echo -e "${CYAN}Type your prompts and press Enter. Use 'exit' or Ctrl+C to quit.${NC}"
    echo
    
    # Start interactive session with proper error handling
    (
        trap 'echo -e "\n${YELLOW}Chat session ended${NC}"' INT TERM
        
        # Use chat.sh instead of direct Python call
        "${PROJECT_ROOT}/scripts/shell/chat.sh" \
            --model-config "${MODEL_CONFIG}" \
            --adapter-path "adapters/lora_weights.npz" || {
                print_error "Chat session failed"
                return 1
            }
            
        chat_status=$?
        
        if [ $chat_status -eq 0 ]; then
            print_success "Chat session completed successfully!"
            print_info "Start a new chat anytime with: ./scripts/shell/chat.sh"
            return 0
        else
            print_error "Chat session failed"
            print_info "Check the error messages above"
            return 1
        fi
    )
}

# Function to run evaluation
run_evaluation() {
    print_step "Evaluation"
    CURRENT_STEP=5
    save_progress
    
    # Run evaluation
    print_info "Running evaluation metrics..."
    python evaluate.py --model-path $(get_model_path $MODEL_CHOICE) \
                      --adapter-path adapters/lora_weights.npz \
                      --test-file data/test.jsonl
    
    # Run benchmarks
    print_info "Running performance benchmarks..."
    python scripts/benchmark.py --model-path $(get_model_path $MODEL_CHOICE) \
                              --adapter-path adapters/lora_weights.npz \
                              --test-file data/test.jsonl
}

# Function to detect hardware and optimize
detect_hardware() {
    print_step "Detecting Hardware Configuration"
    
    # Detect Apple Silicon model
    chip_info=$(sysctl -n machdep.cpu.brand_string)
    total_memory=$(sysctl hw.memsize | awk '{print $2}')
    gpu_info=$(system_profiler SPDisplaysDataType | grep "Metal:")
    
    # Determine optimal configuration
    case $chip_info in
        *"M1"*)
            if [[ $total_memory -ge 16 ]]; then
                RECOMMENDED_MODEL="gemma-2-2b"
                BATCH_SIZE=2
            else
                RECOMMENDED_MODEL="phi-3.5-mini"
                BATCH_SIZE=4
            fi
            ;;
        *"M2"*)
            if [[ $total_memory -ge 32 ]]; then
                RECOMMENDED_MODEL="qwen2.5-7b"
                BATCH_SIZE=1
            elif [[ $total_memory -ge 16 ]]; then
                RECOMMENDED_MODEL="gemma-2-2b"
                BATCH_SIZE=2
            else
                RECOMMENDED_MODEL="phi-3.5-mini"
                BATCH_SIZE=4
            fi
            ;;
        *"M3"* | *"M4"*)
            if [[ $total_memory -ge 32 ]]; then
                RECOMMENDED_MODEL="qwen2.5-7b"
                BATCH_SIZE=2
            else
                RECOMMENDED_MODEL="gemma-2-2b"
                BATCH_SIZE=4
            fi
            ;;
    esac
    
    # Print hardware info
    print_info "Hardware Configuration:"
    print_info "• Processor: $chip_info"
    print_info "• Memory: ${total_memory}GB"
    print_info "• GPU: $gpu_info"
    print_info "\nRecommended Configuration:"
    print_info "• Model: $RECOMMENDED_MODEL"
    print_info "• Batch Size: $BATCH_SIZE"
    
    # Save hardware config
    echo "HARDWARE_CONFIG={"
    echo "  \"chip\": \"$chip_info\","
    echo "  \"memory\": $total_memory,"
    echo "  \"recommended_model\": \"$RECOMMENDED_MODEL\","
    echo "  \"batch_size\": $BATCH_SIZE"
    echo "}" > .hardware_config
}

# Auto-recovery system
setup_auto_recovery() {
    print_step "Setting Up Auto-Recovery"
    
    # Create recovery directory
    mkdir -p .recovery
    
    # Setup checkpoint system
    cat > .recovery/checkpoint.sh << 'EOF'
#!/bin/bash

# Save checkpoint
save_checkpoint() {
    cp lora_config.yaml .recovery/
    cp -r data .recovery/
    cp -r adapters .recovery/
    echo "CHECKPOINT_TIME=$(date +%s)" > .recovery/checkpoint_info
}

# Restore from checkpoint
restore_checkpoint() {
    if [ -f .recovery/checkpoint_info ]; then
        cp .recovery/lora_config.yaml .
        cp -r .recovery/data .
        cp -r .recovery/adapters .
        print_success "Restored from checkpoint"
    fi
}

# Auto-backup every 5 minutes
(while true; do
    save_checkpoint
    sleep 300
done) &
EOF
    
    chmod +x .recovery/checkpoint.sh
    
    print_success "Auto-recovery system initialized"
}

# Enhanced error handling
handle_error() {
    error_code=$1
    error_msg=$2
    
    print_error "Error occurred: $error_msg"
    
    case $error_code in
        1) # Environment setup failed
            print_info "Attempting to fix environment..."
            clean_venv
            setup_environment
            ;;
        2) # Model download failed
            print_info "Retrying model download..."
            for i in {1..3}; do
                if download_model; then
                    break
                fi
                sleep 5
            done
            ;;
        3) # Training interrupted
            print_info "Restoring from last checkpoint..."
            restore_checkpoint
            resume_training
            ;;
        *)
            print_troubleshooting
            ;;
    esac
}

# Enhanced help system
show_help() {
    local context="$1"
    
    case "$context" in
        "model")
            print_header "Model Selection Help"
            echo -e "${CYAN}Available Models:${NC}"
            echo -e "• ${BOLD}Phi-3.5 Mini${NC}: Best for quick experiments (4GB RAM)"
            echo -e "  - Fast training, good for testing workflows"
            echo -e "  - Suitable for text generation and completion"
            echo
            echo -e "${CYAN}2. Gemma 2-2B${NC}"
            echo -e "   - Google's latest compact model"
            echo -e "   - Memory: ~12GB required"
            echo -e "   - Best for: Balanced performance, efficiency"
            echo
            echo -e "${CYAN}3. Qwen 2.5 7B${NC}"
            echo -e "   - Advanced multilingual model"
            echo -e "   - Memory: ~32GB required"
            echo -e "   - Best for: Complex tasks, multilingual support"
            ;;
        "data")
            print_header "Dataset Help"
            echo -e "${CYAN}Dataset Options:${NC}"
            echo -e "1. ${BOLD}Example Datasets${NC}"
            echo -e "   • Pre-formatted and ready to use"
            echo -e "   • Great for learning the workflow"
            echo -e "   • Choose based on your use case:"
            echo -e "     - SQL: Database query generation"
            echo -e "     - Instructions: General task completion"
            echo -e "     - Code: Programming assistance"
            echo
            echo -e "2. ${BOLD}Custom Dataset${NC}"
            echo -e "   • Use your own training data"
            echo -e "   • Supported formats:"
            echo -e "     - JSONL (recommended)"
            echo -e "     - CSV"
            echo -e "     - Text files"
            echo -e "   • Auto-conversion available"
            ;;
        "training")
            print_header "Training Help"
            echo -e "${CYAN}Training Process:${NC}"
            echo -e "• ${BOLD}Auto-checkpointing${NC} every 100 steps"
            echo -e "• Safe to interrupt with Ctrl+C"
            echo -e "• Progress automatically saved"
            echo
            echo -e "${CYAN}Optimization Tips:${NC}"
            echo -e "• Start with smaller datasets to test"
            echo -e "• Monitor training loss for convergence"
            echo -e "• Use evaluation metrics to guide decisions"
            echo
            echo -e "${CYAN}Common Issues:${NC}"
            echo -e "• Out of memory: Try a smaller model"
            echo -e "• Slow training: Check CPU usage"
            echo -e "• Poor results: Adjust learning rate"
            ;;
        "evaluation")
            print_header "Evaluation Help"
            echo -e "${CYAN}Metrics Explained:${NC}"
            echo -e "• ${BOLD}Loss${NC}: Lower is better"
            echo -e "• ${BOLD}Perplexity${NC}: Measure of model confidence"
            echo -e "• ${BOLD}Speed${NC}: Tokens per second"
            echo
            echo -e "${CYAN}Benchmarking:${NC}"
            echo -e "• Compare against base model"
            echo -e "• Check memory usage"
            echo -e "• Measure inference speed"
            ;;
        *)
            print_header "MLX LoRA Tutorial Help"
            echo -e "${CYAN}Available Commands:${NC}"
            echo -e "• ${BOLD}help${NC}: Show this help message"
            echo -e "• ${BOLD}help model${NC}: Model selection guide"
            echo -e "• ${BOLD}help data${NC}: Dataset preparation help"
            echo -e "• ${BOLD}help training${NC}: Training process guide"
            echo -e "• ${BOLD}help evaluation${NC}: Evaluation help"
            echo -e "• ${BOLD}quit${NC}: Safely exit the tutorial"
            echo
            echo -e "${YELLOW}Quick Tips:${NC}"
            echo -e "• Use Ctrl+C to safely interrupt training"
            echo -e "• Progress is automatically saved"
            echo -e "• Run with --resume to continue"
            ;;
    esac
    
    echo
    echo -e "${GREEN}Need more help? Check out our docs:${NC}"
    echo -e "• Docs: github.com/jbarnes850/mlx-lora/docs"
    echo
}

handle_command() {
    local cmd="$1"
    case "$cmd" in
        "help")
            show_help "$2"
            return 0
            ;;
        "quit")
            print_info "Safely exiting tutorial..."
            cleanup_and_exit
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to show completion message
show_completion() {
    clear
    print_logo
    echo -e "${BOLD}${GREEN}🎉 Congratulations! Your Model is Ready${NC}"
    echo
    echo -e "${CYAN}What you've accomplished:${NC}"
    echo -e "✓ Set up MLX development environment"
    echo -e "✓ Prepared and processed training data"
    echo -e "✓ Fine-tuned $(get_model_name $MODEL_CHOICE) with LoRA"
    echo -e "✓ Created a production-ready model"
    echo
    echo -e "${BOLD}Your model is ready at:${NC}"
    echo -e "• ${CYAN}Adapter weights: adapters/lora_weights.safetensors${NC}"
    echo -e "• ${CYAN}Configuration: adapters/config.json${NC}"
    echo
    echo -e "${BOLD}${YELLOW}Try Your Model:${NC}"
    echo -e "1. ${YELLOW}Start chatting${NC}"
    echo -e "2. ${YELLOW}Export model${NC}"
    echo -e "3. ${YELLOW}Exit${NC}"
    echo
    
    while true; do
        read -p "Choose an option (1-3): " choice
        case $choice in
            1)
                # Launch chat with proper model config
                "${PROJECT_ROOT}/scripts/shell/chat.sh" \
                    --model-config "${MLX_MODEL_CONFIG}" \
                    --adapter-path "adapters/lora_weights.safetensors"
                break
                ;;
            2)
                "${PROJECT_ROOT}/scripts/shell/export.sh" --merge
                break
                ;;
            3)
                echo -e "${GREEN}Thank you for using MLX LoRA Framework!${NC}"
                exit 0
                ;;
            *)
                echo -e "${YELLOW}Please choose 1, 2, or 3${NC}"
                ;;
        esac
    done
}

# Function to handle successful training completion
handle_training_success() {
    print_success "Training completed successfully!"
    
    # Save training metadata
    cat > "adapters/training_info.json" << EOF
{
    "model": "$(get_model_name $MODEL_CHOICE)",
    "completed": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "duration": "$TRAINING_DURATION",
    "final_loss": "$FINAL_LOSS",
    "hardware": "$(sysctl -n machdep.cpu.brand_string)"
}
EOF
    
    # Show completion message and start chat
    show_completion
}

# Add this function before main()
verify_all_configs() {
    print_info "Verifying model configurations..."
    
    # Create configs directory if it doesn't exist
    mkdir -p "${PROJECT_ROOT}/configs/models"
    
    # List of required config files
    local required_configs=(
        "phi3.yaml:microsoft/Phi-3.5-mini-instruct"
        "gemma.yaml:google/gemma-2-2b"
        "qwen.yaml:Qwen/Qwen2.5-7B-Instruct"
    )
    
    # Check each config file
    for config_pair in "${required_configs[@]}"; do
        local config_file="${config_pair%%:*}"
        local model_name="${config_pair#*:}"
        local config_path="${PROJECT_ROOT}/configs/models/${config_file}"
        
        if [ ! -f "$config_path" ]; then
            print_error "Missing required config: ${config_file}"
            print_info "Expected path: ${config_path}"
            return 1
        else
            # Verify the config contains the correct model name
            if ! grep -q "name: \"${model_name}\"" "$config_path"; then
                print_error "Invalid model name in ${config_file}"
                print_info "Expected model: ${model_name}"
                return 1
            fi
        fi
    done
    
    print_success "All model configurations verified"
    return 0
}

# Update main() to use this verification
main() {
    # Initialize
    if [ "$FRESH_START" == "1" ]; then
        CURRENT_STEP=1
    fi
    
    print_logo
    print_welcome
    
    # Verify configs exist and are valid
    verify_all_configs || {
        print_error "Model configuration verification failed"
        print_info "Please ensure all model configuration files exist in configs/models/"
        exit 1
    }
    
    # Step 1: Environment Setup
    print_header "Step 1: Environment Setup"
    check_system || exit 1
    setup_environment || exit 1
    print_progress
    
    # Step 2: Data Preparation
    print_header "Step 2: Data Preparation"
    print_info "You can use our example datasets or bring your own custom data."
    prepare_data || exit 1
    print_progress
    
    # Step 3: Model Selection
    print_header "Step 3: Model Selection and Configuration"
    print_info "Choose a model based on your hardware and requirements:"
    select_model || exit 1
    print_progress
    
    # Step 4: Training
    print_header "Step 4: Training with LoRA"
    print_info "Starting the fine-tuning process. This may take 15-30 minutes."
    run_training || exit 1
    print_progress
    
    # Step 5: Evaluation
    print_header "Step 5: Evaluation"
    print_info "Evaluating model performance and generating metrics."
    run_evaluation || exit 1
    print_progress
    
    # Step 6: Testing
    print_header "Step 6: Testing Your Model"
    print_info "Let's test your fine-tuned model in an interactive chat session."
    test_model || exit 1
    print_progress
    
    # Tutorial Complete
    print_header " Tutorial Complete!"
    echo
    echo -e "${GREEN}Congratulations! You've successfully:"
    echo -e "✓ Set up your environment"
    echo -e "✓ Prepared your data"
    echo -e "✓ Fine-tuned a model with LoRA"
    echo -e "✓ Evaluated performance"
    echo -e "✓ Tested your model${NC}"
    echo
    echo -e "${BOLD}Next Steps:${NC}"
    echo -e "1. Run inference anytime:    ${CYAN}./scripts/shell/run_inference.sh${NC}"
    echo -e "2. Train new models:         ${CYAN}./scripts/shell/train_lora.sh${NC}"
    echo -e "3. View training logs:       ${CYAN}less logs/training.log${NC}"
    echo
    echo -e "${YELLOW}Your trained model adapter is saved in: adapters/lora_weights.npz${NC}"
    echo
}

# Function to check available RAM
check_available_ram() {
    local required_ram=$(get_model_ram $MODEL_CHOICE)
    local available_ram=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
    if [ $available_ram -lt $required_ram ]; then
        print_error "Insufficient RAM. Model requires ${required_ram}GB, but only ${available_ram}GB available"
        return 1
    fi
    print_success "RAM check passed: ${available_ram}GB available"
    return 0
}

# Function to download model
download_model() {
    local model_path=$(get_model_path $MODEL_CHOICE)
    print_info "Downloading ${model_path}..."
    
    # Set optimized environment variables for faster downloads
    export HF_HUB_ENABLE_HF_TRANSFER=1          # Enable fast transfer
    export HF_HUB_DOWNLOAD_WORKERS=16           # More parallel workers
    export HF_DATASETS_OFFLINE=1                # Use cached files when available
    export HF_HUB_DISABLE_PROGRESS_BARS=1       # Reduce overhead
    export HF_HUB_DISABLE_TELEMETRY=1          # Disable telemetry
    export HF_HUB_DOWNLOAD_TIMEOUT=300         # Longer timeout for large files
    
    # Additional optimization for Apple Silicon
    if [ "$(uname -m)" = "arm64" ]; then
        export HF_HUB_DISABLE_SYMLINKS=1        # Better performance on Apple Silicon
    fi
    
    # Create cache directory if it doesn't exist
    mkdir -p "${PROJECT_ROOT}/.cache"
    
    # Check if model is already downloaded
    if [ -d "${PROJECT_ROOT}/.cache/${model_path}" ]; then
        print_success "Model already downloaded"
        return 0
    fi
    
    # Download model using optimized Python script
    PYTHONWARNINGS="ignore::UserWarning" python "${PROJECT_ROOT}/mlx_lora_trainer/scripts/python/download.py" \
        --model "$model_path" \
        --show-progress || {
            print_error "Failed to download model"
            return 1
        }
    
    print_success "Model downloaded successfully"
    
    # Clean up environment variables
    unset HF_HUB_ENABLE_HF_TRANSFER
    unset HF_HUB_DOWNLOAD_WORKERS
    unset HF_DATASETS_OFFLINE
    unset HF_HUB_DISABLE_SYMLINKS
    unset HF_HUB_DISABLE_PROGRESS_BARS
    unset HF_HUB_DISABLE_TELEMETRY
    unset HF_HUB_DOWNLOAD_TIMEOUT
    
    return 0
}

# Function to verify model configuration
verify_model_config() {
    local model_name=$1
    local config_file=""
    
    # Create configs directory if it doesn't exist
    mkdir -p "${PROJECT_ROOT}/configs/models"
    
    case $model_name in
        "microsoft/Phi-3.5-mini-instruct")
            config_file="phi3.yaml"
            ;;
        "google/gemma-2-2b")
            config_file="gemma.yaml"
            ;;
        "Qwen/Qwen2.5-7B-Instruct")
            config_file="qwen.yaml"
            ;;
        *)
            print_error "Unknown model: $model_name"
            return 1
            ;;
    esac
    
    local full_path="${PROJECT_ROOT}/configs/models/${config_file}"
    
    # If config doesn't exist, create it using test_config.py
    if [ ! -f "$full_path" ]; then
        print_info "Creating model configuration: ${config_file}"
        python "${PROJECT_ROOT}/mlx_lora_trainer/scripts/python/test_config.py" || {
            print_error "Failed to create model configuration"
            return 1
        }
    fi
    
    # Verify config exists after potential creation
    if [ ! -f "$full_path" ]; then
        print_error "Model configuration not found: ${full_path}"
        print_info "Expected config file: ${config_file}"
        print_info "Current directory: $(pwd)"
        print_info "Project root: ${PROJECT_ROOT}"
        ls -la "${PROJECT_ROOT}/configs/models/" || true
        return 1
    fi
    
    export MLX_MODEL_CONFIG="$full_path"
    print_success "Using model configuration: $full_path"
    return 0
}

# Add this function near the other verification functions
verify_python_env() {
    # Check for required Python packages
    python -c "import huggingface_hub" 2>/dev/null || {
        print_error "huggingface_hub not found. Installing required packages..."
        pip install huggingface_hub transformers rich || return 1
    }
    return 0
}

# Run main function
main
