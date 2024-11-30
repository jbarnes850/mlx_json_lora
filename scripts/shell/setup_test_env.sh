#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check system requirements
check_system() {
    # Check for Apple Silicon
    if [[ $(uname -m) != "arm64" ]]; then
        echo -e "${RED}Error: Requires Apple Silicon Mac${NC}"
        exit 1
    fi
    
    # Check macOS version
    if [[ $(sw_vers -productVersion | cut -d. -f1) -lt 13 ]]; then
        echo -e "${RED}Error: Requires macOS 13.3 or later${NC}"
        exit 1
    fi
    
    # Check available RAM
    total_ram=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
    if [[ $total_ram -lt 8 ]]; then
        echo -e "${RED}Error: Requires at least 8GB RAM${NC}"
        exit 1
    fi
    
    # Check available disk space
    available_space=$(df -k . | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ $available_space -lt 10 ]]; then
        echo -e "${RED}Error: Requires at least 10GB free disk space${NC}"
        exit 1
    fi
}

# Create test directories and files
mkdir -p adapters
mkdir -p exported_model

# Function to print status
print_status() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Check system requirements
print_status "Checking system requirements..."
check_system

# Create and activate virtual environment
print_status "Creating isolated test environment..."
python3 -m venv .venv_test
source .venv_test/bin/activate

# Verify Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if (( $(echo "$python_version < 3.9" | bc -l) )); then
    echo -e "${RED}Error: Requires Python 3.9+${NC}"
    exit 1
fi

# Install specific versions of dependencies
print_status "Installing dependencies..."
pip install --quiet --no-cache-dir \
    "numpy==1.26.4" \
    "mlx>=0.21.0" \
    "mlx-lm" \
    "transformers" \
    "torch==2.4.1" \
    "protobuf<5.0.0" \
    "fastapi<0.111.0" \
    "setuptools>=70.0.0" \
    "aiofiles<24.0.0" \
    "langchain>=0.3.7" \
    "langchain-core>=0.3.15"

# Verify installations
print_status "Verifying installations..."
python3 -c "
import sys
import mlx
import numpy as np

# Check MLX version
mlx_version = mlx.__version__
if float(mlx_version) < 0.21:
    sys.exit('MLX version must be >= 0.21.0')

# Check NumPy version
np_version = np.__version__
if not np_version.startswith('1.26'):
    sys.exit('Incorrect NumPy version')

print('All package versions verified')
" || exit 1

print_status "Creating test weights..."
python3 -c "
import numpy as np
import json

# Create dummy LoRA weights with correct names
dummy_weights = {
    'q_proj.lora.weight': np.zeros((8, 768), dtype=np.float32),
    'k_proj.lora.weight': np.zeros((8, 768), dtype=np.float32),
    'v_proj.lora.weight': np.zeros((8, 768), dtype=np.float32),
}

np.savez('adapters/lora_weights.npz', **dummy_weights)

# Create training info
training_info = {
    'model': 'microsoft/Phi-3.5-mini-instruct',
    'completed': '2024-03-06T12:00:00Z',
    'duration': '30m',
    'final_loss': 2.1,
    'hardware': 'Apple M3 Max'
}

with open('adapters/training_info.json', 'w') as f:
    json.dump(training_info, f, indent=2)
"

echo -e "${GREEN}✓ Test environment setup complete${NC}" 