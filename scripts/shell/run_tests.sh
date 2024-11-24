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

# Install development dependencies
print_header "Installing development dependencies"
pip install -r "${PROJECT_ROOT}/requirements-dev.txt"

# Run code formatting
print_header "Running code formatters"
black "${PROJECT_ROOT}/mlx_lora_trainer" "${PROJECT_ROOT}/tests"
isort "${PROJECT_ROOT}/mlx_lora_trainer" "${PROJECT_ROOT}/tests"

# Run linting
print_header "Running linters"
flake8 "${PROJECT_ROOT}/mlx_lora_trainer" "${PROJECT_ROOT}/tests"
mypy "${PROJECT_ROOT}/mlx_lora_trainer"

# Run tests with coverage
print_header "Running tests with coverage"
pytest "${PROJECT_ROOT}/tests" "$@"

# Check exit code
if [ $? -eq 0 ]; then
    print_success "All tests passed!"
    exit 0
else
    print_error "Tests failed!"
    exit 1
fi
