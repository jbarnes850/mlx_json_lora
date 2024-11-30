#!/bin/bash

# Setup test environment first
./scripts/shell/setup_test_env.sh || {
    echo "Failed to setup test environment"
    exit 1
}

# Test inference
echo "Testing inference..."
./scripts/shell/run_inference.sh --non-interactive \
    --prompt "What is machine learning?" || {
    echo "Inference test failed"
    exit 1
}

# Test chat
echo "Testing chat..."
./scripts/shell/chat.sh --non-interactive \
    --prompt "Hello, how are you?" || {
    echo "Chat test failed"
    exit 1
}

# Test export
echo "Testing export..."
./scripts/shell/export.sh --no-merge || {
    echo "Export test failed"
    exit 1
}

# Add end-to-end tests
test_full_workflow() {
    # Test data preparation
    ./scripts/prepare_data.sh || exit 1
    
    # Test training
    ./scripts/train.sh --model phi3 --test-mode || exit 1
    
    # Test inference
    ./scripts/inference.sh --model phi3 --prompt "Test prompt" || exit 1
    
    # Test export
    ./scripts/export.sh --model phi3 || exit 1
}

echo "All tests completed successfully!" 