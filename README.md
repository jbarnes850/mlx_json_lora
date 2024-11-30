# MLX LoRA Trainer: Professional Model Fine-Tuning on Apple Silicon

> Transform your Mac into a powerful AI development workstation. Built on Apple's MLX framework and optimized for Metal performance.

<div align="center">

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MLX Version](https://img.shields.io/badge/MLX-0.18.1-green.svg)](https://github.com/ml-explore/mlx)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

</div>

## 🎯 Overview

MLX LoRA Trainer is a professional-grade framework for fine-tuning large language models on Apple Silicon. Whether you're an ML engineer, AI researcher, or developer exploring AI, this framework provides:

- **Hardware-Aware Optimization**: Automatic model selection and configuration based on your Mac's capabilities
- **Memory-Efficient Training**: Advanced techniques like gradient checkpointing and LoRA for efficient fine-tuning
- **Production-Ready Pipeline**: From data preparation to model deployment
- **Developer-First Experience**: Clear workflows, comprehensive logging, and intuitive CLI

## ⚡️ Quick Start (2 minutes)

Get started instantly with our quickstart script:

```bash
git clone https://github.com/jbarnes850/mlx_lora_trainer.git
cd mlx_lora_trainer
bash scripts/shell/quickstart.sh
```

The quickstart will:

1. Analyze your hardware
2. Select the optimal model and configuration
3. Download a curated training dataset
4. Start a training run optimized for your system

For a more comprehensive experience:

```bash
bash scripts/shell/run_tutorial.sh
```

## 🎯 Key Features

### 1. Intelligent Hardware Detection

- Automatically detects your Mac's capabilities
- Optimizes batch sizes and model selection
- Prevents out-of-memory errors before they happen

### 2. Supported Models

| Model | Parameters | Min RAM | Strengths |
|-------|------------|---------|-----------|
| [Phi-3.5 Mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3B | 8GB | Fast training, excellent code generation |
| [Gemma 2-2B](https://huggingface.co/google/gemma-2-2b) | 2.2B | 12GB | Efficient, strong reasoning |
| [Qwen 2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7B | 32GB | Advanced tasks, multilingual support |

### 3. Training Options

- **Quick Training**: ~15-30 minutes for initial results
- **Full Training**: Customizable epochs and parameters
- **Custom Datasets**: Support for your own training data

### 4. Advanced Features

- Gradient checkpointing for memory efficiency
- LoRA adapter training for quick iterations
- Automatic mixed precision (AMP)
- Model merging and quantization
- Ollama export support

## 💻 System Requirements

### Hardware

- Apple Silicon Mac (M1/M2/M3/M4)
- RAM:
  - Minimum: 8GB (Phi-3.5 Mini)
  - Recommended: 16GB+ (Gemma 2-2B)
  - Advanced: 32GB+ (Qwen 2.5 7B)

### Software

- macOS Sonoma or newer
- Python 3.9+
- MLX 0.18.1+

## 📚 Usage Guide

### 1. Data Preparation

```json
{
    "conversations": [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."}
    ]
}
```

### 2. Training Configuration

```yaml
model:
  name: "google/gemma-2-2b"
  batch_size: 4
  max_seq_length: 2048

training:
  learning_rate: 1e-4
  num_epochs: 3
  grad_checkpoint: true

lora:
  rank: 8
  alpha: 32
  target_modules: ["q_proj", "v_proj"]
```

### 3. Export Options

```bash
# Export for local use
./scripts/shell/export.sh --merge

# Export for Ollama
./scripts/shell/export.sh --quantize
```

## 🔍 Technical Details

### Memory Optimization

- Gradient checkpointing reduces memory by ~60%
- LoRA reduces trainable parameters by >99%
- Automatic batch size optimization
- Smart attention caching

### Training Performance

Approximate training times per epoch:

- M1 Pro/Max: Base training time
- M2 Pro/Max: ~20% faster
- M3 Pro/Max: ~40% faster

### MLX Integration

- Native Metal performance
- Efficient tensor operations
- Hardware-accelerated training
- Optimized memory management

## 🛠 Advanced Usage

### Custom Dataset Training

```bash
# Prepare your data
python -m mlx_lora_trainer.scripts.python.prepare_data \
    --input your_data.json \
    --output data/custom_train.jsonl

# Start training
./scripts/shell/train_lora.sh --config configs/custom_config.yaml
```

### Model Export and Deployment

```bash
# Export merged model
./scripts/shell/export.sh --merge --output my_model

# Start local inference server
python -m mlx_lora_trainer.server --model my_model --port 8080
```

## 📈 Roadmap

### Upcoming Features

- [ ] Adding more models to the supported list
- [ ] Advanced quantization options
- [ ] Distributed training support
- [ ] Web UI for training monitoring
- [ ] Enhanced data preprocessing tools

### Research Directions

- [ ] Exploration of QLoRA techniques
- [ ] Implementation of Flash Attention
- [ ] Investigation of pruning methods
- [ ] Integration of PEFT approaches

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](docs/contributing.md) for:

- Development setup
- Code style guidelines
- Pull request process
- Feature request guidelines

## 📄 License

[MIT License](LICENSE) - Free for commercial and personal use.

---

<div align="center">
Built with ❤️ by Infinite Canvas
</div>
