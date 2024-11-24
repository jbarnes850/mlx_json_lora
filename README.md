# MLX LoRA Trainer: Professional Model Fine-Tuning on Apple Silicon

> Transform your Mac into a professional model fine-tuning workstation. Built on Apple's MLX framework and optimized for Metal performance.

<div align="center">

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MLX Version](https://img.shields.io/badge/MLX-0.18.1-green.svg)](https://github.com/ml-explore/mlx)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

</div>

## 🎯 Professional-Grade Fine-Tuning

MLX LoRA Trainer brings enterprise-quality model customization to your local environment. Designed for professionals who need:

- Production-ready model adaptation
- Memory-efficient training
- Professional monitoring and logging
- Seamless deployment pipeline

### [Watch Demo Video](demo/tutorial.mp4)

## ⚡️ Quick Start (2 minutes)

Get started with a single command:
```bash
git clone https://github.com/jbarnes850/mlx_lora_trainer.git
cd mlx_lora_trainer
bash scripts/shell/run_tutorial.sh
```

The interactive tutorial will guide you through:
1. Selecting your model (Phi-3-mini, Gemma-2B, or Qwen2.5-4B)
2. Choosing your training data (pre-formatted JSONL or uploading custom data)
3. Starting your first fine-tuning run

## Advanced Features

### Custom Datasets
The framework currently supports pre-formatted JSONL datasets with conversation structure:
```json
{
    "conversations": [
        {"role": "user", "content": "Your input text"},
        {"role": "assistant", "content": "Desired output text"}
    ]
}
```
Place your formatted data in the `data/` directory as `train.jsonl`, or specify a custom path in the config.

> 💡 **Coming Soon**: Support for converting unstructured text data into the required JSONL format.

### Training Times
Approximate training times per epoch on M2 Pro:
- Phi-3-mini (8GB): ~20 minutes
- Gemma-2B (12GB): ~45 minutes
- Qwen2.5-4B (16GB): ~90 minutes

Training time varies based on dataset size and hardware. Expect ~20% faster times on M3/M4 Series and ~20% slower on M1 series.

### Local vs Cloud Training
While cloud solutions may offer faster training times, MLX LoRA Trainer provides:
- Complete data privacy
- No cloud compute costs
- Full control over training process
- Optimized for Apple Silicon

## 💻 Supported Models (More Models Coming Soon)

| Model | Parameters | RAM | Use Case |
|-------|------------|-----|----------|
| Phi-3-mini | 1.3B | 8GB | Rapid prototyping |
| Gemma-2B | 2B | 12GB | Production deployment |
| Qwen2.5-4B | 4B | 16GB | Advanced applications |

## 📊 Technical Implementation

### Memory-Optimized Training
```python
config = Config(
    model_name="gemma-2b",
    lora_rank=8,
    lora_alpha=32,
    grad_checkpoint=True,
    memory_efficient=True
)
```

### Production Data Format
```json
{
    "conversations": [
        {"role": "user", "content": "Query"},
        {"role": "assistant", "content": "Response"}
    ]
}
```

## 🔧 System Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS Sonoma or newer
- Python 3.9+
- RAM requirements:
  * 8GB+ for Phi-3-mini
  * 12GB+ for Gemma-2B
  * 16GB+ for Qwen2.5-4B

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [Best Practices](docs/best_practices.md)
- [Contributing](docs/contributing.md)


## 📈 Roadmap

### High Priority
- [ ] Unstructured Data Support
  * Automatic conversion of text documents to JSONL
  * Smart chunking and context preservation
  * Quality-focused conversation generation
  * Memory-efficient processing pipeline

- [ ] Expanded Model Support
  * Llama-3 Model Family
  * Gemma Model Family
  * Falcon Model Family
  

### Coming Soon
- [ ] Distributed training support
- [ ] Advanced quantization options
- [ ] Custom architecture support
- [ ] Enterprise monitoring dashboard
- [ ] Cloud integration options

## 📄 License

[MIT License](LICENSE) - Free for commercial and personal use.

---

<div align="center">
Built with ❤️ by Infinite Canvas
</div>
