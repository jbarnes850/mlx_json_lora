# Installation Guide

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- 8GB RAM minimum (16GB recommended)
- 10GB free storage space

## Quick Install

```bash
# Clone the repository
git clone https://github.com/jbarnes850/mlx_lora_trainer.git
cd mlx_lora_trainer

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .
```

## Development Install

For contributors and developers:

```bash
# Install development dependencies
pip install -e ".[dev]"
```

## Troubleshooting

### Common Issues

1. **MLX Installation Fails**
   - Ensure you're on macOS with Apple Silicon
   - Try: `pip install --upgrade pip`
   - Install MLX separately: `pip install mlx`

2. **Import Errors**
   - Ensure virtual environment is activated
   - Check PYTHONPATH includes project root
   - Verify installation with: `python -c "import mlx_lora_trainer"`

3. **Memory Issues**
   - Close other memory-intensive applications
   - Reduce batch size in training configuration
   - Consider using smaller model variants

### Getting Help

- Check the [GitHub Issues](https://github.com/jbarnes850/mlx_lora_trainer/issues)
