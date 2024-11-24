# Contributing Guide

Thank you for your interest in contributing to MLX LoRA Trainer! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/jbarnes850/mlx_lora_trainer.git
   cd mlx_lora_trainer
   ```
3. Create a development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

## Code Style

We follow these coding standards:
- PEP 8 for Python code style
- Google-style docstrings
- Type hints for function parameters and returns
- Maximum line length of 100 characters

Use pre-commit hooks to ensure code quality:
```bash
pre-commit install
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

For coverage report:
```bash
pytest --cov=mlx_lora_trainer tests/
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
   We follow [Conventional Commits](https://www.conventionalcommits.org/)

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Any relevant issue numbers
   - Screenshots for UI changes
   - Updated documentation

## Project Structure

```bash
mlx_lora_trainer/
├── mlx_lora_trainer/     # Main package
│   ├── __init__.py
│   ├── model.py         # Model loading and inference
│   ├── trainer.py       # LoRA training implementation
│   └── utils.py         # Utility functions
├── scripts/             # Scripts directory
│   ├── shell/          # Shell scripts
│   └── python/         # Python scripts
├── configs/            # Configuration files
│   ├── models/        # Model configurations
│   └── training/      # Training configurations
├── tests/             # Test files
├── docs/              # Documentation
└── examples/          # Example notebooks and scripts
```

## Documentation

- Update relevant documentation with code changes
- Add docstrings to new functions and classes
- Include examples in docstrings
- Update API documentation for public interfaces

## Need Help?

- Check existing [issues](https://github.com/jbarnes850/mlx_lora_trainer/issues)
