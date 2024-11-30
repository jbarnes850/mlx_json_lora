# Contributing to MLX LoRA Trainer

We love your input! We want to make contributing to MLX LoRA Trainer as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issue tracker](https://github.com/jbarnes850/mlx-lora-trainer/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/jbarnes850/mlx-lora-trainer/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Process

1. Set up your development environment:
   ```bash
   # Clone your fork
   git clone https://github.com/<your-username>/mlx-lora-trainer.git
   cd mlx-lora-trainer
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

2. Run tests:
   ```bash
   # Run all tests
   pytest
   
   # Run specific test file
   pytest tests/test_model.py
   ```

3. Check code style:
   ```bash
   # Run linter
   pylint mlx_lora_trainer
   
   # Run type checker
   mypy mlx_lora_trainer
   ```

## Code Style

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use type hints
- Write docstrings for all public functions
- Keep functions focused and modular
- Add comments for complex logic

## Documentation

- Update the README.md if needed
- Add docstrings to new functions
- Update API documentation if you change interfaces
- Add examples for new features

## License

By contributing, you agree that your contributions will be licensed under its MIT License. 