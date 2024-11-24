from setuptools import setup, find_packages

setup(
    name="mlx_lora_trainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.21.0",
        "transformers>=4.36.0",
        "datasets>=2.15.0",
        "rich>=13.7.0",
        "pyyaml>=6.0.1",
        "torch>=2.1.0",
        "sentencepiece>=0.2.0",
        "accelerate>=1.0.0",
        "einops>=0.8.0",
        "fastapi>=0.115.2",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.9",
    description="MLX LoRA Trainer - Fine-tuning toolkit optimized for Apple Silicon",
    author="Codeium",
    author_email="support@codeium.com",
    url="https://github.com/codeium/mlx-lora-trainer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
