"""Integration tests for MLX LoRA Trainer."""

import os
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest
from transformers import AutoTokenizer

from mlx_lora_trainer.model import ModelRegistry
from mlx_lora_trainer.trainer import LoraTrainer, TrainingArgs
from mlx_lora_trainer.inference import generate_text


@pytest.fixture
def test_data():
    """Create a tiny test dataset."""
    return [
        {"instruction": "Say hello", "response": "Hello! How can I help you today?"},
        {"instruction": "Count to 3", "response": "1, 2, 3"},
    ]


@pytest.fixture
def training_args():
    """Create test training arguments."""
    return TrainingArgs(
        batch_size=1,
        num_epochs=1,
        learning_rate=1e-4,
        max_seq_length=128,
        lora_rank=8,
        lora_alpha=16,
        target_modules=[
            "model.layers.*.self_attn.qkv_proj",
            "model.layers.*.self_attn.o_proj",
            "model.layers.*.mlp.gate_up_proj",
            "model.layers.*.mlp.down_proj"
        ],
        model_name="microsoft/phi-3-mini-4k-instruct",
        model_class="phi"
    )


@pytest.mark.slow
def test_training_pipeline(test_data, training_args, tmp_path):
    """Test the complete training pipeline."""
    # Save test data
    data_path = tmp_path / "test_data.jsonl"
    with open(data_path, "w") as f:
        for item in test_data:
            f.write(f"{item}\n")
    
    # Initialize trainer
    trainer = LoraTrainer(training_args)
    
    # Train model
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)
    trainer.train(str(data_path))
    trainer.save_adapter(str(output_dir / "adapter.safetensors"))
    
    # Verify adapter was saved
    adapter_path = output_dir / "adapter.safetensors"
    assert adapter_path.exists(), "Adapter weights not saved"
    
    # Test loading and inference
    response = generate_text(
        "Say hello",
        str(output_dir),
        max_tokens=20
    )
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.slow
def test_inference_quality(test_data, training_args, tmp_path):
    """Test that fine-tuning improves model output."""
    # Initialize model and tokenizer
    model_cls = ModelRegistry.get_model_class(training_args.model_class)
    model = model_cls.from_pretrained(training_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name)
    
    prompt = "Say hello"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].numpy()
    
    # Get base model response
    base_logits = model(mx.array(inputs))
    base_tokens = mx.argmax(base_logits[:, -1, :], axis=-1)
    base_response = tokenizer.decode(base_tokens.tolist(), skip_special_tokens=True)
    
    # Train model
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)
    data_path = tmp_path / "test_data.jsonl"
    with open(data_path, "w") as f:
        for item in test_data:
            f.write(f"{item}\n")
            
    trainer = LoraTrainer(training_args)
    trainer.train(str(data_path))
    trainer.save_adapter(str(output_dir / "adapter.safetensors"))
    
    # Get fine-tuned response
    tuned_response = generate_text(
        prompt,
        str(output_dir),
        max_tokens=20
    )
    
    # Verify responses are different
    assert tuned_response != base_response, "Fine-tuning did not change model output"
    
    # Check response quality
    assert len(tuned_response) > 0, "Empty response from fine-tuned model"
    assert "hello" in tuned_response.lower(), "Response not relevant to prompt"


@pytest.mark.slow
def test_model_save_load(training_args, tmp_path):
    """Test model saving and loading."""
    # Initialize trainer
    trainer = LoraTrainer(training_args)
    
    # Add LoRA layers
    trainer._setup_model()
    
    # Save adapter
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)
    adapter_path = output_dir / "adapter.safetensors"
    trainer.save_adapter(str(adapter_path))
    
    # Load adapter in new trainer
    new_trainer = LoraTrainer(training_args)
    new_trainer._setup_model()
    new_trainer.load_adapter(str(adapter_path))
    
    # Compare model outputs
    prompt = "Test prompt"
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name)
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].numpy()
    
    out1 = trainer.model(mx.array(inputs))
    out2 = new_trainer.model(mx.array(inputs))
    
    assert mx.allclose(out1, out2, rtol=1e-5), "Model outputs differ after save/load"
