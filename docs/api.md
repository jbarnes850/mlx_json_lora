# API Reference

## Command Line Interface

### Training Script
```bash
bash scripts/shell/train_lora.sh
```

**Options:**
- Model selection is interactive
- Configuration is handled through YAML files

### Chat Interface
```bash
bash scripts/shell/chat.sh
```

**Options:**
- Uses the most recently trained model by default
- Interactive chat session with the model

### Export Script
```bash
bash scripts/shell/export.sh
```

**Options:**
- Exports the trained LoRA weights to a deployable format

## Configuration

### Model Configurations

Located in `configs/models/`:

```yaml
# Example config.yaml
model: "gemma-2b"
train: true
data: "data"
adapter_path: "adapters"

# Memory optimization settings
memory:
  grad_checkpoint: true      # Enable gradient checkpointing
  checkpoint_layers: "all"   # Apply to all transformer layers
  memory_efficient: true     # Use memory efficient attention
```

### Training Parameters

```yaml
# Training settings
training:
  batch_size: 2
  learning_rate: 1.5e-4
  num_epochs: 3
  save_every: 100
  grad_checkpoint: true
```

## Python API

### Trainer Class

```python
from mlx_lora_trainer import Trainer, Config

# Initialize trainer
config = Config(
    model_name="gemma-2b",
    lora_rank=8,
    lora_alpha=32,
    grad_checkpoint=True
)

trainer = Trainer(config)
trainer.train(
    train_data="data/train.jsonl",
    val_data="data/val.jsonl",
    epochs=3
)
```

### Data Format

Training data should be in JSONL format:
```json
{
    "conversations": [
        {"role": "user", "content": "Query"},
        {"role": "assistant", "content": "Response"}
    ]
}
```

## Error Handling

The framework provides detailed error messages for common issues:
- Memory constraints
- Hardware compatibility
- Data format issues
- Training interruptions

## Logging

Logs are stored in `logs/`:
- `training.log`: Training progress and metrics
- Error logs with timestamps and stack traces
- Memory usage statistics
