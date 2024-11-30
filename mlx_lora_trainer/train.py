WEIGHT_MAPPINGS = {
    "microsoft/Phi-3.5-mini-instruct": {
        "self_attn.q_proj": "q_proj",
        "self_attn.k_proj": "k_proj",
        "self_attn.v_proj": "v_proj",
        "self_attn.o_proj": "o_proj",
        "mlp.gate_proj": "gate_proj",
        "mlp.up_proj": "up_proj",
        "mlp.down_proj": "down_proj"
    },
    "google/gemma-2-2b": {
        "self_attn.q_proj": "q_proj",
        "self_attn.k_proj": "k_proj",
        "self_attn.v_proj": "v_proj",
        "self_attn.o_proj": "o_proj",
        "mlp.gate_proj": "gate_proj",
        "mlp.up_proj": "up_proj",
        "mlp.down_proj": "down_proj"
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "self_attn.q_proj": "q_proj",
        "self_attn.k_proj": "k_proj",
        "self_attn.v_proj": "v_proj",
        "self_attn.o_proj": "o_proj",
        "mlp.gate_proj": "gate_proj",
        "mlp.up_proj": "up_proj",
        "mlp.down_proj": "down_proj"
    }
}

def save_adapter_weights(model: nn.Module, save_path: str) -> None:
    """Save adapter weights with consistent naming."""
    try:
        # Extract LoRA weights with consistent naming
        lora_weights = {}
        state_dict = model.state_dict()
        
        # Get model type from config
        model_name = model.config.name_or_path
        mapping = WEIGHT_MAPPINGS.get(model_name, {})
        
        # Map and save weights using consistent naming
        for target_name, source_name in mapping.items():
            full_name = f"{target_name}.lora.weight"
            if full_name in state_dict:
                lora_weights[source_name] = state_dict[full_name]
        
        # Save weights
        np.savez(save_path, **lora_weights)
        print(f"[INFO] Saved adapter weights to {save_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save weights: {str(e)}")
        raise 

def validate_adapter_weights(weights: Dict[str, mx.array], config: Dict) -> bool:
    required_modules = set(config["lora"]["target_modules"])
    weight_modules = {name.split(".")[0] for name in weights.keys()}
    return required_modules.issubset(weight_modules)

# Add gradient accumulation for better memory efficiency
def train_step(model, optimizer, batch, grad_accumulation_steps=1):
    """Training step with gradient accumulation."""
    accumulated_gradients = None
    for micro_batch in split_batch(batch, grad_accumulation_steps):
        loss, grads = value_and_grad(model)(micro_batch)
        if accumulated_gradients is None:
            accumulated_gradients = grads
        else:
            accumulated_gradients = tree_map(lambda x, y: x + y, accumulated_gradients, grads)
    
    # Scale gradients by accumulation steps
    accumulated_gradients = tree_map(lambda x: x / grad_accumulation_steps, accumulated_gradients)
    optimizer.update(model, accumulated_gradients)
    return loss

class TrainingMonitor:
    """Monitor training progress and resources."""
    def __init__(self):
        self.loss_history = []
        self.memory_usage = []
        self.grad_norms = []
        
    def log_metrics(self, step: int, loss: float, model: nn.Module):
        """Log training metrics."""
        grad_norm = compute_grad_norm(model)
        memory_used = psutil.Process().memory_info().rss / 1024**3
        
        self.loss_history.append(loss)
        self.memory_usage.append(memory_used)
        self.grad_norms.append(grad_norm)
        
        # Alert if metrics exceed thresholds
        if grad_norm > 100:
            print(f"WARNING: Large gradient norm: {grad_norm}")
        if memory_used > 0.9 * psutil.virtual_memory().total / 1024**3:
            print("WARNING: High memory usage")