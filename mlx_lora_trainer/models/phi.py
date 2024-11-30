"""Phi model implementation."""

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoConfig, PretrainedConfig, AutoModelForCausalLM
from typing import Optional, List, Dict, Any

class PhiModel(nn.Module):
    """Phi model implementation."""
    
    def __init__(self, config: Optional[PretrainedConfig] = None):
        """Initialize Phi model."""
        super().__init__()
        if config is None:
            config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        self.config = config
        
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        
        # Initialize model components
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = []
        
        for _ in range(config.num_hidden_layers):
            self.layers.append({
                "attention": {
                    "q_proj": nn.Linear(config.hidden_size, config.hidden_size),
                    "k_proj": nn.Linear(config.hidden_size, config.hidden_size),
                    "v_proj": nn.Linear(config.hidden_size, config.hidden_size),
                    "o_proj": nn.Linear(config.hidden_size, config.hidden_size),
                },
                "mlp": {
                    "gate_proj": nn.Linear(config.hidden_size, config.intermediate_size),
                    "up_proj": nn.Linear(config.hidden_size, config.intermediate_size),
                    "down_proj": nn.Linear(config.intermediate_size, config.hidden_size),
                },
                "ln_1": nn.LayerNorm(config.hidden_size),
                "ln_2": nn.LayerNorm(config.hidden_size),
            })
        
        self.ln_f = nn.LayerNorm(config.hidden_size)
    
    def forward(self, input_ids: mx.array, cache=None) -> mx.array:
        """Forward pass."""
        hidden_states = self.embeddings(input_ids)
        
        for i, layer in enumerate(self.layers):
            residual = hidden_states
            hidden_states = layer["ln_1"](hidden_states)
            
            # Self-attention
            query = layer["attention"]["q_proj"](hidden_states)
            key = layer["attention"]["k_proj"](hidden_states)
            value = layer["attention"]["v_proj"](hidden_states)
            
            # Update cache if provided
            if cache is not None:
                key_cache, value_cache = cache.update_and_fetch(i, key, value)
                key = key_cache
                value = value_cache
            
            # Compute attention
            attn_output = self._attention(query, key, value)
            attn_output = layer["attention"]["o_proj"](attn_output)
            hidden_states = residual + attn_output
            
            # MLP
            residual = hidden_states
            hidden_states = layer["ln_2"](hidden_states)
            
            gate_output = layer["mlp"]["gate_proj"](hidden_states)
            up_output = layer["mlp"]["up_proj"](hidden_states)
            mlp_output = mx.sigmoid(gate_output) * up_output
            mlp_output = layer["mlp"]["down_proj"](mlp_output)
            
            hidden_states = residual + mlp_output
        
        hidden_states = self.ln_f(hidden_states)
        logits = hidden_states @ self.embeddings.weight.T
        
        return logits
    
    def _attention(self, query: mx.array, key: mx.array, value: mx.array) -> mx.array:
        """Compute scaled dot-product attention."""
        attention_scores = (query @ key.transpose()) / mx.sqrt(float(self.head_dim))
        attention_probs = mx.softmax(attention_scores, axis=-1)
        return attention_probs @ value
    
    @classmethod
    def from_pretrained(cls, model_name: str) -> "PhiModel":
        """Load pretrained model."""
        config = AutoConfig.from_pretrained(model_name)
        model = cls(config)
        
        # Load pretrained weights
        torch_model = AutoModelForCausalLM.from_pretrained(model_name)
        state_dict = {k: mx.array(v.detach().numpy()) 
                     for k, v in torch_model.state_dict().items()}
        
        model.update(state_dict)
        return model 
    
    def load_weights(self, weights: Dict[str, mx.array]) -> None:
        """Load weights with proper mapping."""
        try:
            # Map weight names if needed
            mapped_weights = {}
            for name, weight in weights.items():
                # Handle different weight name formats
                if name.startswith("attention."):
                    # Remove 'attention.' prefix for compatibility
                    mapped_name = name.replace("attention.", "")
                    mapped_weights[mapped_name] = weight
                else:
                    mapped_weights[name] = weight
            
            # Update model with mapped weights
            self.update(mapped_weights)
            print("[INFO] Successfully loaded adapter weights")
            
        except Exception as e:
            print(f"[ERROR] Failed to load weights: {str(e)}")
            raise