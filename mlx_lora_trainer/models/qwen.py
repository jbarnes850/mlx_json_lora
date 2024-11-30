"""Qwen model implementation."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoConfig, PretrainedConfig, AutoModelForCausalLM

class QwenModel(nn.Module):
    """Qwen2.5 model implementation."""
    
    def __init__(self, config: Optional[PretrainedConfig] = None):
        super().__init__()
        if config is None:
            config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.config = config
        
        # Model dimensions
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads  # GQA support
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        
        # Initialize components
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = []
        
        for _ in range(config.num_hidden_layers):
            self.layers.append({
                "self_attn": {
                    "q_proj": nn.Linear(config.hidden_size, config.hidden_size),
                    "k_proj": nn.Linear(config.hidden_size, 
                                      (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads),
                    "v_proj": nn.Linear(config.hidden_size, 
                                      (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads),
                    "o_proj": nn.Linear(config.hidden_size, config.hidden_size),
                },
                "mlp": {
                    "gate_proj": nn.Linear(config.hidden_size, config.intermediate_size),
                    "up_proj": nn.Linear(config.hidden_size, config.intermediate_size),
                    "down_proj": nn.Linear(config.intermediate_size, config.hidden_size),
                },
                "input_layernorm": nn.RMSNorm(config.hidden_size),
                "post_attention_layernorm": nn.RMSNorm(config.hidden_size),
            })
        
        self.norm_f = nn.RMSNorm(config.hidden_size)
        
    def forward(self, input_ids: mx.array, cache=None) -> mx.array:
        """Forward pass with Grouped Query Attention (GQA)."""
        hidden_states = self.embedding(input_ids)
        
        for i, layer in enumerate(self.layers):
            residual = hidden_states
            hidden_states = layer["input_layernorm"](hidden_states)
            
            # Self-attention with GQA
            query = layer["self_attn"]["q_proj"](hidden_states)
            key = layer["self_attn"]["k_proj"](hidden_states)
            value = layer["self_attn"]["v_proj"](hidden_states)
            
            # Reshape for GQA
            query = self._reshape_for_gqa(query, is_query=True)
            key = self._reshape_for_gqa(key, is_query=False)
            value = self._reshape_for_gqa(value, is_query=False)
            
            # Update cache if provided
            if cache is not None:
                key_cache, value_cache = cache.update_and_fetch(i, key, value)
                key = key_cache
                value = value_cache
            
            # Compute attention with GQA
            attn_output = self._grouped_attention(query, key, value)
            attn_output = layer["self_attn"]["o_proj"](attn_output)
            hidden_states = residual + attn_output
            
            # MLP with SwiGLU activation
            residual = hidden_states
            hidden_states = layer["post_attention_layernorm"](hidden_states)
            
            gate_output = layer["mlp"]["gate_proj"](hidden_states)
            up_output = layer["mlp"]["up_proj"](hidden_states)
            mlp_output = mx.sigmoid(gate_output) * up_output
            mlp_output = layer["mlp"]["down_proj"](mlp_output)
            
            hidden_states = residual + mlp_output
        
        hidden_states = self.norm_f(hidden_states)
        logits = hidden_states @ self.embedding.weight.T
        
        return logits
    
    def _reshape_for_gqa(self, x: mx.array, is_query: bool = False) -> mx.array:
        """Reshape for Grouped Query Attention."""
        batch_size, seq_len, hidden_size = x.shape
        num_heads = self.num_attention_heads if is_query else self.num_key_value_heads
        head_dim = hidden_size // (self.num_attention_heads if is_query else self.num_key_value_heads)
        
        # Current: Simple reshape
        # Need: Proper head grouping and scaling
        return x.reshape(batch_size, seq_len, num_heads, head_dim)
    
    def _grouped_attention(self, query: mx.array, key: mx.array, value: mx.array) -> mx.array:
        """Compute grouped query attention."""
        # Apply rotary position embeddings
        query = self._apply_rope(query)
        key = self._apply_rope(key)
        
        # Repeat KV heads to match query heads for attention
        key = self._repeat_kv(key)
        value = self._repeat_kv(value)
        
        # Compute scaled dot-product attention
        attention_scores = (query @ key.transpose(0, 1, 3, 2)) / mx.sqrt(float(self.head_dim))
        attention_probs = mx.softmax(attention_scores, axis=-1)
        
        # [batch, num_heads, seq_len, head_dim]
        context = attention_probs @ value
        
        # Reshape back to [batch, seq_len, hidden_size]
        batch_size, _, seq_len, _ = context.shape
        context = context.transpose(0, 2, 1, 3)
        return context.reshape(batch_size, seq_len, self.hidden_size)
    
    def _repeat_kv(self, x: mx.array) -> mx.array:
        """Repeat KV heads to match number of query heads for GQA."""
        repeat_factor = self.num_attention_heads // self.num_key_value_heads
        if repeat_factor == 1:
            return x
            
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :]  # [batch, num_kv_heads, 1, seq_len, head_dim]
        x = x.repeat(1, 1, repeat_factor, 1, 1)  # [batch, num_kv_heads, repeat, seq_len, head_dim]
        return x.reshape(batch_size, num_kv_heads * repeat_factor, seq_len, head_dim)
    
    def _apply_rope(self, x: mx.array) -> mx.array:
        """Apply rotary position embeddings."""
        # TODO: Implement proper RoPE
        # For now, return unchanged to get the architecture working
        return x
    
    @classmethod
    def from_pretrained(cls, model_name: str) -> "QwenModel":
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
        """Load adapter weights with proper mapping."""
        try:
            # Map weight names
            mapped_weights = {}
            for name, weight in weights.items():
                if name.startswith("self_attn."):
                    mapped_name = name
                elif name.startswith("attention."):
                    mapped_name = name.replace("attention.", "self_attn.")
                else:
                    mapped_name = name
                mapped_weights[mapped_name] = weight
            
            # Update model
            self.update(mapped_weights)
            print("[INFO] Successfully loaded adapter weights")
            
        except Exception as e:
            print(f"[ERROR] Failed to load weights: {str(e)}")
            raise