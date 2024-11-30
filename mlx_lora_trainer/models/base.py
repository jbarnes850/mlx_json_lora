class BaseModel(nn.Module):
    """Base class for all models."""
    
    @abstractmethod
    def validate_weights(self) -> bool:
        """Validate model weights."""
        pass
    
    def verify_forward_pass(self, batch_size: int = 1, seq_len: int = 32) -> bool:
        """Verify model forward pass."""
        try:
            dummy_input = mx.random.randint(0, 100, (batch_size, seq_len))
            output = self(dummy_input)
            
            # Check output shape
            expected_shape = (batch_size, seq_len, self.config.vocab_size)
            assert output.shape == expected_shape
            
            # Check for NaN values
            assert not mx.isnan(output).any()
            
            return True
        except Exception as e:
            print(f"Forward pass verification failed: {str(e)}")
            return False 