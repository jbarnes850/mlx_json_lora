"""Batch processing utilities."""

from typing import List, Any
import mlx.core as mx

def split_batch(batch: Any, num_splits: int) -> List[Any]:
    """Split a batch into smaller micro-batches.
    
    Args:
        batch: Input batch to split
        num_splits: Number of splits to create
        
    Returns:
        List of micro-batches
    """
    if isinstance(batch, (list, tuple)):
        return [batch[i::num_splits] for i in range(num_splits)]
    elif isinstance(batch, dict):
        return [{k: v[i::num_splits] for k, v in batch.items()} 
                for i in range(num_splits)]
    elif isinstance(batch, mx.array):
        return mx.split(batch, num_splits)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}") 