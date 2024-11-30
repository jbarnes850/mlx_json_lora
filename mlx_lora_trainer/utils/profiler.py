"""Performance profiling utilities for MLX LoRA Trainer."""

import time
import psutil
import mlx.core as mx
from typing import Dict, List, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

@dataclass
class ProfileMetrics:
    """Training performance metrics."""
    batch_time: float
    memory_used: float
    throughput: float  # tokens/second
    gpu_utilization: Optional[float]

class PerformanceProfiler:
    """Performance profiling for training runs."""
    
    def __init__(self, model_name: str, batch_size: int):
        self.console = Console()
        self.model_name = model_name
        self.batch_size = batch_size
        self.metrics_history: List[ProfileMetrics] = []
        self.start_time = time.time()
        
        # Initialize memory tracking
        self.initial_memory = psutil.Process().memory_info().rss / 1024**3
        
    def profile_step(self, 
                    batch_tokens: int,
                    step_time: float,
                    loss: float) -> ProfileMetrics:
        """Profile a single training step."""
        current_memory = psutil.Process().memory_info().rss / 1024**3
        memory_used = current_memory - self.initial_memory
        
        # Calculate throughput
        throughput = batch_tokens / step_time if step_time > 0 else 0
        
        # Get GPU utilization if available
        try:
            gpu_util = mx.metal.get_gpu_utilization()
        except:
            gpu_util = None
            
        metrics = ProfileMetrics(
            batch_time=step_time,
            memory_used=memory_used,
            throughput=throughput,
            gpu_utilization=gpu_util
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze memory growth
        memory_growth = [m.memory_used for m in self.metrics_history]
        if len(memory_growth) > 1:
            growth_rate = (memory_growth[-1] - memory_growth[0]) / len(memory_growth)
            if growth_rate > 0.1:  # GB per step
                recommendations.append(
                    "⚠️ High memory growth detected. Consider:"
                    "\n  - Reducing batch size"
                    "\n  - Enabling gradient checkpointing"
                    "\n  - Using a smaller model"
                )
        
        # Analyze throughput
        throughputs = [m.throughput for m in self.metrics_history]
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
        if avg_throughput < 100:  # tokens/second threshold
            recommendations.append(
                "💡 Low throughput detected. Consider:"
                "\n  - Increasing batch size if memory allows"
                "\n  - Reducing model size or sequence length"
                "\n  - Closing other applications"
            )
        
        return recommendations
    
    def print_summary(self) -> None:
        """Print performance summary."""
        table = Table(title=f"Training Performance Summary: {self.model_name}")
        
        table.add_column("Metric", justify="right", style="cyan")
        table.add_column("Value", justify="left", style="green")
        
        # Calculate summary statistics
        avg_metrics = self._calculate_averages()
        
        table.add_row("Average Throughput", f"{avg_metrics['throughput']:.2f} tokens/sec")
        table.add_row("Memory Usage", f"{avg_metrics['memory_used']:.2f} GB")
        table.add_row("Batch Time", f"{avg_metrics['batch_time']:.3f} sec")
        if avg_metrics['gpu_util'] is not None:
            table.add_row("GPU Utilization", f"{avg_metrics['gpu_util']:.1f}%")
            
        self.console.print(table)
        
        # Print recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            self.console.print("\n[bold yellow]Recommendations:[/]")
            for rec in recommendations:
                self.console.print(rec)
                
    def _calculate_averages(self) -> Dict[str, float]:
        """Calculate average metrics."""
        if not self.metrics_history:
            return {
                'throughput': 0.0,
                'memory_used': 0.0,
                'batch_time': 0.0,
                'gpu_util': None
            }
            
        n = len(self.metrics_history)
        return {
            'throughput': sum(m.throughput for m in self.metrics_history) / n,
            'memory_used': self.metrics_history[-1].memory_used,  # Current memory
            'batch_time': sum(m.batch_time for m in self.metrics_history) / n,
            'gpu_util': (sum(m.gpu_utilization for m in self.metrics_history if m.gpu_utilization is not None) / n
                        if any(m.gpu_utilization is not None for m in self.metrics_history) else None)
        } 