#!/usr/bin/env python3
"""
Benchmarking script for MLX LoRA fine-tuned models.
Measures inference latency, memory usage, and perplexity.
"""

import os
import json
import time
import psutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from rich.console import Console
from rich.table import Table
from mlx_lm import load, generate
from transformers import AutoTokenizer

console = Console()

class ModelBenchmark:
    def __init__(self, model_path: str, adapter_path: str = None):
        """Initialize benchmark with model and adapter paths."""
        self.model_path = model_path
        self.adapter_path = adapter_path
        
        # Load model and tokenizer
        console.print("[bold green]Loading model...[/bold green]")
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
    
    def measure_memory(self) -> Dict[str, float]:
        """Measure memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
            'vms': memory_info.vms / (1024 * 1024)   # VMS in MB
        }
    
    def measure_latency(self, prompt: str, num_runs: int = 5) -> Dict[str, float]:
        """Measure inference latency."""
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.time()
            _ = generate(
                prompt=prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                max_tokens=100
            )
            latencies.append(time.time() - start_time)
        
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies)
        }
    
    def calculate_perplexity(self, test_file: str, num_samples: int = None) -> float:
        """Calculate perplexity on test set."""
        total_loss = 0
        total_tokens = 0
        
        # Load test data
        with open(test_file, 'r') as f:
            test_data = [json.loads(line) for line in f]
        
        if num_samples:
            test_data = test_data[:num_samples]
        
        for sample in test_data:
            # Get completion text
            if 'completion' in sample:
                text = sample['completion']
            elif 'messages' in sample:
                text = sample['messages'][-1]['content']
            else:
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(text)
            total_tokens += len(tokens)
            
            # Calculate loss
            logits = self.model(tokens)
            loss = -np.mean(np.log(logits))
            total_loss += loss * len(tokens)
        
        return np.exp(total_loss / total_tokens)
    
    def run_benchmarks(self, test_file: str, num_samples: int = None) -> Dict[str, Any]:
        """Run all benchmarks."""
        results = {}
        
        # Measure memory usage
        console.print("\n[cyan]Measuring memory usage...[/cyan]")
        results['memory'] = self.measure_memory()
        
        # Measure latency
        console.print("\n[cyan]Measuring inference latency...[/cyan]")
        sample_prompt = "Once upon a time"
        results['latency'] = self.measure_latency(sample_prompt)
        
        # Calculate perplexity
        console.print("\n[cyan]Calculating perplexity...[/cyan]")
        results['perplexity'] = self.calculate_perplexity(test_file, num_samples)
        
        return results
    
    def display_results(self, results: Dict[str, Any]):
        """Display benchmark results in a formatted table."""
        # Memory table
        memory_table = Table(title="Memory Usage")
        memory_table.add_column("Metric")
        memory_table.add_column("Value (MB)")
        
        for metric, value in results['memory'].items():
            memory_table.add_row(metric.upper(), f"{value:.2f}")
        
        console.print(memory_table)
        
        # Latency table
        latency_table = Table(title="Inference Latency (seconds)")
        latency_table.add_column("Metric")
        latency_table.add_column("Value")
        
        for metric, value in results['latency'].items():
            latency_table.add_row(metric.capitalize(), f"{value:.4f}")
        
        console.print(latency_table)
        
        # Perplexity
        console.print(f"\n[bold green]Perplexity:[/bold green] {results['perplexity']:.4f}")
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save benchmark results to file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[green]Results saved to {output_file}[/green]")

def main():
    parser = argparse.ArgumentParser(description='Benchmark MLX LoRA fine-tuned model')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to model or model name on HuggingFace')
    parser.add_argument('--adapter-path', type=str,
                      help='Path to LoRA adapter weights')
    parser.add_argument('--test-file', type=str, required=True,
                      help='Path to test data file')
    parser.add_argument('--num-samples', type=int,
                      help='Number of samples to use for perplexity calculation')
    parser.add_argument('--output-file', type=str, default='benchmark_results.json',
                      help='Path to save benchmark results')
    args = parser.parse_args()
    
    # Run benchmarks
    benchmark = ModelBenchmark(args.model_path, args.adapter_path)
    results = benchmark.run_benchmarks(args.test_file, args.num_samples)
    
    # Display and save results
    benchmark.display_results(results)
    benchmark.save_results(results, args.output_file)

if __name__ == "__main__":
    main()
