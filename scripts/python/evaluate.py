#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from rich.console import Console
from rich.progress import Progress
from rouge_score import rouge_scorer
import mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import generate
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

class Evaluator:
    def __init__(self, model_path, adapter_path=None, model_config=None):
        """Initialize evaluator with model and tokenizer."""
        self.console = Console()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load model and tokenizer
        self.console.print("[bold green]Loading model...[/bold green]")
        model_path = model_path or "microsoft/phi-3-mini-4k-instruct"
        self.model_config = model_config or {
            'max_seq_length': 512,
            'max_tokens': 100
        }
        
        # Download model if needed
        if not os.path.exists(model_path):
            model_path = snapshot_download(model_path)
            
        # Load model and tokenizer
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        
    def evaluate_sample(self, input_text, target_text):
        """Evaluate a single sample."""
        # Generate response
        prompt_tokens = self.tokenizer.encode(input_text)
        output_tokens = generate(
            prompt=prompt_tokens,
            model=self.model,
            max_tokens=self.model_config['max_tokens']
        )
        output = self.tokenizer.decode(output_tokens)
        
        # Calculate ROUGE scores
        scores = self.scorer.score(target_text, output)
        
        return {
            'prediction': output,
            'target': target_text,
            'rouge_scores': scores
        }
    
    def evaluate_dataset(self, test_file, num_samples=None):
        """Evaluate the model on a test dataset."""
        # Load test data
        with open(test_file, 'r') as f:
            test_data = [json.loads(line) for line in f]
        
        # Limit number of samples if specified
        if num_samples is not None:
            test_data = test_data[:num_samples]
        
        # Initialize metrics
        results = []
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougeL = 0
        
        # Evaluate each sample
        with Progress() as progress:
            task = progress.add_task("[cyan]Evaluating...", total=len(test_data))
            for sample in test_data:
                result = self.evaluate_sample(sample['prompt'], sample['completion'])
                results.append({
                    'prompt': sample['prompt'],
                    'completion': sample['completion'],
                    'prediction': result['prediction'],
                    'rouge_scores': result['rouge_scores']
                })
                
                # Update metrics
                total_rouge1 += result['rouge_scores']['rouge1'].fmeasure
                total_rouge2 += result['rouge_scores']['rouge2'].fmeasure
                total_rougeL += result['rouge_scores']['rougeL'].fmeasure
                
                progress.update(task, advance=1)
        
        # Calculate average metrics
        num_samples = len(test_data)
        avg_metrics = {
            'rouge1': total_rouge1 / num_samples,
            'rouge2': total_rouge2 / num_samples,
            'rougeL': total_rougeL / num_samples
        }
        
        return results, avg_metrics
    
    def display_results(self, results, avg_metrics):
        """Display evaluation results."""
        # Print average metrics
        self.console.print("\n[bold green]Average Metrics:[/bold green]")
        for metric, value in avg_metrics.items():
            self.console.print(f"[cyan]{metric}:[/cyan] {value:.4f}")
        
        # Print example predictions
        self.console.print("\n[bold green]Example Predictions:[/bold green]")
        for i, result in enumerate(results[:3]):  # Show first 3 examples
            self.console.print(f"\n[bold]Example {i+1}:[/bold]")
            self.console.print(f"[cyan]Prompt:[/cyan] {result['prompt']}")
            self.console.print(f"[cyan]Target:[/cyan] {result['completion']}")
            self.console.print(f"[yellow]Prediction:[/yellow] {result['prediction']}")
            self.console.print(f"[magenta]ROUGE-L:[/magenta] {result['rouge_scores']['rougeL'].fmeasure:.4f}")
    
    def save_results(self, results, avg_metrics, output_file):
        """Save evaluation results to file."""
        output = {
            'average_metrics': avg_metrics,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.console.print(f"\n[green]Results saved to {output_file}[/green]")

def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned model')
    parser.add_argument('--model-path', type=str, default="",
                      help='Path to model or model name on HuggingFace')
    parser.add_argument('--adapter-path', type=str, required=True,
                      help='Path to LoRA adapter weights')
    parser.add_argument('--test-file', type=str, required=True,
                      help='Path to test data file')
    parser.add_argument('--num-samples', type=int, default=None,
                      help='Number of samples to evaluate')
    parser.add_argument('--output-file', type=str, default='evaluation_results.json',
                      help='Path to save evaluation results')
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Evaluator(args.model_path, args.adapter_path)
    
    # Run evaluation
    results, avg_metrics = evaluator.evaluate_dataset(args.test_file, args.num_samples)
    
    # Display and save results
    evaluator.display_results(results, avg_metrics)
    evaluator.save_results(results, avg_metrics, args.output_file)

if __name__ == "__main__":
    main()
