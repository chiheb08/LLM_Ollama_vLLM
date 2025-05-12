#!/usr/bin/env python3
import os
import time
import json
import argparse
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from monitoring.prometheus_metrics import metrics

# Benchmark configuration
DEFAULT_MODELS = ["tinyllama"]
DEFAULT_PROMPTS = [
    "Explain what is machine learning in simple terms.",
    "Write a short poem about artificial intelligence.",
    "Describe the differences between CPU and GPU computing.",
    "What is vLLM and how does it make language models faster?",
    "Provide 3 tips for optimizing Docker containers.",
    "Explain the concept of attention in transformer models.",
    "What are the ethical concerns in AI development?",
    "Describe the process of fine-tuning a language model.",
    "Explain how PagedAttention works in vLLM.",
    "What are the benefits of using Docker for deployment?"
]

# Service endpoints
OLLAMA_API = "http://ollama:11434/api/generate"

def setup_args():
    parser = argparse.ArgumentParser(description="LLM Benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, 
                        help="Models to benchmark")
    parser.add_argument("--prompts-file", type=str, 
                        help="JSON file with prompts to use for benchmarking")
    parser.add_argument("--iterations", type=int, default=3, 
                        help="Number of iterations for each prompt")
    parser.add_argument("--concurrent", type=int, default=1, 
                        help="Number of concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=512, 
                        help="Maximum tokens to generate")
    parser.add_argument("--output-dir", type=str, default="/app/results", 
                        help="Directory to save results")
    parser.add_argument("--plot", action="store_true", 
                        help="Generate plots of the results")
    
    return parser.parse_args()

def load_prompts(args):
    """Load prompts from file or use defaults"""
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            return json.load(f)
    return DEFAULT_PROMPTS

def run_ollama_inference(model, prompt, max_tokens):
    """Run inference using Ollama API"""
    start_time = time.time()
    
    try:
        metrics.record_request(model, "ollama")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": max_tokens,
            }
        }
        
        response = requests.post(OLLAMA_API, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Calculate metrics
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        
        # Ollama doesn't provide token counts directly, estimate based on text length
        # This is a rough approximation
        input_tokens = len(prompt.split())
        output_tokens = len(result.get("response", "").split())
        
        metrics.record_response_time(model, "ollama", elapsed_time_ms)
        metrics.record_tokens(model, "ollama", input_tokens, output_tokens, elapsed_time_ms)
        
        return {
            "model": model,
            "service": "ollama",
            "time_ms": elapsed_time_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "text": result.get("response", "")
        }
    
    except Exception as e:
        metrics.record_request(model, "ollama", "error")
        return {
            "model": model,
            "service": "ollama",
            "error": str(e),
            "time_ms": (time.time() - start_time) * 1000
        }

def run_single_benchmark(model, prompt, max_tokens):
    """Run benchmark for Ollama on a single prompt"""
    # Run with Ollama
    ollama_result = run_ollama_inference(model, prompt, max_tokens)
    ollama_result["prompt"] = prompt
    
    return [ollama_result]

def generate_plots(df, output_dir):
    """Generate comparison plots for Ollama performance"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # 1. Response Time Comparison by Model
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="model", y="time_ms", data=df)
    ax.set_title("Response Time by Model")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "response_time_comparison.png"))
    plt.close()
    
    # 2. Tokens per Second Comparison
    df['tokens_per_second'] = df['output_tokens'] / (df['time_ms'] / 1000)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="model", y="tokens_per_second", data=df)
    ax.set_title("Tokens per Second by Model")
    ax.set_ylabel("Tokens/s")
    ax.set_xlabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tokens_per_second_comparison.png"))
    plt.close()
    
    # 3. Box plot of response times
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x="model", y="time_ms", data=df)
    ax.set_title("Response Time Distribution")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "response_time_distribution.png"))
    plt.close()
    
    # 4. Create summary table
    summary = df.groupby(['model']).agg({
        'time_ms': ['mean', 'min', 'max', 'std'],
        'tokens_per_second': ['mean', 'min', 'max', 'std']
    }).reset_index()
    
    # Save summary as CSV
    summary_path = os.path.join(output_dir, "summary_stats.csv")
    summary.to_csv(summary_path)
    
    print(f"Plots and summary saved to {output_dir}")

def main():
    args = setup_args()
    prompts = load_prompts(args)
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for model in args.models:
        print(f"Benchmarking model: {model}")
        
        for iteration in range(args.iterations):
            print(f"  Iteration {iteration+1}/{args.iterations}")
            
            with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
                futures = []
                for prompt in prompts:
                    futures.append(executor.submit(
                        run_single_benchmark, model, prompt, args.max_tokens
                    ))
                
                for future in tqdm(futures, desc="Processing prompts"):
                    results = future.result()
                    all_results.extend(results)
    
    # Save raw results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, "benchmark_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Generate plots if requested
    if args.plot:
        generate_plots(results_df, output_dir)

if __name__ == "__main__":
    main() 