#!/usr/bin/env python3
import os
import time
import torch
import json
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify

from vllm import LLM, SamplingParams

# Get environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/tinyllama")
MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Initialize Flask app
app = Flask(__name__)

# Check if model exists locally, otherwise download from HF
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}, downloading from {MODEL_ID}...")
    os.makedirs(MODEL_PATH, exist_ok=True)
    # We'll use the model_id directly in vLLM which will handle the download

# Initialize vLLM
try:
    # Try to load from local path first
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading from {MODEL_PATH}: {e}")
    print(f"Loading model from HuggingFace {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, tensor_parallel_size=1)
    print(f"Loaded model from {MODEL_ID}")

# Default sampling parameters
default_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/metrics', methods=['GET'])
def metrics():
    # Return some basic metrics about the model
    metrics = {
        'model_id': MODEL_ID,
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        metrics['gpu_name'] = torch.cuda.get_device_name(0)
        metrics['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory
        metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated(0)
        metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved(0)
    
    return jsonify(metrics)

@app.route('/generate', methods=['POST'])
def generate():
    start_time = time.time()
    
    # Parse request
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    # Get prompt and parameters
    prompt = request_data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Optional parameters
    temperature = request_data.get('temperature', default_params.temperature)
    top_p = request_data.get('top_p', default_params.top_p)
    max_tokens = request_data.get('max_tokens', default_params.max_tokens)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # Generate response
    try:
        outputs = llm.generate(prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Calculate metrics
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        input_tokens = len(outputs[0].prompt_token_ids)
        output_tokens = len(outputs[0].outputs[0].token_ids)
        tokens_per_second = output_tokens / (elapsed_time_ms / 1000) if elapsed_time_ms > 0 else 0
        
        response = {
            "text": generated_text,
            "metrics": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "time_ms": elapsed_time_ms,
                "tokens_per_second": tokens_per_second
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 