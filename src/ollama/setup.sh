#!/bin/bash
# This script pulls models into Ollama

# Set the model we want to use
MODEL="tinyllama"

echo "Pulling model: $MODEL"
ollama pull $MODEL

# List available models
echo "Available models:"
ollama list

echo "Setup complete!" 