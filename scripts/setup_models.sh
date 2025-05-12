#!/bin/bash
# Script to setup the models for Ollama

# Create data directories
mkdir -p data/ollama

# Setup Ollama model
echo "Setting up Ollama model..."
docker exec -it ollama /bin/bash -c "/app/src/ollama/setup.sh"

# Wait for models to be ready
echo "Waiting for services to be ready..."
sleep 10

echo "Setup complete! You can now run the benchmark." 