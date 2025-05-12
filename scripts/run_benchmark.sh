#!/bin/bash
# Script to run the benchmark

# Define model(s) to benchmark
MODEL="tinyllama"

# Enter the benchmark container
echo "Running benchmark with model $MODEL..."
docker exec -it benchmark python /app/src/run_benchmark.py --models $MODEL --iterations 3 --concurrent 1 --plot

echo "Benchmark complete! Results are saved in the 'results' directory." 