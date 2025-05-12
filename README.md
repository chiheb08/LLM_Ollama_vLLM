# LLM Benchmarking Platform: Ollama vs. vLLM

A Docker-based platform for benchmarking and monitoring LLM inference engines (Ollama and vLLM) with real-time visualization. Designed to provide detailed performance comparisons between different inference engines, models, and configurations.

![Benchmarking Dashboard](https://raw.githubusercontent.com/username/llm-benchmarking-platform/main/docs/images/dashboard_preview.png)

## Features

- 🚀 Benchmark multiple LLM models across different inference engines (Ollama and vLLM)
- 📊 Real-time visualization of performance metrics with Grafana dashboards
- 🔍 Comprehensive hardware monitoring (CPU, memory, GPU) for resource utilization analysis
- 📈 Automated metric collection, persistence, and visualization
- 🔄 Support for concurrent inference requests to simulate load testing
- 🖥️ Cross-platform support with CPU-only mode for Mac/non-NVIDIA systems
- 📱 Ollama WebUI for interactive testing and model management
- 📁 Persistent model storage and benchmark results

```

## Comparison: Ollama vs. vLLM

| Feature               | Ollama                  | vLLM                        |
|-----------------------|-------------------------|------------------------------|
| **Implementation**    | C++ & Rust              | Python & CUDA               |
| **Key Technology**    | GGML/GGUF Quantization  | PagedAttention              |
| **Primary Focus**     | Ease of use, low memory | High throughput, GPU scaling|
| **Best For**          | Personal use, CPU systems| Production, GPU systems    |
| **Quantization**      | Built-in                | Optional                    |
| **Typical Performance**| 20-30 tokens/sec       | 50-100+ tokens/sec          |
| **Memory Usage**      | Lower                   | Higher                      |
| **GPU Utilization**   | Moderate               | High                         |
| **API Compatibility** | Custom API              | OpenAI API compatible       |

## Components

- **Ollama**: Fast, local LLM inference engine with built-in model management
- **vLLM**: High-performance LLM inference with PagedAttention (GPU accelerated)
- **Prometheus**: Time-series metrics collection for performance data
- **Grafana**: Dashboard visualization with custom panels for LLM metrics
- **Node Exporter**: Hardware metrics collection for system resources
- **DCGM Exporter**: NVIDIA GPU metrics collection for GPU utilization
- **Benchmark Tool**: Python-based benchmarking with customizable parameters

## Requirements

- Docker and Docker Compose
- 8GB+ RAM for basic models (16GB+ recommended)
- For vLLM: NVIDIA GPU with CUDA support (at least 8GB VRAM)
- For Ollama: Works on CPU, but GPU recommended for larger models
- macOS, Linux, or Windows with WSL2 (Windows native Docker has limited GPU support)

## Quick Start

### CPU-Only Mode (Recommended for Mac/non-NVIDIA systems)

For testing with Ollama on systems without NVIDIA GPUs (e.g., Macs with Apple Silicon):

```bash
# Clone this repository
git clone https://github.com/chiheb08/LLM_Ollama_vLLM.git
cd LLM_Ollama_vLLM

# Start the platform in CPU-only mode
docker-compose -f docker-compose.cpu.yml up -d

# Download a model to test with
docker exec -it ollama ollama pull tinyllama

# Run a benchmark against Ollama
docker exec -it benchmark python /app/src/run_benchmark.py --models tinyllama --iterations 3 --plot
```

### GPU Mode with Both Ollama and vLLM

If you have an NVIDIA GPU with CUDA support:

```bash
# Clone this repository
git clone https://github.com/chiheb08/LLM_Ollama_vLLM.git
cd LLM_Ollama_vLLM

# Start the platform with GPU services
docker-compose up -d

# Or to specifically include vLLM:
docker-compose --profile gpu up -d

# Download a model to test with
docker exec -it ollama ollama pull tinyllama

# Run a benchmark against both Ollama and vLLM
docker exec -it benchmark python /app/src/run_benchmark.py --models tinyllama --engines ollama vllm --iterations 3 --plot
```

### Accessing Dashboards

- **Grafana**: http://localhost:3001 (admin/password)
- **Prometheus**: http://localhost:9090
- **Ollama WebUI**: http://localhost:3000

## Benchmark Options

```
usage: run_benchmark.py [-h] [--models MODELS [MODELS ...]] [--engines ENGINES [ENGINES ...]]
                        [--iterations ITERATIONS] [--concurrent CONCURRENT]
                        [--max-tokens MAX_TOKENS] [--plot]

arguments:
  --models MODELS [MODELS ...]     Models to benchmark (default: ["tinyllama"])
  --engines ENGINES [ENGINES ...]  Inference engines to benchmark (ollama, vllm)
  --iterations ITERATIONS          Number of iterations per model (default: 5)
  --concurrent CONCURRENT          Number of concurrent requests (default: 1)
  --max-tokens MAX_TOKENS          Maximum tokens to generate (default: 100)
  --plot                          Generate plots of the results
```

## Benchmarking Methodology

The benchmark process:

1. Selects a set of representative prompts from various domains
2. Measures key performance metrics:
   - Response time (ms)
   - Tokens per second (throughput)
   - First token latency
   - Memory usage during inference
   - GPU utilization (when applicable)
3. Runs multiple iterations to account for variance
4. Supports concurrent requests to simulate real-world load
5. Generates statistical analysis (mean, min, max, standard deviation)
6. Creates visualizations for comparative analysis

## Dashboard Features

The platform provides multiple dashboards:

1. **Home Dashboard**: Overview and navigation hub
   - Summary of total requests processed
   - Links to engine-specific dashboards
   - Quick stats on system health

2. **Ollama Dashboard**: Detailed metrics for Ollama performance
   - Response time tracking
   - Tokens/second performance
   - Request volume
   - Token generation metrics
   - System resource utilization

3. **vLLM Dashboard**: GPU-accelerated performance metrics
   - GPU utilization and memory consumption
   - Temperature and power monitoring
   - Response time tracking
   - Throughput metrics
   - Detailed token statistics

## Results and Analysis

Benchmark results are saved to the `results/` directory:
- CSV files with detailed metrics for further analysis
- PNG plots showing performance comparisons between engines and models
- Real-time metrics in Grafana dashboards for ongoing monitoring

### Sample Findings

When comparing Ollama and vLLM on the same model (e.g., TinyLlama):

- **Response Time**: vLLM typically shows 30-50% lower latency, especially with GPU acceleration
- **Throughput**: vLLM processes 2-3x more tokens per second on equivalent hardware
- **Memory Usage**: vLLM requires more VRAM but can handle larger contexts more efficiently
- **Scaling**: vLLM scales better with batch size and context length, crucial for production
- **First Token Latency**: Ollama often has faster time-to-first-token in smaller models
- **CPU Utilization**: Ollama has efficient CPU execution with quantized models

## Understanding vLLM

vLLM is an open-source library for LLM inference that introduces several optimizations:

### Key Features of vLLM

1. **PagedAttention**: A memory-efficient attention mechanism that manages KV cache (key-value pairs from previous tokens) using a paging system inspired by operating systems.
2. **Continuous Batching**: Efficiently processes multiple requests concurrently, dramatically improving throughput.
3. **Optimized CUDA Kernels**: Highly optimized implementations of key operations.
4. **Tensor Parallelism**: Distributes large models across multiple GPUs.

### How vLLM Improves Performance

* **Memory Efficiency**: By using PagedAttention, vLLM can better manage memory, allowing it to serve more concurrent requests with the same hardware.
* **Throughput**: The continuous batching approach enables processing multiple requests at different stages simultaneously, maximizing GPU utilization.
* **Latency**: Optimized CUDA kernels and efficient memory management reduce the time to generate each token.

## Advanced Customization

### Custom Models

- Add models to Ollama: `docker exec -it ollama ollama pull model_name`
- Use custom GGUF models with Ollama: 
  ```bash
  docker cp your-model.gguf ollama:/tmp/
  docker exec -it ollama ollama create mymodel -f /tmp/your-model.gguf
  ```
- Change vLLM model: Edit the MODEL_ID environment variable in docker-compose.yml

### Hardware Configuration

- Adjust GPU allocation between services by modifying the `deploy.resources` section
- For multi-GPU setups, modify the vLLM container to use tensor parallelism:
  ```yaml
  environment:
    - MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
    - TENSOR_PARALLEL_SIZE=2  # For 2 GPUs
  ```

### Benchmark Customization

- Edit `src/run_benchmark.py` to add custom prompts or metrics
- Modify the Grafana dashboards to display metrics relevant to your use case
- Adjust the Prometheus scrape interval for more/less granular data

## Troubleshooting

- **Issue**: Grafana isn't showing metrics
  - **Solution**: Check that Prometheus can reach the targets: http://localhost:9090/targets
  
- **Issue**: GPU not detected by vLLM
  - **Solution**: Verify NVIDIA drivers and Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

- **Issue**: Containers fail to start
  - **Solution**: Check logs with `docker-compose logs service_name`

- **Issue**: Out of memory errors
  - **Solution**: Use smaller models or increase swap space with `--memory-swap` in Docker

## Project Structure

```
LLM_Ollama_vLLM/
├── config/                      # Configuration files
│   ├── benchmark.Dockerfile     # Dockerfile for benchmark container
│   ├── vllm.Dockerfile          # Dockerfile for vLLM container
│   ├── prometheus.yml           # Prometheus configuration
│   └── grafana/                 # Grafana dashboards and datasources
│       ├── dashboards/          # Dashboard definitions
│       │   ├── dashboard.yml    # Dashboard provisioning config
│       │   ├── home.json        # Home dashboard
│       │   ├── llm_benchmark.json  # Ollama dashboard
│       │   └── vllm_benchmark.json # vLLM dashboard
│       └── datasources/         # Data source definitions
│           └── datasource.yml   # Prometheus data source
├── data/                        # Model and application data
│   ├── ollama/                  # Ollama model storage
│   └── vllm/                    # vLLM model storage
├── results/                     # Benchmark results
├── src/                         # Source code
│   ├── monitoring/              # Prometheus metrics exporter
│   │   ├── __init__.py          # Package initialization
│   │   ├── prometheus_metrics.py # Metrics collection library
│   │   └── metrics_exporter.py  # Metrics simulation and export
│   └── run_benchmark.py         # Main benchmark script
├── docker-compose.yml           # Docker Compose configuration (GPU)
├── docker-compose.cpu.yml       # Docker Compose configuration (CPU-only)
└── requirements.txt             # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ollama](https://github.com/ollama/ollama) for the local inference engine
- [vLLM](https://github.com/vllm-project/vllm) for the high-performance inference
- [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) for monitoring
- [NVIDIA DCGM](https://developer.nvidia.com/dcgm) for GPU metrics 