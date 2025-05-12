# Ollama vs vLLM Benchmarking

This project provides a comprehensive benchmarking environment for comparing the performance of language models using Ollama and vLLM. It includes hardware monitoring and visualization tools to help you understand the performance characteristics of each framework.

## 🚀 Features

- **Complete Docker Environment**: All components run in Docker containers for easy setup and deployment
- **Performance Comparison**: Compare inference performance between Ollama and vLLM using the same models
- **Hardware Monitoring**: Track CPU, memory, and GPU usage during inference
- **Visualization Dashboard**: Grafana dashboard for real-time performance monitoring
- **Metrics Collection**: Prometheus integration for time-series metrics collection
- **Detailed Reports**: Generate CSV reports and visual plots comparing performance metrics
- **User-Friendly Interface**: Ollama WebUI for interactive testing

## 📋 Requirements

- Docker and Docker Compose
- CUDA-compatible GPU (optional but recommended)
- At least 8GB of RAM
- 10GB+ of free disk space (depends on model size)

## 🔧 Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/chiheb08/LLM_Ollama_vLLM
   cd ollama-vllm-comparison
   ```

2. Start the Docker environment:
   ```bash
   docker-compose up -d
   ```

3. Setup the models:
   ```bash
   ./scripts/setup_models.sh
   ```

## 📊 Running Benchmarks

To run a benchmark comparing Ollama and vLLM:

```bash
./scripts/run_benchmark.sh
```

This will:
1. Run inference across a set of prompts using both Ollama and vLLM
2. Collect performance metrics (response time, tokens per second, etc.)
3. Generate comparison charts and tables
4. Save results to the `results` directory

### Customizing Benchmarks

You can customize the benchmark by modifying the script parameters:

```bash
docker exec -it benchmark python /app/src/run_benchmark.py \
  --models llama2 mistral tinyllama \
  --iterations 5 \
  --concurrent 3 \
  --max-tokens 1024 \
  --plot
```

Available parameters:
- `--models`: Space-separated list of models to benchmark
- `--iterations`: Number of runs for each prompt
- `--concurrent`: Number of concurrent requests
- `--max-tokens`: Maximum tokens to generate
- `--plot`: Generate visualization plots
- `--prompts-file`: Custom JSON file with prompts
- `--output-dir`: Directory to save results

## 📈 Monitoring

Access the monitoring dashboards:

- **Grafana**: http://localhost:3001 (Username: admin, Password: password)
- **Prometheus**: http://localhost:9090
- **Ollama WebUI**: http://localhost:3000

### Key Metrics Collected

The benchmarking system collects the following metrics:
- **Response Time**: Total time to generate a response (ms)
- **Tokens per Second**: Generation speed
- **Memory Usage**: RAM consumption during inference
- **CPU/GPU Usage**: Processor load during inference
- **Total Tokens**: Input and output token counts

### Real-time Monitoring Architecture

The metrics collection works as follows:
1. The `prometheus_metrics.py` module runs inside the benchmark container on port 8080
2. During benchmark runs, it records metrics about response times and token generation
3. Prometheus scrapes these metrics from the benchmark container
4. Grafana visualizes the data from Prometheus in dashboards
5. Node Exporter provides system-level metrics about CPU, memory, and GPU usage

## 🔍 Understanding vLLM

[vLLM](https://github.com/vllm-project/vllm) is an open-source library for LLM inference that introduces several optimizations:

### Key Features of vLLM

1. **PagedAttention**: A memory-efficient attention mechanism that manages KV cache (key-value pairs from previous tokens) using a paging system inspired by operating systems.

2. **Continuous Batching**: Efficiently processes multiple requests concurrently, dramatically improving throughput.

3. **Optimized CUDA Kernels**: Highly optimized implementations of key operations.

4. **Tensor Parallelism**: Distributes large models across multiple GPUs.

### How vLLM Improves Performance

- **Memory Efficiency**: By using PagedAttention, vLLM can better manage memory, allowing it to serve more concurrent requests with the same hardware.
  
- **Throughput**: The continuous batching approach enables processing multiple requests at different stages simultaneously, maximizing GPU utilization.

- **Latency**: Optimized CUDA kernels and efficient memory management reduce the time to generate each token.

## 🔄 Comparison with Ollama

Ollama is an easy-to-use tool for running LLMs locally, with excellent developer experience. This project helps you understand the performance trade-offs between:

- **Ollama**: User-friendly, easy setup, good for general-purpose use
- **vLLM**: Optimized for maximum performance, better for production environments or serving multiple users

## 📚 Models

By default, the benchmark uses the TinyLlama-1.1B-Chat model, but you can modify the scripts to test with other models.

### Adding New Models

#### For Ollama:
1. Pull the model using Ollama's CLI:
   ```bash
   docker exec -it ollama ollama pull <model-name>
   ```
2. Update the `MODEL` variable in `src/ollama/setup.sh` or pass the model name to the benchmark script

#### For vLLM:
1. Update the environment variables in `config/vllm.Dockerfile`:
   ```
   ENV MODEL_PATH="/models/<model-name>"
   ENV MODEL_ID="<hf-repo-id>/<model-name>"
   ```
2. Rebuild the vLLM container:
   ```bash
   docker-compose build vllm
   docker-compose up -d vllm
   ```

### Supported Models

The following models have been tested with this setup:
- TinyLlama (1.1B parameters)
- Llama 2 (7B parameters)
- Mistral (7B parameters)

Larger models (13B+) may require more memory and GPU resources.

## 📂 Project Structure

```
ollama-vllm-comparison/
├── config/                  # Configuration files
│   ├── benchmark.Dockerfile # Dockerfile for benchmark container
│   ├── vllm.Dockerfile      # Dockerfile for vLLM container
│   ├── prometheus.yml       # Prometheus configuration
│   └── grafana/             # Grafana dashboards and datasources
├── data/                    # Model and application data
├── results/                 # Benchmark results
├── scripts/                 # Helper scripts
│   ├── run_benchmark.sh     # Script to run benchmarks
│   └── setup_models.sh      # Script to setup models
├── src/                     # Source code
│   ├── monitoring/          # Prometheus metrics exporter
│   │   └── prometheus_metrics.py # Metrics collection
│   ├── ollama/              # Ollama-specific code
│   │   └── setup.sh         # Ollama model setup script
│   ├── vllm/                # vLLM-specific code
│   │   └── server.py        # vLLM API server
│   └── run_benchmark.py     # Main benchmark script
└── docker-compose.yml       # Docker Compose configuration
```

## 🔧 Troubleshooting

### Common Issues

**Issue**: Docker containers fail to start
**Solution**: Check GPU drivers and Docker GPU settings

```bash
# Verify NVIDIA drivers are properly installed
nvidia-smi

# Check Docker GPU support
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Issue**: Models are not downloading
**Solution**: Check network connectivity and model availability

```bash
# Check if Ollama service is running
docker exec -it ollama ollama list

# Check vLLM logs
docker logs vllm
```

**Issue**: Metrics are not showing in Grafana
**Solution**: Verify Prometheus is scraping the metrics

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

## 🤝 Contributing

Contributions are welcome! Feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References & Resources

### Core Technologies

#### Ollama
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Ollama Documentation](https://ollama.ai/docs)
- [Ollama Model Library](https://ollama.ai/library)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)

#### vLLM
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [vLLM API Reference](https://vllm.readthedocs.io/en/latest/serving/openai_compatible_server.html)

#### Monitoring Stack
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Node Exporter Documentation](https://prometheus.io/docs/guides/node-exporter/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)

#### Docker & Container Orchestration
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)

### LLM Models
- [TinyLlama Repository](https://github.com/jzhang38/TinyLlama)
- [Llama 2 Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
- [Mistral AI Documentation](https://docs.mistral.ai/)
- [HuggingFace Model Hub](https://huggingface.co/models)

### Visualization & Data Analysis
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Additional Resources
- [LLM Inference Optimization Guide](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Transformer Models: An Introduction and Catalog](https://arxiv.org/abs/2302.07730)
- [Benchmarking Generation Throughput of LLMs](https://huggingface.co/blog/benchmark-llms)
- [Efficient Memory Management for Large Language Model Serving](https://www.anyscale.com/blog/continuous-batching-llm-inference) 