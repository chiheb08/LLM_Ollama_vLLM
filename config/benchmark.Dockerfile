FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    requests \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    psutil \
    prometheus-client \
    tqdm

# Copy benchmark scripts
COPY src /app/src

# Set up environment
ENV PYTHONUNBUFFERED=1

# Create results directory
RUN mkdir -p /app/results

# Run the benchmark when started
CMD ["python", "/app/src/run_benchmark.py"] 