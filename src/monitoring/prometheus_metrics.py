#!/usr/bin/env python3
import time
import threading
from prometheus_client import start_http_server, Gauge, Counter, Summary

# Define Prometheus metrics
class LLMMetrics:
    def __init__(self, port=8080):
        # Initialize metrics
        self.response_time = Gauge('llm_response_time_ms', 
                                 'Response time in milliseconds', 
                                 ['model', 'method'])
        
        self.tokens_per_second = Gauge('llm_tokens_per_second', 
                                     'Tokens generated per second', 
                                     ['model', 'method'])
        
        self.total_tokens = Counter('llm_total_tokens',
                                  'Total tokens processed',
                                  ['model', 'method', 'token_type'])
        
        self.requests_total = Counter('llm_requests_total',
                                    'Total number of requests',
                                    ['model', 'method', 'status'])
        
        self.response_time_summary = Summary('llm_response_time_summary',
                                          'Summary of response times',
                                          ['model', 'method'])
        
        # Start HTTP server for Prometheus to scrape
        start_http_server(port)
        print(f"Prometheus metrics server started on port {port}")
    
    def record_request(self, model, method, status="success"):
        """Record a request to the LLM service"""
        self.requests_total.labels(model=model, method=method, status=status).inc()
    
    def record_response_time(self, model, method, time_ms):
        """Record response time in milliseconds"""
        self.response_time.labels(model=model, method=method).set(time_ms)
        self.response_time_summary.labels(model=model, method=method).observe(time_ms)
    
    def record_tokens(self, model, method, input_tokens, output_tokens, time_ms):
        """Record token metrics"""
        self.total_tokens.labels(model=model, method=method, token_type="input").inc(input_tokens)
        self.total_tokens.labels(model=model, method=method, token_type="output").inc(output_tokens)
        
        # Calculate tokens per second (for output tokens)
        if time_ms > 0:
            tokens_per_sec = output_tokens / (time_ms / 1000)
            self.tokens_per_second.labels(model=model, method=method).set(tokens_per_sec)

# Create a global instance
metrics = LLMMetrics()

if __name__ == "__main__":
    # For testing
    print("LLM Metrics exporter running. Press Ctrl+C to exit.")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...") 