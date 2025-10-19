#!/usr/bin/env python3

# Chaos Metrics Generator for JadeVectorDB
# This script generates metrics during chaos experiments for monitoring

import time
import random
import json
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import argparse

# Chaos experiment metrics
chaos_experiment_active = Gauge('chaos_experiment_active', 'Indicates if a chaos experiment is active', ['experiment_type'])
chaos_experiment_count = Counter('chaos_experiments_total', 'Total number of chaos experiments executed', ['experiment_type'])
chaos_experiment_duration = Histogram('chaos_experiment_duration_seconds', 'Duration of chaos experiments', ['experiment_type'])

class ChaosMetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            # Generate a simple response with current chaos metrics
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            
            # Add some chaos-related metrics
            metrics = [
                f"# HELP chaos_experiment_active Indicates if a chaos experiment is active",
                f"# TYPE chaos_experiment_active gauge",
                f'chaos_experiment_active{{experiment_type="network_partition"}} {random.choice([0, 1])}',
                f'chaos_experiment_active{{experiment_type="node_failure"}} {random.choice([0, 1])}',
                f'chaos_experiment_active{{experiment_type="resource_exhaustion"}} {random.choice([0, 1])}',
                f"",
                f"# HELP chaos_experiments_total Total number of chaos experiments executed",
                f"# TYPE chaos_experiments_total counter",
                f'chaos_experiments_total{{experiment_type="network_partition"}} {random.randint(0, 100)}',
                f'chaos_experiments_total{{experiment_type="node_failure"}} {random.randint(0, 100)}',
                f'chaos_experiments_total{{experiment_type="resource_exhaustion"}} {random.randint(0, 100)}',
                f"",
                f"# HELP chaos_experiment_duration_seconds Duration of chaos experiments",
                f"# TYPE chaos_experiment_duration_seconds histogram",
                f'chaos_experiment_duration_seconds_sum{{experiment_type="network_partition"}} {random.uniform(30, 300)}',
                f'chaos_experiment_duration_seconds_sum{{experiment_type="node_failure"}} {random.uniform(30, 300)}',
                f'chaos_experiment_duration_seconds_sum{{experiment_type="resource_exhaustion"}} {random.uniform(30, 300)}',
            ]
            
            response = "\n".join(metrics)
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def simulate_chaos_metrics(port=8000):
    """Start an HTTP server to expose chaos metrics"""
    print(f"Starting chaos metrics server on port {port}")
    server = HTTPServer(('', port), ChaosMetricsHandler)
    server.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chaos Metrics Generator for JadeVectorDB')
    parser.add_argument('--port', type=int, default=8000, help='Port to run metrics server on')
    args = parser.parse_args()
    
    # Start the metrics server in a separate thread
    server_thread = threading.Thread(target=simulate_chaos_metrics, args=(args.port,))
    server_thread.daemon = True
    server_thread.start()
    
    print(f"Chaos metrics server started on port {args.port}")
    print("Metrics endpoint: http://localhost:{}/metrics".format(args.port))
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down chaos metrics server...")
        server_thread.join(timeout=1)