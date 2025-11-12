# JadeVectorDB Distributed Deployment Guide

Complete guide for deploying JadeVectorDB in a distributed environment with examples for Docker, Kubernetes, and cloud platforms.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [Docker Compose Cluster](#docker-compose-cluster)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployments](#cloud-deployments)
6. [Production Best Practices](#production-best-practices)
7. [Monitoring and Observability](#monitoring-and-observability)

---

## Quick Start

### Single Node Docker Deployment

```bash
# Build the image
docker build -t jadevectordb:latest .

# Run single node
docker run -d \
  --name jadevectordb \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  jadevectordb:latest
```

### 3-Node Cluster with Docker Compose

```bash
# Start the distributed cluster
docker-compose -f docker-compose.distributed.yml up -d

# Verify cluster status
docker-compose -f docker-compose.distributed.yml ps

# View logs
docker-compose -f docker-compose.distributed.yml logs -f jadevectordb-master
```

---

## Docker Deployment

### Building the Image

The Dockerfile uses a multi-stage build for optimal image size:

```bash
# Development build with tests
docker build \
  --build-arg BUILD_TYPE=Debug \
  --build-arg BUILD_TESTS=ON \
  -t jadevectordb:dev .

# Production build
docker build \
  --build-arg BUILD_TYPE=Release \
  --build-arg BUILD_TESTS=OFF \
  -t jadevectordb:prod .
```

### Running a Single Node

```bash
docker run -d \
  --name jadevectordb-node1 \
  -p 8080:8080 \
  -p 8081:8081 \
  -e JADE_DB_NODE_ID=node1 \
  -e JADE_DB_NODE_ROLE=master \
  -e JADE_DB_PORT=8080 \
  -e JADE_DB_RPC_PORT=8081 \
  -e JADE_DB_LOG_LEVEL=INFO \
  -v jadevectordb-data:/app/data \
  -v jadevectordb-config:/app/config \
  jadevectordb:latest
```

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `JADE_DB_NODE_ID` | Unique node identifier | auto-generated | `node1`, `node2` |
| `JADE_DB_NODE_ROLE` | Node role | `worker` | `master`, `worker` |
| `JADE_DB_PORT` | HTTP API port | `8080` | `8080` |
| `JADE_DB_RPC_PORT` | Internal RPC port | `8081` | `8081` |
| `JADE_DB_CLUSTER_NODES` | Cluster node list | - | `node1,node2,node3` |
| `JADE_DB_MASTER_NODE` | Master node address | - | `jadevectordb-master` |
| `JADE_DB_LOG_LEVEL` | Logging level | `INFO` | `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `JADE_DB_DATA_DIR` | Data directory | `/app/data` | `/app/data` |
| `JADE_DB_CONFIG_DIR` | Config directory | `/app/config` | `/app/config` |

---

## Docker Compose Cluster

### Distributed Cluster Configuration

The `docker-compose.distributed.yml` file defines a complete 3-node cluster:

```yaml
version: '3.8'

services:
  jadevectordb-master:
    image: jadevectordb:latest
    container_name: jadevectordb-master
    ports:
      - "8080:8080"
      - "8081:8081"
    environment:
      - JADE_DB_NODE_ROLE=master
      - JADE_DB_NODE_ID=master-1
      - JADE_DB_CLUSTER_NODES=jadevectordb-master,jadevectordb-worker-1,jadevectordb-worker-2
      - JADE_DB_PORT=8080
      - JADE_DB_RPC_PORT=8081
    volumes:
      - jadevectordb_master_data:/data
      - jadevectordb_master_config:/config
    networks:
      - jadevectordb_cluster_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "/app/jadevectordb", "--health-check"]
      interval: 30s
      timeout: 10s
      retries: 3

  jadevectordb-worker-1:
    image: jadevectordb:latest
    container_name: jadevectordb-worker-1
    ports:
      - "8082:8080"
      - "8083:8081"
    environment:
      - JADE_DB_NODE_ROLE=worker
      - JADE_DB_NODE_ID=worker-1
      - JADE_DB_MASTER_NODE=jadevectordb-master
      - JADE_DB_CLUSTER_NODES=jadevectordb-master,jadevectordb-worker-1,jadevectordb-worker-2
    volumes:
      - jadevectordb_worker_1_data:/data
    networks:
      - jadevectordb_cluster_network
    depends_on:
      - jadevectordb-master
    restart: unless-stopped

  # worker-2 similar to worker-1...

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - jadevectordb_cluster_network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    networks:
      - jadevectordb_cluster_network
```

### Cluster Management Commands

```bash
# Start the cluster
docker-compose -f docker-compose.distributed.yml up -d

# Scale workers
docker-compose -f docker-compose.distributed.yml up -d --scale jadevectordb-worker=5

# Stop the cluster
docker-compose -f docker-compose.distributed.yml down

# Remove all data
docker-compose -f docker-compose.distributed.yml down -v

# View logs for specific service
docker-compose -f docker-compose.distributed.yml logs -f jadevectordb-master

# Restart a specific node
docker-compose -f docker-compose.distributed.yml restart jadevectordb-worker-1

# Execute command in container
docker-compose -f docker-compose.distributed.yml exec jadevectordb-master /bin/bash
```

---

## Kubernetes Deployment

### Prerequisites

```bash
# Ensure kubectl is installed and configured
kubectl version --client

# Create namespace
kubectl create namespace jadevectordb-system
```

### StatefulSet Deployment

```yaml
# k8s/jadevectordb-cluster.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: jadevectordb-cluster
  namespace: jadevectordb-system
spec:
  serviceName: "jadevectordb-headless"
  replicas: 3
  selector:
    matchLabels:
      app: jadevectordb
  template:
    metadata:
      labels:
        app: jadevectordb
    spec:
      containers:
      - name: jadevectordb
        image: jadevectordb/jadevectordb:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: rpc
        env:
        - name: JADE_DB_PORT
          value: "8080"
        - name: JADE_DB_RPC_PORT
          value: "8081"
        - name: JADE_DB_NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: data
          mountPath: /app/data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

### Deploy to Kubernetes

```bash
# Apply the configuration
kubectl apply -f k8s/jadevectordb-cluster.yaml

# Check status
kubectl get statefulsets -n jadevectordb-system
kubectl get pods -n jadevectordb-system

# View logs
kubectl logs -f jadevectordb-cluster-0 -n jadevectordb-system

# Scale the cluster
kubectl scale statefulset jadevectordb-cluster --replicas=5 -n jadevectordb-system

# Rolling update
kubectl set image statefulset/jadevectordb-cluster \
  jadevectordb=jadevectordb/jadevectordb:v2.0 \
  -n jadevectordb-system

# Delete deployment
kubectl delete -f k8s/jadevectordb-cluster.yaml
```

### Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: jadevectordb-service
  namespace: jadevectordb-system
spec:
  type: LoadBalancer
  selector:
    app: jadevectordb
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: rpc
    port: 8081
    targetPort: 8081
```

---

## Cloud Deployments

### AWS EKS Deployment

```bash
# Create EKS cluster
eksctl create cluster \
  --name jadevectordb-cluster \
  --region us-east-1 \
  --nodes 3 \
  --node-type t3.medium \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name jadevectordb-cluster

# Deploy JadeVectorDB
kubectl apply -f deployments/aws/jadevectordb-eks.yaml

# Create load balancer
kubectl apply -f deployments/aws/load-balancer.yaml

# Monitor deployment
kubectl get all -n jadevectordb-system
```

### Google Cloud GKE Deployment

```bash
# Create GKE cluster
gcloud container clusters create jadevectordb-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2 \
  --region=us-central1

# Get credentials
gcloud container clusters get-credentials jadevectordb-cluster --region us-central1

# Deploy
kubectl apply -f deployments/gcp/jadevectordb-deployment.yaml

# Expose service
kubectl expose deployment jadevectordb --type=LoadBalancer --port 8080
```

### Azure AKS Deployment

```bash
# Create resource group
az group create --name jadevectordb-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group jadevectordb-rg \
  --name jadevectordb-cluster \
  --node-count 3 \
  --enable-managed-identity \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group jadevectordb-rg --name jadevectordb-cluster

# Deploy
kubectl apply -f deployments/azure/jadevectordb-deployment.yaml
```

---

## Production Best Practices

### High Availability Configuration

```yaml
# Recommended production setup
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: jadevectordb-ha
spec:
  replicas: 5  # Odd number for quorum
  podManagementPolicy: Parallel
  updateStrategy:
    type: RollingUpdate
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - jadevectordb
            topologyKey: "kubernetes.io/hostname"
      containers:
      - name: jadevectordb
        image: jadevectordb:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Backup and Restore

```bash
# Backup data volumes
kubectl exec jadevectordb-cluster-0 -n jadevectordb-system -- \
  tar czf /tmp/backup.tar.gz /app/data

kubectl cp jadevectordb-system/jadevectordb-cluster-0:/tmp/backup.tar.gz \
  ./backup-$(date +%Y%m%d).tar.gz

# Restore from backup
kubectl cp ./backup-20250131.tar.gz \
  jadevectordb-system/jadevectordb-cluster-0:/tmp/restore.tar.gz

kubectl exec jadevectordb-cluster-0 -n jadevectordb-system -- \
  tar xzf /tmp/restore.tar.gz -C /
```

### Security Hardening

```yaml
# Security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true

# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: jadevectordb-network-policy
spec:
  podSelector:
    matchLabels:
      app: jadevectordb
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: jadevectordb-client
    ports:
    - protocol: TCP
      port: 8080
```

---

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'jadevectordb'
    static_configs:
      - targets: ['jadevectordb-master:8080', 'jadevectordb-worker-1:8080', 'jadevectordb-worker-2:8080']
    metrics_path: '/metrics'
```

### Grafana Dashboards

```bash
# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Login at http://localhost:3000
# Default credentials: admin/admin

# Import JadeVectorDB dashboard
# Dashboard ID: TBD (custom dashboard for JadeVectorDB metrics)
```

### Health Checks

```bash
# Check cluster health
curl http://localhost:8080/health

# Check node status
curl http://localhost:8080/api/cluster/nodes

# Check metrics
curl http://localhost:8080/metrics
```

### Logging Configuration

```yaml
# Fluentd for log aggregation
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/jadevectordb*.log
      pos_file /var/log/fluentd-jadevectordb.log.pos
      tag jadevectordb.*
      format json
    </source>

    <match jadevectordb.**>
      @type elasticsearch
      host elasticsearch
      port 9200
      logstash_format true
    </match>
```

---

## Troubleshooting

### Common Issues

#### Node Not Joining Cluster

```bash
# Check network connectivity
kubectl exec jadevectordb-cluster-0 -- ping jadevectordb-cluster-1

# Check logs
kubectl logs jadevectordb-cluster-0 | grep "cluster"

# Verify seed nodes configuration
kubectl describe pod jadevectordb-cluster-0 | grep SEED
```

#### Performance Issues

```bash
# Check resource usage
kubectl top pods -n jadevectordb-system

# Check for throttling
kubectl describe pod jadevectordb-cluster-0 | grep -i throttl

# Scale up resources
kubectl set resources statefulset jadevectordb-cluster \
  --limits=cpu=4000m,memory=8Gi \
  --requests=cpu=2000m,memory=4Gi
```

#### Data Replication Issues

```bash
# Check replication status via API
curl http://localhost:8080/api/replication/status

# Force replication
curl -X POST http://localhost:8080/api/replication/force

# Check shard distribution
curl http://localhost:8080/api/shards/distribution
```

---

## Example Application

Complete example of using JadeVectorDB in a distributed setup:

```cpp
#include "services/distributed_service_manager.h"
#include <iostream>

int main() {
    // Initialize distributed service manager
    auto manager = std::make_unique<DistributedServiceManager>();

    // Configure for production
    DistributedConfig config;
    config.cluster_host = "jadevectordb-cluster-0.jadevectordb-headless";
    config.cluster_port = 8081;
    config.seed_nodes = {
        "jadevectordb-cluster-1:8081",
        "jadevectordb-cluster-2:8081"
    };

    config.enable_sharding = true;
    config.enable_replication = true;
    config.enable_clustering = true;

    config.sharding_config.num_shards = 16;
    config.replication_config.default_replication_factor = 3;

    // Initialize and start
    if (!manager->initialize(config).value() ||
        !manager->start().value()) {
        std::cerr << "Failed to start distributed services\n";
        return 1;
    }

    // Join the cluster
    manager->join_cluster("jadevectordb-cluster-0", 8081);

    // Create database
    Database db;
    db.databaseId = "production-db";
    db.dimensions = 512;
    manager->create_shards_for_database(db);

    std::cout << "Distributed cluster ready!\n";

    // Keep running
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(60));

        // Health check
        if (!manager->is_cluster_healthy().value()) {
            std::cerr << "Cluster health check failed!\n";
        }
    }

    return 0;
}
```

---

For more information, see:
- [Distributed Services API Documentation](./distributed_services_api.md)
- [Architecture Overview](./architecture.md)
- [Performance Tuning Guide](./performance_tuning.md)
