# Re-ranking Server Deployment Guide

Comprehensive deployment guide for the Python re-ranking server across different environments.

## Table of Contents

1. [Overview](#overview)
2. [Development Environment](#development-environment)
3. [Single-Node Production](#single-node-production)
4. [Distributed Cluster](#distributed-cluster)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Dependency Management](#dependency-management)
8. [Performance Tuning](#performance-tuning)

---

## Overview

### Architecture Decision (Phase 1)

The re-ranking server runs as a **Python subprocess** spawned by each JadeVectorDB instance. This phase is optimal for:
- Development and testing
- Single-node production deployments
- Small clusters (< 5 nodes)

**See**: `docs/architecture.md` for Phases 2 (dedicated microservice) and 3 (ONNX native).

### System Requirements

- **Python**: 3.9+ (3.12+ recommended)
- **RAM**: 1-2GB per subprocess (model + runtime)
- **Disk**: 500MB-1GB (model cache + dependencies)
- **Network**: Internet access for initial model download

---

## Development Environment

### Local Machine (Recommended)

**Best for**: Development, debugging, testing

#### Step 1: Install Dependencies

```bash
cd /path/to/JadeVectorDB/python

# Option A: System-wide (simple)
pip3 install -r requirements.txt

# Option B: Virtual environment (isolated, recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

#### Step 2: Verify Installation

```bash
python3 test_reranking_server.py
```

Expected: All tests pass ✅

#### Step 3: Run JadeVectorDB

```bash
cd ../backend/build
./jadevectordb
```

The C++ application will automatically spawn the Python subprocess.

### Configuration

Set environment variables for development:

```bash
# Optional: Use different model
export RERANKING_MODEL_PATH=cross-encoder/ms-marco-MiniLM-L-6-v2

# Optional: Adjust batch size
export RERANKING_BATCH_SIZE=32

# Optional: Enable debug logging
export RERANKING_LOG_LEVEL=DEBUG

# Run JadeVectorDB
./backend/build/jadevectordb
```

---

## Single-Node Production

### Bare Metal / VM Deployment

**Best for**: Production deployments on a single server

#### Step 1: Create Service User

```bash
# Create dedicated user (security best practice)
sudo useradd -r -s /bin/bash -d /opt/jadevectordb jadevectordb
sudo mkdir -p /opt/jadevectordb
sudo chown jadevectordb:jadevectordb /opt/jadevectordb
```

#### Step 2: Install System-Wide Dependencies

```bash
# Switch to service user
sudo su - jadevectordb

# Install Python dependencies in user space
cd /opt/jadevectordb
python3 -m pip install --user sentence-transformers torch transformers numpy
```

**Why `--user`?**: Installs in `~/.local/lib/python3.x/site-packages`, avoiding sudo requirements.

#### Step 3: Download Model Cache (Pre-warm)

```bash
# Pre-download model to avoid startup delay
python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

Model cached at: `~/.cache/huggingface/`

#### Step 4: Configure SystemD Service

Create `/etc/systemd/system/jadevectordb.service`:

```ini
[Unit]
Description=JadeVectorDB - High-Performance Vector Database
After=network.target

[Service]
Type=simple
User=jadevectordb
Group=jadevectordb
WorkingDirectory=/opt/jadevectordb

# Ensure Python can find dependencies
Environment="PATH=/opt/jadevectordb/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=/opt/jadevectordb/.local/lib/python3.12/site-packages"

# Re-ranking configuration
Environment="RERANKING_MODEL_PATH=cross-encoder/ms-marco-MiniLM-L-6-v2"
Environment="RERANKING_BATCH_SIZE=32"
Environment="RERANKING_LOG_LEVEL=INFO"

# JadeVectorDB configuration
Environment="JADEVECTORDB_DATA_DIR=/var/lib/jadevectordb"
Environment="JADEVECTORDB_PORT=8080"
Environment="JADEVECTORDB_LOG_LEVEL=info"

ExecStart=/opt/jadevectordb/jadevectordb
Restart=on-failure
RestartSec=10s

# Resource limits
LimitNOFILE=65536
MemoryMax=4G

[Install]
WantedBy=multi-user.target
```

#### Step 5: Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable jadevectordb
sudo systemctl start jadevectordb
sudo systemctl status jadevectordb
```

#### Monitoring

```bash
# Check logs
sudo journalctl -u jadevectordb -f

# Check subprocess status
ps aux | grep reranking_server

# Check resource usage
systemctl status jadevectordb
```

---

## Distributed Cluster

### Multi-Node Deployment

**Architecture**: Each worker node runs its own Python subprocess.

```
Master Node:
  - JadeVectorDB (C++)
  - Python Subprocess (reranking_server.py)

Worker Node 1:
  - JadeVectorDB (C++)
  - Python Subprocess (reranking_server.py)

Worker Node 2:
  - JadeVectorDB (C++)
  - Python Subprocess (reranking_server.py)
```

#### Deployment Strategy

**Option A: Identical Setup (Recommended)**

Deploy same configuration to all nodes:

```bash
# On each node
sudo su - jadevectordb
cd /opt/jadevectordb
python3 -m pip install --user -r requirements.txt

# Pre-download model on each node
python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

**Option B: Shared NFS Cache**

Share model cache across nodes (reduces disk usage):

```bash
# On master node
sudo mkdir -p /nfs/jadevectordb/huggingface-cache
sudo chown -R jadevectordb:jadevectordb /nfs/jadevectordb

# On each node, symlink to NFS
ln -s /nfs/jadevectordb/huggingface-cache ~/.cache/huggingface
```

**⚠️ Caution**: NFS latency may impact model loading time.

#### Ansible Playbook (Automated Deployment)

Create `deploy_reranking.yml`:

```yaml
---
- name: Deploy JadeVectorDB Re-ranking Dependencies
  hosts: jadevectordb_cluster
  become: yes
  tasks:
    - name: Install Python dependencies
      pip:
        name:
          - sentence-transformers>=2.2.0
          - torch>=2.0.0
          - transformers>=4.30.0
          - numpy>=1.23.0
        state: present
        extra_args: --user
      become_user: jadevectordb

    - name: Pre-download model cache
      shell: |
        python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
      become_user: jadevectordb
      args:
        creates: ~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2

    - name: Copy Python scripts
      copy:
        src: "{{ item }}"
        dest: /opt/jadevectordb/python/
        owner: jadevectordb
        group: jadevectordb
        mode: '0755'
      loop:
        - reranking_server.py
        - requirements.txt
```

Run:

```bash
ansible-playbook -i inventory.yml deploy_reranking.yml
```

---

## Docker Deployment

### Dockerfile with Python Dependencies

Update the Dockerfile to include Python dependencies:

```dockerfile
#############################################################################
# Stage 1: Builder
#############################################################################
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git ninja-build pkg-config \
    libboost-all-dev libssl-dev libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY backend/ /build/

RUN chmod +x build.sh && \
    ./build.sh --type Release --no-tests --no-benchmarks --jobs 4

#############################################################################
# Stage 2: Runtime
#############################################################################
FROM ubuntu:24.04 AS runtime

LABEL maintainer="JadeVectorDB Team"

# Install runtime dependencies + Python
RUN apt-get update && apt-get install -y \
    ca-certificates libstdc++6 curl \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r jadevectordb && \
    useradd -r -g jadevectordb -s /bin/bash -d /app jadevectordb

WORKDIR /app

# Copy C++ executable
COPY --from=builder /build/build/jadevectordb /app/jadevectordb

# Copy Python scripts
COPY python/ /app/python/

# Install Python dependencies as jadevectordb user
RUN chown -R jadevectordb:jadevectordb /app && \
    mkdir -p /app/data /app/logs /app/config

USER jadevectordb

# Install Python dependencies
RUN python3 -m pip install --user --no-cache-dir -r /app/python/requirements.txt

# Pre-download model cache (reduces startup time)
RUN python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Ensure Python packages are in PATH
ENV PATH="/app/.local/bin:${PATH}"
ENV PYTHONPATH="/app/.local/lib/python3.12/site-packages"

EXPOSE 8080

# Environment variables
ENV JADEVECTORDB_PORT=8080 \
    JADEVECTORDB_LOG_LEVEL=info \
    JADEVECTORDB_DATA_DIR=/app/data \
    RERANKING_MODEL_PATH=cross-encoder/ms-marco-MiniLM-L-6-v2 \
    RERANKING_BATCH_SIZE=32 \
    RERANKING_LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

STOPSIGNAL SIGTERM

CMD ["/app/jadevectordb"]
```

### Build and Run

```bash
# Build image
docker build -t jadevectordb:latest -f Dockerfile .

# Run container
docker run -d \
  --name jadevectordb \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -e RERANKING_BATCH_SIZE=64 \
  jadevectordb:latest
```

### Docker Compose (Multi-Node)

Update `docker-compose.distributed.yml`:

```yaml
services:
  jadevectordb-master:
    build:
      context: .
      dockerfile: Dockerfile
    image: jadevectordb:latest
    environment:
      - JADEVECTORDB_NODE_TYPE=master
      - RERANKING_MODEL_PATH=cross-encoder/ms-marco-MiniLM-L-6-v2
      - RERANKING_BATCH_SIZE=32
    volumes:
      - master_data:/app/data
      # Share model cache across nodes (optional)
      - huggingface_cache:/app/.cache/huggingface
    networks:
      - jadevectordb_network

  jadevectordb-worker-1:
    image: jadevectordb:latest
    environment:
      - JADEVECTORDB_NODE_TYPE=worker
      - RERANKING_MODEL_PATH=cross-encoder/ms-marco-MiniLM-L-6-v2
      - RERANKING_BATCH_SIZE=32
    volumes:
      - worker1_data:/app/data
      - huggingface_cache:/app/.cache/huggingface
    networks:
      - jadevectordb_network

  jadevectordb-worker-2:
    image: jadevectordb:latest
    environment:
      - JADEVECTORDB_NODE_TYPE=worker
      - RERANKING_MODEL_PATH=cross-encoder/ms-marco-MiniLM-L-6-v2
      - RERANKING_BATCH_SIZE=32
    volumes:
      - worker2_data:/app/data
      - huggingface_cache:/app/.cache/huggingface
    networks:
      - jadevectordb_network

volumes:
  master_data:
  worker1_data:
  worker2_data:
  huggingface_cache:  # Shared model cache

networks:
  jadevectordb_network:
    driver: bridge
```

Run:

```bash
docker-compose -f docker-compose.distributed.yml up -d
```

---

## Kubernetes Deployment

### ConfigMap for Python Scripts

Create `k8s/reranking-config.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: reranking-scripts
  namespace: jadevectordb
data:
  reranking_server.py: |
    # Content of reranking_server.py
    # (embed entire script here)
```

### StatefulSet with Init Container

Create `k8s/jadevectordb-statefulset.yaml`:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: jadevectordb
  namespace: jadevectordb
spec:
  serviceName: jadevectordb
  replicas: 3
  selector:
    matchLabels:
      app: jadevectordb
  template:
    metadata:
      labels:
        app: jadevectordb
    spec:
      initContainers:
        # Pre-download model to shared volume
        - name: model-downloader
          image: python:3.12-slim
          command:
            - /bin/bash
            - -c
            - |
              pip install --no-cache-dir sentence-transformers torch transformers
              python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
              cp -r ~/.cache/huggingface /cache/
          volumeMounts:
            - name: model-cache
              mountPath: /cache

      containers:
        - name: jadevectordb
          image: jadevectordb:latest
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: RERANKING_MODEL_PATH
              value: "cross-encoder/ms-marco-MiniLM-L-6-v2"
            - name: RERANKING_BATCH_SIZE
              value: "32"
            - name: PYTHONPATH
              value: "/app/.local/lib/python3.12/site-packages"
          volumeMounts:
            - name: data
              mountPath: /app/data
            - name: model-cache
              mountPath: /app/.cache/huggingface
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10

  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: [ "ReadWriteOnce" ]
        resources:
          requests:
            storage: 10Gi
    - metadata:
        name: model-cache
      spec:
        accessModes: [ "ReadWriteOnce" ]
        resources:
          requests:
            storage: 5Gi
```

Deploy:

```bash
kubectl apply -f k8s/jadevectordb-statefulset.yaml
```

---

## Dependency Management

### Lock File for Reproducibility

Generate a lock file to freeze versions:

```bash
cd python
pip freeze > requirements-lock.txt
```

Use in production:

```bash
pip install -r requirements-lock.txt
```

### Offline Installation (Air-Gapped Environments)

#### Step 1: Download Dependencies

On a machine with internet:

```bash
mkdir -p /tmp/python-packages
pip download -r requirements.txt -d /tmp/python-packages
```

#### Step 2: Transfer and Install

Copy `/tmp/python-packages` to air-gapped machine:

```bash
pip install --no-index --find-links=/tmp/python-packages -r requirements.txt
```

#### Step 3: Download Model

Download model manually:

```bash
# On internet-connected machine
from huggingface_hub import snapshot_download
snapshot_download("cross-encoder/ms-marco-MiniLM-L-6-v2", cache_dir="/tmp/models")
```

Copy `/tmp/models` to air-gapped machine at `~/.cache/huggingface/`

---

## Performance Tuning

### CPU Optimization

```bash
# Increase batch size (if RAM allows)
export RERANKING_BATCH_SIZE=64

# Use all CPU cores for PyTorch
export OMP_NUM_THREADS=$(nproc)

# Enable MKL optimizations (if available)
export MKL_NUM_THREADS=$(nproc)
```

### GPU Acceleration

Install CUDA-enabled PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU:

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Performance Gain**: ~3-5x faster inference

### Memory Optimization

```bash
# Reduce batch size to save memory
export RERANKING_BATCH_SIZE=16

# Use smaller model
export RERANKING_MODEL_PATH=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Monitoring

Add Prometheus metrics (future enhancement):

```python
# In reranking_server.py
from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter('reranking_requests_total', 'Total requests')
LATENCY = Histogram('reranking_latency_seconds', 'Inference latency')
```

---

## Summary

| Deployment Scenario | Installation Method | Notes |
|---------------------|---------------------|-------|
| **Development** | `pip install -r requirements.txt` | Use virtual environment |
| **Single-Node Prod** | `pip install --user -r requirements.txt` | SystemD service |
| **Distributed Cluster** | Ansible playbook or identical setup per node | Each node has subprocess |
| **Docker** | Build with Python deps in Dockerfile | Pre-download model in image |
| **Kubernetes** | InitContainer for model download | Use PVC for model cache |

---

## Next Steps

1. **Install dependencies** (Task #6)
2. **Test integration** with C++ subprocess
3. **Benchmark performance** (Task #5)
4. **Plan Phase 2** (dedicated microservice) for large clusters

**For more information**:
- `python/INSTALL.md` - Installation guide
- `python/README.md` - Usage documentation
- `docs/architecture.md` - Architecture details
- `TasksTracking/16-hybrid-search-reranking-analytics.md` - Task tracking
