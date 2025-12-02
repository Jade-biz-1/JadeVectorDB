# JadeVectorDB Developer Guide

Welcome to the JadeVectorDB project! This guide will help you understand the project structure, set up your development environment, and familiarize you with various important artifacts. This document also explains the general build, deploy, and test processes, as well as local and remote deployments, scaling, and monitoring capabilities.

## 1. Prerequisites

Before you begin, ensure you have the following tools installed on your system:

- **Git:** For version control.
- **C++ Toolchain:** A modern C++ compiler (GCC, Clang, or MSVC) that supports C++20.
- **CMake:** Version 3.20 or higher, for building the C++ backend.
- **Node.js:** Version 18 or higher, for the Next.js frontend.
- **Python:** Version 3.8 or higher, for the Python-based CLI tools.
- **Docker and Docker Compose:** For running the entire application stack locally in containers.
- **Eigen, FlatBuffers, Apache Arrow:** Math and serialization libraries used by the backend.
- **gRPC:** For distributed system communication.
- **Prometheus and Grafana:** For monitoring and observability (optional but recommended).

You can check if most of these are installed by running the provided prerequisite checker script:

```bash
sh .specify/scripts/bash/check-prerequisites.sh
```

## 2. Cloning the Repository

Start by cloning the repository to your local machine:

```bash
git clone <repository-url>
cd JadeVectorDB
```

## 3. Backend Setup (C++)

The backend consists of C++ microservices. It is managed by CMake.

1.  **Install Dependencies:** The project uses several third-party libraries (Eigen, OpenBLAS, FlatBuffers, gRPC, etc.). These are managed via CMake's `FetchContent` or are expected to be available on your system. The build script will handle them.

2.  **Build the Backend:**

    ```bash
    # Navigate to the backend directory
    cd backend

    # Configure the project with CMake
    cmake -B build

    # Build the project
    cmake --build build
    ```

    This will compile the backend services and place the executables in the `backend/build/` directory.

## 4. Frontend Setup (Next.js)

The frontend is a Next.js web application.

1.  **Navigate to the frontend directory:**

    ```bash
    cd frontend
    ```

2.  **Install Dependencies:**

    ```bash
    npm install
    ```

    This will download all the required Node.js packages.

## 5. CLI Setup (Python)

The Python CLI allows for easy interaction with the database from the command line.

1.  **Navigate to the Python CLI directory:**

    ```bash
    cd cli/python
    ```

2.  **Install in Editable Mode:** It's recommended to install the package in editable mode so that your changes are immediately reflected.

    ```bash
    pip install -e .
    ```

## 6. Running the System Locally with Docker Compose

The easiest way to run the entire stack (backend, frontend, and dependencies) is by using the provided Docker Compose configuration.

1.  **Ensure Docker is running.**

2.  **From the project root directory, run:**

    ```bash
    docker-compose up --build
    ```

    The `--build` flag ensures that the Docker images are rebuilt to reflect any code changes you've made. This command will start all the services defined in `docker-compose.yml`.

3.  **Accessing the Services:**
    -   **Web UI:** Open your browser and navigate to `http://localhost:3000`.
    -   **API:** The API will be accessible at `http://localhost:8080` (or the port configured in your environment).

## 7. Project Structure and Architecture

JadeVectorDB is a distributed vector database with a modular architecture consisting of several key components:

- **`backend/`** - Core C++ implementation with services for:
  - Vector storage and retrieval
  - Similarity search algorithms
  - Database management
  - Authentication and authorization
  - Distributed system services (Cluster, Sharding, Replication)

- **`frontend/`** - Next.js web interface with 23+ fully implemented pages for:
  - Database management
  - Vector operations
  - Similarity search
  - Performance monitoring
  - Cluster management
  - Security and access control

- **`cli/`** - Command-line interfaces in:
  - Python (full-featured)
  - Bash (lightweight)
  - JavaScript (Node.js-based)

- **`docker/`** - Containerization and orchestration configurations
- **`docs/`** - Documentation and architectural specifications
- **`scripts/`** - Development and deployment utilities

### Key Backend Services

The backend features several core services that handle different aspects of the system:

- **DatabaseService** - Manages database creation, configuration, and lifecycle
- **VectorStorageService** - Handles vector storage and retrieval operations
- **SimilaritySearchService** - Implements various similarity search algorithms (cosine, euclidean, dot product)
- **AuthenticationService** - JWT-based authentication with API key management
- **ClusterService** - Implements Raft-based consensus for master election
- **ShardingService** - Distributes data across cluster nodes using multiple strategies
- **ReplicationService** - Ensures data availability through configurable replication
- **DistributedServiceManager** - Coordinates all distributed services

### Authentication and Security

JadeVectorDB includes a comprehensive security system with:
- JWT-based authentication
- Role-based access control
- API key management
- Default users for development environments (admin, dev, test)

Default development credentials are automatically created in development and test environments only:

| Username | Password | Roles |
|----------|----------|-------|
| `admin` | `admin123` | admin, developer, user |
| `dev` | `dev123` | developer, user |
| `test` | `test123` | tester, user |

**Note:** These default users are automatically removed in production environments.

## 8. Development Workflow

- **Making Changes:** Make your code changes in the `backend`, `frontend`, or `cli` directories.
- **Testing (Backend):** To run the C++ tests, execute `ctest` from the `backend/build` directory.
- **Rebuilding:** If you are not using Docker, you will need to manually rebuild the specific component you are working on.
- **Running with Docker:** If you are using Docker, simply run `docker-compose up --build` again to restart the system with your latest changes.

## 9. Build and Test Processes

### Building the Backend

```bash
cd backend
mkdir build
cd build
cmake ..
make -j$(nproc)  # Use all available CPU cores for faster compilation
```

### Running Tests

The project includes comprehensive testing with 217+ test cases:

**Backend Tests:**
```bash
cd backend/build
./jadevectordb_tests
```

**Code Coverage:**
```bash
cd backend
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_COVERAGE=ON ..
make
make coverage
```

Coverage report will be available in the `coverage_report` directory.

### Static Analysis and Security Testing

**Static Analysis:**
```bash
cd backend
python3 ../scripts/run-static-analysis.py
```

**Security Testing:**
```bash
python3 scripts/run-security-tests.py --project-dir backend
```

## 10. Deployment Options

JadeVectorDB supports multiple deployment strategies depending on your requirements:

### Local Development Deployment

For local development, use the standard Docker Compose setup:

```bash
docker-compose up --build
```

This creates a single-node setup with the following services:
- JadeVectorDB API server on port 8081
- Web UI on port 3000
- Prometheus monitoring on port 9090
- Grafana dashboard on port 3001

### Distributed Deployment

For production and scaling, JadeVectorDB supports a distributed architecture with master-worker nodes:

```bash
docker-compose -f docker-compose.distributed.yml up --build
```

This creates a 3-node cluster with:
- 1 Master node (jadevectordb-master)
- 2 Worker nodes (jadevectordb-worker-1, jadevectordb-worker-2)
- Web UI connected to the master node
- Monitoring services (Prometheus and Grafana)

The distributed setup includes:
- **ClusterService**: Implements Raft-based consensus for master election
- **ShardingService**: Distributes data across cluster nodes using multiple strategies:
  - Hash-based (MurmurHash, FNV hash)
  - Range-based (partition by vector ID ranges)
  - Vector-based (cluster vectors by similarity)
  - Auto (system-selected optimal strategy)
- **ReplicationService**: Configurable replication with synchronous/asynchronous modes
- **Load balancing**: Health-aware request routing

### Kubernetes Deployment

For cloud deployments, JadeVectorDB can be deployed to Kubernetes using StatefulSets:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: jadevectordb-master
spec:
  serviceName: jadevectordb-master
  replicas: 3
  selector:
    matchLabels:
      app: jadevectordb
      role: master
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: jadevectordb-worker
spec:
  serviceName: jadevectordb-worker
  replicas: 5
  selector:
    matchLabels:
      app: jadevectordb
      role: worker
```

## 11. Scaling Strategies

### Horizontal Scaling

JadeVectorDB scales horizontally by adding worker nodes to the cluster:

1. **Sharding-based scaling**: Data is automatically distributed across nodes based on sharding strategy
2. **Query distribution**: Search requests are distributed across relevant shards in parallel
3. **Load balancing**: Queries are distributed across nodes based on their load and health status

### Vertical Scaling

For single-node deployments, you can scale vertically by increasing:
- CPU and memory resources
- Storage capacity
- Network bandwidth

### Auto-scaling Considerations

While auto-scaling is not yet implemented in the current version, the architecture supports:
- Dynamic node joining/leaving
- Automatic shard rebalancing
- Load-based scaling decisions

## 12. Monitoring and Observability

JadeVectorDB includes comprehensive monitoring and observability capabilities to help you understand system performance and health:

### Built-in Metrics

The system exposes various metrics through the API endpoints:
- **Health checks**: `/health` endpoint for system status
- **Performance metrics**: `/status` endpoint for detailed system status
- **API endpoints**:
  - `GET /health` - System health check
  - `GET /status` - Detailed system status with cluster information
  - `GET /v1/databases` - List all databases and their status
  - `GET /v1/monitoring/performance` - Performance metrics

### Prometheus Integration

The system is configured to work with Prometheus for metric collection:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'jadevectordb'
    static_configs:
      - targets: ['jadevectordb:8080']
    scrape_interval: 15s
    metrics_path: /metrics  # Placeholder - actual implementation may vary
```

### Grafana Dashboards

Grafana dashboards provide visualization of key metrics:
- Cluster status and node health
- Query performance (p50, p95, p99 latencies)
- Throughput metrics
- Resource utilization (CPU, memory, disk, network)
- Replication lag and consistency metrics

### Key Metrics to Monitor

| Metric Category | Key Metrics |
|-----------------|-------------|
| **Cluster Health** | Node status, master status, connectivity |
| **Performance** | Query latency (p50, p95, p99), throughput |
| **Data Distribution** | Shard count per node, data balance |
| **Replication** | Replication lag, consistency level |
| **Resources** | CPU, memory, disk, network per node |

### Distributed Tracing

The distributed system components support distributed tracing across:
- Master-worker communication
- Query execution across shards
- Replication operations
- Cluster management operations

## 13. Distributed System Management

### Cluster Management Commands

Use the command-line interface to manage your distributed cluster:

**Python CLI:**
```bash
# Check cluster status
jade-db cluster status

# List all nodes
jade-db nodes list

# Add a new node to the cluster
jade-db cluster add-node --node-id node-id --node-type worker

# Remove a node from the cluster
jade-db cluster remove-node --node-id node-id
```

**Shell CLI:**
```bash
# Check cluster health
bash cli/shell/scripts/jade-db.sh cluster-status

# List nodes
bash cli/shell/scripts/jade-db.sh list-nodes
```

### Master Election and Failover

The system uses Raft-based consensus for master election:
- Automatic master failover in case of master node failure
- Election timeout: 5-10 seconds for failover
- Consistent leader election with no split-brain scenarios
- Quorum-based decision making

### Data Migration and Rebalancing

The system supports live data migration between nodes:
- Zero-downtime migration for queries
- Automated shard rebalancing
- Configurable migration strategies (LIVE_COPY, SNAPSHOT, INCREMENTAL)
- Progress tracking and rollback capability

### Backup and Restore

Distributed backup and restore capabilities include:
- Cluster-wide snapshots
- Incremental backup strategies
- Point-in-time restore functionality
- Backup verification and integrity checks

### Security in Distributed Mode

Security measures in distributed mode:
- Node authentication within the cluster
- Secure RPC communication (to be implemented with TLS)
- Role-based access control across all nodes
- Secure API key distribution

## 14. Development Best Practices

### Code Organization

The project follows a modular architecture pattern:
- Each service is implemented in its own file with clear interfaces
- Service dependencies are managed through dependency injection
- Configuration is centralized with environment variable support
- Error handling follows a consistent Result<T> pattern

### Testing Strategy

The project maintains high test coverage with:
- Unit tests for individual components (80%+ coverage)
- Integration tests for service interactions
- End-to-end tests for complete workflows
- Chaos tests for failure scenarios
- Performance benchmarks for critical operations

### Performance Considerations

When developing for JadeVectorDB:
- Use efficient data structures (Eigen for linear algebra operations)
- Implement connection pooling for distributed communication
- Optimize query execution plans
- Consider memory usage for large vector operations
- Implement proper caching strategies

### Git Workflow

For contributing to the project:
1. Fork the repository
2. Create a feature branch (`feature/description-of-change`)
3. Make your changes following the existing code style
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request with a clear description

### Continuous Integration

The project uses CI/CD pipelines that:
- Run all tests on each commit
- Build Docker images automatically
- Perform static analysis and security checks
- Generate code coverage reports
- Run performance benchmarks

## 15. Resources and Documentation

For further information about JadeVectorDB, consult these additional resources:

- [API Documentation](docs/api_documentation.md) - Complete REST API reference
- [CLI Documentation](docs/cli-documentation.md) - Command-line interface reference
- [Architecture Documentation](docs/architecture.md) - Detailed system architecture
- [Search Functionality](docs/search_functionality.md) - Search algorithms and metadata filtering
- [Quiz System Documentation](docs/QUIZ_SYSTEM_DOCUMENTATION.md) - Interactive tutorial assessment platform
- [Distributed Implementation Plan](DISTRIBUTED_SYSTEM_IMPLEMENTATION_PLAN.md) - Detailed distributed system architecture
- [Deployment Guide](DOCKER_DEPLOYMENT.md) - Production deployment instructions
- [Contributing Guidelines](CONTRIBUTING.md) - Guidelines for contributing to the project
- [Consistency Report](CONSISTENCY_REPORT.md) - Report on distributed consistency models

## Conclusion

This guide provides a comprehensive overview of developing with JadeVectorDB. The system is designed to be scalable, reliable, and developer-friendly, with support for both single-node and distributed deployments.

For questions not covered in this guide, please check the project's issue tracker or create a new issue for assistance. The project is actively maintained, and the community welcomes contributions and feedback.

Happy developing!