# JadeVectorDB Architecture Documentation

## Overview

JadeVectorDB is a high-performance distributed vector database designed to efficiently store, index, and search vector embeddings. The system is built with C++20 for maximum performance and uses a microservices architecture to enable independent scaling of components.

## System Architecture

### High-Level Architecture

```
                    +------------------+
                    |   Client Apps    |
                    +------------------+
                            |
                    +------------------+
                    |   API Gateway    |  (REST/gRPC)
                    +------------------+
                            |
        +-------------------+-------------------+
        |                   |                   |
+----------------+  +----------------+  +----------------+
|  Vector Storage|  | Similarity     |  |   Database     |
|   Service      |  |   Search       |  |   Service      |
+----------------+  +----------------+  +----------------+
        |                   |                   |
+----------------+  +----------------+  +----------------+
|   Index        |  |   Lifecycle    |  |   Metrics      |
|   Service      |  |   Service      |  |   Service      |
+----------------+  +----------------+  +----------------+
        |                   |                   |
+----------------+  +----------------+  +----------------+
|   Security     |  |   Monitoring   |  |   Embedding    |
|   Service      |  |   Service      |  |   Service      |
+----------------+  +----------------+  +----------------+
```

### Core Services

#### 1. Vector Storage Service
- **Responsibility**: Store, retrieve, update, and delete vector embeddings
- **Technology**: Memory-mapped files for persistence, custom binary format optimized for vector operations
- **Features**:
  - Atomic operations for consistency
  - Batch operations for efficiency
  - Metadata storage alongside vectors
  - Validation against database schema
  - **Persistent storage** with durability guarantees
  - **Cross-platform support** (Unix/Linux mmap, Windows CreateFileMapping)
  - **LRU caching** of open files for efficient memory usage
  - **Lazy loading** for fast startup times

#### 2. Similarity Search Service
- **Responsibility**: Perform vector similarity searches with various algorithms
- **Supported Algorithms**:
  - Cosine similarity
  - Euclidean distance
  - Dot product
- **Optimizations**: SIMD operations, efficient data structures, multi-threading

#### 3. Database Service
- **Responsibility**: Manage database instances and their configurations
- **Features**:
  - Create, read, update, delete operations
  - Configuration validation
  - Schema management
  - Access control

#### 4. Index Service
- **Responsibility**: Manage multiple indexing algorithms for optimized search
- **Supported Index Types**:
  - FLAT (brute-force search, exact results)
  - HNSW (hierarchical navigable small world)
  - IVF (inverted file)
  - LSH (locality sensitive hashing)
- **Features**:
  - Configurable parameters per index type
  - Asynchronous index building
  - Index performance monitoring

### Data Architecture

#### Persistence Layer

JadeVectorDB employs a sophisticated persistence layer that enables durable, high-performance vector storage using memory-mapped files.

##### Overview
- **Storage Model**: Persistent memory-mapped binary files per database
- **Format**: Custom binary format optimized for SIMD operations and cache efficiency
- **Durability**: Automatic flushing with configurable intervals, graceful shutdown handling
- **Performance**: Sub-millisecond access times, minimal memory overhead with LRU eviction
- **Scalability**: Supports millions of vectors with lazy loading and efficient indexing

##### MemoryMappedVectorStore

The `MemoryMappedVectorStore` class provides the core persistence mechanism:

**Architecture**:
- Each database corresponds to one `.jvdb` binary file
- Files are memory-mapped for zero-copy access
- Automatic file growth with exponential allocation strategy
- Cross-platform support (Unix mmap, Windows CreateFileMapping)

**Binary File Format**:
```
┌─────────────────────────────┐
│  Header (64 bytes)          │  Magic: 0x4A564442 ("JVDB")
│  - Magic number (4B)        │  Version, metadata, capacity
│  - Version (4B)             │
│  - Vector count (8B)        │
│  - Dimension (4B)           │
│  - Capacity (8B)            │
│  - Reserved (36B)           │
├─────────────────────────────┤
│  Index Section              │  32 bytes per vector:
│  - Entry 0 (32B)            │  - ID (8B)
│  - Entry 1 (32B)            │  - Offset (8B)
│  - ...                      │  - Size (8B)
│  - Entry N-1 (32B)          │  - Flags (8B)
├─────────────────────────────┤
│  Data Section               │  SIMD-aligned (32-byte):
│  - Vector 0 data            │  - Float32 components
│  - Vector 1 data            │  - Metadata (JSON)
│  - ...                      │  - Padding for alignment
│  - Vector N-1 data          │
└─────────────────────────────┘
```

**Key Features**:
- **Zero-copy access**: Vectors are accessed directly in mapped memory
- **SIMD alignment**: Data aligned to 32-byte boundaries for AVX/AVX2 operations
- **Efficient indexing**: O(log n) vector lookup using sorted index section
- **Atomic updates**: In-place updates with atomic metadata writes
- **Graceful growth**: Files expand automatically with exponential allocation

##### PersistentDatabasePersistence

The `PersistentDatabasePersistence` class integrates memory-mapped storage with the database service:

**Architecture**:
- Manages lifecycle of multiple `MemoryMappedVectorStore` instances
- Implements LRU eviction policy for limiting open file descriptors
- Coordinates flushing across all databases
- Handles database creation, deletion, and schema validation

**LRU Eviction Policy**:
- Configurable `max_open_files` parameter (default: 100)
- Automatically closes least-recently-used stores when limit is reached
- Re-opens stores on-demand with lazy loading
- Maintains consistent state across eviction/reload cycles

**Configuration Parameters**:
```cpp
PersistentDatabasePersistence(
    const std::string& storage_path,      // Root directory for .jvdb files
    size_t max_open_files = 100,          // Max concurrent open files
    std::chrono::seconds flush_interval = std::chrono::seconds(300)  // Auto-flush period
);
```

##### Durability and Crash Recovery

**Flush Mechanisms**:
1. **Automatic periodic flushing**: Background thread flushes all dirty stores at configurable intervals
2. **Manual flushing**: Explicit `flush()` calls for critical checkpoints
3. **Graceful shutdown**: Signal handlers (SIGTERM, SIGINT) ensure clean shutdown with full flush

**VectorFlushManager**:
- Central coordinator for flush operations
- Tracks dirty databases requiring flush
- Handles periodic flush scheduling
- Provides explicit flush APIs for user control

**Crash Recovery**:
- Flushed data is guaranteed to survive process termination
- Unflushed data (since last flush) may be lost on ungraceful shutdown
- Header integrity checks on startup detect corruption
- Automatic fallback to last known good state

**Recovery Process**:
1. Open memory-mapped file
2. Validate magic number (0x4A564442) and version
3. Verify vector count ≤ capacity
4. Rebuild in-memory index from file index section
5. Resume normal operations

##### Performance Characteristics

**Startup Time**:
- Lazy loading: O(1) - only header is read initially
- Full index load: O(n) where n = vector count
- Typical startup: < 10ms for 100K vectors

**Operation Latency**:
- Vector retrieval: < 1µs (memory-mapped, zero-copy)
- Vector storage: ~10-50µs (includes index update, no flush)
- Flush operation: ~1-10ms per database (depends on OS page cache)

**Throughput**:
- Store: 50,000-100,000 vectors/sec (batch operations)
- Retrieve: 200,000-500,000 vectors/sec (hot cache)
- Update: 40,000-80,000 vectors/sec

**Memory Efficiency**:
- Per-database overhead: ~200KB (index structure, LRU tracking)
- LRU eviction keeps memory usage bounded
- Typical memory usage: 2-5MB per 100K vectors (excluding mmap)

##### Integration with Core Services

**DatabaseService Integration**:
- `DatabaseService` uses `PersistentDatabasePersistence` as storage backend
- All database CRUD operations transparently persisted to disk
- Schema metadata stored in separate configuration files

**VectorService Integration**:
- Vector operations routed through `MemoryMappedVectorStore`
- Batch operations optimized for memory-mapped access
- Metadata stored alongside vector data in data section

**IndexService Integration**:
- Indexes built over persistent vector data
- Index metadata stored separately (not memory-mapped)
- Supports incremental index updates

**ReplicationService Integration**:
- Replication syncs `.jvdb` files across cluster nodes
- Delta-based replication for incremental updates
- Crash recovery ensures replica consistency

##### Best Practices

**Production Deployment**:
1. Use SSD storage for optimal performance
2. Configure `max_open_files` based on system limits (`ulimit -n`)
3. Set `flush_interval` based on durability requirements (5-15 minutes typical)
4. Monitor disk space usage (vectors can grow large)
5. Enable automatic backups of `.jvdb` files

**Performance Tuning**:
- Increase `max_open_files` for workloads with many hot databases
- Decrease `flush_interval` for stricter durability (trades performance)
- Use batch operations for bulk insertions
- Pre-allocate databases with known capacity to reduce file growth overhead

**Capacity Planning**:
- Storage per vector: `(dimension × 4 bytes) + metadata_size + 32B index entry`
- Example: 1M vectors × 512 dims × 4B = 2GB + index (32MB) + metadata
- Plan for 10-20% overhead for alignment and reserved space

### Data Architecture (continued)

#### Storage Format
- **Memory-Mapped Files**: For efficient large dataset handling
- **Custom Binary Format**: Optimized for vector operations
- **Apache Arrow**: For in-memory analytics with rich typing
- **FlatBuffers**: For network serialization

#### Indexing Architecture
The system supports multiple indexing algorithms optimized for different use cases:

- **FLAT Index**: Exact search, best for small datasets (< 10K vectors)
- **HNSW Index**: Graph-based, excellent balance of speed and accuracy
- **IVF Index**: Inverted file, good for very large datasets
- **LSH Index**: Hash-based, good for high-dimensional approximate search

### API Architecture

#### REST API
- **Framework**: Crow C++ web framework
- **Endpoints**: Follow RESTful conventions
- **Authentication**: API key-based
- **Rate Limiting**: Per-API key limits

#### gRPC API
- **Protocol**: gRPC for internal service communication
- **Performance**: Lower latency than REST for internal calls
- **Consistency**: Strong typing with Protocol Buffers

### Security Architecture

#### Authentication & Authorization
- **API Keys**: Primary authentication mechanism
- **Role-Based Access Control (RBAC)**: Fine-grained permissions
- **JWT Support**: For advanced authentication scenarios

#### Data Protection
- **Transport Security**: TLS/SSL for all communications
- **Data Encryption**: Optional at-rest encryption
- **Audit Logging**: Comprehensive logging of all operations

### Deployment Architecture

#### Microservices Design
- **Independent Scaling**: Each service can be scaled independently
- **Resilience**: Failure in one service doesn't necessarily affect others
- **Maintainability**: Clear separation of concerns

#### Containerized Deployment
- **Docker**: Containerized for consistent deployments
- **Kubernetes**: Orchestration for scaling and resilience
- **Docker Compose**: For local development and testing

### Performance Architecture

#### Caching Layer
- **Vector Cache**: Frequently accessed vectors cached in memory
- **Index Cache**: Active index structures cached for fast access
- **Query Result Cache**: Recent query results cached when appropriate

#### Threading Model
- **Thread Pool**: Efficient task distribution
- **SIMD Operations**: Vectorized operations for performance
- **Lock-Free Queues**: For internal message passing

#### Memory Management
- **Memory Pools**: Pre-allocated memory pools for performance
- **SIMD-Aligned Allocations**: Optimized memory layout for vector operations
- **Memory-Mapped Files**: Efficient large dataset handling

### Monitoring & Observability

#### Metrics Collection
- **Prometheus Format**: For integration with Prometheus ecosystem
- **Key Metrics**:
  - Query latency percentiles (p50, p95, p99)
  - Request rates
  - Error rates
  - Resource utilization
  - Index build progress

#### Logging
- **Structured Logging**: JSON format for easy parsing
- **Log Levels**: Support for different verbosity levels
- **Audit Logs**: Security-focused logging of access and modifications

#### Health Checks
- **Service Health**: Individual service health status
- **Database Health**: Per-database status monitoring
- **Cluster Health**: Distributed cluster monitoring

### Distributed Architecture

#### DistributedServiceManager
The `DistributedServiceManager` coordinates all distributed components:
- Lifecycle management of distributed services
- Service initialization and configuration
- Health monitoring and status reporting
- Integration with security audit logging and performance benchmarking

#### Cluster Management (ClusterService)
- **Raft Consensus**: For leader election and distributed consensus
- **Member List**: Automatic discovery and failure detection
- **Gossip Protocol**: For cluster state propagation
- **Node Health Monitoring**: Continuous health checks and automatic failover
- **Cluster Membership**: Dynamic node addition and removal

#### Data Distribution
- **Sharding Strategies** (ShardingService): Hash-based, range-based, and vector-based sharding
- **Replication** (ReplicationService): Configurable replication factors for durability with master-slave replication
- **Query Routing** (QueryRouter): Intelligent request distribution across shards and replicas
- **Load Balancing**: Automatic request distribution across nodes

### Configuration Management

#### Runtime Configuration
- **Configuration Service**: Centralized configuration management
- **Dynamic Updates**: Configuration changes without service restart
- **Validation**: Configuration validation before application

## Technology Stack

### Core Technologies
- **Language**: C++20
- **Build System**: CMake
- **Package Management**: C++ package managers or manual dependency management

### Libraries & Frameworks
- **Eigen**: Linear algebra operations
- **OpenBLAS/BLIS**: Optimized BLAS operations
- **FlatBuffers**: Serialization
- **Apache Arrow**: In-memory analytics
- **gRPC**: Service communication
- **Crow**: REST API framework
- **Google Test**: Testing framework

### Deployment & Infrastructure
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Prometheus**: Metrics collection
- **Grafana**: Dashboard and visualization

## Scalability Considerations

### Horizontal Scaling
- **Database Sharding**: Distribute data across multiple nodes
- **Service Scaling**: Independent scaling of each service
- **Index Distribution**: Distribute index computation across nodes

### Vertical Scaling
- **SIMD Optimization**: Leverage CPU vector instructions
- **Memory Optimization**: Efficient memory usage patterns
- **Thread Pool Tuning**: Optimize for available CPU cores

## Performance Benchmarks

### Target Performance Goals
- **Response Time**: < 100ms for similarity search on datasets up to 10M vectors
- **Throughput**: 10,000+ queries per second
- **Vector Ingestion**: 100,000+ vectors per second
- **Memory Usage**: < 2 bytes per dimension for in-memory operations

### Performance Validation
- **Benchmark Suite**: Comprehensive performance test suite
- **Continuous Monitoring**: Performance validation in CI/CD pipeline
- **Regression Detection**: Automated detection of performance regressions