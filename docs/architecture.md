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

**Index Resize Mechanism** (Sprint 2.3):
- Automatic capacity growth when 75% full
- Doubles index capacity (e.g., 1024 → 2048 entries)
- Full rehashing of all active entries to new index
- **Critical Design Decision**: Save old offset values before unmapping
  - When file is resized, memory is unmapped and remapped
  - After remap, header pointer refers to same location but values may be stale
  - Solution: Save `old_data_offset` and `old_vector_ids_offset` before unmapping
  - During rehash, update both `data_offset` and `string_offset` using saved values
  - Formula: `new_offset = new_base + (old_offset - old_base)`
  - This preserves relative positioning within relocated sections
- **Data Integrity**: Bug fixed (December 19, 2025) - all data preserved during resize
- **Thread Safety**: Protected by per-database mutexes
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

## Advanced Search Features

### Hybrid Search Architecture

JadeVectorDB implements hybrid search by combining vector similarity search with BM25 keyword-based search for improved retrieval accuracy.

**Key Components**:
- **BM25 Scorer**: Implements BM25 algorithm for keyword relevance scoring
- **Inverted Index**: Fast keyword lookup with posting lists
- **Score Fusion**: Combines vector and BM25 scores using RRF or weighted linear fusion
- **Index Persistence**: SQLite-based storage for BM25 index durability

**Fusion Methods**:
1. **Reciprocal Rank Fusion (RRF)**: Rank-based fusion, no normalization needed
2. **Weighted Linear Fusion**: Configurable alpha parameter for score combination

**Use Cases**:
- Exact match requirements (product codes, model numbers)
- Technical documentation search
- E-commerce product search
- RAG systems requiring keyword precision

### Re-ranking Architecture

Re-ranking improves search precision by re-scoring top candidates using cross-encoder models. JadeVectorDB supports multiple deployment architectures depending on scale and requirements.

#### Architecture Options

##### Option 1: Python Subprocess (Single-Node Deployments)
```
┌─────────────────────────────────────┐
│   JadeVectorDB Single Node          │
│                                     │
│  ┌──────────────────────────────┐  │
│  │ C++ Main Process             │  │
│  │  - Search Pipeline           │  │
│  │  - RerankingService          │  │
│  └──────────┬───────────────────┘  │
│             │ stdin/stdout          │
│             │ JSON IPC              │
│  ┌──────────▼───────────────────┐  │
│  │ Python Subprocess            │  │
│  │  - sentence-transformers     │  │
│  │  - cross-encoder model       │  │
│  │  - Model: ms-marco-MiniLM    │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Characteristics**:
- **Deployment**: Simple, single process launch
- **Memory**: ~200-500MB per Python subprocess
- **Latency**: ~150-300ms for 100 documents
- **Best For**: Development, small-scale deployments, single-node production

**Pros**:
- Simple implementation and deployment
- No additional services to manage
- Fast development iteration
- Good for proof-of-concept

**Cons**:
- Not optimal for distributed clusters (each worker needs own subprocess)
- No GPU sharing across nodes
- Process management complexity in distributed mode
- Subprocess restart overhead on failures

##### Option 2: Dedicated Re-ranking Service (Distributed Deployments - Recommended)
```
┌──────────────────────────────────────────────────────┐
│            JadeVectorDB Cluster                      │
├──────────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌──────────────┐            │
│  │ Master Node  │      │ Worker Node 1│            │
│  └──────┬───────┘      └──────┬───────┘            │
│         │                     │                     │
│  ┌──────▼───────┐      ┌──────▼───────┐            │
│  │ Worker Node 2│      │ Worker Node 3│            │
│  └──────┬───────┘      └──────┬───────┘            │
│         │                     │                     │
│         └──────────┬──────────┘                     │
│                    │ gRPC                           │
│         ┌──────────▼──────────────────┐            │
│         │  Re-ranking Service         │            │
│         │  (Microservice)             │            │
│         │  ┌──────────────────────┐  │            │
│         │  │ Model Inference Pool │  │            │
│         │  │ - GPU Support        │  │            │
│         │  │ - Batch Processing   │  │            │
│         │  │ - Model Caching      │  │            │
│         │  └──────────────────────┘  │            │
│         └─────────────────────────────┘            │
└──────────────────────────────────────────────────────┘
```

**Characteristics**:
- **Deployment**: Independent microservice, scales separately
- **Memory**: Shared across cluster (single model load ~500MB)
- **Latency**: ~100-200ms for 100 documents (optimized batch processing)
- **Best For**: Production distributed clusters, high-throughput systems

**Pros**:
- **Resource Efficiency**: Single model instance serves entire cluster
- **GPU Support**: Centralized GPU utilization
- **Independent Scaling**: Scale re-ranker independently from DB nodes
- **High Availability**: Multiple re-ranker instances for redundancy
- **Consistent Performance**: Batch optimization, connection pooling
- **Microservices Pattern**: Aligns with existing distributed architecture

**Cons**:
- Additional service to deploy and manage
- Network latency between nodes and re-ranker
- More complex deployment configuration

##### Option 3: ONNX Runtime (C++ Native - Future)
```
┌─────────────────────────────────────┐
│   JadeVectorDB Node                 │
│                                     │
│  ┌──────────────────────────────┐  │
│  │ C++ Main Process             │  │
│  │  - Search Pipeline           │  │
│  │  - ONNXRerankingService      │  │
│  │  - ONNX Runtime (C++)        │  │
│  │  - Cross-encoder.onnx        │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Characteristics**:
- **Deployment**: Native C++ library, no subprocess/service
- **Memory**: ~300-400MB (ONNX model + runtime)
- **Latency**: ~50-100ms for 100 documents (native performance)
- **Best For**: Maximum performance, simplified deployment

**Pros**:
- **Best Performance**: Native C++ execution, no IPC overhead
- **Simplified Deployment**: Single binary, no Python dependencies
- **Memory Efficiency**: No subprocess overhead
- **GPU Support**: ONNX Runtime supports CUDA/TensorRT

**Cons**:
- **Model Conversion**: Requires converting PyTorch models to ONNX
- **Limited Model Support**: Not all models convert cleanly
- **Implementation Complexity**: More code than subprocess approach
- **Debugging Difficulty**: Harder to debug than Python

#### Recommended Implementation Strategy

**Phase 1 (Current - T16.9 to T16.14)**: Python Subprocess
- Implement for rapid development and validation
- Prove quality improvement metrics (+15% precision@5)
- Focus on single-node and small cluster deployments
- Document subprocess management patterns

**Phase 2 (Future Enhancement)**: Dedicated Re-ranking Service
- Design and implement gRPC-based re-ranking microservice
- Support GPU acceleration
- Implement connection pooling and batch optimization
- Deploy alongside existing distributed services

**Phase 3 (Future Optimization)**: ONNX Runtime Support
- Add ONNX as alternative inference backend
- Benchmark performance vs. subprocess approach
- Provide configuration option for deployment flexibility

#### Subprocess Management Implementation

The Python subprocess approach (Phase 1) uses a sophisticated process management layer to ensure reliable communication and fault tolerance.

**SubprocessManager Architecture**:

```cpp
┌─────────────────────────────────────────────────────┐
│          SubprocessManager                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Process Lifecycle Management                │  │
│  │  - fork() + exec() subprocess spawning       │  │
│  │  - Pipe setup (stdin/stdout/stderr)          │  │
│  │  - Signal handling (SIGTERM, SIGPIPE)        │  │
│  │  - Graceful shutdown with timeout            │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  IPC Communication Layer                     │  │
│  │  - Line-based JSON protocol                  │  │
│  │  - Non-blocking I/O with poll()              │  │
│  │  - Request/response correlation              │  │
│  │  - Timeout support (configurable)            │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Health Monitoring Thread                    │  │
│  │  - Periodic heartbeat (30s interval)         │  │
│  │  - Subprocess health detection               │  │
│  │  - Auto-restart on failure                   │  │
│  │  - Circuit breaker pattern                   │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Thread Safety                               │  │
│  │  - Mutex-protected request queue             │  │
│  │  - Thread-safe status tracking               │  │
│  │  - Atomic flags for state management         │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**Communication Protocol**:

The C++ process and Python subprocess communicate via a line-based JSON protocol over stdin/stdout pipes:

```
C++ Process                          Python Subprocess
    │                                        │
    │  1. Launch subprocess                  │
    │────────────────────────────────────────▶
    │                                        │
    │                    {"status": "ready"} │
    │◀────────────────────────────────────────
    │                                        │
    │  2. Send rerank request                │
    │  {"query": "...", "documents": [...]}  │
    │────────────────────────────────────────▶
    │                                        │
    │       {"scores": [...], "latency": 145}│
    │◀────────────────────────────────────────
    │                                        │
    │  3. Heartbeat ping (every 30s)         │
    │  {"command": "ping"}                   │
    │────────────────────────────────────────▶
    │                                        │
    │                    {"status": "pong"}  │
    │◀────────────────────────────────────────
    │                                        │
    │  4. Shutdown command                   │
    │  {"command": "shutdown"}               │
    │────────────────────────────────────────▶
    │                                        │
    │         {"status": "shutting_down"}    │
    │◀────────────────────────────────────────
    │                                        │
```

**Key Implementation Details**:

1. **Process Spawning**:
   - Use `fork()` + `exec()` pattern for clean process creation
   - Setup pipes before exec: stdin (write), stdout (read), stderr (read)
   - Close unused pipe ends in both parent and child
   - Set non-blocking mode on pipes using `fcntl()`

2. **JSON Communication**:
   - Write: `json_string + "\n"` + `flush()`
   - Read: Buffered line reading with `poll()` for timeout support
   - Error handling: Parse stderr on failure for Python tracebacks

3. **Timeout Handling**:
   - Startup timeout: 10 seconds (model loading can be slow)
   - Request timeout: 5 seconds (configurable based on batch size)
   - Heartbeat interval: 30 seconds (detect zombie processes)

4. **Error Recovery**:
   - Subprocess crash: Detected via heartbeat, auto-restart once
   - Communication timeout: Kill request, return error with partial results
   - Python exception: Parse stderr, return descriptive error message
   - OOM error: Detect via stderr patterns, return resource exhaustion error

5. **Thread Safety**:
   - All public methods protected by mutex
   - Request queue for serializing concurrent requests
   - Atomic status flags for health monitoring thread

**Configuration Structure**:

```cpp
struct SubprocessConfig {
    std::string python_executable = "python3";
    std::string script_path = "python/reranking_server.py";
    std::string model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2";
    int batch_size = 32;
    std::chrono::milliseconds startup_timeout = 10000ms;
    std::chrono::milliseconds request_timeout = 5000ms;
    std::chrono::milliseconds heartbeat_interval = 30000ms;
    bool auto_restart = true;
    int max_restart_attempts = 3;
};
```

**RerankingService Layer**:

Built on top of SubprocessManager, provides high-level API:

```cpp
class RerankingService {
public:
    // Initialize subprocess and load model
    Result<void> initialize();

    // Rerank search results with score combination
    Result<std::vector<RerankingResult>> rerank(
        const std::string& query,
        const std::vector<HybridSearchResult>& results
    );

    // Direct text reranking (standalone)
    Result<std::vector<RerankingResult>> rerank_batch(
        const std::string& query,
        const std::vector<std::string>& doc_ids,
        const std::vector<std::string>& documents,
        const std::vector<double>& original_scores
    );

    // Health check
    bool is_ready() const;

    // Statistics
    Statistics get_statistics() const;

private:
    std::unique_ptr<SubprocessManager> subprocess_;
    RerankingConfig config_;
    std::mutex mutex_;
    Statistics stats_;
};
```

**Score Combination Logic**:

```cpp
// Weighted combination of rerank score and original hybrid score
combined_score = rerank_weight * rerank_score +
                 (1.0 - rerank_weight) * original_score;

// Default: 70% rerank, 30% hybrid
// Rationale: Cross-encoder is more accurate but trust original search context
```

#### Integration with Search Pipeline

**Two-Stage Retrieval Architecture**:
```
Query Input
    │
    ▼
┌────────────────────────┐
│ Stage 1: Retrieval     │
│  - Vector Search       │
│  - OR Hybrid Search    │
│  - Retrieve Top-100    │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ Stage 2: Re-ranking    │ (Optional)
│  - Cross-encoder       │
│  - Score Top-100       │
│  - Return Top-K        │
└──────────┬─────────────┘
           │
           ▼
      Final Results
```

**HybridSearchEngine Integration**:

```cpp
std::vector<HybridSearchResult> HybridSearchEngine::search(
    const std::string& query_text,
    const std::vector<float>& query_vector,
    size_t top_k,
    bool enable_reranking = false,
    size_t rerank_top_n = 100
) {
    // Stage 1: Hybrid search (vector + BM25)
    auto candidates = hybrid_search_internal(
        query_text, query_vector, rerank_top_n
    );

    // Stage 2: Optional re-ranking
    if (enable_reranking && reranking_provider_) {
        candidates = reranking_provider_(query_text, candidates);
    }

    // Return top-K results
    candidates.resize(std::min(candidates.size(), top_k));
    return candidates;
}
```

**Dependency Injection Pattern**:

The HybridSearchEngine uses dependency injection for the reranking provider, allowing flexible implementation:

```cpp
// In REST API handler
hybrid_engine->set_reranking_provider(
    [&reranking_service](const std::string& query,
                        const std::vector<HybridSearchResult>& candidates)
    -> std::vector<HybridSearchResult> {
        // Convert to RerankingService format
        std::vector<std::string> doc_ids, documents;
        std::vector<double> scores;

        for (const auto& c : candidates) {
            doc_ids.push_back(c.doc_id);
            documents.push_back(extract_text(c));
            scores.push_back(c.hybrid_score);
        }

        // Call reranking service
        auto rerank_result = reranking_service.rerank_batch(
            query, doc_ids, documents, scores
        );

        // Convert back to HybridSearchResult
        return convert_to_hybrid_results(rerank_result);
    }
);
```

**Configuration**:
- **enable_reranking**: Boolean flag to enable/disable re-ranking
- **rerank_top_n**: Number of candidates to re-rank (default: 100)
- **model**: Cross-encoder model selection
- **batch_size**: Batch size for inference (default: 32)
- **rerank_weight**: Score combination weight (default: 0.7)

#### Performance Characteristics

**Single Document Re-ranking Latency**:
- Python subprocess: ~2-3ms per document
- Dedicated service: ~1-2ms per document (batched)
- ONNX native: ~0.5-1ms per document

**Throughput** (100 documents):
- Python subprocess: ~150-300ms total
- Dedicated service: ~100-200ms total (optimized batching)
- ONNX native: ~50-100ms total

**Quality Improvement**:
- **Precision@5**: +15-20% improvement over bi-encoder alone
- **MRR (Mean Reciprocal Rank)**: +10-15% improvement
- **NDCG@10**: +8-12% improvement

**Resource Requirements**:
- **CPU**: 2-4 cores recommended for inference
- **Memory**: 500MB-1GB per instance (includes model)
- **GPU** (optional): 2GB VRAM (significant speedup)

#### Best Practices

**For Single-Node Deployments**:
1. Use Python subprocess approach (simpler)
2. Configure subprocess health monitoring
3. Implement auto-restart on failure
4. Set reasonable timeout values (5-10 seconds)

**For Distributed Deployments**:
1. Deploy dedicated re-ranking service
2. Use load balancer for multiple re-ranker instances
3. Enable connection pooling
4. Monitor re-ranker service health separately
5. Configure graceful degradation (fallback to no re-ranking)

**Model Selection**:
- **Development**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, good quality)
- **Production**: `cross-encoder/ms-marco-MiniLM-L-12-v2` (better quality, slower)
- **Domain-Specific**: Fine-tune on your domain data for best results

## Performance Benchmarks

### Target Performance Goals
- **Response Time**: < 100ms for similarity search on datasets up to 10M vectors
- **Throughput**: 10,000+ queries per second
- **Vector Ingestion**: 100,000+ vectors per second
- **Memory Usage**: < 2 bytes per dimension for in-memory operations
- **Hybrid Search**: < 30ms total (vector + BM25 + fusion) for 50K documents
- **Re-ranking**: < 200ms for 100 candidates (Python subprocess)
- **Re-ranking Quality**: +15% precision@5 improvement over bi-encoder alone

### Performance Validation
- **Benchmark Suite**: Comprehensive performance test suite including hybrid search and re-ranking
- **Continuous Monitoring**: Performance validation in CI/CD pipeline
- **Regression Detection**: Automated detection of performance regressions
- **Quality Metrics**: Precision@K, MRR, and NDCG tracking for search improvements