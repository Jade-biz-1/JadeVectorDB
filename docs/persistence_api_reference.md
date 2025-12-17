# JadeVectorDB Persistence Layer API Reference

## Overview

This document provides detailed API reference for the persistent storage components introduced in Sprint 2.1. The persistence layer enables durable storage of vector embeddings using memory-mapped files with efficient LRU caching and automatic flushing.

## Core Classes

### MemoryMappedVectorStore

Memory-mapped persistent storage for vector embeddings with zero-copy access and SIMD alignment.

#### Constructor

```cpp
MemoryMappedVectorStore(
    const std::string& file_path,
    size_t dimension,
    size_t initial_capacity = 1000
);
```

**Parameters**:
- `file_path`: Path to the `.jvdb` binary file (created if doesn't exist)
- `dimension`: Vector dimensionality (must match schema)
- `initial_capacity`: Initial vector capacity (grows automatically)

**Throws**:
- `std::runtime_error`: File creation/mapping failure
- `std::invalid_argument`: Invalid dimension or capacity

**Example**:
```cpp
// Create or open persistent store for 512-dimensional vectors
auto store = std::make_shared<MemoryMappedVectorStore>(
    "/data/databases/my_database.jvdb",
    512,        // dimension
    10000       // initial capacity
);
```

#### store_vector

```cpp
bool store_vector(
    int64_t vector_id,
    const std::vector<float>& data,
    const std::string& metadata = ""
);
```

**Purpose**: Store a new vector or update an existing vector.

**Parameters**:
- `vector_id`: Unique vector identifier (must be non-negative)
- `data`: Vector components (size must equal dimension)
- `metadata`: Optional JSON metadata string

**Returns**: `true` on success, `false` on failure

**Thread-Safety**: Thread-safe with internal mutex

**Performance**: ~10-50µs per vector (excludes flush)

**Example**:
```cpp
std::vector<float> embedding(512, 0.5f);
std::string metadata = R"({"label": "cat", "confidence": 0.95})";

if (store->store_vector(12345, embedding, metadata)) {
    std::cout << "Vector stored successfully\n";
}
```

#### retrieve_vector

```cpp
std::optional<std::pair<std::vector<float>, std::string>> 
retrieve_vector(int64_t vector_id) const;
```

**Purpose**: Retrieve a vector and its metadata by ID.

**Parameters**:
- `vector_id`: Vector identifier to retrieve

**Returns**: 
- `std::optional` containing `{vector_data, metadata}` if found
- `std::nullopt` if vector doesn't exist

**Thread-Safety**: Thread-safe (read-only operation)

**Performance**: <1µs (zero-copy memory-mapped access)

**Example**:
```cpp
auto result = store->retrieve_vector(12345);
if (result) {
    auto [data, metadata] = *result;
    std::cout << "Vector dimension: " << data.size() << "\n";
    std::cout << "Metadata: " << metadata << "\n";
}
```

#### delete_vector

```cpp
bool delete_vector(int64_t vector_id);
```

**Purpose**: Delete a vector from storage (marks as deleted, reclaimed on compaction).

**Parameters**:
- `vector_id`: Vector identifier to delete

**Returns**: `true` if vector existed and was deleted, `false` if not found

**Thread-Safety**: Thread-safe with internal mutex

**Performance**: ~5-20µs (marks index entry as deleted)

**Note**: Physical space not reclaimed until compaction (future feature)

**Example**:
```cpp
if (store->delete_vector(12345)) {
    std::cout << "Vector deleted\n";
}
```

#### batch_store

```cpp
bool batch_store(
    const std::vector<int64_t>& vector_ids,
    const std::vector<std::vector<float>>& vectors,
    const std::vector<std::string>& metadata_list
);
```

**Purpose**: Store multiple vectors in a single operation for improved performance.

**Parameters**:
- `vector_ids`: Vector identifiers (must have same size as vectors)
- `vectors`: Vector data (each must have same dimension)
- `metadata_list`: Metadata strings (must have same size as vectors)

**Returns**: `true` if all vectors stored successfully, `false` on any failure

**Thread-Safety**: Thread-safe with internal mutex

**Performance**: 50,000-100,000 vectors/sec

**Example**:
```cpp
std::vector<int64_t> ids = {1, 2, 3, 4, 5};
std::vector<std::vector<float>> vectors(5, std::vector<float>(512, 0.5f));
std::vector<std::string> metadata(5, R"({"batch": true})");

if (store->batch_store(ids, vectors, metadata)) {
    std::cout << "Batch stored 5 vectors\n";
}
```

#### flush

```cpp
bool flush();
```

**Purpose**: Synchronize in-memory state to disk, ensuring durability.

**Returns**: `true` on successful flush, `false` on failure

**Thread-Safety**: Thread-safe

**Performance**: 1-10ms per database (depends on OS page cache)

**Note**: Automatically called on destruction and by VectorFlushManager

**Example**:
```cpp
// Critical checkpoint - ensure data is durable
if (store->flush()) {
    std::cout << "Data flushed to disk\n";
}
```

#### get_vector_count / get_dimension / get_capacity

```cpp
size_t get_vector_count() const;  // Current number of vectors
size_t get_dimension() const;     // Vector dimensionality
size_t get_capacity() const;      // Current file capacity
```

**Purpose**: Query store statistics.

**Thread-Safety**: Thread-safe (read-only)

**Example**:
```cpp
std::cout << "Store contains " << store->get_vector_count() 
          << " vectors of dimension " << store->get_dimension() << "\n";
```

---

### PersistentDatabasePersistence

High-level database persistence manager with LRU eviction and automatic flushing.

#### Constructor

```cpp
PersistentDatabasePersistence(
    const std::string& storage_path,
    size_t max_open_files = 100,
    std::chrono::seconds flush_interval = std::chrono::seconds(300)
);
```

**Parameters**:
- `storage_path`: Root directory for `.jvdb` files (created if doesn't exist)
- `max_open_files`: Maximum concurrent open memory-mapped files (LRU eviction)
- `flush_interval`: Automatic flush interval (0 disables auto-flush)

**Throws**:
- `std::runtime_error`: Storage path creation failure

**Example**:
```cpp
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/var/lib/jadevectordb/data",  // storage path
    100,                             // max 100 open files
    std::chrono::seconds(300)        // flush every 5 minutes
);
```

#### create_database

```cpp
bool create_database(
    const std::string& database_name,
    const DatabaseSchema& schema
);
```

**Purpose**: Create a new database with schema.

**Parameters**:
- `database_name`: Unique database identifier
- `schema`: Schema including dimension, distance metric

**Returns**: `true` on success, `false` if database already exists

**Thread-Safety**: Thread-safe

**Side-Effects**: Creates `{storage_path}/{database_name}.jvdb`

**Example**:
```cpp
DatabaseSchema schema;
schema.dimension = 512;
schema.distance_metric = DistanceMetric::COSINE;
schema.index_type = IndexType::HNSW;

if (persistence->create_database("embeddings_512d", schema)) {
    std::cout << "Database created\n";
}
```

#### delete_database

```cpp
bool delete_database(const std::string& database_name);
```

**Purpose**: Delete a database and remove its `.jvdb` file.

**Parameters**:
- `database_name`: Database identifier to delete

**Returns**: `true` on success, `false` if database doesn't exist

**Thread-Safety**: Thread-safe

**Side-Effects**: Deletes `{storage_path}/{database_name}.jvdb`

**Example**:
```cpp
if (persistence->delete_database("old_database")) {
    std::cout << "Database deleted\n";
}
```

#### store_vector / retrieve_vector / delete_vector

```cpp
bool store_vector(
    const std::string& database_name,
    int64_t vector_id,
    const std::vector<float>& data,
    const std::string& metadata = ""
);

std::optional<std::pair<std::vector<float>, std::string>> 
retrieve_vector(
    const std::string& database_name,
    int64_t vector_id
);

bool delete_vector(
    const std::string& database_name,
    int64_t vector_id
);
```

**Purpose**: Database-scoped vector operations (delegates to MemoryMappedVectorStore).

**Thread-Safety**: Thread-safe

**LRU Behavior**: Automatically opens/closes stores based on `max_open_files` limit

**Example**:
```cpp
// Store vector in database "embeddings_512d"
std::vector<float> vec(512, 0.5f);
persistence->store_vector("embeddings_512d", 1, vec, R"({"tag": "test"})");

// Retrieve vector
auto result = persistence->retrieve_vector("embeddings_512d", 1);

// Delete vector
persistence->delete_vector("embeddings_512d", 1);
```

#### list_databases

```cpp
std::vector<std::string> list_databases() const;
```

**Purpose**: Get list of all database names.

**Returns**: Vector of database names (alphabetically sorted)

**Thread-Safety**: Thread-safe

**Example**:
```cpp
auto databases = persistence->list_databases();
for (const auto& db : databases) {
    std::cout << "Database: " << db << "\n";
}
```

#### flush / flush_database

```cpp
void flush();                                  // Flush all databases
bool flush_database(const std::string& name);  // Flush specific database
```

**Purpose**: Manually trigger flush for durability checkpoints.

**Thread-Safety**: Thread-safe

**Performance**: O(n) where n = number of open databases

**Example**:
```cpp
// Flush specific database
persistence->flush_database("critical_data");

// Flush all databases (e.g., before backup)
persistence->flush();
```

#### get_storage_stats

```cpp
struct StorageStats {
    size_t total_databases;
    size_t total_vectors;
    size_t total_bytes;
    size_t open_files;
};

StorageStats get_storage_stats() const;
```

**Purpose**: Query storage statistics for monitoring.

**Thread-Safety**: Thread-safe

**Example**:
```cpp
auto stats = persistence->get_storage_stats();
std::cout << "Total databases: " << stats.total_databases << "\n"
          << "Total vectors: " << stats.total_vectors << "\n"
          << "Storage size: " << stats.total_bytes / (1024*1024) << " MB\n"
          << "Open files: " << stats.open_files << "\n";
```

---

### VectorFlushManager

Global flush coordinator for periodic and manual flush operations.

#### get_instance

```cpp
static VectorFlushManager& get_instance();
```

**Purpose**: Get singleton instance (thread-safe).

**Example**:
```cpp
auto& flush_mgr = VectorFlushManager::get_instance();
```

#### register_persistence / unregister_persistence

```cpp
void register_persistence(
    const std::string& name,
    std::shared_ptr<PersistentDatabasePersistence> persistence
);

void unregister_persistence(const std::string& name);
```

**Purpose**: Register persistence instances for coordinated flushing.

**Thread-Safety**: Thread-safe

**Example**:
```cpp
auto persistence = std::make_shared<PersistentDatabasePersistence>("/data");
flush_mgr.register_persistence("main_storage", persistence);

// ... later ...
flush_mgr.unregister_persistence("main_storage");
```

#### start_periodic_flush / stop_periodic_flush

```cpp
void start_periodic_flush(std::chrono::seconds interval);
void stop_periodic_flush();
```

**Purpose**: Start/stop background flush thread.

**Parameters**:
- `interval`: Time between automatic flushes

**Thread-Safety**: Thread-safe

**Example**:
```cpp
// Start auto-flush every 5 minutes
flush_mgr.start_periodic_flush(std::chrono::seconds(300));

// Stop auto-flush (e.g., during shutdown)
flush_mgr.stop_periodic_flush();
```

#### flush_all

```cpp
void flush_all();
```

**Purpose**: Manually flush all registered persistence instances.

**Thread-Safety**: Thread-safe

**Performance**: O(n × m) where n = persistence instances, m = databases per instance

**Example**:
```cpp
// Explicit flush for critical checkpoint
flush_mgr.flush_all();
std::cout << "All data flushed to disk\n";
```

---

## Configuration Examples

### Basic Configuration

```cpp
// Create persistence layer with default settings
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/var/lib/jadevectordb",  // storage path
    100,                       // max 100 open files
    std::chrono::seconds(300)  // flush every 5 minutes
);

// Register with flush manager
auto& flush_mgr = VectorFlushManager::get_instance();
flush_mgr.register_persistence("main", persistence);
flush_mgr.start_periodic_flush(std::chrono::seconds(300));
```

### High-Throughput Configuration

```cpp
// Optimized for high write throughput
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/mnt/nvme/jadevectordb",     // Fast SSD storage
    500,                           // Higher open file limit
    std::chrono::seconds(600)      // Less frequent flushes (10 min)
);
```

### High-Durability Configuration

```cpp
// Optimized for maximum durability
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/data/jadevectordb",
    50,                            // Conservative file limit
    std::chrono::seconds(60)       // Frequent flushes (1 min)
);
```

### Memory-Constrained Configuration

```cpp
// Optimized for limited file descriptors
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/data/jadevectordb",
    20,                            // Low open file limit
    std::chrono::seconds(300)
);
```

---

## Error Handling

### Common Error Scenarios

#### File System Errors

```cpp
try {
    auto store = std::make_shared<MemoryMappedVectorStore>(
        "/invalid/path/db.jvdb", 512, 1000
    );
} catch (const std::runtime_error& e) {
    // Handle: Permission denied, disk full, invalid path
    std::cerr << "Failed to create store: " << e.what() << "\n";
}
```

#### Dimension Mismatch

```cpp
auto store = std::make_shared<MemoryMappedVectorStore>("db.jvdb", 512, 1000);

std::vector<float> wrong_dim(128);  // Wrong dimension!
if (!store->store_vector(1, wrong_dim)) {
    // Validation failed - dimension mismatch
    std::cerr << "Dimension mismatch\n";
}
```

#### Corrupted File Detection

```cpp
// Automatic corruption detection on open
try {
    auto store = std::make_shared<MemoryMappedVectorStore>(
        "corrupted.jvdb", 512, 1000
    );
} catch (const std::runtime_error& e) {
    // Invalid magic number or header corruption detected
    std::cerr << "Corrupted file: " << e.what() << "\n";
    // Recovery: Delete file and recreate
}
```

#### LRU Eviction Handling

```cpp
// LRU eviction is transparent - no error handling needed
auto persistence = std::make_shared<PersistentDatabasePersistence>("/data", 10);

// Create 20 databases (exceeds max_open_files=10)
for (int i = 0; i < 20; i++) {
    DatabaseSchema schema{512, DistanceMetric::COSINE};
    persistence->create_database("db_" + std::to_string(i), schema);
}

// Access all databases - LRU automatically handles opening/closing
for (int i = 0; i < 20; i++) {
    std::vector<float> vec(512, 0.5f);
    persistence->store_vector("db_" + std::to_string(i), 1, vec);
    // Oldest stores automatically closed when limit reached
}
```

---

## Performance Tuning

### Batch Operations

```cpp
// SLOW: Individual stores (10µs × 10,000 = 100ms)
for (int i = 0; i < 10000; i++) {
    store->store_vector(i, vec, metadata);
}

// FAST: Batch store (10,000 vectors in ~100ms)
std::vector<int64_t> ids(10000);
std::vector<std::vector<float>> vectors(10000, vec);
std::vector<std::string> metadata_list(10000, metadata);
std::iota(ids.begin(), ids.end(), 0);
store->batch_store(ids, vectors, metadata_list);
```

### Flush Strategy

```cpp
// Strategy 1: High throughput (infrequent flushes)
persistence->start_periodic_flush(std::chrono::seconds(600));  // 10 min

// Strategy 2: Balanced (moderate flushes)
persistence->start_periodic_flush(std::chrono::seconds(300));  // 5 min

// Strategy 3: High durability (frequent flushes)
persistence->start_periodic_flush(std::chrono::seconds(60));   // 1 min

// Strategy 4: Critical operations (explicit flush)
store->batch_store(ids, vectors, metadata_list);
store->flush();  // Ensure durability immediately
```

### LRU Tuning

```cpp
// Check system file descriptor limit
// $ ulimit -n
// 1024

// Set max_open_files to 50-80% of system limit
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/data",
    800,  // 80% of 1024 limit
    std::chrono::seconds(300)
);
```

---

## Signal Handling

Graceful shutdown with signal handlers ensures data durability:

```cpp
#include "SignalHandler.hpp"

// Initialize signal handlers for SIGTERM, SIGINT
SignalHandler::initialize();

// Register cleanup callback
SignalHandler::register_shutdown_callback([]() {
    std::cout << "Shutting down gracefully...\n";
    
    // Flush all data
    VectorFlushManager::get_instance().flush_all();
    
    // Stop periodic flushing
    VectorFlushManager::get_instance().stop_periodic_flush();
    
    std::cout << "Shutdown complete\n";
});

// Normal operation
// ...

// On SIGTERM/SIGINT: callbacks invoked, data flushed, clean exit
```

---

## Monitoring and Observability

### Storage Statistics

```cpp
auto stats = persistence->get_storage_stats();

// Log metrics
log_metric("jadevectordb.storage.databases", stats.total_databases);
log_metric("jadevectordb.storage.vectors", stats.total_vectors);
log_metric("jadevectordb.storage.bytes", stats.total_bytes);
log_metric("jadevectordb.storage.open_files", stats.open_files);

// Alert on approaching file limit
if (stats.open_files > 0.8 * max_open_files) {
    log_warning("Approaching max_open_files limit");
}
```

### Performance Monitoring

```cpp
// Measure flush latency
auto start = std::chrono::high_resolution_clock::now();
persistence->flush();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::high_resolution_clock::now() - start
).count();

log_metric("jadevectordb.flush.latency_ms", duration);
```

### Health Checks

```cpp
bool health_check() {
    try {
        // Verify storage is accessible
        auto databases = persistence->list_databases();
        
        // Verify flush works
        persistence->flush();
        
        return true;
    } catch (const std::exception& e) {
        log_error("Health check failed: ", e.what());
        return false;
    }
}
```

---

## Thread Safety Guarantees

All persistence layer APIs are **fully thread-safe**:

- **MemoryMappedVectorStore**: Internal mutex protects all mutations
- **PersistentDatabasePersistence**: Thread-safe LRU and store management
- **VectorFlushManager**: Thread-safe registration and flush coordination

Concurrent access patterns:

```cpp
// Safe: Multiple threads reading/writing different vectors
std::vector<std::thread> writers;
for (int i = 0; i < 8; i++) {
    writers.emplace_back([&store, i]() {
        for (int j = 0; j < 1000; j++) {
            std::vector<float> vec(512, i * 1000 + j);
            store->store_vector(i * 1000 + j, vec);
        }
    });
}
for (auto& t : writers) t.join();

// Safe: Concurrent read/write to same store
std::thread reader([&store]() {
    for (int i = 0; i < 8000; i++) {
        auto result = store->retrieve_vector(i);
    }
});
reader.join();
```

---

## Migration from InMemoryDatabasePersistence

See [Migration Guide](migration_guide_persistent_storage.md) for step-by-step instructions.

---

## See Also

- [Architecture Documentation](architecture.md) - System architecture overview
- [Migration Guide](migration_guide_persistent_storage.md) - Migration from in-memory storage
- [Performance Tuning Guide](performance_tuning.md) - Advanced optimization techniques
- [Distributed Deployment Guide](distributed_deployment_guide.md) - Cluster configuration
