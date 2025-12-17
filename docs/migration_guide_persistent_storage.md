# Migration Guide: Persistent Vector Storage

## Overview

This guide provides step-by-step instructions for migrating from JadeVectorDB's in-memory storage (`InMemoryDatabasePersistence`) to the new persistent storage layer (`PersistentDatabasePersistence`) introduced in Sprint 2.1.

**Target Audience**: System administrators, DevOps engineers, and developers deploying JadeVectorDB in production.

**Estimated Migration Time**: 30 minutes - 2 hours (depending on data volume)

---

## Table of Contents

1. [Why Migrate?](#why-migrate)
2. [Prerequisites](#prerequisites)
3. [Understanding the Differences](#understanding-the-differences)
4. [Migration Strategies](#migration-strategies)
5. [Step-by-Step Migration](#step-by-step-migration)
6. [Configuration Reference](#configuration-reference)
7. [Validation and Testing](#validation-and-testing)
8. [Rollback Strategy](#rollback-strategy)
9. [Troubleshooting](#troubleshooting)
10. [Performance Tuning](#performance-tuning)

---

## Why Migrate?

### Benefits of Persistent Storage

**Durability**:
- Vectors survive process restarts and crashes
- No data loss on system reboot or ungraceful shutdown
- Automatic periodic flushing ensures recent data is saved

**Performance**:
- Memory-mapped files enable zero-copy access
- Reduced memory footprint with LRU eviction
- Faster startup times with lazy loading (10ms vs seconds for large datasets)

**Scalability**:
- Support for datasets larger than available RAM
- Efficient handling of hundreds of databases with limited file descriptors
- Predictable memory usage independent of data size

**Operational Benefits**:
- Simplified backup and restore (just copy `.jvdb` files)
- Easy database migration between servers
- Reduced memory pressure on host systems

---

## Prerequisites

### System Requirements

**Storage**:
- Available disk space: 2-3x your current memory usage for vector data
- SSD storage strongly recommended for optimal performance
- Filesystem with mmap support (ext4, XFS, NTFS, APFS)

**Operating System**:
- Linux kernel 2.6+ (mmap support)
- Windows 7+ (CreateFileMapping support)
- macOS 10.4+ (mmap support)

**File Descriptors**:
```bash
# Check current limit
ulimit -n

# Recommended: At least 1024 (default on most systems)
# For production: 4096 or higher
# Set persistent limit in /etc/security/limits.conf:
# * soft nofile 4096
# * hard nofile 8192
```

**Permissions**:
- Write access to storage directory (e.g., `/var/lib/jadevectordb`)
- Sufficient disk quota for expected data growth

### Software Requirements

- JadeVectorDB version 2.1.0 or higher
- C++20 compatible compiler (if building from source)
- Existing JadeVectorDB deployment with data to migrate

### Pre-Migration Checklist

- [ ] Backup existing data (export databases to JSON/binary format)
- [ ] Test migration in staging environment first
- [ ] Verify disk space availability
- [ ] Review and increase file descriptor limits if needed
- [ ] Schedule maintenance window (optional, for zero-downtime migration)
- [ ] Prepare rollback plan
- [ ] Document current configuration

---

## Understanding the Differences

### Architecture Comparison

#### InMemoryDatabasePersistence (Old)

```
┌─────────────────────────────┐
│  DatabaseService            │
├─────────────────────────────┤
│  InMemoryDatabasePersistence│
│  - std::map<db, vectors>    │  ← All data in RAM
│  - No durability            │
│  - Fast but volatile        │
└─────────────────────────────┘
```

**Characteristics**:
- All vectors stored in RAM (std::map)
- No persistence across restarts
- Memory usage grows linearly with data
- Instant startup (no load time)
- Simple implementation

#### PersistentDatabasePersistence (New)

```
┌─────────────────────────────────────┐
│  DatabaseService                    │
├─────────────────────────────────────┤
│  PersistentDatabasePersistence      │
│  - LRU cache of open stores         │  ← Smart caching
│  - MemoryMappedVectorStore per DB   │
├─────────────────────────────────────┤
│  Disk Storage (.jvdb files)         │
│  - Binary format                    │  ← Persistent
│  - Memory-mapped access             │
│  - SIMD-aligned data                │
└─────────────────────────────────────┘
```

**Characteristics**:
- Vectors stored in memory-mapped files on disk
- Full durability with automatic flushing
- Memory usage bounded by LRU cache
- Fast startup with lazy loading (header only)
- Zero-copy access to vector data

### API Compatibility

**Good News**: The public API is **fully compatible**! No code changes required for basic operations.

```cpp
// Both implementations support the same interface:
persistence->create_database("my_db", schema);
persistence->store_vector("my_db", 1, vec, metadata);
auto result = persistence->retrieve_vector("my_db", 1);
persistence->delete_database("my_db");
```

**New Features** (optional to use):
- Configurable LRU cache size (`max_open_files`)
- Manual flush operations (`flush()`, `flush_database()`)
- Storage statistics (`get_storage_stats()`)
- VectorFlushManager integration for coordinated flushing

---

## Migration Strategies

### Strategy 1: Clean Migration (Recommended for Small Datasets)

**Best For**: 
- Small datasets (< 1GB)
- Development/staging environments
- Non-critical data that can be re-ingested

**Steps**:
1. Stop JadeVectorDB
2. Export data (optional backup)
3. Update configuration to use PersistentDatabasePersistence
4. Start JadeVectorDB
5. Re-ingest data

**Downtime**: 5-30 minutes

### Strategy 2: Export/Import Migration (Recommended for Production)

**Best For**:
- Medium datasets (1-100GB)
- Production environments
- Critical data requiring validation

**Steps**:
1. Export all databases to intermediate format (JSON/binary)
2. Stop JadeVectorDB
3. Update configuration
4. Start JadeVectorDB with persistent storage
5. Import data from intermediate format
6. Validate data integrity

**Downtime**: 30 minutes - 2 hours

### Strategy 3: Zero-Downtime Migration (Advanced)

**Best For**:
- Large datasets (> 100GB)
- 24/7 production systems
- Distributed deployments

**Steps**:
1. Set up parallel JadeVectorDB instance with persistent storage
2. Configure replication from old to new instance
3. Wait for full synchronization
4. Switch traffic to new instance
5. Decommission old instance

**Downtime**: None (requires replication setup)

---

## Step-by-Step Migration

### Step 1: Backup Current Data

#### Option A: JSON Export (Recommended)

```bash
# Export all databases to JSON
mkdir -p /backup/jadevectordb/$(date +%Y%m%d)
cd /backup/jadevectordb/$(date +%Y%m%d)

# Use JadeVectorDB CLI to export
jadevectordb-cli export --format json --output ./databases.json

# Verify backup
ls -lh databases.json
```

#### Option B: Binary Snapshot

```bash
# If using in-memory persistence, data is not on disk
# Use database export API instead
curl -X POST http://localhost:8080/api/v1/export \
  -H "Content-Type: application/json" \
  -d '{"format": "binary", "output": "/backup/snapshot.bin"}' \
  -o /backup/snapshot.bin
```

### Step 2: Prepare Storage Directory

```bash
# Create persistent storage directory
sudo mkdir -p /var/lib/jadevectordb/data
sudo chown jadevectordb:jadevectordb /var/lib/jadevectordb/data
sudo chmod 755 /var/lib/jadevectordb/data

# Verify write access
sudo -u jadevectordb touch /var/lib/jadevectordb/data/test
sudo -u jadevectordb rm /var/lib/jadevectordb/data/test
```

### Step 3: Update Configuration

#### Before (In-Memory)

```cpp
// src/main.cpp
auto persistence = std::make_shared<InMemoryDatabasePersistence>();
auto db_service = std::make_shared<DatabaseService>(persistence);
```

#### After (Persistent)

```cpp
// src/main.cpp
#include "PersistentDatabasePersistence.hpp"
#include "VectorFlushManager.hpp"
#include "SignalHandler.hpp"

// Initialize signal handlers for graceful shutdown
SignalHandler::initialize();

// Create persistent storage
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/var/lib/jadevectordb/data",  // storage_path
    100,                             // max_open_files (adjust based on ulimit)
    std::chrono::seconds(300)        // flush_interval (5 minutes)
);

// Register with flush manager
auto& flush_mgr = VectorFlushManager::get_instance();
flush_mgr.register_persistence("main", persistence);
flush_mgr.start_periodic_flush(std::chrono::seconds(300));

// Register graceful shutdown
SignalHandler::register_shutdown_callback([&flush_mgr, persistence]() {
    spdlog::info("Graceful shutdown initiated...");
    
    // Flush all data
    flush_mgr.flush_all();
    
    // Stop background tasks
    flush_mgr.stop_periodic_flush();
    
    spdlog::info("Shutdown complete");
});

// Create database service (same as before)
auto db_service = std::make_shared<DatabaseService>(persistence);
```

#### Configuration File (config.json)

```json
{
  "persistence": {
    "type": "persistent",
    "storage_path": "/var/lib/jadevectordb/data",
    "max_open_files": 100,
    "flush_interval_seconds": 300
  },
  "vector_storage": {
    "initial_capacity": 10000,
    "enable_auto_flush": true
  }
}
```

### Step 4: Build and Deploy

#### If Building from Source

```bash
cd /path/to/JadeVectorDB/backend

# Clean build
rm -rf build
mkdir build && cd build

# Configure with persistent storage enabled
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Verify binary
./jadevectordb --version
```

#### If Using Prebuilt Binaries

```bash
# Download latest release with persistent storage support
wget https://github.com/YourOrg/JadeVectorDB/releases/download/v2.1.0/jadevectordb-linux-amd64.tar.gz

# Extract
tar -xzf jadevectordb-linux-amd64.tar.gz
cd jadevectordb

# Verify version
./jadevectordb --version  # Should show >= 2.1.0
```

### Step 5: Start with Persistent Storage

```bash
# Stop old instance
sudo systemctl stop jadevectordb

# Start new instance
sudo systemctl start jadevectordb

# Check logs
sudo journalctl -u jadevectordb -f

# Expected log messages:
# [info] Initializing PersistentDatabasePersistence at /var/lib/jadevectordb/data
# [info] LRU cache configured with max_open_files=100
# [info] Starting periodic flush with interval=300s
# [info] JadeVectorDB started successfully
```

### Step 6: Restore Data

#### Option A: JSON Import

```bash
# Import databases from JSON backup
jadevectordb-cli import --format json --input /backup/databases.json

# Monitor progress
tail -f /var/log/jadevectordb/import.log
```

#### Option B: API-Based Restore

```bash
# Restore each database using API
for db in $(jq -r '.databases[].name' /backup/databases.json); do
  echo "Restoring database: $db"
  curl -X POST http://localhost:8080/api/v1/databases/$db/restore \
    -H "Content-Type: application/json" \
    -d @/backup/$db.json
done
```

#### Option C: Programmatic Import

```cpp
#include <fstream>
#include <nlohmann/json.hpp>

void import_from_json(
    const std::string& json_path,
    std::shared_ptr<PersistentDatabasePersistence> persistence
) {
    // Load JSON
    std::ifstream file(json_path);
    nlohmann::json data;
    file >> data;
    
    // Restore each database
    for (const auto& db_json : data["databases"]) {
        std::string db_name = db_json["name"];
        
        // Create database
        DatabaseSchema schema;
        schema.dimension = db_json["schema"]["dimension"];
        schema.distance_metric = parse_metric(db_json["schema"]["distance_metric"]);
        persistence->create_database(db_name, schema);
        
        // Restore vectors in batches
        const auto& vectors = db_json["vectors"];
        const size_t batch_size = 1000;
        
        for (size_t i = 0; i < vectors.size(); i += batch_size) {
            std::vector<int64_t> ids;
            std::vector<std::vector<float>> data_batch;
            std::vector<std::string> metadata_batch;
            
            for (size_t j = i; j < std::min(i + batch_size, vectors.size()); j++) {
                ids.push_back(vectors[j]["id"]);
                data_batch.push_back(vectors[j]["data"].get<std::vector<float>>());
                metadata_batch.push_back(vectors[j]["metadata"].dump());
            }
            
            persistence->batch_store(db_name, ids, data_batch, metadata_batch);
        }
        
        // Flush after each database
        persistence->flush_database(db_name);
        
        spdlog::info("Restored database: {} ({} vectors)", db_name, vectors.size());
    }
}
```

### Step 7: Validation

```bash
# Verify database count
curl http://localhost:8080/api/v1/databases | jq '.databases | length'

# Verify vector counts
for db in $(curl -s http://localhost:8080/api/v1/databases | jq -r '.databases[].name'); do
  count=$(curl -s http://localhost:8080/api/v1/databases/$db/stats | jq '.vector_count')
  echo "$db: $count vectors"
done

# Check storage statistics
curl http://localhost:8080/api/v1/storage/stats | jq .
# Expected output:
# {
#   "total_databases": 10,
#   "total_vectors": 1000000,
#   "total_bytes": 2147483648,
#   "open_files": 10
# }
```

### Step 8: Monitor and Optimize

```bash
# Monitor .jvdb file sizes
du -sh /var/lib/jadevectordb/data/*.jvdb

# Monitor memory usage
ps aux | grep jadevectordb

# Monitor open file descriptors
lsof -p $(pgrep jadevectordb) | wc -l

# Check flush performance
tail -f /var/log/jadevectordb/server.log | grep "flush completed"
```

---

## Configuration Reference

### Recommended Configurations by Use Case

#### Development / Testing

```cpp
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "./data",                      // Local directory
    20,                            // Small file limit
    std::chrono::seconds(60)       // Frequent flushes for testing
);
```

#### Production - Balanced

```cpp
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/var/lib/jadevectordb/data",  // Production storage
    100,                            // Moderate file limit
    std::chrono::seconds(300)       // 5-minute flushes
);
```

#### Production - High Throughput

```cpp
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/mnt/nvme/jadevectordb",      // Fast NVMe storage
    500,                            // High file limit
    std::chrono::seconds(600)       // Less frequent flushes (10 min)
);
```

#### Production - High Durability

```cpp
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/data/jadevectordb",
    50,                             // Conservative file limit
    std::chrono::seconds(60)        // Frequent flushes (1 min)
);
```

#### Memory-Constrained Systems

```cpp
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/var/lib/jadevectordb/data",
    10,                             // Minimal open files
    std::chrono::seconds(300)
);
```

### Configuration Parameters Explained

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `storage_path` | Required | Valid filesystem path | Storage location for `.jvdb` files |
| `max_open_files` | 100 | 5 - 1000 | Higher = less LRU eviction overhead, more memory |
| `flush_interval` | 300s | 0 - 3600s | Lower = better durability, higher I/O overhead |

**Tuning Guidelines**:
- `max_open_files`: Set to 50-80% of `ulimit -n`
- `flush_interval`: Balance between durability and performance
  - Critical data: 60-120 seconds
  - Standard data: 300-600 seconds
  - Bulk ingestion: 600-1800 seconds (then manual flush)

---

## Validation and Testing

### Functional Testing

#### Test 1: Basic Operations

```bash
# Create database
curl -X POST http://localhost:8080/api/v1/databases/test_db \
  -H "Content-Type: application/json" \
  -d '{"dimension": 128, "distance_metric": "cosine"}'

# Store vector
curl -X POST http://localhost:8080/api/v1/databases/test_db/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vector_id": 1,
    "data": [0.1, 0.2, ...],  # 128 dimensions
    "metadata": {"label": "test"}
  }'

# Retrieve vector
curl http://localhost:8080/api/v1/databases/test_db/vectors/1

# Verify .jvdb file created
ls -lh /var/lib/jadevectordb/data/test_db.jvdb
```

#### Test 2: Restart Persistence

```bash
# Store 100 vectors
for i in {1..100}; do
  curl -X POST http://localhost:8080/api/v1/databases/test_db/vectors \
    -H "Content-Type: application/json" \
    -d "{\"vector_id\": $i, \"data\": $(python3 -c "import random; print([random.random() for _ in range(128)])"), \"metadata\": \"{}\"}"
done

# Manually flush
curl -X POST http://localhost:8080/api/v1/storage/flush

# Restart server
sudo systemctl restart jadevectordb

# Verify vectors survived restart
for i in {1..100}; do
  curl -s http://localhost:8080/api/v1/databases/test_db/vectors/$i | jq -e '.data' > /dev/null
  if [ $? -ne 0 ]; then
    echo "Vector $i missing after restart!"
    exit 1
  fi
done

echo "✓ All vectors persisted across restart"
```

#### Test 3: LRU Eviction

```bash
# Create 150 databases (exceeds default max_open_files=100)
for i in {1..150}; do
  curl -X POST http://localhost:8080/api/v1/databases/db_$i \
    -H "Content-Type: application/json" \
    -d '{"dimension": 64, "distance_metric": "euclidean"}'
  
  # Store 10 vectors in each
  for j in {1..10}; do
    curl -X POST http://localhost:8080/api/v1/databases/db_$i/vectors \
      -H "Content-Type: application/json" \
      -d "{\"vector_id\": $j, \"data\": $(python3 -c "import random; print([random.random() for _ in range(64)])"), \"metadata\": \"{}\"}"
  done
done

# Check open files (should be <= max_open_files)
open_files=$(lsof -p $(pgrep jadevectordb) | grep "\.jvdb" | wc -l)
echo "Open .jvdb files: $open_files (should be <= 100)"

# Access all databases - should work despite LRU eviction
for i in {1..150}; do
  count=$(curl -s http://localhost:8080/api/v1/databases/db_$i/stats | jq '.vector_count')
  if [ "$count" != "10" ]; then
    echo "Database db_$i has incorrect count: $count"
    exit 1
  fi
done

echo "✓ LRU eviction working correctly"
```

### Performance Testing

#### Baseline Benchmark

```bash
# Run performance tests
cd /path/to/JadeVectorDB/backend/benchmarks
./vector_persistence_benchmark --benchmark_filter=PersistentVectorStore

# Expected output:
# BM_PersistentVectorStore/64          5000 ns/op     (20,000 vectors/sec)
# BM_PersistentVectorStore/512         8000 ns/op     (12,500 vectors/sec)
# BM_PersistentVectorRetrieve/512       800 ns/op     (1,250,000 vectors/sec)
# BM_PersistentBatchStore/1000        50000 ns/op     (20,000 vectors/sec)
```

#### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test vector insertion throughput
ab -n 10000 -c 10 -p vector.json -T application/json \
  http://localhost:8080/api/v1/databases/test_db/vectors

# Expected: 500-1000 requests/sec for single-vector inserts
```

### Data Integrity Testing

```python
# test_integrity.py
import requests
import numpy as np

def test_data_integrity():
    base_url = "http://localhost:8080/api/v1"
    db_name = "integrity_test"
    
    # Create database
    requests.post(f"{base_url}/databases/{db_name}", json={
        "dimension": 128,
        "distance_metric": "cosine"
    })
    
    # Store 1000 vectors with known data
    stored_vectors = {}
    for i in range(1000):
        vec = np.random.rand(128).tolist()
        stored_vectors[i] = vec
        
        requests.post(f"{base_url}/databases/{db_name}/vectors", json={
            "vector_id": i,
            "data": vec,
            "metadata": f'{{"index": {i}}}'
        })
    
    # Flush to ensure durability
    requests.post(f"{base_url}/storage/flush")
    
    # Restart server (manual step)
    input("Restart server and press Enter to continue...")
    
    # Retrieve and validate all vectors
    errors = 0
    for i in range(1000):
        response = requests.get(f"{base_url}/databases/{db_name}/vectors/{i}")
        retrieved = response.json()["data"]
        
        # Check if vectors match (within floating point tolerance)
        if not np.allclose(stored_vectors[i], retrieved, rtol=1e-5):
            errors += 1
            print(f"Mismatch at vector {i}")
    
    print(f"Data integrity test: {1000 - errors}/1000 vectors correct")
    return errors == 0

if __name__ == "__main__":
    assert test_data_integrity(), "Data integrity test failed!"
    print("✓ Data integrity test passed")
```

---

## Rollback Strategy

### When to Rollback

- Data corruption detected during validation
- Unacceptable performance degradation
- Critical bugs in persistent storage implementation
- Insufficient disk space or I/O capacity

### Rollback Procedure

#### Step 1: Stop New Instance

```bash
sudo systemctl stop jadevectordb
```

#### Step 2: Revert Configuration

```cpp
// Revert src/main.cpp to use InMemoryDatabasePersistence
auto persistence = std::make_shared<InMemoryDatabasePersistence>();
auto db_service = std::make_shared<DatabaseService>(persistence);
```

#### Step 3: Rebuild and Deploy

```bash
cd /path/to/JadeVectorDB/backend
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo systemctl start jadevectordb
```

#### Step 4: Restore Data

```bash
# Import from JSON backup created in Step 1
jadevectordb-cli import --format json --input /backup/databases.json
```

#### Step 5: Validate

```bash
# Verify database count and vector counts
curl http://localhost:8080/api/v1/databases | jq .
```

### Rollback Testing

**Test rollback procedure in staging** before attempting in production:

```bash
# Staging rollback test
1. Perform migration
2. Store test data
3. Execute rollback procedure
4. Verify data restored correctly
5. Document any issues encountered
```

---

## Troubleshooting

### Issue 1: File Permission Errors

**Symptoms**:
```
[error] Failed to create memory-mapped file: Permission denied
[error] Cannot open /var/lib/jadevectordb/data/my_db.jvdb
```

**Solution**:
```bash
# Fix permissions
sudo chown -R jadevectordb:jadevectordb /var/lib/jadevectordb/data
sudo chmod -R 755 /var/lib/jadevectordb/data

# Verify
ls -la /var/lib/jadevectordb/data
```

### Issue 2: Disk Space Exhausted

**Symptoms**:
```
[error] Failed to extend memory-mapped file: No space left on device
```

**Solution**:
```bash
# Check disk usage
df -h /var/lib/jadevectordb/data

# Free up space or move to larger volume
sudo systemctl stop jadevectordb
sudo mv /var/lib/jadevectordb/data /mnt/large_volume/jadevectordb/data
sudo ln -s /mnt/large_volume/jadevectordb/data /var/lib/jadevectordb/data
sudo systemctl start jadevectordb
```

### Issue 3: Too Many Open Files

**Symptoms**:
```
[error] Failed to open database store: Too many open files
[warning] LRU eviction cannot free file descriptors
```

**Solution**:
```bash
# Check current limits
ulimit -n

# Increase limit temporarily
ulimit -n 4096

# Increase permanently (/etc/security/limits.conf)
echo "jadevectordb soft nofile 4096" | sudo tee -a /etc/security/limits.conf
echo "jadevectordb hard nofile 8192" | sudo tee -a /etc/security/limits.conf

# Or reduce max_open_files in configuration
# Change from 100 to 50
```

### Issue 4: Slow Startup After Migration

**Symptoms**:
- Server takes 30+ seconds to start
- High disk I/O during startup

**Solution**:
```cpp
// Enable lazy loading (default, but ensure not overridden)
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    storage_path,
    max_open_files,
    flush_interval
    // Do not pre-load all databases
);

// If pre-loading is necessary, do it asynchronously
std::thread([persistence]() {
    for (const auto& db_name : persistence->list_databases()) {
        // Warm up LRU cache
        persistence->get_storage_stats();
    }
}).detach();
```

### Issue 5: Data Corruption After Crash

**Symptoms**:
```
[error] Invalid magic number in /var/lib/jadevectordb/data/db.jvdb
[error] Header validation failed for database: db
```

**Solution**:
```bash
# Attempt to recover from backup
cp /backup/databases/db.jvdb /var/lib/jadevectordb/data/db.jvdb

# If no backup, delete corrupted file and re-ingest
rm /var/lib/jadevectordb/data/db.jvdb
jadevectordb-cli import --database db --input /backup/db.json

# Prevention: Reduce flush_interval for critical databases
```

### Issue 6: High Flush Latency

**Symptoms**:
- Flush operations take 10+ seconds
- Server unresponsive during flush

**Solution**:
```bash
# Check disk I/O performance
iostat -x 1 10

# If I/O is slow, consider:
# 1. Migrate to SSD
# 2. Increase flush_interval
# 3. Reduce number of databases per flush batch

# Tune kernel I/O scheduler (for HDDs)
echo "deadline" | sudo tee /sys/block/sda/queue/scheduler
```

### Issue 7: Memory Usage Higher Than Expected

**Symptoms**:
- RSS memory usage exceeds expectations
- OOM killer terminates process

**Solution**:
```bash
# Check actual memory usage
ps aux | grep jadevectordb

# Monitor memory-mapped regions
pmap -x $(pgrep jadevectordb)

# Solutions:
# 1. Reduce max_open_files to limit mmap regions
# 2. Enable madvise for better memory management:

// In code:
#ifdef __linux__
madvise(mapped_memory, size, MADV_RANDOM);  // Optimize for random access
#endif
```

---

## Performance Tuning

### Optimize for Throughput

```cpp
// Configuration
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/mnt/nvme/jadevectordb",      // Use fastest storage
    500,                            // High file limit
    std::chrono::seconds(600)       // Less frequent flushes
);

// Use batch operations
std::vector<int64_t> ids(10000);
std::vector<std::vector<float>> vectors(10000, std::vector<float>(512));
std::vector<std::string> metadata(10000);
persistence->batch_store("db", ids, vectors, metadata);

// Disable auto-flush during bulk ingestion
flush_mgr.stop_periodic_flush();
// ... bulk ingest ...
persistence->flush();  // Single flush at end
flush_mgr.start_periodic_flush(std::chrono::seconds(600));
```

### Optimize for Latency

```cpp
// Keep databases hot in LRU cache
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/mnt/nvme/jadevectordb",
    200,                            // High file limit for hot databases
    std::chrono::seconds(300)
);

// Pre-warm critical databases
for (const auto& db : critical_databases) {
    // Trigger load by accessing
    persistence->retrieve_vector(db, 0);
}
```

### Optimize for Memory

```cpp
// Minimize memory footprint
auto persistence = std::make_shared<PersistentDatabasePersistence>(
    "/var/lib/jadevectordb/data",
    20,                             // Low file limit
    std::chrono::seconds(180)       // More frequent flushes to reduce dirty pages
);

// Use madvise for better memory management
#ifdef __linux__
// In MemoryMappedVectorStore::map_file():
madvise(mapped_data, file_size, MADV_SEQUENTIAL);  // If access is sequential
madvise(mapped_data, file_size, MADV_DONTNEED);    // Aggressively free pages
#endif
```

### Monitoring for Performance

```bash
# Monitor flush performance
tail -f /var/log/jadevectordb/server.log | grep "flush_duration_ms"

# Monitor LRU eviction rate
curl http://localhost:8080/api/v1/storage/stats | jq '.lru_evictions_per_minute'

# Monitor I/O wait
iostat -x 1

# Monitor memory pressure
vmstat 1
```

---

## Best Practices

### 1. Regular Backups

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup/jadevectordb/$(date +%Y%m%d_%H%M%S)"
DATA_DIR="/var/lib/jadevectordb/data"

# Flush before backup
curl -X POST http://localhost:8080/api/v1/storage/flush

# Wait for flush to complete
sleep 10

# Copy .jvdb files
mkdir -p "$BACKUP_DIR"
cp "$DATA_DIR"/*.jvdb "$BACKUP_DIR/"

# Compress
tar -czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_DIR" .
rm -rf "$BACKUP_DIR"

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

### 2. Monitoring and Alerts

```yaml
# Prometheus alert rules
groups:
  - name: jadevectordb
    rules:
      - alert: HighFlushLatency
        expr: jadevectordb_flush_duration_ms > 10000
        for: 5m
        annotations:
          summary: "Flush operations are slow"
      
      - alert: ApproachingFileLimit
        expr: jadevectordb_open_files / jadevectordb_max_open_files > 0.8
        annotations:
          summary: "Approaching max_open_files limit"
      
      - alert: LowDiskSpace
        expr: node_filesystem_avail_bytes{mountpoint="/var/lib/jadevectordb"} < 10737418240  # 10GB
        annotations:
          summary: "Low disk space for vector storage"
```

### 3. Capacity Planning

```python
# Estimate storage requirements
def estimate_storage(num_databases, vectors_per_db, dimension, metadata_size_bytes=100):
    bytes_per_vector = (dimension * 4) + metadata_size_bytes + 32  # data + metadata + index
    total_vectors = num_databases * vectors_per_db
    total_bytes = total_vectors * bytes_per_vector
    
    # Add 20% overhead for headers, alignment, fragmentation
    total_bytes *= 1.2
    
    return {
        "total_vectors": total_vectors,
        "storage_gb": total_bytes / (1024**3),
        "recommended_disk_gb": (total_bytes / (1024**3)) * 2  # 2x for safety
    }

# Example
est = estimate_storage(
    num_databases=100,
    vectors_per_db=100000,
    dimension=512,
    metadata_size_bytes=200
)
print(f"Estimated storage: {est['storage_gb']:.2f} GB")
print(f"Recommended disk: {est['recommended_disk_gb']:.2f} GB")
```

### 4. Testing Strategy

- **Before Migration**: Full backup + rollback test in staging
- **During Migration**: Incremental validation after each step
- **After Migration**: 24-hour monitoring period with rollback plan ready
- **Long-term**: Regular data integrity checks, performance benchmarks

---

## Conclusion

Migrating to persistent vector storage provides significant benefits in durability, scalability, and operational simplicity. By following this guide, you can migrate safely with minimal downtime and full data integrity.

**Key Takeaways**:
- Test migration in staging first
- Always maintain backups before migration
- Monitor performance during and after migration
- Tune configuration based on workload characteristics
- Have a tested rollback plan ready

For questions or issues, please refer to:
- [Persistence API Reference](persistence_api_reference.md)
- [Architecture Documentation](architecture.md)
- [Community Forum](https://forum.jadevectordb.com)
- [GitHub Issues](https://github.com/YourOrg/JadeVectorDB/issues)

---

**Document Version**: 1.0  
**Last Updated**: Sprint 2.1 (2025-01-XX)  
**Applies to**: JadeVectorDB >= 2.1.0
