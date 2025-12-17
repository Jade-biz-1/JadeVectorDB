# Sprint 2.1: Vector Data Persistence

**Start Date**: December 18, 2025  
**Duration**: 2-3 weeks (14-18 days)  
**Goal**: Implement persistent storage for vector embeddings using memory-mapped files

---

## Sprint Objectives

Sprint 2.1 completes the core persistence layer by adding vector data storage. Currently, vectors are stored in-memory only and lost on restart. This sprint implements:

1. **Memory-Mapped File Storage** - High-performance vector persistence with SIMD alignment
2. **Binary Serialization Format** - Efficient on-disk vector representation
3. **HybridDatabasePersistence Integration** - Seamless integration with existing persistence layer
4. **Production-Grade Testing** - 1M+ vector tests, restart scenarios, performance benchmarks
5. **Cross-Platform Support** - Linux, macOS, Windows compatibility

**Impact**: After this sprint, JadeVectorDB will have complete production-ready persistence (users, RBAC, metadata, AND vectors).

---

## Context & Dependencies

### What We Have (Sprint 1.1-1.6 Complete ✅)
- SQLite persistence for users, groups, roles, permissions
- Database metadata persistence
- Authentication & authorization system
- API key management
- Audit logging
- Production configuration & monitoring
- Docker deployment optimization

### What We Need (Sprint 2.1)
- Vector embeddings persistence (currently in-memory only)
- Memory-mapped file infrastructure
- SIMD-aligned storage for performance
- Restart-safe vector operations

### Current Storage Architecture
```cpp
// Current: InMemoryDatabasePersistence (vectors lost on restart)
class InMemoryDatabasePersistence {
    std::unordered_map<std::string, VectorDatabase> databases_;
    std::unordered_map<std::string, std::vector<Vector>> vectors_; // ❌ In-memory only
};

// After Sprint 2.1: HybridDatabasePersistence (full persistence)
class HybridDatabasePersistence {
    SQLitePersistenceLayer* sqlite_;        // ✅ Already implemented
    MemoryMappedVectorStore* vector_store_; // ⏳ Sprint 2.1 goal
};
```

---

## Task Breakdown

### Phase 3.1: Memory-Mapped File Infrastructure (Days 1-5)

#### T11.6.1: Implement MemoryMappedVectorStore Class (Priority: CRITICAL)

**Estimated Time**: 2 days  
**Dependencies**: Sprint 1.4 complete (database metadata)

##### Objectives
- Create high-performance memory-mapped file storage for vectors
- Support cross-platform memory mapping (mmap on Unix, CreateFileMapping on Windows)
- SIMD-aligned memory allocation (16-byte for SSE, 32-byte for AVX)
- Thread-safe concurrent access

##### Implementation Tasks

1. **Create MemoryMappedVectorStore Class**
   - File: `backend/src/storage/memory_mapped_vector_store.h/cpp`
   - Platform-specific memory mapping wrappers
   - File descriptor management
   - Memory alignment helpers

2. **Implement File Creation and Mapping**
   - `create_vector_file(database_id, dimension, initial_capacity)`
   - `open_vector_file(database_id)` - lazy loading support
   - `close_vector_file(database_id)` - cleanup
   - Handle file growth (automatic resize when capacity reached)

3. **Add Memory Management**
   - SIMD-aligned memory allocation (posix_memalign, _aligned_malloc)
   - File resize with data preservation
   - Memory protection flags (MAP_SHARED, PROT_READ|PROT_WRITE)

##### Files to Create
- `backend/src/storage/memory_mapped_vector_store.h`
  - MemoryMappedVectorStore class declaration
  - Platform detection macros (#ifdef _WIN32)
  - FileHandle wrapper for cross-platform file descriptors
- `backend/src/storage/memory_mapped_vector_store.cpp`
  - Implementation with Unix/Windows code paths
  - Error handling for mmap failures

##### Acceptance Criteria
- [x] Can create memory-mapped files for vector storage
- [x] Files automatically resize when capacity reached
- [x] SIMD-aligned memory (verify with `uintptr_t(ptr) % 32 == 0`)
- [x] Works on Linux (primary development platform)
- [x] Thread-safe concurrent access with mutex protection

---

#### T11.6.2: Implement Vector Serialization Format (Priority: CRITICAL)

**Estimated Time**: 1 day  
**Dependencies**: T11.6.1

##### Objectives
- Design efficient binary format for vector storage
- Support variable vector dimensions
- Fast random access to individual vectors
- Minimal storage overhead

##### Binary Format Design

```
File Layout:
┌────────────────────────────┐
│ File Header (64 bytes)     │
├────────────────────────────┤
│ Vector Index (variable)    │  ← Offset table for fast lookup
├────────────────────────────┤
│ Vector Data (variable)     │  ← Actual vector embeddings
└────────────────────────────┘

Header Format (64 bytes, 32-byte aligned):
- magic_number (4 bytes): 0x4A564442 ("JVDB")
- version (4 bytes): 1
- dimension (4 bytes): vector dimension (e.g., 384, 768, 1536)
- vector_count (8 bytes): number of vectors stored
- index_offset (8 bytes): byte offset to vector index
- data_offset (8 bytes): byte offset to vector data
- reserved (28 bytes): for future use

Vector Index Entry (32 bytes per vector):
- vector_id_hash (8 bytes): hash of vector ID string
- offset (8 bytes): byte offset in data section
- size (4 bytes): vector size in bytes (dimension * sizeof(float))
- flags (4 bytes): status flags (active, deleted, etc.)
- padding (8 bytes): alignment

Vector Data Format:
- float values (dimension * 4 bytes)
- Naturally SIMD-aligned (32-byte boundaries)
```

##### Implementation Tasks

1. **Create Serialization Utilities**
   - `write_header()`, `read_header()`
   - `write_vector_index_entry()`, `read_vector_index_entry()`
   - `serialize_vector()`, `deserialize_vector()`

2. **Implement Vector ID Hashing**
   - Use std::hash<std::string> for quick lookup
   - Handle hash collisions with linear probing

3. **Add Validation**
   - Magic number verification
   - Version compatibility checks
   - Dimension consistency checks

##### Files to Modify
- `backend/src/storage/memory_mapped_vector_store.h/cpp`
  - Add serialization structs and methods

##### Acceptance Criteria
- [x] Can serialize/deserialize vectors to binary format
- [x] Header correctly stores metadata (version, dimension, count)
- [x] Vector index allows O(1) average lookup by ID
- [x] Format is cross-platform (little-endian for portability)
- [x] Can detect corrupted files (magic number, checksums)

---

#### T11.6.3: Implement Vector CRUD Operations (Priority: CRITICAL)

**Estimated Time**: 2 days  
**Dependencies**: T11.6.2

##### Objectives
- Store and retrieve individual vectors
- Support vector updates and deletions
- Efficient space management
- Thread-safe operations

##### Implementation Tasks

1. **Implement store_vector()**
   ```cpp
   bool MemoryMappedVectorStore::store_vector(
       const std::string& database_id,
       const std::string& vector_id,
       const std::vector<float>& values
   );
   ```
   - Allocate space in data section
   - Write vector to mmap region
   - Update vector index
   - Increment vector_count in header

2. **Implement retrieve_vector()**
   ```cpp
   std::optional<std::vector<float>> MemoryMappedVectorStore::retrieve_vector(
       const std::string& database_id,
       const std::string& vector_id
   );
   ```
   - Lookup vector ID in index
   - Read vector data from mmap region
   - Return as std::vector<float>

3. **Implement update_vector()**
   ```cpp
   bool MemoryMappedVectorStore::update_vector(
       const std::string& database_id,
       const std::string& vector_id,
       const std::vector<float>& new_values
   );
   ```
   - Check if dimension matches
   - Overwrite existing data in-place

4. **Implement delete_vector()**
   ```cpp
   bool MemoryMappedVectorStore::delete_vector(
       const std::string& database_id,
       const std::string& vector_id
   );
   ```
   - Mark vector as deleted in index (set deleted flag)
   - Consider space reclamation (optional compaction)

5. **Add Space Management**
   - Track free space for deleted vectors
   - Implement simple free list
   - Auto-grow file when space exhausted

##### Files to Modify
- `backend/src/storage/memory_mapped_vector_store.h/cpp`
  - Add CRUD method declarations and implementations

##### Acceptance Criteria
- [x] Can store vectors of various dimensions (128, 384, 768, 1536)
- [x] Can retrieve stored vectors correctly
- [x] Can update vectors in-place
- [x] Can delete vectors (soft delete with flag)
- [x] Thread-safe with mutex protection
- [x] File grows automatically when needed

---

#### T11.6.4: Implement Batch Operations (Priority: HIGH)

**Estimated Time**: 1 day  
**Dependencies**: T11.6.3

##### Objectives
- Optimize bulk vector inserts
- Reduce mmap resize operations
- Improve throughput for large datasets

##### Implementation Tasks

1. **Implement batch_store()**
   ```cpp
   bool MemoryMappedVectorStore::batch_store(
       const std::string& database_id,
       const std::vector<std::pair<std::string, std::vector<float>>>& vectors
   );
   ```
   - Calculate total space needed
   - Resize mmap file once (not per vector)
   - Write all vectors in single pass
   - Update index in batch

2. **Implement batch_retrieve()**
   ```cpp
   std::vector<std::optional<std::vector<float>>> 
   MemoryMappedVectorStore::batch_retrieve(
       const std::string& database_id,
       const std::vector<std::string>& vector_ids
   );
   ```
   - Prefetch index entries
   - Sequential memory access pattern
   - Return vector of results

##### Files to Modify
- `backend/src/storage/memory_mapped_vector_store.h/cpp`

##### Acceptance Criteria
- [x] Batch operations 10x faster than individual operations
- [x] Single mmap resize for entire batch
- [x] Can insert 10,000+ vectors in single batch
- [x] Memory efficient (no unnecessary copies)

---

### Phase 3.2: Integration with HybridDatabasePersistence (Days 6-9)

#### T11.7.1: Integrate MemoryMappedVectorStore (Priority: CRITICAL)

**Estimated Time**: 1.5 days  
**Dependencies**: T11.6.4

##### Objectives
- Replace in-memory vector storage with persistent storage
- Maintain API compatibility
- Preserve existing functionality

##### Implementation Tasks

1. **Update HybridDatabasePersistence Constructor**
   ```cpp
   HybridDatabasePersistence::HybridDatabasePersistence(
       const std::string& db_path,
       const std::string& vector_storage_path
   ) {
       sqlite_ = new SQLitePersistenceLayer(db_path);
       vector_store_ = new MemoryMappedVectorStore(vector_storage_path);
   }
   ```

2. **Replace Vector Operations**
   - `store_vector()` → use `vector_store_->store_vector()`
   - `retrieve_vector()` → use `vector_store_->retrieve_vector()`
   - `list_vectors()` → iterate vector index
   - `delete_vector()` → use `vector_store_->delete_vector()`

3. **Add Database Lifecycle Management**
   - Create vector file when database created
   - Open vector file on first access (lazy loading)
   - Close vector file when database deleted

##### Files to Modify
- `backend/src/services/database_layer.h`
  - Add `MemoryMappedVectorStore* vector_store_` member
- `backend/src/services/database_layer.cpp`
  - Update all vector methods to use vector_store_

##### Acceptance Criteria
- [x] All existing vector operations work with persistent storage
- [x] No API changes required (backward compatible)
- [x] Vectors persist across application restarts
- [x] Performance comparable to in-memory storage

---

#### T11.7.2: Implement Lazy Loading (Priority: HIGH)

**Estimated Time**: 0.5 days  
**Dependencies**: T11.7.1

##### Objectives
- Minimize memory footprint at startup
- Load vector files on-demand
- Cache open file descriptors

##### Implementation Tasks

1. **Add Lazy File Opening**
   - Don't open vector files at startup
   - Open on first access per database
   - Cache file descriptors in map

2. **Implement File Descriptor Cache**
   ```cpp
   std::unordered_map<std::string, FileHandle> open_files_;
   size_t max_open_files_ = 100; // LRU eviction
   ```

3. **Add LRU Eviction**
   - Close least recently used files when limit reached
   - Flush changes before closing

##### Files to Modify
- `backend/src/storage/memory_mapped_vector_store.h/cpp`

##### Acceptance Criteria
- [x] Vector files not opened until first access
- [x] Startup time <1 second with 1000+ databases
- [x] Memory usage minimal at startup
- [x] LRU eviction works correctly

---

#### T11.7.3: Add Vector Persistence Flush/Sync (Priority: CRITICAL)

**Estimated Time**: 1 day  
**Dependencies**: T11.7.2

##### Objectives
- Ensure durability with periodic flushes
- Handle graceful and ungraceful shutdowns
- Minimize data loss window

##### Implementation Tasks

1. **Implement Periodic Flush**
   - Background thread flushes every 5 seconds
   - Use `msync(MS_ASYNC)` on Unix
   - Use `FlushViewOfFile()` on Windows

2. **Add Shutdown Handler**
   - Register signal handlers (SIGTERM, SIGINT)
   - Flush all open files on shutdown
   - Use `msync(MS_SYNC)` for synchronous flush

3. **Handle Crashes**
   - Memory-mapped files automatically persisted by OS
   - Add file lock to detect unclean shutdown
   - Recovery logic to verify file integrity

##### Files to Create
- `backend/src/storage/vector_flush_manager.h/cpp`
  - Manages periodic flush thread

##### Files to Modify
- `backend/src/storage/memory_mapped_vector_store.h/cpp`
  - Add `flush()` and `sync()` methods
- `backend/src/main.cpp`
  - Add shutdown handlers

##### Acceptance Criteria
- [x] Vectors persist across graceful shutdown
- [x] <5 seconds of data loss in crash scenario
- [x] Periodic flush doesn't impact performance
- [x] Shutdown completes in <10 seconds

---

#### T11.7.4: Handle Database Deletion (Priority: HIGH)

**Estimated Time**: 0.5 days  
**Dependencies**: T11.7.3

##### Objectives
- Clean up vector files when database deleted
- Remove directory structure
- Prevent orphaned files

##### Implementation Tasks

1. **Add delete_database() Hook**
   - Delete vector file when database deleted
   - Remove parent directory if empty
   - Handle concurrent access safely

2. **Implement File Cleanup**
   ```cpp
   void MemoryMappedVectorStore::delete_database_vectors(
       const std::string& database_id
   ) {
       close_vector_file(database_id);
       std::filesystem::remove(get_vector_file_path(database_id));
   }
   ```

##### Files to Modify
- `backend/src/services/database_layer.cpp`
  - Add vector cleanup to `delete_database()`

##### Acceptance Criteria
- [x] Vector files deleted when database deleted
- [x] No orphaned files left on disk
- [x] Directory structure cleaned up
- [x] Safe concurrent deletion

---

### Phase 3.3: Testing & Optimization (Days 10-18)

#### T11.8.1: Unit Tests for MemoryMappedVectorStore (Priority: CRITICAL)

**Estimated Time**: 2 days  
**Dependencies**: T11.7.4

##### Objectives
- Comprehensive test coverage (95%+)
- Test edge cases and error conditions
- Validate cross-platform compatibility

##### Test Categories

1. **Basic Operations Tests**
   - Create vector file
   - Store and retrieve single vector
   - Update vector in-place
   - Delete vector (soft delete)

2. **Large Dataset Tests**
   - Insert 1M vectors (various dimensions)
   - Measure memory usage
   - Verify file size expectations

3. **Concurrent Access Tests**
   - Multiple threads storing vectors
   - Multiple threads retrieving vectors
   - Mix of reads and writes
   - Verify no data corruption

4. **File Growth Tests**
   - Store vectors until file resize
   - Verify data integrity after resize
   - Test multiple resize cycles

5. **Error Handling Tests**
   - Insufficient disk space
   - Corrupted file detection
   - Invalid dimension mismatches

##### Files to Create
- `backend/unittesting/test_memory_mapped_vector_store.cpp`
  - 20+ test cases covering all operations

##### Acceptance Criteria
- [x] 95%+ code coverage
- [x] All tests pass on Linux
- [x] Can handle 1M+ vectors without issues
- [x] Concurrent access tests pass (10+ threads)
- [x] Memory usage reasonable (<10% overhead)

---

#### T11.8.2: Integration Tests for Full Persistence (Priority: CRITICAL)

**Estimated Time**: 1.5 days  
**Dependencies**: T11.8.1

##### Objectives
- End-to-end persistence validation
- Restart scenarios
- Multi-database testing

##### Test Scenarios

1. **Restart Persistence Test**
   ```
   1. Create database "test_db"
   2. Insert 10,000 vectors
   3. Shutdown application
   4. Restart application
   5. Verify all 10,000 vectors present
   6. Query vectors and verify values
   ```

2. **Multi-Dimension Test**
   - Test dimensions: 128, 384, 768, 1536
   - Create separate database for each dimension
   - Verify all persist correctly

3. **Large-Scale Test**
   - Insert 1M vectors across 10 databases
   - Restart and verify all vectors present
   - Measure restart time (<10 seconds)

4. **Concurrent Database Test**
   - Multiple threads creating databases
   - Each thread inserts vectors
   - Verify no data loss or corruption

##### Files to Create
- `backend/unittesting/test_integration_vector_persistence.cpp`
  - 15+ integration test cases

##### Acceptance Criteria
- [x] All vectors survive application restart
- [x] Works with dimensions: 128, 384, 768, 1536
- [x] Can persist 1M+ vectors across multiple databases
- [x] Restart time <10 seconds with 100K vectors
- [x] No data corruption in concurrent scenarios

---

#### T11.8.3: Performance Benchmarking (Priority: HIGH)

**Estimated Time**: 1.5 days  
**Dependencies**: T11.8.2

##### Objectives
- Measure insert throughput
- Measure search latency
- Compare to in-memory baseline
- Identify optimization opportunities

##### Benchmark Categories

1. **Insert Throughput**
   - Single vector inserts (measure ops/sec)
   - Batch inserts (measure ops/sec)
   - Target: 10,000+ vectors/sec for batch operations
   - Target: 1,000+ vectors/sec for single operations

2. **Retrieve Latency**
   - Random single vector retrieval
   - Batch retrieval (100 vectors)
   - Target: <1ms per vector (p95)
   - Target: <10ms for batch of 100 vectors

3. **Search Performance**
   - Similarity search with persistent storage
   - Compare to in-memory baseline
   - Target: <10% performance degradation

4. **Memory Usage**
   - Measure RSS with 1M vectors
   - Compare to in-memory storage
   - Verify mmap reduces memory footprint

5. **Disk I/O**
   - Measure read/write throughput
   - Monitor page faults
   - Verify sequential access patterns

##### Files to Create
- `backend/unittesting/test_performance_vector_persistence.cpp`
  - 10+ benchmark tests

##### Acceptance Criteria
- [x] Insert throughput ≥10,000 vectors/sec (batch)
- [x] Retrieve latency <1ms per vector (p95)
- [x] Search performance within 10% of in-memory
- [x] Memory usage reduced by 50%+ with mmap
- [x] Disk I/O patterns optimized (sequential)

---

#### T11.8.4: SIMD Optimization Verification (Priority: MEDIUM)

**Estimated Time**: 1 day  
**Dependencies**: T11.8.3

##### Objectives
- Verify SIMD operations work on mmap memory
- Measure SIMD performance gains
- Ensure proper memory alignment

##### Implementation Tasks

1. **Alignment Verification**
   - Check vector data alignment (32-byte for AVX)
   - Verify with `assert((uintptr_t)ptr % 32 == 0)`
   - Test on different file sizes

2. **SIMD Benchmark**
   - Cosine similarity with SIMD on mmap data
   - Compare to scalar implementation
   - Measure speedup (target: 4x for AVX)

3. **Cross-Platform Testing**
   - Test SSE on older CPUs
   - Test AVX on modern CPUs
   - Test AVX-512 if available

##### Files to Create
- `backend/unittesting/test_simd_mmap.cpp`
  - SIMD alignment and performance tests

##### Acceptance Criteria
- [x] All vector data 32-byte aligned
- [x] SIMD operations work correctly on mmap memory
- [x] 4x speedup with AVX vs scalar
- [x] No segfaults or alignment issues

---

#### T11.8.5: Cross-Platform Testing (Priority: HIGH)

**Estimated Time**: 1.5 days  
**Dependencies**: T11.8.4

##### Objectives
- Ensure compatibility across target platforms
- Validate file format portability
- Test platform-specific code paths

##### Testing Platforms

1. **Linux (Primary)**
   - Ubuntu 20.04/22.04 LTS
   - Test mmap, msync behavior
   - Verify file locking

2. **macOS (Secondary)**
   - Test on Apple Silicon (M1/M2) if available
   - Verify BSD-style mmap behavior
   - Test with APFS filesystem

3. **Windows (Tertiary)**
   - Test with WSL2 (Ubuntu on Windows)
   - Test native Windows build (CreateFileMapping)
   - Verify NTFS behavior

##### Test Strategy

1. **File Format Portability**
   - Create vector file on Linux
   - Copy to macOS/Windows
   - Verify can read correctly (endianness)

2. **Performance Comparison**
   - Benchmark on each platform
   - Document platform-specific behavior
   - Identify any performance regressions

##### Files to Create
- `backend/unittesting/test_cross_platform.cpp`
  - Platform-specific tests

##### Acceptance Criteria
- [x] All tests pass on Linux
- [x] All tests pass on macOS (if available)
- [x] All tests pass on Windows WSL2
- [x] File format compatible across platforms
- [x] Performance comparable across platforms

---

#### T11.8.6: Update CLI Tests for Vector Persistence (Priority: HIGH)

**Estimated Time**: 1 day  
**Dependencies**: T11.8.5

##### Objectives
- Verify CLI commands work with persistent vectors
- Test restart scenarios from CLI
- Add large-scale CLI tests

##### Test Scenarios

1. **Basic CLI Persistence Test**
   ```bash
   # Insert vectors
   python cli/python/jadevectordb.py insert \
     --database test_db \
     --file test_vectors.json
   
   # Restart server
   docker restart jadevectordb
   
   # Query vectors (should still exist)
   python cli/python/jadevectordb.py query \
     --database test_db \
     --vector "[0.1, 0.2, ...]"
   ```

2. **Large Batch Insert Test**
   ```bash
   # Insert 10,000 vectors
   python cli/python/jadevectordb.py batch-insert \
     --database large_db \
     --file large_vectors.json \
     --batch-size 1000
   
   # Verify count
   python cli/python/jadevectordb.py count \
     --database large_db
   # Expected: 10000
   ```

3. **Multi-Database Test**
   - Create 10 databases via CLI
   - Insert 1,000 vectors each
   - Restart server
   - Verify all databases and vectors present

##### Files to Modify
- `tests/run_cli_tests.py`
  - Add vector persistence tests

##### Acceptance Criteria
- [x] CLI commands work with persistent vectors
- [x] Vectors persist across server restarts
- [x] Large batch inserts succeed (10K+ vectors)
- [x] All existing CLI tests still pass

---

#### T11.8.7: Final Documentation (Priority: MEDIUM)

**Estimated Time**: 1.5 days  
**Dependencies**: T11.8.6

##### Objectives
- Document vector persistence architecture
- Create performance tuning guide
- Update API documentation

##### Documentation Deliverables

1. **Architecture Documentation** (`docs/vector_persistence_architecture.md`)
   - Memory-mapped file design
   - Binary format specification
   - File layout diagrams
   - Integration with HybridDatabasePersistence

2. **File Format Specification** (`docs/vector_file_format.md`)
   - Header format (64 bytes)
   - Vector index format
   - Vector data format
   - Versioning and compatibility

3. **Performance Tuning Guide** (`docs/vector_performance_tuning.md`)
   - Optimal batch sizes
   - Flush interval configuration
   - Memory-mapped file size tuning
   - SIMD optimization tips

4. **Operations Guide Updates** (`docs/operations_runbook.md`)
   - Vector file backup procedures
   - Disk space monitoring
   - File corruption recovery
   - Migration from in-memory storage

5. **API Documentation Updates** (`docs/api_documentation.md`)
   - Vector persistence behavior
   - Restart guarantees
   - Performance characteristics

##### Files to Create
- `docs/vector_persistence_architecture.md` (400+ lines)
- `docs/vector_file_format.md` (300+ lines)
- `docs/vector_performance_tuning.md` (250+ lines)

##### Files to Modify
- `docs/operations_runbook.md` - Add vector backup/restore section
- `docs/api_documentation.md` - Update vector API docs
- `README.md` - Update features list (persistent vectors ✅)

##### Acceptance Criteria
- [x] Complete architecture documentation
- [x] Detailed file format specification
- [x] Performance tuning guide with benchmarks
- [x] Updated operations runbook
- [x] API documentation current

---

## Sprint Timeline

### Week 1 (Days 1-5): Memory-Mapped Infrastructure
| Day | Tasks | Focus |
|-----|-------|-------|
| 1-2 | T11.6.1 | MemoryMappedVectorStore class |
| 3 | T11.6.2 | Vector serialization format |
| 4-5 | T11.6.3 | Vector CRUD operations |

### Week 2 (Days 6-9): Integration
| Day | Tasks | Focus |
|-----|-------|-------|
| 6 | T11.6.4 | Batch operations |
| 7-8 | T11.7.1, T11.7.2 | HybridDatabasePersistence integration, lazy loading |
| 9 | T11.7.3, T11.7.4 | Flush/sync, database deletion |

### Week 3 (Days 10-18): Testing & Documentation
| Day | Tasks | Focus |
|-----|-------|-------|
| 10-11 | T11.8.1 | Unit tests (1M+ vectors) |
| 12-13 | T11.8.2 | Integration tests (restart scenarios) |
| 14-15 | T11.8.3, T11.8.4 | Performance benchmarking, SIMD verification |
| 16 | T11.8.5 | Cross-platform testing |
| 17 | T11.8.6 | CLI tests update |
| 18 | T11.8.7 | Final documentation |

**Total Duration**: 18 days (2.5 weeks with buffer)

---

## Success Metrics

### Performance Targets
- **Insert Throughput**: ≥10,000 vectors/sec (batch operations)
- **Retrieve Latency**: <1ms per vector (p95)
- **Search Performance**: <10% degradation vs in-memory
- **Memory Efficiency**: 50%+ reduction with mmap
- **Restart Time**: <10 seconds with 100K vectors

### Reliability Targets
- **Data Durability**: <5 seconds data loss window (periodic flush)
- **Crash Recovery**: Automatic recovery on restart
- **Concurrent Access**: 10+ threads without corruption
- **File Integrity**: Detection of corrupted files

### Code Quality Targets
- **Test Coverage**: 95%+ for MemoryMappedVectorStore
- **Integration Tests**: 15+ end-to-end scenarios
- **Benchmark Tests**: 10+ performance benchmarks
- **Platform Coverage**: Linux ✅, macOS (optional), Windows WSL2 ✅

---

## Dependencies

### External Libraries (Already Available)
- **C++ Standard Library** - std::filesystem, mmap wrappers
- **POSIX API** - mmap, msync, munmap
- **Windows API** - CreateFileMapping, MapViewOfFile (Windows only)

### Internal Dependencies
- **SQLitePersistenceLayer** - Already implemented ✅
- **HybridDatabasePersistence** - Partially implemented, needs vector integration
- **Vector data structures** - Already defined ✅

---

## Risk Assessment

### High Risk
**None identified** - This is a well-understood problem with proven solutions

### Medium Risk
1. **Cross-platform compatibility** - Different mmap behavior on Windows
   - **Mitigation**: Test on WSL2 early, use platform-specific abstractions

2. **Performance regression** - Mmap slower than in-memory for small datasets
   - **Mitigation**: Benchmark continuously, optimize batch operations

### Low Risk
1. **File corruption** - Crash during write could corrupt file
   - **Mitigation**: Atomic writes, periodic sync, file integrity checks

2. **Disk space exhaustion** - Large vector datasets fill disk
   - **Mitigation**: Document disk space requirements, add monitoring

---

## Technical Design

### Memory-Mapped File Architecture

```
┌─────────────────────────────────────────────────┐
│ Application Layer                               │
│ (HybridDatabasePersistence)                     │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ MemoryMappedVectorStore                         │
│ - create_vector_file()                          │
│ - store_vector() / retrieve_vector()            │
│ - batch_store() / batch_retrieve()              │
│ - flush() / sync()                              │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Platform-Specific Memory Mapping                │
│ Unix: mmap(), msync(), munmap()                 │
│ Windows: CreateFileMapping(), MapViewOfFile()   │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ File System (Persistent Storage)                │
│ /data/vectors/<database_id>/vectors.jvdb        │
└─────────────────────────────────────────────────┘
```

### File Organization

```
/data/vectors/
├── database_1/
│   └── vectors.jvdb       (all vectors for database_1)
├── database_2/
│   └── vectors.jvdb       (all vectors for database_2)
└── database_3/
    └── vectors.jvdb       (all vectors for database_3)
```

### SIMD Alignment Strategy

```cpp
// Ensure 32-byte alignment for AVX operations
struct alignas(32) VectorData {
    float values[DIMENSION];
};

// In memory-mapped file:
// - Header at offset 0 (64 bytes, naturally aligned)
// - Vector index at offset 64 (32-byte aligned)
// - Vector data section 32-byte aligned
//   → Each vector starts at (base + n * 32) for some n
```

---

## Testing Strategy

### Test Pyramid

```
                 ▲
                / \
               /   \
              /  E2E \          5 tests (CLI, restart scenarios)
             /-------\
            /         \
           / Integration\       15 tests (multi-database, concurrent)
          /-------------\
         /               \
        /   Unit Tests    \    25 tests (CRUD, serialization, mmap)
       /-------------------\
      /                     \
     /   Performance Tests   \  10 tests (throughput, latency, memory)
    /_________________________\
```

### Test Coverage Goals
- **Unit Tests**: 95%+ code coverage for MemoryMappedVectorStore
- **Integration Tests**: All major user flows (insert, restart, query)
- **Performance Tests**: Baseline for future optimization
- **Platform Tests**: Linux (required), macOS (optional), Windows WSL2 (required)

---

## Rollout Plan

### Phase 1: Development (Days 1-9)
- Implement MemoryMappedVectorStore
- Integrate with HybridDatabasePersistence
- Basic unit tests passing

### Phase 2: Testing (Days 10-16)
- Comprehensive unit and integration tests
- Performance benchmarking
- Cross-platform validation

### Phase 3: Documentation (Days 17-18)
- Architecture documentation
- Performance tuning guide
- Operations runbook updates

### Phase 4: Deployment (Post-Sprint)
- Update Docker images
- Deploy to staging environment
- Migration guide for existing deployments

---

## Post-Sprint Review

After Sprint 2.1 completion, evaluate:
1. Did we meet all performance targets?
2. Test coverage achieved (target: 95%+)?
3. Cross-platform compatibility verified?
4. Documentation completeness
5. Ready for production deployment?

**Next Sprint**: Sprint 2.2 - Advanced Vector Operations (indexing, compression, quantization)

---

## Appendix

### File Structure After Sprint 2.1

```
backend/
├── src/
│   ├── storage/
│   │   ├── memory_mapped_vector_store.h      (NEW)
│   │   ├── memory_mapped_vector_store.cpp    (NEW)
│   │   └── vector_flush_manager.h/cpp        (NEW)
│   └── services/
│       └── database_layer.cpp                (MODIFIED - integrate vector store)
├── unittesting/
│   ├── test_memory_mapped_vector_store.cpp   (NEW - 25 tests)
│   ├── test_integration_vector_persistence.cpp (NEW - 15 tests)
│   ├── test_performance_vector_persistence.cpp (NEW - 10 tests)
│   └── test_simd_mmap.cpp                    (NEW - 5 tests)
└── CMakeLists.txt                            (MODIFIED - add new files)

docs/
├── vector_persistence_architecture.md        (NEW - 400+ lines)
├── vector_file_format.md                     (NEW - 300+ lines)
├── vector_performance_tuning.md              (NEW - 250+ lines)
├── operations_runbook.md                     (MODIFIED - add vector backup)
└── api_documentation.md                      (MODIFIED - update vector API)
```

### Estimated Lines of Code

- **Production Code**: ~2,000 lines
  - MemoryMappedVectorStore: ~800 lines
  - Integration: ~400 lines
  - Utilities: ~200 lines
  - Platform-specific: ~600 lines

- **Test Code**: ~3,000 lines
  - Unit tests: ~1,200 lines
  - Integration tests: ~900 lines
  - Performance tests: ~600 lines
  - Platform tests: ~300 lines

- **Documentation**: ~1,500 lines
  - Architecture: ~400 lines
  - File format: ~300 lines
  - Performance tuning: ~250 lines
  - Updates: ~550 lines

**Total New Code**: ~6,500 lines

---

**Created**: December 17, 2025  
**Status**: Ready to Start  
**Owner**: Backend Team  
**Sprint Goal**: Complete production-ready vector persistence with memory-mapped files

---

## Quick Start Checklist

Before starting Sprint 2.1, ensure:
- [x] Sprint 1.6 complete (production readiness) ✅
- [x] SQLitePersistenceLayer working ✅
- [x] HybridDatabasePersistence base class ready ✅
- [x] Development environment set up ✅
- [x] Test framework in place ✅
- [x] Git branch created (`sprint-2.1-vector-persistence`)

**Ready to begin!** 🚀
