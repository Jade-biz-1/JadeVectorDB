#include <benchmark/benchmark.h>
#include "services/database_layer.h"
#include "storage/memory_mapped_vector_store.h"
#include <filesystem>
#include <random>
#include <memory>

// Helper function to generate random vectors
std::vector<float> generate_random_vector(size_t dimension, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> vec(dimension);
    for (size_t i = 0; i < dimension; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

// Benchmark 1: Single Vector Store - Persistent
static void BM_PersistentVectorStore(benchmark::State& state) {
    std::string storage_path = "/tmp/jade_benchmark_persistent";
    std::filesystem::remove_all(storage_path);
    std::filesystem::create_directories(storage_path);
    
    auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
        storage_path, nullptr, nullptr, nullptr);
    
    jadevectordb::Database db;
    db.name = "bench_db";
    db.vectorDimension = state.range(0);
    
    auto create_result = persistence->create_database(db);
    std::string db_id = *create_result;
    
    int counter = 0;
    for (auto _ : state) {
        jadevectordb::Vector vec;
        vec.id = "vec_" + std::to_string(counter++);
        vec.values = generate_random_vector(state.range(0), counter);
        vec.databaseId = db_id;
        
        benchmark::DoNotOptimize(persistence->store_vector(db_id, vec));
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float));
    
    std::filesystem::remove_all(storage_path);
}

// Benchmark 2: Single Vector Store - In-Memory
static void BM_InMemoryVectorStore(benchmark::State& state) {
    auto persistence = std::make_unique<jadevectordb::InMemoryDatabasePersistence>(
        nullptr, nullptr, nullptr);
    
    jadevectordb::Database db;
    db.name = "bench_db";
    db.vectorDimension = state.range(0);
    
    auto create_result = persistence->create_database(db);
    std::string db_id = *create_result;
    
    int counter = 0;
    for (auto _ : state) {
        jadevectordb::Vector vec;
        vec.id = "vec_" + std::to_string(counter++);
        vec.values = generate_random_vector(state.range(0), counter);
        vec.databaseId = db_id;
        
        benchmark::DoNotOptimize(persistence->store_vector(db_id, vec));
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float));
}

// Benchmark 3: Single Vector Retrieve - Persistent
static void BM_PersistentVectorRetrieve(benchmark::State& state) {
    std::string storage_path = "/tmp/jade_benchmark_persistent_retrieve";
    std::filesystem::remove_all(storage_path);
    std::filesystem::create_directories(storage_path);
    
    auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
        storage_path, nullptr, nullptr, nullptr);
    
    jadevectordb::Database db;
    db.name = "bench_db";
    db.vectorDimension = state.range(0);
    
    auto create_result = persistence->create_database(db);
    std::string db_id = *create_result;
    
    // Pre-populate with 1000 vectors
    for (int i = 0; i < 1000; i++) {
        jadevectordb::Vector vec;
        vec.id = "vec_" + std::to_string(i);
        vec.values = generate_random_vector(state.range(0), i);
        vec.databaseId = db_id;
        persistence->store_vector(db_id, vec);
    }
    persistence->flush_all();
    
    int counter = 0;
    for (auto _ : state) {
        std::string vec_id = "vec_" + std::to_string(counter++ % 1000);
        benchmark::DoNotOptimize(persistence->retrieve_vector(db_id, vec_id));
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float));
    
    std::filesystem::remove_all(storage_path);
}

// Benchmark 4: Single Vector Retrieve - In-Memory
static void BM_InMemoryVectorRetrieve(benchmark::State& state) {
    auto persistence = std::make_unique<jadevectordb::InMemoryDatabasePersistence>(
        nullptr, nullptr, nullptr);
    
    jadevectordb::Database db;
    db.name = "bench_db";
    db.vectorDimension = state.range(0);
    
    auto create_result = persistence->create_database(db);
    std::string db_id = *create_result;
    
    // Pre-populate with 1000 vectors
    for (int i = 0; i < 1000; i++) {
        jadevectordb::Vector vec;
        vec.id = "vec_" + std::to_string(i);
        vec.values = generate_random_vector(state.range(0), i);
        vec.databaseId = db_id;
        persistence->store_vector(db_id, vec);
    }
    
    int counter = 0;
    for (auto _ : state) {
        std::string vec_id = "vec_" + std::to_string(counter++ % 1000);
        benchmark::DoNotOptimize(persistence->retrieve_vector(db_id, vec_id));
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float));
}

// Benchmark 5: Batch Store - Persistent
static void BM_PersistentBatchStore(benchmark::State& state) {
    std::string storage_path = "/tmp/jade_benchmark_persistent_batch";
    std::filesystem::remove_all(storage_path);
    std::filesystem::create_directories(storage_path);
    
    auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
        storage_path, nullptr, nullptr, nullptr);
    
    jadevectordb::Database db;
    db.name = "bench_db";
    db.vectorDimension = state.range(0);
    
    auto create_result = persistence->create_database(db);
    std::string db_id = *create_result;
    
    const int BATCH_SIZE = 100;
    int counter = 0;
    
    for (auto _ : state) {
        std::vector<jadevectordb::Vector> batch;
        for (int i = 0; i < BATCH_SIZE; i++) {
            jadevectordb::Vector vec;
            vec.id = "vec_" + std::to_string(counter++);
            vec.values = generate_random_vector(state.range(0), counter);
            vec.databaseId = db_id;
            batch.push_back(vec);
        }
        
        benchmark::DoNotOptimize(persistence->batch_store_vectors(db_id, batch));
    }
    
    state.SetItemsProcessed(state.iterations() * BATCH_SIZE);
    state.SetBytesProcessed(state.iterations() * BATCH_SIZE * state.range(0) * sizeof(float));
    
    std::filesystem::remove_all(storage_path);
}

// Benchmark 6: Batch Store - In-Memory
static void BM_InMemoryBatchStore(benchmark::State& state) {
    auto persistence = std::make_unique<jadevectordb::InMemoryDatabasePersistence>(
        nullptr, nullptr, nullptr);
    
    jadevectordb::Database db;
    db.name = "bench_db";
    db.vectorDimension = state.range(0);
    
    auto create_result = persistence->create_database(db);
    std::string db_id = *create_result;
    
    const int BATCH_SIZE = 100;
    int counter = 0;
    
    for (auto _ : state) {
        std::vector<jadevectordb::Vector> batch;
        for (int i = 0; i < BATCH_SIZE; i++) {
            jadevectordb::Vector vec;
            vec.id = "vec_" + std::to_string(counter++);
            vec.values = generate_random_vector(state.range(0), counter);
            vec.databaseId = db_id;
            batch.push_back(vec);
        }
        
        benchmark::DoNotOptimize(persistence->batch_store_vectors(db_id, batch));
    }
    
    state.SetItemsProcessed(state.iterations() * BATCH_SIZE);
    state.SetBytesProcessed(state.iterations() * BATCH_SIZE * state.range(0) * sizeof(float));
}

// Benchmark 7: Startup Time - Persistent (with existing data)
static void BM_PersistentStartupTime(benchmark::State& state) {
    std::string storage_path = "/tmp/jade_benchmark_persistent_startup";
    std::filesystem::remove_all(storage_path);
    std::filesystem::create_directories(storage_path);
    
    // Pre-create data
    {
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            storage_path, nullptr, nullptr, nullptr);
        
        for (int db_idx = 0; db_idx < 10; db_idx++) {
            jadevectordb::Database db;
            db.name = "bench_db_" + std::to_string(db_idx);
            db.vectorDimension = 128;
            
            auto create_result = persistence->create_database(db);
            std::string db_id = *create_result;
            
            for (int i = 0; i < 100; i++) {
                jadevectordb::Vector vec;
                vec.id = "vec_" + std::to_string(i);
                vec.values = generate_random_vector(128, i);
                vec.databaseId = db_id;
                persistence->store_vector(db_id, vec);
            }
        }
        persistence->flush_all();
    }
    
    // Measure startup time
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            storage_path, nullptr, nullptr, nullptr);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        state.SetIterationTime(elapsed.count() / 1e6);
    }
    
    std::filesystem::remove_all(storage_path);
}

// Benchmark 8: Flush Performance
static void BM_PersistentFlush(benchmark::State& state) {
    std::string storage_path = "/tmp/jade_benchmark_persistent_flush";
    std::filesystem::remove_all(storage_path);
    std::filesystem::create_directories(storage_path);
    
    auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
        storage_path, nullptr, nullptr, nullptr);
    
    jadevectordb::Database db;
    db.name = "bench_db";
    db.vectorDimension = state.range(0);
    
    auto create_result = persistence->create_database(db);
    std::string db_id = *create_result;
    
    // Pre-populate
    for (int i = 0; i < 1000; i++) {
        jadevectordb::Vector vec;
        vec.id = "vec_" + std::to_string(i);
        vec.values = generate_random_vector(state.range(0), i);
        vec.databaseId = db_id;
        persistence->store_vector(db_id, vec);
    }
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(persistence->flush_all());
    }
    
    std::filesystem::remove_all(storage_path);
}

// Register benchmarks with different vector dimensions
BENCHMARK(BM_PersistentVectorStore)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);
BENCHMARK(BM_InMemoryVectorStore)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);

BENCHMARK(BM_PersistentVectorRetrieve)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);
BENCHMARK(BM_InMemoryVectorRetrieve)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);

BENCHMARK(BM_PersistentBatchStore)->Arg(128)->Arg(256)->Arg(512);
BENCHMARK(BM_InMemoryBatchStore)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK(BM_PersistentStartupTime)->UseManualTime();

BENCHMARK(BM_PersistentFlush)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK_MAIN();
