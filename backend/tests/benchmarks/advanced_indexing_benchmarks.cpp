#include <benchmark/benchmark.h>
#include <memory>
#include <vector>
#include <random>
#include <chrono>

#include "services/index/pq_index.h"
#include "services/index/opq_index.h"
#include "services/index/sq_index.h"
#include "services/index/composite_index.h"
#include "models/vector.h"
#include "lib/logging.h"

using namespace jadevectordb;

// Benchmark fixture for Product Quantization (PQ) index
class PqIndexBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // Initialize PQ index with parameters
        PqIndex::PqParams pq_params;
        pq_params.subvector_dimension = 8;
        pq_params.num_centroids = 256;
        pq_params.use_residual = false;
        pq_params.max_iterations = 100;
        pq_params.tolerance = 1e-4f;
        pq_params.random_seed = 100;
        
        pq_index_ = std::make_unique<PqIndex>(pq_params);
        pq_index_->initialize(pq_params);
        
        // Generate test vectors
        size_t num_vectors = state.range(0);
        size_t vector_dimension = state.range(1);
        generateTestVectors(num_vectors, vector_dimension);
    }
    
    void TearDown(const ::benchmark::State& state) override {
        pq_index_.reset();
        test_vectors_.clear();
    }
    
    void generateTestVectors(size_t count, size_t dimension) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        test_vectors_.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            Vector v;
            v.id = "vector_" + std::to_string(i);
            v.values = std::vector<float>(dimension);
            
            for (auto& val : v.values) {
                val = dist(gen);
            }
            
            test_vectors_.push_back(v);
        }
    }
    
    std::unique_ptr<PqIndex> pq_index_;
    std::vector<Vector> test_vectors_;
};

// Benchmark PQ index building performance
BENCHMARK_DEFINE_F(PqIndexBenchmark, BuildPerformance)(benchmark::State& state) {
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    
    // Add vectors to the index
    std::vector<std::pair<int, std::vector<float>>> pq_vectors;
    pq_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        pq_vectors.emplace_back(static_cast<int>(i), test_vectors_[i].values);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = pq_index_->build_from_vectors(pq_vectors);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("PQ index build failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

// Benchmark PQ index search performance
BENCHMARK_DEFINE_F(PqIndexBenchmark, SearchPerformance)(benchmark::State& state) {
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    int top_k = state.range(2);
    
    // First build the index
    std::vector<std::pair<int, std::vector<float>>> pq_vectors;
    pq_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        pq_vectors.emplace_back(static_cast<int>(i), test_vectors_[i].values);
    }
    
    auto build_result = pq_index_->build_from_vectors(pq_vectors);
    if (!build_result.has_value()) {
        state.SkipWithError("PQ index build failed");
        return;
    }
    
    // Create a query vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> query_vector(vector_dimension);
    for (auto& val : query_vector) {
        val = dist(gen);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = pq_index_->search(query_vector, top_k, 0.0f);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("PQ index search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

// Register PQ benchmarks
BENCHMARK_REGISTER_F(PqIndexBenchmark, BuildPerformance)
    ->Args({1000, 128})    // 1K vectors, 128 dimensions
    ->Args({10000, 128})   // 10K vectors, 128 dimensions
    ->Args({100000, 128})  // 100K vectors, 128 dimensions
    ->Args({1000000, 128}) // 1M vectors, 128 dimensions
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(PqIndexBenchmark, SearchPerformance)
    ->Args({10000, 128, 10})   // 10K vectors, 128 dimensions, top-10 results
    ->Args({100000, 128, 10})  // 100K vectors, 128 dimensions, top-10 results
    ->Args({1000000, 128, 10}) // 1M vectors, 128 dimensions, top-10 results
    ->Unit(benchmark::kMillisecond);

// Benchmark fixture for Optimized Product Quantization (OPQ) index
class OpqIndexBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // Initialize OPQ index with parameters
        OpqIndex::OpqParams opq_params;
        opq_params.subvector_dimension = 8;
        opq_params.num_centroids = 256;
        opq_params.rotation_optimization_iterations = 10;
        opq_params.use_residual = false;
        opq_params.max_iterations = 100;
        opq_params.tolerance = 1e-4f;
        opq_params.random_seed = 100;
        
        opq_index_ = std::make_unique<OpqIndex>(opq_params);
        opq_index_->initialize(opq_params);
        
        // Generate test vectors
        size_t num_vectors = state.range(0);
        size_t vector_dimension = state.range(1);
        generateTestVectors(num_vectors, vector_dimension);
    }
    
    void TearDown(const ::benchmark::State& state) override {
        opq_index_.reset();
        test_vectors_.clear();
    }
    
    void generateTestVectors(size_t count, size_t dimension) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        test_vectors_.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            Vector v;
            v.id = "vector_" + std::to_string(i);
            v.values = std::vector<float>(dimension);
            
            for (auto& val : v.values) {
                val = dist(gen);
            }
            
            test_vectors_.push_back(v);
        }
    }
    
    std::unique_ptr<OpqIndex> opq_index_;
    std::vector<Vector> test_vectors_;
};

// Benchmark OPQ index building performance
BENCHMARK_DEFINE_F(OpqIndexBenchmark, BuildPerformance)(benchmark::State& state) {
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    
    // Add vectors to the index
    std::vector<std::pair<int, std::vector<float>>> opq_vectors;
    opq_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        opq_vectors.emplace_back(static_cast<int>(i), test_vectors_[i].values);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = opq_index_->build_from_vectors(opq_vectors);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("OPQ index build failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

// Benchmark OPQ index search performance
BENCHMARK_DEFINE_F(OpqIndexBenchmark, SearchPerformance)(benchmark::State& state) {
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    int top_k = state.range(2);
    
    // First build the index
    std::vector<std::pair<int, std::vector<float>>> opq_vectors;
    opq_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        opq_vectors.emplace_back(static_cast<int>(i), test_vectors_[i].values);
    }
    
    auto build_result = opq_index_->build_from_vectors(opq_vectors);
    if (!build_result.has_value()) {
        state.SkipWithError("OPQ index build failed");
        return;
    }
    
    // Create a query vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> query_vector(vector_dimension);
    for (auto& val : query_vector) {
        val = dist(gen);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = opq_index_->search(query_vector, top_k, 0.0f);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("OPQ index search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

// Register OPQ benchmarks
BENCHMARK_REGISTER_F(OpqIndexBenchmark, BuildPerformance)
    ->Args({1000, 128})    // 1K vectors, 128 dimensions
    ->Args({10000, 128})   // 10K vectors, 128 dimensions
    ->Args({100000, 128})  // 100K vectors, 128 dimensions
    ->Args({1000000, 128}) // 1M vectors, 128 dimensions
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(OpqIndexBenchmark, SearchPerformance)
    ->Args({10000, 128, 10})   // 10K vectors, 128 dimensions, top-10 results
    ->Args({100000, 128, 10})  // 100K vectors, 128 dimensions, top-10 results
    ->Args({1000000, 128, 10}) // 1M vectors, 128 dimensions, top-10 results
    ->Unit(benchmark::kMillisecond);

// Benchmark fixture for Scalar Quantization (SQ) index
class SqIndexBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // Initialize SQ index with parameters
        SqIndex::SqParams sq_params;
        sq_params.bits_per_dimension = 8;
        sq_params.normalize_vectors = false;
        sq_params.random_seed = 100;
        
        sq_index_ = std::make_unique<SqIndex>(sq_params);
        sq_index_->initialize(sq_params);
        
        // Generate test vectors
        size_t num_vectors = state.range(0);
        size_t vector_dimension = state.range(1);
        generateTestVectors(num_vectors, vector_dimension);
    }
    
    void TearDown(const ::benchmark::State& state) override {
        sq_index_.reset();
        test_vectors_.clear();
    }
    
    void generateTestVectors(size_t count, size_t dimension) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        test_vectors_.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            Vector v;
            v.id = "vector_" + std::to_string(i);
            v.values = std::vector<float>(dimension);
            
            for (auto& val : v.values) {
                val = dist(gen);
            }
            
            test_vectors_.push_back(v);
        }
    }
    
    std::unique_ptr<SqIndex> sq_index_;
    std::vector<Vector> test_vectors_;
};

// Benchmark SQ index building performance
BENCHMARK_DEFINE_F(SqIndexBenchmark, BuildPerformance)(benchmark::State& state) {
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    
    // Add vectors to the index
    std::vector<std::pair<int, std::vector<float>>> sq_vectors;
    sq_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        sq_vectors.emplace_back(static_cast<int>(i), test_vectors_[i].values);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = sq_index_->build_from_vectors(sq_vectors);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("SQ index build failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

// Benchmark SQ index search performance
BENCHMARK_DEFINE_F(SqIndexBenchmark, SearchPerformance)(benchmark::State& state) {
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    int top_k = state.range(2);
    
    // First build the index
    std::vector<std::pair<int, std::vector<float>>> sq_vectors;
    sq_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        sq_vectors.emplace_back(static_cast<int>(i), test_vectors_[i].values);
    }
    
    auto build_result = sq_index_->build_from_vectors(sq_vectors);
    if (!build_result.has_value()) {
        state.SkipWithError("SQ index build failed");
        return;
    }
    
    // Create a query vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> query_vector(vector_dimension);
    for (auto& val : query_vector) {
        val = dist(gen);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = sq_index_->search(query_vector, top_k, 0.0f);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("SQ index search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

// Register SQ benchmarks
BENCHMARK_REGISTER_F(SqIndexBenchmark, BuildPerformance)
    ->Args({1000, 128})    // 1K vectors, 128 dimensions
    ->Args({10000, 128})   // 10K vectors, 128 dimensions
    ->Args({100000, 128})  // 100K vectors, 128 dimensions
    ->Args({1000000, 128}) // 1M vectors, 128 dimensions
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SqIndexBenchmark, SearchPerformance)
    ->Args({10000, 128, 10})   // 10K vectors, 128 dimensions, top-10 results
    ->Args({100000, 128, 10})  // 100K vectors, 128 dimensions, top-10 results
    ->Args({1000000, 128, 10}) // 1M vectors, 128 dimensions, top-10 results
    ->Unit(benchmark::kMillisecond);

// Benchmark fixture for Composite Index
class CompositeIndexBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // Initialize composite index with parameters
        CompositeIndex::CompositeIndexParams composite_params;
        composite_params.fusion_method = CompositeIndex::CompositeIndexParams::RRF;
        composite_params.rrf_k = 60;
        composite_params.enable_filtering = true;
        composite_params.allow_multiple_searches = true;
        
        composite_index_ = std::make_unique<CompositeIndex>(composite_params);
        composite_index_->initialize(composite_params);
        
        // Generate test vectors
        size_t num_vectors = state.range(0);
        size_t vector_dimension = state.range(1);
        generateTestVectors(num_vectors, vector_dimension);
    }
    
    void TearDown(const ::benchmark::State& state) override {
        composite_index_.reset();
        test_vectors_.clear();
    }
    
    void generateTestVectors(size_t count, size_t dimension) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        test_vectors_.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            Vector v;
            v.id = "vector_" + std::to_string(i);
            v.values = std::vector<float>(dimension);
            
            for (auto& val : v.values) {
                val = dist(gen);
            }
            
            test_vectors_.push_back(v);
        }
    }
    
    std::unique_ptr<CompositeIndex> composite_index_;
    std::vector<Vector> test_vectors_;
};

// Benchmark Composite Index building performance
BENCHMARK_DEFINE_F(CompositeIndexBenchmark, BuildPerformance)(benchmark::State& state) {
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    
    // Add vectors to the index
    std::vector<std::pair<int, std::vector<float>>> composite_vectors;
    composite_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        composite_vectors.emplace_back(static_cast<int>(i), test_vectors_[i].values);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = composite_index_->build_from_vectors(composite_vectors);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Composite index build failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

// Benchmark Composite Index search performance
BENCHMARK_DEFINE_F(CompositeIndexBenchmark, SearchPerformance)(benchmark::State& state) {
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    int top_k = state.range(2);
    
    // First build the index
    std::vector<std::pair<int, std::vector<float>>> composite_vectors;
    composite_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        composite_vectors.emplace_back(static_cast<int>(i), test_vectors_[i].values);
    }
    
    auto build_result = composite_index_->build_from_vectors(composite_vectors);
    if (!build_result.has_value()) {
        state.SkipWithError("Composite index build failed");
        return;
    }
    
    // Create a query vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> query_vector(vector_dimension);
    for (auto& val : query_vector) {
        val = dist(gen);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = composite_index_->search(query_vector, top_k, 0.0f);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Composite index search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

// Register Composite Index benchmarks
BENCHMARK_REGISTER_F(CompositeIndexBenchmark, BuildPerformance)
    ->Args({1000, 128})    // 1K vectors, 128 dimensions
    ->Args({10000, 128})   // 10K vectors, 128 dimensions
    ->Args({100000, 128})  // 100K vectors, 128 dimensions
    ->Args({1000000, 128}) // 1M vectors, 128 dimensions
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(CompositeIndexBenchmark, SearchPerformance)
    ->Args({10000, 128, 10})   // 10K vectors, 128 dimensions, top-10 results
    ->Args({100000, 128, 10})  // 100K vectors, 128 dimensions, top-10 results
    ->Args({1000000, 128, 10}) // 1M vectors, 128 dimensions, top-10 results
    ->Unit(benchmark::kMillisecond);

// Performance validation benchmarks to ensure requirements are met

// PQ index performance validation
static void BM_PqIndexPerformanceValidation(benchmark::State& state) {
    // This benchmark validates that PQ index meets performance requirements
    // According to spec PB-004: "Advanced indexing algorithms provide similarity search 
    // with 95% accuracy for datasets up to 10 million vectors with response times under 50ms"
    
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    int top_k = state.range(2);
    
    // Initialize PQ index with parameters
    PqIndex::PqParams pq_params;
    pq_params.subvector_dimension = 8;
    pq_params.num_centroids = 256;
    pq_params.use_residual = false;
    pq_params.max_iterations = 100;
    pq_params.tolerance = 1e-4f;
    pq_params.random_seed = 100;
    
    auto pq_index = std::make_unique<PqIndex>(pq_params);
    pq_index->initialize(pq_params);
    
    // Generate test vectors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<std::pair<int, std::vector<float>>> pq_vectors;
    pq_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> values(vector_dimension);
        for (auto& val : values) {
            val = dist(gen);
        }
        pq_vectors.emplace_back(static_cast<int>(i), std::move(values));
    }
    
    // Build the index
    auto build_result = pq_index->build_from_vectors(pq_vectors);
    if (!build_result.has_value()) {
        state.SkipWithError("PQ index build failed");
        return;
    }
    
    // Create a query vector
    std::vector<float> query_vector(vector_dimension);
    for (auto& val : query_vector) {
        val = dist(gen);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = pq_index->search(query_vector, top_k, 0.0f);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Validate performance requirement: under 50ms for 10M vectors
        // For smaller datasets, we'll scale the requirement proportionally
        double max_allowed_time_ms = 50.0 * (static_cast<double>(num_vectors) / 10000000.0);
        double actual_time_ms = duration.count() / 1000.0;
        
        if (actual_time_ms > max_allowed_time_ms) {
            state.SkipWithError("PQ index exceeded performance requirement");
        }
        
        if (result.has_value()) {
            state.SetIterationTime(actual_time_ms / 1000.0);
        } else {
            state.SkipWithError("PQ index search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

BENCHMARK(BM_PqIndexPerformanceValidation)
    ->Args({100000, 128, 10})   // 100K vectors, 128 dimensions, top-10 results
    ->Args({1000000, 128, 10})  // 1M vectors, 128 dimensions, top-10 results
    ->Unit(benchmark::kMillisecond);

// OPQ index performance validation
static void BM_OpqIndexPerformanceValidation(benchmark::State& state) {
    // This benchmark validates that OPQ index meets performance requirements
    // According to spec PB-004: "Advanced indexing algorithms provide similarity search 
    // with 95% accuracy for datasets up to 10 million vectors with response times under 50ms"
    
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    int top_k = state.range(2);
    
    // Initialize OPQ index with parameters
    OpqIndex::OpqParams opq_params;
    opq_params.subvector_dimension = 8;
    opq_params.num_centroids = 256;
    opq_params.rotation_optimization_iterations = 10;
    opq_params.use_residual = false;
    opq_params.max_iterations = 100;
    opq_params.tolerance = 1e-4f;
    opq_params.random_seed = 100;
    
    auto opq_index = std::make_unique<OpqIndex>(opq_params);
    opq_index->initialize(opq_params);
    
    // Generate test vectors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<std::pair<int, std::vector<float>>> opq_vectors;
    opq_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> values(vector_dimension);
        for (auto& val : values) {
            val = dist(gen);
        }
        opq_vectors.emplace_back(static_cast<int>(i), std::move(values));
    }
    
    // Build the index
    auto build_result = opq_index->build_from_vectors(opq_vectors);
    if (!build_result.has_value()) {
        state.SkipWithError("OPQ index build failed");
        return;
    }
    
    // Create a query vector
    std::vector<float> query_vector(vector_dimension);
    for (auto& val : query_vector) {
        val = dist(gen);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = opq_index->search(query_vector, top_k, 0.0f);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Validate performance requirement: under 50ms for 10M vectors
        // For smaller datasets, we'll scale the requirement proportionally
        double max_allowed_time_ms = 50.0 * (static_cast<double>(num_vectors) / 10000000.0);
        double actual_time_ms = duration.count() / 1000.0;
        
        if (actual_time_ms > max_allowed_time_ms) {
            state.SkipWithError("OPQ index exceeded performance requirement");
        }
        
        if (result.has_value()) {
            state.SetIterationTime(actual_time_ms / 1000.0);
        } else {
            state.SkipWithError("OPQ index search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

BENCHMARK(BM_OpqIndexPerformanceValidation)
    ->Args({100000, 128, 10})   // 100K vectors, 128 dimensions, top-10 results
    ->Args({1000000, 128, 10})  // 1M vectors, 128 dimensions, top-10 results
    ->Unit(benchmark::kMillisecond);

// SQ index performance validation
static void BM_SqIndexPerformanceValidation(benchmark::State& state) {
    // This benchmark validates that SQ index meets performance requirements
    // According to spec PB-004: "Advanced indexing algorithms provide similarity search 
    // with 95% accuracy for datasets up to 10 million vectors with response times under 50ms"
    
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    int top_k = state.range(2);
    
    // Initialize SQ index with parameters
    SqIndex::SqParams sq_params;
    sq_params.bits_per_dimension = 8;
    sq_params.normalize_vectors = false;
    sq_params.random_seed = 100;
    
    auto sq_index = std::make_unique<SqIndex>(sq_params);
    sq_index->initialize(sq_params);
    
    // Generate test vectors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<std::pair<int, std::vector<float>>> sq_vectors;
    sq_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> values(vector_dimension);
        for (auto& val : values) {
            val = dist(gen);
        }
        sq_vectors.emplace_back(static_cast<int>(i), std::move(values));
    }
    
    // Build the index
    auto build_result = sq_index->build_from_vectors(sq_vectors);
    if (!build_result.has_value()) {
        state.SkipWithError("SQ index build failed");
        return;
    }
    
    // Create a query vector
    std::vector<float> query_vector(vector_dimension);
    for (auto& val : query_vector) {
        val = dist(gen);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = sq_index->search(query_vector, top_k, 0.0f);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Validate performance requirement: under 50ms for 10M vectors
        // For smaller datasets, we'll scale the requirement proportionally
        double max_allowed_time_ms = 50.0 * (static_cast<double>(num_vectors) / 10000000.0);
        double actual_time_ms = duration.count() / 1000.0;
        
        if (actual_time_ms > max_allowed_time_ms) {
            state.SkipWithError("SQ index exceeded performance requirement");
        }
        
        if (result.has_value()) {
            state.SetIterationTime(actual_time_ms / 1000.0);
        } else {
            state.SkipWithError("SQ index search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

BENCHMARK(BM_SqIndexPerformanceValidation)
    ->Args({100000, 128, 10})   // 100K vectors, 128 dimensions, top-10 results
    ->Args({1000000, 128, 10})  // 1M vectors, 128 dimensions, top-10 results
    ->Unit(benchmark::kMillisecond);

// Composite Index performance validation
static void BM_CompositeIndexPerformanceValidation(benchmark::State& state) {
    // This benchmark validates that Composite Index meets performance requirements
    // According to spec PB-004: "Advanced indexing algorithms provide similarity search 
    // with 95% accuracy for datasets up to 10 million vectors with response times under 50ms"
    
    size_t num_vectors = state.range(0);
    size_t vector_dimension = state.range(1);
    int top_k = state.range(2);
    
    // Initialize composite index with parameters
    CompositeIndex::CompositeIndexParams composite_params;
    composite_params.fusion_method = CompositeIndex::CompositeIndexParams::RRF;
    composite_params.rrf_k = 60;
    composite_params.enable_filtering = true;
    composite_params.allow_multiple_searches = true;
    
    auto composite_index = std::make_unique<CompositeIndex>(composite_params);
    composite_index->initialize(composite_params);
    
    // Generate test vectors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<std::pair<int, std::vector<float>>> composite_vectors;
    composite_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> values(vector_dimension);
        for (auto& val : values) {
            val = dist(gen);
        }
        composite_vectors.emplace_back(static_cast<int>(i), std::move(values));
    }
    
    // Build the index
    auto build_result = composite_index->build_from_vectors(composite_vectors);
    if (!build_result.has_value()) {
        state.SkipWithError("Composite index build failed");
        return;
    }
    
    // Create a query vector
    std::vector<float> query_vector(vector_dimension);
    for (auto& val : query_vector) {
        val = dist(gen);
    }
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = composite_index->search(query_vector, top_k, 0.0f);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Validate performance requirement: under 50ms for 10M vectors
        // For smaller datasets, we'll scale the requirement proportionally
        double max_allowed_time_ms = 50.0 * (static_cast<double>(num_vectors) / 10000000.0);
        double actual_time_ms = duration.count() / 1000.0;
        
        if (actual_time_ms > max_allowed_time_ms) {
            state.SkipWithError("Composite index exceeded performance requirement");
        }
        
        if (result.has_value()) {
            state.SetIterationTime(actual_time_ms / 1000.0);
        } else {
            state.SkipWithError("Composite index search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

BENCHMARK(BM_CompositeIndexPerformanceValidation)
    ->Args({100000, 128, 10})   // 100K vectors, 128 dimensions, top-10 results
    ->Args({1000000, 128, 10})  // 1M vectors, 128 dimensions, top-10 results
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();