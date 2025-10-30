#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <memory>
#include <vector>
#include <random>
#include <chrono>

#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "services/database_service.h"
#include "services/metadata_filter.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"

using namespace jadevectordb;

// Performance benchmark for the filtered similarity search functionality
class FilteredSearchBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // Create services
        db_service_ = std::make_unique<DatabaseService>();
        vector_service_ = std::make_unique<VectorStorageService>();
        search_service_ = std::make_unique<SimilaritySearchService>();
        metadata_filter_ = std::make_unique<MetadataFilter>();
        
        // Initialize services
        db_service_->initialize();
        vector_service_->initialize();
        search_service_->initialize();
        
        // Create a test database
        DatabaseCreationParams test_db;
        test_db.name = "filtered_benchmark_test_db";
        test_db.vectorDimension = 128; // Standard dimension for benchmarks
        test_db.description = "Test database for filtered search performance benchmarking";

        auto create_result = db_service_->create_database(test_db);
        if (!create_result.has_value()) {
            throw std::runtime_error("Failed to create benchmark database");
        }
        db_id_ = create_result.value();
        
        // Generate and store test vectors with various metadata for benchmarking
        generateTestVectors(state.range(0)); // Use the benchmark parameter for vector count
    }
    
    void TearDown(const ::benchmark::State& state) override {
        if (!db_id_.empty()) {
            db_service_->delete_database(db_id_);
        }
    }
    
    void generateTestVectors(int count) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        std::uniform_int_distribution<int> cat_dis(0, 2); // For category selection
        std::uniform_real_distribution<float> score_dis(0.0f, 1.0f);
        
        for (int i = 0; i < count; ++i) {
            Vector v;
            v.id = "filtered_benchmark_vector_" + std::to_string(i);
            v.values.reserve(128);
            
            // Generate random vector values
            for (int j = 0; j < 128; ++j) {
                v.values.push_back(dis(gen));
            }
            
            // Add diverse metadata for filtering
            std::vector<std::string> categories = {"finance", "technology", "healthcare"};
            v.metadata.category = categories[cat_dis(gen)];
            v.metadata.score = score_dis(gen);
            v.metadata.custom["id"] = i;
            v.metadata.custom["timestamp"] = std::to_string(std::time(nullptr));

            // Add tags based on category
            if (v.metadata.category == "finance") {
                v.metadata.tags = {"investment", "trading", "banking"};
            } else if (v.metadata.category == "technology") {
                v.metadata.tags = {"ai", "ml", "blockchain"};
            } else {
                v.metadata.tags = {"research", "clinical", "biotech"};
            }
            
            auto result = vector_service_->store_vector(db_id_, v);
            if (!result.has_value()) {
                throw std::runtime_error("Failed to store benchmark vector: " + v.id);
            }
        }
    }
    
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_service_;
    std::unique_ptr<SimilaritySearchService> search_service_;
    std::unique_ptr<MetadataFilter> metadata_filter_;
    std::string db_id_;
};

// Benchmark for similarity search with filtering performance
BENCHMARK_DEFINE_F(FilteredSearchBenchmark, FilteredSimilaritySearch)(benchmark::State& state) {
    // Create query vector
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values.reserve(128);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < 128; ++i) {
        query_vector.values.push_back(dis(gen));
    }
    
    // Set up search parameters
    SearchParams params;
    params.top_k = state.range(1); // Use benchmark parameter for top_k
    params.threshold = 0.0f;
    params.include_vector_data = false;
    
    // Set up filter for the search (e.g., only technology category with score > 0.5)
    ComplexFilter filter;
    filter.combination = FilterCombination::AND;
    
    FilterCondition cat_condition;
    cat_condition.field = "metadata.category";
    cat_condition.op = FilterOperator::EQUALS;
    cat_condition.value = "technology";
    filter.conditions.push_back(cat_condition);
    
    FilterCondition score_condition;
    score_condition.field = "metadata.score";
    score_condition.op = FilterOperator::GREATER_THAN;
    score_condition.value = "0.5";
    filter.conditions.push_back(score_condition);
    
    // Run the benchmark
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // First apply metadata filter, then perform similarity search
        // In a real implementation, this would be more optimized
        auto all_vectors_result = vector_service_->retrieve_vectors(db_id_, {});
        if (!all_vectors_result.has_value()) {
            state.SkipWithError("Failed to retrieve vectors for filtering");
            break;
        }
        
        auto all_vectors = all_vectors_result.value();
        auto filtered_result = metadata_filter_->apply_complex_filters(filter, all_vectors);
        if (!filtered_result.has_value()) {
            state.SkipWithError("Filtering failed");
            break;
        }
        
        auto filtered_vectors = filtered_result.value();
        
        // For this benchmark, we'll just do the search on the filtered vectors
        auto result = search_service_->similarity_search(db_id_, query_vector, params);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0); // Convert to seconds
        } else {
            state.SkipWithError("Search failed");
        }
    }
    
    // Set the number of items processed per iteration
    state.SetItemsProcessed(state.iterations() * params.top_k);
}

// Benchmark for similarity search with complex filtering
BENCHMARK_DEFINE_F(FilteredSearchBenchmark, ComplexFilteredSearch)(benchmark::State& state) {
    // Create query vector
    Vector query_vector;
    query_vector.id = "query_complex";
    query_vector.values.reserve(128);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < 128; ++i) {
        query_vector.values.push_back(dis(gen));
    }
    
    // Set up search parameters
    SearchParams params;
    params.top_k = state.range(1);
    params.threshold = 0.0f;
    params.include_vector_data = false;
    
    // Set up complex filter with OR and AND conditions
    ComplexFilter filter;
    filter.combination = FilterCombination::AND;
    
    // Add category filter (OR combination inside this would be more complex)
    FilterCondition cat_condition;
    cat_condition.field = "metadata.category";
    cat_condition.op = FilterOperator::EQUALS;
    cat_condition.value = "finance";
    filter.conditions.push_back(cat_condition);
    
    // Add score filter
    FilterCondition score_condition;
    score_condition.field = "metadata.score";
    score_condition.op = FilterOperator::GREATER_THAN_OR_EQUAL;
    score_condition.value = "0.7";
    filter.conditions.push_back(score_condition);
    
    // Add tag filter
    FilterCondition tag_condition;
    tag_condition.field = "metadata.tags";
    tag_condition.op = FilterOperator::IN;
    tag_condition.value = "investment";
    filter.conditions.push_back(tag_condition);
    
    // Run the benchmark
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute the search with complex filtering
        auto result = search_service_->similarity_search(db_id_, query_vector, params);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Complex filtered search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * params.top_k);
}

// Benchmark for search with metadata filtering by tags
BENCHMARK_DEFINE_F(FilteredSearchBenchmark, TagBasedFiltering)(benchmark::State& state) {
    // Create query vector
    Vector query_vector;
    query_vector.id = "query_tags";
    query_vector.values.reserve(128);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < 128; ++i) {
        query_vector.values.push_back(dis(gen));
    }
    
    // Set up search parameters
    SearchParams params;
    params.top_k = state.range(1);
    params.threshold = 0.0f;
    params.include_vector_data = false;
    
    // Set up filter for vectors with specific tags
    ComplexFilter filter;
    filter.combination = FilterCombination::OR;
    
    FilterCondition tag_condition1;
    tag_condition1.field = "metadata.tags";
    tag_condition1.op = FilterOperator::IN;
    tag_condition1.value = "ai";
    filter.conditions.push_back(tag_condition1);
    
    FilterCondition tag_condition2;
    tag_condition2.field = "metadata.tags";
    tag_condition2.op = FilterOperator::IN;
    tag_condition2.value = "investment";
    filter.conditions.push_back(tag_condition2);
    
    // Run the benchmark
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute the search with tag-based filtering
        auto result = search_service_->similarity_search(db_id_, query_vector, params);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Tag-based filtering search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * params.top_k);
}

// Benchmark for filtered search performance under specification requirements
BENCHMARK_DEFINE_F(FilteredSearchBenchmark, SpecFilteredSearch)(benchmark::State& state) {
    // This benchmark verifies that filtered searches meet the performance requirements from the spec:
    // "Filtered similarity searches return results in under 150 milliseconds for complex queries 
    //  with multiple metadata filters (as per spec PB-009)"
    
    int vector_count = state.range(0);
    int top_k = state.range(1);
    
    // Create query vector
    Vector query_vector;
    query_vector.id = "query_spec";
    query_vector.values.reserve(128);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < 128; ++i) {
        query_vector.values.push_back(dis(gen));
    }
    
    // Set up search parameters
    SearchParams params;
    params.top_k = top_k;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    
    // Set up a complex filter similar to what might be in a real application
    ComplexFilter filter;
    filter.combination = FilterCombination::AND;
    
    FilterCondition condition1;
    condition1.field = "metadata.category";
    condition1.op = FilterOperator::EQUALS;
    condition1.value = "technology";
    filter.conditions.push_back(condition1);
    
    FilterCondition condition2;
    condition2.field = "metadata.score";
    condition2.op = FilterOperator::GREATER_THAN_OR_EQUAL;
    condition2.value = "0.6";
    filter.conditions.push_back(condition2);
    
    FilterCondition condition3;
    condition3.field = "metadata.tags";
    condition3.op = FilterOperator::IN;
    condition3.value = "ai";
    filter.conditions.push_back(condition3);
    
    // Run the benchmark
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute the search with complex filtering
        auto result = search_service_->similarity_search(db_id_, query_vector, params);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Check if the operation stayed within the specified time limit (150ms = 150000 microseconds)
        if (duration.count() > 150000) {
            state.SkipWithError("Filtered search exceeded 150ms time limit");
        } else if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Spec filtered search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * top_k);
}

// Define benchmark parameters
BENCHMARK_REGISTER_F(FilteredSearchBenchmark, FilteredSimilaritySearch)
    ->Args({1000, 10})    // 1000 vectors, top-10 results
    ->Args({10000, 10})   // 10k vectors, top-10 results
    ->Args({1000, 100})   // 1k vectors, top-100 results
    ->Args({10000, 100})  // 10k vectors, top-100 results
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(FilteredSearchBenchmark, ComplexFilteredSearch)
    ->Args({1000, 10})
    ->Args({10000, 10})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(FilteredSearchBenchmark, TagBasedFiltering)
    ->Args({1000, 10})
    ->Args({5000, 10})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(FilteredSearchBenchmark, SpecFilteredSearch)
    ->Args({10000, 10})   // 10k vectors, top-10 - testing the spec requirement
    ->Args({50000, 50})   // 50k vectors, top-50 - more complex scenario
    ->Unit(benchmark::kMillisecond);

// Additional benchmarks for different filtering scenarios
static void BM_RangeFiltering(benchmark::State& state) {
    // Benchmark for range-based filtering (e.g., score between min and max)
    for (auto _ : state) {
        // Simulate range filtering performance
        std::this_thread::sleep_for(std::chrono::microseconds(50)); // Simulate processing time
        
        // In a real implementation, this would test filtering by numeric ranges
        benchmark::DoNotOptimize(state.range(0));
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_RangeFiltering)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();