#include <benchmark/benchmark.h>
#include <memory>
#include <vector>
#include <random>
#include <chrono>

#include "services/metadata_filter.h"
#include "models/vector.h"
#include "lib/logging.h"

using namespace jadevectordb;

// Benchmark fixture for advanced metadata filtering
class AdvancedFilteringBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        filter_ = std::make_unique<MetadataFilter>();
        
        // Generate test vectors with various metadata
        size_t num_vectors = state.range(0);
        generateTestVectors(num_vectors);
    }
    
    void TearDown(const ::benchmark::State& state) override {
        filter_.reset();
        test_vectors_.clear();
    }
    
    void generateTestVectors(size_t count) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> float_dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> int_dist(0, 1000);
        std::uniform_int_distribution<int> coord_dist(-90, 90);  // Simplified coordinate range
        
        test_vectors_.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            Vector v;
            v.id = "vector_" + std::to_string(i);
            v.values = std::vector<float>(128);  // 128-dimensional vectors
            
            // Generate random vector values
            for (auto& val : v.values) {
                val = float_dist(gen);
            }
            
            // Add metadata
            v.metadata.owner = "user" + std::to_string(i % 100);  // 100 different users
            v.metadata.category = "category" + std::to_string(i % 10);  // 10 categories
            v.metadata.status = (i % 3 == 0) ? "active" : (i % 3 == 1) ? "draft" : "archived";
            v.metadata.score = float_dist(gen);
            v.metadata.created_at = "2025-01-01T" + std::to_string(i % 24) + ":00:00Z";
            v.metadata.updated_at = "2025-01-02T" + std::to_string(i % 24) + ":00:00Z";
            
            // Add tags
            v.metadata.tags = {
                "tag" + std::to_string(i % 5),
                "tag" + std::to_string((i + 1) % 5),
                "tag" + std::to_string((i + 2) % 5)
            };
            
            // Add permissions
            v.metadata.permissions = {
                "read",
                (i % 2 == 0) ? "write" : "execute"
            };
            
            // Add custom fields
            v.metadata.custom["project"] = "project-" + std::to_string(i % 20);
            v.metadata.custom["department"] = "dept-" + std::to_string(i % 5);
            v.metadata.custom["location"] = std::to_string(coord_dist(gen)) + "," + std::to_string(coord_dist(gen));
            v.metadata.custom["timestamp"] = "2025-01-01T" + std::to_string(i % 24) + ":" + 
                                          std::to_string((i * 2) % 60) + ":00Z";
            v.metadata.custom["description"] = "This is a sample document for testing advanced filtering capabilities with ID " + 
                                             std::to_string(i) + " and various metadata fields";
            
            test_vectors_.push_back(v);
        }
    }
    
    std::unique_ptr<MetadataFilter> filter_;
    std::vector<Vector> test_vectors_;
};

// Benchmark geospatial filtering performance
BENCHMARK_DEFINE_F(AdvancedFilteringBenchmark, GeospatialFiltering)(benchmark::State& state) {
    // Create a geospatial query: find vectors within a radius
    GeoQuery geo_query(GeospatialOperator::WITHIN_RADIUS, "metadata.custom.location");
    geo_query.center = Point(0.0, 0.0);  // Center point
    geo_query.radius = 1000.0;  // 1000 meters
    
    std::vector<GeoQuery> geo_queries = {geo_query};
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter_->apply_geo_filters(geo_queries, test_vectors_);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Geospatial filtering failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * test_vectors_.size());
}

// Benchmark temporal filtering performance
BENCHMARK_DEFINE_F(AdvancedFilteringBenchmark, TemporalFiltering)(benchmark::State& state) {
    // Create a temporal query: find vectors before a certain time
    TemporalQuery temporal_query(TemporalOperator::BEFORE, "metadata.custom.timestamp");
    
    std::tm tm_time = {};
    std::istringstream ss("2025-01-01T12:00:00Z");
    ss >> std::get_time(&tm_time, "%Y-%m-%dT%H:%M:%SZ");
    temporal_query.time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm_time));
    
    std::vector<TemporalQuery> temporal_queries = {temporal_query};
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter_->apply_temporal_filters(temporal_queries, test_vectors_);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Temporal filtering failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * test_vectors_.size());
}

// Benchmark nested object filtering performance
BENCHMARK_DEFINE_F(AdvancedFilteringBenchmark, NestedObjectFiltering)(benchmark::State& state) {
    // Create a nested query: check if a path exists
    NestedQuery nested_query("metadata.custom.location", NestedOperator::EXISTS_PATH, "");
    
    std::vector<NestedQuery> nested_queries = {nested_query};
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter_->apply_nested_filters(nested_queries, test_vectors_);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Nested object filtering failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * test_vectors_.size());
}

// Benchmark full-text search performance
BENCHMARK_DEFINE_F(AdvancedFilteringBenchmark, FullTextSearch)(benchmark::State& state) {
    // Create a full-text query: match all terms
    FullTextQuery fulltext_query("metadata.custom.description", "sample document testing", FullTextOperator::MATCHES_ALL_TERMS);
    
    std::vector<FullTextQuery> fulltext_queries = {fulltext_query};
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter_->apply_fulltext_filters(fulltext_queries, test_vectors_);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Full-text search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * test_vectors_.size());
}

// Benchmark fuzzy matching performance
BENCHMARK_DEFINE_F(AdvancedFilteringBenchmark, FuzzyMatching)(benchmark::State& state) {
    // Create a fuzzy matching query
    FullTextQuery fuzzy_query("metadata.custom.description", "sampel", FullTextOperator::FUZZY_MATCH);
    fuzzy_query.max_edit_distance = 2;
    
    std::vector<FullTextQuery> fuzzy_queries = {fuzzy_query};
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter_->apply_fulltext_filters(fuzzy_queries, test_vectors_);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Fuzzy matching failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * test_vectors_.size());
}

// Benchmark combined advanced filtering performance
BENCHMARK_DEFINE_F(AdvancedFilteringBenchmark, CombinedAdvancedFiltering)(benchmark::State& state) {
    // Create multiple types of queries
    
    // Geospatial query
    GeoQuery geo_query(GeospatialOperator::WITHIN_BOUNDING_BOX, "metadata.custom.location");
    geo_query.bbox = BoundingBox(Point(-10.0, -10.0), Point(10.0, 10.0));
    std::vector<GeoQuery> geo_queries = {geo_query};
    
    // Temporal query
    TemporalQuery temporal_query(TemporalOperator::BETWEEN, "metadata.custom.timestamp");
    std::tm tm_start = {};
    std::tm tm_end = {};
    std::istringstream ss_start("2025-01-01T00:00:00Z");
    std::istringstream ss_end("2025-01-01T12:00:00Z");
    ss_start >> std::get_time(&tm_start, "%Y-%m-%dT%H:%M:%SZ");
    ss_end >> std::get_time(&tm_end, "%Y-%m-%dT%H:%M:%SZ");
    temporal_query.time_range = TimeRange(
        std::chrono::system_clock::from_time_t(std::mktime(&tm_start)),
        std::chrono::system_clock::from_time_t(std::mktime(&tm_end))
    );
    std::vector<TemporalQuery> temporal_queries = {temporal_query};
    
    // Nested query
    NestedQuery nested_query("metadata.custom.description", NestedOperator::EXISTS_PATH, "");
    std::vector<NestedQuery> nested_queries = {nested_query};
    
    // Full-text query
    FullTextQuery fulltext_query("metadata.custom.description", "sample document", FullTextOperator::MATCHES_ALL_TERMS);
    std::vector<FullTextQuery> fulltext_queries = {fulltext_query};
    
    // Regular filter conditions
    std::vector<FilterCondition> conditions;
    FilterCondition owner_condition("metadata.owner", FilterOperator::EQUALS, "user1");
    conditions.push_back(owner_condition);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter_->apply_advanced_filters(conditions, geo_queries, temporal_queries,
                                                   nested_queries, fulltext_queries, test_vectors_);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.has_value()) {
            state.SetIterationTime(duration.count() / 1000000.0);
        } else {
            state.SkipWithError("Combined advanced filtering failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * test_vectors_.size());
}

// Register benchmarks with different dataset sizes
BENCHMARK_REGISTER_F(AdvancedFilteringBenchmark, GeospatialFiltering)
    ->Arg(1000)    // 1K vectors
    ->Arg(10000)   // 10K vectors
    ->Arg(100000)  // 100K vectors
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(AdvancedFilteringBenchmark, TemporalFiltering)
    ->Arg(1000)    // 1K vectors
    ->Arg(10000)   // 10K vectors
    ->Arg(100000)  // 100K vectors
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(AdvancedFilteringBenchmark, NestedObjectFiltering)
    ->Arg(1000)    // 1K vectors
    ->Arg(10000)   // 10K vectors
    ->Arg(100000)  // 100K vectors
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(AdvancedFilteringBenchmark, FullTextSearch)
    ->Arg(1000)    // 1K vectors
    ->Arg(10000)   // 10K vectors
    ->Arg(100000)  // 100K vectors
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(AdvancedFilteringBenchmark, FuzzyMatching)
    ->Arg(1000)    // 1K vectors
    ->Arg(10000)   // 10K vectors
    ->Arg(100000)  // 100K vectors
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(AdvancedFilteringBenchmark, CombinedAdvancedFiltering)
    ->Arg(1000)    // 1K vectors
    ->Arg(10000)   // 10K vectors
    ->Arg(100000)  // 100K vectors
    ->Unit(benchmark::kMillisecond);

// Additional specific benchmarks for performance validation

// Benchmark to validate that geospatial filtering meets performance requirements
static void BM_GeospatialFilteringPerformanceValidation(benchmark::State& state) {
    // This benchmark validates that geospatial filtering performs within specified limits
    // According to spec PB-009: "Geospatial filtering operations return results in under 50 milliseconds
    // for datasets up to 10 million vectors with 95% accuracy"
    
    size_t num_vectors = state.range(0);
    std::unique_ptr<MetadataFilter> filter = std::make_unique<MetadataFilter>();
    
    // Generate test vectors
    std::vector<Vector> test_vectors;
    test_vectors.reserve(num_vectors);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> coord_dist(-90.0f, 90.0f);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.values = std::vector<float>(128);
        
        // Add location metadata
        v.metadata.custom["location"] = std::to_string(coord_dist(gen)) + "," + std::to_string(coord_dist(gen));
        
        test_vectors.push_back(v);
    }
    
    // Create a geospatial query
    GeoQuery geo_query(GeospatialOperator::WITHIN_RADIUS, "metadata.custom.location");
    geo_query.center = Point(0.0, 0.0);
    geo_query.radius = 10000.0;  // 10km radius
    
    std::vector<GeoQuery> geo_queries = {geo_query};
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter->apply_geo_filters(geo_queries, test_vectors);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Validate performance requirement: under 50ms for 10M vectors
        // For smaller datasets, we'll scale the requirement proportionally
        double max_allowed_time_ms = 50.0 * (static_cast<double>(num_vectors) / 10000000.0);
        double actual_time_ms = duration.count() / 1000.0;
        
        if (actual_time_ms > max_allowed_time_ms) {
            state.SkipWithError("Geospatial filtering exceeded performance requirement");
        }
        
        if (result.has_value()) {
            state.SetIterationTime(actual_time_ms / 1000.0);
        } else {
            state.SkipWithError("Geospatial filtering failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

BENCHMARK(BM_GeospatialFilteringPerformanceValidation)
    ->Arg(100000)   // 100K vectors
    ->Arg(1000000)  // 1M vectors
    ->Unit(benchmark::kMillisecond);

// Benchmark to validate that temporal filtering meets performance requirements
static void BM_TemporalFilteringPerformanceValidation(benchmark::State& state) {
    // This benchmark validates that temporal filtering performs within specified limits
    // According to spec PB-009: "Temporal filtering operations return results in under 30 milliseconds
    // for datasets up to 10 million vectors with 95% accuracy"
    
    size_t num_vectors = state.range(0);
    std::unique_ptr<MetadataFilter> filter = std::make_unique<MetadataFilter>();
    
    // Generate test vectors
    std::vector<Vector> test_vectors;
    test_vectors.reserve(num_vectors);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> hour_dist(0, 23);
    std::uniform_int_distribution<int> minute_dist(0, 59);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.values = std::vector<float>(128);
        
        // Add timestamp metadata
        v.metadata.custom["timestamp"] = "2025-01-01T" + std::to_string(hour_dist(gen)) + ":" + 
                                      std::to_string(minute_dist(gen)) + ":00Z";
        
        test_vectors.push_back(v);
    }
    
    // Create a temporal query
    TemporalQuery temporal_query(TemporalOperator::AFTER, "metadata.custom.timestamp");
    std::tm tm_time = {};
    std::istringstream ss("2025-01-01T12:00:00Z");
    ss >> std::get_time(&tm_time, "%Y-%m-%dT%H:%M:%SZ");
    temporal_query.time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm_time));
    
    std::vector<TemporalQuery> temporal_queries = {temporal_query};
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter->apply_temporal_filters(temporal_queries, test_vectors);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Validate performance requirement: under 30ms for 10M vectors
        // For smaller datasets, we'll scale the requirement proportionally
        double max_allowed_time_ms = 30.0 * (static_cast<double>(num_vectors) / 10000000.0);
        double actual_time_ms = duration.count() / 1000.0;
        
        if (actual_time_ms > max_allowed_time_ms) {
            state.SkipWithError("Temporal filtering exceeded performance requirement");
        }
        
        if (result.has_value()) {
            state.SetIterationTime(actual_time_ms / 1000.0);
        } else {
            state.SkipWithError("Temporal filtering failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

BENCHMARK(BM_TemporalFilteringPerformanceValidation)
    ->Arg(100000)   // 100K vectors
    ->Arg(1000000)  // 1M vectors
    ->Unit(benchmark::kMillisecond);

// Benchmark to validate that nested object filtering meets performance requirements
static void BM_NestedObjectFilteringPerformanceValidation(benchmark::State& state) {
    // This benchmark validates that nested object filtering performs within specified limits
    // According to spec PB-009: "Nested object filtering operations return results in under 40 milliseconds
    // for datasets up to 10 million vectors with 95% accuracy"
    
    size_t num_vectors = state.range(0);
    std::unique_ptr<MetadataFilter> filter = std::make_unique<MetadataFilter>();
    
    // Generate test vectors
    std::vector<Vector> test_vectors;
    test_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.values = std::vector<float>(128);
        
        // Add nested metadata
        v.metadata.custom["nested_field"] = "value_" + std::to_string(i % 1000);
        
        test_vectors.push_back(v);
    }
    
    // Create a nested query
    NestedQuery nested_query("metadata.custom.nested_field", NestedOperator::EXISTS_PATH, "");
    
    std::vector<NestedQuery> nested_queries = {nested_query};
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter->apply_nested_filters(nested_queries, test_vectors);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Validate performance requirement: under 40ms for 10M vectors
        // For smaller datasets, we'll scale the requirement proportionally
        double max_allowed_time_ms = 40.0 * (static_cast<double>(num_vectors) / 10000000.0);
        double actual_time_ms = duration.count() / 1000.0;
        
        if (actual_time_ms > max_allowed_time_ms) {
            state.SkipWithError("Nested object filtering exceeded performance requirement");
        }
        
        if (result.has_value()) {
            state.SetIterationTime(actual_time_ms / 1000.0);
        } else {
            state.SkipWithError("Nested object filtering failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

BENCHMARK(BM_NestedObjectFilteringPerformanceValidation)
    ->Arg(100000)   // 100K vectors
    ->Arg(1000000)  // 1M vectors
    ->Unit(benchmark::kMillisecond);

// Benchmark to validate that full-text search meets performance requirements
static void BM_FullTextSearchPerformanceValidation(benchmark::State& state) {
    // This benchmark validates that full-text search performs within specified limits
    // According to spec PB-009: "Full-text search operations return results in under 100 milliseconds
    // for datasets up to 10 million vectors with 95% accuracy"
    
    size_t num_vectors = state.range(0);
    std::unique_ptr<MetadataFilter> filter = std::make_unique<MetadataFilter>();
    
    // Generate test vectors
    std::vector<Vector> test_vectors;
    test_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.values = std::vector<float>(128);
        
        // Add description metadata
        v.metadata.custom["description"] = "This is a sample document for testing full-text search capabilities with ID " + 
                                         std::to_string(i) + " and various textual content for performance validation";
        
        test_vectors.push_back(v);
    }
    
    // Create a full-text query
    FullTextQuery fulltext_query("metadata.custom.description", "sample document testing", FullTextOperator::MATCHES_ALL_TERMS);
    
    std::vector<FullTextQuery> fulltext_queries = {fulltext_query};
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter->apply_fulltext_filters(fulltext_queries, test_vectors);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Validate performance requirement: under 100ms for 10M vectors
        // For smaller datasets, we'll scale the requirement proportionally
        double max_allowed_time_ms = 100.0 * (static_cast<double>(num_vectors) / 10000000.0);
        double actual_time_ms = duration.count() / 1000.0;
        
        if (actual_time_ms > max_allowed_time_ms) {
            state.SkipWithError("Full-text search exceeded performance requirement");
        }
        
        if (result.has_value()) {
            state.SetIterationTime(actual_time_ms / 1000.0);
        } else {
            state.SkipWithError("Full-text search failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

BENCHMARK(BM_FullTextSearchPerformanceValidation)
    ->Arg(100000)   // 100K vectors
    ->Arg(1000000)  // 1M vectors
    ->Unit(benchmark::kMillisecond);

// Benchmark to validate that fuzzy matching meets performance requirements
static void BM_FuzzyMatchingPerformanceValidation(benchmark::State& state) {
    // This benchmark validates that fuzzy matching performs within specified limits
    // According to spec PB-009: "Fuzzy matching operations return results in under 150 milliseconds
    // for datasets up to 10 million vectors with 95% accuracy"
    
    size_t num_vectors = state.range(0);
    std::unique_ptr<MetadataFilter> filter = std::make_unique<MetadataFilter>();
    
    // Generate test vectors
    std::vector<Vector> test_vectors;
    test_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.values = std::vector<float>(128);
        
        // Add description metadata with slight variations
        v.metadata.custom["description"] = "This is a sampl document for testng fuzzy matching capabilities with ID " + 
                                         std::to_string(i) + " and various textual content for performance validation";
        
        test_vectors.push_back(v);
    }
    
    // Create a fuzzy matching query
    FullTextQuery fuzzy_query("metadata.custom.description", "sample document testing", FullTextOperator::FUZZY_MATCH);
    fuzzy_query.max_edit_distance = 3;
    
    std::vector<FullTextQuery> fuzzy_queries = {fuzzy_query};
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = filter->apply_fulltext_filters(fuzzy_queries, test_vectors);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Validate performance requirement: under 150ms for 10M vectors
        // For smaller datasets, we'll scale the requirement proportionally
        double max_allowed_time_ms = 150.0 * (static_cast<double>(num_vectors) / 10000000.0);
        double actual_time_ms = duration.count() / 1000.0;
        
        if (actual_time_ms > max_allowed_time_ms) {
            state.SkipWithError("Fuzzy matching exceeded performance requirement");
        }
        
        if (result.has_value()) {
            state.SetIterationTime(actual_time_ms / 1000.0);
        } else {
            state.SkipWithError("Fuzzy matching failed");
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

BENCHMARK(BM_FuzzyMatchingPerformanceValidation)
    ->Arg(100000)   // 100K vectors
    ->Arg(1000000)  // 1M vectors
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();