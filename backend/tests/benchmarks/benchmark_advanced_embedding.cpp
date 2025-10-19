#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <random>

#include "services/model_versioning_service.h"
#include "services/custom_model_training_service.h"

using namespace jadevectordb;

// Benchmark for model versioning creation
static void BM_ModelVersionCreation(benchmark::State& state) {
    ModelVersioningService service;
    
    for (auto _ : state) {
        ModelVersion version_info;
        version_info.version_id = "v1_0_0_" + std::to_string(state.iterations());
        version_info.model_id = "benchmark_model";
        version_info.version_number = "1.0.0";
        version_info.path_to_model = "/path/to/model_" + std::to_string(state.iterations()) + ".onnx";
        version_info.author = "benchmark";
        version_info.changelog = "Version for benchmarking";
        version_info.status = "inactive";
        
        service.create_model_version("benchmark_model", version_info);
    }
}
BENCHMARK(BM_ModelVersionCreation);

// Benchmark for A/B test model selection
static void BM_ABModelSelection(benchmark::State& state) {
    ModelVersioningService service;
    
    // Create an A/B test with two models
    ABTestConfig config;
    config.test_id = "benchmark_ab_test";
    config.model_ids = {"model_a", "model_b"};
    config.traffic_split = {0.5f, 0.5f};
    config.test_name = "Benchmark A/B Test";
    config.description = "For performance benchmarking";
    config.is_active = true;
    
    service.create_ab_test(config);
    service.start_ab_test("benchmark_ab_test");
    
    for (auto _ : state) {
        std::string selected = service.select_model_for_ab_test("benchmark_ab_test");
        benchmark::DoNotOptimize(selected);
    }
}
BENCHMARK(BM_ABModelSelection);

// Benchmark for model usage recording
static void BM_RecordModelUsage(benchmark::State& state) {
    ModelVersioningService service;
    
    // Create an A/B test
    ABTestConfig config;
    config.test_id = "benchmark_usage_test";
    config.model_ids = {"model_x", "model_y"};
    config.traffic_split = {0.5f, 0.5f};
    config.test_name = "Benchmark Usage Test";
    config.description = "For usage recording benchmarking";
    config.is_active = true;
    
    service.create_ab_test(config);
    service.start_ab_test("benchmark_usage_test");
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    std::vector<std::string> models = {"model_x", "model_y"};
    
    for (auto _ : state) {
        std::string model = models[dis(gen)];
        service.record_model_usage("benchmark_usage_test", model, 50.0, true);
    }
}
BENCHMARK(BM_RecordModelUsage);

// Benchmark for model version retrieval
static void BM_GetModelVersion(benchmark::State& state) {
    ModelVersioningService service;
    
    // Create a model version first
    ModelVersion version_info;
    version_info.version_id = "v1_0_0_benchmark";
    version_info.model_id = "benchmark_get_model";
    version_info.version_number = "1.0.0";
    version_info.path_to_model = "/path/to/benchmark_model.onnx";
    version_info.author = "benchmark";
    version_info.changelog = "Version for retrieval benchmarking";
    version_info.status = "active";
    
    service.create_model_version("benchmark_get_model", version_info);
    
    for (auto _ : state) {
        ModelVersion retrieved = service.get_model_version("benchmark_get_model", "1.0.0");
        benchmark::DoNotOptimize(retrieved);
    }
}
BENCHMARK(BM_GetModelVersion);

// Benchmark for listing model versions
static void BM_ListModelVersions(benchmark::State& state) {
    ModelVersioningService service;
    
    // Create multiple model versions
    for (int i = 0; i < 100; ++i) {
        ModelVersion version_info;
        version_info.version_id = "v1_0_" + std::to_string(i);
        version_info.model_id = "benchmark_list_model";
        version_info.version_number = "1.0." + std::to_string(i);
        version_info.path_to_model = "/path/to/benchmark_model_" + std::to_string(i) + ".onnx";
        version_info.author = "benchmark";
        version_info.changelog = "Version for listing benchmarking - " + std::to_string(i);
        version_info.status = "inactive";
        
        service.create_model_version("benchmark_list_model", version_info);
    }
    
    for (auto _ : state) {
        std::vector<ModelVersion> versions = service.list_model_versions("benchmark_list_model");
        benchmark::DoNotOptimize(versions.size());
    }
}
BENCHMARK(BM_ListModelVersions);

BENCHMARK_MAIN();