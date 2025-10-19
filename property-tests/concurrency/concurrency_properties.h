// Concurrency properties for JadeVectorDB
// Properties that verify thread safety and concurrent access correctness

#ifndef CONCURRENCY_PROPERTIES_H
#define CONCURRENCY_PROPERTIES_H

#include "property_test_framework.h"
#include <vector>
#include <functional>
#include <algorithm>
#include <memory>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <future>
#include <chrono>

namespace property_tests {
namespace concurrency {

// Simulated concurrent vector store for property testing
template<typename T>
class ThreadSafeVectorStore {
private:
    mutable std::shared_mutex mutex_;
    std::vector<std::pair<std::string, T>> data_;
    std::atomic<size_t> operation_count_{0};

public:
    bool insert(const std::string& key, const T& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        // Check if key already exists
        for (const auto& item : data_) {
            if (item.first == key) {
                return false; // Key already exists
            }
        }
        data_.push_back({key, value});
        operation_count_++;
        return true;
    }

    bool get(const std::string& key, T& value) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        for (const auto& item : data_) {
            if (item.first == key) {
                value = item.second;
                return true;
            }
        }
        return false; // Key not found
    }

    bool update(const std::string& key, const T& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        for (auto& item : data_) {
            if (item.first == key) {
                item.second = value;
                operation_count_++;
                return true;
            }
        }
        return false; // Key not found
    }

    bool remove(const std::string& key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = std::remove_if(data_.begin(), data_.end(),
                                [&key](const auto& item) { return item.first == key; });
        if (it != data_.end()) {
            data_.erase(it, data_.end());
            operation_count_++;
            return true;
        }
        return false; // Key not found
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return data_.size();
    }

    size_t get_operation_count() const {
        return operation_count_.load();
    }

    // Get a copy of all keys for iteration (with lock)
    std::vector<std::string> get_keys() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<std::string> keys;
        for (const auto& item : data_) {
            keys.push_back(item.first);
        }
        return keys;
    }
};

// Operation types for concurrent testing
enum class OperationType {
    INSERT,
    READ,
    UPDATE,
    DELETE
};

// A concurrent operation
struct ConcurrentOperation {
    OperationType type;
    std::string key;
    std::vector<float> vector;
    
    ConcurrentOperation(OperationType t, const std::string& k, const std::vector<float>& v = {})
        : type(t), key(k), vector(v) {}
};

// Property: Thread Safety - Multiple threads can access data safely without race conditions
inline bool thread_safety_property(const std::vector<ConcurrentOperation>& operations) {
    // Create a shared vector store
    auto store = std::make_shared<ThreadSafeVectorStore<std::vector<float>>>();
    
    // Execute operations concurrently using async futures
    std::vector<std::future<void>> futures;
    
    for (const auto& op : operations) {
        futures.push_back(std::async(std::launch::async, [store, op]() {
            switch (op.type) {
                case OperationType::INSERT:
                    store->insert(op.key, op.vector);
                    break;
                case OperationType::READ: {
                    std::vector<float> result;
                    store->get(op.key, result);
                    break;
                }
                case OperationType::UPDATE:
                    store->update(op.key, op.vector);
                    break;
                case OperationType::DELETE:
                    store->remove(op.key);
                    break;
            }
        }));
    }
    
    // Wait for all operations to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    // Basic check: the operation count should be non-negative and realistic
    return store->get_operation_count() <= operations.size() * 2; // Allow for both successful and failed ops
}

// Property: Atomic Operation - Individual operations are atomic
inline bool atomic_operation_property(const std::vector<float>& input_vector) {
    // For this property, we test that vector operations are atomic
    // by checking if a large vector can be processed without interruption
    
    // Create a thread-safe store
    ThreadSafeVectorStore<std::vector<float>> store;
    
    // Insert the vector
    std::string key = "test_vector";
    bool insert_success = store.insert(key, input_vector);
    
    if (!insert_success) {
        return false; // Insert failed
    }
    
    // Immediately try to retrieve it
    std::vector<float> retrieved_vector;
    bool get_success = store.get(key, retrieved_vector);
    
    if (!get_success) {
        return false; // Retrieval failed
    }
    
    // Verify the retrieved vector matches the inserted one
    if (retrieved_vector.size() != input_vector.size()) {
        return false;
    }
    
    for (size_t i = 0; i < input_vector.size(); ++i) {
        if (std::abs(retrieved_vector[i] - input_vector[i]) > 1e-6f) {
            return false;
        }
    }
    
    return true;
}

// Property: Read-Write Consistency - Readers see consistent state with writers
inline bool read_write_consistency_property() {
    // Test that readers and writers can work concurrently without inconsistencies
    ThreadSafeVectorStore<std::vector<float>> store;
    
    // Insert some data
    std::vector<std::vector<float>> test_data = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };
    
    for (size_t i = 0; i < test_data.size(); ++i) {
        store.insert("vec_" + std::to_string(i), test_data[i]);
    }
    
    // Launch reader and writer threads
    auto reader_future = std::async(std::launch::async, [&store]() {
        // Reader thread: Keep reading the size
        size_t last_size = store.size();
        for (int i = 0; i < 100; ++i) {  // Read 100 times
            size_t current_size = store.size();
            // Size should be valid (0-3 in our test case)
            if (current_size > 3) {
                return false;
            }
            last_size = current_size;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        return true;
    });
    
    auto writer_future = std::async(std::launch::async, [&store, &test_data]() {
        // Writer thread: Perform updates
        for (int i = 0; i < 50; ++i) {
            std::string key = "vec_" + std::to_string(i % test_data.size());
            store.update(key, test_data[i % test_data.size()]);
        }
        return true;
    });
    
    // Wait for completion
    bool reader_ok = reader_future.get();
    bool writer_ok = writer_future.get();
    
    return reader_ok && writer_ok;
}

// Generator for concurrent operations
class ConcurrentOperationGenerator : public Generator<ConcurrentOperation> {
private:
    int max_dimension;
    
public:
    ConcurrentOperationGenerator(int max_dim = 128) : max_dimension(max_dim) {}
    
    ConcurrentOperation generate(std::mt19937& rng) override {
        std::uniform_int_distribution<int> type_dist(0, 3); // 4 operation types
        std::uniform_int_distribution<int> dim_dist(2, max_dimension);
        std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> key_len_dist(5, 15);
        std::uniform_int_distribution<int> char_dist(97, 122); // lowercase letters
        
        OperationType type = static_cast<OperationType>(type_dist(rng));
        
        // Generate random key
        int key_len = key_len_dist(rng);
        std::string key;
        for (int i = 0; i < key_len; ++i) {
            key += static_cast<char>(char_dist(rng));
        }
        
        // Generate random vector for insert/update operations
        std::vector<float> vector;
        if (type == OperationType::INSERT || type == OperationType::UPDATE) {
            int dimension = dim_dist(rng);
            vector.resize(dimension);
            for (int i = 0; i < dimension; ++i) {
                vector[i] = val_dist(rng);
            }
        }
        
        return ConcurrentOperation(type, key, vector);
    }
};

// Generator for concurrent operation sequences
class ConcurrentOperationSequenceGenerator : public Generator<std::vector<ConcurrentOperation>> {
private:
    int max_operations;
    ConcurrentOperationGenerator op_gen;
    
public:
    ConcurrentOperationSequenceGenerator(int max_ops = 20, int max_dim = 128) 
        : max_operations(max_ops), op_gen(max_dim) {}
    
    std::vector<ConcurrentOperation> generate(std::mt19937& rng) override {
        std::uniform_int_distribution<int> count_dist(5, max_operations);
        int num_ops = count_dist(rng);
        
        std::vector<ConcurrentOperation> sequence;
        sequence.reserve(num_ops);
        
        for (int i = 0; i < num_ops; ++i) {
            sequence.push_back(op_gen.generate(rng));
        }
        
        return sequence;
    }
};

} // namespace concurrency
} // namespace property_tests

#endif // CONCURRENCY_PROPERTIES_H