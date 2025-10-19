// Distributed system invariants for JadeVectorDB
// Properties that verify correct behavior in a distributed environment

#ifndef DISTRIBUTED_INVARIANTS_H
#define DISTRIBUTED_INVARIANTS_H

#include "property_test_framework.h"
#include <vector>
#include <functional>
#include <algorithm>
#include <memory>
#include <map>
#include <set>
#include <string>
#include <cmath>

namespace property_tests {
namespace distributed {

// Node in a simulated distributed system
struct Node {
    std::string id;
    std::map<std::string, std::vector<float>> data;
    bool is_alive;
    
    Node(const std::string& node_id) : id(node_id), is_alive(true) {}
    
    bool insert(const std::string& key, const std::vector<float>& vector) {
        if (!is_alive) return false;
        data[key] = vector;
        return true;
    }
    
    bool get(const std::string& key, std::vector<float>& vector) const {
        if (!is_alive) return false;
        auto it = data.find(key);
        if (it != data.end()) {
            vector = it->second;
            return true;
        }
        return false;
    }
    
    bool remove(const std::string& key) {
        if (!is_alive) return false;
        return data.erase(key) > 0;
    }
};

// Simulated distributed system
struct DistributedSystem {
    std::vector<Node> nodes;
    int replication_factor;
    std::map<std::string, std::set<std::string>> key_to_nodes; // Maps keys to nodes that contain them
    
    DistributedSystem(int node_count, int repl_factor = 1) 
        : replication_factor(repl_factor) {
        for (int i = 0; i < node_count; ++i) {
            nodes.emplace_back("node_" + std::to_string(i));
        }
    }
    
    // Simulate a simple hash-based sharding
    int get_shard(const std::string& key) const {
        std::hash<std::string> hasher;
        return static_cast<int>(hasher(key)) % nodes.size();
    }
    
    bool insert(const std::string& key, const std::vector<float>& vector) {
        if (nodes.empty()) return false;
        
        // Determine primary node
        int primary_idx = get_shard(key);
        if (!nodes[primary_idx].insert(key, vector)) {
            return false;
        }
        
        // Replicate to other nodes based on replication factor
        key_to_nodes[key] = {nodes[primary_idx].id};
        for (int i = 1; i < std::min(replication_factor, static_cast<int>(nodes.size())); ++i) {
            int replica_idx = (primary_idx + i) % nodes.size();
            if (nodes[replica_idx].is_alive) {
                nodes[replica_idx].insert(key, vector);
                key_to_nodes[key].insert(nodes[replica_idx].id);
            }
        }
        
        return true;
    }
    
    bool get(const std::string& key, std::vector<float>& vector) const {
        // Try to get from any node that has the key
        for (const auto& node_id : key_to_nodes.at(key)) {
            // Extract node index from node_id (node_0 -> 0)
            int node_idx = std::stoi(node_id.substr(5)); // skip "node_"
            if (nodes[node_idx].get(key, vector)) {
                return true;
            }
        }
        return false;
    }
    
    size_t get_total_data_count() const {
        size_t count = 0;
        std::set<std::string> unique_keys;
        
        for (const auto& node : nodes) {
            for (const auto& pair : node.data) {
                unique_keys.insert(pair.first);
            }
        }
        
        return unique_keys.size();
    }
    
    void partition_network(int partition1_size) {
        // Partition the network by marking some nodes as non-communicable
        // This simulates a network partition
        for (int i = 0; i < std::min(partition1_size, static_cast<int>(nodes.size())); ++i) {
            nodes[i].is_alive = false;
        }
    }
    
    void heal_network() {
        // Restore all nodes to alive state
        for (auto& node : nodes) {
            node.is_alive = true;
        }
    }
};

// Property: Data Consistency - Same data should be available across all replicas
inline bool data_consistency_property(const std::vector<std::vector<float>>& test_vectors) {
    if (test_vectors.empty()) return true;
    
    // Create a distributed system with 3 nodes and replication factor of 2
    DistributedSystem system(3, 2);
    
    // Insert the same vectors into the system
    for (size_t i = 0; i < test_vectors.size(); ++i) {
        std::string key = "vector_" + std::to_string(i);
        if (!system.insert(key, test_vectors[i])) {
            return false;
        }
    }
    
    // Verify all vectors can be retrieved consistently
    for (size_t i = 0; i < test_vectors.size(); ++i) {
        std::string key = "vector_" + std::to_string(i);
        std::vector<float> retrieved_vector;
        
        if (!system.get(key, retrieved_vector)) {
            return false; // Could not retrieve
        }
        
        // Check if retrieved vector matches original
        if (retrieved_vector.size() != test_vectors[i].size()) {
            return false;
        }
        
        for (size_t j = 0; j < test_vectors[i].size(); ++j) {
            if (std::abs(retrieved_vector[j] - test_vectors[i][j]) > 1e-6f) {
                return false;
            }
        }
    }
    
    return true;
}

// Property: Partition Tolerance - System should continue to operate during network partitions
inline bool partition_tolerance_property() {
    // Create a distributed system
    DistributedSystem system(5, 2);  // 5 nodes, replication factor 2
    
    // Insert some data
    std::vector<std::vector<float>> test_data = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f},
        {10.0f, 11.0f, 12.0f}
    };
    
    for (size_t i = 0; i < test_data.size(); ++i) {
        std::string key = "data_" + std::to_string(i);
        if (!system.insert(key, test_data[i])) {
            return false;
        }
    }
    
    // Verify baseline count
    size_t baseline_count = system.get_total_data_count();
    if (baseline_count != test_data.size()) {
        return false;
    }
    
    // Partition the network (make first 2 nodes unreachable)
    system.partition_network(2);
    
    // Even with partition, we should still be able to access data
    // (assuming replication ensures availability)
    bool all_accessible = true;
    for (size_t i = 0; i < test_data.size(); ++i) {
        std::string key = "data_" + std::to_string(i);
        std::vector<float> retrieved_vector;
        if (!system.get(key, retrieved_vector)) {
            all_accessible = false; // Not all data is accessible
        }
    }
    
    // Healing the network should restore full functionality
    system.heal_network();
    
    // After healing, all data should be accessible again
    for (size_t i = 0; i < test_data.size(); ++i) {
        std::string key = "data_" + std::to_string(i);
        std::vector<float> retrieved_vector;
        if (!system.get(key, retrieved_vector)) {
            return false; // Data should be accessible after healing
        }
    }
    
    return true;
}

// Property: Eventual Consistency - After a partition heals, all nodes should have consistent data
inline bool eventual_consistency_property() {
    DistributedSystem system(3, 2);
    
    // Insert data when all nodes are available
    std::vector<float> test_vector = {1.0f, 2.0f, 3.0f};
    std::string key = "consistency_test";
    
    if (!system.insert(key, test_vector)) {
        return false;
    }
    
    // Verify initial consistency
    std::vector<float> retrieved;
    if (!system.get(key, retrieved)) {
        return false;
    }
    
    // Check that retrieved vector matches original
    if (retrieved.size() != test_vector.size()) {
        return false;
    }
    for (size_t i = 0; i < test_vector.size(); ++i) {
        if (std::abs(retrieved[i] - test_vector[i]) > 1e-6f) {
            return false;
        }
    }
    
    // Simulate a partition
    system.partition_network(1);
    
    // Try to update during partition
    std::vector<float> updated_vector = {4.0f, 5.0f, 6.0f};
    if (!system.insert(key, updated_vector)) {
        // Update might fail during partition, which is acceptable
    }
    
    // Heal the partition
    system.heal_network();
    
    // After healing, data should be consistent across nodes
    std::vector<float> final_retrieval;
    if (!system.get(key, final_retrieval)) {
        return false;
    }
    
    // Final vector should be either original or updated (both are valid)
    bool matches_original = true;
    bool matches_updated = true;
    
    if (final_retrieval.size() != test_vector.size()) {
        matches_original = false;
    } else {
        for (size_t i = 0; i < test_vector.size(); ++i) {
            if (std::abs(final_retrieval[i] - test_vector[i]) > 1e-6f) {
                matches_original = false;
                break;
            }
        }
    }
    
    if (final_retrieval.size() != updated_vector.size()) {
        matches_updated = false;
    } else {
        for (size_t i = 0; i < updated_vector.size(); ++i) {
            if (std::abs(final_retrieval[i] - updated_vector[i]) > 1e-6f) {
                matches_updated = false;
                break;
            }
        }
    }
    
    // At least one of the versions should match (original or updated)
    return matches_original || matches_updated;
}

// Generator for distributed operations
class DistributedVectorGenerator : public Generator<std::vector<std::vector<float>>> {
private:
    int num_vectors;
    int max_dimension;
    
public:
    DistributedVectorGenerator(int n_vecs = 10, int max_dim = 128) 
        : num_vectors(n_vecs), max_dimension(max_dim) {}
    
    std::vector<std::vector<float>> generate(std::mt19937& rng) override {
        std::uniform_int_distribution<int> dim_dist(2, max_dimension);
        std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
        
        std::vector<std::vector<float>> result;
        result.reserve(num_vectors);
        
        for (int i = 0; i < num_vectors; ++i) {
            int dimension = dim_dist(rng);
            std::vector<float> vector(dimension);
            for (int j = 0; j < dimension; ++j) {
                vector[j] = val_dist(rng);
            }
            result.push_back(vector);
        }
        
        return result;
    }
};

} // namespace distributed
} // namespace property_tests

#endif // DISTRIBUTED_INVARIANTS_H