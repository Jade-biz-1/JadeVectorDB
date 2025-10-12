#ifndef JADEVECTORDB_HNSW_INDEX_H
#define JADEVECTORDB_HNSW_INDEX_H

#include "models/index.h"
#include "services/similarity_search.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <random>

namespace jadevectordb {

struct HnswNode {
    int id;
    std::vector<float> vector;  // The actual vector data
    std::vector<std::vector<int>> neighbors;  // neighbors[layer][neighbor_index]
    int max_level;
    std::mt19937 rng;
    
    HnswNode(int node_id, const std::vector<float>& vec, int max_lv)
        : id(node_id), vector(vec), max_level(max_lv), rng(node_id) {
        neighbors.resize(max_lv + 1);
    }
};

/**
 * @brief Implementation of Hierarchical Navigable Small World (HNSW) index algorithm
 * 
 * This implementation provides efficient similarity search with good balance
 * between accuracy and speed for high-dimensional vector spaces.
 */
class HnswIndex {
public:
    struct HnswParams {
        int max_elements = 1000000;  // Maximum number of elements
        int M = 16;                  // Max number of connections per element
        int ef_construction = 200;   // Construction parameter
        int ef_search = 64;          // Search parameter
        int random_seed = 100;
        float level_mult = 1.0 / log(1.0 * M);  // Multiplier for level generation
        
        // Constructor
        HnswParams() = default;
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Index parameters
    HnswParams params_;
    
    // Data structures
    std::vector<std::unique_ptr<HnswNode>> nodes_;
    std::unordered_map<int, int> id_to_idx_;  // Maps vector ID to internal index
    std::vector<int> element_levels_;         // Level of each element
    std::vector<std::mutex> link_locks_;      // Mutex for each element's links
    
    // Current state
    int cur_element_count_ = 0;
    int max_level_ = 0;
    int entry_point_ = -1;
    
    // Random generator for level assignment
    std::mt19937 level_generator_;
    std::uniform_real_distribution<float> uniform_distribution_;
    
    // Thread safety
    mutable std::shared_mutex index_mutex_;
    
public:
    explicit HnswIndex(const HnswParams& params = HnswParams{});
    ~HnswIndex() = default;
    
    // Initialize the index with parameters
    bool initialize(const HnswParams& params);
    
    // Add a vector to the index
    Result<bool> add_vector(int vector_id, const std::vector<float>& vector);
    
    // Search for similar vectors
    Result<std::vector<std::pair<int, float>>> search(const std::vector<float>& query,
                                                     int k = 10,
                                                     float threshold = 0.0f) const;
    
    // Build the index from a set of vectors (batch operation)
    Result<bool> build_from_vectors(const std::vector<std::pair<int, std::vector<float>>>& vectors);
    
    // Check if the index contains a specific vector
    bool contains(int vector_id) const;
    
    // Remove a vector from the index
    Result<bool> remove_vector(int vector_id);
    
    // Update a vector in the index
    Result<bool> update_vector(int vector_id, const std::vector<float>& new_vector);
    
    // Get the number of vectors in the index
    size_t size() const;
    
    // Get index statistics
    Result<std::unordered_map<std::string, std::string>> get_stats() const;
    
    // Clear the index
    void clear();
    
    // Check if the index is empty
    bool empty() const;
    
    // Get the dimension of vectors in the index
    int get_dimension() const;
    
    // Get the current number of levels in the index
    int get_current_levels() const;

private:
    // Internal implementation methods
    
    // Generate random level for a new element
    int getRandomLevel();
    
    // Find the appropriate insertion point in a specific level
    int searchLevel(const std::vector<float>& query, int enterpoint, int level) const;
    
    // Perform greedy search in a level
    int greedySearch(const std::vector<float>& query, int enterpoint, int level) const;
    
    // Connect nodes during insertion
    void connectNewElement(int new_cur_c, int cur_c, int level, bool isUpdate);
    
    // Calculate distance between two vectors
    float calculateDistance(const std::vector<float>& a, const std::vector<float>& b) const;
    
    // Find the most similar neighbors to a vector
    std::vector<std::pair<float, int>> getNeighborsByHeuristic(
        const std::vector<std::pair<float, int>>& candidates, int max_size) const;
    
    // Link nodes at a specific level
    void link(int src_id, int dest_id, int level);
    
    // Remove links to a specific node
    void removeLink(int node_idx, int target_idx, int level);
    
    // Validate index state
    bool validate() const;
    
    // Optimize the index structure
    void optimize();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_HNSW_INDEX_H