#ifndef JADEVECTORDB_IVF_INDEX_H
#define JADEVECTORDB_IVF_INDEX_H

#include "models/index.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <list>

namespace jadevectordb {

/**
 * @brief Implementation of Inverted File (IVF) index algorithm
 * 
 * This implementation provides efficient similarity search by clustering
 * vectors into partitions and only searching relevant partitions.
 */
class IvfIndex {
public:
    struct IvfParams {
        int num_clusters = 1000;          // Number of clusters/partitions
        int max_iterations = 100;         // Max iterations for k-means clustering
        int num_probes = 10;              // Number of clusters to probe during search
        float tolerance = 1e-4;           // Tolerance for k-means convergence
        bool use_product_quantization = false;  // Whether to use PQ for compression
        int pq_subvector_dimension = 16;  // Dimension of each subvector in PQ
        
        // Constructor
        IvfParams() = default;
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Index parameters
    IvfParams params_;
    
    // Data structures
    std::vector<std::vector<float>> centroids_;              // Cluster centroids
    std::unordered_map<int, std::vector<float>> vectors_;    // Vector ID -> vector data
    std::unordered_map<int, int> vector_cluster_map_;        // Vector ID -> cluster ID
    std::unordered_map<int, std::vector<int>> cluster_map_;  // Cluster ID -> vector IDs
    
    // For Product Quantization (if enabled)
    std::vector<std::vector<std::vector<float>>> pq_centroids_;  // PQ centroids
    std::unordered_map<int, std::vector<uint8_t>> pq_codes_;     // PQ codes for vectors
    
    // Current state
    int dimension_ = 0;
    bool is_trained_ = false;
    
    // Thread safety
    mutable std::shared_mutex index_mutex_;
    
public:
    explicit IvfIndex(const IvfParams& params = IvfParams{});
    ~IvfIndex() = default;
    
    // Initialize the index with parameters
    bool initialize(const IvfParams& params);
    
    // Add a vector to the index (will be clustered during build phase)
    Result<bool> add_vector(int vector_id, const std::vector<float>& vector);
    
    // Build the index (cluster vectors, compute centroids)
    Result<bool> build();
    
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
    
    // Get the number of clusters
    int get_num_clusters() const;
    
    // Rebuild the index after updates
    Result<bool> rebuild();

private:
    // Internal implementation methods
    
    // Perform k-means clustering to build centroids
    Result<bool> perform_kmeans_clustering();
    
    // Assign vectors to clusters
    Result<bool> assign_vectors_to_clusters();
    
    // Compute squared distance between two vectors
    float compute_squared_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    
    // Find nearest centroids to a vector
    std::vector<std::pair<float, int>> find_nearest_centroids(const std::vector<float>& query, int n) const;
    
    // Apply Product Quantization to vectors
    Result<bool> apply_product_quantization();
    
    // Encode a vector using Product Quantization
    std::vector<uint8_t> encode_vector_pq(const std::vector<float>& vector) const;
    
    // Decode a PQ-encoded vector
    std::vector<float> decode_vector_pq(const std::vector<uint8_t>& codes) const;
    
    // Compute distance using PQ approximation
    float compute_pq_distance(const std::vector<uint8_t>& code1, 
                             const std::vector<uint8_t>& code2) const;
    
    // Validate index state
    bool validate() const;
    
    // Update cluster assignments after adding/removing vectors
    Result<bool> update_cluster_assignments();
    
    // Optimize clustering after changes
    Result<bool> optimize_clustering();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_IVF_INDEX_H