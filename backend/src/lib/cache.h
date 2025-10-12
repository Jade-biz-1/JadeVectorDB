#ifndef JADEVECTORDB_EMBEDDING_CACHE_H
#define JADEVECTORDB_EMBEDDING_CACHE_H

#include "models/embedding_model.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>
#include <list>

namespace jadevectordb {

/**
 * @brief Cache for storing frequently generated embeddings
 * 
 * This cache stores embeddings that have been previously generated to avoid
 * recomputation of the same inputs, improving performance for repeated queries.
 */
class EmbeddingCache {
private:
    struct CacheEntry {
        std::vector<float> embedding;
        std::chrono::steady_clock::time_point timestamp;
        size_t access_count;
        
        CacheEntry(const std::vector<float>& emb) 
            : embedding(emb), timestamp(std::chrono::steady_clock::now()), access_count(1) {}
    };
    
    std::unordered_map<std::string, CacheEntry> cache_;
    mutable std::shared_mutex cache_mutex_;
    
    // For LRU implementation
    std::list<std::string> lru_list_;
    std::unordered_map<std::string, decltype(lru_list_)::iterator> lru_map_;
    
    size_t max_size_;
    std::chrono::seconds ttl_seconds_;
    
public:
    explicit EmbeddingCache(size_t max_size = 10000, 
                          std::chrono::seconds ttl = std::chrono::seconds(3600)); // 1 hour default TTL
    ~EmbeddingCache() = default;
    
    // Get cached embedding by input hash
    Result<std::vector<float>> get(const std::string& input_hash);
    
    // Put embedding in cache
    bool put(const std::string& input_hash, const std::vector<float>& embedding);
    
    // Check if input is in cache
    bool contains(const std::string& input_hash) const;
    
    // Remove specific entry from cache
    bool remove(const std::string& input_hash);
    
    // Clear the entire cache
    void clear();
    
    // Get current cache size
    size_t size() const;
    
    // Get cache capacity
    size_t capacity() const { return max_size_; }
    
    // Evict expired entries
    void evict_expired();
    
    // Get cache hit ratio
    double hit_ratio() const;
    
private:
    // Generate hash for input
    std::string generate_input_hash(const std::string& input) const;
    
    // Update LRU position
    void update_lru(const std::string& input_hash);
    
    // Remove least recently used entry
    void evict_lru();
    
    // Check if entry is expired
    bool is_expired(const CacheEntry& entry) const;
};

/**
 * @brief Cache for storing loaded embedding models
 * 
 * This cache stores loaded embedding models in memory to avoid repeated
 * loading/unloading of models, improving performance for repeated operations.
 */
class ModelCache {
private:
    struct ModelEntry {
        std::shared_ptr<IEmbeddingProvider> model;
        std::chrono::steady_clock::time_point timestamp;
        size_t access_count;
        
        ModelEntry(std::shared_ptr<IEmbeddingProvider> m) 
            : model(m), timestamp(std::chrono::steady_clock::now()), access_count(1) {}
    };
    
    std::unordered_map<std::string, ModelEntry> cache_;
    mutable std::shared_mutex cache_mutex_;
    
    // For LRU implementation
    std::list<std::string> lru_list_;
    std::unordered_map<std::string, decltype(lru_list_)::iterator> lru_map_;
    
    size_t max_size_;
    std::chrono::seconds ttl_seconds_;
    
public:
    explicit ModelCache(size_t max_size = 100, 
                      std::chrono::seconds ttl = std::chrono::seconds(7200)); // 2 hour default TTL
    ~ModelCache() = default;
    
    // Get cached model by model ID
    Result<std::shared_ptr<IEmbeddingProvider>> get(const std::string& model_id);
    
    // Put model in cache
    bool put(const std::string& model_id, std::shared_ptr<IEmbeddingProvider> model);
    
    // Check if model is in cache
    bool contains(const std::string& model_id) const;
    
    // Remove specific model from cache
    bool remove(const std::string& model_id);
    
    // Clear the entire cache
    void clear();
    
    // Get current cache size
    size_t size() const;
    
    // Get cache capacity
    size_t capacity() const { return max_size_; }
    
    // Evict expired entries
    void evict_expired();
    
    // Get cache hit ratio
    double hit_ratio() const;
    
private:
    // Update LRU position
    void update_lru(const std::string& model_id);
    
    // Remove least recently used entry
    void evict_lru();
    
    // Check if entry is expired
    bool is_expired(const ModelEntry& entry) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_EMBEDDING_CACHE_H