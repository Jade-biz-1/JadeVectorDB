#include "permission_cache.h"
#include <algorithm>

namespace jadevectordb {
namespace cache {

PermissionCache::PermissionCache(size_t max_size, int ttl_seconds)
    : max_size_(max_size)
    , ttl_(ttl_seconds)
    , hits_(0)
    , misses_(0)
    , evictions_(0) {
}

std::string PermissionCache::make_key(const std::string& user_id,
                                     const std::string& database_id,
                                     const std::string& permission) const {
    return user_id + ":" + database_id + ":" + permission;
}

std::optional<bool> PermissionCache::get(const std::string& user_id,
                                        const std::string& database_id,
                                        const std::string& permission) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string key = make_key(user_id, database_id, permission);
    auto it = cache_.find(key);
    
    if (it == cache_.end()) {
        ++misses_;
        return std::nullopt;
    }
    
    // Check if expired
    if (is_expired(it->second.first)) {
        // Remove expired entry
        lru_list_.erase(it->second.second);
        cache_.erase(it);
        ++misses_;
        return std::nullopt;
    }
    
    // Move to front of LRU list (most recently used)
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second.second);
    
    ++hits_;
    return it->second.first.granted;
}

void PermissionCache::put(const std::string& user_id,
                         const std::string& database_id,
                         const std::string& permission,
                         bool granted) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string key = make_key(user_id, database_id, permission);
    auto it = cache_.find(key);
    
    if (it != cache_.end()) {
        // Update existing entry and move to front
        it->second.first = CacheEntry(granted);
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second.second);
    } else {
        // Check if we need to evict
        if (cache_.size() >= max_size_) {
            evict_lru();
        }
        
        // Add new entry at front of LRU list
        lru_list_.push_front(key);
        auto list_it = lru_list_.begin();
        cache_.emplace(key, std::make_pair(CacheEntry(granted), list_it));
    }
}

void PermissionCache::invalidate_user(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string prefix = user_id + ":";
    
    auto it = cache_.begin();
    while (it != cache_.end()) {
        if (it->first.substr(0, prefix.length()) == prefix) {
            lru_list_.erase(it->second.second);
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void PermissionCache::invalidate_database(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.begin();
    while (it != cache_.end()) {
        // Key format: "user_id:database_id:permission"
        // Find second colon to extract database_id
        size_t first_colon = it->first.find(':');
        if (first_colon != std::string::npos) {
            size_t second_colon = it->first.find(':', first_colon + 1);
            if (second_colon != std::string::npos) {
                std::string db_id = it->first.substr(first_colon + 1, 
                                                     second_colon - first_colon - 1);
                if (db_id == database_id) {
                    lru_list_.erase(it->second.second);
                    it = cache_.erase(it);
                    continue;
                }
            }
        }
        ++it;
    }
}

void PermissionCache::invalidate(const std::string& user_id,
                                const std::string& database_id,
                                const std::string& permission) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string key = make_key(user_id, database_id, permission);
    auto it = cache_.find(key);
    
    if (it != cache_.end()) {
        lru_list_.erase(it->second.second);
        cache_.erase(it);
    }
}

void PermissionCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_list_.clear();
}

PermissionCache::CacheStats PermissionCache::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    CacheStats stats;
    stats.hits = hits_;
    stats.misses = misses_;
    stats.evictions = evictions_;
    stats.current_size = cache_.size();
    stats.max_size = max_size_;
    
    return stats;
}

void PermissionCache::cleanup_expired() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.begin();
    while (it != cache_.end()) {
        if (is_expired(it->second.first)) {
            lru_list_.erase(it->second.second);
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void PermissionCache::evict_lru() {
    // Assumes mutex is already held
    if (lru_list_.empty()) {
        return;
    }
    
    // Remove least recently used (back of list)
    std::string key = lru_list_.back();
    cache_.erase(key);
    lru_list_.pop_back();
    ++evictions_;
}

bool PermissionCache::is_expired(const CacheEntry& entry) const {
    auto now = std::chrono::steady_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::seconds>(now - entry.timestamp);
    return age >= ttl_;
}

} // namespace cache
} // namespace jadevectordb
