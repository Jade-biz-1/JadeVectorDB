#include "memory_mapped_vector_store.h"
#include <filesystem>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace jadevectordb {

namespace {
    constexpr uint32_t MAGIC_NUMBER = 0x4A564442;  // "JVDB"
    constexpr uint32_t FORMAT_VERSION = 1;
    constexpr size_t HEADER_SIZE = 128;  // Increased for vector_ids_offset field
    constexpr size_t INDEX_ENTRY_SIZE = 64;  // Increased for string_offset field
    constexpr size_t ALIGNMENT = 32;  // AVX alignment
    
    uint64_t get_timestamp() {
        return std::chrono::system_clock::now().time_since_epoch().count();
    }
}

// ============================================================================
// FileHandle Implementation
// ============================================================================

FileHandle::~FileHandle() {
    close();
}

#ifdef _WIN32
bool FileHandle::open(const std::string& path, bool create) {
    path_ = path;
    DWORD access = GENERIC_READ | GENERIC_WRITE;
    DWORD share = FILE_SHARE_READ;
    DWORD disposition = create ? CREATE_ALWAYS : OPEN_EXISTING;
    
    handle_ = CreateFileA(path.c_str(), access, share, nullptr,
                         disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    return handle_ != INVALID_HANDLE_VALUE;
}

void FileHandle::close() {
    if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(handle_);
        handle_ = nullptr;
    }
}

bool FileHandle::is_open() const {
    return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
}

bool FileHandle::resize(size_t new_size) {
    if (!is_open()) return false;
    
    LARGE_INTEGER li;
    li.QuadPart = new_size;
    
    if (!SetFilePointerEx(handle_, li, nullptr, FILE_BEGIN)) {
        return false;
    }
    
    return SetEndOfFile(handle_);
}

size_t FileHandle::get_size() const {
    if (!is_open()) return 0;
    
    LARGE_INTEGER size;
    if (!GetFileSizeEx(handle_, &size)) {
        return 0;
    }
    
    return static_cast<size_t>(size.QuadPart);
}

#else  // Unix/Linux

bool FileHandle::open(const std::string& path, bool create) {
    path_ = path;
    int flags = O_RDWR;
    if (create) {
        flags |= O_CREAT | O_TRUNC;
    }
    
    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    fd_ = ::open(path.c_str(), flags, mode);
    
    return fd_ != -1;
}

void FileHandle::close() {
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool FileHandle::is_open() const {
    return fd_ != -1;
}

bool FileHandle::resize(size_t new_size) {
    if (!is_open()) return false;
    return ftruncate(fd_, new_size) == 0;
}

size_t FileHandle::get_size() const {
    if (!is_open()) return 0;
    
    struct stat st;
    if (fstat(fd_, &st) != 0) {
        return 0;
    }
    
    return static_cast<size_t>(st.st_size);
}

#endif

// ============================================================================
// MemoryMappedVectorStore Implementation
// ============================================================================

MemoryMappedVectorStore::MemoryMappedVectorStore(const std::string& storage_path)
    : storage_path_(storage_path) {
    // Create storage directory if it doesn't exist
    std::filesystem::create_directories(storage_path_);
}

MemoryMappedVectorStore::~MemoryMappedVectorStore() {
    // Close all open files with flush
    std::lock_guard<std::mutex> lock(files_mutex_);
    for (auto& [db_id, db_file] : open_files_) {
        if (db_file && db_file->mapped_memory) {
            sync_file(db_file->mapped_memory, db_file->mapped_size, true);
            unmap_file(db_file->mapped_memory, db_file->mapped_size);
        }
    }
    open_files_.clear();
}

std::string MemoryMappedVectorStore::get_database_directory(const std::string& database_id) const {
    return storage_path_ + "/" + database_id;
}

std::string MemoryMappedVectorStore::get_vector_file_path(const std::string& database_id) const {
    return get_database_directory(database_id) + "/vectors.jvdb";
}

bool MemoryMappedVectorStore::create_vector_file(const std::string& database_id,
                                                 int dimension,
                                                 size_t initial_capacity) {
    // Create database directory
    std::string db_dir = get_database_directory(database_id);
    if (!std::filesystem::create_directories(db_dir)) {
        if (!std::filesystem::exists(db_dir)) {
            return false;
        }
    }
    
    // Create file
    std::string file_path = get_vector_file_path(database_id);
    auto handle = std::make_unique<FileHandle>();
    if (!handle->open(file_path, true)) {
        return false;
    }
    
    // Calculate initial file size
    size_t index_size = initial_capacity * INDEX_ENTRY_SIZE;
    size_t data_size = initial_capacity * dimension * sizeof(float);
    size_t string_size = initial_capacity * 64;  // 64 bytes per vector ID string
    size_t total_size = HEADER_SIZE + index_size + data_size + string_size;
    
    // Resize file
    if (!handle->resize(total_size)) {
        return false;
    }
    
    // Map file
    void* mapped = map_file(handle.get(), total_size);
    if (!mapped) {
        return false;
    }
    
    // Initialize header
    auto* header = reinterpret_cast<VectorFileHeader*>(mapped);
    std::memset(header, 0, HEADER_SIZE);
    header->magic_number = MAGIC_NUMBER;
    header->version = FORMAT_VERSION;
    header->dimension = dimension;
    header->vector_count = 0;
    header->active_count = 0;
    header->index_offset = HEADER_SIZE;
    header->data_offset = HEADER_SIZE + index_size;
    header->vector_ids_offset = HEADER_SIZE + index_size + data_size;
    header->index_capacity = initial_capacity;
    header->data_capacity = data_size;
    
    // Zero out index, data, and string sections
    std::memset(static_cast<char*>(mapped) + HEADER_SIZE, 0, index_size + data_size + string_size);
    
    // Flush to disk
    sync_file(mapped, total_size, true);
    
    // Store in open files
    auto db_file = std::make_unique<DatabaseFile>();
    db_file->handle = std::move(handle);
    db_file->mapped_memory = mapped;
    db_file->file_size = total_size;
    db_file->mapped_size = total_size;
    db_file->header = header;
    db_file->last_access_time = get_timestamp();
    
    std::lock_guard<std::mutex> lock(files_mutex_);
    open_files_[database_id] = std::move(db_file);
    
    return true;
}

bool MemoryMappedVectorStore::open_vector_file(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(files_mutex_);
    
    // Check if already open
    if (open_files_.find(database_id) != open_files_.end()) {
        return true;
    }
    
    // Open file
    std::string file_path = get_vector_file_path(database_id);
    if (!std::filesystem::exists(file_path)) {
        return false;
    }
    
    auto handle = std::make_unique<FileHandle>();
    if (!handle->open(file_path, false)) {
        return false;
    }
    
    size_t file_size = handle->get_size();
    if (file_size < HEADER_SIZE) {
        return false;
    }
    
    // Map file
    void* mapped = map_file(handle.get(), file_size);
    if (!mapped) {
        return false;
    }
    
    // Verify header
    auto* header = reinterpret_cast<VectorFileHeader*>(mapped);
    if (header->magic_number != MAGIC_NUMBER) {
        unmap_file(mapped, file_size);
        return false;
    }
    
    if (header->version != FORMAT_VERSION) {
        unmap_file(mapped, file_size);
        return false;
    }
    
    // Store in open files
    auto db_file = std::make_unique<DatabaseFile>();
    db_file->handle = std::move(handle);
    db_file->mapped_memory = mapped;
    db_file->file_size = file_size;
    db_file->mapped_size = file_size;
    db_file->header = header;
    db_file->last_access_time = get_timestamp();
    
    open_files_[database_id] = std::move(db_file);
    
    // Rebuild in-memory vector ID index from file
    rebuild_vector_id_index(open_files_[database_id].get());
    
    return true;
}

void MemoryMappedVectorStore::close_vector_file(const std::string& database_id, bool flush_changes) {
    std::lock_guard<std::mutex> lock(files_mutex_);
    
    auto it = open_files_.find(database_id);
    if (it == open_files_.end()) {
        return;
    }
    
    auto& db_file = it->second;
    if (db_file && db_file->mapped_memory) {
        if (flush_changes) {
            sync_file(db_file->mapped_memory, db_file->mapped_size, true);
        }
        unmap_file(db_file->mapped_memory, db_file->mapped_size);
    }
    
    open_files_.erase(it);
}

MemoryMappedVectorStore::DatabaseFile* 
MemoryMappedVectorStore::get_or_open_file(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(files_mutex_);
    
    // Check if already open
    auto it = open_files_.find(database_id);
    if (it != open_files_.end()) {
        it->second->last_access_time = get_timestamp();
        return it->second.get();
    }
    
    // Try to open file
    files_mutex_.unlock();
    bool opened = open_vector_file(database_id);
    files_mutex_.lock();
    
    if (!opened) {
        return nullptr;
    }
    
    // Check LRU eviction
    if (open_files_.size() > max_open_files_) {
        evict_lru_file();
    }
    
    it = open_files_.find(database_id);
    if (it != open_files_.end()) {
        return it->second.get();
    }
    
    return nullptr;
}

void MemoryMappedVectorStore::evict_lru_file() {
    // Find LRU file (excluding currently accessed file)
    std::string lru_db_id;
    uint64_t oldest_time = UINT64_MAX;
    
    for (const auto& [db_id, db_file] : open_files_) {
        if (db_file->last_access_time < oldest_time) {
            oldest_time = db_file->last_access_time;
            lru_db_id = db_id;
        }
    }
    
    if (!lru_db_id.empty()) {
        auto& db_file = open_files_[lru_db_id];
        if (db_file && db_file->mapped_memory) {
            sync_file(db_file->mapped_memory, db_file->mapped_size, false);
            unmap_file(db_file->mapped_memory, db_file->mapped_size);
        }
        open_files_.erase(lru_db_id);
    }
}

uint64_t MemoryMappedVectorStore::hash_vector_id(const std::string& vector_id) const {
    return std::hash<std::string>{}(vector_id);
}

VectorIndexEntry* MemoryMappedVectorStore::find_index_entry(DatabaseFile* db_file,
                                                            const std::string& vector_id) {
    if (!db_file || !db_file->header) return nullptr;
    
    uint64_t hash = hash_vector_id(vector_id);
    auto* header = db_file->header;
    auto* index_base = reinterpret_cast<VectorIndexEntry*>(
        static_cast<char*>(db_file->mapped_memory) + header->index_offset);
    
    // Linear probing for hash collision resolution
    size_t probe = hash % header->index_capacity;
    size_t attempts = 0;
    
    while (attempts < header->index_capacity) {
        auto* entry = &index_base[probe];
        
        if (entry->vector_id_hash == 0) {
            // Empty slot
            return nullptr;
        }
        
        if (entry->vector_id_hash == hash && (entry->flags & VectorIndexEntry::FLAG_ACTIVE)) {
            return entry;
        }
        
        probe = (probe + 1) % header->index_capacity;
        attempts++;
    }
    
    return nullptr;
}

VectorIndexEntry* MemoryMappedVectorStore::allocate_index_entry(DatabaseFile* db_file, const std::string& vector_id) {
    if (!db_file || !db_file->header) return nullptr;
    
    auto* header = db_file->header;
    
    // Check if we need to grow index
    if (header->vector_count >= header->index_capacity * 0.75) {
        // Index is 75% full, need to resize
        return nullptr;  // TODO: Implement index resize
    }
    
    uint64_t hash = hash_vector_id(vector_id);
    auto* index_base = reinterpret_cast<VectorIndexEntry*>(
        static_cast<char*>(db_file->mapped_memory) + header->index_offset);
    
    // Use hash-based probing to find slot (same as find_index_entry)
    size_t probe = hash % header->index_capacity;
    size_t attempts = 0;
    
    while (attempts < header->index_capacity) {
        auto* entry = &index_base[probe];
        if (entry->vector_id_hash == 0 || (entry->flags & VectorIndexEntry::FLAG_DELETED)) {
            return entry;
        }
        probe = (probe + 1) % header->index_capacity;
        attempts++;
    }
    
    return nullptr;
}

uint64_t MemoryMappedVectorStore::allocate_vector_space(DatabaseFile* db_file, size_t size) {
    if (!db_file || !db_file->header) return 0;
    
    auto* header = db_file->header;
    
    // Simple allocation: append to end
    // TODO: Implement free list for deleted vectors
    uint64_t current_used = header->vector_count * header->dimension * sizeof(float);
    
    if (current_used + size > header->data_capacity) {
        // Need to grow data section
        return 0;
    }
    
    return header->data_offset + current_used;
}

void* MemoryMappedVectorStore::get_vector_data_pointer(DatabaseFile* db_file, uint64_t offset) {
    if (!db_file || !db_file->mapped_memory) return nullptr;
    return static_cast<char*>(db_file->mapped_memory) + offset;
}

bool MemoryMappedVectorStore::store_vector(const std::string& database_id,
                                           const std::string& vector_id,
                                           const std::vector<float>& values) {
    auto* db_file = get_or_open_file(database_id);
    if (!db_file) return false;
    
    std::lock_guard<std::mutex> lock(db_file->mutex);
    
    // Check dimension
    if (values.size() != db_file->header->dimension) {
        return false;
    }
    
    // Check if vector already exists
    auto* existing = find_index_entry(db_file, vector_id);
    if (existing) {
        // Update existing vector
        auto* data = reinterpret_cast<float*>(
            get_vector_data_pointer(db_file, existing->data_offset));
        std::memcpy(data, values.data(), values.size() * sizeof(float));
        existing->timestamp = get_timestamp();
        return true;
    }
    
    // Allocate new entry
    auto* entry = allocate_index_entry(db_file, vector_id);
    if (!entry) return false;
    
    // Allocate space for vector data
    uint64_t data_offset = allocate_vector_space(db_file, values.size() * sizeof(float));
    if (data_offset == 0) return false;
    
    // Write vector data
    auto* data = reinterpret_cast<float*>(get_vector_data_pointer(db_file, data_offset));
    std::memcpy(data, values.data(), values.size() * sizeof(float));
    
    // Update index entry
    entry->vector_id_hash = hash_vector_id(vector_id);
    entry->data_offset = data_offset;
    entry->size = values.size() * sizeof(float);
    entry->flags = VectorIndexEntry::FLAG_ACTIVE;
    entry->timestamp = get_timestamp();
    
    // Store vector ID string at vector_ids_offset + current position
    if (db_file->header->vector_ids_offset > 0) {
        size_t entry_index = (reinterpret_cast<char*>(entry) - 
                              (static_cast<char*>(db_file->mapped_memory) + db_file->header->index_offset)) 
                             / INDEX_ENTRY_SIZE;
        uint64_t string_offset = db_file->header->vector_ids_offset + (entry_index * 64);
        entry->string_offset = string_offset;
        
        // Write vector ID string
        char* string_ptr = static_cast<char*>(db_file->mapped_memory) + string_offset;
        std::strncpy(string_ptr, vector_id.c_str(), 63);
        string_ptr[63] = '\0';  // Ensure null termination
    }
    
    // Update header
    db_file->header->vector_count++;
    db_file->header->active_count++;
    
    // Track vector ID in index (active)
    db_file->vector_id_index_[vector_id] = true;
    
    return true;
}

std::optional<std::vector<float>> MemoryMappedVectorStore::retrieve_vector(
    const std::string& database_id,
    const std::string& vector_id) {
    
    auto* db_file = get_or_open_file(database_id);
    if (!db_file) return std::nullopt;
    
    std::lock_guard<std::mutex> lock(db_file->mutex);
    
    auto* entry = find_index_entry(db_file, vector_id);
    if (!entry) return std::nullopt;
    
    // Read vector data
    auto* data = reinterpret_cast<const float*>(
        get_vector_data_pointer(db_file, entry->data_offset));
    
    size_t count = entry->size / sizeof(float);
    return std::vector<float>(data, data + count);
}

bool MemoryMappedVectorStore::update_vector(const std::string& database_id,
                                            const std::string& vector_id,
                                            const std::vector<float>& new_values) {
    // For now, update is same as store (overwrites in-place)
    return store_vector(database_id, vector_id, new_values);
}

bool MemoryMappedVectorStore::delete_vector(const std::string& database_id,
                                            const std::string& vector_id) {
    auto* db_file = get_or_open_file(database_id);
    if (!db_file) return false;
    
    std::lock_guard<std::mutex> lock(db_file->mutex);
    
    auto* entry = find_index_entry(db_file, vector_id);
    if (!entry) return false;
    
    // Soft delete: set flag
    entry->flags |= VectorIndexEntry::FLAG_DELETED;
    entry->flags &= ~VectorIndexEntry::FLAG_ACTIVE;
    db_file->header->active_count--;
    
    // Mark as deleted in index (false = deleted)
    db_file->vector_id_index_[vector_id] = false;
    
    return true;
}

size_t MemoryMappedVectorStore::batch_store(
    const std::string& database_id,
    const std::vector<std::pair<std::string, std::vector<float>>>& vectors) {
    
    size_t stored = 0;
    for (const auto& [vector_id, values] : vectors) {
        if (store_vector(database_id, vector_id, values)) {
            stored++;
        }
    }
    return stored;
}

std::vector<std::optional<std::vector<float>>> MemoryMappedVectorStore::batch_retrieve(
    const std::string& database_id,
    const std::vector<std::string>& vector_ids) {
    
    std::vector<std::optional<std::vector<float>>> results;
    results.reserve(vector_ids.size());
    
    for (const auto& vector_id : vector_ids) {
        results.push_back(retrieve_vector(database_id, vector_id));
    }
    
    return results;
}

void MemoryMappedVectorStore::flush(const std::string& database_id, bool sync) {
    std::lock_guard<std::mutex> lock(files_mutex_);
    
    auto it = open_files_.find(database_id);
    if (it == open_files_.end()) return;
    
    auto& db_file = it->second;
    if (db_file && db_file->mapped_memory) {
        sync_file(db_file->mapped_memory, db_file->mapped_size, sync);
    }
}

void MemoryMappedVectorStore::flush_all(bool sync) {
    std::lock_guard<std::mutex> lock(files_mutex_);
    
    for (auto& [db_id, db_file] : open_files_) {
        if (db_file && db_file->mapped_memory) {
            sync_file(db_file->mapped_memory, db_file->mapped_size, sync);
        }
    }
}

bool MemoryMappedVectorStore::delete_database_vectors(const std::string& database_id) {
    // Close file first
    close_vector_file(database_id, false);
    
    // Delete file and directory
    std::string file_path = get_vector_file_path(database_id);
    std::string db_dir = get_database_directory(database_id);
    
    std::error_code ec;
    std::filesystem::remove_all(db_dir, ec);
    
    return !ec;
}

size_t MemoryMappedVectorStore::get_vector_count(const std::string& database_id) {
    auto* db_file = get_or_open_file(database_id);
    if (!db_file || !db_file->header) return 0;
    
    return db_file->header->active_count;
}

int MemoryMappedVectorStore::get_dimension(const std::string& database_id) {
    auto* db_file = get_or_open_file(database_id);
    if (!db_file || !db_file->header) return 0;
    
    return db_file->header->dimension;
}

bool MemoryMappedVectorStore::has_database(const std::string& database_id) {
    return std::filesystem::exists(get_vector_file_path(database_id));
}

std::vector<std::string> MemoryMappedVectorStore::list_vector_ids(const std::string& database_id) {
    auto* db_file = get_or_open_file(database_id);
    if (!db_file) return {};
    
    std::lock_guard<std::mutex> lock(db_file->mutex);
    
    std::vector<std::string> active_ids;
    active_ids.reserve(db_file->header->active_count);
    
    // Return only active vector IDs (where value is true)
    for (const auto& [vector_id, is_active] : db_file->vector_id_index_) {
        if (is_active) {
            active_ids.push_back(vector_id);
        }
    }
    
    return active_ids;
}

size_t MemoryMappedVectorStore::get_deleted_count(const std::string& database_id) {
    auto* db_file = get_or_open_file(database_id);
    if (!db_file) return 0;
    
    std::lock_guard<std::mutex> lock(db_file->mutex);
    
    size_t deleted_count = 0;
    for (const auto& [vector_id, is_active] : db_file->vector_id_index_) {
        if (!is_active) {
            deleted_count++;
        }
    }
    
    return deleted_count;
}

std::string MemoryMappedVectorStore::get_database_vector_file_path(const std::string& database_id) const {
    return get_vector_file_path(database_id);
}

void MemoryMappedVectorStore::rebuild_vector_id_index(DatabaseFile* db_file) {
    if (!db_file || !db_file->header) return;
    
    // Clear existing index
    db_file->vector_id_index_.clear();
    
    // If no vector_ids_offset, file was created with old format
    if (db_file->header->vector_ids_offset == 0) {
        return;  // Can't rebuild without stored strings
    }
    
    auto* header = db_file->header;
    auto* index_base = reinterpret_cast<VectorIndexEntry*>(
        static_cast<char*>(db_file->mapped_memory) + header->index_offset);
    
    // Scan through all index entries
    for (size_t i = 0; i < header->index_capacity; i++) {
        auto* entry = &index_base[i];
        
        // Skip empty entries
        if (entry->vector_id_hash == 0) {
            continue;
        }
        
        // Check if entry has a string offset
        if (entry->string_offset > 0 && entry->string_offset < db_file->file_size) {
            // Read vector ID string
            const char* string_ptr = static_cast<const char*>(db_file->mapped_memory) + entry->string_offset;
            std::string vector_id(string_ptr);
            
            // Add to in-memory index based on active/deleted flag
            bool is_active = (entry->flags & VectorIndexEntry::FLAG_ACTIVE) != 0;
            db_file->vector_id_index_[vector_id] = is_active;
        }
    }
}

// Platform-specific mmap operations

#ifdef _WIN32

void* MemoryMappedVectorStore::map_file(FileHandle* handle, size_t size) {
    if (!handle || !handle->is_open()) return nullptr;
    
    HANDLE mapping = CreateFileMappingA(
        handle->native_handle(),
        nullptr,
        PAGE_READWRITE,
        static_cast<DWORD>(size >> 32),
        static_cast<DWORD>(size & 0xFFFFFFFF),
        nullptr
    );
    
    if (!mapping) return nullptr;
    
    void* addr = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size);
    CloseHandle(mapping);
    
    return addr;
}

void MemoryMappedVectorStore::unmap_file(void* addr, size_t size) {
    if (addr) {
        UnmapViewOfFile(addr);
    }
}

bool MemoryMappedVectorStore::sync_file(void* addr, size_t size, bool synchronous) {
    if (!addr) return false;
    return FlushViewOfFile(addr, size) != 0;
}

#else  // Unix/Linux

void* MemoryMappedVectorStore::map_file(FileHandle* handle, size_t size) {
    if (!handle || !handle->is_open()) return nullptr;
    
    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     handle->native_handle(), 0);
    
    if (addr == MAP_FAILED) {
        return nullptr;
    }
    
    return addr;
}

void MemoryMappedVectorStore::unmap_file(void* addr, size_t size) {
    if (addr) {
        munmap(addr, size);
    }
}

bool MemoryMappedVectorStore::sync_file(void* addr, size_t size, bool synchronous) {
    if (!addr) return false;
    
    int flags = synchronous ? MS_SYNC : MS_ASYNC;
    return msync(addr, size, flags) == 0;
}

#endif

} // namespace jadevectordb
