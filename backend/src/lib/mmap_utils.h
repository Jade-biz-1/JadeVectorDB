#ifndef JADEVECTORDB_MMAP_UTILS_H
#define JADEVECTORDB_MMAP_UTILS_H

#include <string>
#include <memory>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace jadevectordb {

class MMapFile {
private:
    int fd_;
    void* mapped_addr_;
    size_t file_size_;
    bool is_mapped_;
    std::string file_path_;

public:
    // Constructor
    explicit MMapFile(const std::string& file_path);
    
    // Destructor
    ~MMapFile();
    
    // Disable copy constructor and assignment operator
    MMapFile(const MMapFile&) = delete;
    MMapFile& operator=(const MMapFile&) = delete;
    
    // Move constructor and assignment operator
    MMapFile(MMapFile&& other) noexcept;
    MMapFile& operator=(MMapFile&& other) noexcept;
    
    // Methods
    bool map(size_t size_hint = 0);
    bool unmap();
    void* get_address() const;
    size_t get_size() const;
    bool is_mapped() const;
    const std::string& get_file_path() const;
    
    // Resize file
    bool resize(size_t new_size);
    
    // Sync to disk
    bool sync();
    
    // Close file
    void close();
};

// Utility functions for memory mapping
namespace mmap_utils {
    
    // Create a memory-mapped file
    std::unique_ptr<MMapFile> create_mmap_file(const std::string& file_path, size_t initial_size = 0);
    
    // Map a portion of a file
    void* map_file_region(int fd, size_t offset, size_t length, bool read_only = false);
    
    // Unmap a region
    bool unmap_file_region(void* addr, size_t length);
    
    // Check if a file can be memory-mapped
    bool can_mmap_file(const std::string& file_path);
    
    // Get file size
    size_t get_file_size(const std::string& file_path);
    
} // namespace mmap_utils

} // namespace jadevectordb

#endif // JADEVECTORDB_MMAP_UTILS_H