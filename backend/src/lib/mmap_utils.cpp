#include "mmap_utils.h"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <sys/stat.h>

namespace jadevectordb {

MMapFile::MMapFile(const std::string& file_path) 
    : fd_(-1), mapped_addr_(nullptr), file_size_(0), is_mapped_(false), file_path_(file_path) {
}

MMapFile::~MMapFile() {
    if (is_mapped_) {
        unmap();
    }
    if (fd_ != -1) {
        close();
    }
}

MMapFile::MMapFile(MMapFile&& other) noexcept
    : fd_(other.fd_), mapped_addr_(other.mapped_addr_), file_size_(other.file_size_),
      is_mapped_(other.is_mapped_), file_path_(std::move(other.file_path_)) {
    other.fd_ = -1;
    other.mapped_addr_ = nullptr;
    other.file_size_ = 0;
    other.is_mapped_ = false;
}

MMapFile& MMapFile::operator=(MMapFile&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        if (is_mapped_) {
            unmap();
        }
        if (fd_ != -1) {
            close();
        }
        
        // Move resources
        fd_ = other.fd_;
        mapped_addr_ = other.mapped_addr_;
        file_size_ = other.file_size_;
        is_mapped_ = other.is_mapped_;
        file_path_ = std::move(other.file_path_);
        
        // Reset other
        other.fd_ = -1;
        other.mapped_addr_ = nullptr;
        other.file_size_ = 0;
        other.is_mapped_ = false;
    }
    return *this;
}

bool MMapFile::map(size_t size_hint) {
    if (is_mapped_) {
        return false;
    }
    
    // Open file
    fd_ = open(file_path_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ == -1) {
        return false;
    }
    
    // Get file size
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    
    file_size_ = sb.st_size;
    
    // If file is empty and size hint is provided, extend it
    if (file_size_ == 0 && size_hint > 0) {
        if (!resize(size_hint)) {
            close(fd_);
            fd_ = -1;
            return false;
        }
        file_size_ = size_hint;
    }
    
    // If file is still empty, we can't map it
    if (file_size_ == 0) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    
    // Map file
    mapped_addr_ = mmap(nullptr, file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapped_addr_ == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    
    is_mapped_ = true;
    return true;
}

bool MMapFile::unmap() {
    if (!is_mapped_) {
        return false;
    }
    
    if (munmap(mapped_addr_, file_size_) == -1) {
        return false;
    }
    
    mapped_addr_ = nullptr;
    file_size_ = 0;
    is_mapped_ = false;
    return true;
}

void* MMapFile::get_address() const {
    return mapped_addr_;
}

size_t MMapFile::get_size() const {
    return file_size_;
}

bool MMapFile::is_mapped() const {
    return is_mapped_;
}

const std::string& MMapFile::get_file_path() const {
    return file_path_;
}

bool MMapFile::resize(size_t new_size) {
    // If currently mapped, unmap first
    bool was_mapped = is_mapped_;
    if (was_mapped) {
        if (!unmap()) {
            return false;
        }
    }
    
    // Resize file
    if (ftruncate(fd_, new_size) == -1) {
        // Remap if it was mapped before
        if (was_mapped) {
            map(file_size_);
        }
        return false;
    }
    
    file_size_ = new_size;
    
    // Remap if it was mapped before
    if (was_mapped) {
        return map(new_size);
    }
    
    return true;
}

bool MMapFile::sync() {
    if (!is_mapped_) {
        return false;
    }
    
    return msync(mapped_addr_, file_size_, MS_SYNC) == 0;
}

void MMapFile::close() {
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
}

// Utility functions implementation
namespace mmap_utils {
    
    std::unique_ptr<MMapFile> create_mmap_file(const std::string& file_path, size_t initial_size) {
        auto mmap_file = std::make_unique<MMapFile>(file_path);
        if (initial_size > 0) {
            if (!mmap_file->resize(initial_size)) {
                return nullptr;
            }
        }
        return mmap_file;
    }
    
    void* map_file_region(int fd, size_t offset, size_t length, bool read_only) {
        int prot = read_only ? PROT_READ : (PROT_READ | PROT_WRITE);
        return mmap(nullptr, length, prot, MAP_SHARED, fd, offset);
    }
    
    bool unmap_file_region(void* addr, size_t length) {
        return munmap(addr, length) == 0;
    }
    
    bool can_mmap_file(const std::string& file_path) {
        int fd = open(file_path.c_str(), O_RDONLY);
        if (fd == -1) {
            return false;
        }
        
        struct stat sb;
        bool result = (fstat(fd, &sb) == 0) && (sb.st_size > 0);
        close(fd);
        return result;
    }
    
    size_t get_file_size(const std::string& file_path) {
        struct stat sb;
        if (stat(file_path.c_str(), &sb) == 0) {
            return sb.st_size;
        }
        return 0;
    }
    
} // namespace mmap_utils

} // namespace jadevectordb