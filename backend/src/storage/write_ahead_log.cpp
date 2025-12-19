#include "write_ahead_log.h"
#include <filesystem>
#include <chrono>
#include <cstring>
#include <algorithm>

namespace jadevectordb {

WriteAheadLog::WriteAheadLog(const std::string& database_id,
                             const std::string& wal_directory)
    : database_id_(database_id)
    , wal_directory_(wal_directory)
    , sequence_number_(0)
    , entry_count_(0)
    , log_size_(0) {
    
    wal_file_path_ = get_wal_file_path();
}

WriteAheadLog::~WriteAheadLog() {
    flush(true);  // Sync on shutdown
    if (wal_file_.is_open()) {
        wal_file_.close();
    }
}

std::string WriteAheadLog::get_wal_file_path() const {
    return wal_directory_ + "/" + database_id_ + ".wal";
}

bool WriteAheadLog::initialize() {
    // Create WAL directory if needed
    std::filesystem::create_directories(wal_directory_);
    
    // Check if WAL file exists
    bool exists = std::filesystem::exists(wal_file_path_);
    
    // Open WAL file in read/write mode, create if doesn't exist
    wal_file_.open(wal_file_path_, 
                   std::ios::in | std::ios::out | std::ios::binary | std::ios::ate);
    
    if (!wal_file_.is_open()) {
        // Try to create file
        wal_file_.open(wal_file_path_,
                      std::ios::out | std::ios::binary);
        if (!wal_file_.is_open()) {
            return false;
        }
        wal_file_.close();
        
        // Reopen in read/write mode
        wal_file_.open(wal_file_path_,
                      std::ios::in | std::ios::out | std::ios::binary | std::ios::ate);
        if (!wal_file_.is_open()) {
            return false;
        }
    }
    
    // Get file size
    wal_file_.seekg(0, std::ios::end);
    log_size_ = wal_file_.tellg();
    
    // If file exists, scan to get sequence number and entry count
    if (exists && log_size_ > 0) {
        wal_file_.seekg(0, std::ios::beg);
        
        while (wal_file_.tellg() < static_cast<std::streampos>(log_size_)) {
            WALEntryHeader header;
            wal_file_.read(reinterpret_cast<char*>(&header), sizeof(header));
            
            if (!wal_file_) break;
            
            if (header.magic_number != WAL_MAGIC) {
                // Corrupted log, stop here
                break;
            }
            
            // Update sequence number
            if (header.sequence_number > sequence_number_) {
                sequence_number_ = header.sequence_number;
            }
            
            entry_count_++;
            
            // Skip entry data
            wal_file_.seekg(header.data_size, std::ios::cur);
        }
        
        // Position at end for appending
        wal_file_.seekp(0, std::ios::end);
    }
    
    return true;
}

uint64_t WriteAheadLog::get_timestamp_micros() const {
    auto now = std::chrono::system_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
    return static_cast<uint64_t>(micros);
}

uint32_t WriteAheadLog::calculate_checksum(const void* data, size_t size) const {
    // Simple CRC32-like checksum
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < size; i++) {
        crc ^= bytes[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    
    return ~crc;
}

bool WriteAheadLog::verify_checksum(const WALEntryHeader& header, const void* data) const {
    return calculate_checksum(data, header.data_size) == header.checksum;
}

bool WriteAheadLog::write_entry(WALEntryType type, const void* data, size_t size) {
    std::lock_guard<std::mutex> lock(wal_mutex_);
    
    if (!wal_file_.is_open()) {
        return false;
    }
    
    // Create header
    WALEntryHeader header;
    std::memset(&header, 0, sizeof(header));
    header.magic_number = WAL_MAGIC;
    header.entry_type = type;
    header.sequence_number = ++sequence_number_;
    header.timestamp = get_timestamp_micros();
    header.data_size = size;
    header.checksum = calculate_checksum(data, size);
    
    // Write header
    wal_file_.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!wal_file_) {
        return false;
    }
    
    // Write data
    wal_file_.write(static_cast<const char*>(data), size);
    if (!wal_file_) {
        return false;
    }
    
    log_size_ += sizeof(header) + size;
    entry_count_++;
    
    return true;
}

bool WriteAheadLog::log_vector_store(const std::string& database_id,
                                     const std::string& vector_id,
                                     const std::vector<float>& values) {
    // Prepare entry data
    WALVectorEntry entry;
    std::memset(&entry, 0, sizeof(entry));
    
    std::strncpy(entry.database_id, database_id.c_str(), 63);
    std::strncpy(entry.vector_id, vector_id.c_str(), 63);
    entry.dimension = values.size();
    
    // Combine entry + vector data
    std::vector<uint8_t> data(sizeof(entry) + values.size() * sizeof(float));
    std::memcpy(data.data(), &entry, sizeof(entry));
    std::memcpy(data.data() + sizeof(entry), values.data(), values.size() * sizeof(float));
    
    return write_entry(WALEntryType::VECTOR_STORE, data.data(), data.size());
}

bool WriteAheadLog::log_vector_update(const std::string& database_id,
                                      const std::string& vector_id,
                                      const std::vector<float>& values) {
    // Prepare entry data
    WALVectorEntry entry;
    std::memset(&entry, 0, sizeof(entry));
    
    std::strncpy(entry.database_id, database_id.c_str(), 63);
    std::strncpy(entry.vector_id, vector_id.c_str(), 63);
    entry.dimension = values.size();
    
    // Combine entry + vector data
    std::vector<uint8_t> data(sizeof(entry) + values.size() * sizeof(float));
    std::memcpy(data.data(), &entry, sizeof(entry));
    std::memcpy(data.data() + sizeof(entry), values.data(), values.size() * sizeof(float));
    
    return write_entry(WALEntryType::VECTOR_UPDATE, data.data(), data.size());
}

bool WriteAheadLog::log_vector_delete(const std::string& database_id,
                                      const std::string& vector_id) {
    // Prepare entry data
    WALVectorEntry entry;
    std::memset(&entry, 0, sizeof(entry));
    
    std::strncpy(entry.database_id, database_id.c_str(), 63);
    std::strncpy(entry.vector_id, vector_id.c_str(), 63);
    entry.dimension = 0;  // No vector data for delete
    
    return write_entry(WALEntryType::VECTOR_DELETE, &entry, sizeof(entry));
}

bool WriteAheadLog::log_index_resize(const std::string& database_id,
                                     size_t old_capacity,
                                     size_t new_capacity) {
    // Prepare entry data
    struct IndexResizeEntry {
        char database_id[64];
        uint64_t old_capacity;
        uint64_t new_capacity;
    } entry;
    
    std::memset(&entry, 0, sizeof(entry));
    std::strncpy(entry.database_id, database_id.c_str(), 63);
    entry.old_capacity = old_capacity;
    entry.new_capacity = new_capacity;
    
    return write_entry(WALEntryType::INDEX_RESIZE, &entry, sizeof(entry));
}

bool WriteAheadLog::write_checkpoint() {
    // Empty checkpoint marker
    uint8_t marker = 0;
    return write_entry(WALEntryType::CHECKPOINT, &marker, sizeof(marker));
}

bool WriteAheadLog::flush(bool sync) {
    std::lock_guard<std::mutex> lock(wal_mutex_);
    
    if (!wal_file_.is_open()) {
        return false;
    }
    
    wal_file_.flush();
    
    if (sync) {
        // Force sync to disk (platform-specific)
#ifdef _WIN32
        // Windows: Use _commit
        int fd = _fileno(wal_file_.rdbuf()->_Filebuffer);
        if (fd >= 0) {
            _commit(fd);
        }
#else
        // Unix: fsync - Get file descriptor from stream
        // Note: This is implementation-specific and may not work on all platforms
        // A more portable solution would be to use POSIX open/write directly
        wal_file_.sync();
#endif
    }
    
    return true;
}

int WriteAheadLog::replay(std::function<bool(const WALEntryHeader&, const std::vector<uint8_t>&)> replay_callback) {
    std::lock_guard<std::mutex> lock(wal_mutex_);
    
    if (!wal_file_.is_open()) {
        return -1;
    }
    
    // Seek to beginning
    wal_file_.seekg(0, std::ios::beg);
    
    int replayed_count = 0;
    
    while (wal_file_.tellg() < static_cast<std::streampos>(log_size_)) {
        WALEntryHeader header;
        wal_file_.read(reinterpret_cast<char*>(&header), sizeof(header));
        
        if (!wal_file_) break;
        
        // Verify magic number
        if (header.magic_number != WAL_MAGIC) {
            // Corrupted log
            break;
        }
        
        // Read entry data
        std::vector<uint8_t> data(header.data_size);
        wal_file_.read(reinterpret_cast<char*>(data.data()), header.data_size);
        
        if (!wal_file_) break;
        
        // Verify checksum
        if (!verify_checksum(header, data.data())) {
            // Corrupted entry, stop replay
            break;
        }
        
        // Skip checkpoint markers
        if (header.entry_type == WALEntryType::CHECKPOINT) {
            continue;
        }
        
        // Execute callback
        if (!replay_callback(header, data)) {
            // Callback failed, stop replay
            break;
        }
        
        replayed_count++;
    }
    
    // Position at end for future writes
    wal_file_.seekp(0, std::ios::end);
    
    return replayed_count;
}

bool WriteAheadLog::truncate() {
    std::lock_guard<std::mutex> lock(wal_mutex_);
    
    if (!wal_file_.is_open()) {
        return false;
    }
    
    // Close file
    wal_file_.close();
    
    // Delete file
    std::error_code ec;
    std::filesystem::remove(wal_file_path_, ec);
    
    // Recreate empty file
    wal_file_.open(wal_file_path_,
                  std::ios::out | std::ios::binary);
    if (!wal_file_.is_open()) {
        return false;
    }
    wal_file_.close();
    
    // Reopen in read/write mode
    wal_file_.open(wal_file_path_,
                  std::ios::in | std::ios::out | std::ios::binary | std::ios::ate);
    
    if (!wal_file_.is_open()) {
        return false;
    }
    
    // Reset counters
    log_size_ = 0;
    entry_count_ = 0;
    // Keep sequence_number_ to maintain monotonicity
    
    return true;
}

size_t WriteAheadLog::get_log_size() const {
    return log_size_;
}

uint64_t WriteAheadLog::get_entry_count() const {
    return entry_count_;
}

bool WriteAheadLog::needs_checkpoint(size_t size_threshold) const {
    return log_size_ >= size_threshold;
}

} // namespace jadevectordb
