#include "storage_format.h"
#include <fstream>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <set>
#include <zlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include "lib/mmap_utils.h"

namespace jadevectordb {

namespace storage_format {

    // Storage file manager implementation
    StorageFileManager::StorageFileManager(const std::string& file_path)
        : file_path_(file_path), is_open_(false), file_descriptor_(-1) {
    }
    
    StorageFileManager::~StorageFileManager() {
        if (is_open_) {
            close_file();
        }
    }
    
    bool StorageFileManager::create_file() {
        std::ofstream file(file_path_, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            return false;
        }
        
        // Write initial header
        StorageHeader header = create_header(VECTOR_DATA, 0);
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        file.close();
        
        return file.good();
    }
    
    bool StorageFileManager::open_file() {
        // Check if file exists
        std::ifstream test_file(file_path_, std::ios::binary);
        if (!test_file.good()) {
            // File doesn't exist, create it
            if (!create_file()) {
                return false;
            }
        }
        test_file.close();

        is_open_ = true;
        return true;
    }

    bool StorageFileManager::close_file() {
        if (is_open_) {
            // Release any locks before closing
            if (file_descriptor_ >= 0) {
                release_lock();
                ::close(file_descriptor_);
                file_descriptor_ = -1;
            }
            is_open_ = false;
        }
        return true;
    }

    bool StorageFileManager::is_open() const {
        return is_open_;
    }

    bool StorageFileManager::acquire_read_lock() {
        // Open file descriptor if not already open
        if (file_descriptor_ < 0) {
            file_descriptor_ = ::open(file_path_.c_str(), O_RDONLY);
            if (file_descriptor_ < 0) {
                return false;
            }
        }

        // Acquire shared (read) lock using flock
        // LOCK_SH = shared lock, allows multiple readers
        if (::flock(file_descriptor_, LOCK_SH) == -1) {
            return false;
        }

        return true;
    }

    bool StorageFileManager::acquire_write_lock() {
        // Open file descriptor if not already open
        if (file_descriptor_ < 0) {
            file_descriptor_ = ::open(file_path_.c_str(), O_RDWR | O_CREAT, 0644);
            if (file_descriptor_ < 0) {
                return false;
            }
        }

        // Acquire exclusive (write) lock using flock
        // LOCK_EX = exclusive lock, blocks all other readers and writers
        if (::flock(file_descriptor_, LOCK_EX) == -1) {
            return false;
        }

        return true;
    }

    bool StorageFileManager::release_lock() {
        if (file_descriptor_ < 0) {
            return true;  // No lock to release
        }

        // Release lock using LOCK_UN
        if (::flock(file_descriptor_, LOCK_UN) == -1) {
            return false;
        }

        return true;
    }

    bool StorageFileManager::verify_file_integrity() {
        if (!is_open_) {
            return false;
        }

        try {
            std::ifstream file(file_path_, std::ios::binary);
            if (!file.is_open()) {
                return false;
            }

            size_t valid_entries = 0;
            size_t corrupted_entries = 0;

            while (file.good() && !file.eof()) {
                StorageHeader header;
                file.read(reinterpret_cast<char*>(&header), sizeof(header));

                if (file.gcount() != sizeof(header)) {
                    break; // End of file
                }

                // Verify header magic number and version
                if (!verify_header(header)) {
                    corrupted_entries++;
                    // Try to skip this entry
                    file.seekg(header.data_size, std::ios::cur);
                    continue;
                }

                // Read data section
                std::vector<uint8_t> data(header.data_size);
                file.read(reinterpret_cast<char*>(data.data()), header.data_size);

                if (file.gcount() != static_cast<std::streamsize>(header.data_size)) {
                    corrupted_entries++;
                    break; // Truncated file
                }

                // Verify checksum
                uint64_t calculated_checksum = calculate_checksum(data.data(), data.size());
                if (calculated_checksum != header.checksum) {
                    corrupted_entries++;
                } else {
                    valid_entries++;
                }
            }

            file.close();

            // File is considered intact if no corrupted entries found
            return corrupted_entries == 0;

        } catch (const std::exception& e) {
            return false;
        }
    }

    std::vector<size_t> StorageFileManager::find_corrupted_entries() {
        std::vector<size_t> corrupted_offsets;

        try {
            std::ifstream file(file_path_, std::ios::binary);
            if (!file.is_open()) {
                return corrupted_offsets;
            }

            size_t entry_index = 0;
            while (file.good() && !file.eof()) {
                size_t current_offset = file.tellg();
                StorageHeader header;
                file.read(reinterpret_cast<char*>(&header), sizeof(header));

                if (file.gcount() != sizeof(header)) {
                    break;
                }

                if (!verify_header(header)) {
                    corrupted_offsets.push_back(current_offset);
                    file.seekg(header.data_size, std::ios::cur);
                    entry_index++;
                    continue;
                }

                std::vector<uint8_t> data(header.data_size);
                file.read(reinterpret_cast<char*>(data.data()), header.data_size);

                if (file.gcount() != static_cast<std::streamsize>(header.data_size)) {
                    corrupted_offsets.push_back(current_offset);
                    break;
                }

                uint64_t calculated_checksum = calculate_checksum(data.data(), data.size());
                if (calculated_checksum != header.checksum) {
                    corrupted_offsets.push_back(current_offset);
                }

                entry_index++;
            }

            file.close();
            return corrupted_offsets;

        } catch (const std::exception& e) {
            return corrupted_offsets;
        }
    }

    bool StorageFileManager::repair_corrupted_file(const std::string& backup_path) {
        try {
            // Find corrupted entries
            auto corrupted_offsets = find_corrupted_entries();

            if (corrupted_offsets.empty()) {
                return true; // Nothing to repair
            }

            // Create backup
            std::filesystem::copy_file(file_path_, backup_path,
                                      std::filesystem::copy_options::overwrite_existing);

            // Open original file for reading
            std::ifstream input(file_path_, std::ios::binary);
            if (!input.is_open()) {
                return false;
            }

            // Create temporary repaired file
            std::string temp_path = file_path_ + ".repaired";
            std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
            if (!output.is_open()) {
                input.close();
                return false;
            }

            // Copy only valid entries to repaired file
            size_t current_offset = 0;
            std::set<size_t> corrupted_set(corrupted_offsets.begin(), corrupted_offsets.end());

            while (input.good() && !input.eof()) {
                size_t entry_offset = input.tellg();

                StorageHeader header;
                input.read(reinterpret_cast<char*>(&header), sizeof(header));

                if (input.gcount() != sizeof(header)) {
                    break;
                }

                std::vector<uint8_t> data(header.data_size);
                input.read(reinterpret_cast<char*>(data.data()), header.data_size);

                if (input.gcount() != static_cast<std::streamsize>(header.data_size)) {
                    break;
                }

                // Only write if this entry is not corrupted
                if (corrupted_set.find(entry_offset) == corrupted_set.end()) {
                    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
                    output.write(reinterpret_cast<const char*>(data.data()), data.size());
                }
            }

            input.close();
            output.close();

            // Replace original file with repaired file
            std::filesystem::remove(file_path_);
            std::filesystem::rename(temp_path, file_path_);

            return true;

        } catch (const std::exception& e) {
            return false;
        }
    }

    bool StorageFileManager::create_recovery_checkpoint() {
        try {
            // Verify file integrity first
            if (!verify_file_integrity()) {
                return false;
            }

            // Create checkpoint file with timestamp
            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();

            std::string checkpoint_path = file_path_ + ".checkpoint." + std::to_string(timestamp);

            // Copy file to checkpoint location
            std::filesystem::copy_file(file_path_, checkpoint_path,
                                      std::filesystem::copy_options::overwrite_existing);

            return true;

        } catch (const std::exception& e) {
            return false;
        }
    }

    bool StorageFileManager::write_vector(const Vector& vector) {
        if (!is_open_) {
            return false;
        }

        try {
            // Convert to storage format
            VectorStorageFormat storage_format = convert_to_storage_format(vector);

            // Serialize the vector data to binary
            std::vector<uint8_t> serialized_data;

            // Write ID (length + data)
            uint32_t id_length = static_cast<uint32_t>(storage_format.id.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&id_length),
                                 reinterpret_cast<const uint8_t*>(&id_length) + sizeof(id_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.id.begin(),
                                 storage_format.id.end());

            // Write dimension
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&storage_format.dimension),
                                 reinterpret_cast<const uint8_t*>(&storage_format.dimension) + sizeof(storage_format.dimension));

            // Write vector values
            for (float value : storage_format.values) {
                serialized_data.insert(serialized_data.end(),
                                     reinterpret_cast<const uint8_t*>(&value),
                                     reinterpret_cast<const uint8_t*>(&value) + sizeof(value));
            }

            // Write metadata JSON (length + data)
            uint32_t metadata_length = static_cast<uint32_t>(storage_format.metadata_json.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&metadata_length),
                                 reinterpret_cast<const uint8_t*>(&metadata_length) + sizeof(metadata_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.metadata_json.begin(),
                                 storage_format.metadata_json.end());

            // Write timestamps
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&storage_format.created_timestamp),
                                 reinterpret_cast<const uint8_t*>(&storage_format.created_timestamp) + sizeof(storage_format.created_timestamp));
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&storage_format.updated_timestamp),
                                 reinterpret_cast<const uint8_t*>(&storage_format.updated_timestamp) + sizeof(storage_format.updated_timestamp));

            // Write version and deleted flag
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&storage_format.version),
                                 reinterpret_cast<const uint8_t*>(&storage_format.version) + sizeof(storage_format.version));
            uint8_t deleted_flag = storage_format.deleted ? 1 : 0;
            serialized_data.push_back(deleted_flag);

            // Create header with checksum
            StorageHeader header = create_header(VECTOR_DATA, serialized_data.size());
            header.checksum = calculate_checksum(serialized_data.data(), serialized_data.size());

            // Open file in append mode
            std::ofstream file(file_path_, std::ios::binary | std::ios::app);
            if (!file.is_open()) {
                return false;
            }

            // Write header and data
            file.write(reinterpret_cast<const char*>(&header), sizeof(header));
            file.write(reinterpret_cast<const char*>(serialized_data.data()), serialized_data.size());

            file.close();
            return file.good();

        } catch (const std::exception& e) {
            return false;
        }
    }
    
    bool StorageFileManager::write_database(const Database& database) {
        if (!is_open_) {
            return false;
        }

        try {
            DatabaseStorageFormat storage_format = convert_to_storage_format(database);
            std::vector<uint8_t> serialized_data;

            // Write database_id (length + data)
            uint32_t db_id_length = static_cast<uint32_t>(storage_format.database_id.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&db_id_length),
                                 reinterpret_cast<const uint8_t*>(&db_id_length) + sizeof(db_id_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.database_id.begin(),
                                 storage_format.database_id.end());

            // Write name (length + data)
            uint32_t name_length = static_cast<uint32_t>(storage_format.name.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&name_length),
                                 reinterpret_cast<const uint8_t*>(&name_length) + sizeof(name_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.name.begin(),
                                 storage_format.name.end());

            // Write description (length + data)
            uint32_t desc_length = static_cast<uint32_t>(storage_format.description.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&desc_length),
                                 reinterpret_cast<const uint8_t*>(&desc_length) + sizeof(desc_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.description.begin(),
                                 storage_format.description.end());

            // Write vector_dimension
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&storage_format.vector_dimension),
                                 reinterpret_cast<const uint8_t*>(&storage_format.vector_dimension) + sizeof(storage_format.vector_dimension));

            // Write index_type (length + data)
            uint32_t index_type_length = static_cast<uint32_t>(storage_format.index_type.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&index_type_length),
                                 reinterpret_cast<const uint8_t*>(&index_type_length) + sizeof(index_type_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.index_type.begin(),
                                 storage_format.index_type.end());

            // Write all JSON fields (index_parameters, sharding_config, etc.)
            auto write_json_field = [&](const std::string& json_str) {
                uint32_t length = static_cast<uint32_t>(json_str.size());
                serialized_data.insert(serialized_data.end(),
                                     reinterpret_cast<const uint8_t*>(&length),
                                     reinterpret_cast<const uint8_t*>(&length) + sizeof(length));
                serialized_data.insert(serialized_data.end(), json_str.begin(), json_str.end());
            };

            write_json_field(storage_format.index_parameters_json);
            write_json_field(storage_format.sharding_config_json);
            write_json_field(storage_format.replication_config_json);
            write_json_field(storage_format.embedding_models_json);
            write_json_field(storage_format.metadata_schema_json);
            write_json_field(storage_format.retention_policy_json);
            write_json_field(storage_format.access_control_json);

            // Write timestamps
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&storage_format.created_timestamp),
                                 reinterpret_cast<const uint8_t*>(&storage_format.created_timestamp) + sizeof(storage_format.created_timestamp));
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&storage_format.updated_timestamp),
                                 reinterpret_cast<const uint8_t*>(&storage_format.updated_timestamp) + sizeof(storage_format.updated_timestamp));

            // Create header with checksum
            StorageHeader header = create_header(DATABASE_DATA, serialized_data.size());
            header.checksum = calculate_checksum(serialized_data.data(), serialized_data.size());

            // Write to file
            std::ofstream file(file_path_, std::ios::binary | std::ios::app);
            if (!file.is_open()) {
                return false;
            }

            file.write(reinterpret_cast<const char*>(&header), sizeof(header));
            file.write(reinterpret_cast<const char*>(serialized_data.data()), serialized_data.size());

            file.close();
            return file.good();

        } catch (const std::exception& e) {
            return false;
        }
    }
    
    bool StorageFileManager::write_index(const Index& index) {
        if (!is_open_) {
            return false;
        }
        
        // Convert to storage format
        IndexStorageFormat storage_format = convert_to_storage_format(index);
        
        // In a real implementation, we would write to the file
        // This is a placeholder implementation
        return true;
    }
    
    bool StorageFileManager::write_embedding_model(const EmbeddingModel& model) {
        if (!is_open_) {
            return false;
        }
        
        // Convert to storage format
        EmbeddingModelStorageFormat storage_format = convert_to_storage_format(model);
        
        // In a real implementation, we would write to the file
        // This is a placeholder implementation
        return true;
    }
    
    bool StorageFileManager::write_vector_batch(const std::vector<Vector>& vectors) {
        if (!is_open_) {
            return false;
        }
        
        // Convert to storage format
        std::vector<VectorStorageFormat> storage_formats = convert_to_storage_format_batch(vectors);
        
        // In a real implementation, we would write to the file
        // This is a placeholder implementation
        return true;
    }
    
    Vector StorageFileManager::read_vector(const std::string& id) {
        if (!is_open_) {
            return Vector();
        }

        try {
            std::ifstream file(file_path_, std::ios::binary);
            if (!file.is_open()) {
                return Vector();
            }

            // Read through file to find the vector with matching ID
            while (file.good() && !file.eof()) {
                StorageHeader header;
                file.read(reinterpret_cast<char*>(&header), sizeof(header));

                if (file.gcount() != sizeof(header)) {
                    break; // End of file or incomplete header
                }

                // Verify header
                if (!verify_header(header)) {
                    break; // Invalid file format
                }

                // Read the data section
                std::vector<uint8_t> data(header.data_size);
                file.read(reinterpret_cast<char*>(data.data()), header.data_size);

                if (file.gcount() != static_cast<std::streamsize>(header.data_size)) {
                    break; // Incomplete data
                }

                // Verify checksum
                uint64_t calculated_checksum = calculate_checksum(data.data(), data.size());
                if (calculated_checksum != header.checksum) {
                    continue; // Checksum mismatch, skip this entry
                }

                // Deserialize only if this is vector data
                if (header.data_type != VECTOR_DATA) {
                    continue; // Not a vector, skip
                }

                // Deserialize to check ID
                size_t offset = 0;

                // Read ID length
                uint32_t id_length;
                std::memcpy(&id_length, data.data() + offset, sizeof(id_length));
                offset += sizeof(id_length);

                // Read ID
                std::string vector_id(reinterpret_cast<const char*>(data.data() + offset), id_length);
                offset += id_length;

                // Check if this is the vector we're looking for
                if (vector_id != id) {
                    continue; // Not the right vector, keep searching
                }

                // Found it! Deserialize the rest
                VectorStorageFormat storage_format;
                storage_format.id = vector_id;

                // Read dimension
                std::memcpy(&storage_format.dimension, data.data() + offset, sizeof(storage_format.dimension));
                offset += sizeof(storage_format.dimension);

                // Read vector values
                storage_format.values.resize(storage_format.dimension);
                for (uint32_t i = 0; i < storage_format.dimension; ++i) {
                    std::memcpy(&storage_format.values[i], data.data() + offset, sizeof(float));
                    offset += sizeof(float);
                }

                // Read metadata JSON length
                uint32_t metadata_length;
                std::memcpy(&metadata_length, data.data() + offset, sizeof(metadata_length));
                offset += sizeof(metadata_length);

                // Read metadata JSON
                storage_format.metadata_json = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), metadata_length);
                offset += metadata_length;

                // Read timestamps
                std::memcpy(&storage_format.created_timestamp, data.data() + offset,
                          sizeof(storage_format.created_timestamp));
                offset += sizeof(storage_format.created_timestamp);
                std::memcpy(&storage_format.updated_timestamp, data.data() + offset,
                          sizeof(storage_format.updated_timestamp));
                offset += sizeof(storage_format.updated_timestamp);

                // Read version
                std::memcpy(&storage_format.version, data.data() + offset, sizeof(storage_format.version));
                offset += sizeof(storage_format.version);

                // Read deleted flag
                uint8_t deleted_flag = data[offset];
                storage_format.deleted = (deleted_flag != 0);

                file.close();
                return convert_from_storage_format(storage_format);
            }

            file.close();
            return Vector(); // Not found

        } catch (const std::exception& e) {
            return Vector();
        }
    }
    
    Database StorageFileManager::read_database(const std::string& database_id) {
        if (!is_open_) {
            return Database();
        }

        try {
            std::ifstream file(file_path_, std::ios::binary);
            if (!file.is_open()) {
                return Database();
            }

            // Helper lambda to read length-prefixed string
            auto read_string = [](const std::vector<uint8_t>& data, size_t& offset) -> std::string {
                uint32_t length;
                std::memcpy(&length, data.data() + offset, sizeof(length));
                offset += sizeof(length);
                std::string result(reinterpret_cast<const char*>(data.data() + offset), length);
                offset += length;
                return result;
            };

            // Scan through file
            while (file.good() && !file.eof()) {
                StorageHeader header;
                file.read(reinterpret_cast<char*>(&header), sizeof(header));

                if (file.gcount() != sizeof(header)) {
                    break;
                }

                if (!verify_header(header)) {
                    break;
                }

                std::vector<uint8_t> data(header.data_size);
                file.read(reinterpret_cast<char*>(data.data()), header.data_size);

                if (file.gcount() != static_cast<std::streamsize>(header.data_size)) {
                    break;
                }

                if (calculate_checksum(data.data(), data.size()) != header.checksum) {
                    continue;
                }

                if (header.data_type != DATABASE_DATA) {
                    continue;
                }

                // Deserialize to check database_id
                size_t offset = 0;
                std::string db_id = read_string(data, offset);

                if (db_id != database_id) {
                    continue;
                }

                // Found it! Deserialize the rest
                DatabaseStorageFormat storage_format;
                storage_format.database_id = db_id;
                storage_format.name = read_string(data, offset);
                storage_format.description = read_string(data, offset);

                std::memcpy(&storage_format.vector_dimension, data.data() + offset, sizeof(storage_format.vector_dimension));
                offset += sizeof(storage_format.vector_dimension);

                storage_format.index_type = read_string(data, offset);
                storage_format.index_parameters_json = read_string(data, offset);
                storage_format.sharding_config_json = read_string(data, offset);
                storage_format.replication_config_json = read_string(data, offset);
                storage_format.embedding_models_json = read_string(data, offset);
                storage_format.metadata_schema_json = read_string(data, offset);
                storage_format.retention_policy_json = read_string(data, offset);
                storage_format.access_control_json = read_string(data, offset);

                std::memcpy(&storage_format.created_timestamp, data.data() + offset, sizeof(storage_format.created_timestamp));
                offset += sizeof(storage_format.created_timestamp);
                std::memcpy(&storage_format.updated_timestamp, data.data() + offset, sizeof(storage_format.updated_timestamp));

                file.close();
                return convert_from_storage_format(storage_format);
            }

            file.close();
            return Database();

        } catch (const std::exception& e) {
            return Database();
        }
    }
    
    Index StorageFileManager::read_index(const std::string& index_id) {
        // In a real implementation, we would read from the file
        // This is a placeholder implementation
        return Index();
    }
    
    EmbeddingModel StorageFileManager::read_embedding_model(const std::string& model_id) {
        // In a real implementation, we would read from the file
        // This is a placeholder implementation
        return EmbeddingModel();
    }
    
    std::vector<Vector> StorageFileManager::read_vector_batch(const std::vector<std::string>& ids) {
        // In a real implementation, we would read from the file
        // This is a placeholder implementation
        return std::vector<Vector>();
    }
    
    uint64_t StorageFileManager::calculate_checksum(const uint8_t* data, size_t size) {
        // Use CRC32 for better data integrity (zlib provides optimized implementation)
        unsigned long crc = crc32(0L, Z_NULL, 0);
        crc = crc32(crc, data, size);

        // Extend CRC32 to 64-bit by combining with simple hash
        uint64_t extended_checksum = crc;

        // Add a secondary hash for additional verification
        uint64_t secondary_hash = 0;
        for (size_t i = 0; i < size; i += 8) {
            secondary_hash ^= (i < size) ? static_cast<uint64_t>(data[i]) << ((i % 8) * 8) : 0;
        }

        extended_checksum = (extended_checksum << 32) | (secondary_hash & 0xFFFFFFFF);
        return extended_checksum;
    }
    
    StorageHeader StorageFileManager::create_header(DataType data_type, uint64_t data_size) {
        StorageHeader header;
        header.magic_number = MAGIC_NUMBER;
        header.format_version = FORMAT_VERSION;
        header.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        header.data_type = static_cast<uint32_t>(data_type);
        header.reserved = 0;
        header.data_size = data_size;
        header.checksum = 0; // Will be calculated later
        return header;
    }
    
    bool StorageFileManager::verify_header(const StorageHeader& header) {
        return header.magic_number == MAGIC_NUMBER && 
               header.format_version == FORMAT_VERSION;
    }
    
    std::string StorageFileManager::get_file_extension() {
        return ".jdb";
    }
    
    void* StorageFileManager::map_file_region(size_t offset, size_t length) {
        // In a real implementation, we would use memory mapping
        // This is a placeholder implementation
        return nullptr;
    }
    
    bool StorageFileManager::unmap_file_region(void* addr, size_t length) {
        // In a real implementation, we would unmap the region
        // This is a placeholder implementation
        return true;
    }
    
    // Conversion functions implementation
    VectorStorageFormat convert_to_storage_format(const Vector& vector) {
        VectorStorageFormat storage_format;
        storage_format.id = vector.id;
        storage_format.dimension = static_cast<uint32_t>(vector.values.size());
        storage_format.values = vector.values;
        
        // In a real implementation, we would serialize the metadata to JSON
        // This is a placeholder implementation
        storage_format.metadata_json = "{}";
        
        storage_format.created_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        storage_format.updated_timestamp = storage_format.created_timestamp;
        storage_format.version = static_cast<uint32_t>(vector.version);
        storage_format.deleted = vector.deleted;
        
        return storage_format;
    }
    
    Vector convert_from_storage_format(const VectorStorageFormat& storage_format) {
        Vector vector;
        vector.id = storage_format.id;
        vector.values = storage_format.values;
        vector.version = static_cast<int>(storage_format.version);
        vector.deleted = storage_format.deleted;
        
        // In a real implementation, we would deserialize the metadata from JSON
        // This is a placeholder implementation
        vector.metadata.status = "active";
        
        return vector;
    }
    
    DatabaseStorageFormat convert_to_storage_format(const Database& database) {
        DatabaseStorageFormat storage_format;
        storage_format.database_id = database.databaseId;
        storage_format.name = database.name;
        storage_format.description = database.description;
        storage_format.vector_dimension = static_cast<uint32_t>(database.vectorDimension);
        storage_format.index_type = database.indexType;
        
        // In a real implementation, we would serialize the parameters to JSON
        // This is a placeholder implementation
        storage_format.index_parameters_json = "{}";
        storage_format.sharding_config_json = "{}";
        storage_format.replication_config_json = "{}";
        storage_format.embedding_models_json = "[]";
        storage_format.metadata_schema_json = "{}";
        storage_format.retention_policy_json = "{}";
        storage_format.access_control_json = "{}";
        
        storage_format.created_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        storage_format.updated_timestamp = storage_format.created_timestamp;
        
        return storage_format;
    }
    
    Database convert_from_storage_format(const DatabaseStorageFormat& storage_format) {
        Database database;
        database.databaseId = storage_format.database_id;
        database.name = storage_format.name;
        database.description = storage_format.description;
        database.vectorDimension = static_cast<int>(storage_format.vector_dimension);
        database.indexType = storage_format.index_type;
        
        // In a real implementation, we would deserialize from JSON
        // This is a placeholder implementation
        database.sharding.strategy = "hash";
        database.sharding.numShards = 1;
        database.replication.factor = 1;
        database.replication.sync = true;
        
        return database;
    }
    
    IndexStorageFormat convert_to_storage_format(const Index& index) {
        IndexStorageFormat storage_format;
        storage_format.index_id = index.indexId;
        storage_format.database_id = index.databaseId;
        storage_format.type = index.type;
        
        // In a real implementation, we would serialize the parameters to JSON
        // This is a placeholder implementation
        storage_format.parameters_json = "{}";
        
        storage_format.status = index.status;
        storage_format.created_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        storage_format.updated_timestamp = storage_format.created_timestamp;
        storage_format.vector_count = index.stats ? index.stats->vectorCount : 0;
        storage_format.size_bytes = index.stats ? index.stats->sizeBytes : 0;
        
        return storage_format;
    }
    
    Index convert_from_storage_format(const IndexStorageFormat& storage_format) {
        Index index;
        index.indexId = storage_format.index_id;
        index.databaseId = storage_format.database_id;
        index.type = storage_format.type;
        index.status = storage_format.status;
        
        // In a real implementation, we would deserialize from JSON
        // This is a placeholder implementation
        index.stats = std::make_unique<Index::Stats>();
        index.stats->vectorCount = static_cast<int>(storage_format.vector_count);
        index.stats->sizeBytes = static_cast<long long>(storage_format.size_bytes);
        index.stats->buildTimeMs = 0;
        
        return index;
    }
    
    EmbeddingModelStorageFormat convert_to_storage_format(const EmbeddingModel& model) {
        EmbeddingModelStorageFormat storage_format;
        storage_format.model_id = model.modelId;
        storage_format.name = model.name;
        storage_format.version = model.version;
        storage_format.provider = model.provider;
        storage_format.input_type = model.inputType;
        storage_format.output_dimension = static_cast<uint32_t>(model.outputDimension);
        
        // In a real implementation, we would serialize the parameters to JSON
        // This is a placeholder implementation
        storage_format.parameters_json = "{}";
        
        storage_format.status = model.status;
        
        return storage_format;
    }
    
    EmbeddingModel convert_from_storage_format(const EmbeddingModelStorageFormat& storage_format) {
        EmbeddingModel model;
        model.modelId = storage_format.model_id;
        model.name = storage_format.name;
        model.version = storage_format.version;
        model.provider = storage_format.provider;
        model.inputType = storage_format.input_type;
        model.outputDimension = static_cast<int>(storage_format.output_dimension);
        model.status = storage_format.status;
        
        // In a real implementation, we would deserialize from JSON
        // This is a placeholder implementation
        
        return model;
    }
    
    std::vector<VectorStorageFormat> convert_to_storage_format_batch(const std::vector<Vector>& vectors) {
        std::vector<VectorStorageFormat> storage_formats;
        storage_formats.reserve(vectors.size());
        
        for (const auto& vector : vectors) {
            storage_formats.push_back(convert_to_storage_format(vector));
        }
        
        return storage_formats;
    }
    
    std::vector<Vector> convert_from_storage_format_batch(const std::vector<VectorStorageFormat>& storage_formats) {
        std::vector<Vector> vectors;
        vectors.reserve(storage_formats.size());
        
        for (const auto& storage_format : storage_formats) {
            vectors.push_back(convert_from_storage_format(storage_format));
        }
        
        return vectors;
    }
    
    // Utility functions for storage optimization
    size_t estimate_storage_size(const Vector& vector) {
        return sizeof(VectorStorageFormat) + 
               vector.id.size() + 
               vector.values.size() * sizeof(float) +
               1024; // Approximate size for metadata JSON
    }
    
    size_t estimate_storage_size(const Database& database) {
        return sizeof(DatabaseStorageFormat) + 
               database.databaseId.size() + 
               database.name.size() + 
               database.description.size() +
               2048; // Approximate size for JSON serialized fields
    }
    
    size_t estimate_storage_size(const Index& index) {
        return sizeof(IndexStorageFormat) + 
               index.indexId.size() + 
               index.databaseId.size() + 
               index.type.size() + 
               index.status.size() +
               1024; // Approximate size for JSON serialized fields
    }
    
    size_t estimate_storage_size(const EmbeddingModel& model) {
        return sizeof(EmbeddingModelStorageFormat) + 
               model.modelId.size() + 
               model.name.size() + 
               model.version.size() + 
               model.provider.size() + 
               model.inputType.size() +
               1024; // Approximate size for JSON serialized fields
    }
    
    // Compression utilities
    std::vector<uint8_t> compress_data(const uint8_t* data, size_t size) {
        // In a real implementation, we would use zlib or another compression library
        // This is a placeholder implementation that returns uncompressed data
        return std::vector<uint8_t>(data, data + size);
    }
    
    std::vector<uint8_t> decompress_data(const uint8_t* compressed_data, size_t compressed_size, size_t original_size) {
        // In a real implementation, we would use zlib or another compression library
        // This is a placeholder implementation that returns the data as-is
        return std::vector<uint8_t>(compressed_data, compressed_data + compressed_size);
    }

} // namespace storage_format

} // namespace jadevectordb