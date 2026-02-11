#include "storage_format.h"
#include <fstream>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <set>
#include <unordered_set>
#include <zlib.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#endif
#include <nlohmann/json.hpp>  // For JSON serialization/deserialization
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

            // Timestamps not present in struct - skipping serialization
            // If timestamps are needed, add them to EmbeddingModelStorageFormat struct first

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

            // Timestamps not present in struct - skipping serialization
            // If timestamps are needed, add them to EmbeddingModelStorageFormat struct first

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

        try {
            // Convert to storage format
            IndexStorageFormat storage_format = convert_to_storage_format(index);

            // Serialize the index data to binary format
            std::vector<uint8_t> serialized_data;

            // Write index_id (length + data)
            uint32_t id_length = static_cast<uint32_t>(storage_format.index_id.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&id_length),
                                 reinterpret_cast<const uint8_t*>(&id_length) + sizeof(id_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.index_id.begin(),
                                 storage_format.index_id.end());

            // Write database_id (length + data)
            uint32_t db_id_length = static_cast<uint32_t>(storage_format.database_id.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&db_id_length),
                                 reinterpret_cast<const uint8_t*>(&db_id_length) + sizeof(db_id_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.database_id.begin(),
                                 storage_format.database_id.end());

            // Write type (length + data)
            uint32_t type_length = static_cast<uint32_t>(storage_format.type.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&type_length),
                                 reinterpret_cast<const uint8_t*>(&type_length) + sizeof(type_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.type.begin(),
                                 storage_format.type.end());

            // Write parameters JSON (length + data)
            uint32_t params_length = static_cast<uint32_t>(storage_format.parameters_json.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&params_length),
                                 reinterpret_cast<const uint8_t*>(&params_length) + sizeof(params_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.parameters_json.begin(),
                                 storage_format.parameters_json.end());

            // Write status (length + data)
            uint32_t status_length = static_cast<uint32_t>(storage_format.status.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&status_length),
                                 reinterpret_cast<const uint8_t*>(&status_length) + sizeof(status_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.status.begin(),
                                 storage_format.status.end());

            // Timestamps not present in struct - skipping serialization
            // If timestamps are needed, add them to EmbeddingModelStorageFormat struct first

            // Write counts and sizes
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&storage_format.vector_count),
                                 reinterpret_cast<const uint8_t*>(&storage_format.vector_count) + sizeof(storage_format.vector_count));
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&storage_format.size_bytes),
                                 reinterpret_cast<const uint8_t*>(&storage_format.size_bytes) + sizeof(storage_format.size_bytes));

            // Create header with checksum
            StorageHeader header = create_header(INDEX_DATA, serialized_data.size());
            header.checksum = calculate_checksum(serialized_data.data(), serialized_data.size());

            // Write to file
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
    
    bool StorageFileManager::write_embedding_model(const EmbeddingModel& model) {
        if (!is_open_) {
            return false;
        }

        try {
            // Convert to storage format
            EmbeddingModelStorageFormat storage_format = convert_to_storage_format(model);

            // Serialize the model data to binary format
            std::vector<uint8_t> serialized_data;

            // Write model_id (length + data)
            uint32_t id_length = static_cast<uint32_t>(storage_format.model_id.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&id_length),
                                 reinterpret_cast<const uint8_t*>(&id_length) + sizeof(id_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.model_id.begin(),
                                 storage_format.model_id.end());

            // Write name (length + data)
            uint32_t name_length = static_cast<uint32_t>(storage_format.name.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&name_length),
                                 reinterpret_cast<const uint8_t*>(&name_length) + sizeof(name_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.name.begin(),
                                 storage_format.name.end());

            // Write version (length + data)
            uint32_t version_length = static_cast<uint32_t>(storage_format.version.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&version_length),
                                 reinterpret_cast<const uint8_t*>(&version_length) + sizeof(version_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.version.begin(),
                                 storage_format.version.end());

            // Write provider (length + data)
            uint32_t provider_length = static_cast<uint32_t>(storage_format.provider.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&provider_length),
                                 reinterpret_cast<const uint8_t*>(&provider_length) + sizeof(provider_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.provider.begin(),
                                 storage_format.provider.end());

            // Write input_type (length + data)
            uint32_t input_type_length = static_cast<uint32_t>(storage_format.input_type.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&input_type_length),
                                 reinterpret_cast<const uint8_t*>(&input_type_length) + sizeof(input_type_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.input_type.begin(),
                                 storage_format.input_type.end());

            // Write output_dimension
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&storage_format.output_dimension),
                                 reinterpret_cast<const uint8_t*>(&storage_format.output_dimension) + sizeof(storage_format.output_dimension));

            // Write parameters_json (length + data)
            uint32_t params_length = static_cast<uint32_t>(storage_format.parameters_json.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&params_length),
                                 reinterpret_cast<const uint8_t*>(&params_length) + sizeof(params_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.parameters_json.begin(),
                                 storage_format.parameters_json.end());

            // Write status (length + data)
            uint32_t status_length = static_cast<uint32_t>(storage_format.status.size());
            serialized_data.insert(serialized_data.end(),
                                 reinterpret_cast<const uint8_t*>(&status_length),
                                 reinterpret_cast<const uint8_t*>(&status_length) + sizeof(status_length));
            serialized_data.insert(serialized_data.end(),
                                 storage_format.status.begin(),
                                 storage_format.status.end());

            // Timestamps not present in struct - skipping serialization
            // If timestamps are needed, add them to EmbeddingModelStorageFormat struct first

            // Create header with checksum
            StorageHeader header = create_header(EMBEDDING_MODEL_DATA, serialized_data.size());
            header.checksum = calculate_checksum(serialized_data.data(), serialized_data.size());

            // Write to file
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
    
    bool StorageFileManager::write_vector_batch(const std::vector<Vector>& vectors) {
        if (!is_open_ || vectors.empty()) {
            return false;
        }

        try {
            // Convert all vectors to storage format
            std::vector<VectorStorageFormat> storage_formats = convert_to_storage_format_batch(vectors);

            // Write each vector sequentially
            for (const auto& storage_format : storage_formats) {
                // Serialize the vector data to binary format
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

                // Write to file
                std::ofstream file(file_path_, std::ios::binary | std::ios::app);
                if (!file.is_open()) {
                    return false;
                }

                // Write header and data
                file.write(reinterpret_cast<const char*>(&header), sizeof(header));
                file.write(reinterpret_cast<const char*>(serialized_data.data()), serialized_data.size());

                file.close();
            }

            return true;

        } catch (const std::exception& e) {
            return false;
        }
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
        if (!is_open_) {
            return Index();
        }

        try {
            std::ifstream file(file_path_, std::ios::binary);
            if (!file.is_open()) {
                return Index();
            }

            // Search through file for the index with matching ID
            while (file.good() && !file.eof()) {
                StorageHeader header;
                file.read(reinterpret_cast<char*>(&header), sizeof(header));

                if (file.gcount() != sizeof(header)) {
                    break; // End of file or incomplete header
                }

                // Verify header
                if (!verify_header(header)) {
                    continue; // Invalid header, skip
                }

                // Skip if this is not index data
                if (header.data_type != INDEX_DATA) {
                    file.seekg(header.data_size, std::ios::cur); // Skip this entry
                    continue;
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

                // Deserialize the index data
                size_t offset = 0;

                // Read index_id length
                uint32_t id_length;
                std::memcpy(&id_length, data.data() + offset, sizeof(id_length));
                offset += sizeof(id_length);

                // Read index_id
                std::string read_index_id(
                    reinterpret_cast<const char*>(data.data() + offset), id_length);
                offset += id_length;

                // Check if this is the index we're looking for
                if (read_index_id != index_id) {
                    continue; // Not the right index, keep searching
                }

                // We found the index! Now deserialize the rest
                IndexStorageFormat storage_format;
                storage_format.index_id = read_index_id;

                // Read database_id length
                uint32_t db_id_length;
                std::memcpy(&db_id_length, data.data() + offset, sizeof(db_id_length));
                offset += sizeof(db_id_length);

                // Read database_id
                storage_format.database_id = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), db_id_length);
                offset += db_id_length;

                // Read type length
                uint32_t type_length;
                std::memcpy(&type_length, data.data() + offset, sizeof(type_length));
                offset += sizeof(type_length);

                // Read type
                storage_format.type = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), type_length);
                offset += type_length;

                // Read parameters JSON length
                uint32_t params_length;
                std::memcpy(&params_length, data.data() + offset, sizeof(params_length));
                offset += sizeof(params_length);

                // Read parameters JSON
                storage_format.parameters_json = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), params_length);
                offset += params_length;

                // Read status length
                uint32_t status_length;
                std::memcpy(&status_length, data.data() + offset, sizeof(status_length));
                offset += sizeof(status_length);

                // Read status
                storage_format.status = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), status_length);
                offset += status_length;

                // Read timestamp
                std::memcpy(&storage_format.created_timestamp, data.data() + offset, sizeof(storage_format.created_timestamp));
                offset += sizeof(storage_format.created_timestamp);

                // Read updated timestamp
                std::memcpy(&storage_format.updated_timestamp, data.data() + offset, sizeof(storage_format.updated_timestamp));
                offset += sizeof(storage_format.updated_timestamp);

                // Read vector count
                std::memcpy(&storage_format.vector_count, data.data() + offset, sizeof(storage_format.vector_count));
                offset += sizeof(storage_format.vector_count);

                // Read size in bytes
                std::memcpy(&storage_format.size_bytes, data.data() + offset, sizeof(storage_format.size_bytes));

                // Convert back to Index object
                Index index;
                index.indexId = storage_format.index_id;
                index.databaseId = storage_format.database_id;
                index.type = storage_format.type;
                index.status = storage_format.status;

                // Set up statistics
                index.stats = std::make_unique<Index::Stats>();
                index.stats->vectorCount = static_cast<int>(storage_format.vector_count);
                index.stats->sizeBytes = static_cast<long long>(storage_format.size_bytes);

                file.close();
                return index;

            } // end while loop

            file.close();
            return Index(); // Not found

        } catch (const std::exception& e) {
            return Index(); // Return empty index on error
        }
    }
    
    EmbeddingModel StorageFileManager::read_embedding_model(const std::string& model_id) {
        if (!is_open_) {
            return EmbeddingModel();
        }

        try {
            std::ifstream file(file_path_, std::ios::binary);
            if (!file.is_open()) {
                return EmbeddingModel();
            }

            // Search through file for the model with matching ID
            while (file.good() && !file.eof()) {
                StorageHeader header;
                file.read(reinterpret_cast<char*>(&header), sizeof(header));

                if (file.gcount() != sizeof(header)) {
                    break; // End of file or incomplete header
                }

                // Verify header
                if (!verify_header(header)) {
                    continue; // Invalid header, skip
                }

                // Skip if this is not embedding model data
                if (header.data_type != EMBEDDING_MODEL_DATA) {
                    file.seekg(header.data_size, std::ios::cur); // Skip this entry
                    continue;
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

                // Deserialize the model data
                size_t offset = 0;

                // Read model_id length
                uint32_t id_length;
                std::memcpy(&id_length, data.data() + offset, sizeof(id_length));
                offset += sizeof(id_length);

                // Read model_id
                std::string read_model_id(
                    reinterpret_cast<const char*>(data.data() + offset), id_length);
                offset += id_length;

                // Check if this is the model we're looking for
                if (read_model_id != model_id) {
                    continue; // Not the right model, keep searching
                }

                // We found the model! Now deserialize the rest
                EmbeddingModelStorageFormat storage_format;
                storage_format.model_id = read_model_id;

                // Read name length
                uint32_t name_length;
                std::memcpy(&name_length, data.data() + offset, sizeof(name_length));
                offset += sizeof(name_length);

                storage_format.name = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), name_length);
                offset += name_length;

                // Read version length
                uint32_t version_length;
                std::memcpy(&version_length, data.data() + offset, sizeof(version_length));
                offset += sizeof(version_length);

                storage_format.version = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), version_length);
                offset += version_length;

                // Read provider length
                uint32_t provider_length;
                std::memcpy(&provider_length, data.data() + offset, sizeof(provider_length));
                offset += sizeof(provider_length);

                storage_format.provider = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), provider_length);
                offset += provider_length;

                // Read input_type length
                uint32_t input_type_length;
                std::memcpy(&input_type_length, data.data() + offset, sizeof(input_type_length));
                offset += sizeof(input_type_length);

                storage_format.input_type = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), input_type_length);
                offset += input_type_length;

                // Read output_dimension
                std::memcpy(&storage_format.output_dimension, data.data() + offset, sizeof(storage_format.output_dimension));
                offset += sizeof(storage_format.output_dimension);

                // Read parameters_json length
                uint32_t params_length;
                std::memcpy(&params_length, data.data() + offset, sizeof(params_length));
                offset += sizeof(params_length);

                storage_format.parameters_json = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), params_length);
                offset += params_length;

                // Read status length
                uint32_t status_length;
                std::memcpy(&status_length, data.data() + offset, sizeof(status_length));
                offset += sizeof(status_length);

                storage_format.status = std::string(
                    reinterpret_cast<const char*>(data.data() + offset), status_length);
                offset += status_length;

                // Timestamps not present in struct - skipping deserialization
                // If timestamps were serialized, we need to skip them here
                // For now, we assume they weren't serialized (see write_embedding_model)

                // Convert back to EmbeddingModel object
                EmbeddingModel model;
                model.modelId = storage_format.model_id;
                model.name = storage_format.name;
                model.version = storage_format.version;
                model.provider = storage_format.provider;
                model.inputType = storage_format.input_type;
                model.outputDimension = static_cast<int>(storage_format.output_dimension);
                model.status = storage_format.status;

                file.close();
                return model;

            } // end while loop

            file.close();
            return EmbeddingModel(); // Not found

        } catch (const std::exception& e) {
            return EmbeddingModel(); // Return empty model on error
        }
    }
    
    std::vector<Vector> StorageFileManager::read_vector_batch(const std::vector<std::string>& ids) {
        if (!is_open_ || ids.empty()) {
            return std::vector<Vector>();
        }

        std::vector<Vector> result;
        std::unordered_set<std::string> id_set(ids.begin(), ids.end()); // For efficient lookup

        try {
            std::ifstream file(file_path_, std::ios::binary);
            if (!file.is_open()) {
                return std::vector<Vector>();
            }

            // Search through the file for vectors with matching IDs
            while (file.good() && !file.eof() && result.size() < ids.size()) {
                StorageHeader header;
                file.read(reinterpret_cast<char*>(&header), sizeof(header));

                if (file.gcount() != sizeof(header)) {
                    break; // End of file or incomplete header
                }

                // Verify header
                if (!verify_header(header)) {
                    file.seekg(header.data_size, std::ios::cur); // Skip this invalid entry
                    continue;
                }

                // Skip if this is not vector data
                if (header.data_type != VECTOR_DATA) {
                    file.seekg(header.data_size, std::ios::cur); // Skip this entry
                    continue;
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

                // Deserialize vector ID to check if it's one we're looking for
                size_t offset = 0;

                // Read vector ID length
                uint32_t id_length;
                std::memcpy(&id_length, data.data() + offset, sizeof(id_length));
                offset += sizeof(id_length);

                // Read vector ID
                std::string vector_id(
                    reinterpret_cast<const char*>(data.data() + offset), id_length);
                offset += id_length;

                // Check if this is one of the vectors we're looking for
                if (id_set.count(vector_id) > 0) {
                    // This is a vector we need - deserialize the complete vector
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
                    std::memcpy(&storage_format.created_timestamp, data.data() + offset, sizeof(storage_format.created_timestamp));
                    offset += sizeof(storage_format.created_timestamp);

                    std::memcpy(&storage_format.updated_timestamp, data.data() + offset, sizeof(storage_format.updated_timestamp));
                    offset += sizeof(storage_format.updated_timestamp);

                    // Read version
                    std::memcpy(&storage_format.version, data.data() + offset, sizeof(storage_format.version));
                    offset += sizeof(storage_format.version);

                    // Read deleted flag
                    uint8_t deleted_flag;
                    std::memcpy(&deleted_flag, data.data() + offset, sizeof(deleted_flag));
                    storage_format.deleted = (deleted_flag != 0);

                    // Convert to Vector and add to result
                    Vector vector = convert_from_storage_format(storage_format);
                    result.push_back(vector);
                }
            }

            file.close();
            return result;

        } catch (const std::exception& e) {
            return std::vector<Vector>(); // Return empty vector on error
        }
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
        // Use mmap for memory mapping of a specific region of the file
#ifdef __linux__
        int fd = open(file_path_.c_str(), O_RDONLY);
        if (fd < 0) {
            return nullptr;
        }

        // Map the file region into memory
        void* mapped_addr = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, static_cast<off_t>(offset));

        // Close the file descriptor after mapping (the mapping stays)
        close(fd);

        if (mapped_addr == MAP_FAILED) {
            return nullptr;
        }

        return mapped_addr;
#else
        // On non-Linux systems, we might use different APIs
        // For now, return null since cross-platform memory mapping is complex
        return nullptr;
#endif
    }

    bool StorageFileManager::unmap_file_region(void* addr, size_t length) {
        // Unmap the previously mapped region
        if (addr == nullptr) {
            return false;
        }

#ifdef __linux__
        int result = munmap(addr, length);
        return (result == 0);
#else
        // On non-Linux systems, use appropriate API
        // For now, return true as a placeholder
        return true;
#endif
    }
    
    // Conversion functions implementation
    VectorStorageFormat convert_to_storage_format(const Vector& vector) {
        VectorStorageFormat storage_format;
        storage_format.id = vector.id;
        storage_format.dimension = static_cast<uint32_t>(vector.values.size());
        storage_format.values = vector.values;
        
        // In a real implementation, we would serialize the metadata to JSON
        // This is a placeholder implementation
        // Serialize the metadata to JSON using nlohmann json
        nlohmann::json metadata_json;
        metadata_json["source"] = vector.metadata.source;
        metadata_json["owner"] = vector.metadata.owner;
        metadata_json["category"] = vector.metadata.category;
        metadata_json["score"] = vector.metadata.score;
        metadata_json["status"] = vector.metadata.status;

        // Convert tags vector to JSON array
        nlohmann::json tags_array = nlohmann::json::array();
        for (const auto& tag : vector.metadata.tags) {
            tags_array.push_back(tag);
        }
        metadata_json["tags"] = tags_array;

        storage_format.metadata_json = metadata_json.dump();
        
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
        try {
            nlohmann::json metadata_json = nlohmann::json::parse(storage_format.metadata_json);

            vector.metadata.source = metadata_json.value("source", "");
            vector.metadata.owner = metadata_json.value("owner", "");
            vector.metadata.category = metadata_json.value("category", "");
            vector.metadata.score = metadata_json.value("score", 0.0f);
            vector.metadata.status = metadata_json.value("status", "active");

            // Parse tags array from JSON
            if (metadata_json.contains("tags") && metadata_json["tags"].is_array()) {
                const auto& tags_array = metadata_json["tags"];
                for (const auto& tag : tags_array) {
                    vector.metadata.tags.push_back(tag.get<std::string>());
                }
            }
        } catch (const std::exception& e) {
            // If deserialization fails, set default values
            vector.metadata.status = "active";
        }
        
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
        try {
            nlohmann::json index_params;
            index_params["type"] = database.indexType;
            index_params["dimensions"] = database.vectorDimension;
            storage_format.index_parameters_json = index_params.dump();

            nlohmann::json sharding_config;
            sharding_config["strategy"] = database.sharding.strategy;
            sharding_config["num_shards"] = database.sharding.numShards;
            storage_format.sharding_config_json = sharding_config.dump();

            nlohmann::json replication_config;
            replication_config["factor"] = database.replication.factor;
            replication_config["sync"] = database.replication.sync;
            storage_format.replication_config_json = replication_config.dump();
        } catch (const std::exception& e) {
            // If serialization fails, use default empty JSON
            storage_format.index_parameters_json = "{}";
            storage_format.sharding_config_json = "{}";
            storage_format.replication_config_json = "{}";
        }
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
        try {
            nlohmann::json index_params = nlohmann::json::parse(storage_format.index_parameters_json);
            database.indexType = index_params.value("type", database.indexType);
            database.vectorDimension = index_params.value("dimensions", database.vectorDimension);

            nlohmann::json sharding_config = nlohmann::json::parse(storage_format.sharding_config_json);
            database.sharding.strategy = sharding_config.value("strategy", "hash");
            database.sharding.numShards = sharding_config.value("num_shards", 1);

            nlohmann::json replication_config = nlohmann::json::parse(storage_format.replication_config_json);
            database.replication.factor = replication_config.value("factor", 1);
            database.replication.sync = replication_config.value("sync", true);
        } catch (const std::exception& e) {
            // If deserialization fails, use default values
            database.sharding.strategy = "hash";
            database.sharding.numShards = 1;
            database.replication.factor = 1;
            database.replication.sync = true;
        }
        
        return database;
    }
    
    IndexStorageFormat convert_to_storage_format(const Index& index) {
        IndexStorageFormat storage_format;
        storage_format.index_id = index.indexId;
        storage_format.database_id = index.databaseId;
        storage_format.type = index.type;

        // Serialize the parameters to JSON using nlohmann json
        try {
            nlohmann::json params_json;
            params_json["type"] = index.type;
            params_json["status"] = index.status;

            // Add index-specific parameters to the JSON (if available)
            if (!index.parameters.empty()) {
                params_json["params"] = index.parameters;
            } else {
                // Default parameters if no params available
                params_json["params"] = {
                    {"algorithm", index.type},
                    {"metric", "cosine"}
                };
            }

            storage_format.parameters_json = params_json.dump();
        } catch (const std::exception& e) {
            // If serialization fails, use default empty JSON
            storage_format.parameters_json = "{}";
        }

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
        try {
            nlohmann::json params_json = nlohmann::json::parse(storage_format.parameters_json);

            // Extract index parameters if available
            if (params_json.contains("params")) {
                // In a full implementation, we'd deserialize the actual parameters
                // For now, extracting basic values if they exist
                if (params_json["params"].contains("build_time_ms")) {
                    index.stats->buildTimeMs = params_json["params"]["build_time_ms"].get<long long>();
                }
            }
        } catch (const std::exception& e) {
            // If deserialization fails, use default values
            index.stats = std::make_unique<Index::Stats>();
            index.stats->buildTimeMs = 0;
        }

        if (!index.stats) {
            index.stats = std::make_unique<Index::Stats>();
        }

        index.stats->vectorCount = static_cast<int>(storage_format.vector_count);
        index.stats->sizeBytes = static_cast<long long>(storage_format.size_bytes);
        if (index.stats->buildTimeMs == 0) {
            index.stats->buildTimeMs = 0; // Default if not set during deserialization
        }
        
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
        
        // Serialize the parameters to JSON using nlohmann json
        try {
            nlohmann::json params_json;
            params_json["name"] = model.name;
            params_json["dimensions"] = model.outputDimension;
            params_json["provider"] = model.provider;
            params_json["input_type"] = model.inputType;
            params_json["version"] = model.version;

            // Add model parameters if available
            if (!model.parameters.empty()) {
                params_json["parameters"] = model.parameters;
            }

            storage_format.parameters_json = params_json.dump();
        } catch (const std::exception& e) {
            // If serialization fails, use default empty JSON
            storage_format.parameters_json = "{}";
        }
        
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
        try {
            nlohmann::json params_json = nlohmann::json::parse(storage_format.parameters_json);

            // Set model properties from JSON if available
            if (params_json.contains("parameters")) {
                // Parse parameters map
                auto params_obj = params_json["parameters"];
                for (auto& [key, value] : params_obj.items()) {
                    model.parameters[key] = value.dump();
                }
            }
        } catch (const std::exception& e) {
            // If deserialization fails, continue with default values
        }

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
        if (!data || size == 0) {
            return std::vector<uint8_t>();
        }

#ifdef ZLIB_AVAILABLE
        // Use zlib for compression
        uLong source_len = static_cast<uLong>(size);
        uLong dest_len = compressBound(source_len);  // Maximum possible compressed size

        std::vector<uint8_t> compressed_data(dest_len);

        int result = compress(
            compressed_data.data(),
            &dest_len,
            data,
            source_len
        );

        if (result == Z_OK) {
            // Resize to actual compressed size
            compressed_data.resize(dest_len);
            return compressed_data;
        } else {
            // Compression failed, return original data
            return std::vector<uint8_t>(data, data + size);
        }
#else
        // Without zlib, return original data
        return std::vector<uint8_t>(data, data + size);
#endif
    }

    std::vector<uint8_t> decompress_data(const uint8_t* compressed_data, size_t compressed_size, size_t original_size) {
        if (!compressed_data || compressed_size == 0) {
            return std::vector<uint8_t>();
        }

#ifdef ZLIB_AVAILABLE
        // Use zlib for decompression
        uLong dest_len = static_cast<uLong>(original_size);
        std::vector<uint8_t> decompressed_data(original_size);

        int result = uncompress(
            decompressed_data.data(),
            &dest_len,
            compressed_data,
            static_cast<uLong>(compressed_size)
        );

        if (result == Z_OK) {
            // Successfully decompressed
            return decompressed_data;
        } else {
            // Decompression failed, return empty vector
            return std::vector<uint8_t>();
        }
#else
        // Without zlib, return compressed data as-is (though this is unusual)
        return std::vector<uint8_t>(compressed_data, compressed_data + compressed_size);
#endif
    }

} // namespace storage_format

} // namespace jadevectordb