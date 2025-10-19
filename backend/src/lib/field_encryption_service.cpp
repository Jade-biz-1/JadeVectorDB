#include "field_encryption_service.h"
#include "../models/vector.h"
#include <stdexcept>

namespace jadevectordb {
namespace encryption {

FieldEncryptionServiceImpl::FieldEncryptionServiceImpl(std::unique_ptr<EncryptionManager> encryption_manager)
    : encryption_manager_(std::move(encryption_manager)) {
}

std::vector<uint8_t> FieldEncryptionServiceImpl::encrypt_field(const std::string& field_path,
                                                              const std::vector<uint8_t>& field_value,
                                                              const std::map<std::string, std::string>& context) {
    auto it = field_configs_.find(field_path);
    if (it == field_configs_.end()) {
        // If no configuration exists for this field, return the original value
        return field_value;
    }
    
    // Use the encryption manager to encrypt the field
    const auto& field_config = it->second;
    
    // Generate field-specific context for additional authentication
    auto field_context = generate_field_context(field_path, context);
    
    // Perform encryption using the field configuration
    return encryption_manager_->encrypt_data(field_value, field_config.encryption_config);
}

std::vector<uint8_t> FieldEncryptionServiceImpl::decrypt_field(const std::string& field_path,
                                                              const std::vector<uint8_t>& encrypted_value,
                                                              const std::map<std::string, std::string>& context) {
    auto it = field_configs_.find(field_path);
    if (it == field_configs_.end()) {
        // If no configuration exists for this field, return the original value
        return encrypted_value;
    }
    
    // Use the encryption manager to decrypt the field
    const auto& field_config = it->second;
    
    // Perform decryption using the field configuration
    return encryption_manager_->decrypt_data(encrypted_value, field_config.encryption_config);
}

bool FieldEncryptionServiceImpl::is_field_encrypted(const std::string& field_path) const {
    return field_configs_.find(field_path) != field_configs_.end();
}

EncryptionConfig FieldEncryptionServiceImpl::get_field_encryption_config(const std::string& field_path) const {
    auto it = field_configs_.find(field_path);
    if (it == field_configs_.end()) {
        throw std::runtime_error("Field not configured for encryption: " + field_path);
    }
    
    return it->second.encryption_config;
}

void FieldEncryptionServiceImpl::configure_field(const std::string& field_path, 
                                                const EncryptionConfig& config) {
    FieldEncryptionConfig field_config;
    field_config.field_name = field_path;
    field_config.encryption_config = config;
    field_config.enable_deterministic_encryption = config.searchable_encryption;  // Use searchable setting as deterministic indicator
    field_config.searchable = config.searchable_encryption;
    
    field_configs_[field_path] = field_config;
    
    // Also configure the encryption manager for this field
    encryption_manager_->configure_field_encryption(field_config);
}

void FieldEncryptionServiceImpl::remove_field_configuration(const std::string& field_path) {
    field_configs_.erase(field_path);
}

std::vector<uint8_t> FieldEncryptionServiceImpl::generate_field_context(const std::string& field_path,
                                                                       const std::map<std::string, std::string>& context) const {
    // Create a context string that combines the field path and additional context
    std::string combined_context = field_path;
    for (const auto& pair : context) {
        combined_context += "|" + pair.first + "=" + pair.second;
    }
    
    // Convert to bytes
    return std::vector<uint8_t>(combined_context.begin(), combined_context.end());
}

// VectorDataEncryptor implementation
VectorDataEncryptor::VectorDataEncryptor(std::shared_ptr<FieldEncryptionServiceImpl> field_encryption_service)
    : field_encryption_service_(field_encryption_service) {
}

jadevectordb::Vector VectorDataEncryptor::encrypt_vector(const jadevectordb::Vector& vector) {
    jadevectordb::Vector encrypted_vector = vector;  // Start with a copy of the original
    
    // Encrypt vector values if configured
    if (field_encryption_service_->is_field_encrypted("vector.values")) {
        std::vector<uint8_t> values_bytes;
        values_bytes.reserve(vector.values.size() * sizeof(float));
        
        // Serialize float values to bytes
        for (float val : vector.values) {
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&val);
            values_bytes.insert(values_bytes.end(), bytes, bytes + sizeof(float));
        }
        
        auto encrypted_bytes = field_encryption_service_->encrypt_field("vector.values", values_bytes);
        
        // Deserialize bytes back to float values
        encrypted_vector.values.clear();
        for (size_t i = 0; i < encrypted_bytes.size(); i += sizeof(float)) {
            if (i + sizeof(float) <= encrypted_bytes.size()) {
                float val;
                std::memcpy(&val, encrypted_bytes.data() + i, sizeof(float));
                encrypted_vector.values.push_back(val);
            }
        }
    }
    
    // Encrypt metadata fields if configured
    encrypted_vector.metadata = encrypt_metadata_fields(vector.metadata);
    
    return encrypted_vector;
}

jadevectordb::Vector VectorDataEncryptor::decrypt_vector(const jadevectordb::Vector& encrypted_vector) {
    jadevectordb::Vector decrypted_vector = encrypted_vector;  // Start with a copy of the encrypted data
    
    // Decrypt vector values if configured
    if (field_encryption_service_->is_field_encrypted("vector.values")) {
        std::vector<uint8_t> encrypted_values_bytes;
        encrypted_values_bytes.reserve(encrypted_vector.values.size() * sizeof(float));
        
        // Serialize encrypted float values to bytes
        for (float val : encrypted_vector.values) {
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&val);
            encrypted_values_bytes.insert(encrypted_values_bytes.end(), bytes, bytes + sizeof(float));
        }
        
        auto decrypted_bytes = field_encryption_service_->decrypt_field("vector.values", encrypted_values_bytes);
        
        // Deserialize bytes back to float values
        decrypted_vector.values.clear();
        for (size_t i = 0; i < decrypted_bytes.size(); i += sizeof(float)) {
            if (i + sizeof(float) <= decrypted_bytes.size()) {
                float val;
                std::memcpy(&val, decrypted_bytes.data() + i, sizeof(float));
                decrypted_vector.values.push_back(val);
            }
        }
    }
    
    // Decrypt metadata fields if configured
    decrypted_vector.metadata = decrypt_metadata_fields(encrypted_vector.metadata);
    
    return decrypted_vector;
}

jadevectordb::Vector::Metadata VectorDataEncryptor::encrypt_metadata_fields(const jadevectordb::Vector::Metadata& metadata) {
    jadevectordb::Vector::Metadata encrypted_metadata = metadata;  // Start with a copy
    
    // Encrypt the source field if configured
    if (field_encryption_service_->is_field_encrypted("vector.metadata.source")) {
        std::vector<uint8_t> source_bytes(metadata.source.begin(), metadata.source.end());
        auto encrypted_source = field_encryption_service_->encrypt_field("vector.metadata.source", source_bytes);
        encrypted_metadata.source = std::string(encrypted_source.begin(), encrypted_source.end());
    }
    
    // Encrypt the owner field if configured
    if (field_encryption_service_->is_field_encrypted("vector.metadata.owner")) {
        std::vector<uint8_t> owner_bytes(metadata.owner.begin(), metadata.owner.end());
        auto encrypted_owner = field_encryption_service_->encrypt_field("vector.metadata.owner", owner_bytes);
        encrypted_metadata.owner = std::string(encrypted_owner.begin(), encrypted_owner.end());
    }
    
    // Encrypt tags if configured
    if (field_encryption_service_->is_field_encrypted("vector.metadata.tags")) {
        // For simplicity, we'll encrypt each tag individually
        // In a real implementation, we might encrypt the whole list
        for (auto& tag : encrypted_metadata.tags) {
            if (field_encryption_service_->is_field_encrypted("vector.metadata.tags.element")) {
                std::vector<uint8_t> tag_bytes(tag.begin(), tag.end());
                auto encrypted_tag = field_encryption_service_->encrypt_field("vector.metadata.tags.element", tag_bytes);
                tag = std::string(encrypted_tag.begin(), encrypted_tag.end());
            }
        }
    }
    
    return encrypted_metadata;
}

jadevectordb::Vector::Metadata VectorDataEncryptor::decrypt_metadata_fields(const jadevectordb::Vector::Metadata& encrypted_metadata) {
    jadevectordb::Vector::Metadata decrypted_metadata = encrypted_metadata;  // Start with a copy
    
    // Decrypt the source field if configured
    if (field_encryption_service_->is_field_encrypted("vector.metadata.source")) {
        std::vector<uint8_t> encrypted_source_bytes(encrypted_metadata.source.begin(), encrypted_metadata.source.end());
        auto source_bytes = field_encryption_service_->decrypt_field("vector.metadata.source", encrypted_source_bytes);
        decrypted_metadata.source = std::string(source_bytes.begin(), source_bytes.end());
    }
    
    // Decrypt the owner field if configured
    if (field_encryption_service_->is_field_encrypted("vector.metadata.owner")) {
        std::vector<uint8_t> encrypted_owner_bytes(encrypted_metadata.owner.begin(), encrypted_metadata.owner.end());
        auto owner_bytes = field_encryption_service_->decrypt_field("vector.metadata.owner", encrypted_owner_bytes);
        decrypted_metadata.owner = std::string(owner_bytes.begin(), owner_bytes.end());
    }
    
    // Decrypt tags if configured
    if (field_encryption_service_->is_field_encrypted("vector.metadata.tags")) {
        // For simplicity, we'll decrypt each tag individually
        for (auto& tag : decrypted_metadata.tags) {
            if (field_encryption_service_->is_field_encrypted("vector.metadata.tags.element")) {
                std::vector<uint8_t> encrypted_tag_bytes(tag.begin(), tag.end());
                auto tag_bytes = field_encryption_service_->decrypt_field("vector.metadata.tags.element", encrypted_tag_bytes);
                tag = std::string(tag_bytes.begin(), tag_bytes.end());
            }
        }
    }
    
    return decrypted_metadata;
}

} // namespace encryption
} // namespace jadevectordb