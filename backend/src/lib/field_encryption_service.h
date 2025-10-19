#ifndef JADEVECTORDB_FIELD_ENCRYPTION_SERVICE_H
#define JADEVECTORDB_FIELD_ENCRYPTION_SERVICE_H

#include "encryption.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace jadevectordb {
namespace encryption {

    /**
     * @brief Interface for field-level encryption
     * 
     * This service handles encryption and decryption of specific fields in data structures
     */
    class IFieldEncryptionService {
    public:
        virtual ~IFieldEncryptionService() = default;
        
        /**
         * @brief Encrypt a specific field in a data structure
         * @param field_path Path to the field to encrypt (e.g., "user.profile.email")
         * @param field_value Value of the field to encrypt
         * @param context Additional context for encryption (e.g., record ID, user ID)
         * @return Encrypted field value
         */
        virtual std::vector<uint8_t> encrypt_field(const std::string& field_path,
                                                 const std::vector<uint8_t>& field_value,
                                                 const std::map<std::string, std::string>& context = {}) = 0;
        
        /**
         * @brief Decrypt a specific field in a data structure
         * @param field_path Path to the field to decrypt
         * @param encrypted_value Encrypted value of the field
         * @param context Additional context for decryption
         * @return Decrypted field value
         */
        virtual std::vector<uint8_t> decrypt_field(const std::string& field_path,
                                                 const std::vector<uint8_t>& encrypted_value,
                                                 const std::map<std::string, std::string>& context = {}) = 0;
        
        /**
         * @brief Check if a field is configured for encryption
         * @param field_path Path to the field to check
         * @return True if the field is configured for encryption
         */
        virtual bool is_field_encrypted(const std::string& field_path) const = 0;
        
        /**
         * @brief Get the encryption configuration for a field
         * @param field_path Path to the field
         * @return Encryption configuration for the field
         */
        virtual EncryptionConfig get_field_encryption_config(const std::string& field_path) const = 0;
        
        /**
         * @brief Configure encryption for a field
         * @param field_path Path to the field to configure
         * @param config Encryption configuration for the field
         */
        virtual void configure_field(const std::string& field_path, 
                                   const EncryptionConfig& config) = 0;
        
        /**
         * @brief Remove encryption configuration for a field
         * @param field_path Path to the field to remove configuration for
         */
        virtual void remove_field_configuration(const std::string& field_path) = 0;
    };

    /**
     * @brief Implementation of field-level encryption service
     * 
     * This service manages encryption/decryption for specific fields in data structures
     */
    class FieldEncryptionServiceImpl : public IFieldEncryptionService {
    private:
        std::map<std::string, FieldEncryptionConfig> field_configs_;
        std::unique_ptr<EncryptionManager> encryption_manager_;
        
    public:
        explicit FieldEncryptionServiceImpl(std::unique_ptr<EncryptionManager> encryption_manager);
        
        std::vector<uint8_t> encrypt_field(const std::string& field_path,
                                         const std::vector<uint8_t>& field_value,
                                         const std::map<std::string, std::string>& context = {}) override;
        
        std::vector<uint8_t> decrypt_field(const std::string& field_path,
                                         const std::vector<uint8_t>& encrypted_value,
                                         const std::map<std::string, std::string>& context = {}) override;
        
        bool is_field_encrypted(const std::string& field_path) const override;
        
        EncryptionConfig get_field_encryption_config(const std::string& field_path) const override;
        
        void configure_field(const std::string& field_path, 
                           const EncryptionConfig& config) override;
        
        void remove_field_configuration(const std::string& field_path) override;
        
    private:
        // Helper method to generate field-specific context for encryption
        std::vector<uint8_t> generate_field_context(const std::string& field_path,
                                                   const std::map<std::string, std::string>& context) const;
    };

    /**
     * @brief Encryptor for vector data structures
     * 
     * This class handles encryption of vector-related data structures
     */
    class VectorDataEncryptor {
    private:
        std::shared_ptr<FieldEncryptionServiceImpl> field_encryption_service_;
        
    public:
        explicit VectorDataEncryptor(std::shared_ptr<FieldEncryptionServiceImpl> field_encryption_service);
        
        /**
         * @brief Encrypt a Vector structure
         * @param vector The vector to encrypt (some fields may be encrypted based on configuration)
         * @return Encrypted vector
         */
        jadevectordb::Vector encrypt_vector(const jadevectordb::Vector& vector);
        
        /**
         * @brief Decrypt a Vector structure
         * @param encrypted_vector The encrypted vector to decrypt
         * @return Decrypted vector
         */
        jadevectordb::Vector decrypt_vector(const jadevectordb::Vector& encrypted_vector);
        
        /**
         * @brief Encrypt specific fields in the vector metadata
         * @param metadata The metadata to encrypt fields in
         * @return Metadata with encrypted fields
         */
        jadevectordb::Vector::Metadata encrypt_metadata_fields(const jadevectordb::Vector::Metadata& metadata);
        
        /**
         * @brief Decrypt specific fields in the vector metadata
         * @param encrypted_metadata The encrypted metadata to decrypt fields in
         * @return Metadata with decrypted fields
         */
        jadevectordb::Vector::Metadata decrypt_metadata_fields(const jadevectordb::Vector::Metadata& encrypted_metadata);
    };

} // namespace encryption
} // namespace jadevectordb

#endif // JADEVECTORDB_FIELD_ENCRYPTION_SERVICE_H