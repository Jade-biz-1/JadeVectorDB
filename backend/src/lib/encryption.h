#ifndef JADEVECTORDB_ENCRYPTION_H
#define JADEVECTORDB_ENCRYPTION_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <chrono>

namespace jadevectordb {
namespace encryption {

    // Enum for encryption algorithms
    enum class EncryptionAlgorithm {
        AES_256_GCM,      // Advanced Encryption Standard with Galois/Counter Mode
        CHACHA20_POLY1305, // Stream cipher with AEAD
        RSA_4096,         // RSA with 4096-bit keys
        HOMOMORPHIC_SIMPLE, // Simplified homomorphic encryption for searchable operations
        CUSTOM            // Custom encryption algorithm
    };

    // Enum for key types
    enum class KeyType {
        SYMMETRIC,        // Symmetric key (same key for encryption/decryption)
        ASYMMETRIC_PUBLIC, // Public key (for encryption)
        ASYMMETRIC_PRIVATE // Private key (for decryption)
    };

    // Structure for encryption key
    struct EncryptionKey {
        std::string key_id;           // Unique identifier for the key
        KeyType type;                 // Type of key
        EncryptionAlgorithm algorithm; // Algorithm this key is intended for
        std::vector<uint8_t> key_data; // The actual key material
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point expires_at;
        std::string description;
        bool is_active;
        
        EncryptionKey() : key_id(""), type(KeyType::SYMMETRIC), 
                         algorithm(EncryptionAlgorithm::AES_256_GCM),
                         is_active(true) {}
    };

    // Configuration for encryption operations
    struct EncryptionConfig {
        EncryptionAlgorithm algorithm = EncryptionAlgorithm::AES_256_GCM;
        std::string key_id;           // ID of the key to use
        int key_size_bits = 256;      // Size of the encryption key in bits
        bool enable_hardware_acceleration = true; // Use hardware acceleration if available
        std::map<std::string, std::string> custom_params; // Additional parameters for specific algorithms
        bool searchable_encryption = false; // Enable searchable encryption (affects performance and security)
    };

    /**
     * @brief Interface for encryption algorithms
     * 
     * This interface defines the contract for all encryption algorithms
     * used in the vector database system.
     */
    class IEncryptionAlgorithm {
    public:
        virtual ~IEncryptionAlgorithm() = default;
        
        /**
         * @brief Encrypt data
         * @param plaintext Data to encrypt
         * @param key Key to use for encryption
         * @param associated_data Additional data authenticated but not encrypted (for AEAD ciphers)
         * @return Encrypted data
         */
        virtual std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext, 
                                           const EncryptionKey& key,
                                           const std::vector<uint8_t>& associated_data = {}) = 0;
        
        /**
         * @brief Decrypt data
         * @param ciphertext Data to decrypt
         * @param key Key to use for decryption
         * @param associated_data Additional data authenticated but not encrypted (for AEAD ciphers)
         * @return Decrypted data
         */
        virtual std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext,
                                           const EncryptionKey& key,
                                           const std::vector<uint8_t>& associated_data = {}) = 0;
        
        /**
         * @brief Generate a new encryption key
         * @param config Configuration for the key
         * @return Generated encryption key
         */
        virtual EncryptionKey generate_key(const EncryptionConfig& config) = 0;
        
        /**
         * @brief Validate an encryption key for this algorithm
         * @param key Key to validate
         * @return True if valid, false otherwise
         */
        virtual bool validate_key(const EncryptionKey& key) const = 0;
        
        /**
         * @brief Get the algorithm identifier
         */
        virtual EncryptionAlgorithm get_algorithm() const = 0;
        
        /**
         * @brief Get the name of the algorithm
         */
        virtual std::string get_name() const = 0;
        
        /**
         * @brief Get the recommended key size in bits
         */
        virtual int get_recommended_key_size() const = 0;
    };

    /**
     * @brief Field-level encryption configuration
     * 
     * Specifies which fields should be encrypted at the field level
     */
    struct FieldEncryptionConfig {
        std::string field_name;        // Name of the field to encrypt
        EncryptionConfig encryption_config; // Encryption algorithm to use
        bool enable_deterministic_encryption; // Enable deterministic encryption (same plaintext always encrypts to same ciphertext)
        bool searchable;               // Whether this field supports searching
        std::string associated_data;   // Additional data for authenticated encryption
    };

    /**
     * @brief Key management service interface
     * 
     * Manages encryption keys including generation, rotation, and lifecycle
     */
    class IKeyManagementService {
    public:
        virtual ~IKeyManagementService() = default;
        
        /**
         * @brief Create a new encryption key
         * @param config Configuration for the key
         * @return ID of the created key
         */
        virtual std::string create_key(const EncryptionConfig& config) = 0;
        
        /**
         * @brief Rotate an existing key
         * @param key_id ID of the key to rotate
         * @param new_config New configuration for the rotated key
         * @return ID of the new key
         */
        virtual std::string rotate_key(const std::string& key_id, 
                                     const EncryptionConfig& new_config) = 0;
        
        /**
         * @brief Retrieve an encryption key
         * @param key_id ID of the key to retrieve
         * @return Encryption key
         */
        virtual EncryptionKey get_key(const std::string& key_id) const = 0;
        
        /**
         * @brief Mark an encryption key as inactive
         * @param key_id ID of the key to deactivate
         */
        virtual void deactivate_key(const std::string& key_id) = 0;
        
        /**
         * @brief Delete an encryption key
         * @param key_id ID of the key to delete
         */
        virtual void delete_key(const std::string& key_id) = 0;
        
        /**
         * @brief Check if a key is active
         * @param key_id ID of the key to check
         * @return True if active, false otherwise
         */
        virtual bool is_key_active(const std::string& key_id) const = 0;
        
        /**
         * @brief List all keys matching criteria
         * @param algorithm Algorithm type to filter by, if any
         * @param active_only Whether to return only active keys
         * @return Vector of key IDs
         */
        virtual std::vector<std::string> list_keys(EncryptionAlgorithm algorithm = EncryptionAlgorithm::CUSTOM, 
                                                 bool active_only = true) const = 0;
        
        /**
         * @brief Schedule key rotation
         * @param key_id ID of the key to rotate
         * @param interval Rotation interval
         */
        virtual void schedule_key_rotation(const std::string& key_id, 
                                         const std::chrono::hours& interval) = 0;
    };

    /**
     * @brief Encryption manager to handle different encryption algorithms and keys
     * 
     * This class manages multiple encryption algorithms and provides
     * a unified interface for encryption/decryption operations.
     */
    class EncryptionManager {
    private:
        std::unique_ptr<IKeyManagementService> key_management_service_;
        std::map<EncryptionAlgorithm, std::unique_ptr<IEncryptionAlgorithm>> algorithms_;
        std::map<std::string, FieldEncryptionConfig> field_encryption_configs_;
        
    public:
        EncryptionManager();
        ~EncryptionManager() = default;
        
        /**
         * @brief Initialize the encryption manager with a key management service
         * @param kms Key management service to use
         */
        void initialize(std::unique_ptr<IKeyManagementService> kms);
        
        /**
         * @brief Encrypt data using the specified configuration
         * @param data Data to encrypt
         * @param config Encryption configuration
         * @return Encrypted data
         */
        std::vector<uint8_t> encrypt_data(const std::vector<uint8_t>& data,
                                        const EncryptionConfig& config);
        
        /**
         * @brief Decrypt data using the specified configuration
         * @param data Data to decrypt
         * @param config Encryption configuration
         * @return Decrypted data
         */
        std::vector<uint8_t> decrypt_data(const std::vector<uint8_t>& data,
                                        const EncryptionConfig& config);
        
        /**
         * @brief Configure field-level encryption
         * @param field_config Field encryption configuration
         */
        void configure_field_encryption(const FieldEncryptionConfig& field_config);
        
        /**
         * @brief Encrypt a specific field in a data structure
         * @param field_name Name of the field to encrypt
         * @param data Data to encrypt
         * @return Encrypted data
         */
        std::vector<uint8_t> encrypt_field(const std::string& field_name,
                                         const std::vector<uint8_t>& data);
        
        /**
         * @brief Decrypt a specific field in a data structure
         * @param field_name Name of the field to decrypt
         * @param data Data to decrypt
         * @return Decrypted data
         */
        std::vector<uint8_t> decrypt_field(const std::string& field_name,
                                         const std::vector<uint8_t>& data);
        
        /**
         * @brief Generate a new encryption key using the key management service
         * @param config Configuration for the key
         * @return ID of the generated key
         */
        std::string generate_key(const EncryptionConfig& config);
        
        /**
         * @brief Get the key management service
         */
        IKeyManagementService* get_key_management_service() const;
        
        /**
         * @brief Check if searchable encryption is enabled for a field
         * @param field_name Name of the field to check
         * @return True if searchable encryption is enabled, false otherwise
         */
        bool is_searchable_encryption_enabled(const std::string& field_name) const;
    };

    // Forward declaration of specific encryption implementations
    class AES256GCMEncryption;
    class ChaCha20Poly1305Encryption;
    class SimpleHomomorphicEncryption;  // Simplified for searchable encryption
    class KeyManagementServiceImpl;

} // namespace encryption
} // namespace jadevectordb

#endif // JADEVECTORDB_ENCRYPTION_H