#include "encryption.h"
#include <stdexcept>
#include <random>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cstring>
#include <memory>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/err.h>

namespace jadevectordb {
namespace encryption {

// AES-256-GCM implementation using OpenSSL

class AES256GCMEncryption : public IEncryptionAlgorithm {
private:
    static constexpr int GCM_IV_LENGTH = 12;  // 96 bits recommended for GCM
    static constexpr int GCM_TAG_LENGTH = 16; // 128 bits

public:
    std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext,
                                const EncryptionKey& key,
                                const std::vector<uint8_t>& associated_data = {}) override {
        // Validate key
        if (key.key_data.size() != 32) {
            throw std::runtime_error("Invalid key size for AES-256-GCM");
        }

        // Generate random IV (nonce)
        std::vector<uint8_t> iv(GCM_IV_LENGTH);
        if (RAND_bytes(iv.data(), GCM_IV_LENGTH) != 1) {
            throw std::runtime_error("Failed to generate random IV");
        }

        // Create and initialize the context
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx) {
            throw std::runtime_error("Failed to create cipher context");
        }

        // Initialize encryption operation
        if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to initialize AES-256-GCM");
        }

        // Set IV length
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, GCM_IV_LENGTH, nullptr) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to set IV length");
        }

        // Initialize key and IV
        if (EVP_EncryptInit_ex(ctx, nullptr, nullptr, key.key_data.data(), iv.data()) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to set key and IV");
        }

        // Provide any AAD data (authenticated but not encrypted)
        if (!associated_data.empty()) {
            int len;
            if (EVP_EncryptUpdate(ctx, nullptr, &len, associated_data.data(), associated_data.size()) != 1) {
                EVP_CIPHER_CTX_free(ctx);
                throw std::runtime_error("Failed to set associated data");
            }
        }

        // Encrypt plaintext
        std::vector<uint8_t> ciphertext(plaintext.size());
        int len = 0;
        if (EVP_EncryptUpdate(ctx, ciphertext.data(), &len, plaintext.data(), plaintext.size()) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Encryption failed");
        }
        int ciphertext_len = len;

        // Finalize encryption
        if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + len, &len) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Encryption finalization failed");
        }
        ciphertext_len += len;
        ciphertext.resize(ciphertext_len);

        // Get the authentication tag
        std::vector<uint8_t> tag(GCM_TAG_LENGTH);
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, GCM_TAG_LENGTH, tag.data()) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to get authentication tag");
        }

        EVP_CIPHER_CTX_free(ctx);

        // Build result: IV || ciphertext || tag
        std::vector<uint8_t> result;
        result.reserve(GCM_IV_LENGTH + ciphertext.size() + GCM_TAG_LENGTH);
        result.insert(result.end(), iv.begin(), iv.end());
        result.insert(result.end(), ciphertext.begin(), ciphertext.end());
        result.insert(result.end(), tag.begin(), tag.end());

        return result;
    }

    std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext,
                                const EncryptionKey& key,
                                const std::vector<uint8_t>& associated_data = {}) override {
        // Validate key
        if (key.key_data.size() != 32) {
            throw std::runtime_error("Invalid key size for AES-256-GCM");
        }

        // Validate ciphertext size (must have at least IV + tag)
        if (ciphertext.size() < GCM_IV_LENGTH + GCM_TAG_LENGTH) {
            throw std::runtime_error("Ciphertext too small");
        }

        // Extract IV, ciphertext, and tag
        std::vector<uint8_t> iv(ciphertext.begin(), ciphertext.begin() + GCM_IV_LENGTH);
        size_t ct_len = ciphertext.size() - GCM_IV_LENGTH - GCM_TAG_LENGTH;
        std::vector<uint8_t> ct(ciphertext.begin() + GCM_IV_LENGTH,
                               ciphertext.begin() + GCM_IV_LENGTH + ct_len);
        std::vector<uint8_t> tag(ciphertext.end() - GCM_TAG_LENGTH, ciphertext.end());

        // Create and initialize the context
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx) {
            throw std::runtime_error("Failed to create cipher context");
        }

        // Initialize decryption operation
        if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to initialize AES-256-GCM");
        }

        // Set IV length
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, GCM_IV_LENGTH, nullptr) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to set IV length");
        }

        // Initialize key and IV
        if (EVP_DecryptInit_ex(ctx, nullptr, nullptr, key.key_data.data(), iv.data()) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to set key and IV");
        }

        // Provide any AAD data
        if (!associated_data.empty()) {
            int len;
            if (EVP_DecryptUpdate(ctx, nullptr, &len, associated_data.data(), associated_data.size()) != 1) {
                EVP_CIPHER_CTX_free(ctx);
                throw std::runtime_error("Failed to set associated data");
            }
        }

        // Decrypt ciphertext
        std::vector<uint8_t> plaintext(ct.size());
        int len = 0;
        if (EVP_DecryptUpdate(ctx, plaintext.data(), &len, ct.data(), ct.size()) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Decryption failed");
        }
        int plaintext_len = len;

        // Set expected tag value
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, GCM_TAG_LENGTH, tag.data()) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to set authentication tag");
        }

        // Finalize decryption and verify tag
        int ret = EVP_DecryptFinal_ex(ctx, plaintext.data() + len, &len);
        EVP_CIPHER_CTX_free(ctx);

        if (ret <= 0) {
            throw std::runtime_error("Decryption failed: authentication tag verification failed");
        }

        plaintext_len += len;
        plaintext.resize(plaintext_len);

        return plaintext;
    }
    
    EncryptionKey generate_key(const EncryptionConfig& config) override {
        EncryptionKey key;
        key.key_id = "key_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        key.type = KeyType::SYMMETRIC;
        key.algorithm = EncryptionAlgorithm::AES_256_GCM;
        key.created_at = std::chrono::system_clock::now();
        key.expires_at = key.created_at + std::chrono::hours(8760); // 1 year

        // Generate cryptographically secure random key material using OpenSSL
        // AES-256 requires 256 bits = 32 bytes
        key.key_data.resize(32);
        if (RAND_bytes(key.key_data.data(), 32) != 1) {
            throw std::runtime_error("Failed to generate secure random key");
        }

        key.is_active = true;
        return key;
    }
    
    bool validate_key(const EncryptionKey& key) const override {
        return key.key_data.size() == 32 &&  // 256 bits
               key.algorithm == EncryptionAlgorithm::AES_256_GCM &&
               key.type == KeyType::SYMMETRIC;
    }
    
    EncryptionAlgorithm get_algorithm() const override {
        return EncryptionAlgorithm::AES_256_GCM;
    }
    
    std::string get_name() const override {
        return "AES-256-GCM";
    }
    
    int get_recommended_key_size() const override {
        return 256;
    }
};

class ChaCha20Poly1305Encryption : public IEncryptionAlgorithm {
public:
    std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext, 
                                const EncryptionKey& key,
                                const std::vector<uint8_t>& associated_data = {}) override {
        // In a real implementation, this would perform actual ChaCha20-Poly1305 encryption
        // For now, return the plaintext as placeholder
        return plaintext;  // Placeholder
    }
    
    std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext,
                                const EncryptionKey& key,
                                const std::vector<uint8_t>& associated_data = {}) override {
        // In a real implementation, this would perform actual ChaCha20-Poly1305 decryption
        return ciphertext;  // Placeholder
    }
    
    EncryptionKey generate_key(const EncryptionConfig& config) override {
        EncryptionKey key;
        key.key_id = "chacha_key_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        key.type = KeyType::SYMMETRIC;
        key.algorithm = EncryptionAlgorithm::CHACHA20_POLY1305;
        key.created_at = std::chrono::system_clock::now();
        key.expires_at = key.created_at + std::chrono::hours(8760); // 1 year
        
        // Generate random key material (256 bits = 32 bytes)
        key.key_data.resize(32);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        for (auto& byte : key.key_data) {
            byte = static_cast<uint8_t>(dis(gen));
        }
        
        key.is_active = true;
        return key;
    }
    
    bool validate_key(const EncryptionKey& key) const override {
        return key.key_data.size() == 32 &&  // 256 bits
               key.algorithm == EncryptionAlgorithm::CHACHA20_POLY1305 &&
               key.type == KeyType::SYMMETRIC;
    }
    
    EncryptionAlgorithm get_algorithm() const override {
        return EncryptionAlgorithm::CHACHA20_POLY1305;
    }
    
    std::string get_name() const override {
        return "ChaCha20-Poly1305";
    }
    
    int get_recommended_key_size() const override {
        return 256;
    }
};

class SimpleHomomorphicEncryption : public IEncryptionAlgorithm {
private:
    // This would store parameters for homomorphic operations in a real implementation
    // For this simplified version, we'll use a basic approach

public:
    std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext, 
                                const EncryptionKey& key,
                                const std::vector<uint8_t>& associated_data = {}) override {
        // For demonstration purposes, we'll implement a simplified somewhat-homomorphic encryption
        // This is NOT cryptographically secure but demonstrates the concept
        
        std::vector<uint8_t> result;
        result.reserve(plaintext.size() * 2); // Reserve space for "encrypted" data
        
        // Simple approach: double each byte value and add a key-derived offset
        // This allows for some basic homomorphic addition
        for (size_t i = 0; i < plaintext.size(); ++i) {
            uint8_t key_byte = key.key_data[i % key.key_data.size()];
            uint8_t enc_byte = (plaintext[i] * 2 + key_byte) % 256;
            result.push_back(enc_byte);
        }
        
        return result;
    }
    
    std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext,
                                const EncryptionKey& key,
                                const std::vector<uint8_t>& associated_data = {}) override {
        // Reverse the homomorphic transformation
        std::vector<uint8_t> result;
        result.reserve(ciphertext.size());
        
        for (size_t i = 0; i < ciphertext.size(); ++i) {
            uint8_t key_byte = key.key_data[i % key.key_data.size()];
            int temp = ciphertext[i] - key_byte;
            if (temp < 0) temp += 256; // Handle underflow
            uint8_t orig_byte = temp / 2; // Reverse the doubling
            result.push_back(orig_byte);
        }
        
        return result;
    }
    
    EncryptionKey generate_key(const EncryptionConfig& config) override {
        EncryptionKey key;
        key.key_id = "homo_key_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        key.type = KeyType::SYMMETRIC;
        key.algorithm = EncryptionAlgorithm::HOMOMORPHIC_SIMPLE;
        key.created_at = std::chrono::system_clock::now();
        key.expires_at = key.created_at + std::chrono::hours(8760); // 1 year
        
        // Generate random key material
        key.key_data.resize(32);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        for (auto& byte : key.key_data) {
            byte = static_cast<uint8_t>(dis(gen));
        }
        
        key.is_active = true;
        return key;
    }
    
    bool validate_key(const EncryptionKey& key) const override {
        return !key.key_data.empty() &&
               key.algorithm == EncryptionAlgorithm::HOMOMORPHIC_SIMPLE &&
               key.type == KeyType::SYMMETRIC;
    }
    
    EncryptionAlgorithm get_algorithm() const override {
        return EncryptionAlgorithm::HOMOMORPHIC_SIMPLE;
    }
    
    std::string get_name() const override {
        return "Simple Homomorphic";
    }
    
    int get_recommended_key_size() const override {
        return 256;
    }
    
    // Additional method to demonstrate homomorphic addition
    std::vector<uint8_t> homomorphic_add(const std::vector<uint8_t>& enc_a, 
                                        const std::vector<uint8_t>& enc_b) const {
        if (enc_a.size() != enc_b.size()) {
            throw std::invalid_argument("Encrypted vectors must have the same size for addition");
        }
        
        std::vector<uint8_t> result;
        result.reserve(enc_a.size());
        
        for (size_t i = 0; i < enc_a.size(); ++i) {
            // Homomorphic addition in our simplified scheme
            uint8_t sum = (enc_a[i] + enc_b[i]) % 256;
            result.push_back(sum);
        }
        
        return result;
    }
    
    // Additional method to demonstrate scalar multiplication
    std::vector<uint8_t> homomorphic_scalar_multiply(const std::vector<uint8_t>& encrypted,
                                                   uint8_t scalar) const {
        std::vector<uint8_t> result;
        result.reserve(encrypted.size());
        
        for (uint8_t value : encrypted) {
            // Homomorphic scalar multiplication in our simplified scheme
            uint8_t product = (value * scalar) % 256;
            result.push_back(product);
        }
        
        return result;
    }
};

// Key Management Service Implementation
class KeyManagementServiceImpl : public IKeyManagementService {
private:
    std::map<std::string, EncryptionKey> keys_;
    mutable std::mutex keys_mutex_;
    
public:
    std::string create_key(const EncryptionConfig& config) override {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        
        // Create a specific algorithm instance to generate the key
        std::unique_ptr<IEncryptionAlgorithm> algorithm;
        switch (config.algorithm) {
            case EncryptionAlgorithm::AES_256_GCM:
                algorithm = std::make_unique<AES256GCMEncryption>();
                break;
            case EncryptionAlgorithm::CHACHA20_POLY1305:
                algorithm = std::make_unique<ChaCha20Poly1305Encryption>();
                break;
            case EncryptionAlgorithm::HOMOMORPHIC_SIMPLE:
                algorithm = std::make_unique<SimpleHomomorphicEncryption>();
                break;
            default:
                algorithm = std::make_unique<AES256GCMEncryption>();
                break;
        }
        
        EncryptionKey key = algorithm->generate_key(config);
        keys_[key.key_id] = key;
        
        return key.key_id;
    }
    
    std::string rotate_key(const std::string& key_id, 
                          const EncryptionConfig& new_config) override {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        
        auto it = keys_.find(key_id);
        if (it == keys_.end()) {
            throw std::runtime_error("Key not found: " + key_id);
        }
        
        // Deactivate the old key
        it->second.is_active = false;
        
        // Create new key with the same configuration
        std::string new_key_id = create_key(new_config);
        
        return new_key_id;
    }
    
    EncryptionKey get_key(const std::string& key_id) const override {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        
        auto it = keys_.find(key_id);
        if (it == keys_.end()) {
            throw std::runtime_error("Key not found: " + key_id);
        }
        
        return it->second;
    }
    
    void deactivate_key(const std::string& key_id) override {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        
        auto it = keys_.find(key_id);
        if (it != keys_.end()) {
            it->second.is_active = false;
        }
    }
    
    void delete_key(const std::string& key_id) override {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        keys_.erase(key_id);
    }
    
    bool is_key_active(const std::string& key_id) const override {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        
        auto it = keys_.find(key_id);
        if (it == keys_.end()) {
            return false;
        }
        
        return it->second.is_active;
    }
    
    std::vector<std::string> list_keys(EncryptionAlgorithm algorithm, 
                                      bool active_only) const override {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        
        std::vector<std::string> result;
        for (const auto& pair : keys_) {
            if ((!active_only || pair.second.is_active) &&
                (algorithm == EncryptionAlgorithm::CUSTOM || 
                 pair.second.algorithm == algorithm)) {
                result.push_back(pair.first);
            }
        }
        
        return result;
    }
    
    void schedule_key_rotation(const std::string& key_id, 
                              const std::chrono::hours& interval) override {
        // In a real implementation, this would schedule key rotation
        // For now, we'll just log that this was called
        // A proper implementation would use a background thread/service
    }
};

// EncryptionManager implementation
EncryptionManager::EncryptionManager() {
    // Initialize with default algorithms
    algorithms_[EncryptionAlgorithm::AES_256_GCM] = std::make_unique<AES256GCMEncryption>();
    algorithms_[EncryptionAlgorithm::CHACHA20_POLY1305] = std::make_unique<ChaCha20Poly1305Encryption>();
    algorithms_[EncryptionAlgorithm::HOMOMORPHIC_SIMPLE] = std::make_unique<SimpleHomomorphicEncryption>();
}

void EncryptionManager::initialize(std::unique_ptr<IKeyManagementService> kms) {
    key_management_service_ = std::move(kms);
}

std::vector<uint8_t> EncryptionManager::encrypt_data(const std::vector<uint8_t>& data,
                                                    const EncryptionConfig& config) {
    if (!key_management_service_) {
        throw std::runtime_error("Key management service not initialized");
    }
    
    // Get the algorithm implementation
    auto algorithm_it = algorithms_.find(config.algorithm);
    if (algorithm_it == algorithms_.end()) {
        throw std::runtime_error("Algorithm not supported: " + 
                                std::to_string(static_cast<int>(config.algorithm)));
    }
    
    // Get the key
    EncryptionKey key = key_management_service_->get_key(config.key_id);
    
    // Perform encryption
    return algorithm_it->second->encrypt(data, key);
}

std::vector<uint8_t> EncryptionManager::decrypt_data(const std::vector<uint8_t>& data,
                                                    const EncryptionConfig& config) {
    if (!key_management_service_) {
        throw std::runtime_error("Key management service not initialized");
    }
    
    // Get the algorithm implementation
    auto algorithm_it = algorithms_.find(config.algorithm);
    if (algorithm_it == algorithms_.end()) {
        throw std::runtime_error("Algorithm not supported: " + 
                                std::to_string(static_cast<int>(config.algorithm)));
    }
    
    // Get the key
    EncryptionKey key = key_management_service_->get_key(config.key_id);
    
    // Perform decryption
    return algorithm_it->second->decrypt(data, key);
}

void EncryptionManager::configure_field_encryption(const FieldEncryptionConfig& field_config) {
    field_encryption_configs_[field_config.field_name] = field_config;
}

std::vector<uint8_t> EncryptionManager::encrypt_field(const std::string& field_name,
                                                     const std::vector<uint8_t>& data) {
    auto it = field_encryption_configs_.find(field_name);
    if (it == field_encryption_configs_.end()) {
        throw std::runtime_error("Field encryption not configured for: " + field_name);
    }
    
    return encrypt_data(data, it->second.encryption_config);
}

std::vector<uint8_t> EncryptionManager::decrypt_field(const std::string& field_name,
                                                     const std::vector<uint8_t>& data) {
    auto it = field_encryption_configs_.find(field_name);
    if (it == field_encryption_configs_.end()) {
        throw std::runtime_error("Field encryption not configured for: " + field_name);
    }
    
    return decrypt_data(data, it->second.encryption_config);
}

std::string EncryptionManager::generate_key(const EncryptionConfig& config) {
    if (!key_management_service_) {
        throw std::runtime_error("Key management service not initialized");
    }
    
    return key_management_service_->create_key(config);
}

IKeyManagementService* EncryptionManager::get_key_management_service() const {
    return key_management_service_.get();
}

bool EncryptionManager::is_searchable_encryption_enabled(const std::string& field_name) const {
    auto it = field_encryption_configs_.find(field_name);
    if (it == field_encryption_configs_.end()) {
        return false;
    }
    
    return it->second.searchable;
}

} // namespace encryption
} // namespace jadevectordb