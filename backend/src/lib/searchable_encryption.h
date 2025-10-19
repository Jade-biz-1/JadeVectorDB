#ifndef JADEVECTORDB_SEARCHABLE_ENCRYPTION_H
#define JADEVECTORDB_SEARCHABLE_ENCRYPTION_H

#include "encryption.h"
#include <string>
#include <vector>
#include <memory>

namespace jadevectordb {
namespace encryption {

    /**
     * @brief Interface for searchable encryption
     * 
     * This interface defines methods for performing searches on encrypted data
     */
    class ISearchableEncryption {
    public:
        virtual ~ISearchableEncryption() = default;
        
        /**
         * @brief Create a searchable token for encrypted data
         * @param plaintext The original plaintext value
         * @param key Key used for encryption
         * @return Encrypted searchable token
         */
        virtual std::vector<uint8_t> create_searchable_token(const std::string& plaintext, 
                                                           const EncryptionKey& key) = 0;
        
        /**
         * @brief Create a trapdoor for searching
         * @param search_term The term to search for
         * @param key Key used for encryption
         * @return Search trapdoor
         */
        virtual std::vector<uint8_t> create_search_trapdoor(const std::string& search_term, 
                                                          const EncryptionKey& key) = 0;
        
        /**
         * @brief Test if a searchable token matches a trapdoor
         * @param token Searchable token
         * @param trapdoor Search trapdoor
         * @return True if they match, false otherwise
         */
        virtual bool test_match(const std::vector<uint8_t>& token, 
                              const std::vector<uint8_t>& trapdoor) = 0;
    };

    /**
     * @brief Deterministic encryption for searchable fields
     * 
     * This class implements deterministic encryption where the same plaintext
     * always encrypts to the same ciphertext, enabling equality searches
     */
    class DeterministicEncryption : public ISearchableEncryption {
    private:
        std::unique_ptr<IEncryptionAlgorithm> algorithm_;
        
    public:
        explicit DeterministicEncryption(std::unique_ptr<IEncryptionAlgorithm> algorithm);
        
        std::vector<uint8_t> create_searchable_token(const std::string& plaintext, 
                                                   const EncryptionKey& key) override;
        
        std::vector<uint8_t> create_search_trapdoor(const std::string& search_term, 
                                                   const EncryptionKey& key) override;
        
        bool test_match(const std::vector<uint8_t>& token, 
                       const std::vector<uint8_t>& trapdoor) override;
    };

    /**
     * @brief Order-preserving encryption for range searches
     * 
     * This class implements order-preserving encryption for range queries
     */
    class OrderPreservingEncryption {
    private:
        // This would require a specialized cryptographic implementation
        // for maintaining order relationships in encrypted data
        // We'll implement a simplified version for demonstration
        
    public:
        /**
         * @brief Encrypt a value while preserving order
         * @param value The value to encrypt
         * @param key Key to use for encryption
         * @return Encrypted value that preserves order relations
         */
        std::vector<uint8_t> encrypt_preserving_order(double value, const EncryptionKey& key);
        
        /**
         * @brief Compare two order-preserving encrypted values
         * @param enc1 First encrypted value
         * @param enc2 Second encrypted value
         * @return -1 if enc1 < enc2, 0 if equal, 1 if enc1 > enc2
         */
        int compare_encrypted_values(const std::vector<uint8_t>& enc1, 
                                   const std::vector<uint8_t>& enc2);
    };

    /**
     * @brief Implementation of homomorphic encryption for vector operations
     * 
     * This class implements simplified homomorphic operations that allow
     * computations on encrypted vector data
     */
    class VectorHomomorphicEncryption {
    private:
        std::unique_ptr<IEncryptionAlgorithm> algorithm_;
        
    public:
        explicit VectorHomomorphicEncryption(std::unique_ptr<IEncryptionAlgorithm> algorithm);
        
        /**
         * @brief Perform an operation on encrypted vectors
         * @param op Operation to perform (e.g., addition, multiplication)
         * @param encrypted_a First encrypted vector
         * @param encrypted_b Second encrypted vector
         * @return Result of the operation on encrypted data
         */
        std::vector<uint8_t> perform_operation(const std::string& op,
                                             const std::vector<uint8_t>& encrypted_a,
                                             const std::vector<uint8_t>& encrypted_b,
                                             const EncryptionKey& key);
        
        /**
         * @brief Perform scalar multiplication on encrypted vector
         * @param encrypted_vector Encrypted vector
         * @param scalar Scalar value to multiply with
         * @param key Key to use for operations
         * @return Scalar multiplication result
         */
        std::vector<uint8_t> scalar_multiply(const std::vector<uint8_t>& encrypted_vector,
                                           double scalar,
                                           const EncryptionKey& key);
    };

} // namespace encryption
} // namespace jadevectordb

#endif // JADEVECTORDB_SEARCHABLE_ENCRYPTION_H