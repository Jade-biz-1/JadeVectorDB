#include "certificate_manager.h"
#include <random>
#include <thread>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>

namespace jadevectordb {
namespace encryption {

CertificateManagerImpl::CertificateManagerImpl() {
    // Start the monitoring thread for scheduled renewals
    std::thread(&CertificateManagerImpl::monitoring_loop, this).detach();
}

CertificateInfo CertificateManagerImpl::generate_certificate(const std::string& subject_name,
                                                           int validity_days,
                                                           bool is_ca) {
    std::lock_guard<std::mutex> lock(certificates_mutex_);
    
    CertificateInfo cert;
    cert.certificate_id = generate_cert_id();
    cert.common_name = subject_name;
    cert.issuer = "JadeVectorDB CA";  // For self-signed or internal CA
    cert.is_active = true;
    cert.is_self_signed = true;
    cert.not_before = std::chrono::system_clock::now();
    cert.not_after = cert.not_before + std::chrono::hours(24 * validity_days);
    
    // Generate mock certificate data with realistic format
    std::stringstream cert_stream;
    cert_stream << "-----BEGIN CERTIFICATE-----\n";
    cert_stream << "MIIDXTCCAkWgAwIBAgIJAJC1HiIAZAiIMA0GCSqGSIb3DQEBCwUAMEUxCzAJBgNV\n";
    cert_stream << "BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX\n";
    cert_stream << "aWRnaXRzIFB0eSBMdGQwHhcNMjUwMTEwMTIwMDAwWhcNMjYwMTEwMTIwMDAwWjBF\n";
    cert_stream << "MQswCQYDVQQGEwJBVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50\n";
    cert_stream << "ZXJuZXQgV2lkZ2l0cyBQdHkgTHRkMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB\n";
    cert_stream << "CgKCAQEAuCqL4IijUG9M4xX5xZf0y6nFtQwJ8kzNvQIDAQABo1AwTjAdBgNVHQ4E\n";
    cert_stream << "FgQUFv3J9NmJtzNRJravwuZLNJq0XBMwHwYDVR0jBBgwFoAUFv3J9NmJtzNRJrav\n";
    cert_stream << "wuZLNJq0XBowDAYDVR0TBAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAo4y4r6yJ\n";
    cert_stream << "sJ6y1mPz2nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6hL3qA5vG1cR9eP6uF4tX3wZ\n";
    cert_stream << "8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6hL3qA5vG1cR9eP6uF4tX3wZ\n";
    cert_stream << "8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6hL3qA5vG1cR9eP6uF4tX3wZ\n";
    cert_stream << "8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6hL3qA5vG1cR9eP6uF4tX3wZ\n";
    cert_stream << "8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6hL3qA5vG1cR9eP6uF4tX3wZ\n";
    cert_stream << "8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6hL3qA5vG1cR9eP6uF4tX3wZ\n";
    cert_stream << "8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6hL3qA5vG1cR9eP6uF4tX3wZ\n";
    cert_stream << "-----END CERTIFICATE-----";

    cert.certificate_data = cert_stream.str();
    cert.public_key = "-----BEGIN PUBLIC KEY-----\n" + cert.certificate_id + "_PUBLIC_KEY\n-----END PUBLIC KEY-----";
    
    certificates_[cert.certificate_id] = cert;
    
    return cert;
}

CertificateInfo CertificateManagerImpl::load_certificate(const std::string& pem_data) {
    std::lock_guard<std::mutex> lock(certificates_mutex_);
    
    CertificateInfo cert;
    cert.certificate_id = generate_cert_id();
    cert.certificate_data = pem_data;
    
    // Parse the PEM certificate using OpenSSL
    BIO* bio = BIO_new_mem_buf(pem_data.data(), static_cast<int>(pem_data.length()));
    if (!bio) {
        throw std::runtime_error("Failed to create BIO for certificate");
    }
    
    X509* x509_cert = PEM_read_bio_X509(bio, nullptr, nullptr, nullptr);
    BIO_free(bio);
    
    if (!x509_cert) {
        throw std::runtime_error("Failed to parse PEM certificate: " + 
                                std::string(ERR_error_string(ERR_get_error(), nullptr)));
    }
    
    // Extract common name from subject
    X509_NAME* subject_name = X509_get_subject_name(x509_cert);
    if (subject_name) {
        char cn_buf[256];
        int cn_len = X509_NAME_get_text_by_NID(subject_name, NID_commonName, cn_buf, sizeof(cn_buf));
        if (cn_len > 0) {
            cert.common_name = std::string(cn_buf, cn_len);
        } else {
            cert.common_name = "unknown";
        }
    } else {
        cert.common_name = "unknown";
    }
    
    // Extract issuer name
    X509_NAME* issuer_name = X509_get_issuer_name(x509_cert);
    if (issuer_name) {
        char issuer_buf[256];
        int issuer_len = X509_NAME_get_text_by_NID(issuer_name, NID_commonName, issuer_buf, sizeof(issuer_buf));
        if (issuer_len > 0) {
            cert.issuer = std::string(issuer_buf, issuer_len);
        } else {
            cert.issuer = "unknown";
        }
    } else {
        cert.issuer = "unknown";
    }
    
    // Extract validity period
    ASN1_TIME* not_before = X509_get_notBefore(x509_cert);
    ASN1_TIME* not_after = X509_get_notAfter(x509_cert);
    
    if (not_before) {
        struct tm tm_not_before;
        ASN1_TIME_to_tm(not_before, &tm_not_before);
        cert.not_before = std::chrono::system_clock::from_time_t(mktime(&tm_not_before));
    }
    
    if (not_after) {
        struct tm tm_not_after;
        ASN1_TIME_to_tm(not_after, &tm_not_after);
        cert.not_after = std::chrono::system_clock::from_time_t(mktime(&tm_not_after));
    }
    
    // Extract Subject Alternative Names (SANs)
    STACK_OF(GENERAL_NAME)* san_names = static_cast<STACK_OF(GENERAL_NAME)*>(
        X509_get_ext_d2i(x509_cert, NID_subject_alt_name, nullptr, nullptr));
    
    if (san_names) {
        int san_count = sk_GENERAL_NAME_num(san_names);
        for (int i = 0; i < san_count; i++) {
            GENERAL_NAME* gen_name = sk_GENERAL_NAME_value(san_names, i);
            if (gen_name->type == GEN_DNS) {
                ASN1_STRING* dns_name = gen_name->d.dNSName;
                cert.san_list.push_back(std::string(
                    reinterpret_cast<const char*>(ASN1_STRING_get0_data(dns_name)),
                    ASN1_STRING_length(dns_name)));
            }
        }
        GENERAL_NAMES_free(san_names);
    }
    
    // Check if self-signed
    cert.is_self_signed = (X509_check_issued(x509_cert, x509_cert) == X509_V_OK);
    
    // Extract public key
    EVP_PKEY* pkey = X509_get_pubkey(x509_cert);
    if (pkey) {
        BIO* pubkey_bio = BIO_new(BIO_s_mem());
        if (PEM_write_bio_PUBKEY(pubkey_bio, pkey)) {
            char* pubkey_data = nullptr;
            long pubkey_len = BIO_get_mem_data(pubkey_bio, &pubkey_data);
            if (pubkey_data && pubkey_len > 0) {
                cert.public_key = std::string(pubkey_data, pubkey_len);
            }
        }
        BIO_free(pubkey_bio);
        EVP_PKEY_free(pkey);
    }
    
    cert.is_active = true;
    
    X509_free(x509_cert);
    
    certificates_[cert.certificate_id] = cert;
    
    return cert;
}

bool CertificateManagerImpl::validate_certificate(const std::string& cert_id) const {
    std::lock_guard<std::mutex> lock(certificates_mutex_);
    
    auto it = certificates_.find(cert_id);
    if (it == certificates_.end()) {
        return false;
    }
    
    return perform_validation_checks(it->second);
}

bool CertificateManagerImpl::is_certificate_expired(const std::string& cert_id) const {
    std::lock_guard<std::mutex> lock(certificates_mutex_);
    
    auto it = certificates_.find(cert_id);
    if (it == certificates_.end()) {
        return true;  // Certificate not found is considered expired
    }
    
    auto now = std::chrono::system_clock::now();
    return now > it->second.not_after;
}

CertificateInfo CertificateManagerImpl::get_certificate_info(const std::string& cert_id) const {
    std::lock_guard<std::mutex> lock(certificates_mutex_);
    
    auto it = certificates_.find(cert_id);
    if (it == certificates_.end()) {
        throw std::runtime_error("Certificate not found: " + cert_id);
    }
    
    return it->second;
}

CertificateInfo CertificateManagerImpl::renew_certificate(const std::string& cert_id,
                                                        int new_validity_days,
                                                        std::function<void(bool, const std::string&)> callback) {
    std::lock_guard<std::mutex> lock(certificates_mutex_);
    
    auto it = certificates_.find(cert_id);
    if (it == certificates_.end()) {
        if (callback) {
            callback(false, "Certificate not found: " + cert_id);
        }
        throw std::runtime_error("Certificate not found: " + cert_id);
    }
    
    // Create a new certificate with the same properties but extended validity
    CertificateInfo old_cert = it->second;
    CertificateInfo new_cert;
    new_cert.certificate_id = generate_cert_id();
    new_cert.common_name = old_cert.common_name;
    new_cert.issuer = old_cert.issuer;
    new_cert.is_active = true;
    new_cert.is_self_signed = old_cert.is_self_signed;
    new_cert.not_before = std::chrono::system_clock::now();
    new_cert.not_after = new_cert.not_before + std::chrono::hours(24 * new_validity_days);
    
    // Generate realistic certificate data based on the old certificate
    // In production, this would use a proper certificate library like OpenSSL
    std::stringstream cert_stream;
    cert_stream << "-----BEGIN CERTIFICATE-----\n";

    // Generate deterministic base64-like data based on cert properties
    std::string cert_info = new_cert.common_name + new_cert.issuer + new_cert.certificate_id;
    std::hash<std::string> hasher;
    size_t hash_val = hasher(cert_info);

    // Create pseudo-certificate data in base64 format (simplified)
    for (int i = 0; i < 20; i++) {
        hash_val = hash_val * 1103515245 + 12345; // LCG
        char line[65];
        for (int j = 0; j < 64; j++) {
            static const char b64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            line[j] = b64_chars[(hash_val >> (j % 32)) % 64];
        }
        line[64] = '\0';
        cert_stream << line << "\n";
    }
    cert_stream << "-----END CERTIFICATE-----";

    new_cert.certificate_data = cert_stream.str();

    // Generate public key in similar fashion
    std::stringstream pubkey_stream;
    pubkey_stream << "-----BEGIN PUBLIC KEY-----\n";
    hash_val = hasher(new_cert.certificate_id + "_pubkey");
    for (int i = 0; i < 6; i++) {
        hash_val = hash_val * 1103515245 + 12345;
        char line[65];
        for (int j = 0; j < 64; j++) {
            static const char b64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            line[j] = b64_chars[(hash_val >> (j % 32)) % 64];
        }
        line[64] = '\0';
        pubkey_stream << line << "\n";
    }
    pubkey_stream << "-----END PUBLIC KEY-----";

    new_cert.public_key = pubkey_stream.str();
    
    // Mark the old certificate as inactive
    it->second.is_active = false;
    
    // Add the new certificate
    certificates_[new_cert.certificate_id] = new_cert;
    
    if (callback) {
        callback(true, "Certificate renewed successfully: " + old_cert.certificate_id + 
                " -> " + new_cert.certificate_id);
    }
    
    return new_cert;
}

void CertificateManagerImpl::schedule_certificate_renewal(const std::string& cert_id,
                                                        int renewal_threshold_days) {
    std::lock_guard<std::mutex> lock(certificates_mutex_);
    
    auto it = certificates_.find(cert_id);
    if (it == certificates_.end()) {
        throw std::runtime_error("Certificate not found: " + cert_id);
    }
    
    // Calculate renewal time (threshold days before expiration)
    auto renewal_time = it->second.not_after - std::chrono::hours(24 * renewal_threshold_days);
    renewal_schedule_[cert_id] = renewal_time;
}

void CertificateManagerImpl::revoke_certificate(const std::string& cert_id,
                                              const std::string& reason) {
    std::lock_guard<std::mutex> lock(certificates_mutex_);
    
    auto it = certificates_.find(cert_id);
    if (it != certificates_.end()) {
        it->second.is_active = false;
        
        // Add to Certificate Revocation List (CRL)
        std::lock_guard<std::mutex> revoke_lock(revocation_mutex_);
        revoked_certificates_.insert(cert_id);
    }
}

std::string CertificateManagerImpl::create_certificate_chain(const std::vector<std::string>& cert_ids) const {
    std::lock_guard<std::mutex> lock(certificates_mutex_);
    
    std::string chain;
    for (const auto& cert_id : cert_ids) {
        auto it = certificates_.find(cert_id);
        if (it != certificates_.end()) {
            if (!chain.empty()) {
                chain += "\n";  // Separate certificates
            }
            chain += it->second.certificate_data;
        }
    }
    
    return chain;
}

bool CertificateManagerImpl::verify_certificate_chain(const std::string& cert_chain_pem,
                                                     const std::string& trusted_ca_pem) const {
    // Create a new X509 store for chain verification
    X509_STORE* store = X509_STORE_new();
    if (!store) {
        return false;
    }
    
    // Load trusted CA if provided
    if (!trusted_ca_pem.empty()) {
        BIO* ca_bio = BIO_new_mem_buf(trusted_ca_pem.data(), 
                                     static_cast<int>(trusted_ca_pem.length()));
        if (ca_bio) {
            X509* ca_cert = PEM_read_bio_X509(ca_bio, nullptr, nullptr, nullptr);
            BIO_free(ca_bio);
            
            if (ca_cert) {
                X509_STORE_add_cert(store, ca_cert);
                X509_free(ca_cert);
            }
        }
    }
    
    // Parse certificate chain
    BIO* chain_bio = BIO_new_mem_buf(cert_chain_pem.data(), 
                                     static_cast<int>(cert_chain_pem.length()));
    if (!chain_bio) {
        X509_STORE_free(store);
        return false;
    }
    
    // Read all certificates from the chain
    STACK_OF(X509)* chain_stack = sk_X509_new_null();
    X509* leaf_cert = nullptr;
    X509* cert = nullptr;
    
    // First certificate is the leaf
    leaf_cert = PEM_read_bio_X509(chain_bio, nullptr, nullptr, nullptr);
    if (!leaf_cert) {
        BIO_free(chain_bio);
        X509_STORE_free(store);
        return false;
    }
    
    // Read remaining certificates in the chain
    while ((cert = PEM_read_bio_X509(chain_bio, nullptr, nullptr, nullptr)) != nullptr) {
        sk_X509_push(chain_stack, cert);
    }
    
    BIO_free(chain_bio);
    
    // Create verification context
    X509_STORE_CTX* ctx = X509_STORE_CTX_new();
    if (!ctx) {
        X509_free(leaf_cert);
        sk_X509_pop_free(chain_stack, X509_free);
        X509_STORE_free(store);
        return false;
    }
    
    // Initialize and verify
    if (!X509_STORE_CTX_init(ctx, store, leaf_cert, chain_stack)) {
        X509_STORE_CTX_free(ctx);
        X509_free(leaf_cert);
        sk_X509_pop_free(chain_stack, X509_free);
        X509_STORE_free(store);
        return false;
    }
    
    int verify_result = X509_verify_cert(ctx);
    
    // Check for revocation
    if (verify_result == 1) {
        // Additional check: verify none of the certificates in chain are revoked
        std::lock_guard<std::mutex> lock(certificates_mutex_);
        std::lock_guard<std::mutex> revoke_lock(revocation_mutex_);
        
        for (const auto& [cert_id, cert_info] : certificates_) {
            if (cert_info.certificate_data == cert_chain_pem) {
                if (revoked_certificates_.find(cert_id) != revoked_certificates_.end()) {
                    verify_result = 0;  // Certificate is revoked
                    break;
                }
            }
        }
    }
    
    X509_STORE_CTX_free(ctx);
    X509_free(leaf_cert);
    sk_X509_pop_free(chain_stack, X509_free);
    X509_STORE_free(store);
    
    return verify_result == 1;
}

bool CertificateManagerImpl::is_certificate_revoked(const std::string& cert_id) const {
    std::lock_guard<std::mutex> lock(revocation_mutex_);
    return revoked_certificates_.find(cert_id) != revoked_certificates_.end();
}

std::string CertificateManagerImpl::generate_cert_id() {
    // Generate a unique certificate ID
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Add some randomness to avoid collisions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(10000, 99999);
    
    return "cert_" + std::to_string(now) + "_" + std::to_string(dis(gen));
}

bool CertificateManagerImpl::perform_validation_checks(const CertificateInfo& cert) const {
    auto now = std::chrono::system_clock::now();
    
    // Check if certificate is within validity period
    if (now < cert.not_before || now > cert.not_after) {
        return false;
    }
    
    // Check if certificate is marked as active
    if (!cert.is_active) {
        return false;
    }
    
    // Perform OpenSSL-based validation
    BIO* bio = BIO_new_mem_buf(cert.certificate_data.data(), 
                               static_cast<int>(cert.certificate_data.length()));
    if (!bio) {
        return false;
    }
    
    X509* x509_cert = PEM_read_bio_X509(bio, nullptr, nullptr, nullptr);
    BIO_free(bio);
    
    if (!x509_cert) {
        return false;
    }
    
    // Create a verification context
    X509_STORE* store = X509_STORE_new();
    if (!store) {
        X509_free(x509_cert);
        return false;
    }
    
    X509_STORE_CTX* ctx = X509_STORE_CTX_new();
    if (!ctx) {
        X509_STORE_free(store);
        X509_free(x509_cert);
        return false;
    }
    
    // Initialize verification context
    if (!X509_STORE_CTX_init(ctx, store, x509_cert, nullptr)) {
        X509_STORE_CTX_free(ctx);
        X509_STORE_free(store);
        X509_free(x509_cert);
        return false;
    }
    
    // Perform verification
    int verify_result = X509_verify_cert(ctx);
    
    // Get detailed error if verification failed
    if (verify_result != 1) {
        int error_code = X509_STORE_CTX_get_error(ctx);
        // For self-signed certificates, allow self-signed error
        if (cert.is_self_signed && error_code == X509_V_ERR_DEPTH_ZERO_SELF_SIGNED_CERT) {
            verify_result = 1;  // Accept self-signed for internal use
        }
    }
    
    X509_STORE_CTX_free(ctx);
    X509_STORE_free(store);
    X509_free(x509_cert);
    
    return verify_result == 1;
}

void CertificateManagerImpl::monitoring_loop() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::minutes(5));  // Check every 5 minutes

        auto now = std::chrono::system_clock::now();

        std::lock_guard<std::mutex> lock(certificates_mutex_);

        // Check for expiring certificates and log warnings
        for (const auto& [cert_id, cert_info] : certificates_) {
            if (!cert_info.is_active) {
                continue;
            }
            
            auto time_to_expiry = std::chrono::duration_cast<std::chrono::hours>(
                cert_info.not_after - now).count();
            
            // Log warnings for certificates expiring soon
            if (time_to_expiry <= 24 * 7) {  // 7 days
                // In production, this would trigger an alert/notification
                // For now, we just track it
                if (time_to_expiry <= 0) {
                    // Certificate has expired
                    certificates_[cert_id].is_active = false;
                }
            }
        }

        // Check for scheduled renewals
        for (auto& [cert_id, renewal_time] : renewal_schedule_) {
            if (now >= renewal_time) {
                // Time to renew this certificate
                try {
                    // Find the certificate to renew
                    auto cert_it = certificates_.find(cert_id);
                    if (cert_it != certificates_.end() && cert_it->second.is_active) {
                        // Attempt automatic renewal
                        // In production, this would use generate_certificate or load from ACME
                        
                        // Mark old certificate as inactive
                        cert_it->second.is_active = false;

                        // Create new certificate with same properties but extended validity
                        CertificateInfo new_cert;
                        new_cert.certificate_id = "cert_" + std::to_string(
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count());
                        new_cert.common_name = cert_it->second.common_name;
                        new_cert.issuer = cert_it->second.issuer;
                        new_cert.is_active = true;
                        new_cert.is_self_signed = cert_it->second.is_self_signed;
                        new_cert.not_before = std::chrono::system_clock::now();
                        new_cert.not_after = new_cert.not_before + std::chrono::hours(24 * 365);

                        // Generate new mock certificate data
                        std::stringstream cert_stream;
                        cert_stream << "-----BEGIN CERTIFICATE-----\n";
                        cert_stream << "MIIDYTCCAkmgAwIBAgIJAOhLvA3Lu+36MA0GCSqGSIb3DQEBCwUAMEUxCzAJBgNV\n";
                        cert_stream << "BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX\n";
                        cert_stream << "aWRnaXRzIFB0eSBMdGQwHhcNMjYwMTEwMTIwMDAwWhcNMjcwMTEwMTIwMDAwWjBF\n";
                        cert_stream << "MQswCQYDVQQGEwJBVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50\n";
                        cert_stream << "ZXJuZXQgV2lkZ2l0cyBQdHkgTHRkMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB\n";
                        cert_stream << "CgKCAQEAxQPBbhtMPx6r5QJzHqC2vFvFwB6bF6lL5G8ZJ6vG8mN9kP3tY2s4d6hL\n";
                        cert_stream << "3qA5vG1cR9eP6uF4tX3wZ8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6h\n";
                        cert_stream << "L3qA5vG1cR9eP6uF4tX3wZ8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6h\n";
                        cert_stream << "L3qA5vG1cR9eP6uF4tX3wZ8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6h\n";
                        cert_stream << "L3qA5vG1cR9eP6uF4tX3wZ8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6h\n";
                        cert_stream << "L3qA5vG1cR9eP6uF4tX3wZ8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6h\n";
                        cert_stream << "L3qA5vG1cR9eP6uF4tX3wZ8cV2pH7fJ6nQ9v8K7xR3mN2oV5j1pX8w7tY2s4d6h\n";
                        cert_stream << "-----END CERTIFICATE-----";

                        new_cert.certificate_data = cert_stream.str();
                        new_cert.public_key = "-----BEGIN PUBLIC KEY-----\n" + new_cert.certificate_id + "_PUBLIC_KEY\n-----END PUBLIC KEY-----";

                        // Store the new certificate
                        certificates_[new_cert.certificate_id] = new_cert;

                        // Update the renewal schedule for the new certificate
                        renewal_schedule_[new_cert.certificate_id] =
                            new_cert.not_after - std::chrono::hours(24 * 30); // Renew 30 days before expiration
                    }
                } catch (const std::exception& e) {
                    // Handle renewal failure
                    // In a real implementation, we would log this
                }
            }
        }
    }
}

// CertificateRotationService implementation
CertificateRotationService::CertificateRotationService(std::shared_ptr<ICertificateManager> cert_manager)
    : cert_manager_(cert_manager), check_interval_(std::chrono::hours(1)), running_(false) {
}

CertificateRotationService::~CertificateRotationService() {
    stop();
}

void CertificateRotationService::start() {
    if (running_) {
        return;  // Already running
    }
    
    running_ = true;
    monitoring_thread_ = std::thread([this]() {
        while (running_) {
            check_and_rotate_certificates();
            std::this_thread::sleep_for(check_interval_);
        }
    });
}

void CertificateRotationService::stop() {
    if (!running_) {
        return;  // Already stopped
    }
    
    running_ = false;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
}

void CertificateRotationService::configure_rotation(const std::string& cert_id,
                                                  int threshold_days,
                                                  std::function<void(bool, const CertificateInfo&)> callback) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    rotation_config_[cert_id] = threshold_days;
    
    // Schedule the certificate for renewal at the threshold
    cert_manager_->schedule_certificate_renewal(cert_id, threshold_days);
}

void CertificateRotationService::check_and_rotate_certificates() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    for (const auto& config : rotation_config_) {
        const std::string& cert_id = config.first;
        int threshold_days = config.second;
        
        // Check if the certificate needs renewal
        auto cert_info = cert_manager_->get_certificate_info(cert_id);
        auto now = std::chrono::system_clock::now();
        auto time_to_expiration = std::chrono::duration_cast<std::chrono::hours>(
            cert_info.not_after - now).count();
        
        if (time_to_expiration <= threshold_days * 24) {
            // Certificate needs renewal
            try {
                auto new_cert = cert_manager_->renew_certificate(cert_id);
                // In a real implementation, we might notify via the callback
            } catch (const std::exception& e) {
                // Handle renewal failure
                // In a real implementation, we would log this
            }
        }
    }
}

bool CertificateRotationService::is_running() const {
    return running_;
}

} // namespace encryption
} // namespace jadevectordb