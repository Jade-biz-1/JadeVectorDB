#ifndef JADEVECTORDB_CERTIFICATE_MANAGER_H
#define JADEVECTORDB_CERTIFICATE_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>

namespace jadevectordb {
namespace encryption {

    struct CertificateInfo {
        std::string certificate_id;
        std::string common_name;
        std::string issuer;
        std::chrono::system_clock::time_point not_before;
        std::chrono::system_clock::time_point not_after;
        std::string public_key;
        std::string certificate_data;  // PEM format
        std::vector<std::string> san_list;  // Subject Alternative Names
        bool is_active;
        bool is_self_signed;
    };

    /**
     * @brief Certificate management service interface
     * 
     * This service handles certificate generation, validation, and rotation
     */
    class ICertificateManager {
    public:
        virtual ~ICertificateManager() = default;
        
        /**
         * @brief Generate a new certificate
         * @param subject_name Subject name for the certificate
         * @param validity_days Number of days the certificate should be valid
         * @param is_ca Whether this is a Certificate Authority certificate
         * @return Certificate information
         */
        virtual CertificateInfo generate_certificate(const std::string& subject_name,
                                                   int validity_days = 365,
                                                   bool is_ca = false) = 0;
        
        /**
         * @brief Load an existing certificate from PEM data
         * @param pem_data Certificate data in PEM format
         * @return Certificate information
         */
        virtual CertificateInfo load_certificate(const std::string& pem_data) = 0;
        
        /**
         * @brief Validate a certificate
         * @param cert_id ID of the certificate to validate
         * @return True if valid, false otherwise
         */
        virtual bool validate_certificate(const std::string& cert_id) const = 0;
        
        /**
         * @brief Check if a certificate is expired
         * @param cert_id ID of the certificate to check
         * @return True if expired, false otherwise
         */
        virtual bool is_certificate_expired(const std::string& cert_id) const = 0;
        
        /**
         * @brief Get certificate information
         * @param cert_id ID of the certificate
         * @return Certificate information
         */
        virtual CertificateInfo get_certificate_info(const std::string& cert_id) const = 0;
        
        /**
         * @brief Renew a certificate before expiration
         * @param cert_id ID of the certificate to renew
         * @param new_validity_days New validity period in days
         * @param callback Optional callback to notify about renewal status
         * @return New certificate information
         */
        virtual CertificateInfo renew_certificate(const std::string& cert_id,
                                                int new_validity_days = 365,
                                                std::function<void(bool, const std::string&)> callback = nullptr) = 0;
        
        /**
         * @brief Schedule automatic certificate renewal
         * @param cert_id ID of the certificate to schedule renewal for
         * @param renewal_threshold_days Days before expiration to initiate renewal
         */
        virtual void schedule_certificate_renewal(const std::string& cert_id,
                                                int renewal_threshold_days = 30) = 0;
        
        /**
         * @brief Revoke a certificate
         * @param cert_id ID of the certificate to revoke
         * @param reason Reason for revocation
         */
        virtual void revoke_certificate(const std::string& cert_id,
                                      const std::string& reason = "unspecified") = 0;
        
        /**
         * @brief Create a certificate chain
         * @param cert_ids List of certificate IDs in the chain
         * @return Combined certificate chain
         */
        virtual std::string create_certificate_chain(const std::vector<std::string>& cert_ids) const = 0;
        
        /**
         * @brief Verify a certificate chain
         * @param cert_chain_pem PEM-encoded certificate chain
         * @param trusted_ca_pem PEM-encoded trusted CA certificate (optional)
         * @return True if chain is valid, false otherwise
         */
        virtual bool verify_certificate_chain(const std::string& cert_chain_pem,
                                             const std::string& trusted_ca_pem = "") const = 0;
        
        /**
         * @brief Check certificate revocation status
         * @param cert_id ID of the certificate to check
         * @return True if revoked, false otherwise
         */
        virtual bool is_certificate_revoked(const std::string& cert_id) const = 0;
    };

    /**
     * @brief Implementation of certificate management service
     * 
     * This service handles certificate generation, validation, and rotation
     */
    class CertificateManagerImpl : public ICertificateManager {
    private:
        std::map<std::string, CertificateInfo> certificates_;
        std::map<std::string, std::chrono::system_clock::time_point> renewal_schedule_;
        std::mutex certificates_mutex_;
        std::atomic<bool> running_{true};
        std::thread monitoring_thread_;
        
    public:
        CertificateManagerImpl();
        
        CertificateInfo generate_certificate(const std::string& subject_name,
                                           int validity_days = 365,
                                           bool is_ca = false) override;
        
        CertificateInfo load_certificate(const std::string& pem_data) override;
        
        bool validate_certificate(const std::string& cert_id) const override;
        
        bool is_certificate_expired(const std::string& cert_id) const override;
        
        CertificateInfo get_certificate_info(const std::string& cert_id) const override;
        
        CertificateInfo renew_certificate(const std::string& cert_id,
                                        int new_validity_days = 365,
                                        std::function<void(bool, const std::string&)> callback = nullptr) override;
        
        void schedule_certificate_renewal(const std::string& cert_id,
                                        int renewal_threshold_days = 30) override;
        
        void revoke_certificate(const std::string& cert_id,
                              const std::string& reason = "unspecified") override;
        
        std::string create_certificate_chain(const std::vector<std::string>& cert_ids) const override;
        
        bool verify_certificate_chain(const std::string& cert_chain_pem,
                                     const std::string& trusted_ca_pem = "") const override;
        
        bool is_certificate_revoked(const std::string& cert_id) const override;
        
    private:
        // Certificate Revocation List (CRL)
        mutable std::set<std::string> revoked_certificates_;
        mutable std::mutex revocation_mutex_;
        
        // Helper method to generate a unique certificate ID
        std::string generate_cert_id();
        
        // Helper method to perform certificate validation checks
        bool perform_validation_checks(const CertificateInfo& cert) const;
        
        // Background thread for monitoring renewal schedules
        void monitoring_loop();
    };

    /**
     * @brief Certificate rotation automation service
     * 
     * This service automatically manages certificate rotation based on configuration
     */
    class CertificateRotationService {
    private:
        std::shared_ptr<ICertificateManager> cert_manager_;
        std::chrono::minutes check_interval_;
        std::thread monitoring_thread_;
        std::atomic<bool> running_;
        std::mutex config_mutex_;
        std::map<std::string, int> rotation_config_;  // cert_id -> renewal threshold in days
        
    public:
        explicit CertificateRotationService(std::shared_ptr<ICertificateManager> cert_manager);
        
        ~CertificateRotationService();
        
        /**
         * @brief Start the automated certificate rotation service
         */
        void start();
        
        /**
         * @brief Stop the automated certificate rotation service
         */
        void stop();
        
        /**
         * @brief Configure rotation for a certificate
         * @param cert_id ID of the certificate to configure rotation for
         * @param threshold_days Days before expiration to rotate
         * @param callback Optional callback when rotation occurs
         */
        void configure_rotation(const std::string& cert_id,
                              int threshold_days,
                              std::function<void(bool, const CertificateInfo&)> callback = nullptr);
        
        /**
         * @brief Check for certificates that need renewal and rotate them
         */
        void check_and_rotate_certificates();
        
        /**
         * @brief Get the status of the rotation service
         */
        bool is_running() const;
    };

} // namespace encryption
} // namespace jadevectordb

#endif // JADEVECTORDB_CERTIFICATE_MANAGER_H