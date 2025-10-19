#ifndef JADEVECTORDB_ZERO_TRUST_H
#define JADEVECTORDB_ZERO_TRUST_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <functional>

namespace jadevectordb {
namespace zero_trust {

    // Enum for trust levels
    enum class TrustLevel {
        NONE = 0,           // No trust
        LOW = 1,            // Low trust
        MEDIUM = 2,         // Medium trust
        HIGH = 3,           // High trust
        VERIFIED = 4        // Fully verified
    };

    // Enum for access types
    enum class AccessType {
        READ,
        WRITE,
        DELETE,
        ADMIN,
        CUSTOM
    };

    // Structure for device identity and attributes
    struct DeviceIdentity {
        std::string device_id;
        std::string hardware_id;
        std::string os_type;
        std::string os_version;
        std::string certificate_thumbprint;
        std::vector<std::string> trusted_certificates;
        std::string public_key;
        std::chrono::system_clock::time_point last_verification;
        TrustLevel trust_level;
        bool is_managed;
    };

    // Structure for session information
    struct SessionInfo {
        std::string session_id;
        std::string user_id;
        std::string device_id;
        std::string ip_address;
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point last_activity;
        std::chrono::system_clock::time_point expires_at;
        TrustLevel trust_level;
        std::vector<std::string> permissions;
        std::string origin;
    };

    // Structure for access request
    struct AccessRequest {
        std::string resource_id;
        AccessType access_type;
        std::string requester_id;
        std::string device_id;
        std::string ip_address;
        std::string justification;  // Reason for access request
        std::chrono::system_clock::time_point requested_at;
        std::chrono::seconds requested_duration;  // For JIT access
    };

    // Structure for access decision
    struct AccessDecision {
        bool approved;
        std::chrono::system_clock::time_point decision_time;
        std::string reason;
        TrustLevel trust_level;
        std::chrono::system_clock::time_point expires_at;
        std::vector<std::string> granted_permissions;
    };

    /**
     * @brief Interface for continuous authentication
     * 
     * This service handles continuous authentication and trust evaluation
     */
    class IContinuousAuthentication {
    public:
        virtual ~IContinuousAuthentication() = default;
        
        /**
         * @brief Authenticate a request with continuous trust evaluation
         * @param request Access request to authenticate
         * @param session_info Current session information
         * @return Trust level for the request
         */
        virtual TrustLevel authenticate_request(const AccessRequest& request, 
                                              const SessionInfo& session_info) = 0;
        
        /**
         * @brief Re-evaluate trust for an existing session
         * @param session_id ID of the session to evaluate
         * @return Updated trust level
         */
        virtual TrustLevel reevaluate_session(const std::string& session_id) = 0;
        
        /**
         * @brief Register behavioral patterns for a user
         * @param user_id ID of the user
         * @param patterns Behavioral patterns
         */
        virtual void register_behavioral_patterns(const std::string& user_id, 
                                                const std::vector<std::string>& patterns) = 0;
        
        /**
         * @brief Update trust based on risk factors
         * @param session_info Session information to evaluate
         * @param risk_factors Risk factors to consider
         * @return Adjusted trust level
         */
        virtual TrustLevel update_trust_from_risk(const SessionInfo& session_info,
                                                const std::vector<std::string>& risk_factors) = 0;
    };

    /**
     * @brief Interface for micro-segmentation
     * 
     * This service handles network and service segmentation
     */
    class IMicroSegmentation {
    public:
        virtual ~IMicroSegmentation() = default;
        
        /**
         * @brief Check if communication is allowed between two endpoints
         * @param source_endpoint Source endpoint identifier
         * @param destination_endpoint Destination endpoint identifier
         * @param protocol Network protocol
         * @param port Destination port
         * @return True if communication is allowed
         */
        virtual bool is_communication_allowed(const std::string& source_endpoint,
                                            const std::string& destination_endpoint,
                                            const std::string& protocol = "tcp",
                                            int port = 0) = 0;
        
        /**
         * @brief Create a security policy between endpoints
         * @param source_endpoint Source endpoint identifier
         * @param destination_endpoint Destination endpoint identifier
         * @param allowed_protocols List of allowed protocols
         * @param allowed_ports List of allowed ports
         */
        virtual void create_security_policy(const std::string& source_endpoint,
                                          const std::string& destination_endpoint,
                                          const std::vector<std::string>& allowed_protocols,
                                          const std::vector<int>& allowed_ports) = 0;
        
        /**
         * @brief Get applicable security policies for an endpoint
         * @param endpoint Endpoint identifier
         * @return List of applicable policies
         */
        virtual std::vector<std::string> get_applicable_policies(const std::string& endpoint) const = 0;
        
        /**
         * @brief Remove a security policy
         * @param source_endpoint Source endpoint identifier
         * @param destination_endpoint Destination endpoint identifier
         */
        virtual void remove_security_policy(const std::string& source_endpoint,
                                          const std::string& destination_endpoint) = 0;
    };

    /**
     * @brief Interface for just-in-time access provisioning
     * 
     * This service handles temporary access provisioning
     */
    class IJustInTimeAccess {
    public:
        virtual ~IJustInTimeAccess() = default;
        
        /**
         * @brief Request temporary access to a resource
         * @param request Access request with duration
         * @param approver_id Optional ID of the approving entity
         * @return Access decision with temporary credentials
         */
        virtual AccessDecision request_temporary_access(const AccessRequest& request,
                                                      const std::string& approver_id = "") = 0;
        
        /**
         * @brief Approve a JIT access request
         * @param request_id ID of the request to approve
         * @param approver_id ID of the entity approving
         * @return True if approval was successful
         */
        virtual bool approve_access_request(const std::string& request_id,
                                          const std::string& approver_id) = 0;
        
        /**
         * @brief Revoke temporary access
         * @param access_token Token identifying the temporary access
         * @return True if revocation was successful
         */
        virtual bool revoke_temporary_access(const std::string& access_token) = 0;
        
        /**
         * @brief Check if temporary access is still valid
         * @param access_token Token to check
         * @return True if access is still valid
         */
        virtual bool is_temporary_access_valid(const std::string& access_token) const = 0;
    };

    /**
     * @brief Interface for device trust attestation
     * 
     * This service handles verification of device trustworthiness
     */
    class IDeviceAttestation {
    public:
        virtual ~IDeviceAttestation() = default;
        
        /**
         * @brief Attest a device's trustworthiness
         * @param device_identity Device identity to attest
         * @return Trust level for the device
         */
        virtual TrustLevel attest_device(const DeviceIdentity& device_identity) = 0;
        
        /**
         * @brief Register a device for attestation
         * @param device_identity Device identity information
         * @param initial_trust_level Initial trust level
         * @return Device ID for the registered device
         */
        virtual std::string register_device(const DeviceIdentity& device_identity,
                                          TrustLevel initial_trust_level = TrustLevel::LOW) = 0;
        
        /**
         * @brief Verify a device's attestation certificate
         * @param certificate Certificate to verify
         * @param device_id ID of the device presenting the certificate
         * @return True if certificate is valid
         */
        virtual bool verify_attestation_certificate(const std::string& certificate,
                                                 const std::string& device_id) = 0;
        
        /**
         * @brief Update device trust based on health assessment
         * @param device_id ID of the device
         * @param health_indicators Device health indicators
         * @return Updated trust level
         */
        virtual TrustLevel update_device_trust(const std::string& device_id,
                                             const std::unordered_map<std::string, std::string>& health_indicators) = 0;
        
        /**
         * @brief Get trust assessment for a device
         * @param device_id ID of the device
         * @return Trust assessment information
         */
        virtual TrustLevel get_device_trust_level(const std::string& device_id) const = 0;
    };

    /**
     * @brief Zero Trust Orchestrator
     * 
     * This class coordinates all zero-trust components to make access decisions
     */
    class ZeroTrustOrchestrator {
    private:
        std::unique_ptr<IContinuousAuthentication> continuous_auth_;
        std::unique_ptr<IMicroSegmentation> microsegmentation_;
        std::unique_ptr<IJustInTimeAccess> jit_access_;
        std::unique_ptr<IDeviceAttestation> device_attestation_;
        
    public:
        ZeroTrustOrchestrator(
            std::unique_ptr<IContinuousAuthentication> continuous_auth,
            std::unique_ptr<IMicroSegmentation> microsegmentation,
            std::unique_ptr<IJustInTimeAccess> jit_access,
            std::unique_ptr<IDeviceAttestation> device_attestation
        );
        
        ~ZeroTrustOrchestrator() = default;
        
        /**
         * @brief Evaluate access request using all zero-trust components
         * @param request Access request to evaluate
         * @param session_info Session information
         * @param device_identity Requesting device identity
         * @return Final access decision
         */
        AccessDecision evaluate_access_request(const AccessRequest& request,
                                             const SessionInfo& session_info,
                                             const DeviceIdentity& device_identity);
        
        /**
         * @brief Perform continuous trust evaluation for a session
         * @param session_id ID of the session to evaluate
         * @return Updated trust level
         */
        TrustLevel continuous_evaluation(const std::string& session_id);
        
        /**
         * @brief Register a new device in the zero-trust system
         * @param device_identity Device identity information
         * @param initial_trust_level Initial trust level
         * @return Device ID if registration is successful
         */
        std::string register_device(const DeviceIdentity& device_identity,
                                  TrustLevel initial_trust_level = TrustLevel::LOW);
        
        /**
         * @brief Request JIT access to a resource
         * @param request Access request with duration
         * @param user_id ID of the requesting user
         * @return Access decision
         */
        AccessDecision request_jit_access(const AccessRequest& request, 
                                        const std::string& user_id);
        
        /**
         * @brief Check network communication permissions
         * @param source_endpoint Source endpoint
         * @param destination_endpoint Destination endpoint
         * @param protocol Network protocol
         * @param port Destination port
         * @return True if communication is allowed
         */
        bool is_network_access_allowed(const std::string& source_endpoint,
                                     const std::string& destination_endpoint,
                                     const std::string& protocol = "tcp",
                                     int port = 0);
        
        // Getters for individual components
        IContinuousAuthentication* get_continuous_auth() const;
        IMicroSegmentation* get_microsegmentation() const;
        IJustInTimeAccess* get_jit_access() const;
        IDeviceAttestation* get_device_attestation() const;
    };

} // namespace zero_trust
} // namespace jadevectordb

#endif // JADEVECTORDB_ZERO_TRUST_H