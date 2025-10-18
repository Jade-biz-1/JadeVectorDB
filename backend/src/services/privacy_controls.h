#ifndef JADEVECTORDB_PRIVACY_CONTROLS_H
#define JADEVECTORDB_PRIVACY_CONTROLS_H

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <functional>

#include "models/database.h"
#include "models/vector.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include "services/database_layer.h"

namespace jadevectordb {

// GDPR compliance requirements
enum class DataSubjectRight {
    RIGHT_OF_ACCESS,           // Right to access personal data
    RIGHT_TO_RECTIFICATION,    // Right to correct inaccurate data
    RIGHT_TO_ERASURE,          // Right to erasure ('right to be forgotten')
    RIGHT_TO_RESTRICT_PROCESSING,  // Right to restrict processing
    RIGHT_TO_DATA_PORTABILITY,     // Right to data portability
    RIGHT_TO_OBJECT,              // Right to object to processing
    RIGHT_NOT_TO_BE_SUBJECTED_TO_AUTOMATED_DECISION_MAKING  // Right to not be subject to automated decision-making
};

// Types of personal data that may be stored
enum class PersonalDataType {
    NAME,
    EMAIL,
    PHONE,
    ADDRESS,
    IP_ADDRESS,
    USER_BEHAVIOR,
    PROFILE_DATA,
    PREFERENCES,
    OTHER
};

// Request types for GDPR compliance
enum class PrivacyRequestType {
    DATA_ACCESS,
    DATA_EXPORT,
    DATA_DELETION,
    DATA_CORRECTION,
    OBJECTION_TO_PROCESSING,
    RESTRICT_PROCESSING
};

struct PrivacyRequest {
    std::string request_id;
    std::string user_id;
    PrivacyRequestType type;
    std::vector<PersonalDataType> data_types;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point processed_at;
    std::string status;  // "pending", "approved", "rejected", "completed"
    std::string requested_by;
    std::string reason;
    std::string additional_details;
    
    PrivacyRequest() : status("pending") {}
};

struct PersonalDataRecord {
    std::string record_id;
    std::string user_id;
    PersonalDataType type;
    std::string data_value;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_accessed;
    std::string source;  // Where the data came from
    bool is_erased;      // Whether the data has been erased
    std::chrono::system_clock::time_point erased_at;
    std::string erased_by;
    std::string retention_category;  // Category for retention policy
};

struct DataExportResult {
    std::string export_id;
    std::string user_id;
    std::string format;  // JSON, CSV, etc.
    std::string file_path;
    std::chrono::system_clock::time_point created_at;
    bool success;
    std::string error_message;
    std::vector<std::string> exported_files;
};

class PrivacyControls {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::unique_ptr<DatabaseLayer> db_layer_;
    mutable std::mutex privacy_mutex_;
    
    // Request tracking
    std::unordered_map<std::string, PrivacyRequest> privacy_requests_;
    
    // Personal data tracking
    std::unordered_map<std::string, std::vector<PersonalDataRecord>> user_personal_data_;
    
    // Configuration
    bool gdpr_compliance_enabled_;
    bool right_to_erasure_enabled_;
    bool data_portability_enabled_;
    bool data_export_enabled_;
    std::chrono::hours data_retention_period_;
    std::string data_export_format_;
    
    // Callbacks for compliance events
    std::function<void(const PrivacyRequest&)> on_privacy_request_callback_;
    std::function<void(const std::string& user_id)> on_data_erasure_callback_;
    std::function<void(const std::string& user_id, const std::string& export_path)> on_data_export_callback_;

public:
    explicit PrivacyControls(std::unique_ptr<DatabaseLayer> db_layer = nullptr);
    ~PrivacyControls() = default;
    
    // Initialize the privacy controls
    Result<void> initialize();
    
    // Configuration methods
    void enable_gdpr_compliance(bool enable) { gdpr_compliance_enabled_ = enable; }
    void enable_right_to_erasure(bool enable) { right_to_erasure_enabled_ = enable; }
    void enable_data_portability(bool enable) { data_portability_enabled_ = enable; }
    void enable_data_export(bool enable) { data_export_enabled_ = enable; }
    void set_data_retention_period(std::chrono::hours period) { data_retention_period_ = period; }
    void set_data_export_format(const std::string& format) { data_export_format_ = format; }
    
    // Privacy request management
    Result<std::string> submit_privacy_request(const PrivacyRequest& request);
    Result<PrivacyRequest> get_privacy_request(const std::string& request_id) const;
    Result<void> update_privacy_request_status(const std::string& request_id, const std::string& status);
    Result<std::vector<PrivacyRequest>> get_user_privacy_requests(const std::string& user_id) const;
    Result<std::vector<PrivacyRequest>> get_privacy_requests_by_type(PrivacyRequestType type) const;
    
    // Personal data management
    Result<void> register_personal_data(const std::string& user_id, 
                                      const PersonalDataRecord& record);
    Result<std::vector<PersonalDataRecord>> get_personal_data_for_user(const std::string& user_id) const;
    Result<void> update_personal_data(const std::string& record_id, 
                                    const PersonalDataRecord& updated_record);
    Result<void> erase_personal_data(const std::string& user_id, 
                                   PersonalDataType data_type = PersonalDataType::OTHER);
    Result<void> erase_user_data_completely(const std::string& user_id);
    
    // Data export functionality
    Result<DataExportResult> export_user_data(const std::string& user_id, 
                                            const std::string& format = "json");
    Result<std::string> generate_data_portability_export(const std::string& user_id,
                                                        const std::string& target_system);
    
    // Right to be forgotten implementation
    Result<bool> request_right_to_be_forgotten(const std::string& user_id, 
                                             const std::string& reason = "");
    Result<bool> process_right_to_be_forgotten(const std::string& user_id);
    
    // Data access and portability
    Result<std::vector<std::string>> get_user_data_locations(const std::string& user_id) const;
    Result<std::string> get_user_data_summary(const std::string& user_id) const;
    
    // Audit and tracking
    Result<std::vector<PrivacyRequest>> get_compliance_audit_log() const;
    Result<std::vector<std::string>> get_erased_data_records(const std::string& user_id) const;
    
    // Retention policy enforcement
    Result<void> enforce_retention_policy();
    Result<void> check_and_purge_expired_data();
    
    // Set callback functions
    void set_privacy_request_callback(std::function<void(const PrivacyRequest&)> callback) {
        on_privacy_request_callback_ = callback;
    }
    void set_data_erasure_callback(std::function<void(const std::string&)> callback) {
        on_data_erasure_callback_ = callback;
    }
    void set_data_export_callback(std::function<void(const std::string&, const std::string&)> callback) {
        on_data_export_callback_ = callback;
    }
    
    // Utility methods
    bool is_gdpr_compliant() const { return gdpr_compliance_enabled_; }
    std::string personal_data_type_to_string(PersonalDataType type) const;
    std::string privacy_request_type_to_string(PrivacyRequestType type) const;

private:
    // Helper methods
    std::string generate_request_id() const;
    std::string generate_export_id() const;
    bool validate_user_data_access(const std::string& user_id, const std::string& requesting_user) const;
    Result<void> persist_privacy_request(const PrivacyRequest& request);
    Result<void> persist_personal_data_record(const PersonalDataRecord& record);
    Result<void> remove_personal_data_record(const std::string& record_id);
    Result<std::string> export_data_to_format(const std::vector<PersonalDataRecord>& records, 
                                            const std::string& format) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_PRIVACY_CONTROLS_H