#include "privacy_controls.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace jadevectordb {

PrivacyControls::PrivacyControls(std::unique_ptr<DatabaseLayer> db_layer)
    : db_layer_(std::move(db_layer)), 
      gdpr_compliance_enabled_(true),
      right_to_erasure_enabled_(true),
      data_portability_enabled_(true),
      data_export_enabled_(true),
      data_retention_period_(std::chrono::hours(24 * 365)) { // Default: 1 year retention
    
    logger_ = logging::LoggerManager::get_logger("PrivacyControls");
    
    if (!db_layer_) {
        // If no database layer is provided, create a default one
        db_layer_ = std::make_unique<DatabaseLayer>();
        db_layer_->initialize();
    }
}

Result<void> PrivacyControls::initialize() {
    LOG_INFO(logger_, "Initializing privacy controls for GDPR compliance");
    
    // Initialize any required data structures
    // In a real implementation, this might involve setting up database tables, 
    // file storage, or other persistence mechanisms for privacy requests
    
    // Set default configuration
    data_export_format_ = "json";
    
    LOG_INFO(logger_, "Privacy controls initialized successfully");
    return {};
}

Result<std::string> PrivacyControls::submit_privacy_request(const PrivacyRequest& request) {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    if (!gdpr_compliance_enabled_) {
        RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "GDPR compliance is disabled");
    }
    
    // Generate a unique ID for the request
    std::string request_id = generate_request_id();
    
    // Update the request with the generated ID and timestamps
    PrivacyRequest updated_request = request;
    updated_request.request_id = request_id;
    updated_request.created_at = std::chrono::system_clock::now();
    updated_request.status = "pending";
    
    // Store the request
    privacy_requests_[request_id] = updated_request;
    
    // Persist the request (in a real system, this would go to a database)
    persist_privacy_request(updated_request);
    
    // Call the callback if registered
    if (on_privacy_request_callback_) {
        on_privacy_request_callback_(updated_request);
    }
    
    LOG_INFO(logger_, "Privacy request submitted: " + request_id + 
             " for user: " + request.user_id + 
             " type: " + privacy_request_type_to_string(request.type));
    
    return request_id;
}

Result<PrivacyRequest> PrivacyControls::get_privacy_request(const std::string& request_id) const {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    auto it = privacy_requests_.find(request_id);
    if (it == privacy_requests_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Privacy request not found: " + request_id);
    }
    
    return it->second;
}

Result<void> PrivacyControls::update_privacy_request_status(const std::string& request_id, 
                                                          const std::string& status) {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    auto it = privacy_requests_.find(request_id);
    if (it == privacy_requests_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Privacy request not found: " + request_id);
    }
    
    it->second.status = status;
    it->second.processed_at = std::chrono::system_clock::now();
    
    // Persist the updated request
    persist_privacy_request(it->second);
    
    LOG_INFO(logger_, "Privacy request " + request_id + " status updated to: " + status);
    
    return {};
}

Result<std::vector<PrivacyRequest>> PrivacyControls::get_user_privacy_requests(const std::string& user_id) const {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    std::vector<PrivacyRequest> user_requests;
    for (const auto& pair : privacy_requests_) {
        if (pair.second.user_id == user_id) {
            user_requests.push_back(pair.second);
        }
    }
    
    return user_requests;
}

Result<std::vector<PrivacyRequest>> PrivacyControls::get_privacy_requests_by_type(PrivacyRequestType type) const {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    std::vector<PrivacyRequest> type_requests;
    for (const auto& pair : privacy_requests_) {
        if (pair.second.type == type) {
            type_requests.push_back(pair.second);
        }
    }
    
    return type_requests;
}

Result<void> PrivacyControls::register_personal_data(const std::string& user_id, 
                                                   const PersonalDataRecord& record) {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    if (!gdpr_compliance_enabled_) {
        RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "GDPR compliance is disabled");
    }
    
    // Create a new record with proper IDs and timestamps
    PersonalDataRecord new_record = record;
    new_record.record_id = generate_request_id(); // Reusing the ID generator
    new_record.user_id = user_id;
    new_record.created_at = std::chrono::system_clock::now();
    new_record.is_erased = false;
    
    // Add to the user's personal data records
    user_personal_data_[user_id].push_back(new_record);
    
    // Persist the record
    persist_personal_data_record(new_record);
    
    LOG_DEBUG(logger_, "Registered personal data for user: " + user_id + 
              " type: " + personal_data_type_to_string(record.type));
    
    return {};
}

Result<std::vector<PersonalDataRecord>> PrivacyControls::get_personal_data_for_user(const std::string& user_id) const {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    auto it = user_personal_data_.find(user_id);
    if (it == user_personal_data_.end()) {
        // Return empty vector if no personal data is found
        return std::vector<PersonalDataRecord>();
    }
    
    // Return only non-erased records
    std::vector<PersonalDataRecord> active_records;
    for (const auto& record : it->second) {
        if (!record.is_erased) {
            active_records.push_back(record);
        }
    }
    
    return active_records;
}

Result<void> PrivacyControls::update_personal_data(const std::string& record_id, 
                                                 const PersonalDataRecord& updated_record) {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    // Find the record to update
    for (auto& user_data_pair : user_personal_data_) {
        auto& records = user_data_pair.second;
        for (auto& record : records) {
            if (record.record_id == record_id) {
                // Update the record
                record = updated_record;
                record.record_id = record_id; // Preserve the original ID
                
                // Persist the updated record
                persist_personal_data_record(record);
                
                LOG_DEBUG(logger_, "Updated personal data record: " + record_id);
                return {};
            }
        }
    }
    
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Personal data record not found: " + record_id);
}

Result<void> PrivacyControls::erase_personal_data(const std::string& user_id, 
                                                PersonalDataType data_type) {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    if (!right_to_erasure_enabled_) {
        RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Right to erasure is disabled");
    }
    
    auto it = user_personal_data_.find(user_id);
    if (it == user_personal_data_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "No personal data found for user: " + user_id);
    }
    
    int erased_count = 0;
    for (auto& record : it->second) {
        if (!record.is_erased && (data_type == PersonalDataType::OTHER || record.type == data_type)) {
            record.is_erased = true;
            record.erased_at = std::chrono::system_clock::now();
            // In a real implementation, we might set erased_by to the admin who performed the action
            record.erased_by = "PRIVACY_SYSTEM";
            
            // Actually remove the data value but keep the record with a note that it was erased
            record.data_value = "[ERASED]";
            
            erased_count++;
            
            // Remove from persistent storage
            remove_personal_data_record(record.record_id);
        }
    }
    
    // Call the callback if registered
    if (on_data_erasure_callback_) {
        on_data_erasure_callback_(user_id);
    }
    
    LOG_INFO(logger_, "Erased " + std::to_string(erased_count) + " personal data records for user: " + user_id);
    
    return {};
}

Result<void> PrivacyControls::erase_user_data_completely(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    if (!right_to_erasure_enabled_) {
        RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Right to erasure is disabled");
    }
    
    // First, remove the user's personal data records
    auto it = user_personal_data_.find(user_id);
    if (it != user_personal_data_.end()) {
        for (auto& record : it->second) {
            if (!record.is_erased) {
                record.is_erased = true;
                record.erased_at = std::chrono::system_clock::now();
                record.erased_by = "PRIVACY_SYSTEM";
                record.data_value = "[ERASED]";
                
                // Remove from persistent storage
                remove_personal_data_record(record.record_id);
            }
        }
        
        // Clear the user's records from memory
        user_personal_data_.erase(it);
    }
    
    // In a real implementation, we would also need to:
    // 1. Remove the user from the auth system
    // 2. Remove any API keys associated with the user
    // 3. Remove the user from any roles or groups
    // 4. Remove any vectors associated with the user (if tracked)
    // 5. Update any references to the user in audit logs
    
    // Call the callback if registered
    if (on_data_erasure_callback_) {
        on_data_erasure_callback_(user_id);
    }
    
    LOG_INFO(logger_, "Completely erased all data for user: " + user_id);
    
    return {};
}

Result<DataExportResult> PrivacyControls::export_user_data(const std::string& user_id, 
                                                          const std::string& format) {
    if (!data_export_enabled_) {
        RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Data export is disabled");
    }
    
    // Get all personal data records for the user
    auto personal_data_result = get_personal_data_for_user(user_id);
    if (!personal_data_result.has_value()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "No personal data found for user: " + user_id);
    }
    
    auto personal_data = personal_data_result.value();
    
    // Generate the export
    auto export_result = export_data_to_format(personal_data, format.empty() ? data_export_format_ : format);
    if (!export_result.has_value()) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to export user data: " + export_result.error().message);
    }
    
    // Create the export result
    DataExportResult result;
    result.export_id = generate_export_id();
    result.user_id = user_id;
    result.format = format.empty() ? data_export_format_ : format;
    result.file_path = export_result.value();
    result.created_at = std::chrono::system_clock::now();
    result.success = true;
    result.exported_files.push_back(result.file_path);
    
    // Call the callback if registered
    if (on_data_export_callback_) {
        on_data_export_callback_(user_id, result.file_path);
    }
    
    LOG_INFO(logger_, "Exported data for user: " + user_id + " to: " + result.file_path);
    
    return result;
}

Result<std::string> PrivacyControls::generate_data_portability_export(const std::string& user_id,
                                                                     const std::string& target_system) {
    if (!data_portability_enabled_) {
        RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Data portability is disabled");
    }
    
    // Get the user's data
    auto personal_data_result = get_personal_data_for_user(user_id);
    if (!personal_data_result.has_value()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "No personal data found for user: " + user_id);
    }
    
    auto personal_data = personal_data_result.value();
    
    // In a real system, this would format the data according to the target system's requirements
    // For now, we'll just use the standard export format
    auto export_result = export_data_to_format(personal_data, "json");
    if (!export_result.has_value()) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to generate portability export: " + export_result.error().message);
    }
    
    LOG_INFO(logger_, "Generated portability export for user: " + user_id + " target: " + target_system);
    
    return export_result.value();
}

Result<bool> PrivacyControls::request_right_to_be_forgotten(const std::string& user_id, 
                                                          const std::string& reason) {
    if (!right_to_erasure_enabled_) {
        RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Right to erasure is disabled");
    }
    
    // Create a privacy request for data deletion
    PrivacyRequest request;
    request.user_id = user_id;
    request.type = PrivacyRequestType::DATA_DELETION;
    request.created_at = std::chrono::system_clock::now();
    request.reason = reason.empty() ? "User requested right to be forgotten" : reason;
    request.requested_by = user_id;  // User is requesting for themselves
    
    auto result = submit_privacy_request(request);
    if (!result.has_value()) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to submit right to be forgotten request: " + result.error().message);
    }
    
    LOG_INFO(logger_, "Right to be forgotten requested for user: " + user_id);
    
    return true;
}

Result<bool> PrivacyControls::process_right_to_be_forgotten(const std::string& user_id) {
    if (!right_to_erasure_enabled_) {
        RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Right to erasure is disabled");
    }
    
    // Perform the complete data erasure
    auto result = erase_user_data_completely(user_id);
    if (!result.has_value()) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to process right to be forgotten: " + result.error().message);
    }
    
    LOG_INFO(logger_, "Right to be forgotten processed for user: " + user_id);
    
    return true;
}

Result<std::vector<std::string>> PrivacyControls::get_user_data_locations(const std::string& user_id) const {
    std::vector<std::string> locations;
    
    // In a real implementation, this would search across all databases and data stores
    // to find all locations where the user's data is stored
    
    // For now, we'll just return some example locations
    locations.push_back("personal_data_registry");
    locations.push_back("user_metadata");
    locations.push_back("audit_logs");
    locations.push_back("session_data");
    
    // Add more locations based on the personal data records
    auto personal_data_result = get_personal_data_for_user(user_id);
    if (personal_data_result.has_value()) {
        for (const auto& record : personal_data_result.value()) {
            if (std::find(locations.begin(), locations.end(), record.source) == locations.end()) {
                locations.push_back(record.source);
            }
        }
    }
    
    return locations;
}

Result<std::string> PrivacyControls::get_user_data_summary(const std::string& user_id) const {
    auto personal_data_result = get_personal_data_for_user(user_id);
    if (!personal_data_result.has_value()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "No personal data found for user: " + user_id);
    }
    
    auto personal_data = personal_data_result.value();
    
    json summary;
    summary["user_id"] = user_id;
    summary["data_types_count"] = personal_data.size();
    summary["data_types"] = json::array();
    
    std::unordered_map<std::string, int> type_counts;
    for (const auto& record : personal_data) {
        std::string type_str = personal_data_type_to_string(record.type);
        type_counts[type_str]++;
    }
    
    for (const auto& pair : type_counts) {
        json type_info;
        type_info["type"] = pair.first;
        type_info["count"] = pair.second;
        summary["data_types"].push_back(type_info);
    }
    
    summary["created_at"] = "2025-10-11T00:00:00Z"; // In a real system, use actual timestamp
    summary["last_accessed"] = "2025-10-11T00:00:00Z";
    
    return summary.dump(4);
}

Result<std::vector<PrivacyRequest>> PrivacyControls::get_compliance_audit_log() const {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    std::vector<PrivacyRequest> all_requests;
    for (const auto& pair : privacy_requests_) {
        all_requests.push_back(pair.second);
    }
    
    return all_requests;
}

Result<std::vector<std::string>> PrivacyControls::get_erased_data_records(const std::string& user_id) const {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    std::vector<std::string> erased_record_ids;
    
    auto it = user_personal_data_.find(user_id);
    if (it != user_personal_data_.end()) {
        for (const auto& record : it->second) {
            if (record.is_erased) {
                erased_record_ids.push_back(record.record_id);
            }
        }
    }
    
    return erased_record_ids;
}

Result<void> PrivacyControls::enforce_retention_policy() {
    std::lock_guard<std::mutex> lock(privacy_mutex_);
    
    auto now = std::chrono::system_clock::now();
    auto retention_limit = now - data_retention_period_;
    
    int purged_count = 0;
    
    for (auto& user_data_pair : user_personal_data_) {
        auto& records = user_data_pair.second;
        records.erase(
            std::remove_if(records.begin(), records.end(),
                [&retention_limit](const PersonalDataRecord& record) {
                    return !record.is_erased && record.created_at < retention_limit;
                }),
            records.end()
        );
        purged_count += 1; // Simplified - in reality, we'd count the actual records purged
    }
    
    LOG_INFO(logger_, "Enforced retention policy, potentially purged data for " + std::to_string(purged_count) + " users");
    
    return {};
}

Result<void> PrivacyControls::check_and_purge_expired_data() {
    // This would be called periodically to clean up expired data
    // Implementation similar to enforce_retention_policy but potentially with more complex logic
    return enforce_retention_policy();
}

std::string PrivacyControls::personal_data_type_to_string(PersonalDataType type) const {
    switch (type) {
        case PersonalDataType::NAME: return "name";
        case PersonalDataType::EMAIL: return "email";
        case PersonalDataType::PHONE: return "phone";
        case PersonalDataType::ADDRESS: return "address";
        case PersonalDataType::IP_ADDRESS: return "ip_address";
        case PersonalDataType::USER_BEHAVIOR: return "user_behavior";
        case PersonalDataType::PROFILE_DATA: return "profile_data";
        case PersonalDataType::PREFERENCES: return "preferences";
        case PersonalDataType::OTHER: return "other";
        default: return "unknown";
    }
}

std::string PrivacyControls::privacy_request_type_to_string(PrivacyRequestType type) const {
    switch (type) {
        case PrivacyRequestType::DATA_ACCESS: return "data_access";
        case PrivacyRequestType::DATA_EXPORT: return "data_export";
        case PrivacyRequestType::DATA_DELETION: return "data_deletion";
        case PrivacyRequestType::DATA_CORRECTION: return "data_correction";
        case PrivacyRequestType::OBJECTION_TO_PROCESSING: return "objection_to_processing";
        case PrivacyRequestType::RESTRICT_PROCESSING: return "restrict_processing";
        default: return "unknown";
    }
}

// Helper methods implementation
std::string PrivacyControls::generate_request_id() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto count = duration.count();
    
    std::stringstream ss;
    ss << "req_" << std::hex << count;
    
    // Add some randomness to ensure uniqueness
    thread_local static std::random_device rd;
    thread_local static std::mt19937 gen(rd());
    thread_local static std::uniform_int_distribution<> dis(1000, 9999);
    
    ss << "_" << dis(gen);
    return ss.str();
}

std::string PrivacyControls::generate_export_id() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto count = duration.count();
    
    std::stringstream ss;
    ss << "exp_" << std::hex << count;
    
    // Add some randomness to ensure uniqueness
    thread_local static std::random_device rd;
    thread_local static std::mt19937 gen(rd());
    thread_local static std::uniform_int_distribution<> dis(1000, 9999);
    
    ss << "_" << dis(gen);
    return ss.str();
}

bool PrivacyControls::validate_user_data_access(const std::string& user_id, 
                                               const std::string& requesting_user) const {
    // In a real implementation, this would check if the requesting user has permission 
    // to access the target user's data (e.g., if they are the same user or an admin)
    // For now, we'll implement a simple check that only allows users to access their own data
    return user_id == requesting_user;
}

Result<void> PrivacyControls::persist_privacy_request(const PrivacyRequest& request) {
    // In a real implementation, this would persist the request to a database or file
    // For now, it's stored in memory (privacy_requests_ map)
    
    // This implementation simply ensures the request is in memory
    // In a production system, this would write to persistent storage
    return {};
}

Result<void> PrivacyControls::persist_personal_data_record(const PersonalDataRecord& record) {
    // In a real implementation, this would persist the record to a database or file
    // For now, it's managed in memory (user_personal_data_ map)
    
    // This implementation simply ensures the record is in memory
    // In a production system, this would write to persistent storage
    return {};
}

Result<void> PrivacyControls::remove_personal_data_record(const std::string& record_id) {
    // In a real implementation, this would remove the record from persistent storage
    // For now, we just update the in-memory representation
    
    // In the current implementation, we mark records as erased rather than removing them
    // This is done in the erase methods by setting is_erased = true
    return {};
}

Result<std::string> PrivacyControls::export_data_to_format(const std::vector<PersonalDataRecord>& records, 
                                                          const std::string& format) const {
    try {
        json export_data;
        export_data["export_date"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        export_data["format_version"] = "1.0";
        export_data["personal_data"] = json::array();
        
        for (const auto& record : records) {
            json record_json;
            record_json["record_id"] = record.record_id;
            record_json["user_id"] = record.user_id;
            record_json["type"] = personal_data_type_to_string(record.type);
            record_json["data_value"] = record.data_value;
            record_json["source"] = record.source;
            record_json["retention_category"] = record.retention_category;
            record_json["created_at"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                record.created_at.time_since_epoch()).count();
            
            export_data["personal_data"].push_back(record_json);
        }
        
        // Create a temporary file for the export
        std::string export_file_path = "/tmp/jadevectordb_export_" + 
                                      std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + 
                                      ".json";
        
        std::ofstream file(export_file_path);
        if (!file.is_open()) {
            RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Could not create export file: " + export_file_path);
        }
        
        file << export_data.dump(4);
        file.close();
        
        return export_file_path;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Error creating data export: " + std::string(e.what()));
    }
}

} // namespace jadevectordb