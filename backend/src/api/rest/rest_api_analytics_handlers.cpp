// REST API Analytics Handlers - Query analytics and insights endpoints
#include "rest_api.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include "analytics/query_analytics_manager.h"
#include "analytics/analytics_engine.h"
#include "analytics/batch_processor.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

namespace jadevectordb {

// Helper function to convert ISO timestamp to milliseconds since epoch
int64_t parse_timestamp(const std::string& timestamp) {
    // Simple implementation: assume format "YYYY-MM-DDTHH:MM:SS" or Unix milliseconds
    if (timestamp.empty()) {
        return 0;
    }

    // Try parsing as number (Unix timestamp in ms)
    try {
        return std::stoll(timestamp);
    } catch (...) {
        // If not a number, return current time - 24 hours
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() - 86400000;
    }
}

// GET /v1/databases/{id}/analytics/stats
crow::response RestApiImpl::handle_analytics_stats_request(const crow::request& req, const std::string& database_id) {
    LOG_INFO(logger_, "Analytics stats request for database: " << database_id);

    try {
        // Get query parameters
        auto start_time_param = req.url_params.get("start_time");
        auto end_time_param = req.url_params.get("end_time");
        auto granularity_param = req.url_params.get("granularity");

        // Parse timestamps (default: last 24 hours)
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        int64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        int64_t start_time = end_time - 86400000;  // 24 hours ago

        if (start_time_param) {
            start_time = parse_timestamp(start_time_param);
        }
        if (end_time_param) {
            end_time = parse_timestamp(end_time_param);
        }

        // Parse granularity (default: hourly)
        jadedb::analytics::TimeBucket bucket = jadedb::analytics::TimeBucket::HOURLY;
        if (granularity_param) {
            std::string gran = granularity_param;
            if (gran == "hourly") {
                bucket = jadedb::analytics::TimeBucket::HOURLY;
            } else if (gran == "daily") {
                bucket = jadedb::analytics::TimeBucket::DAILY;
            } else if (gran == "weekly") {
                bucket = jadedb::analytics::TimeBucket::WEEKLY;
            } else if (gran == "monthly") {
                bucket = jadedb::analytics::TimeBucket::MONTHLY;
            }
        }

        // Get analytics engine
        auto engine = get_or_create_analytics_engine(database_id);
        if (!engine) {
            return crow::response(500, "Failed to create analytics engine");
        }

        // Compute statistics
        auto stats_result = engine->compute_statistics(database_id, start_time, end_time, bucket);
        if (!stats_result.has_value()) {
            return crow::response(500, stats_result.error().message);
        }

        // Build response
        json response;
        response["database_id"] = database_id;
        response["start_time"] = start_time;
        response["end_time"] = end_time;
        response["granularity"] = granularity_param ? std::string(granularity_param) : "hourly";

        json stats_array = json::array();
        for (const auto& stat : stats_result.value()) {
            json stat_obj;
            stat_obj["time_bucket"] = stat.time_bucket;
            stat_obj["bucket_type"] = static_cast<int>(stat.bucket_type);
            stat_obj["total_queries"] = stat.total_queries;
            stat_obj["successful_queries"] = stat.successful_queries;
            stat_obj["failed_queries"] = stat.failed_queries;
            stat_obj["zero_result_queries"] = stat.zero_result_queries;
            stat_obj["unique_users"] = stat.unique_users;
            stat_obj["unique_sessions"] = stat.unique_sessions;
            stat_obj["avg_latency_ms"] = stat.avg_latency_ms;
            stat_obj["p50_latency_ms"] = stat.p50_latency_ms;
            stat_obj["p95_latency_ms"] = stat.p95_latency_ms;
            stat_obj["p99_latency_ms"] = stat.p99_latency_ms;
            stat_obj["vector_queries"] = stat.vector_queries;
            stat_obj["hybrid_queries"] = stat.hybrid_queries;
            stat_obj["bm25_queries"] = stat.bm25_queries;
            stat_obj["reranked_queries"] = stat.reranked_queries;
            stats_array.push_back(stat_obj);
        }

        response["stats"] = stats_array;

        return crow::response(200, response.dump());

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Analytics stats error: " << e.what());
        return crow::response(500, std::string("Error: ") + e.what());
    }
}

// GET /v1/databases/{id}/analytics/queries
crow::response RestApiImpl::handle_analytics_queries_request(const crow::request& req, const std::string& database_id) {
    LOG_INFO(logger_, "Analytics queries request for database: " << database_id);

    try {
        // Get query parameters
        auto start_time_param = req.url_params.get("start_time");
        auto end_time_param = req.url_params.get("end_time");
        // auto query_type_param = req.url_params.get("query_type");  // For future filtering
        auto limit_param = req.url_params.get("limit");
        auto offset_param = req.url_params.get("offset");

        // Parse parameters
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        int64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        int64_t start_time = end_time - 86400000;  // 24 hours ago

        if (start_time_param) {
            start_time = parse_timestamp(start_time_param);
        }
        if (end_time_param) {
            end_time = parse_timestamp(end_time_param);
        }

        size_t limit = 100;
        size_t offset = 0;
        if (limit_param) {
            limit = std::stoul(limit_param);
        }
        if (offset_param) {
            offset = std::stoul(offset_param);
        }

        // Get analytics manager
        auto manager = get_or_create_analytics_manager(database_id);
        if (!manager) {
            return crow::response(500, "Failed to create analytics manager");
        }

        // Get recent queries (implementation would query database)
        // For now, return a placeholder response
        json response;
        response["database_id"] = database_id;
        response["start_time"] = start_time;
        response["end_time"] = end_time;
        response["limit"] = limit;
        response["offset"] = offset;
        response["queries"] = json::array();

        return crow::response(200, response.dump());

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Analytics queries error: " << e.what());
        return crow::response(500, std::string("Error: ") + e.what());
    }
}

// GET /v1/databases/{id}/analytics/patterns
crow::response RestApiImpl::handle_analytics_patterns_request(const crow::request& req, const std::string& database_id) {
    LOG_INFO(logger_, "Analytics patterns request for database: " << database_id);

    try {
        // Get query parameters
        auto start_time_param = req.url_params.get("start_time");
        auto end_time_param = req.url_params.get("end_time");
        auto min_count_param = req.url_params.get("min_count");
        auto limit_param = req.url_params.get("limit");

        // Parse parameters
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        int64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        int64_t start_time = end_time - 86400000;  // 24 hours ago

        if (start_time_param) {
            start_time = parse_timestamp(start_time_param);
        }
        if (end_time_param) {
            end_time = parse_timestamp(end_time_param);
        }

        size_t min_count = 2;
        size_t limit = 100;
        if (min_count_param) {
            min_count = std::stoul(min_count_param);
        }
        if (limit_param) {
            limit = std::stoul(limit_param);
        }

        // Get analytics engine
        auto engine = get_or_create_analytics_engine(database_id);
        if (!engine) {
            return crow::response(500, "Failed to create analytics engine");
        }

        // Identify patterns
        auto patterns_result = engine->identify_patterns(database_id, start_time, end_time, min_count, limit);
        if (!patterns_result.has_value()) {
            return crow::response(500, patterns_result.error().message);
        }

        // Build response
        json response;
        response["database_id"] = database_id;
        response["start_time"] = start_time;
        response["end_time"] = end_time;
        response["min_count"] = min_count;

        json patterns_array = json::array();
        for (const auto& pattern : patterns_result.value()) {
            json pattern_obj;
            pattern_obj["normalized_text"] = pattern.normalized_text;
            pattern_obj["query_count"] = pattern.query_count;
            pattern_obj["avg_latency_ms"] = pattern.avg_latency_ms;
            pattern_obj["avg_results"] = pattern.avg_results;
            pattern_obj["first_seen"] = pattern.first_seen;
            pattern_obj["last_seen"] = pattern.last_seen;
            patterns_array.push_back(pattern_obj);
        }

        response["patterns"] = patterns_array;

        return crow::response(200, response.dump());

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Analytics patterns error: " << e.what());
        return crow::response(500, std::string("Error: ") + e.what());
    }
}

// GET /v1/databases/{id}/analytics/insights
crow::response RestApiImpl::handle_analytics_insights_request(const crow::request& req, const std::string& database_id) {
    LOG_INFO(logger_, "Analytics insights request for database: " << database_id);

    try {
        // Get query parameters
        auto start_time_param = req.url_params.get("start_time");
        auto end_time_param = req.url_params.get("end_time");

        // Parse parameters
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        int64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        int64_t start_time = end_time - 86400000;  // 24 hours ago

        if (start_time_param) {
            start_time = parse_timestamp(start_time_param);
        }
        if (end_time_param) {
            end_time = parse_timestamp(end_time_param);
        }

        // Get analytics engine
        auto engine = get_or_create_analytics_engine(database_id);
        if (!engine) {
            return crow::response(500, "Failed to create analytics engine");
        }

        // Generate insights
        auto insights_result = engine->generate_insights(database_id, start_time, end_time);
        if (!insights_result.has_value()) {
            return crow::response(500, insights_result.error().message);
        }

        const auto& insights = insights_result.value();

        // Build response
        json response;
        response["database_id"] = database_id;
        response["start_time"] = start_time;
        response["end_time"] = end_time;

        // Statistics summary from AnalyticsInsights
        json stats_summary;
        stats_summary["overall_success_rate"] = insights.overall_success_rate;
        stats_summary["qps_avg"] = insights.qps_avg;
        stats_summary["qps_peak"] = insights.qps_peak;
        stats_summary["peak_hour"] = insights.peak_hour;
        response["summary"] = stats_summary;

        // Common patterns
        json patterns_array = json::array();
        for (const auto& pattern : insights.top_patterns) {
            json pattern_obj;
            pattern_obj["pattern"] = pattern.normalized_text;
            pattern_obj["count"] = pattern.query_count;
            pattern_obj["avg_latency_ms"] = pattern.avg_latency_ms;
            patterns_array.push_back(pattern_obj);
        }
        response["common_patterns"] = patterns_array;

        // Slow queries
        json slow_queries_array = json::array();
        for (const auto& slow_query : insights.slow_queries) {
            json query_obj;
            query_obj["query_text"] = slow_query.query_text;
            query_obj["latency_ms"] = slow_query.total_time_ms;
            query_obj["timestamp"] = slow_query.timestamp;
            slow_queries_array.push_back(query_obj);
        }
        response["slow_queries"] = slow_queries_array;

        // Zero-result queries
        json zero_result_array = json::array();
        for (const auto& zero_query : insights.zero_result_queries) {
            json query_obj;
            query_obj["query_text"] = zero_query.query_text;
            query_obj["count"] = zero_query.occurrence_count;
            zero_result_array.push_back(query_obj);
        }
        response["zero_result_queries"] = zero_result_array;

        // Trending queries
        json trending_array = json::array();
        for (const auto& trending : insights.trending_queries) {
            json query_obj;
            query_obj["query_text"] = trending.query_text;
            query_obj["current_count"] = trending.current_count;
            query_obj["previous_count"] = trending.previous_count;
            query_obj["growth_rate"] = trending.growth_rate;
            trending_array.push_back(query_obj);
        }
        response["trending"] = trending_array;

        return crow::response(200, response.dump());

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Analytics insights error: " << e.what());
        return crow::response(500, std::string("Error: ") + e.what());
    }
}

// GET /v1/databases/{id}/analytics/trending
crow::response RestApiImpl::handle_analytics_trending_request(const crow::request& req, const std::string& database_id) {
    LOG_INFO(logger_, "Analytics trending request for database: " << database_id);

    try {
        // Get query parameters
        auto start_time_param = req.url_params.get("start_time");
        auto end_time_param = req.url_params.get("end_time");
        auto min_growth_param = req.url_params.get("min_growth");
        auto limit_param = req.url_params.get("limit");

        // Parse parameters
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        int64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        int64_t start_time = end_time - 86400000;  // 24 hours ago

        if (start_time_param) {
            start_time = parse_timestamp(start_time_param);
        }
        if (end_time_param) {
            end_time = parse_timestamp(end_time_param);
        }

        double min_growth = 0.5;  // 50% growth minimum
        size_t limit = 100;
        if (min_growth_param) {
            min_growth = std::stod(min_growth_param);
        }
        if (limit_param) {
            limit = std::stoul(limit_param);
        }

        // Get analytics engine
        auto engine = get_or_create_analytics_engine(database_id);
        if (!engine) {
            return crow::response(500, "Failed to create analytics engine");
        }

        // Detect trending queries (bucket_type defaults to DAILY, then min_growth_rate, then limit)
        auto trending_result = engine->detect_trending(database_id, start_time, end_time,
            jadedb::analytics::TimeBucket::DAILY, min_growth, limit);
        if (!trending_result.has_value()) {
            return crow::response(500, trending_result.error().message);
        }

        // Build response
        json response;
        response["database_id"] = database_id;
        response["start_time"] = start_time;
        response["end_time"] = end_time;
        response["min_growth"] = min_growth;

        json trending_array = json::array();
        for (const auto& trending : trending_result.value()) {
            json query_obj;
            query_obj["query_text"] = trending.query_text;
            query_obj["current_count"] = trending.current_count;
            query_obj["previous_count"] = trending.previous_count;
            query_obj["growth_rate"] = trending.growth_rate;
            trending_array.push_back(query_obj);
        }

        response["trending"] = trending_array;

        return crow::response(200, response.dump());

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Analytics trending error: " << e.what());
        return crow::response(500, std::string("Error: ") + e.what());
    }
}

// POST /v1/databases/{id}/analytics/feedback
crow::response RestApiImpl::handle_analytics_feedback_request(const crow::request& req, const std::string& database_id) {
    LOG_INFO(logger_, "Analytics feedback request for database: " << database_id);

    try {
        // Parse request body
        auto body = json::parse(req.body);

        std::string query_id = body.value("query_id", "");
        // Extract feedback data (to be stored in future implementation)
        // std::string user_id = body.value("user_id", "");
        // int rating = body.value("rating", 0);
        // std::string feedback_text = body.value("feedback_text", "");
        // std::string clicked_result_id = body.value("clicked_result_id", "");
        // int clicked_rank = body.value("clicked_rank", 0);

        if (query_id.empty()) {
            return crow::response(400, "query_id is required");
        }

        // Get analytics manager
        auto manager = get_or_create_analytics_manager(database_id);
        if (!manager) {
            return crow::response(500, "Failed to create analytics manager");
        }

        // Store feedback (would need to implement in QueryLogger/AnalyticsEngine)
        // For now, return success
        json response;
        response["success"] = true;
        response["query_id"] = query_id;
        response["message"] = "Feedback recorded";

        return crow::response(200, response.dump());

    } catch (const json::exception& e) {
        return crow::response(400, std::string("JSON error: ") + e.what());
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Analytics feedback error: " << e.what());
        return crow::response(500, std::string("Error: ") + e.what());
    }
}

// GET /v1/databases/{id}/analytics/export
crow::response RestApiImpl::handle_analytics_export_request(const crow::request& req, const std::string& database_id) {
    LOG_INFO(logger_, "Analytics export request for database: " << database_id);

    try {
        // Get query parameters
        auto format_param = req.url_params.get("format");
        auto start_time_param = req.url_params.get("start_time");
        auto end_time_param = req.url_params.get("end_time");

        std::string format = format_param ? std::string(format_param) : "json";

        // Parse timestamps
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        int64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        int64_t start_time = end_time - 86400000;  // 24 hours ago

        if (start_time_param) {
            start_time = parse_timestamp(start_time_param);
        }
        if (end_time_param) {
            end_time = parse_timestamp(end_time_param);
        }

        // Get analytics engine
        auto engine = get_or_create_analytics_engine(database_id);
        if (!engine) {
            return crow::response(500, "Failed to create analytics engine");
        }

        // Generate insights for export
        auto insights_result = engine->generate_insights(database_id, start_time, end_time);
        if (!insights_result.has_value()) {
            return crow::response(500, insights_result.error().message);
        }

        const auto& insights = insights_result.value();

        if (format == "csv") {
            // Generate CSV export
            std::ostringstream csv;
            csv << "Type,Data,Count,Metric,Value\n";

            // Export patterns
            for (const auto& pattern : insights.top_patterns) {
                csv << "Pattern,\"" << pattern.normalized_text << "\","
                    << pattern.query_count << ",avg_latency,"
                    << pattern.avg_latency_ms << "\n";
            }

            // Export slow queries
            for (const auto& slow : insights.slow_queries) {
                csv << "SlowQuery,\"" << slow.query_text << "\",1,latency,"
                    << slow.total_time_ms << "\n";
            }

            crow::response resp(200, csv.str());
            resp.add_header("Content-Type", "text/csv");
            resp.add_header("Content-Disposition",
                "attachment; filename=analytics_" + database_id + ".csv");
            return resp;

        } else {
            // JSON export (same as insights endpoint)
            json response;
            response["database_id"] = database_id;
            response["start_time"] = start_time;
            response["end_time"] = end_time;
            response["export_format"] = "json";

            // Include all insights data
            json patterns_array = json::array();
            for (const auto& pattern : insights.top_patterns) {
                json pattern_obj;
                pattern_obj["pattern"] = pattern.normalized_text;
                pattern_obj["count"] = pattern.query_count;
                pattern_obj["avg_latency_ms"] = pattern.avg_latency_ms;
                patterns_array.push_back(pattern_obj);
            }
            response["patterns"] = patterns_array;

            json slow_queries_array = json::array();
            for (const auto& slow_query : insights.slow_queries) {
                json query_obj;
                query_obj["query_text"] = slow_query.query_text;
                query_obj["latency_ms"] = slow_query.total_time_ms;
                slow_queries_array.push_back(query_obj);
            }
            response["slow_queries"] = slow_queries_array;

            crow::response resp(200, response.dump());
            resp.add_header("Content-Type", "application/json");
            resp.add_header("Content-Disposition",
                "attachment; filename=analytics_" + database_id + ".json");
            return resp;
        }

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Analytics export error: " << e.what());
        return crow::response(500, std::string("Error: ") + e.what());
    }
}

// Helper: Get or create analytics manager
std::shared_ptr<jadedb::analytics::QueryAnalyticsManager>
RestApiImpl::get_or_create_analytics_manager(const std::string& database_id) {
    auto it = analytics_managers_.find(database_id);
    if (it != analytics_managers_.end()) {
        return it->second;
    }

    // Create new analytics manager
    std::string analytics_db_path = "./data/analytics_" + database_id + ".db";

    auto manager = std::make_shared<jadedb::analytics::QueryAnalyticsManager>(database_id, analytics_db_path);
    auto init_result = manager->initialize();
    if (!init_result.has_value()) {
        LOG_ERROR(logger_, "Failed to initialize analytics manager: " << init_result.error().message);
        return nullptr;
    }

    analytics_managers_[database_id] = manager;
    return manager;
}

// Helper: Get or create analytics engine
std::shared_ptr<jadedb::analytics::AnalyticsEngine>
RestApiImpl::get_or_create_analytics_engine(const std::string& database_id) {
    auto it = analytics_engines_.find(database_id);
    if (it != analytics_engines_.end()) {
        return it->second;
    }

    // Create new analytics engine
    std::string analytics_db_path = "./data/analytics_" + database_id + ".db";
    auto engine = std::make_shared<jadedb::analytics::AnalyticsEngine>(analytics_db_path);
    auto init_result = engine->initialize();
    if (!init_result.has_value()) {
        LOG_ERROR(logger_, "Failed to initialize analytics engine: " << init_result.error().message);
        return nullptr;
    }

    analytics_engines_[database_id] = engine;
    return engine;
}

// Helper: Get or create batch processor
std::shared_ptr<jadedb::analytics::BatchProcessor>
RestApiImpl::get_or_create_batch_processor(const std::string& database_id) {
    auto it = batch_processors_.find(database_id);
    if (it != batch_processors_.end()) {
        return it->second;
    }

    // Get or create analytics engine first
    auto engine = get_or_create_analytics_engine(database_id);
    if (!engine) {
        return nullptr;
    }

    // Create new batch processor
    jadedb::analytics::BatchProcessorConfig config;
    config.enable_hourly_aggregation = true;
    config.enable_daily_cleanup = true;
    config.retention_days = 30;

    auto processor = std::make_shared<jadedb::analytics::BatchProcessor>(database_id, engine, config);
    auto start_result = processor->start();
    if (!start_result.has_value()) {
        LOG_ERROR(logger_, "Failed to start batch processor: " << start_result.error().message);
        return nullptr;
    }

    batch_processors_[database_id] = processor;
    return processor;
}

} // namespace jadevectordb
