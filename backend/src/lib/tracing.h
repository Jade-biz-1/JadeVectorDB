#ifndef JADEVECTORDB_TRACING_H
#define JADEVECTORDB_TRACING_H

#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>

namespace jadevectordb {

// Represents a trace span
struct TraceSpan {
    std::string trace_id;
    std::string span_id;
    std::string parent_span_id;
    std::string operation_name;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    std::unordered_map<std::string, std::string> tags;      // Key-value metadata
    std::unordered_map<std::string, std::string> logs;      // Log entries
    bool is_root_span;
    
    TraceSpan() : is_root_span(false) {}
    TraceSpan(const std::string& trace_id, const std::string& span_id, 
              const std::string& op_name)
        : trace_id(trace_id), span_id(span_id), operation_name(op_name),
          start_time(std::chrono::system_clock::now()), is_root_span(false) {}
};

// Configuration for tracing
struct TracingConfig {
    bool enabled = true;
    std::string collector_endpoint = "http://localhost:4317";  // OpenTelemetry collector
    std::string service_name = "jadevectordb";
    int flush_interval_ms = 1000;  // How often to send traces
    int max_queue_size = 1000;     // Max traces to queue before dropping
    bool sample_rate = 1.0;        // Fraction of requests to trace (1.0 = all, 0.5 = half, etc.)
    
    TracingConfig() = default;
};

/**
 * @brief Service for distributed tracing based on OpenTelemetry
 * 
 * This service implements distributed tracing to track request flow
 * across microservices and components for observability and debugging.
 */
class TracingService {
private:
    TracingConfig config_;
    std::string current_trace_id_;
    std::unordered_map<std::string, TraceSpan> active_spans_;
    std::mutex tracing_mutex_;
    
public:
    explicit TracingService();
    ~TracingService() = default;
    
    // Initialize the tracing service
    bool initialize(const TracingConfig& config);
    
    // Start a new trace with root span
    std::string start_trace(const std::string& operation_name);
    
    // Start a new span as a child of an existing trace
    std::string start_span(const std::string& trace_id, 
                          const std::string& parent_span_id,
                          const std::string& operation_name);
    
    // End a span (record end time)
    void end_span(const std::string& span_id);
    
    // Add tag to a span
    void add_tag(const std::string& span_id, const std::string& key, const std::string& value);
    
    // Add log to a span
    void add_log(const std::string& span_id, const std::string& key, const std::string& value);
    
    // Get trace by ID
    std::shared_ptr<TraceSpan> get_trace(const std::string& trace_id) const;
    
    // Get span by ID
    std::shared_ptr<TraceSpan> get_span(const std::string& span_id) const;
    
    // Export completed traces to collector
    bool export_traces();
    
    // Update tracing configuration
    void update_config(const TracingConfig& new_config);
    
    // Get current tracing configuration
    TracingConfig get_config() const;
    
    // Generate a new trace ID
    static std::string generate_trace_id();
    
    // Generate a new span ID
    static std::string generate_span_id();

private:
    // Internal helper methods
    void send_trace_to_collector(const TraceSpan& span);
    bool is_tracing_enabled() const;
    bool should_sample_request() const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_TRACING_H