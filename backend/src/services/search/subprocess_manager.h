#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/types.h>
#endif
#include <nlohmann/json.hpp>
#include "lib/error_handling.h"
#include "lib/logging.h"

namespace jadedb {
namespace search {

/**
 * @brief Configuration for Python subprocess
 */
struct SubprocessConfig {
    std::string python_executable = "python3";
    std::string script_path = "python/reranking_server.py";
    std::string model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2";
    int batch_size = 32;
    std::chrono::milliseconds startup_timeout{10000};
    std::chrono::milliseconds request_timeout{5000};
    std::chrono::milliseconds heartbeat_interval{30000};

    SubprocessConfig() = default;
};

/**
 * @brief Status of subprocess
 */
enum class SubprocessStatus {
    NOT_STARTED,
    STARTING,
    READY,
    BUSY,
    ERROR,
    TERMINATED
};

/**
 * @brief Manages Python subprocess for reranking
 *
 * Handles:
 * - Process lifecycle (spawn, monitor, terminate)
 * - Bidirectional JSON communication via stdin/stdout
 * - Heartbeat monitoring
 * - Error recovery and auto-restart
 * - Thread-safe request handling
 */
class SubprocessManager {
public:
    /**
     * @brief Construct subprocess manager
     * @param config Subprocess configuration
     */
    explicit SubprocessManager(const SubprocessConfig& config = SubprocessConfig());

    ~SubprocessManager();

    // Non-copyable
    SubprocessManager(const SubprocessManager&) = delete;
    SubprocessManager& operator=(const SubprocessManager&) = delete;

    /**
     * @brief Start the Python subprocess
     * @return Result<void> Success or error
     */
    jadevectordb::Result<void> start();

    /**
     * @brief Stop the Python subprocess gracefully
     */
    void stop();

    /**
     * @brief Send JSON request and receive response
     * @param request JSON request object
     * @return Result<nlohmann::json> Response or error
     */
    jadevectordb::Result<nlohmann::json> send_request(const nlohmann::json& request);

    /**
     * @brief Send heartbeat to check subprocess health
     * @return Result<bool> True if alive, false otherwise
     */
    jadevectordb::Result<bool> send_heartbeat();

    /**
     * @brief Get subprocess status
     */
    SubprocessStatus get_status() const;

    /**
     * @brief Check if subprocess is alive and ready
     */
    bool is_ready() const;

    /**
     * @brief Get subprocess PID
     */
    pid_t get_pid() const;

private:
    SubprocessConfig config_;
    std::shared_ptr<jadevectordb::logging::Logger> logger_;

    // Process handles
    FILE* stdin_pipe_;
    FILE* stdout_pipe_;
    FILE* stderr_pipe_;
    pid_t pid_;

    // Status tracking
    std::atomic<SubprocessStatus> status_;
    mutable std::mutex comm_mutex_;

    // Heartbeat thread
    std::thread heartbeat_thread_;
    std::atomic<bool> heartbeat_running_;
    std::atomic<bool> stop_requested_;
    std::chrono::system_clock::time_point last_heartbeat_;

    // Restart tracking
    std::atomic<int> restart_count_;
    static constexpr int MAX_RESTART_ATTEMPTS = 3;

    /**
     * @brief Launch subprocess using popen-style approach
     */
    jadevectordb::Result<void> launch_process();

    /**
     * @brief Write JSON line to stdin
     */
    jadevectordb::Result<void> write_json(const nlohmann::json& json);

    /**
     * @brief Read JSON line from stdout with timeout
     */
    jadevectordb::Result<nlohmann::json> read_json(
        std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)
    );

    /**
     * @brief Read stderr for error messages
     */
    std::string read_stderr();

    /**
     * @brief Heartbeat monitoring loop
     */
    void heartbeat_loop();

    /**
     * @brief Check if process is still alive
     */
    bool is_process_alive() const;

    /**
     * @brief Kill subprocess forcefully
     */
    void kill_process();

    /**
     * @brief Attempt to restart subprocess
     */
    jadevectordb::Result<void> attempt_restart();
};

} // namespace search
} // namespace jadedb
