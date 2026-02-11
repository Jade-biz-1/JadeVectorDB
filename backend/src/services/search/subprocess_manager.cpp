#include "subprocess_manager.h"
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <poll.h>
#endif
#include <sstream>
#include <iostream>
#include <cstring>

namespace jadedb {
namespace search {

SubprocessManager::SubprocessManager(const SubprocessConfig& config)
    : config_(config),
      logger_(jadevectordb::logging::LoggerManager::get_logger("SubprocessManager")),
      stdin_pipe_(nullptr),
      stdout_pipe_(nullptr),
      stderr_pipe_(nullptr),
      pid_(-1),
      status_(SubprocessStatus::NOT_STARTED),
      heartbeat_running_(false),
      stop_requested_(false),
      restart_count_(0) {
}

SubprocessManager::~SubprocessManager() {
    stop();
}

jadevectordb::Result<void> SubprocessManager::start() {
    std::lock_guard<std::mutex> lock(comm_mutex_);

    if (status_ != SubprocessStatus::NOT_STARTED &&
        status_ != SubprocessStatus::ERROR &&
        status_ != SubprocessStatus::TERMINATED) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Subprocess already started"
        ));
    }

    LOG_INFO(logger_, "Starting Python reranking subprocess");
    status_ = SubprocessStatus::STARTING;

    // Launch the process
    auto launch_result = launch_process();
    if (!launch_result.has_value()) {
        status_ = SubprocessStatus::ERROR;
        return launch_result;
    }

    // Wait for ready signal
    LOG_DEBUG(logger_, "Waiting for 'ready' signal from subprocess");
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > config_.startup_timeout) {
            LOG_ERROR(logger_, "Subprocess startup timeout");
            kill_process();
            status_ = SubprocessStatus::ERROR;
            return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                jadevectordb::ErrorCode::TIMEOUT,
                "Subprocess failed to start within timeout"
            ));
        }

        // Try to read response
        auto remaining = config_.startup_timeout -
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        auto response = read_json(remaining);

        if (response.has_value()) {
            auto json = response.value();
            if (json.contains("type") && json["type"] == "ready") {
                LOG_INFO(logger_, "Subprocess ready, model: " << json.value("model", "unknown"));
                status_ = SubprocessStatus::READY;
                last_heartbeat_ = std::chrono::system_clock::now();

                // Start heartbeat thread
                heartbeat_running_ = true;
                heartbeat_thread_ = std::thread(&SubprocessManager::heartbeat_loop, this);

                return jadevectordb::Result<void>{};
            } else if (json.contains("error")) {
                LOG_ERROR(logger_, "Subprocess reported error: " << json["error"]);
                kill_process();
                status_ = SubprocessStatus::ERROR;
                return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                    jadevectordb::ErrorCode::INITIALIZE_ERROR,
                    "Subprocess initialization error: " + json["error"].get<std::string>()
                ));
            }
        }

        // Check if process died
        if (!is_process_alive()) {
            std::string stderr_output = read_stderr();
            LOG_ERROR(logger_, "Subprocess terminated during startup. stderr: " << stderr_output);
            status_ = SubprocessStatus::TERMINATED;
            return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                jadevectordb::ErrorCode::SERVICE_ERROR,
                "Subprocess terminated: " + stderr_output
            ));
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void SubprocessManager::stop() {
    LOG_INFO(logger_, "Stopping subprocess");
    stop_requested_ = true;

    // Stop heartbeat thread
    if (heartbeat_running_) {
        heartbeat_running_ = false;
        if (heartbeat_thread_.joinable()) {
            heartbeat_thread_.join();
        }
    }

    std::lock_guard<std::mutex> lock(comm_mutex_);

    // Send shutdown command
    if (status_ == SubprocessStatus::READY || status_ == SubprocessStatus::BUSY) {
        try {
            nlohmann::json shutdown_cmd;
            shutdown_cmd["type"] = "shutdown";
            write_json(shutdown_cmd);

            // Wait a bit for graceful shutdown
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        } catch (...) {
            LOG_WARN(logger_, "Failed to send shutdown command, will force kill");
        }
    }

    kill_process();
    status_ = SubprocessStatus::TERMINATED;
}

jadevectordb::Result<nlohmann::json> SubprocessManager::send_request(
    const nlohmann::json& request
) {
    std::lock_guard<std::mutex> lock(comm_mutex_);

    if (status_ != SubprocessStatus::READY) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_UNAVAILABLE,
            "Subprocess not ready"
        ));
    }

    status_ = SubprocessStatus::BUSY;

    // Write request
    auto write_result = write_json(request);
    if (!write_result.has_value()) {
        status_ = SubprocessStatus::ERROR;
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_ERROR,
            "Failed to write request: " + write_result.error().message
        ));
    }

    // Read response
    auto response = read_json(config_.request_timeout);

    status_ = SubprocessStatus::READY;

    if (!response.has_value()) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_ERROR,
            "Failed to read response: " + response.error().message
        ));
    }

    auto json = response.value();

    // Check for error response
    if (json.contains("error")) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_ERROR,
            "Subprocess error: " + json["error"].get<std::string>()
        ));
    }

    return response;
}

jadevectordb::Result<bool> SubprocessManager::send_heartbeat() {
    try {
        nlohmann::json heartbeat;
        heartbeat["type"] = "heartbeat";

        auto response = send_request(heartbeat);
        if (!response.has_value()) {
            return jadevectordb::Result<bool>(false);
        }

        auto json = response.value();
        if (json.contains("type") && json["type"] == "heartbeat" &&
            json.contains("status") && json["status"] == "alive") {
            last_heartbeat_ = std::chrono::system_clock::now();
            return jadevectordb::Result<bool>(true);
        }

        return jadevectordb::Result<bool>(false);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Heartbeat failed: " << e.what());
        return jadevectordb::Result<bool>(false);
    }
}

SubprocessStatus SubprocessManager::get_status() const {
    return status_.load();
}

bool SubprocessManager::is_ready() const {
    return status_.load() == SubprocessStatus::READY;
}

pid_t SubprocessManager::get_pid() const {
    return pid_;
}

jadevectordb::Result<void> SubprocessManager::launch_process() {
    // Create pipes for stdin, stdout, stderr
    int stdin_pipe[2], stdout_pipe[2], stderr_pipe[2];

    if (pipe(stdin_pipe) == -1 || pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_ERROR,
            "Failed to create pipes: " + std::string(strerror(errno))
        ));
    }

    pid_ = fork();

    if (pid_ == -1) {
        // Fork failed
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_ERROR,
            "Failed to fork process: " + std::string(strerror(errno))
        ));
    }

    if (pid_ == 0) {
        // Child process

        // Redirect stdin
        dup2(stdin_pipe[0], STDIN_FILENO);
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);

        // Redirect stdout
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);

        // Redirect stderr
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);

        // Execute Python script
        std::vector<const char*> args;
        args.push_back(config_.python_executable.c_str());
        args.push_back(config_.script_path.c_str());
        args.push_back("--model");
        args.push_back(config_.model_name.c_str());
        args.push_back("--batch-size");
        std::string batch_size_str = std::to_string(config_.batch_size);
        args.push_back(batch_size_str.c_str());
        args.push_back(nullptr);

        execvp(config_.python_executable.c_str(), const_cast<char* const*>(args.data()));

        // If execvp returns, it failed
        std::cerr << "Failed to execute Python script: " << strerror(errno) << std::endl;
        _exit(1);
    }

    // Parent process
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    // Convert file descriptors to FILE*
    stdin_pipe_ = fdopen(stdin_pipe[1], "w");
    stdout_pipe_ = fdopen(stdout_pipe[0], "r");
    stderr_pipe_ = fdopen(stderr_pipe[0], "r");

    if (!stdin_pipe_ || !stdout_pipe_ || !stderr_pipe_) {
        kill_process();
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_ERROR,
            "Failed to create FILE streams"
        ));
    }

    // Set non-blocking mode for stdout to enable timeout
    int stdout_fd = fileno(stdout_pipe_);
    int flags = fcntl(stdout_fd, F_GETFL, 0);
    fcntl(stdout_fd, F_SETFL, flags | O_NONBLOCK);

    LOG_INFO(logger_, "Subprocess launched with PID: " << pid_);
    return jadevectordb::Result<void>{};
}

jadevectordb::Result<void> SubprocessManager::write_json(const nlohmann::json& json) {
    try {
        std::string json_str = json.dump();
        json_str += "\n";

        size_t written = fwrite(json_str.c_str(), 1, json_str.size(), stdin_pipe_);
        if (written != json_str.size()) {
            return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                jadevectordb::ErrorCode::SERVICE_ERROR,
                "Failed to write complete JSON to subprocess"
            ));
        }

        fflush(stdin_pipe_);
        LOG_DEBUG(logger_, "Sent to subprocess: " << json_str.substr(0, 100));

        return jadevectordb::Result<void>{};
    } catch (const std::exception& e) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_ERROR,
            "JSON write error: " + std::string(e.what())
        ));
    }
}

jadevectordb::Result<nlohmann::json> SubprocessManager::read_json(
    std::chrono::milliseconds timeout
) {
    try {
        int fd = fileno(stdout_pipe_);
        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLIN;

        auto start_time = std::chrono::steady_clock::now();
        std::string line;

        while (true) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed >= timeout) {
                return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                    jadevectordb::ErrorCode::TIMEOUT,
                    "Timeout reading from subprocess"
                ));
            }

            auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                timeout - elapsed
            ).count();

            int poll_result = poll(&pfd, 1, static_cast<int>(remaining_ms));

            if (poll_result < 0) {
                return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                    jadevectordb::ErrorCode::SERVICE_ERROR,
                    "Poll error: " + std::string(strerror(errno))
                ));
            }

            if (poll_result == 0) {
                // Timeout
                return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                    jadevectordb::ErrorCode::TIMEOUT,
                    "Timeout reading from subprocess"
                ));
            }

            if (pfd.revents & POLLIN) {
                // Data available
                char buffer[4096];
                if (fgets(buffer, sizeof(buffer), stdout_pipe_) != nullptr) {
                    line += buffer;

                    // Check if we have a complete line
                    if (!line.empty() && line.back() == '\n') {
                        line.pop_back();  // Remove newline
                        LOG_DEBUG(logger_, "Received from subprocess: " << line.substr(0, 100));

                        // Parse JSON
                        try {
                            auto json = nlohmann::json::parse(line);
                            return jadevectordb::Result<nlohmann::json>(json);
                        } catch (const nlohmann::json::exception& e) {
                            return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                                jadevectordb::ErrorCode::DESERIALIZATION_ERROR,
                                "JSON parse error: " + std::string(e.what())
                            ));
                        }
                    }
                } else {
                    // Read error or EOF
                    if (feof(stdout_pipe_)) {
                        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                            jadevectordb::ErrorCode::SERVICE_ERROR,
                            "Subprocess closed stdout (EOF)"
                        ));
                    }
                    return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                        jadevectordb::ErrorCode::SERVICE_ERROR,
                        "Error reading from subprocess: " + std::string(strerror(errno))
                    ));
                }
            }

            if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
                return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
                    jadevectordb::ErrorCode::SERVICE_ERROR,
                    "Subprocess pipe error or closed"
                ));
            }
        }
    } catch (const std::exception& e) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_ERROR,
            "Exception reading JSON: " + std::string(e.what())
        ));
    }
}

std::string SubprocessManager::read_stderr() {
    if (!stderr_pipe_) {
        return "";
    }

    std::stringstream ss;
    char buffer[1024];

    // Set non-blocking mode temporarily
    int fd = fileno(stderr_pipe_);
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    while (fgets(buffer, sizeof(buffer), stderr_pipe_) != nullptr) {
        ss << buffer;
    }

    // Restore flags
    fcntl(fd, F_SETFL, flags);

    return ss.str();
}

void SubprocessManager::heartbeat_loop() {
    LOG_DEBUG(logger_, "Heartbeat thread started");

    while (heartbeat_running_ && !stop_requested_) {
        std::this_thread::sleep_for(config_.heartbeat_interval);

        if (!heartbeat_running_ || stop_requested_) {
            break;
        }

        // Send heartbeat
        auto result = send_heartbeat();
        if (!result.has_value() || !result.value()) {
            LOG_WARN(logger_, "Heartbeat failed");

            // Check if process is still alive
            if (!is_process_alive()) {
                LOG_ERROR(logger_, "Subprocess died, attempting restart");
                status_ = SubprocessStatus::ERROR;

                // Attempt restart if not too many attempts
                if (restart_count_ < MAX_RESTART_ATTEMPTS) {
                    auto restart_result = attempt_restart();
                    if (restart_result.has_value()) {
                        LOG_INFO(logger_, "Subprocess restarted successfully");
                    } else {
                        LOG_ERROR(logger_, "Restart failed: " << restart_result.error().message);
                    }
                } else {
                    LOG_ERROR(logger_, "Max restart attempts reached, giving up");
                    heartbeat_running_ = false;
                }
            }
        }
    }

    LOG_DEBUG(logger_, "Heartbeat thread stopped");
}

bool SubprocessManager::is_process_alive() const {
    if (pid_ <= 0) {
        return false;
    }

    int status;
    pid_t result = waitpid(pid_, &status, WNOHANG);

    if (result == 0) {
        // Process still running
        return true;
    }

    if (result == pid_) {
        // Process exited
        if (WIFEXITED(status)) {
            LOG_DEBUG(logger_, "Process exited with code: " << WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
            LOG_DEBUG(logger_, "Process killed by signal: " << WTERMSIG(status));
        }
        return false;
    }

    // Error or other case
    return false;
}

void SubprocessManager::kill_process() {
    if (pid_ > 0) {
        LOG_DEBUG(logger_, "Killing subprocess PID: " << pid_);

        // Try SIGTERM first (graceful)
        kill(pid_, SIGTERM);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // If still alive, SIGKILL
        if (is_process_alive()) {
            LOG_DEBUG(logger_, "Process still alive, sending SIGKILL");
            kill(pid_, SIGKILL);
        }

        // Wait for process to clean up
        waitpid(pid_, nullptr, 0);
        pid_ = -1;
    }

    // Close pipes
    if (stdin_pipe_) {
        fclose(stdin_pipe_);
        stdin_pipe_ = nullptr;
    }
    if (stdout_pipe_) {
        fclose(stdout_pipe_);
        stdout_pipe_ = nullptr;
    }
    if (stderr_pipe_) {
        fclose(stderr_pipe_);
        stderr_pipe_ = nullptr;
    }
}

jadevectordb::Result<void> SubprocessManager::attempt_restart() {
    LOG_INFO(logger_, "Attempting to restart subprocess (attempt "
             << (restart_count_ + 1) << "/" << MAX_RESTART_ATTEMPTS << ")");

    restart_count_++;

    // Clean up old process
    kill_process();

    // Reset status
    status_ = SubprocessStatus::NOT_STARTED;

    // Try to start again
    return start();
}

} // namespace search
} // namespace jadedb
