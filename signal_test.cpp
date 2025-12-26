#include <iostream>
#include <csignal>
#include <unistd.h>
#include <fstream>
#include <ctime>

// Global flag
volatile bool running = true;

// Signal handler
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        // Write to a file to confirm signal handler is called
        std::ofstream signal_log("signal_test.log", std::ios::app);
        signal_log << "Received signal: " << signal << " at " << std::time(nullptr) << std::endl;
        signal_log.close();

        std::cout << "\nReceived shutdown signal (signal=" << signal << "). Setting running=false..." << std::endl;
        std::cout.flush();

        running = false;
    }
}

int main() {
    std::cout << "Signal handler test program. Press Ctrl+C to test signal handling." << std::endl;

    // Install signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Main loop
    while (running) {
        std::cout << "Running... (press Ctrl+C)" << std::endl;
        sleep(1);
    }

    std::cout << "Exiting gracefully." << std::endl;
    return 0;
}