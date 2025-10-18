#!/bin/bash

# Script to install static analysis tools (clang-tidy, cppcheck) needed for C++ compliance verification

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up static analysis tools..."

# Check if running on a supported system (Ubuntu/Debian)
if [ -f /etc/debian_version ]; then
    echo "Installing clang-tidy and cppcheck via apt..."
    sudo apt-get update
    sudo apt-get install -y clang-tidy cppcheck
elif [ -f /etc/redhat-release ]; then
    echo "Installing clang-tidy and cppcheck via yum/dnf..."
    # For CentOS/RHEL/Fedora - might need EPEL repository for cppcheck
    sudo dnf install -y clang-tools-extra cppcheck || {
        sudo yum install -y clang-tools-extra cppcheck
    }
elif [ "$(uname)" = "Darwin" ]; then
    echo "Installing clang-tidy and cppcheck via homebrew..."
    brew install clang-tidy cppcheck
else
    echo "Unsupported OS. Please install clang-tidy and cppcheck manually."
    echo "For Ubuntu/Debian: sudo apt-get install clang-tidy cppcheck"
    echo "For CentOS/RHEL/Fedora: sudo dnf install clang-tools-extra cppcheck"
    echo "For macOS: brew install clang-tidy cppcheck"
    exit 1
fi

echo "Static analysis tools installed successfully."
echo ""
echo "To run static analysis on the project, use:"
echo "  cd backend"
echo "  python3 ../scripts/run-static-analysis.py"
echo ""
echo "This will run both clang-tidy and cppcheck on the project code."