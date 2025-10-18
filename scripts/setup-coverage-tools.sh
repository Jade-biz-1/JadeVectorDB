#!/bin/bash

# Script to install coverage tools (lcov) needed for test coverage measurement

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up coverage measurement tools..."

# Check if running on a supported system (Ubuntu/Debian)
if [ -f /etc/debian_version ]; then
    echo "Installing lcov via apt..."
    sudo apt-get update
    sudo apt-get install -y lcov
elif [ -f /etc/redhat-release ]; then
    echo "Installing lcov via yum/dnf..."
    # For CentOS/RHEL/Fedora
    sudo dnf install -y lcov || sudo yum install -y lcov
elif [ "$(uname)" = "Darwin" ]; then
    echo "Installing lcov via homebrew..."
    brew install lcov
else
    echo "Unsupported OS. Please install lcov manually."
    echo "For Ubuntu/Debian: sudo apt-get install lcov"
    echo "For CentOS/RHEL/Fedora: sudo dnf install lcov or sudo yum install lcov"
    echo "For macOS: brew install lcov"
    exit 1
fi

echo "Coverage tools installed successfully."
echo ""
echo "To build the project with coverage enabled, run:"
echo "  cd backend"
echo "  mkdir build"
echo "  cd build"
echo "  cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_COVERAGE=ON .."
echo "  make"
echo "  make coverage"
echo ""
echo "The coverage report will be available in the 'coverage_report' directory."