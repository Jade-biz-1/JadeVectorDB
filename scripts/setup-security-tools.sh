#!/bin/bash

# Script to install security testing tools needed for security hardening

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up security testing tools..."

# Define the list of tools to install
SECURITY_TOOLS=("nmap" "nikto" "sqlmap" "gosec" "bandit" "brakeman" "snyk")

# Determine the OS and install tools accordingly
if [ -f /etc/debian_version ]; then
    echo "Installing security tools via apt..."
    sudo apt-get update
    
    # Install general security testing tools
    sudo apt-get install -y nmap nikto sqlmap
    
    # Install Golang and gosec for Go security analysis
    if ! command -v go &> /dev/null; then
        echo "Installing Go..."
        sudo apt-get install -y golang
    fi
    
    if ! command -v gosec &> /dev/null; then
        echo "Installing gosec..."
        go install github.com/securego/gosec/v2/cmd/gosec@latest
        # Add go bin to PATH if not already there
        export PATH=$PATH:$(go env GOPATH)/bin
    fi
    
elif [ -f /etc/redhat-release ]; then
    echo "Installing security tools via yum/dnf..."
    # For CentOS/RHEL/Fedora
    sudo dnf install -y nmap nikto sqlmap || sudo yum install -y nmap nikto sqlmap
    
    # Install Golang and gosec for Go security analysis
    if ! command -v go &> /dev/null; then
        echo "Installing Go..."
        sudo dnf install -y golang || sudo yum install -y golang
    fi
    
    if ! command -v gosec &> /dev/null; then
        echo "Installing gosec..."
        go install github.com/securego/gosec/v2/cmd/gosec@latest
        # Add go bin to PATH if not already there
        export PATH=$PATH:$(go env GOPATH)/bin
    fi
    
elif [ "$(uname)" = "Darwin" ]; then
    echo "Installing security tools via homebrew..."
    # Install general security testing tools
    brew install nmap nikto sqlmap
    
    # Install Golang and gosec for Go security analysis
    if ! command -v go &> /dev/null; then
        echo "Installing Go..."
        brew install go
    fi
    
    if ! command -v gosec &> /dev/null; then
        echo "Installing gosec..."
        go install github.com/securego/gosec/v2/cmd/gosec@latest
    fi
else
    echo "Unsupported OS. Please install security tools manually."
    echo "Essential tools to install:"
    for tool in "${SECURITY_TOOLS[@]}"; do
        echo "  - $tool"
    done
    echo ""
    echo "For Ubuntu/Debian: sudo apt-get install nmap nikto sqlmap golang"
    echo "For CentOS/RHEL/Fedora: sudo dnf install nmap nikto sqlmap golang"
    echo "For macOS: brew install nmap nikto sqlmap go"
    exit 1
fi

# Install Node.js tools (npm required)
if command -v npm &> /dev/null; then
    echo "Installing Node.js security tools..."
    npm install -g snyk
else
    echo "Node.js/npm not found. Skipping snyk installation."
    echo "To install Node.js: https://nodejs.org/"
    echo "After installing Node.js, run: npm install -g snyk"
fi

echo "Security testing tools installed successfully."
echo ""
echo "To run security tests, use:"
echo "  python3 ../scripts/run-security-tests.py"
echo ""
echo "This will run various security assessments on the project."