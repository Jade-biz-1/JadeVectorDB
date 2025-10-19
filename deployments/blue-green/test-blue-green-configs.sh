#!/bin/bash

# Test script to validate blue-green deployment configurations

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting validation of blue-green deployment configurations..."

# Check that all required files exist
echo "Checking for required files..."
required_files=(
    "k8s/blue-green-deployment.yaml"
    "k8s/ingress-routing.yaml"
    "k8s/traffic-routing.yaml"
    "k8s/health-check-services.yaml"
    "k8s/monitoring.yaml"
    "k8s/canary-deployment.yaml"
    "switch-traffic.sh"
    "health-check.sh"
    "rollback-monitor.sh"
    "canary-deploy.sh"
    "deploy-blue-green.sh"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✓ All required files are present"
else
    echo "✗ Missing files:"
    printf '%s\n' "${missing_files[@]}"
    exit 1
fi

# Validate YAML files syntax
echo "Validating YAML configuration files..."
yaml_files=(
    "k8s/blue-green-deployment.yaml"
    "k8s/ingress-routing.yaml"
    "k8s/traffic-routing.yaml"
    "k8s/health-check-services.yaml"
    "k8s/monitoring.yaml"
    "k8s/canary-deployment.yaml"
)

for yaml_file in "${yaml_files[@]}"; do
    if command -v yamllint >/dev/null 2>&1; then
        if yamllint "$yaml_file"; then
            echo "✓ $yaml_file is valid YAML"
        else
            echo "✗ $yaml_file has YAML syntax errors"
            exit 1
        fi
    else
        echo "yamllint not found, skipping YAML validation for $yaml_file (install yamllint for validation)"
    fi
done

# Check shell script syntax
echo "Validating shell script syntax..."
shell_scripts=(
    "switch-traffic.sh"
    "health-check.sh"
    "rollback-monitor.sh"
    "canary-deploy.sh"
    "deploy-blue-green.sh"
)

for script in "${shell_scripts[@]}"; do
    if bash -n "$script"; then
        echo "✓ $script has valid syntax"
    else
        echo "✗ $script has syntax errors"
        exit 1
    fi
done

# Check that shell scripts are executable
echo "Checking script permissions..."
for script in "${shell_scripts[@]}"; do
    if [ -x "$script" ]; then
        echo "✓ $script is executable"
    else
        echo "✗ $script is not executable"
        exit 1
    fi
done

# Check Kubernetes resource validity (basic check)
echo "Validating basic Kubernetes resource structure..."
for yaml_file in "${yaml_files[@]}"; do
    # Basic check for required Kubernetes fields
    if grep -q "apiVersion:" "$yaml_file" && grep -q "kind:" "$yaml_file" && grep -q "metadata:" "$yaml_file"; then
        echo "✓ $yaml_file has basic Kubernetes resource structure"
    else
        echo "✗ $yaml_file is missing required Kubernetes resource fields"
        exit 1
    fi
done

echo "All blue-green deployment configurations validated successfully!"
echo
echo "The blue-green deployment system includes:"
echo "- Kubernetes deployments for blue and green environments"
echo "- Traffic routing mechanisms (Ingress and Istio)"
echo "- Health checking services"
echo "- Automated rollback capabilities"
echo "- Canary deployment support"
echo "- Comprehensive deployment scripts"