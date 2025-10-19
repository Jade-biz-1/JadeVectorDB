#!/bin/bash

# Health check script for blue-green deployment
# This script checks the health of both blue and green environments

set -e  # Exit immediately if a command exits with a non-zero status

NAMESPACE=${1:-"jadevectordb-system"}
HEALTH_PATH=${2:-"/health"}
TIMEOUT=${3:-30}  # seconds

echo "Starting health checks for blue-green deployment in namespace $NAMESPACE..."

# Check if kubectl is available
if ! command -v kubectl >/dev/null 2>&1; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Function to check health of a specific environment
check_environment_health() {
    local env=$1
    local service_name="jadevectordb-${env}-service"
    
    echo "Checking health for ${env} environment..."
    
    # Get pods for this environment
    pods=$(kubectl get pods -n $NAMESPACE -l app=jadevectordb,version=$env --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}')
    
    if [ -z "$pods" ]; then
        echo "ERROR: No running pods found for ${env} environment"
        return 1
    fi
    
    # Check each pod for readiness
    for pod in $pods; do
        echo "Checking readiness of pod: $pod"
        if ! kubectl wait --for=condition=ready pod $pod -n $NAMESPACE --timeout=${TIMEOUT}s; then
            echo "ERROR: Pod $pod is not ready"
            return 1
        fi
        
        # Check health endpoint
        echo "Checking health endpoint for pod: $pod"
        if ! kubectl exec -n $NAMESPACE $pod -- curl -f -s http://localhost:8080/health > /dev/null; then
            echo "ERROR: Health check failed for pod $pod"
            return 1
        fi
    done
    
    # Check service connectivity
    service_ip=$(kubectl get svc $service_name -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    if [ -z "$service_ip" ]; then
        echo "ERROR: Could not get service IP for ${env} environment"
        return 1
    fi
    
    echo "Service IP for ${env} environment: $service_ip"
    
    # Test service health endpoint
    echo "Checking health endpoint via service for ${env} environment..."
    if ! curl -f -s --max-time $TIMEOUT http://$service_ip:8080/health > /dev/null; then
        echo "ERROR: Health check failed for ${env} service"
        return 1
    fi
    
    echo "Health check passed for ${env} environment"
    return 0
}

# Check health of both environments
blue_healthy=0
green_healthy=0

if check_environment_health "blue"; then
    blue_healthy=1
    echo "✓ Blue environment is healthy"
else
    echo "✗ Blue environment is not healthy"
fi

if check_environment_health "green"; then
    green_healthy=1
    echo "✓ Green environment is healthy"
else
    echo "✗ Green environment is not healthy"
fi

# Summary
echo
echo "Health check summary:"
echo "Blue environment: $([ $blue_healthy -eq 1 ] && echo 'HEALTHY' || echo 'UNHEALTHY')"
echo "Green environment: $([ $green_healthy -eq 1 ] && echo 'HEALTHY' || echo 'UNHEALTHY')"

# Exit with appropriate status
if [ $blue_healthy -eq 1 ] && [ $green_healthy -eq 1 ]; then
    echo "Both environments are healthy"
    exit 0
elif [ $blue_healthy -eq 1 ] || [ $green_healthy -eq 1 ]; then
    echo "At least one environment is healthy"
    exit 0
else
    echo "ERROR: Both environments are unhealthy"
    exit 1
fi