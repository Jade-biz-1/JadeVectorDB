#!/bin/bash

# Automated Rollback Script for Blue-Green Deployment
# This script performs automated rollback if health checks fail after traffic switch

set -e  # Exit immediately if a command exits with a non-zero status

NAMESPACE=${1:-"jadevectordb-system"}
TIMEOUT=${2:-60}  # seconds to wait for health check after traffic switch
CHECK_INTERVAL=${3:-10}  # seconds between health checks

echo "Starting automated rollback monitor in namespace $NAMESPACE..."
echo "Timeout: ${TIMEOUT}s, Check interval: ${CHECK_INTERVAL}s"

# Check if kubectl is available
if ! command -v kubectl >/dev/null 2>&1; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Get the current active environment from the main service
get_current_active_env() {
    kubectl get svc jadevectordb-service -n $NAMESPACE -o jsonpath='{.spec.selector.version}' 2>/dev/null || echo "unknown"
}

# Get current active environment
ACTIVE_ENV=$(get_current_active_env)
echo "Current active environment: $ACTIVE_ENV"

# If active environment is unknown, exit
if [ "$ACTIVE_ENV" = "unknown" ]; then
    echo "ERROR: Could not determine active environment"
    exit 1
fi

# Calculate the previous environment (the one that was active before current)
if [ "$ACTIVE_ENV" = "blue" ]; then
    PREVIOUS_ENV="green"
else
    PREVIOUS_ENV="blue"
fi

echo "Previous environment (will be reverted to if needed): $PREVIOUS_ENV"

# Wait for the specified timeout before starting health checks
echo "Waiting ${TIMEOUT}s for traffic to switch and services to stabilize..."
sleep $TIMEOUT

# Function to check if environment is healthy
is_environment_healthy() {
    local env=$1
    local service_name="jadevectordb-${env}-service"
    
    # Get service IP
    service_ip=$(kubectl get svc $service_name -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    if [ -z "$service_ip" ]; then
        echo "Could not get service IP for ${env} environment"
        return 1
    fi
    
    # Test health endpoint
    if curl -f -s --max-time 10 http://$service_ip:8080/health > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Continuously monitor the health of the active environment
echo "Starting health monitoring for active environment: $ACTIVE_ENV"
start_time=$(date +%s)
timeout_time=$((start_time + 300))  # 5 minutes total monitoring time

while [ $(date +%s) -lt $timeout_time ]; do
    if is_environment_healthy $ACTIVE_ENV; then
        echo "Active environment ($ACTIVE_ENV) is healthy"
    else
        echo "ERROR: Active environment ($ACTIVE_ENV) is not healthy"
        
        # Rollback to the previous environment
        echo "Initiating rollback to $PREVIOUS_ENV environment..."
        
        # Update the main service to point to the previous (stable) environment
        cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: jadevectordb-service
  namespace: $NAMESPACE
spec:
  selector:
    app: jadevectordb
    version: $PREVIOUS_ENV
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080
    - protocol: TCP
      port: 8081
      targetPort: 8081
  type: LoadBalancer
EOF
        
        echo "Rollback to $PREVIOUS_ENV completed"
        echo "Active environment is now: $PREVIOUS_ENV"
        
        # Exit with error status to indicate rollback happened
        exit 1
    fi
    
    # Wait for the next check
    sleep $CHECK_INTERVAL
done

echo "Health monitoring completed. Active environment remains: $ACTIVE_ENV"