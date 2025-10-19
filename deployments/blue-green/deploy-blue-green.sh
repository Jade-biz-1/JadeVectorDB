#!/bin/bash

# Comprehensive Blue-Green Deployment Script for JadeVectorDB
# This script orchestrates the complete blue-green deployment process

set -e  # Exit immediately if a command exits with a non-zero status

# Default configuration
NAMESPACE="jadevectordb-system"
BLUE_VERSION=${1:-"1.0.0"}
GREEN_VERSION=${2:-"1.1.0"}
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
ROLLOUT_DELAY=${3:-60}  # seconds to wait before switching traffic
ACTION=${4:-"deploy"}  # Options: deploy, promote, rollback, info

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}JadeVectorDB Blue-Green Deployment Script${NC}"
echo "==============================================="

# Check prerequisites
echo "Checking prerequisites..."
if ! command -v kubectl >/dev/null 2>&1; then
    echo -e "${RED}Error: kubectl is not installed or not in PATH${NC}"
    exit 1
fi

if ! command -v helm >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: helm is not installed or not in PATH (some features may be limited)${NC}"
fi

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get current active environment
get_current_active_env() {
    local current_env=$(kubectl get svc jadevectordb-service -n $NAMESPACE -o jsonpath='{.spec.selector.version}' 2>/dev/null || echo "unknown")
    echo $current_env
}

# Function to check if a deployment exists
deployment_exists() {
    local version=$1
    if kubectl get deployment jadevectordb-$version -n $NAMESPACE >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to check if all pods in an environment are ready
check_environment_ready() {
    local version=$1
    local timeout=${2:-300}  # 5 minutes default
    
    print_status "Waiting for $version environment to be ready..."
    
    local end_time=$(($(date +%s) + timeout))
    while [ $(date +%s) -lt $end_time ]; do
        # Check if deployment exists
        if ! deployment_exists $version; then
            print_error "Deployment jadevectordb-$version not found"
            return 1
        fi
        
        # Get the number of ready replicas
        ready_replicas=$(kubectl get deployment jadevectordb-$version -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
        desired_replicas=$(kubectl get deployment jadevectordb-$version -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null)
        
        if [ "$ready_replicas" = "$desired_replicas" ] && [ -n "$ready_replicas" ] && [ "$ready_replicas" -gt 0 ]; then
            print_status "All $ready_replicas replicas are ready in $version environment"
            return 0
        fi
        
        echo -n "."
        sleep 10
    done
    
    print_error "Timeout waiting for $version environment to be ready"
    return 1
}

# Function to run health checks
run_health_checks() {
    local version=$1
    
    print_status "Running health checks for $version environment..."
    
    # Get the service IP for the environment
    service_ip=$(kubectl get svc jadevectordb-${version}-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    
    if [ -z "$service_ip" ]; then
        print_error "Could not get service IP for $version environment"
        return 1
    fi
    
    print_status "Testing health endpoint at http://$service_ip:8080/health"
    
    # Try the health endpoint
    for i in {1..5}; do
        if curl -f -s --max-time 10 http://$service_ip:8080/health > /dev/null; then
            print_status "Health check passed for $version environment"
            return 0
        fi
        echo -n "."
        sleep 5
    done
    
    print_error "Health check failed for $version environment"
    return 1
}

# Function to switch traffic
switch_traffic() {
    local target_env=$1
    
    print_status "Switching traffic to $target_env environment..."
    
    # Update the main service to point to the target environment
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: jadevectordb-service
  namespace: $NAMESPACE
spec:
  selector:
    app: jadevectordb
    version: $target_env
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080
    - protocol: TCP
      port: 8081
      targetPort: 8081
  type: LoadBalancer
EOF
    
    print_status "Traffic switched to $target_env environment"
    
    # Get the external IP of the service
    sleep 10  # Allow time for service update
    service_external_ip=$(kubectl get svc jadevectordb-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    if [ -z "$service_external_ip" ]; then
        service_external_ip=$(kubectl get svc jadevectordb-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
        print_status "Service ClusterIP: $service_external_ip"
    else
        print_status "Service External IP: $service_external_ip"
    fi
}

# Function to deploy both environments (blue and green)
deploy_environments() {
    print_status "Deploying JadeVectorDB blue-green environment..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace $NAMESPACE >/dev/null 2>&1; then
        print_status "Creating namespace: $NAMESPACE"
        kubectl create namespace $NAMESPACE
    fi
    
    # Apply blue-green deployment configuration
    print_status "Applying blue-green deployment configuration..."
    kubectl apply -f ./k8s/blue-green-deployment.yaml -n $NAMESPACE
    
    # Update blue environment to use old version
    print_status "Updating blue environment to version $BLUE_VERSION..."
    kubectl patch deployment jadevectordb-blue -n $NAMESPACE -p "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"jadevectordb\",\"image\":\"jadevectordb/jadevectordb:$BLUE_VERSION\"}]}}}}"
    
    # Update green environment to use new version
    print_status "Updating green environment to version $GREEN_VERSION..."
    kubectl patch deployment jadevectordb-green -n $NAMESPACE -p "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"jadevectordb\",\"image\":\"jadevectordb/jadevectordb:$GREEN_VERSION\"}]}}}}"
    
    # Wait for blue environment to be ready (this is our current prod)
    if check_environment_ready "blue"; then
        print_status "Blue environment is ready"
    else
        print_error "Blue environment failed to become ready"
        exit 1
    fi
    
    # Wait for green environment to be ready (this is our new version)
    if check_environment_ready "green"; then
        print_status "Green environment is ready"
    else
        print_error "Green environment failed to become ready"
        exit 1
    fi
    
    # Run health checks on both environments
    if run_health_checks "blue"; then
        print_status "Blue environment health check passed"
    else
        print_error "Blue environment health check failed"
    fi
    
    if run_health_checks "green"; then
        print_status "Green environment health check passed"
    else
        print_warning "Green environment health check failed, but continuing with deployment"
    fi
    
    print_status "Both environments are deployed and ready"
    
    # Initially route traffic to blue (current stable)
    switch_traffic "blue"
}

# Function to promote green to production
promote_green() {
    print_status "Promoting green environment to production..."
    
    # Check if green environment is ready
    if ! check_environment_ready "green" 300; then
        print_error "Green environment is not ready for promotion"
        exit 1
    fi
    
    # Run health checks on green environment
    if ! run_health_checks "green"; then
        print_error "Green environment health check failed, aborting promotion"
        exit 1
    fi
    
    # Wait before switching traffic
    print_status "Waiting $ROLLOUT_DELAY seconds before switching traffic..."
    sleep $ROLLOUT_DELAY
    
    # Switch traffic to green
    switch_traffic "green"
    
    # Start monitoring for rollback
    print_status "Starting rollback monitor for green environment..."
    ./rollback-monitor.sh $NAMESPACE 300 &
    ROLLBACK_PID=$!
    
    print_status "Traffic successfully switched to green environment"
    print_status "Rollback monitor started with PID: $ROLLBACK_PID"
    print_status "If issues are detected, the system will automatically roll back to blue environment"
}

# Function to rollback to blue
rollback_to_blue() {
    print_status "Rolling back to blue environment..."
    
    # Check if blue environment is ready
    if ! check_environment_ready "blue" 300; then
        print_error "Blue environment is not ready for rollback"
        exit 1
    fi
    
    # Run health checks on blue environment
    if ! run_health_checks "blue"; then
        print_error "Blue environment health check failed, cannot rollback"
        exit 1
    fi
    
    # Switch traffic back to blue
    switch_traffic "blue"
    
    print_status "Successfully rolled back to blue environment"
}

# Function to show deployment info
show_info() {
    print_status "JadeVectorDB Blue-Green Deployment Information"
    echo "================================================"
    
    current_active=$(get_current_active_env)
    print_status "Current active environment: $current_active"
    
    blue_status=$(kubectl get deployment jadevectordb-blue -n $NAMESPACE -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null || echo "Not found")
    green_status=$(kubectl get deployment jadevectordb-green -n $NAMESPACE -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null || echo "Not found")
    
    print_status "Blue environment status: $blue_status (ready/total)"
    print_status "Green environment status: $green_status (ready/total)"
    
    service_ip=$(kubectl get svc jadevectordb-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    if [ -z "$service_ip" ]; then
        service_ip=$(kubectl get svc jadevectordb-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    fi
    print_status "Service IP: $service_ip"
    
    blue_service_ip=$(kubectl get svc jadevectordb-blue-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    green_service_ip=$(kubectl get svc jadevectordb-green-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    
    print_status "Blue service IP: $blue_service_ip"
    print_status "Green service IP: $green_service_ip"
}

# Main action switch
case $ACTION in
    "deploy")
        print_status "Starting blue-green deployment..."
        deploy_environments
        ;;
    "promote")
        print_status "Promoting green environment..."
        promote_green
        ;;
    "rollback")
        print_status "Initiating rollback..."
        rollback_to_blue
        ;;
    "info")
        show_info
        ;;
    *)
        print_error "Invalid action: $ACTION"
        echo "Usage: $0 [blue-version] [green-version] [delay] [action]"
        echo "Actions: deploy, promote, rollback, info"
        echo "Example: $0 1.0.0 1.1.0 60 promote"
        exit 1
        ;;
esac

print_status "Blue-green deployment operation completed successfully"