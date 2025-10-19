#!/bin/bash

# Network Partition Simulation Script for JadeVectorDB
# This script simulates network partitions that might occur in a distributed system

set -e  # Exit immediately if a command exits with a non-zero status

# Default parameters
NAMESPACE=${1:-"jadevectordb-system"}
PARTITION_TYPE=${2:-"random"}  # Options: random, specific, all
DURATION=${3:-"60"}  # Duration in seconds
NODES=${4:-""}  # Comma-separated list of specific nodes to partition

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Starting network partition simulation in namespace: $NAMESPACE"
print_status "Partition type: $PARTITION_TYPE, Duration: ${DURATION}s"

# Check prerequisites
if ! command -v kubectl >/dev/null 2>&1; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

if ! command -v iptables >/dev/null 2>&1; then
    print_warning "iptables is not available, will use kubectl to simulate network issues"
fi

# Function to get all JadeVectorDB pods in the namespace
get_jadevectordb_pods() {
    kubectl get pods -n $NAMESPACE -l app=jadevectordb -o jsonpath='{.items[*].metadata.name}' 2>/dev/null
}

# Function to block network traffic to/from a pod using iptables (if available) or simulate via kubectl
simulate_partition() {
    local pod_name=$1
    local partition_duration=$2
    
    print_status "Simulating network partition for pod: $pod_name"
    
    # Method 1: Use iptables if available (works if running inside the pod)
    if command -v iptables >/dev/null 2>&1; then
        print_status "Using iptables to block traffic for $pod_name"
        kubectl exec -n $NAMESPACE $pod_name -- iptables -A OUTPUT -j DROP
        kubectl exec -n $NAMESPACE $pod_name -- iptables -A INPUT -j DROP
        sleep $partition_duration
        kubectl exec -n $NAMESPACE $pod_name -- iptables -D OUTPUT -j DROP || true
        kubectl exec -n $NAMESPACE $pod_name -- iptables -D INPUT -j DROP || true
    else
        # Method 2: Use kubectl debug to temporarily stop the application process
        print_status "Using kubectl to simulate network partition for $pod_name"
        
        # Get the container name
        local container_name=$(kubectl get pod $pod_name -n $NAMESPACE -o jsonpath='{.spec.containers[0].name}' 2>/dev/null || echo "jadevectordb")
        
        # Stop the main process inside the container to simulate network partition
        kubectl exec -n $NAMESPACE $pod_name -c $container_name -- kill -STOP 1 2>/dev/null || true
        
        sleep $partition_duration
        
        # Resume the process
        kubectl exec -n $NAMESPACE $pod_name -c $container_name -- kill -CONT 1 2>/dev/null || true
    fi
    
    print_status "Network partition simulation completed for pod: $pod_name"
}

# Get all pods in the namespace
all_pods=($(get_jadevectordb_pods))

if [ ${#all_pods[@]} -eq 0 ]; then
    print_error "No JadeVectorDB pods found in namespace: $NAMESPACE"
    exit 1
fi

print_status "Found ${#all_pods[@]} JadeVectorDB pods"

# Execute network partition simulation based on type
case $PARTITION_TYPE in
    "random")
        # Randomly select a small subset of pods (20% of total)
        total_pods=${#all_pods[@]}
        partition_count=$((total_pods * 2 / 10))
        [ $partition_count -lt 1 ] && partition_count=1  # At least 1
        
        print_status "Partitioning $partition_count random pods out of $total_pods"
        
        # Shuffle pods and select random subset
        shuffled_pods=()
        for pod in "${all_pods[@]}"; do
            shuffled_pods+=("$pod")
        done
        
        # Simple shuffle by taking pods at different intervals
        selected_pods=()
        for ((i=0; i<partition_count && i<${#shuffled_pods[@]}; i++)); do
            selected_pods+=("${shuffled_pods[i]}")
        done
        
        for pod in "${selected_pods[@]}"; do
            simulate_partition $pod $DURATION &
        done
        ;;
    
    "specific")
        if [ -z "$NODES" ]; then
            print_error "NODES parameter required for specific partition type"
            exit 1
        fi
        
        IFS=',' read -ra specific_pods <<< "$NODES"
        for pod in "${specific_pods[@]}"; do
            if [[ " ${all_pods[@]} " =~ " $pod " ]]; then
                simulate_partition $pod $DURATION &
            else
                print_warning "Pod $pod not found in namespace, skipping"
            fi
        done
        ;;
    
    "all")
        print_status "Partitioning all ${#all_pods[@]} pods"
        for pod in "${all_pods[@]}"; do
            simulate_partition $pod $DURATION &
        done
        ;;
    
    *)
        print_error "Invalid partition type: $PARTITION_TYPE. Use random, specific, or all"
        exit 1
        ;;
esac

# Wait for all background processes to complete
wait

print_status "Network partition simulation completed"