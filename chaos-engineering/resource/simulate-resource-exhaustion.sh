#!/bin/bash

# Resource Exhaustion Simulation Script for JadeVectorDB
# This script simulates resource exhaustion (CPU, memory, disk, network) in the system

set -e  # Exit immediately if a command exits with a non-zero status

# Default parameters
NAMESPACE=${1:-"jadevectordb-system"}
RESOURCE_TYPE=${2:-"memory"}  # Options: memory, cpu, disk, network
DURATION=${3:-"120"}  # Duration in seconds
INTENSITY=${4:-"medium"}  # Options: low, medium, high
TARGET_PODS=${5:-""}  # Comma-separated list of specific pods to target

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

print_status "Starting resource exhaustion simulation in namespace: $NAMESPACE"
print_status "Resource type: $RESOURCE_TYPE, Duration: ${DURATION}s, Intensity: $INTENSITY"

# Check prerequisites
if ! command -v kubectl >/dev/null 2>&1; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Function to get all JadeVectorDB pods in the namespace
get_jadevectordb_pods() {
    kubectl get pods -n $NAMESPACE -l app=jadevectordb -o jsonpath='{.items[*].metadata.name}' 2>/dev/null
}

# Function to simulate memory exhaustion
simulate_memory_exhaustion() {
    local pod_name=$1
    local duration=$2
    local intensity=$3
    
    print_status "Simulating memory exhaustion on pod: $pod_name for ${duration}s (intensity: $intensity)"
    
    # Determine memory allocation based on intensity
    case $intensity in
        "low")
            memory_mb=100
            ;;
        "medium")
            memory_mb=500
            ;;
        "high")
            memory_mb=1024
            ;;
        *)
            memory_mb=500
            ;;
    esac
    
    # Execute memory exhaustion in the pod
    kubectl exec -n $NAMESPACE $pod_name -c jadevectordb -- sh -c "
        # Create a background process to consume memory
        (
            # Allocate memory using dd and temporary files
            dd if=/dev/zero of=/tmp/chaos_mem_exhaustion bs=1M count=$memory_mb &
            sleep $duration
            rm -f /tmp/chaos_mem_exhaustion 2>/dev/null || true
        ) &
        echo \$! > /tmp/chaos_mem_pid
    " 2>/dev/null || print_warning "Could not execute memory exhaustion in $pod_name"
    
    print_status "Memory exhaustion initiated on pod: $pod_name (allocated $memory_mb MB)"
}

# Function to simulate CPU exhaustion
simulate_cpu_exhaustion() {
    local pod_name=$1
    local duration=$2
    local intensity=$3
    
    print_status "Simulating CPU exhaustion on pod: $pod_name for ${duration}s (intensity: $intensity)"
    
    # Determine CPU load based on intensity
    case $intensity in
        "low")
            cpu_processes=2
            ;;
        "medium")
            cpu_processes=4
            ;;
        "high")
            cpu_processes=8
            ;;
        *)
            cpu_processes=4
            ;;
    esac
    
    # Execute CPU exhaustion in the pod
    kubectl exec -n $NAMESPACE $pod_name -c jadevectordb -- sh -c "
        # Create background processes to consume CPU
        for i in \$(seq 1 $cpu_processes); do
            (
                # Run infinite loop to consume CPU
                while [ \$((SECONDS)) -lt $duration ]; do
                    :  # No-op command that consumes CPU
                done
            ) &
        done
        echo \$! > /tmp/chaos_cpu_pid
    " 2>/dev/null || print_warning "Could not execute CPU exhaustion in $pod_name"
    
    print_status "CPU exhaustion initiated on pod: $pod_name (using $cpu_processes processes)"
}

# Function to simulate disk exhaustion
simulate_disk_exhaustion() {
    local pod_name=$1
    local duration=$2
    local intensity=$3
    
    print_status "Simulating disk exhaustion on pod: $pod_name for ${duration}s (intensity: $intensity)"
    
    # Determine disk allocation based on intensity
    case $intensity in
        "low")
            disk_mb=100
            ;;
        "medium")
            disk_mb=500
            ;;
        "high")
            disk_mb=1024
            ;;
        *)
            disk_mb=500
            ;;
    esac
    
    # Execute disk exhaustion in the pod
    kubectl exec -n $NAMESPACE $pod_name -c jadevectordb -- sh -c "
        # Create a large file to consume disk space
        (
            dd if=/dev/zero of=/tmp/chaos_disk_exhaustion bs=1M count=$disk_mb &
            sleep $duration
            rm -f /tmp/chaos_disk_exhaustion 2>/dev/null || true
        ) &
    " 2>/dev/null || print_warning "Could not execute disk exhaustion in $pod_name"
    
    print_status "Disk exhaustion initiated on pod: $pod_name (allocated $disk_mb MB)"
}

# Function to simulate network exhaustion
simulate_network_exhaustion() {
    local pod_name=$1
    local duration=$2
    local intensity=$3
    
    print_status "Simulating network exhaustion on pod: $pod_name for ${duration}s (intensity: $intensity)"
    
    # Determine network load based on intensity
    case $intensity in
        "low")
            network_concurrent=10
            network_duration=60
            ;;
        "medium")
            network_concurrent=50
            network_duration=120
            ;;
        "high")
            network_concurrent=100
            network_duration=180
            ;;
        *)
            network_concurrent=50
            network_duration=120
            ;;
    esac
    
    # Execute network exhaustion in the pod
    kubectl exec -n $NAMESPACE $pod_name -c jadevectordb -- sh -c "
        # Create background network load
        (
            for i in \$(seq 1 $network_concurrent); do
                # Generate network traffic by pinging a common IP address
                timeout $network_duration ping -c $((network_duration/2)) 8.8.8.8 > /dev/null 2>&1 &
            done
            sleep $duration
        ) &
    " 2>/dev/null || print_warning "Could not execute network exhaustion in $pod_name"
    
    print_status "Network exhaustion initiated on pod: $pod_name (using $network_concurrent concurrent connections)"
}

# Get all pods in the namespace
all_pods=($(get_jadevectordb_pods))

if [ ${#all_pods[@]} -eq 0 ]; then
    print_error "No JadeVectorDB pods found in namespace: $NAMESPACE"
    exit 1
fi

print_status "Found ${#all_pods[@]} JadeVectorDB pods"

# Execute resource exhaustion based on type
if [ -z "$TARGET_PODS" ]; then
    # Select random pods based on intensity (higher intensity = more pods affected)
    total_pods=${#all_pods[@]}
    case $INTENSITY in
        "low")
            target_count=1
            ;;
        "medium")
            target_count=$((total_pods * 3 / 10))
            [ $target_count -lt 1 ] && target_count=1
            ;;
        "high")
            target_count=$((total_pods * 6 / 10))
            [ $target_count -lt 1 ] && target_count=1
            ;;
        *)
            target_count=1
            ;;
    esac
    
    print_status "Targeting $target_count out of $total_pods pods for resource exhaustion"
    
    # Shuffle and select random pods
    shuffled_pods=()
    for pod in "${all_pods[@]}"; do
        shuffled_pods+=("$pod")
    done
    
    selected_pods=("${shuffled_pods[@]:0:$target_count}")
else
    IFS=',' read -ra selected_pods <<< "$TARGET_PODS"
    # Verify selected pods exist
    valid_pods=()
    for pod in "${selected_pods[@]}"; do
        if [[ " ${all_pods[@]} " =~ " $pod " ]]; then
            valid_pods+=("$pod")
        else
            print_warning "Pod $pod not found in namespace, skipping"
        fi
    done
    selected_pods=("${valid_pods[@]}")
fi

# Apply exhaustion to selected pods
for pod in "${selected_pods[@]}"; do
    case $RESOURCE_TYPE in
        "memory")
            simulate_memory_exhaustion $pod $DURATION $INTENSITY &
            ;;
        "cpu")
            simulate_cpu_exhaustion $pod $DURATION $INTENSITY &
            ;;
        "disk")
            simulate_disk_exhaustion $pod $DURATION $INTENSITY &
            ;;
        "network")
            simulate_network_exhaustion $pod $DURATION $INTENSITY &
            ;;
        *)
            print_error "Invalid resource type: $RESOURCE_TYPE. Use memory, cpu, disk, or network"
            exit 1
            ;;
    esac
done

# Wait for all background processes to complete
wait

print_status "Resource exhaustion simulation completed"