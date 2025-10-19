#!/bin/bash

# Automated Chaos Experiment Execution System for JadeVectorDB
# This script orchestrates automated chaos experiments based on experiment definitions

set -e  # Exit immediately if a command exits with a non-zero status

NAMESPACE=${1:-"jadevectordb-system"}
EXPERIMENT_FILE=${2:-"default-experiment.json"}
DURATION=${3:-"300"}  # Total execution time in seconds
GRACE_PERIOD=${4:-"30"}  # Time to allow system to stabilize before/after experiment

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

print_status "Starting automated chaos experiment execution"
print_status "Namespace: $NAMESPACE"
print_status "Experiment file: $EXPERIMENT_FILE"
print_status "Duration: ${DURATION}s"
print_status "Grace period: ${GRACE_PERIOD}s"

# Check prerequisites
if ! command -v kubectl >/dev/null 2>&1; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    print_error "jq is not installed or not in PATH (required for JSON processing)"
    exit 1
fi

# Function to run pre-experiment checks
run_pre_experiment_checks() {
    print_status "Running pre-experiment checks..."
    
    # Check that JadeVectorDB pods are running and healthy
    local pod_count=$(kubectl get pods -n $NAMESPACE -l app=jadevectordb --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}' | wc -w)
    
    if [ $pod_count -eq 0 ]; then
        print_error "No running JadeVectorDB pods found in namespace: $NAMESPACE"
        exit 1
    fi
    
    print_status "$pod_count JadeVectorDB pods are running"
    
    # Check that services are available
    local service_count=$(kubectl get svc -n $NAMESPACE -l app=jadevectordb -o name | wc -l)
    print_status "$service_count JadeVectorDB services found"
    
    print_status "Pre-experiment checks completed successfully"
}

# Function to run post-experiment checks
run_post_experiment_checks() {
    print_status "Running post-experiment checks..."
    
    # Wait a bit for system to recover
    sleep $GRACE_PERIOD
    
    # Check that JadeVectorDB pods are running and healthy
    local pod_count=$(kubectl get pods -n $NAMESPACE -l app=jadevectordb --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}' | wc -w)
    local expected_pod_count=$(kubectl get deployment -n $NAMESPACE jadevectordb -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "3")
    
    print_status "$pod_count JadeVectorDB pods are running (expected: $expected_pod_count)"
    
    # Check that services are available
    local service_count=$(kubectl get svc -n $NAMESPACE -l app=jadevectordb -o name | wc -l)
    print_status "$service_count JadeVectorDB services found"
    
    print_status "Post-experiment checks completed"
}

# Function to execute a single chaos experiment
execute_chaos_experiment() {
    local exp_type=$1
    local exp_params=$2
    local exp_duration=$3
    local exp_interval=$4
    
    print_status "Executing chaos experiment: $exp_type with params: $exp_params for ${exp_duration}s"
    
    # Based on experiment type, call the appropriate chaos script
    case $exp_type in
        "network_partition")
            local partition_type=$(echo $exp_params | jq -r '.partition_type // "random"')
            local targets=$(echo $exp_params | jq -r '.targets // ""')
            
            ./../network/simulate-network-partition.sh $NAMESPACE $partition_type $exp_duration $targets &
            local exp_pid=$!
            sleep $exp_duration
            ;;
        
        "node_failure")
            local failure_type=$(echo $exp_params | jq -r '.failure_type // "pod-delete"')
            local targets=$(echo $exp_params | jq -r '.targets // ""')
            
            ./../node/simulate-node-failure.sh $NAMESPACE $failure_type $exp_duration $targets &
            local exp_pid=$!
            sleep $exp_duration
            ;;
        
        "resource_exhaustion")
            local resource_type=$(echo $exp_params | jq -r '.resource_type // "memory"')
            local intensity=$(echo $exp_params | jq -r '.intensity // "medium"')
            local targets=$(echo $exp_params | jq -r '.targets // ""')
            
            ./../resource/simulate-resource-exhaustion.sh $NAMESPACE $resource_type $exp_duration $intensity $targets &
            local exp_pid=$!
            sleep $exp_duration
            ;;
        
        *)
            print_warning "Unknown experiment type: $exp_type, skipping"
            return 0
            ;;
    esac
    
    print_status "Chaos experiment $exp_type completed"
    sleep $exp_interval
}

# Default experiment if no file specified
if [ "$EXPERIMENT_FILE" = "default-experiment.json" ]; then
    print_status "Using default experiment configuration"
    
    # Create a default experiment file
    cat > ./experiments/default-experiment.json << EOF
{
  "name": "Default Chaos Experiment Suite",
  "description": "A basic chaos experiment suite to test system resilience",
  "experiments": [
    {
      "name": "Memory Exhaustion Test",
      "type": "resource_exhaustion",
      "params": {
        "resource_type": "memory",
        "intensity": "medium"
      },
      "duration": 60,
      "interval": 30
    },
    {
      "name": "Network Partition Test",
      "type": "network_partition",
      "params": {
        "partition_type": "random"
      },
      "duration": 45,
      "interval": 30
    },
    {
      "name": "Node Failure Test",
      "type": "node_failure",
      "params": {
        "failure_type": "pod-kill"
      },
      "duration": 60,
      "interval": 30
    }
  ]
}
EOF
    
    EXPERIMENT_FILE="./experiments/default-experiment.json"
fi

# Validate experiment file exists and is valid JSON
if [ ! -f "$EXPERIMENT_FILE" ]; then
    print_error "Experiment file does not exist: $EXPERIMENT_FILE"
    exit 1
fi

if ! jq empty "$EXPERIMENT_FILE" 2>/dev/null; then
    print_error "Experiment file is not valid JSON: $EXPERIMENT_FILE"
    exit 1
fi

# Run pre-experiment checks
run_pre_experiment_checks

# Wait for grace period
print_status "Waiting $GRACE_PERIOD seconds for system stability before starting experiments..."
sleep $GRACE_PERIOD

# Execute experiments in sequence
experiment_count=$(jq '.experiments | length' "$EXPERIMENT_FILE")
print_status "Starting execution of $experiment_count experiments..."

for i in $(seq 0 $((experiment_count - 1))); do
    print_status "Executing experiment $((i + 1))/$experiment_count"
    
    exp_name=$(jq -r ".experiments[$i].name" "$EXPERIMENT_FILE")
    exp_type=$(jq -r ".experiments[$i].type" "$EXPERIMENT_FILE")
    exp_params=$(jq -r ".experiments[$i].params" "$EXPERIMENT_FILE")
    exp_duration=$(jq -r ".experiments[$i].duration // 60" "$EXPERIMENT_FILE")
    exp_interval=$(jq -r ".experiments[$i].interval // 30" "$EXPERIMENT_FILE")
    
    print_debug "Experiment details - Name: $exp_name, Type: $exp_type, Duration: $exp_duration, Interval: $exp_interval"
    
    # Record start time for this experiment
    local exp_start_time=$(date +%s)
    
    execute_chaos_experiment "$exp_type" "$exp_params" "$exp_duration" "$exp_interval"
    
    # Calculate elapsed time and ensure we don't go over total duration
    local elapsed_time=$(( $(date +%s) - exp_start_time ))
    print_debug "Experiment $exp_name took $elapsed_time seconds"
done

# Run post-experiment checks
run_post_experiment_checks

print_status "All chaos experiments completed successfully"
print_status "Total execution time: $((DURATION + 2*GRACE_PERIOD)) seconds"