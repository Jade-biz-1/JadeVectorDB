#!/bin/bash

# Chaos Engineering Test Suite for JadeVectorDB
# This script runs comprehensive chaos engineering tests

set -e  # Exit immediately if a command exits with a non-zero status

NAMESPACE=${1:-"jadevectordb-system"}
TEST_SUITE=${2:-"all"}  # Options: all, network, node, resource, automated
DURATION=${3:-"300"}  # Duration for each test in seconds
REPORT_FILE=${4:-"/tmp/chaos-test-report-$(date +%Y%m%d-%H%M%S).txt"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "[INFO] $(date): $1" >> $REPORT_FILE
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[WARN] $(date): $1" >> $REPORT_FILE
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $(date): $1" >> $REPORT_FILE
}

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
    echo "[DEBUG] $(date): $1" >> $REPORT_FILE
}

print_status "Starting Chaos Engineering Test Suite"
print_status "Namespace: $NAMESPACE"
print_status "Test Suite: $TEST_SUITE"
print_status "Duration: ${DURATION}s"
print_status "Report File: $REPORT_FILE"

# Initialize report file
echo "Chaos Engineering Test Report" > $REPORT_FILE
echo "=============================" >> $REPORT_FILE
echo "Timestamp: $(date)" >> $REPORT_FILE
echo "Namespace: $NAMESPACE" >> $REPORT_FILE
echo "Test Suite: $TEST_SUITE" >> $REPORT_FILE
echo "" >> $REPORT_FILE

# Check prerequisites
if ! command -v kubectl >/dev/null 2>&1; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    print_error "jq is not installed or not in PATH (required for JSON processing)"
    exit 1
fi

# Function to run network partition tests
run_network_tests() {
    print_status "Starting network partition tests..."
    
    # Test 1: Random network partition
    print_status "Test 1: Random network partition simulation"
    ./network/simulate-network-partition.sh $NAMESPACE random $((DURATION/3)) 2>&1 | tee -a $REPORT_FILE
    sleep 10
    
    # Test 2: All pods network partition
    print_status "Test 2: All pods network partition simulation"
    ./network/simulate-network-partition.sh $NAMESPACE all $((DURATION/3)) 2>&1 | tee -a $REPORT_FILE
    sleep 10
    
    # Test 3: Specific pod network partition
    local pod_list=$(kubectl get pods -n $NAMESPACE -l app=jadevectordb -o jsonpath='{.items[*].metadata.name}' 2>/dev/null | cut -d' ' -f1)
    if [ -n "$pod_list" ]; then
        print_status "Test 3: Specific pod network partition simulation for pod: $pod_list"
        ./network/simulate-network-partition.sh $NAMESPACE specific $((DURATION/3)) $pod_list 2>&1 | tee -a $REPORT_FILE
        sleep 10
    else
        print_warning "No pods found for specific network partition test"
    fi
    
    print_status "Network partition tests completed"
}

# Function to run node failure tests
run_node_tests() {
    print_status "Starting node failure tests..."
    
    # Test 1: Pod deletion
    print_status "Test 1: Pod deletion simulation"
    ./node/simulate-node-failure.sh $NAMESPACE pod-delete $((DURATION/3)) 2>&1 | tee -a $REPORT_FILE
    sleep 30  # Wait for pod to be recreated
    
    # Test 2: Pod termination
    print_status "Test 2: Pod termination simulation"
    ./node/simulate-node-failure.sh $NAMESPACE pod-kill $((DURATION/3)) 2>&1 | tee -a $REPORT_FILE
    sleep 30  # Wait for pod to be recreated
    
    # Test 3: Crashloop simulation
    print_status "Test 3: Crashloop simulation"
    ./node/simulate-node-failure.sh $NAMESPACE crashloop $((DURATION/3)) 2>&1 | tee -a $REPORT_FILE
    sleep 30  # Wait for pods to stabilize
    
    print_status "Node failure tests completed"
}

# Function to run resource exhaustion tests
run_resource_tests() {
    print_status "Starting resource exhaustion tests..."
    
    # Test 1: Memory exhaustion (low intensity)
    print_status "Test 1: Low intensity memory exhaustion"
    ./resource/simulate-resource-exhaustion.sh $NAMESPACE memory $((DURATION/4)) low 2>&1 | tee -a $REPORT_FILE
    sleep 20
    
    # Test 2: CPU exhaustion (medium intensity)
    print_status "Test 2: Medium intensity CPU exhaustion"
    ./resource/simulate-resource-exhaustion.sh $NAMESPACE cpu $((DURATION/4)) medium 2>&1 | tee -a $REPORT_FILE
    sleep 20
    
    # Test 3: Disk exhaustion (medium intensity)
    print_status "Test 3: Medium intensity disk exhaustion"
    ./resource/simulate-resource-exhaustion.sh $NAMESPACE disk $((DURATION/4)) medium 2>&1 | tee -a $REPORT_FILE
    sleep 20
    
    # Test 4: Network exhaustion (high intensity)
    print_status "Test 4: High intensity network exhaustion"
    ./resource/simulate-resource-exhaustion.sh $NAMESPACE network $((DURATION/4)) high 2>&1 | tee -a $REPORT_FILE
    sleep 20
    
    print_status "Resource exhaustion tests completed"
}

# Function to run automated chaos experiment
run_automated_tests() {
    print_status "Starting automated chaos experiment tests..."
    
    # Create a test experiment file
    cat > ./experiments/test-experiment.json << EOF
{
  "name": "Test Chaos Experiment Suite",
  "description": "A test chaos experiment suite to validate the chaos engineering framework",
  "experiments": [
    {
      "name": "Quick Memory Exhaustion",
      "type": "resource_exhaustion",
      "params": {
        "resource_type": "memory",
        "intensity": "low"
      },
      "duration": 30,
      "interval": 10
    },
    {
      "name": "Quick Network Partition",
      "type": "network_partition",
      "params": {
        "partition_type": "random"
      },
      "duration": 20,
      "interval": 10
    },
    {
      "name": "Quick Node Kill",
      "type": "node_failure",
      "params": {
        "failure_type": "pod-kill"
      },
      "duration": 30,
      "interval": 10
    }
  ]
}
EOF
    
    # Run the automated experiment
    ./automation/execute-chaos-experiment.sh $NAMESPACE ./experiments/test-experiment.json $DURATION 2>&1 | tee -a $REPORT_FILE
    
    print_status "Automated chaos experiment tests completed"
}

# Execute tests based on suite selection
case $TEST_SUITE in
    "all")
        print_status "Running all chaos engineering tests..."
        run_network_tests
        run_node_tests
        run_resource_tests
        run_automated_tests
        ;;
    "network")
        print_status "Running network partition tests only..."
        run_network_tests
        ;;
    "node")
        print_status "Running node failure tests only..."
        run_node_tests
        ;;
    "resource")
        print_status "Running resource exhaustion tests only..."
        run_resource_tests
        ;;
    "automated")
        print_status "Running automated chaos experiment tests only..."
        run_automated_tests
        ;;
    *)
        print_error "Invalid test suite: $TEST_SUITE. Use all, network, node, resource, or automated"
        exit 1
        ;;
esac

# Final summary to report
echo "" >> $REPORT_FILE
echo "Test Suite Summary" >> $REPORT_FILE
echo "==================" >> $REPORT_FILE
echo "Completed: $(date)" >> $REPORT_FILE
echo "Test Suite: $TEST_SUITE" >> $REPORT_FILE
echo "Namespace: $NAMESPACE" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "Report saved to: $REPORT_FILE" >> $REPORT_FILE

print_status "Chaos Engineering Test Suite completed"
print_status "Report saved to: $REPORT_FILE"