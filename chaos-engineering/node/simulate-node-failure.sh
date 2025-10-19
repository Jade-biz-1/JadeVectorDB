#!/bin/bash

# Node Failure Injection Script for JadeVectorDB
# This script simulates node failures in the distributed system

set -e  # Exit immediately if a command exits with a non-zero status

# Default parameters
NAMESPACE=${1:-"jadevectordb-system"}
FAILURE_TYPE=${2:-"pod-delete"}  # Options: pod-delete, pod-kill, node-drain, crashloop
DURATION=${3:-"120"}  # Duration in seconds for some failure types
NODES=${4:-""}  # Comma-separated list of specific nodes to target

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

print_status "Starting node failure injection in namespace: $NAMESPACE"
print_status "Failure type: $FAILURE_TYPE, Duration: ${DURATION}s"

# Check prerequisites
if ! command -v kubectl >/dev/null 2>&1; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Function to get all JadeVectorDB pods in the namespace
get_jadevectordb_pods() {
    kubectl get pods -n $NAMESPACE -l app=jadevectordb -o jsonpath='{.items[*].metadata.name}' 2>/dev/null
}

# Function to get nodes where JadeVectorDB pods are running
get_jadevectordb_nodes() {
    kubectl get pods -n $NAMESPACE -l app=jadevectordb -o jsonpath='{range .items[*]}{.spec.nodeName}{" "}{end}' 2>/dev/null | tr ' ' '\n' | sort -u
}

# Function to simulate pod deletion
simulate_pod_deletion() {
    local pod_name=$1
    
    print_status "Deleting pod: $pod_name"
    kubectl delete pod $pod_name -n $NAMESPACE --grace-period=0 --force
    print_status "Pod $pod_name deletion initiated"
}

# Function to simulate pod termination
simulate_pod_kill() {
    local pod_name=$1
    local signal=${2:-"TERM"}
    
    print_status "Terminating pod: $pod_name with signal $signal"
    # First try graceful termination
    kubectl exec -n $NAMESPACE $pod_name -c jadevectordb -- kill -$signal 1 2>/dev/null || true
    sleep 5
    # If still running, force termination
    kubectl exec -n $NAMESPACE $pod_name -c jadevectordb -- kill -KILL 1 2>/dev/null || true
    print_status "Pod $pod_name termination initiated"
}

# Function to simulate crashlooping a pod
simulate_crashloop() {
    local pod_name=$1
    local duration=$2
    
    print_status "Making pod $pod_name crashloop for ${duration}s"
    
    # Get the original command/args to restore later
    local container_name=$(kubectl get pod $pod_name -n $NAMESPACE -o jsonpath='{.spec.containers[0].name}' 2>/dev/null || echo "jadevectordb")
    
    # Temporarily patch the deployment to make the container crash
    # We'll do this by temporarily patching the deployment that created this pod
    local deployment_name=$(kubectl get pod $pod_name -n $NAMESPACE -o jsonpath='{.metadata.ownerReferences[0].name}' 2>/dev/null || echo "")
    
    if [ -n "$deployment_name" ]; then
        print_status "Patching deployment $deployment_name to simulate crashloop"
        # Backup current image
        local current_image=$(kubectl get deployment $deployment_name -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
        
        # Change to an invalid command that will always crash
        kubectl patch deployment $deployment_name -n $NAMESPACE --type='json' -p='[
            {"op": "replace", "path": "/spec/template/spec/containers/0/command", "value": ["/bin/sh", "-c", "exit 1"]}
        ]' 2>/dev/null || print_warning "Could not patch deployment, trying direct pod modification"
        
        sleep $duration
        
        # Restore the original command
        kubectl patch deployment $deployment_name -n $NAMESPACE --type='json' -p='[
            {"op": "remove", "path": "/spec/template/spec/containers/0/command"}
        ]' 2>/dev/null || print_warning "Could not restore deployment, manual intervention may be needed"
    else
        print_warning "Could not identify parent deployment for $pod_name, skipping crashloop simulation"
    fi
}

# Get all pods in the namespace
all_pods=($(get_jadevectordb_pods))

if [ ${#all_pods[@]} -eq 0 ]; then
    print_error "No JadeVectorDB pods found in namespace: $NAMESPACE"
    exit 1
fi

print_status "Found ${#all_pods[@]} JadeVectorDB pods"

# Execute node failure injection based on type
case $FAILURE_TYPE in
    "pod-delete")
        if [ -z "$NODES" ]; then
            # Select random pods to delete (limit to max 30% to maintain cluster stability)
            total_pods=${#all_pods[@]}
            delete_count=$((total_pods * 3 / 10))
            [ $delete_count -lt 1 ] && delete_count=1  # At least 1
            
            print_status "Deleting $delete_count random pods out of $total_pods"
            
            # Shuffle and select random pods
            shuffled_pods=()
            for pod in "${all_pods[@]}"; do
                shuffled_pods+=("$pod")
            done
            
            for ((i=0; i<delete_count && i<${#shuffled_pods[@]}; i++)); do
                simulate_pod_deletion "${shuffled_pods[i]}"
            done
        else
            IFS=',' read -ra specific_pods <<< "$NODES"
            for pod in "${specific_pods[@]}"; do
                if [[ " ${all_pods[@]} " =~ " $pod " ]]; then
                    simulate_pod_deletion $pod
                else
                    print_warning "Pod $pod not found in namespace, skipping"
                fi
            done
        fi
        ;;
    
    "pod-kill")
        if [ -z "$NODES" ]; then
            # Select random pods to kill with SIGTERM
            total_pods=${#all_pods[@]}
            kill_count=$((total_pods * 3 / 10))
            [ $kill_count -lt 1 ] && kill_count=1  # At least 1
            
            print_status "Terminating $kill_count random pods out of $total_pods"
            
            # Shuffle and select random pods
            shuffled_pods=()
            for pod in "${all_pods[@]}"; do
                shuffled_pods+=("$pod")
            done
            
            for ((i=0; i<kill_count && i<${#shuffled_pods[@]}; i++)); do
                simulate_pod_kill "${shuffled_pods[i]}"
            done
        else
            IFS=',' read -ra specific_pods <<< "$NODES"
            for pod in "${specific_pods[@]}"; do
                if [[ " ${all_pods[@]} " =~ " $pod " ]]; then
                    simulate_pod_kill $pod
                else
                    print_warning "Pod $pod not found in namespace, skipping"
                fi
            done
        fi
        ;;
    
    "crashloop")
        if [ -z "$NODES" ]; then
            # Select random pods to make crashloop
            total_pods=${#all_pods[@]}
            crashloop_count=$((total_pods * 2 / 10))
            [ $crashloop_count -lt 1 ] && crashloop_count=1  # At least 1
            
            print_status "Making $crashloop_count random pods crashloop for ${DURATION}s"
            
            # Shuffle and select random pods
            shuffled_pods=()
            for pod in "${all_pods[@]}"; do
                shuffled_pods+=("$pod")
            done
            
            for pod in "${shuffled_pods[@]:0:$crashloop_count}"; do
                simulate_crashloop $pod $DURATION &
            done
        else
            IFS=',' read -ra specific_pods <<< "$NODES"
            for pod in "${specific_pods[@]}"; do
                if [[ " ${all_pods[@]} " =~ " $pod " ]]; then
                    simulate_crashloop $pod $DURATION &
                else
                    print_warning "Pod $pod not found in namespace, skipping"
                fi
            done
        fi
        # Wait for background processes
        wait
        ;;
    
    "node-drain")
        print_warning "Node drain simulation requires cluster admin privileges"
        jadevectordb_nodes=($(get_jadevectordb_nodes))
        
        if [ ${#jadevectordb_nodes[@]} -eq 0 ]; then
            print_error "No nodes with JadeVectorDB pods found"
            exit 1
        fi
        
        print_status "Found ${#jadevectordb_nodes[@]} nodes with JadeVectorDB pods"
        
        if [ -z "$NODES" ]; then
            # Select a random node to simulate drain
            random_node_index=$((RANDOM % ${#jadevectordb_nodes[@]}))
            target_node=${jadevectordb_nodes[$random_node_index]}
            print_status "Simulating drain of node: $target_node"
            
            # Backup current pod disruption budgets
            kubectl get pdb -n $NAMESPACE -o yaml > /tmp/jadevectordb-pdb-backup.yaml 2>/dev/null || print_warning "Could not backup PDBs"
            
            # Temporarily allow disruptions
            kubectl create -f - <<EOF || print_warning "Could not create pdb to allow disruptions"
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: chaos-pdb
  namespace: $NAMESPACE
spec:
  minAvailable: 0
  selector:
    matchLabels:
      app: jadevectordb
EOF
            
            # Cordon and drain the node (with --delete-emptydir-data to simulate real node failure)
            kubectl drain $target_node --ignore-daemonsets --delete-emptydir-data --timeout=${DURATION}s --grace-period=30 || print_warning "Node drain failed, continuing"
            
            # Restore original PDBs
            kubectl delete pdb chaos-pdb -n $NAMESPACE 2>/dev/null || true
            kubectl apply -f /tmp/jadevectordb-pdb-backup.yaml 2>/dev/null || print_warning "Could not restore PDBs"
            
            # Uncordon the node
            kubectl uncordon $target_node || print_warning "Could not uncordon node $target_node"
        else
            IFS=',' read -ra specific_nodes <<< "$NODES"
            for node in "${specific_nodes[@]}"; do
                print_status "Simulating drain of node: $node"
                
                # Similar process for specific nodes
                kubectl drain $node --ignore-daemonsets --delete-emptydir-data --timeout=${DURATION}s --grace-period=30 || print_warning "Node drain failed for $node, continuing"
                kubectl uncordon $node || print_warning "Could not uncordon node $node"
            done
        fi
        ;;
    
    *)
        print_error "Invalid failure type: $FAILURE_TYPE. Use pod-delete, pod-kill, node-drain, or crashloop"
        exit 1
        ;;
esac

print_status "Node failure injection completed"