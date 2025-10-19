#!/bin/bash

# Canary Deployment Script for Blue-Green Setup
# This script enables gradual traffic shifting between blue and green environments

set -e  # Exit immediately if a command exits with a non-zero status

NAMESPACE=${1:-"jadevectordb-system"}
CANARY_TYPE=${2:-"full-switch"}  # Options: partial, half, mostly-green, full-switch, rollback

echo "Starting canary deployment operation: $CANARY_TYPE in namespace $NAMESPACE..."

# Check if kubectl is available
if ! command -v kubectl >/dev/null 2>&1; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if istioctl is available (for Istio configurations)
ISTIOCTL_AVAILABLE=false
if command -v istioctl >/dev/null 2>&1; then
    ISTIOCTL_AVAILABLE=true
fi

case $CANARY_TYPE in
    "partial")
        echo "Shifting 10% of traffic to green environment..."
        if [ "$ISTIOCTL_AVAILABLE" = true ]; then
            # Update Istio VirtualService to route 10% to green
            kubectl patch virtualservice jadevectordb-canary-routing -n $NAMESPACE -p '{"spec":{"http":[{"match":[{"headers":{"canary":{"exact":"partial"}}}],"route":[{"destination":{"host":"jadevectordb-blue-service","subset":"blue"},"weight":90},{"destination":{"host":"jadevectordb-green-service","subset":"green"},"weight":10}]}]}}'
        else
            echo "Istio not available, please manually update traffic routing"
            exit 1
        fi
        ;;
    "half")
        echo "Shifting 50% of traffic to green environment..."
        if [ "$ISTIOCTL_AVAILABLE" = true ]; then
            # Update Istio VirtualService to route 50% to each
            kubectl patch virtualservice jadevectordb-canary-routing -n $NAMESPACE -p '{"spec":{"http":[{"match":[{"headers":{"canary":{"exact":"half"}}}],"route":[{"destination":{"host":"jadevectordb-blue-service","subset":"blue"},"weight":50},{"destination":{"host":"jadevectordb-green-service","subset":"green"},"weight":50}]}]}}'
        else
            echo "Istio not available, please manually update traffic routing"
            exit 1
        fi
        ;;
    "mostly-green")
        echo "Shifting 90% of traffic to green environment..."
        if [ "$ISTIOCTL_AVAILABLE" = true ]; then
            # Update Istio VirtualService to route 90% to green
            kubectl patch virtualservice jadevectordb-canary-routing -n $NAMESPACE -p '{"spec":{"http":[{"match":[{"headers":{"canary":{"exact":"mostly-green"}}}],"route":[{"destination":{"host":"jadevectordb-blue-service","subset":"blue"},"weight":10},{"destination":{"host":"jadevectordb-green-service","subset":"green"},"weight":90}]}]}}'
        else
            echo "Istio not available, please manually update traffic routing"
            exit 1
        fi
        ;;
    "full-switch")
        echo "Switching 100% of traffic to green environment..."
        if [ "$ISTIOCTL_AVAILABLE" = true ]; then
            # Update Istio VirtualService to route 100% to green
            kubectl patch virtualservice jadevectordb-canary-routing -n $NAMESPACE -p '{"spec":{"http":[{"match":[{"headers":{"canary":{"exact":"enabled"}}}],"route":[{"destination":{"host":"jadevectordb-green-service","subset":"green"},"weight":100}]}]}}'
        else
            echo "Switching service selector to green environment..."
            # Update the main service to point to green
            cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: jadevectordb-service
  namespace: $NAMESPACE
spec:
  selector:
    app: jadevectordb
    version: green
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080
    - protocol: TCP
      port: 8081
      targetPort: 8081
  type: LoadBalancer
EOF
        fi
        ;;
    "rollback")
        echo "Rolling back to blue environment (100% traffic)..."
        if [ "$ISTIOCTL_AVAILABLE" = true ]; then
            # Update Istio VirtualService to route 100% to blue
            kubectl patch virtualservice jadevectordb-canary-routing -n $NAMESPACE -p '{"spec":{"http":[{"match":[{"headers":{"canary":{"exact":"disabled"}}}],"route":[{"destination":{"host":"jadevectordb-blue-service","subset":"blue"},"weight":100}]}]}}'
        else
            echo "Switching service selector back to blue environment..."
            # Update the main service to point to blue
            cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: jadevectordb-service
  namespace: $NAMESPACE
spec:
  selector:
    app: jadevectordb
    version: blue
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080
    - protocol: TCP
      port: 8081
      targetPort: 8081
  type: LoadBalancer
EOF
        fi
        ;;
    *)
        echo "Error: Invalid canary type. Use one of: partial, half, mostly-green, full-switch, rollback"
        exit 1
        ;;
esac

echo "Canary operation '$CANARY_TYPE' completed successfully!"
echo "To test the new configuration, use: curl -H 'canary: $CANARY_TYPE' http://jadevectordb.example.com/health"