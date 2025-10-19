#!/bin/bash

# Script to switch traffic between blue and green environments
# Usage: ./switch-traffic.sh <blue|green> [namespace]

set -e  # Exit immediately if a command exits with a non-zero status

TARGET_ENV=${1:-"blue"}  # Default to blue
NAMESPACE=${2:-"jadevectordb-system"}

if [ "$TARGET_ENV" != "blue" && "$TARGET_ENV" != "green" ]; then
    echo "Error: Target environment must be 'blue' or 'green'"
    echo "Usage: $0 <blue|green> [namespace]"
    exit 1
fi

echo "Switching traffic to $TARGET_ENV environment in namespace $NAMESPACE..."

# Check if kubectl is available
if ! command -v kubectl >/dev/null 2>&1; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Update the main service to point to the target environment
echo "Updating main service to point to $TARGET_ENV environment..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: jadevectordb-service
  namespace: $NAMESPACE
spec:
  selector:
    app: jadevectordb
    version: $TARGET_ENV
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080
    - protocol: TCP
      port: 8081
      targetPort: 8081
  type: LoadBalancer
EOF

echo "Traffic successfully switched to $TARGET_ENV environment"
echo "Validating service is ready..."

# Wait for service to be ready
sleep 10

# Check if pods in the target environment are ready
POD_COUNT=$(kubectl get pods -n $NAMESPACE -l app=jadevectordb,version=$TARGET_ENV --field-selector=status.phase=Running --no-headers | wc -l)
if [ $POD_COUNT -eq 0 ]; then
    echo "Warning: No running pods found for $TARGET_ENV environment"
else
    echo "$POD_COUNT pods are running in $TARGET_ENV environment"
fi

# Get the external IP of the service
echo "Getting service external IP..."
SERVICE_IP=$(kubectl get svc jadevectordb-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$SERVICE_IP" ]; then
    SERVICE_IP=$(kubectl get svc jadevectordb-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    echo "Service ClusterIP: $SERVICE_IP"
else
    echo "Service External IP: $SERVICE_IP"
fi

echo "Traffic switch completed successfully!"