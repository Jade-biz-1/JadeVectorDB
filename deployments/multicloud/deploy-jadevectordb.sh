#!/bin/bash

# Cloud-agnostic deployment script for JadeVectorDB
# Supports AWS, Azure, and GCP with minimal configuration changes

set -e  # Exit immediately if a command exits with a non-zero status

# Default configuration
CLOUD_PROVIDER=""
CLUSTER_NAME="jadevectordb-cluster"
REGION=""
PROJECT_ID=""
CONFIG_FILE="jadevectordb-config.yaml"
NAMESPACE="jadevectordb"
HELM_VALUES_FILE="helm-values.yaml"

# Help function
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -p, --provider PROVIDER    Cloud provider (aws, azure, gcp)"
    echo "  -n, --name NAME           Cluster name (default: jadevectordb-cluster)"
    echo "  -r, --region REGION       Region for deployment"
    echo "  -i, --project-id PROJECT  Project ID (for GCP)"
    echo "  -c, --config CONFIG       Configuration file (default: jadevectordb-config.yaml)"
    echo "  --create-cluster          Create new cluster (default: false)"
    echo "  --update-config           Update existing cluster configuration (default: false)"
    echo "  -h, --help               Show this help message"
    exit 1
}

# Parse command line arguments
CREATE_CLUSTER=false
UPDATE_CONFIG=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--provider)
            CLOUD_PROVIDER="$2"
            shift 2
            ;;
        -n|--name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -i|--project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --create-cluster)
            CREATE_CLUSTER=true
            shift
            ;;
        --update-config)
            UPDATE_CONFIG=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$CLOUD_PROVIDER" ]; then
    echo "Error: Cloud provider is required (-p or --provider)"
    usage
fi

if [ -z "$REGION" ]; then
    echo "Error: Region is required (-r or --region)"
    usage
fi

# Function to check prerequisites
check_prerequisites() {
    echo "Checking prerequisites for $CLOUD_PROVIDER deployment..."
    
    case $CLOUD_PROVIDER in
        aws)
            if ! command -v aws >/dev/null 2>&1; then
                echo "Error: AWS CLI is not installed"
                exit 1
            fi
            if ! command -v kubectl >/dev/null 2>&1; then
                echo "Error: kubectl is not installed"
                exit 1
            fi
            if ! command -v helm >/dev/null 2>&1; then
                echo "Error: Helm is not installed"
                exit 1
            fi
            ;;
        azure)
            if ! command -v az >/dev/null 2>&1; then
                echo "Error: Azure CLI is not installed"
                exit 1
            fi
            if ! command -v kubectl >/dev/null 2>&1; then
                echo "Error: kubectl is not installed"
                exit 1
            fi
            if ! command -v helm >/dev/null 2>&1; then
                echo "Error: Helm is not installed"
                exit 1
            fi
            ;;
        gcp)
            if ! command -v gcloud >/dev/null 2>&1; then
                echo "Error: Google Cloud SDK is not installed"
                exit 1
            fi
            if ! command -v kubectl >/dev/null 2>&1; then
                echo "Error: kubectl is not installed"
                exit 1
            fi
            if ! command -v helm >/dev/null 2>&1; then
                echo "Error: Helm is not installed"
                exit 1
            fi
            ;;
        *)
            echo "Unsupported cloud provider: $CLOUD_PROVIDER"
            exit 1
            ;;
    esac
    
    echo "Prerequisites check passed for $CLOUD_PROVIDER"
}

# Function to create cluster based on cloud provider
create_cluster() {
    echo "Creating cluster $CLUSTER_NAME in $REGION on $CLOUD_PROVIDER..."
    
    case $CLOUD_PROVIDER in
        aws)
            # Check if cluster already exists
            if aws eks list-clusters --region $REGION | grep -q $CLUSTER_NAME; then
                echo "Cluster $CLUSTER_NAME already exists"
                return 0
            fi
            
            # Create EKS cluster using eksctl
            cat > eks-cluster-config.yaml << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $REGION
  version: "1.27"

nodeGroups:
  - name: ng-primary
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 1
    maxSize: 10
    volumeSize: 100
    ssh:
      allow: true
    iam:
      withAddonPolicies:
        ebs: true
        fsx: true
        efs: true
        albIngress: true
EOF

            eksctl create cluster -f eks-cluster-config.yaml
            ;;
        azure)
            # Check if resource group exists, create if not
            az group exists --name $CLUSTER_NAME-rg >/dev/null 2>&1 || \
            az group create --name $CLUSTER_NAME-rg --location $REGION
            
            # Check if cluster already exists
            if az aks show --name $CLUSTER_NAME --resource-group $CLUSTER_NAME-rg >/dev/null 2>&1; then
                echo "Cluster $CLUSTER_NAME already exists"
                return 0
            fi
            
            # Create AKS cluster
            az aks create \
                --resource-group $CLUSTER_NAME-rg \
                --name $CLUSTER_NAME \
                --node-count 3 \
                --enable-addons monitoring \
                --generate-ssh-keys \
                --kubernetes-version 1.27 \
                --node-vm-size Standard_D4_v2
            ;;
        gcp)
            if [ -z "$PROJECT_ID" ]; then
                echo "Error: Project ID is required for GCP deployments (-i or --project-id)"
                exit 1
            fi
            
            # Check if cluster already exists
            if gcloud container clusters list --project $PROJECT_ID --filter="name=$CLUSTER_NAME" | grep -q $CLUSTER_NAME; then
                echo "Cluster $CLUSTER_NAME already exists"
                return 0
            fi
            
            # Create GKE cluster
            gcloud container clusters create $CLUSTER_NAME \
                --project=$PROJECT_ID \
                --zone=$REGION-a \
                --num-nodes=3 \
                --machine-type=n1-standard-4 \
                --disk-size=100GB \
                --enable-autoscaling \
                --min-nodes=1 \
                --max-nodes=10 \
                --enable-cloud-logging \
                --enable-cloud-monitoring \
                --enable-autorepair \
                --enable-autoupgrade
            ;;
    esac
    
    echo "Cluster $CLUSTER_NAME created successfully"
}

# Function to configure kubectl context
configure_kubectl() {
    echo "Configuring kubectl for $CLUSTER_NAME..."
    
    case $CLOUD_PROVIDER in
        aws)
            aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION
            ;;
        azure)
            az aks get-credentials --resource-group $CLUSTER_NAME-rg --name $CLUSTER_NAME
            ;;
        gcp)
            gcloud container clusters get-credentials $CLUSTER_NAME --zone=$REGION-a --project=$PROJECT_ID
            ;;
    esac
    
    echo "kubectl configured for cluster $CLUSTER_NAME"
}

# Function to deploy JadeVectorDB using Helm
deploy_jadevectordb() {
    echo "Deploying JadeVectorDB to $CLUSTER_NAME..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Generate Helm values file from config (simplified approach)
    cat > $HELM_VALUES_FILE << EOF
jadevectordb:
  replicaCount: 3
  image:
    repository: jadevectordb/jadevectordb
    tag: latest
  config:
    logLevel: INFO
    port: 8080
    rpcPort: 8081
  resources:
    limits:
      cpu: "2"
      memory: "4Gi"
    requests:
      cpu: "500m"
      memory: "1Gi"
  service:
    type: LoadBalancer
    port: 8080

cluster:
  enabled: true
  size: 3

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
EOF

    # Install or upgrade JadeVectorDB using Helm
    if helm status jadevectordb -n $NAMESPACE >/dev/null 2>&1; then
        echo "Upgrading existing JadeVectorDB installation..."
        helm upgrade jadevectordb ./../../charts/jadevectordb -f $HELM_VALUES_FILE -n $NAMESPACE
    else
        echo "Installing new JadeVectorDB..."
        helm install jadevectordb ./../../charts/jadevectordb -f $HELM_VALUES_FILE -n $NAMESPACE
    fi
    
    echo "JadeVectorDB deployed successfully"
}

# Main deployment flow
main() {
    echo "Starting deployment of JadeVectorDB to $CLOUD_PROVIDER..."
    
    check_prerequisites
    
    if [ "$CREATE_CLUSTER" = true ]; then
        create_cluster
    fi
    
    configure_kubectl
    
    deploy_jadevectordb
    
    # Wait for deployment to be ready
    echo "Waiting for JadeVectorDB to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=jadevectordb -n $NAMESPACE --timeout=300s
    
    echo "JadeVectorDB deployment completed successfully!"
    echo "Service external IP can be found with: kubectl get svc -n $NAMESPACE"
}

# Run main function
main "$@"