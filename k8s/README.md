# JadeVectorDB Kubernetes Manifests

This directory contains raw Kubernetes manifest files for deploying JadeVectorDB. These manifests provide direct control over Kubernetes resources without using Helm.

## 📋 Contents

| File | Description | Type | Use Case |
|------|-------------|------|----------|
| `jadevectordb-deployment.yaml` | Single-instance deployment | Deployment | Development, testing, single-node deployments |
| `jadevectordb-cluster.yaml` | Distributed cluster deployment | StatefulSet | Production, high-availability deployments |
| `monitoring.yaml` | Prometheus and Grafana stack | Multiple | Monitoring and observability |

## 🚀 Quick Start

### Prerequisites

- Kubernetes cluster (1.19+)
- `kubectl` configured to access your cluster
- Container runtime with JadeVectorDB image available

### Deploy Standalone Instance

```bash
# Create namespace and deploy single instance
kubectl apply -f jadevectordb-deployment.yaml

# Verify deployment
kubectl get pods -n jadevectordb-system
kubectl get svc -n jadevectordb-system
```

### Deploy Distributed Cluster

```bash
# Deploy 3-node cluster with StatefulSet
kubectl apply -f jadevectordb-cluster.yaml

# Verify StatefulSet
kubectl get statefulset -n jadevectordb-system
kubectl get pods -n jadevectordb-system -l app=jadevectordb-cluster
```

### Deploy Monitoring Stack

```bash
# Deploy Prometheus and Grafana
kubectl apply -f monitoring.yaml

# Verify monitoring components
kubectl get pods -n jadevectordb-system -l component=monitoring
```

## 📦 Deployment Files

### jadevectordb-deployment.yaml

**Purpose**: Single-node deployment suitable for development and testing.

**Resources Created**:
- Namespace: `jadevectordb-system`
- ConfigMap: `jadevectordb-config` (environment variables)
- Deployment: `jadevectordb` (1 replica)
- Service: `jadevectordb-service` (LoadBalancer)
- PersistentVolumeClaim: `jadevectordb-data-pvc` (10Gi)

**Configuration**:
```yaml
Replicas: 1
CPU Request: 500m
CPU Limit: 2
Memory Request: 1Gi
Memory Limit: 4Gi
Storage: 10Gi
Port: 8080
```

**Access**:
```bash
# Get service external IP (if LoadBalancer)
kubectl get svc jadevectordb-service -n jadevectordb-system

# Or use port-forward for local access
kubectl port-forward svc/jadevectordb-service 8080:8080 -n jadevectordb-system
```

### jadevectordb-cluster.yaml

**Purpose**: Production-ready distributed cluster with 3 nodes.

**Resources Created**:
- Service: `jadevectordb-headless-service` (headless for StatefulSet)
- StatefulSet: `jadevectordb-cluster` (3 replicas)
- VolumeClaimTemplate: Automatic PVC creation for each pod

**Configuration**:
```yaml
Replicas: 3
CPU Request: 1
CPU Limit: 4
Memory Request: 2Gi
Memory Limit: 8Gi
Storage per node: 50Gi
API Port: 8080
RPC Port: 8081
```

**Features**:
- **Stable Network Identity**: Each pod gets a stable hostname
- **Ordered Deployment**: Pods created sequentially
- **Persistent Storage**: Each pod has its own PersistentVolume
- **Headless Service**: Direct pod-to-pod communication

**Access Individual Pods**:
```bash
# Access specific pod
kubectl exec -it jadevectordb-cluster-0 -n jadevectordb-system -- /bin/bash

# Pod DNS names (within cluster)
jadevectordb-cluster-0.jadevectordb-headless-service.jadevectordb-system.svc.cluster.local
jadevectordb-cluster-1.jadevectordb-headless-service.jadevectordb-system.svc.cluster.local
jadevectordb-cluster-2.jadevectordb-headless-service.jadevectordb-system.svc.cluster.local
```

### monitoring.yaml

**Purpose**: Observability stack with Prometheus and Grafana.

**Resources Created**:
- Deployment: `prometheus` (metrics collection)
- Service: `prometheus-service`
- Deployment: `grafana` (visualization)
- Service: `grafana-service`
- ConfigMap: `prometheus-config` (scrape configuration)

**Access Monitoring**:
```bash
# Prometheus
kubectl port-forward svc/prometheus-service 9090:9090 -n jadevectordb-system
# Open http://localhost:9090

# Grafana (default credentials: admin/admin)
kubectl port-forward svc/grafana-service 3000:3000 -n jadevectordb-system
# Open http://localhost:3000
```

## 🔧 Customization

### Modify Resource Limits

Edit the YAML files to adjust CPU/memory:

```yaml
resources:
  limits:
    cpu: "4"
    memory: "8Gi"
  requests:
    cpu: "1"
    memory: "2Gi"
```

### Change Replica Count

For cluster deployment:

```yaml
spec:
  replicas: 5  # Change from 3 to 5 nodes
```

### Adjust Storage Size

```yaml
storage: 100Gi  # Change from default
```

### Configure Environment Variables

Edit the ConfigMap in `jadevectordb-deployment.yaml`:

```yaml
data:
  JADE_DB_LOG_LEVEL: "DEBUG"
  JADE_DB_PORT: "8080"
  JADE_DB_MAX_CONNECTIONS: "1000"
```

## 📊 Monitoring and Debugging

### Check Pod Status

```bash
# All pods
kubectl get pods -n jadevectordb-system

# Detailed pod info
kubectl describe pod <pod-name> -n jadevectordb-system

# Pod logs
kubectl logs <pod-name> -n jadevectordb-system

# Follow logs
kubectl logs -f <pod-name> -n jadevectordb-system
```

### Check Services

```bash
# List services
kubectl get svc -n jadevectordb-system

# Service details
kubectl describe svc <service-name> -n jadevectordb-system

# Test service connectivity
kubectl run -it --rm debug --image=alpine --restart=Never -n jadevectordb-system -- sh
# Inside pod: wget -qO- http://jadevectordb-service:8080/health
```

### Check Persistent Volumes

```bash
# List PVCs
kubectl get pvc -n jadevectordb-system

# PVC details
kubectl describe pvc <pvc-name> -n jadevectordb-system

# List PVs
kubectl get pv
```

### Check StatefulSet

```bash
# StatefulSet status
kubectl get statefulset jadevectordb-cluster -n jadevectordb-system

# StatefulSet details
kubectl describe statefulset jadevectordb-cluster -n jadevectordb-system

# Scale StatefulSet
kubectl scale statefulset jadevectordb-cluster --replicas=5 -n jadevectordb-system
```

## 🔄 Common Operations

### Update Deployment

```bash
# After modifying YAML file
kubectl apply -f jadevectordb-deployment.yaml

# Force rolling update
kubectl rollout restart deployment/jadevectordb -n jadevectordb-system

# Check rollout status
kubectl rollout status deployment/jadevectordb -n jadevectordb-system
```

### Update StatefulSet

```bash
kubectl apply -f jadevectordb-cluster.yaml

# Delete and recreate specific pod (will use new config)
kubectl delete pod jadevectordb-cluster-0 -n jadevectordb-system
```

### Scale Cluster

```bash
# Scale up
kubectl scale statefulset jadevectordb-cluster --replicas=5 -n jadevectordb-system

# Scale down
kubectl scale statefulset jadevectordb-cluster --replicas=2 -n jadevectordb-system
```

### Backup Data

```bash
# Create snapshot of PVC (depends on storage class)
kubectl get pvc -n jadevectordb-system

# Or copy data from pod
kubectl cp jadevectordb-cluster-0:/data ./backup-data -n jadevectordb-system
```

## 🛠️ Troubleshooting

### Pod Won't Start

```bash
# Check events
kubectl get events -n jadevectordb-system --sort-by='.lastTimestamp'

# Check pod description for errors
kubectl describe pod <pod-name> -n jadevectordb-system

# Common issues:
# - Image pull errors: Check image name and registry access
# - Resource constraints: Check node capacity with 'kubectl describe nodes'
# - PVC binding issues: Check storage class and PV availability
```

### StatefulSet Pod Stuck in Pending

```bash
# Check PVC status
kubectl get pvc -n jadevectordb-system

# If PVC pending, check storage class
kubectl get storageclass

# Describe PVC for errors
kubectl describe pvc <pvc-name> -n jadevectordb-system
```

### Service Not Accessible

```bash
# Verify service endpoints
kubectl get endpoints -n jadevectordb-system

# Check if pods are selected
kubectl get pods -n jadevectordb-system --show-labels

# Test from within cluster
kubectl run -it --rm curl --image=curlimages/curl --restart=Never -- \
  curl http://jadevectordb-service.jadevectordb-system:8080/health
```

### High Resource Usage

```bash
# Check pod resource usage
kubectl top pods -n jadevectordb-system

# Check node resource usage
kubectl top nodes

# Adjust resource limits in YAML if needed
```

## 🌐 Production Considerations

### High Availability

1. **Use StatefulSet** for distributed deployment
2. **Set replicas ≥ 3** for quorum-based operations
3. **Configure anti-affinity** to spread pods across nodes
4. **Use quality storage class** (e.g., SSD-backed)

### Security

1. **Use secrets** for sensitive data instead of ConfigMaps
2. **Enable RBAC** and limit service account permissions
3. **Use NetworkPolicies** to restrict pod communication
4. **Scan images** for vulnerabilities before deployment

### Monitoring

1. **Deploy monitoring.yaml** for observability
2. **Set up alerts** in Prometheus for critical metrics
3. **Configure Grafana dashboards** for visualization
4. **Enable pod resource metrics** with metrics-server

## 🔀 Migration from Helm

If you prefer Helm charts over raw manifests:

```bash
# Use Helm chart instead
cd ../charts/jadevectordb
helm install jadevectordb .
```

See [Helm Chart README](../charts/jadevectordb/README.md) for details.

## 📚 Additional Resources

- [JadeVectorDB Documentation](../docs/)
- [Distributed Deployment Guide](../docs/distributed_deployment_guide.md)
- [Docker Deployment Guide](../docs/DOCKER_DEPLOYMENT.md)
- [Kubernetes Official Docs](https://kubernetes.io/docs/)
- [StatefulSets Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)

## ⚠️ Important Notes

1. **Namespace**: All resources are created in `jadevectordb-system` namespace
2. **Storage**: Ensure your cluster has a default StorageClass or specify one
3. **Resources**: Adjust CPU/memory based on your workload requirements
4. **Images**: Update image tags to specific versions in production
5. **Monitoring**: Configure persistent storage for Prometheus data in production

## 🤝 Contributing

When updating manifests:
1. Test deployments with `kubectl apply --dry-run=client`
2. Validate YAML with `kubectl apply --validate=true`
3. Update this README with changes
4. Test in development cluster before production

## 📞 Support

For issues and questions:
- GitHub Issues: [JadeVectorDB Issues](https://github.com/Jade-biz-1/JadeVectorDB/issues)
- Documentation: [Main README](../README.md)
