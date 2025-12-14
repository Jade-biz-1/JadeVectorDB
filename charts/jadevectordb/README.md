# JadeVectorDB Helm Chart

This Helm chart provides an easy way to deploy JadeVectorDB on Kubernetes with production-ready configurations.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- PersistentVolume provisioner support in the underlying infrastructure (if persistence is enabled)

## Quick Start

### Install the Chart

```bash
# From the repository root
helm install jadevectordb ./charts/jadevectordb

# With custom values
helm install jadevectordb ./charts/jadevectordb -f custom-values.yaml

# In a specific namespace
helm install jadevectordb ./charts/jadevectordb --namespace jadevectordb --create-namespace
```

### Upgrade the Release

```bash
helm upgrade jadevectordb ./charts/jadevectordb

# With new values
helm upgrade jadevectordb ./charts/jadevectordb -f updated-values.yaml
```

### Uninstall the Chart

```bash
helm uninstall jadevectordb
```

## Configuration

### Key Configuration Options

The following table lists the most important configurable parameters of the JadeVectorDB chart and their default values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | JadeVectorDB Docker image repository | `jadevectordb/jadevectordb` |
| `image.tag` | Image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `jadevectordb.replicaCount` | Number of replicas (standalone mode) | `1` |
| `jadevectordb.config.port` | API server port | `8080` |
| `jadevectordb.config.rpcPort` | RPC server port | `8081` |
| `jadevectordb.config.logLevel` | Logging level | `INFO` |
| `jadevectordb.resources.limits.cpu` | CPU limit | `2` |
| `jadevectordb.resources.limits.memory` | Memory limit | `4Gi` |
| `jadevectordb.resources.requests.cpu` | CPU request | `500m` |
| `jadevectordb.resources.requests.memory` | Memory request | `1Gi` |
| `jadevectordb.persistence.enabled` | Enable data persistence | `true` |
| `jadevectordb.persistence.size` | Persistent Volume size | `10Gi` |
| `jadevectordb.persistence.storageClass` | Storage class for PVC | `""` (default) |
| `cluster.enabled` | Enable cluster mode (StatefulSet) | `false` |
| `cluster.size` | Number of nodes in cluster | `3` |
| `cluster.persistence.size` | Persistent Volume size for cluster | `50Gi` |
| `monitoring.enabled` | Enable monitoring stack | `false` |
| `monitoring.prometheus.enabled` | Enable Prometheus | `false` |
| `monitoring.grafana.enabled` | Enable Grafana | `false` |
| `ingress.enabled` | Enable ingress controller | `false` |

### Deployment Modes

#### 1. Standalone Mode (Default)

Deploys a single JadeVectorDB instance using a Deployment.

```bash
helm install jadevectordb ./charts/jadevectordb
```

#### 2. Cluster Mode

Deploys a distributed cluster with multiple nodes using StatefulSets.

```bash
helm install jadevectordb ./charts/jadevectordb \
  --set cluster.enabled=true \
  --set cluster.size=3
```

#### 3. With Monitoring

Enables Prometheus and Grafana for monitoring.

```bash
helm install jadevectordb ./charts/jadevectordb \
  --set monitoring.enabled=true \
  --set monitoring.prometheus.enabled=true \
  --set monitoring.grafana.enabled=true
```

#### 4. With Ingress

Exposes JadeVectorDB through an Ingress controller.

```bash
helm install jadevectordb ./charts/jadevectordb \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=jadevectordb.example.com \
  --set ingress.hosts[0].paths[0].path=/ \
  --set ingress.hosts[0].paths[0].pathType=Prefix
```

## Examples

### Production Deployment with Persistence

Create a `production-values.yaml` file:

```yaml
image:
  tag: "1.0.0"
  pullPolicy: Always

jadevectordb:
  replicaCount: 2
  config:
    logLevel: WARN
  resources:
    limits:
      cpu: 4
      memory: 8Gi
    requests:
      cpu: 1
      memory: 2Gi
  persistence:
    enabled: true
    size: 100Gi
    storageClass: "fast-ssd"

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: jadevectordb.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: jadevectordb-tls
      hosts:
        - jadevectordb.example.com
```

Deploy:

```bash
helm install jadevectordb ./charts/jadevectordb -f production-values.yaml
```

### Development Deployment

```bash
helm install jadevectordb ./charts/jadevectordb \
  --set image.tag=latest \
  --set jadevectordb.replicaCount=1 \
  --set jadevectordb.persistence.enabled=false \
  --set jadevectordb.config.logLevel=DEBUG
```

### Cluster Mode with Custom Resources

```bash
helm install jadevectordb ./charts/jadevectordb \
  --set cluster.enabled=true \
  --set cluster.size=5 \
  --set cluster.resources.limits.cpu=4 \
  --set cluster.resources.limits.memory=16Gi \
  --set cluster.persistence.size=200Gi
```

## Accessing JadeVectorDB

### Port Forward (Development)

```bash
kubectl port-forward svc/jadevectordb 8080:8080
```

Then access at `http://localhost:8080`

### Through Ingress (Production)

If ingress is enabled, access through the configured hostname:
```
http://jadevectordb.example.com
```

### Get Service Details

```bash
kubectl get svc
kubectl describe svc jadevectordb
```

## Monitoring

When monitoring is enabled, you can access:

- **Prometheus**: Port-forward to access metrics
  ```bash
  kubectl port-forward svc/prometheus 9090:9090
  ```

- **Grafana**: Port-forward to access dashboards
  ```bash
  kubectl port-forward svc/grafana 3000:3000
  ```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### Check Persistent Volume Claims

```bash
kubectl get pvc
kubectl describe pvc <pvc-name>
```

### Common Issues

**Issue**: Pods stuck in `Pending` state
- **Solution**: Check PVC status and ensure storage class is available

**Issue**: Database not accessible
- **Solution**: Verify service and ingress configurations
  ```bash
  kubectl get svc
  kubectl get ingress
  ```

**Issue**: Out of Memory errors
- **Solution**: Increase memory limits in values.yaml

## Chart Structure

```
charts/jadevectordb/
├── Chart.yaml           # Chart metadata
├── README.md           # This file
├── values.yaml         # Default configuration values
└── templates/
    ├── _helpers.tpl    # Template helpers
    ├── deployment.yaml # Deployment/StatefulSet templates
    └── service.yaml    # Service templates
```

## Related Documentation

- [Kubernetes Deployment Guide](../../docs/distributed_deployment_guide.md)
- [Main Documentation](../../README.md)
- [Raw Kubernetes Manifests](../../k8s/)

## Version History

- **v0.1.0** - Initial Helm chart release
  - Standalone and cluster deployment modes
  - Configurable resource limits
  - Persistent storage support
  - Optional monitoring and ingress

## Contributing

When updating the chart:
1. Update version in `Chart.yaml`
2. Update this README with new features
3. Test deployment with `helm install --dry-run`
4. Validate with `helm lint`

## Support

For issues and questions:
- GitHub Issues: [JadeVectorDB Issues](https://github.com/Jade-biz-1/JadeVectorDB/issues)
- Documentation: [JadeVectorDB Docs](../../docs/)
