# Blue-Green Deployment for JadeVectorDB

## Overview

This document describes the blue-green deployment strategy implemented for JadeVectorDB, enabling zero-downtime deployments with automated rollback capabilities.

Blue-green deployment is a technique that reduces downtime and risk by running two identical production environments called blue and green. At any time, only one of the environments is live, serving all production traffic. For this implementation:

- **Blue Environment**: The current stable production environment
- **Green Environment**: The new version of the application being deployed

## Architecture

### Components

The blue-green deployment system consists of:

1. **Two identical environments**: Blue and green with separate deployments, services, and storage
2. **Traffic routing mechanisms**: Using Kubernetes services, Ingress, and Istio for traffic management
3. **Health checking**: Automated validation of both environments
4. **Automated rollback**: System that monitors the new environment and reverts if issues are detected
5. **Canary deployment support**: Gradual traffic shifting capabilities
6. **Monitoring and alerting**: Prometheus-based monitoring with alerting rules

### Kubernetes Resources

- `blue-green-deployment.yaml`: Contains deployments and services for both blue and green environments
- `ingress-routing.yaml`: Ingress-based traffic routing configurations
- `traffic-routing.yaml`: Istio-based traffic routing configurations
- `health-check-services.yaml`: Services for health checking each environment
- `monitoring.yaml`: Prometheus rules and service monitors
- `canary-deployment.yaml`: Istio configurations for canary deployment

## Deployment Process

### Initial Setup

1. Deploy both blue and green environments
2. Initially route all traffic to the blue environment (the stable version)
3. Keep the green environment ready for new deployments

### Promotion Process

1. Deploy the new version to the green environment
2. Wait for the green environment to be ready and pass health checks
3. Gradually shift traffic to the green environment (canary approach)
4. Monitor for errors and performance issues
5. If successful, switch all traffic to the green environment
6. If issues are detected, automatically or manually roll back to the blue environment

## Usage

### Prerequisites

- Kubernetes cluster with necessary permissions
- `kubectl` installed and configured
- `istioctl` (optional, for Istio features)
- Helm (optional, for additional features)

### Deployment Scripts

The blue-green deployment system includes several scripts:

1. **Main Deployment Script**:
   ```bash
   ./deploy-blue-green.sh [blue-version] [green-version] [delay] [action]
   ```
   - Actions: `deploy`, `promote`, `rollback`, `info`
   - Example: `./deploy-blue-green.sh 1.0.0 1.1.0 60 promote`

2. **Traffic Switching**:
   ```bash
   ./switch-traffic.sh [blue|green] [namespace]
   ```
   - Example: `./switch-traffic.sh green jadevectordb-system`

3. **Health Checking**:
   ```bash
   ./health-check.sh [namespace] [path] [timeout]
   ```
   - Example: `./health-check.sh jadevectordb-system /health 30`

4. **Canary Deployment**:
   ```bash
   ./canary-deploy.sh [namespace] [canary-type]
   ```
   - Types: `partial`, `half`, `mostly-green`, `full-switch`, `rollback`
   - Example: `./canary-deploy.sh jadevectordb-system mostly-green`

5. **Rollback Monitoring**:
   ```bash
   ./rollback-monitor.sh [namespace] [timeout]
   ```

### Complete Deployment Example

1. **Deploy both environments**:
   ```bash
   ./deploy-blue-green.sh 1.0.0 1.1.0 60 deploy
   ```

2. **Verify both environments are healthy**:
   ```bash
   ./health-check.sh jadevectordb-system
   ```

3. **Promote the green environment**:
   ```bash
   ./deploy-blue-green.sh 1.0.0 1.1.0 60 promote
   ```

4. **Monitor the deployment**:
   ```bash
   ./deploy-blue-green.sh jadevectordb-system info
   ```

## Traffic Routing Mechanisms

The system supports multiple traffic routing approaches:

1. **Kubernetes Services**: Direct service switching between blue and green
2. **Ingress**: Using annotations to route specific traffic
3. **Istio**: Advanced traffic splitting with weighted routing

## Health Checking

Health checks are performed on both environments:

1. Pod readiness checks
2. Service connectivity tests
3. Application health endpoint validation
4. Performance and error rate monitoring

## Automated Rollback

The system includes automated rollback capabilities:

1. Monitors the health of the active environment after traffic switch
2. Automatically reverts to the previous stable environment if issues are detected
3. Provides configurable timeout and check intervals
4. Can be triggered manually if needed

## Canary Deployment Support

The system supports gradual traffic shifting:

- 10% to new version initially
- 50% for testing
- 90% for near-complete switch
- 100% for full production
- Easy rollback to previous version

## Best Practices

### Before Deployment

- Ensure sufficient cluster resources for both environments
- Test the deployment process in a staging environment
- Validate the rollback procedure
- Set appropriate resource limits and requests

### During Deployment

- Monitor application metrics during traffic shift
- Watch for error rate increases
- Check performance metrics
- Verify data consistency

### After Deployment

- Validate the new environment remains stable
- Clean up old resources when appropriate
- Update documentation if needed
- Plan the next deployment cycle

## Troubleshooting

### Common Issues

1. **Insufficient Resources**: Ensure the cluster has enough resources for both environments
2. **Service Discovery**: Verify that services are correctly routing to appropriate environments
3. **Health Checks**: Check that health endpoints are accessible and returning correct responses
4. **Configuration Issues**: Ensure environment-specific configurations are correctly applied

### Useful Commands

```bash
# Check deployment status
kubectl get deployments -n jadevectordb-system

# Check service endpoints
kubectl get endpoints -n jadevectordb-system

# Monitor logs
kubectl logs -l app=jadevectordb,version=blue -n jadevectordb-system
kubectl logs -l app=jadevectordb,version=green -n jadevectordb-system

# Check current traffic routing
kubectl describe service jadevectordb-service -n jadevectordb-system
```

## Monitoring and Alerting

The system includes:

- Prometheus rules to alert on environment health
- Service monitors for metric collection
- Dashboard templates for visualization
- Custom metrics for application-specific monitoring

## Security Considerations

- Ensure network policies allow communication between environments during deployment
- Use appropriate RBAC permissions for deployment scripts
- Secure access to management APIs
- Validate input parameters to deployment scripts

## Scaling and Performance

- Adjust replica counts in deployment configurations as needed
- Monitor resource utilization during deployment
- Consider the performance impact of running both environments
- Plan for capacity during the deployment window