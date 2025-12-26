# JadeVectorDB Operations Runbook

## Overview

This runbook provides operational procedures for deploying, maintaining, and troubleshooting JadeVectorDB in production environments.

**Last Updated**: December 26, 2025
**Version**: 1.7

---

## Table of Contents

1. [Deployment](#deployment)
2. [Health Checks](#health-checks)
3. [Backup and Restore](#backup-and-restore)
4. [Scaling](#scaling)
5. [Performance Tuning](#performance-tuning)
6. [Security Operations](#security-operations)
7. [Shutdown Procedures](#shutdown-procedures)

---

## Deployment

### Prerequisites

- Docker 20.10+ or Kubernetes 1.20+
- 2GB RAM minimum (4GB recommended)
- 10GB storage minimum
- Network connectivity (port 8080)

### Environment Configuration

Set required environment variables:

```bash
# Required in production
export JWT_SECRET="your-secret-key-min-32-chars"
export JADEVECTORDB_ENV=production

# Optional overrides
export JADEVECTORDB_PORT=8080
export JADEVECTORDB_HOST=0.0.0.0
export JADEVECTORDB_DB_PATH=/data/jadevectordb.db
export JADEVECTORDB_AUTH_DB_PATH=/data/jadevectordb_auth.db
export JADEVECTORDB_LOG_LEVEL=info
```

### Docker Deployment

```bash
# 1. Pull image
docker pull jadevectordb:latest

# 2. Run container
docker run -d \
  --name jadevectordb \
  -p 8080:8080 \
  -v /data:/data \
  -e JADEVECTORDB_ENV=production \
  -e JWT_SECRET=$JWT_SECRET \
  --restart unless-stopped \
  jadevectordb:latest

# 3. Verify deployment
curl http://localhost:8080/health
```

### Kubernetes Deployment

```bash
# Apply configuration
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify pods
kubectl get pods -l app=jadevectordb
kubectl logs -f deployment/jadevectordb
```

---

## Health Checks

### Endpoint Monitoring

**Basic Health Check**:
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": 1703000000000,
  "version": "1.0.0",
  "checks": {
    "database": "ok",
    "storage": "ok",
    "network": "ok"
  }
}
```

**Database Health Check**:
```bash
curl http://localhost:8080/health/db
```

**Prometheus Metrics**:
```bash
curl http://localhost:8080/metrics
```

### Automated Monitoring

Configure your monitoring system to:
- Poll `/health` every 30 seconds
- Alert if 3 consecutive failures
- Alert if response time > 5 seconds
- Alert on HTTP 503 responses

---

## Backup and Restore

### Database Backup

**Manual Backup**:
```bash
# Stop writes (optional for consistency)
curl -X POST http://localhost:8080/admin/readonly

# Copy database files
cp /data/jadevectordb.db /backup/jadevectordb.db.$(date +%Y%m%d_%H%M%S)
cp /data/jadevectordb_auth.db /backup/jadevectordb_auth.db.$(date +%Y%m%d_%H%M%S)

# Resume writes
curl -X POST http://localhost:8080/admin/readwrite
```

**Automated Backup Script**:
```bash
#!/bin/bash
# backup.sh - Run daily via cron

BACKUP_DIR=/backup/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# Backup databases
docker exec jadevectordb sqlite3 /data/jadevectordb.db ".backup /data/backup.db"
docker cp jadevectordb:/data/backup.db $BACKUP_DIR/jadevectordb.db

# Backup authentication database
docker cp jadevectordb:/data/jadevectordb_auth.db $BACKUP_DIR/

# Compress backups
gzip $BACKUP_DIR/*.db

# Cleanup old backups (keep 30 days)
find /backup -name "*.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR"
```

### Restore Procedure

```bash
# 1. Stop application
docker stop jadevectordb

# 2. Restore database files
cp /backup/20250117/jadevectordb.db.gz /data/
gunzip /data/jadevectordb.db.gz

cp /backup/20250117/jadevectordb_auth.db.gz /data/
gunzip /data/jadevectordb_auth.db.gz

# 3. Verify file permissions
chown jadevectordb:jadevectordb /data/*.db
chmod 600 /data/*.db

# 4. Start application
docker start jadevectordb

# 5. Verify restoration
curl http://localhost:8080/health/db
```

---

## Scaling

### Vertical Scaling

**Increase Memory**:
```bash
# Docker
docker update --memory=8g --memory-swap=8g jadevectordb

# Kubernetes
kubectl edit deployment jadevectordb
# Update resources.limits.memory: 8Gi
```

**Increase CPU**:
```bash
# Docker
docker update --cpus=4 jadevectordb

# Kubernetes
# Update resources.limits.cpu: "4"
```

### Horizontal Scaling

JadeVectorDB supports read replicas for query scaling:

```bash
# Deploy read replicas
kubectl scale deployment jadevectordb-replica --replicas=3

# Configure load balancer
kubectl apply -f k8s/service-lb.yaml
```

**Connection Pooling**:
- Configure `database.connection_pool_size` based on workload
- Default: 20 connections
- Recommended: (2 * CPU cores) + disk count

---

## Performance Tuning

### Database Configuration

Edit `config/performance.json`:

```json
{
  "database": {
    "connection_pool_size": 20,
    "query_timeout_seconds": 30,
    "max_retries": 3
  },
  "cache": {
    "permission_cache_size": 100000,
    "permission_cache_ttl_seconds": 300,
    "enable_query_cache": true
  }
}
```

### Performance Monitoring

**Key Metrics to Track**:
- `jadevectordb_db_operation_duration_seconds` (p95 < 100ms)
- `jadevectordb_permission_check_duration_seconds` (p95 < 1ms)
- `jadevectordb_auth_duration_seconds` (p95 < 200ms)
- `jadevectordb_permission_cache_hits_total / _misses_total` (hit rate > 80%)

**Performance Baseline**:
- Permission checks: 0.97μs (with cache)
- Database queries: < 10ms average
- Authentication: < 100ms average

### Optimization Checklist

- [ ] Enable permission caching (5min TTL)
- [ ] Configure appropriate connection pool size
- [ ] Enable rate limiting for endpoints
- [ ] Monitor cache hit rates
- [ ] Index frequently queried fields
- [ ] Use prepared statements
- [ ] Enable compression for large responses

---

## Security Operations

### User Management

**Create Admin User**:
```bash
curl -X POST http://localhost:8080/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "email": "admin@example.com",
    "password": "SecurePassword123!",
    "is_admin": true
  }'
```

**Reset User Password**:
```bash
curl -X POST http://localhost:8080/v1/auth/forgot-password \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com"}'
```

### Security Monitoring

**Check Failed Logins**:
```bash
# View metrics
curl http://localhost:8080/metrics | grep failed_logins

# Check audit logs
docker exec jadevectordb tail -f /var/log/jadevectordb/audit.log
```

**IP Block Management**:
```bash
# View blocked IPs
curl http://localhost:8080/metrics | grep ip_blocks

# Unblock IP (requires admin)
curl -X DELETE http://localhost:8080/v1/admin/ip-blocks/192.168.1.100 \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Security Best Practices

1. **Rotate Secrets**: Rotate JWT_SECRET every 90 days
2. **Enable 2FA**: Required for admin accounts in production
3. **Monitor Metrics**: Alert on spike in failed logins
4. **Regular Audits**: Review security audit logs weekly
5. **Rate Limiting**: Enforce on all public endpoints
6. **TLS/SSL**: Use HTTPS in production
7. **Firewall**: Restrict access to port 8080

---

## Emergency Procedures

### Service Restart

```bash
# Graceful restart
docker restart jadevectordb

# Force restart (if frozen)
docker stop jadevectordb
docker start jadevectordb
```

### Rollback Deployment

```bash
# Docker
docker stop jadevectordb
docker run -d --name jadevectordb \
  ... \
  jadevectordb:1.5.0

# Kubernetes
kubectl rollout undo deployment/jadevectordb
kubectl rollout status deployment/jadevectordb
```

### Database Corruption Recovery

```bash
# 1. Stop service
docker stop jadevectordb

# 2. Verify database integrity
sqlite3 /data/jadevectordb.db "PRAGMA integrity_check;"

# 3. If corrupted, restore from backup
cp /backup/latest/jadevectordb.db /data/

# 4. Restart service
docker start jadevectordb
```

---

## Shutdown Procedures

### Graceful Shutdown via Admin Endpoint

JadeVectorDB provides a secure admin endpoint for graceful server shutdown. This is the recommended method for controlled shutdowns.

**Prerequisites**:
- Admin user credentials
- Valid JWT authentication token
- Admin role assigned to user

**Step-by-Step Procedure**:

```bash
# Step 1: Login as admin user
TOKEN=$(curl -s -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | \
  jq -r '.token')

# Step 2: Initiate graceful shutdown
curl -X POST http://localhost:8080/admin/shutdown \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"

# Expected response:
# {
#   "status": "shutting_down",
#   "message": "Server shutdown initiated"
# }

# Step 3: Verify server has stopped (after ~1 second)
curl http://localhost:8080/health
# Should fail with connection refused
```

**What Happens During Graceful Shutdown**:
1. Authentication and authorization are verified
2. HTTP response is sent to client
3. After 500ms delay:
   - Server stops accepting new connections
   - In-flight requests are completed
   - Database connections are closed properly
   - Persistent data is flushed to disk
   - Server process exits cleanly

**Using the Frontend Dashboard**:

Admin users can also shutdown the server from the web dashboard:

1. Login to the dashboard as admin user
2. Navigate to the dashboard page (http://localhost:8080/dashboard)
3. Click the red "Shutdown Server" button (visible only to admin users)
4. Confirm the shutdown in the dialog
5. Server will initiate graceful shutdown

**When to Use Graceful Shutdown**:
- Planned maintenance windows
- Configuration changes requiring restart
- Version upgrades
- Clean environment teardown
- Before backup operations (optional)

### Emergency Shutdown Methods

If the admin endpoint is unavailable, use these alternative methods:

**Docker Container Shutdown**:
```bash
# Graceful stop (30 second timeout)
docker stop jadevectordb

# Force stop (immediate)
docker stop -t 0 jadevectordb

# Stop and remove
docker stop jadevectordb && docker rm jadevectordb
```

**Kubernetes Pod Shutdown**:
```bash
# Delete deployment (graceful)
kubectl delete deployment jadevectordb

# Delete pod (will be recreated if deployment exists)
kubectl delete pod jadevectordb-<pod-id>

# Scale to zero
kubectl scale deployment jadevectordb --replicas=0
```

**Process-Level Shutdown**:
```bash
# Find process ID
PID=$(ps aux | grep jadevectordb | grep -v grep | awk '{print $2}')

# Graceful shutdown (SIGTERM)
kill $PID

# Force shutdown if frozen (SIGKILL)
kill -9 $PID
```

### Pre-Shutdown Checklist

Before initiating shutdown in production:

- [ ] Notify affected users/teams
- [ ] Drain active connections from load balancer
- [ ] Complete or pause long-running operations
- [ ] Verify recent backup exists
- [ ] Document reason for shutdown
- [ ] Schedule maintenance window
- [ ] Prepare rollback plan

### Post-Shutdown Verification

After shutdown, verify:

```bash
# Confirm process is stopped
ps aux | grep jadevectordb
# Should return no results

# Check data persistence
ls -lh /data/jadevectordb*.db
# Database files should exist with recent timestamps

# Verify backup integrity (if backup was triggered)
sqlite3 /backup/latest/jadevectordb.db "PRAGMA integrity_check;"
```

### Restart After Shutdown

```bash
# Docker
docker start jadevectordb

# Kubernetes
kubectl scale deployment jadevectordb --replicas=1

# Verify startup
curl http://localhost:8080/health
```

### Troubleshooting Shutdown Issues

**Issue: Shutdown endpoint returns 401 Unauthorized**

```bash
# Verify user has admin role
curl -X GET http://localhost:8080/v1/users/<user-id> \
  -H "Authorization: Bearer $TOKEN"

# Check roles array includes "admin"
```

**Issue: Server hangs and doesn't shutdown**

```bash
# Wait 30 seconds for graceful shutdown
sleep 30

# If still running, force shutdown
docker stop -t 0 jadevectordb
# or
kill -9 <PID>
```

**Issue: Data loss after shutdown**

```bash
# This should not happen with graceful shutdown
# If data is missing, restore from backup:
cp /backup/latest/jadevectordb.db /data/
docker start jadevectordb
```

### Security Considerations

- Admin shutdown endpoint is protected by RBAC
- All shutdown attempts are logged in audit logs
- Unauthorized attempts will be blocked
- Consider restricting network access to admin endpoints
- Monitor for unusual shutdown patterns

**Audit Log Example**:
```
[2025-12-26 20:55:41] [INFO] Shutdown authorized by user: admin
[2025-12-26 20:55:41] [INFO] Shutdown initiated successfully
[2025-12-26 20:55:41] [INFO] Executing shutdown callback...
```

For detailed information about the admin shutdown endpoint, see [Admin Endpoints Reference](admin_endpoints.md).

---

## Support Contacts

- **On-Call Engineer**: See PagerDuty schedule
- **Database Team**: dba-team@company.com
- **Security Team**: security@company.com
- **Documentation**: https://docs.jadevectordb.io

---

## Appendix

### Quick Reference Commands

```bash
# Health check
curl http://localhost:8080/health

# View logs
docker logs -f jadevectordb

# Check metrics
curl http://localhost:8080/metrics | grep jadevectordb

# Backup database
docker exec jadevectordb sqlite3 /data/jadevectordb.db ".backup /data/backup.db"

# View active sessions
curl http://localhost:8080/metrics | grep active_sessions
```

### Configuration Files

- Production: `/config/production.json`
- Development: `/config/development.json`
- Performance: `/config/performance.json`
- Logging: `/config/logging.json`

### Log Locations

- Application: `/var/log/jadevectordb/app.log`
- Audit: `/var/log/jadevectordb/audit.log`
- Access: `/var/log/jadevectordb/access.log`
- Error: `/var/log/jadevectordb/error.log`
