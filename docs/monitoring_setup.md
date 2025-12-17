# JadeVectorDB Monitoring Setup

## Prometheus Configuration

### Prometheus Setup

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'jadevectordb'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

### Key Metrics

**Authentication Metrics**:
- `jadevectordb_auth_requests_total` - Total auth requests
- `jadevectordb_auth_duration_seconds` - Auth latency histogram
- `jadevectordb_failed_logins_total` - Failed login attempts
- `jadevectordb_active_sessions` - Active user sessions

**Database Metrics**:
- `jadevectordb_db_operations_total` - Total DB operations
- `jadevectordb_db_operation_duration_seconds` - Query latency
- `jadevectordb_db_connection_errors_total` - Connection failures
- `jadevectordb_db_query_retries_total` - Query retry count

**Permission Metrics**:
- `jadevectordb_permission_checks_total` - Total permission checks
- `jadevectordb_permission_cache_hits_total` - Cache hits
- `jadevectordb_permission_cache_misses_total` - Cache misses

**Security Metrics**:
- `jadevectordb_rate_limit_exceeded_total` - Rate limit violations
- `jadevectordb_ip_blocks_total` - IP blocks issued

---

## Grafana Dashboard

### Import Dashboard

1. Open Grafana → Dashboards → Import
2. Upload `grafana/jadevectordb-dashboard.json`
3. Select Prometheus data source
4. Click Import

### Key Panels

**System Health**:
- Service uptime
- Request rate
- Error rate
- Response time (p50, p95, p99)

**Authentication**:
- Login success rate
- Active sessions trend
- Failed logins per hour
- Average auth duration

**Database Performance**:
- Query latency histogram
- Operations per second
- Connection pool utilization
- Query retry rate

**Cache Performance**:
- Permission cache hit rate
- Cache size over time
- Eviction rate

---

## Alert Rules

### Critical Alerts

**High Error Rate**:
```yaml
- alert: HighErrorRate
  expr: rate(jadevectordb_auth_errors_total[5m]) > 10
  for: 5m
  annotations:
    summary: "High authentication error rate"
    description: "Error rate is {{ $value }} errors/sec"
```

**Database Connection Failures**:
```yaml
- alert: DatabaseConnectionFailures
  expr: rate(jadevectordb_db_connection_errors_total[5m]) > 1
  for: 2m
  annotations:
    summary: "Database connection failures detected"
```

**Circuit Breaker Open**:
```yaml
- alert: CircuitBreakerOpen
  expr: jadevectordb_circuit_breaker_open == 1
  for: 1m
  annotations:
    summary: "Circuit breaker is open - database unavailable"
```

---

## Monitoring Best Practices

1. **Alert Thresholds**: Review and adjust based on baseline
2. **On-Call Rotation**: Ensure 24/7 coverage
3. **Alert Fatigue**: Tune alerts to reduce false positives
4. **Dashboards**: Review weekly for trends
5. **Capacity Planning**: Monitor growth trends monthly

---

## Example Queries

### Authentication Success Rate
```promql
rate(jadevectordb_auth_requests_total{status="success"}[5m]) /
rate(jadevectordb_auth_requests_total[5m]) * 100
```

### Cache Hit Rate
```promql
rate(jadevectordb_permission_cache_hits_total[5m]) /
(rate(jadevectordb_permission_cache_hits_total[5m]) +
 rate(jadevectordb_permission_cache_misses_total[5m])) * 100
```

### P95 Database Latency
```promql
histogram_quantile(0.95, 
  rate(jadevectordb_db_operation_duration_seconds_bucket[5m]))
```
