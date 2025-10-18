# Filtered Search Metrics and Monitoring Dashboard

## Overview

This document describes the filtered search metrics and monitoring capabilities implemented in JadeVectorDB. The system provides comprehensive observability into search performance, filter effectiveness, and system health.

## Key Metrics

### Search Volume Metrics

| Metric Name | Description | Type |
|-------------|-------------|------|
| `search_requests_total` | Total number of search requests | Counter |
| `filtered_search_requests_total` | Total number of filtered search requests | Counter |
| `search_results_total` | Total number of search results returned | Counter |
| `filtered_search_results_total` | Total number of filtered search results returned | Counter |

### Performance Metrics

| Metric Name | Description | Type |
|-------------|-------------|------|
| `search_request_duration_seconds` | Time spent processing search requests | Histogram |
| `filtered_search_request_duration_seconds` | Time spent processing filtered search requests | Histogram |
| `filter_application_duration_seconds` | Time spent applying filters to vectors | Histogram |
| `active_searches` | Number of currently active searches | Gauge |
| `active_filtered_searches` | Number of currently active filtered searches | Gauge |

### Cache Metrics

| Metric Name | Description | Type |
|-------------|-------------|------|
| `filter_cache_hits_total` | Total number of filter cache hits | Counter |
| `filter_cache_misses_total` | Total number of filter cache misses | Counter |
| `filter_cache_hit_ratio` | Ratio of cache hits to total cache accesses | Gauge |

## Dashboard Views

### 1. Search Performance Overview

This dashboard provides a high-level view of search system performance:

- **Search Rate**: Requests per second (RPS) for both regular and filtered searches
- **Latency Distribution**: 50th, 95th, and 99th percentile latencies
- **Error Rate**: Percentage of failed search requests
- **Active Searches**: Current number of concurrent searches

### 2. Filter Effectiveness Dashboard

This dashboard focuses on filter performance:

- **Filter Application Time**: Average and percentile times for filter application
- **Filter Reduction Rate**: Percentage reduction in candidate vectors after filtering
- **Cache Hit Ratio**: Effectiveness of filter caching
- **Popular Filters**: Most frequently used filter combinations

### 3. Resource Utilization

This dashboard monitors system resource usage:

- **CPU Usage**: Per-core utilization during search operations
- **Memory Usage**: Memory consumption trends
- **Disk I/O**: Storage read/write patterns
- **Network Usage**: Data transfer rates

## Alerting Rules

### Performance Alerts

1. **High Latency Alert**
   - Condition: 95th percentile search latency > 100ms for 5 minutes
   - Severity: Warning
   - Action: Notify performance team

2. **High Error Rate Alert**
   - Condition: Search error rate > 1% for 2 minutes
   - Severity: Critical
   - Action: Page on-call engineer

3. **Resource Exhaustion Alert**
   - Condition: Memory usage > 90% for 10 minutes
   - Severity: Critical
   - Action: Scale up resources or page operations team

### Filter-Specific Alerts

1. **Slow Filter Application Alert**
   - Condition: Average filter application time > 50ms for 3 minutes
   - Severity: Warning
   - Action: Notify optimization team

2. **Low Cache Hit Ratio Alert**
   - Condition: Filter cache hit ratio < 70% for 15 minutes
   - Severity: Warning
   - Action: Notify performance team to investigate caching strategy

## Monitoring Endpoints

### Prometheus Metrics Endpoint

```
GET /metrics
```

Returns all metrics in Prometheus exposition format:

```prometheus
# HELP search_requests_total Total number of search requests
# TYPE search_requests_total counter
search_requests_total 12345

# HELP filtered_search_request_duration_seconds Time spent processing filtered search requests
# TYPE filtered_search_request_duration_seconds histogram
filtered_search_request_duration_seconds_bucket{le="0.005"} 100
filtered_search_request_duration_seconds_bucket{le="0.01"} 250
filtered_search_request_duration_seconds_bucket{le="0.025"} 400
filtered_search_request_duration_seconds_bucket{le="0.05"} 500
filtered_search_request_duration_seconds_bucket{le="+Inf"} 500
filtered_search_request_duration_seconds_sum 12.345
filtered_search_request_duration_seconds_count 500

# HELP active_filtered_searches Number of currently active filtered searches
# TYPE active_filtered_searches gauge
active_filtered_searches 5
```

### Health Check Endpoint

```
GET /health
```

Returns system health status:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00Z",
  "components": {
    "database": "healthy",
    "cache": "healthy",
    "metrics": "healthy"
  },
  "metrics": {
    "search_rps": 125.5,
    "avg_latency_ms": 25.3,
    "error_rate_percent": 0.1
  }
}
```

## Integration with External Monitoring Systems

### Grafana Dashboard

Pre-built Grafana dashboards are available for:

1. **Search Performance Dashboard**: Visualizes search latency, throughput, and error rates
2. **Filter Analytics Dashboard**: Shows filter effectiveness, cache performance, and popular filters
3. **System Resources Dashboard**: Monitors CPU, memory, disk, and network usage

### Alertmanager Integration

Alerts can be configured to send notifications to:

- Slack channels
- Email addresses
- PagerDuty
- Webhooks for custom integrations

## Best Practices for Monitoring

1. **Set Meaningful Thresholds**: Configure alerts based on historical performance data and business requirements
2. **Monitor Trends**: Look for gradual performance degradation, not just sudden failures
3. **Correlate Metrics**: Combine search metrics with system resource metrics to identify root causes
4. **Regular Review**: Periodically review and adjust alert thresholds based on changing usage patterns
5. **Capacity Planning**: Use metrics to predict and plan for future resource needs

## Troubleshooting Common Issues

### High Search Latency

1. Check if there's a correlation with high filter application times
2. Examine cache hit ratios to determine if caching is effective
3. Review system resource usage to identify bottlenecks
4. Analyze query patterns to identify expensive filter combinations

### Low Filter Cache Hit Ratio

1. Review frequently used filters to optimize caching strategy
2. Increase cache size if memory allows
3. Analyze filter patterns to identify cache-unfriendly queries
4. Consider pre-warming cache for common filter combinations

### Resource Exhaustion

1. Scale up or out based on resource usage patterns
2. Optimize queries to reduce resource consumption
3. Implement rate limiting to prevent resource overload
4. Add more granular monitoring to identify resource-intensive operations