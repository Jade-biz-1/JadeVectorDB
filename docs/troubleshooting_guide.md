# JadeVectorDB Troubleshooting Guide

## Common Issues and Solutions

### Issue: Service Won't Start

**Symptoms**: Container exits immediately, "JadeVectorDB" not running

**Common Causes**:
1. Missing JWT_SECRET in production
2. Port 8080 already in use
3. Database file permissions incorrect

**Solutions**:
```bash
# Check logs
docker logs jadevectordb

# Verify JWT_SECRET is set
echo $JWT_SECRET

# Check port availability
netstat -tlnp | grep 8080

# Fix permissions
chmod 600 /data/*.db
chown jadevectordb:jadevectordb /data/*.db
```

---

### Issue: High Response Times

**Symptoms**: API responses > 1 second, metrics show high latency

**Diagnosis**:
```bash
# Check database operation times
curl http://localhost:8080/metrics | grep db_operation_duration

# Check permission cache hit rate
curl http://localhost:8080/metrics | grep permission_cache
```

**Solutions**:
1. Enable permission caching in `performance.json`
2. Increase connection pool size
3. Add database indexes
4. Review slow query logs

---

### Issue: Authentication Failures

**Symptoms**: Users unable to login, 401 errors

**Diagnosis**:
```bash
# Check failed login count
curl http://localhost:8080/metrics | grep failed_logins

# Review audit logs
tail -f /var/log/jadevectordb/audit.log
```

**Solutions**:
- Verify JWT_SECRET hasn't changed
- Check if account is locked (too many failures)
- Unblock IP if needed
- Reset password via forgot-password endpoint

---

### Issue: Rate Limiting Errors

**Symptoms**: HTTP 429 responses, "rate limit exceeded"

**Diagnosis**:
```bash
curl http://localhost:8080/metrics | grep rate_limit_exceeded
```

**Solutions**:
- Wait for rate limit window to expire (check Retry-After header)
- Increase rate limits in `production.json`
- Implement exponential backoff in client
- Use API keys for higher limits

---

### Issue: Database Connection Errors

**Symptoms**: "database connection failed", circuit breaker open

**Diagnosis**:
```bash
# Check database health
curl http://localhost:8080/health/db

# Check connection errors
curl http://localhost:8080/metrics | grep db_connection_errors
```

**Solutions**:
1. Verify database file exists and is readable
2. Check disk space: `df -h /data`
3. Restart service to reset circuit breaker
4. Check for database locks: `lsof | grep jadevectordb.db`

---

### Issue: High Memory Usage

**Symptoms**: OOM kills, memory alerts

**Diagnosis**:
```bash
# Check container memory
docker stats jadevectordb

# Check cache size
curl http://localhost:8080/metrics | grep cache_size
```

**Solutions**:
- Reduce `permission_cache_size` in configuration
- Increase container memory limit
- Monitor for memory leaks
- Restart service to clear cache

---

### Issue: IP Blocked Unexpectedly

**Symptoms**: "Account temporarily locked", HTTP 403

**Diagnosis**:
```bash
curl http://localhost:8080/metrics | grep ip_blocks_total
```

**Solutions**:
```bash
# Admin unblock
curl -X DELETE http://localhost:8080/v1/admin/ip-blocks/1.2.3.4 \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Or wait 1 hour for automatic unblock
```

---

## Error Code Reference

- **400**: Invalid request (check request body)
- **401**: Authentication required (missing/invalid token)
- **403**: IP blocked or permission denied
- **404**: Resource not found
- **429**: Rate limit exceeded (check Retry-After)
- **500**: Internal server error (check logs)
- **503**: Service unavailable (database down, circuit breaker open)

---

## Debug Mode

Enable debug logging:
```bash
export JADEVECTORDB_LOG_LEVEL=debug
docker restart jadevectordb
```

---

## Getting Help

1. Check logs: `docker logs jadevectordb`
2. Review metrics: `curl http://localhost:8080/metrics`
3. Check health: `curl http://localhost:8080/health/db`
4. Contact support with:
   - Error messages
   - Log excerpts
   - Metrics output
   - Configuration (redact secrets!)
