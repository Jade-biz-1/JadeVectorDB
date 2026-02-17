# Exercise 5: Advanced Workflows

## Learning Objectives

By the end of this exercise, you will be able to:
- Monitor JadeVectorDB health and performance
- Implement backup and restore procedures
- Automate routine maintenance tasks
- Create production-ready CLI scripts
- Handle errors and implement retry logic
- Set up alerting and monitoring
- Manage database lifecycle (archival, cleanup)

## Prerequisites

- Completed Exercises 1-4
- Understanding of shell scripting
- Basic understanding of cron jobs (optional)
- JadeVectorDB running at `http://localhost:8080`

## Introduction

This exercise covers production-ready patterns and workflows that go beyond basic operations. You'll learn how to:
- Build robust, error-handling scripts
- Monitor system health
- Automate routine tasks
- Implement disaster recovery procedures

## Exercise Steps

### Step 1: Health Monitoring Script

**Task:** Create a script that continuously monitors JadeVectorDB health.

```bash
#!/bin/bash
# health_monitor.sh - Monitor JadeVectorDB health

INTERVAL=60  # Check every 60 seconds
LOG_FILE="health_monitor.log"

monitor_health() {
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  # Check health endpoint
  response=$(jade-db health 2>&1)
  status=$?

  if [ $status -eq 0 ]; then
    echo "[$timestamp] ✓ HEALTHY: $response" >> "$LOG_FILE"
  else
    echo "[$timestamp] ✗ UNHEALTHY: $response" >> "$LOG_FILE"

    # Send alert (example: write to alert file)
    echo "[$timestamp] Database unhealthy!" >> alerts.log

    # Could also send email, Slack notification, etc.
    # notify_admin "JadeVectorDB is unhealthy"
  fi
}

# Main monitoring loop
echo "Starting health monitoring..."
while true; do
  monitor_health
  sleep $INTERVAL
done
```

**Usage:**
```bash
chmod +x health_monitor.sh
./health_monitor.sh &  # Run in background
```

### Step 2: Performance Metrics Collection

**Task:** Collect and log performance metrics.

```bash
#!/bin/bash
# collect_metrics.sh - Collect performance metrics

METRICS_FILE="metrics.csv"

# Create CSV header if file doesn't exist
if [ ! -f "$METRICS_FILE" ]; then
  echo "timestamp,query_time_ms,result_count,database_id" > "$METRICS_FILE"
fi

collect_metrics() {
  db_id=$1
  query_vector=$2
  top_k=${3:-10}

  timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  # Measure search time
  start=$(date +%s%N)
  results=$(jade-db search \
    --database-id "$db_id" \
    --query-vector "$query_vector" \
    --top-k $top_k 2>&1)
  end=$(date +%s%N)

  # Calculate duration in milliseconds
  duration=$(( (end - start) / 1000000 ))

  # Count results
  result_count=$(echo "$results" | jq 'length' 2>/dev/null || echo "0")

  # Log metrics
  echo "$timestamp,$duration,$result_count,$db_id" >> "$METRICS_FILE"

  echo "Collected metrics: ${duration}ms for $result_count results"
}

# Example usage
collect_metrics "my_database" "[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]" 10
```

### Step 3: Backup Script

**Task:** Create a comprehensive backup script.

```bash
#!/bin/bash
# backup_database.sh - Backup database vectors and metadata

BACKUP_DIR="backups"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

backup_database() {
  db_id=$1
  backup_file="$BACKUP_DIR/${db_id}_${TIMESTAMP}.json"

  echo "Backing up database: $db_id"

  # Create backup directory if it doesn't exist
  mkdir -p "$BACKUP_DIR"

  # Get database info
  db_info=$(jade-db get-db --database-id "$db_id" 2>/dev/null)
  if [ $? -ne 0 ]; then
    echo "Error: Database $db_id not found"
    return 1
  fi

  # Export all vectors (if list-vectors endpoint exists)
  # This is a simplified example - adapt to your API
  vectors=$(jade-db list-vectors --database-id "$db_id" 2>/dev/null)

  # Create backup JSON
  cat > "$backup_file" << EOF
{
  "database_id": "$db_id",
  "backup_timestamp": "$TIMESTAMP",
  "database_info": $db_info,
  "vectors": $vectors
}
EOF

  # Compress backup
  gzip "$backup_file"

  echo "✓ Backup created: ${backup_file}.gz"
  echo "  Size: $(du -h ${backup_file}.gz | cut -f1)"
}

# Backup multiple databases
for db in production_db staging_db; do
  backup_database "$db"
done

# Clean up old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.json.gz" -mtime +7 -delete
echo "✓ Cleaned up old backups"
```

### Step 4: Restore Script

**Task:** Create a restore script for disaster recovery.

```bash
#!/bin/bash
# restore_database.sh - Restore database from backup

restore_database() {
  backup_file=$1

  if [ ! -f "$backup_file" ]; then
    echo "Error: Backup file not found: $backup_file"
    return 1
  fi

  echo "Restoring from backup: $backup_file"

  # Decompress if needed
  if [[ $backup_file == *.gz ]]; then
    gunzip -c "$backup_file" > temp_backup.json
    backup_file="temp_backup.json"
  fi

  # Extract database info
  db_id=$(jq -r '.database_id' "$backup_file")
  dimension=$(jq -r '.database_info.dimension' "$backup_file")
  index_type=$(jq -r '.database_info.index_type' "$backup_file")

  echo "  Database ID: $db_id"
  echo "  Dimension: $dimension"
  echo "  Index Type: $index_type"

  # Recreate database
  echo "Recreating database..."
  jade-db create-db \
    --name "$db_id" \
    --dimension "$dimension" \
    --index-type "$index_type" \
    2>/dev/null

  # Restore vectors
  echo "Restoring vectors..."
  jq -c '.vectors[]' "$backup_file" | while read vector; do
    id=$(echo "$vector" | jq -r '.id')
    values=$(echo "$vector" | jq -c '.values')
    metadata=$(echo "$vector" | jq -c '.metadata')

    jade-db store \
      --database-id "$db_id" \
      --vector-id "$id" \
      --values "$values" \
      --metadata "$metadata" \
      2>/dev/null

    echo -n "."
  done

  echo ""
  echo "✓ Restore completed"

  # Cleanup
  [ -f "temp_backup.json" ] && rm temp_backup.json
}

# Usage
restore_database "backups/production_db_20250115_120000.json.gz"
```

### Step 5: Automated Maintenance Script

**Task:** Create a maintenance script for routine cleanup.

```bash
#!/bin/bash
# maintenance.sh - Automated database maintenance

MAINTENANCE_LOG="maintenance.log"

log_message() {
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$timestamp] $1" | tee -a "$MAINTENANCE_LOG"
}

# Check system health
check_health() {
  log_message "Checking system health..."

  if jade-db health &>/dev/null; then
    log_message "✓ System is healthy"
    return 0
  else
    log_message "✗ System is unhealthy - aborting maintenance"
    return 1
  fi
}

# Clean up test databases
cleanup_test_databases() {
  log_message "Cleaning up test databases..."

  # List all databases
  databases=$(jade-db list-dbs 2>/dev/null | jq -r '.[].id' 2>/dev/null)

  for db in $databases; do
    # Delete test/temporary databases
    if [[ $db == test_* ]] || [[ $db == temp_* ]]; then
      log_message "  Deleting: $db"
      jade-db delete-db --database-id "$db" 2>/dev/null
    fi
  done

  log_message "✓ Cleanup completed"
}

# Vacuum/optimize databases
optimize_databases() {
  log_message "Optimizing databases..."

  # If your DB supports optimization/vacuum
  for db in production_db staging_db; do
    log_message "  Optimizing: $db"
    # jade-db optimize --database-id "$db" 2>/dev/null
  done

  log_message "✓ Optimization completed"
}

# Generate statistics report
generate_report() {
  log_message "Generating statistics report..."

  report_file="reports/report_$(date +%Y%m%d).txt"
  mkdir -p reports

  {
    echo "JadeVectorDB Statistics Report"
    echo "Generated: $(date)"
    echo "================================"
    echo ""

    # System status
    echo "System Status:"
    jade-db status 2>/dev/null | jq .

    echo ""
    echo "Database Statistics:"

    # List all databases with stats
    jade-db list-dbs 2>/dev/null | jq -r '.[].id' | while read db; do
      echo "  $db:"
      jade-db get-db --database-id "$db" 2>/dev/null
    done
  } > "$report_file"

  log_message "✓ Report generated: $report_file"
}

# Main maintenance workflow
main() {
  log_message "========== Starting Maintenance =========="

  if ! check_health; then
    exit 1
  fi

  cleanup_test_databases
  optimize_databases
  generate_report

  log_message "========== Maintenance Complete =========="
}

# Run maintenance
main
```

**Schedule with cron:**
```bash
# Run maintenance every day at 2 AM
0 2 * * * /path/to/maintenance.sh
```

### Step 6: Error Handling and Retry Logic

**Task:** Create robust script with retry logic.

```bash
#!/bin/bash
# robust_import.sh - Import with retry logic

MAX_RETRIES=3
RETRY_DELAY=5

import_with_retry() {
  db_id=$1
  vector_id=$2
  values=$3
  metadata=$4
  retry_count=0

  while [ $retry_count -lt $MAX_RETRIES ]; do
    # Attempt import
    if jade-db store \
      --database-id "$db_id" \
      --vector-id "$vector_id" \
      --values "$values" \
      --metadata "$metadata" \
      2>/dev/null; then
      echo "✓ Successfully imported: $vector_id"
      return 0
    else
      retry_count=$((retry_count + 1))
      if [ $retry_count -lt $MAX_RETRIES ]; then
        echo "✗ Failed to import $vector_id, retrying ($retry_count/$MAX_RETRIES)..."
        sleep $RETRY_DELAY
      else
        echo "✗ Failed to import $vector_id after $MAX_RETRIES attempts"
        echo "$vector_id" >> failed_imports.log
        return 1
      fi
    fi
  done
}

# Bulk import with error handling
bulk_import() {
  db_id=$1
  data_file=$2

  success_count=0
  failure_count=0

  cat "$data_file" | jq -c '.[]' | while read item; do
    id=$(echo "$item" | jq -r '.id')
    values=$(echo "$item" | jq -c '.embedding')
    metadata=$(echo "$item" | jq -c 'del(.id, .embedding)')

    if import_with_retry "$db_id" "$id" "$values" "$metadata"; then
      ((success_count++))
    else
      ((failure_count++))
    fi

    # Progress indicator
    total=$((success_count + failure_count))
    echo "Progress: $total items processed ($success_count succeeded, $failure_count failed)"
  done

  echo "========== Import Summary =========="
  echo "Successful: $success_count"
  echo "Failed: $failure_count"
  echo "See failed_imports.log for details"
}

# Usage
bulk_import "my_database" "../../sample-data/products.json"
```

### Step 7: Alerting System

**Task:** Create an alerting system for critical events.

```bash
#!/bin/bash
# alerting.sh - Monitor and alert on critical events

ALERT_EMAIL="admin@example.com"
ALERT_THRESHOLD_MS=1000  # Alert if search takes > 1s

send_alert() {
  severity=$1
  message=$2
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  # Log alert
  echo "[$timestamp] [$severity] $message" >> alerts.log

  # Send email (requires mail command)
  # echo "$message" | mail -s "JadeVectorDB Alert: $severity" "$ALERT_EMAIL"

  # Send to Slack (requires curl and webhook)
  # curl -X POST -H 'Content-type: application/json' \
  #   --data "{\"text\":\"[$severity] $message\"}" \
  #   "YOUR_SLACK_WEBHOOK_URL"

  # For now, just print
  echo "ALERT [$severity]: $message"
}

# Monitor search performance
monitor_search_performance() {
  db_id=$1
  query=$2

  start=$(date +%s%N)
  jade-db search \
    --database-id "$db_id" \
    --query-vector "$query" \
    --top-k 10 \
    &>/dev/null
  end=$(date +%s%N)

  duration=$(( (end - start) / 1000000 ))

  if [ $duration -gt $ALERT_THRESHOLD_MS ]; then
    send_alert "WARNING" \
      "Slow search detected: ${duration}ms (threshold: ${ALERT_THRESHOLD_MS}ms)"
  fi
}

# Monitor disk space
monitor_disk_space() {
  usage=$(df -h /data | tail -1 | awk '{print $5}' | sed 's/%//')

  if [ $usage -gt 90 ]; then
    send_alert "CRITICAL" \
      "Disk space critically low: ${usage}% used"
  elif [ $usage -gt 80 ]; then
    send_alert "WARNING" \
      "Disk space running low: ${usage}% used"
  fi
}

# Monitor API health
monitor_api_health() {
  if ! jade-db health &>/dev/null; then
    send_alert "CRITICAL" \
      "JadeVectorDB API is down or unresponsive"
  fi
}

# Main monitoring loop
while true; do
  monitor_api_health
  monitor_disk_space
  monitor_search_performance "production_db" "[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]"

  sleep 300  # Check every 5 minutes
done
```

### Step 8: Production Deployment Script

**Task:** Create a complete deployment script.

```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e  # Exit on error

ENVIRONMENT=${1:-production}
CONFIG_FILE="config/${ENVIRONMENT}.conf"

# Load configuration
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Configuration file not found: $CONFIG_FILE"
  exit 1
fi

source "$CONFIG_FILE"

deploy() {
  echo "========================================="
  echo "Deploying to: $ENVIRONMENT"
  echo "Database URL: $JADE_DB_URL"
  echo "========================================="

  # 1. Health check
  echo "Step 1: Checking health..."
  if ! jade-db --url "$JADE_DB_URL" health &>/dev/null; then
    echo "Error: Database is not healthy"
    exit 1
  fi

  # 2. Backup existing data
  echo "Step 2: Creating backup..."
  ./backup_database.sh "$DATABASE_ID"

  # 3. Create/update database schema
  echo "Step 3: Updating database schema..."
  jade-db --url "$JADE_DB_URL" create-db \
    --name "$DATABASE_ID" \
    --dimension "$DIMENSION" \
    --index-type "$INDEX_TYPE" \
    2>/dev/null || echo "Database already exists"

  # 4. Import data
  echo "Step 4: Importing data..."
  ./robust_import.sh "$DATABASE_ID" "$DATA_FILE"

  # 5. Verify deployment
  echo "Step 5: Verifying deployment..."
  db_info=$(jade-db --url "$JADE_DB_URL" get-db \
    --database-id "$DATABASE_ID")
  echo "  Database info: $db_info"

  # 6. Run smoke tests
  echo "Step 6: Running smoke tests..."
  ./smoke_tests.sh "$DATABASE_ID"

  echo "========================================="
  echo "Deployment complete!"
  echo "========================================="
}

# Run deployment
deploy
```

**Configuration file (config/production.conf):**
```bash
# Production configuration
JADE_DB_URL="https://jadevectordb.production.example.com"
JADE_DB_API_KEY="prod_api_key_here"
DATABASE_ID="production_db"
DIMENSION=128
INDEX_TYPE="HNSW"
DATA_FILE="data/production_vectors.json"
```

### Step 9: Smoke Tests

**Task:** Create smoke tests to verify deployment.

```bash
#!/bin/bash
# smoke_tests.sh - Verify basic functionality

DATABASE_ID=$1

run_test() {
  test_name=$1
  test_command=$2

  echo -n "  Testing $test_name... "

  if eval "$test_command" &>/dev/null; then
    echo "✓ PASS"
    return 0
  else
    echo "✗ FAIL"
    return 1
  fi
}

echo "Running smoke tests for: $DATABASE_ID"
echo "======================================="

# Test 1: Database exists
run_test "Database exists" \
  "jade-db get-db --database-id $DATABASE_ID"

# Test 2: Can store vector
run_test "Can store vector" \
  "jade-db store --database-id $DATABASE_ID \
    --vector-id test_vector \
    --values '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]' \
    --metadata '{\"test\": true}'"

# Test 3: Can retrieve vector
run_test "Can retrieve vector" \
  "jade-db retrieve --database-id $DATABASE_ID --vector-id test_vector"

# Test 4: Can search
run_test "Can perform search" \
  "jade-db search --database-id $DATABASE_ID \
    --query-vector '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]' \
    --top-k 5"

# Test 5: Can delete vector
run_test "Can delete vector" \
  "jade-db delete --database-id $DATABASE_ID --vector-id test_vector"

echo "======================================="
echo "Smoke tests complete"
```

### Step 10: Production Workflow Automation

**Task:** Create a master automation script.

```bash
#!/bin/bash
# automation.sh - Master automation orchestrator

# Run daily at 2 AM via cron:
# 0 2 * * * /path/to/automation.sh daily

TASK=$1

daily_tasks() {
  echo "Running daily tasks..."

  # Backup all databases
  ./backup_database.sh production_db
  ./backup_database.sh staging_db

  # Run maintenance
  ./maintenance.sh

  # Generate reports
  ./generate_report.sh

  # Clean up old logs
  find logs/ -name "*.log" -mtime +30 -delete
}

weekly_tasks() {
  echo "Running weekly tasks..."

  # Full system backup
  ./full_backup.sh

  # Performance analysis
  ./analyze_performance.sh

  # Capacity planning
  ./capacity_planning.sh
}

monthly_tasks() {
  echo "Running monthly tasks..."

  # Archive old backups
  ./archive_backups.sh

  # Security audit
  ./security_audit.sh

  # Database optimization
  ./optimize_all_databases.sh
}

case $TASK in
  daily)
    daily_tasks
    ;;
  weekly)
    weekly_tasks
    ;;
  monthly)
    monthly_tasks
    ;;
  *)
    echo "Usage: $0 {daily|weekly|monthly}"
    exit 1
    ;;
esac
```

## Verification

Test your production workflows:

```bash
# Test backup/restore
bash exercises/05-advanced-workflows/test_backup_restore.sh

# Test error handling
bash exercises/05-advanced-workflows/test_error_handling.sh

# Test monitoring
bash exercises/05-advanced-workflows/test_monitoring.sh
```

## Best Practices Summary

### 1. Always Log Everything
```bash
# Redirect both stdout and stderr to log file
./script.sh >> logs/script.log 2>&1
```

### 2. Use Set -e for Critical Scripts
```bash
#!/bin/bash
set -e  # Exit immediately on error
set -u  # Error on undefined variables
set -o pipefail  # Catch errors in pipelines
```

### 3. Implement Graceful Shutdown
```bash
trap cleanup EXIT
cleanup() {
  echo "Cleaning up..."
  # Remove temp files, restore state, etc.
}
```

### 4. Use Configuration Files
```bash
# Don't hardcode values
source config/production.conf
```

### 5. Test Before Production
```bash
# Always test in staging first
./deploy.sh staging
# If successful, deploy to production
./deploy.sh production
```

## Next Steps

Congratulations! You've completed all CLI tutorials. Next:

1. **Explore the Web Tutorial** for visual understanding
2. **Read Production Documentation** for deployment best practices
3. **Set up Monitoring** with Prometheus and Grafana
4. **Implement CI/CD** for automated deployments

## Resources

- **Cron Tutorial:** `man 5 crontab`
- **Shell Scripting Guide:** Advanced Bash-Scripting Guide
- **Monitoring Tools:** Prometheus, Grafana, Datadog
- **Backup Strategies:** 3-2-1 Backup Rule
