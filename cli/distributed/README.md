# JadeVectorDB Distributed Cluster Management CLI

A specialized command-line tool for managing JadeVectorDB distributed clusters. Provides operations for cluster administration, node management, shard operations, and diagnostics.

## Overview

The Distributed CLI (`cluster_cli.py`) is designed specifically for managing JadeVectorDB in distributed mode with multiple nodes. It complements the standard CLI tools by focusing on cluster-wide operations.

## Features

- ✅ **Cluster Status** - Monitor overall cluster health and configuration
- ✅ **Node Management** - Add, remove, and monitor cluster nodes
- ✅ **Shard Operations** - Manage data shards and migrations
- ✅ **Diagnostics** - Detailed health checks and troubleshooting
- ✅ **Metrics** - Performance and resource monitoring
- ✅ **JSON & Table Output** - Flexible output formats for automation

## Installation

The distributed CLI is a Python script with minimal dependencies:

```bash
# Requires Python 3.8+ and requests library
pip install requests

# Or install from requirements
pip install -r requirements.txt
```

## Usage

### Basic Syntax

```bash
python cli/distributed/cluster_cli.py --host <host> --port <port> <command> [options]
```

### Global Options

- `--host <host>` - Cluster master node hostname (default: localhost)
- `--port <port>` - API port (default: 8080)
- `--format <format>` - Output format: json, table, or compact (default: table)

## Commands

### Cluster Operations

#### Get Cluster Status

Display overall cluster status including nodes, shards, and health:

```bash
python cluster_cli.py --host localhost --port 8080 status
```

**Output:**
```
Cluster Status
==============
Master Node: node-master-001
Total Nodes: 3
Active Nodes: 3
Total Shards: 12
Status: HEALTHY
Replication Factor: 2
```

#### Get Cluster Diagnostics

Comprehensive diagnostics including all subsystems:

```bash
python cluster_cli.py --host localhost --port 8080 diagnostics
```

**Output includes:**
- Cluster topology
- Node health status
- Shard distribution
- Replication status
- Performance metrics
- Error logs

#### Get Cluster Metrics

Retrieve performance and resource metrics:

```bash
python cluster_cli.py --host localhost --port 8080 metrics
```

### Node Management

#### List All Nodes

```bash
python cluster_cli.py --host localhost --port 8080 nodes
```

**Output:**
```
┌──────────────────┬─────────┬────────────────────┬──────────┬──────────┐
│ Node ID          │ Role    │ Address            │ Status   │ Shards   │
├──────────────────┼─────────┼────────────────────┼──────────┼──────────┤
│ node-master-001  │ MASTER  │ 192.168.1.10:8080  │ HEALTHY  │ 4        │
│ node-worker-001  │ WORKER  │ 192.168.1.11:8080  │ HEALTHY  │ 4        │
│ node-worker-002  │ WORKER  │ 192.168.1.12:8080  │ HEALTHY  │ 4        │
└──────────────────┴─────────┴────────────────────┴──────────┴──────────┘
```

#### Add a Node

```bash
python cluster_cli.py --host localhost --port 8080 add-node \
  --node-id node-worker-003 \
  --address 192.168.1.13:8080
```

#### Remove a Node

```bash
python cluster_cli.py --host localhost --port 8080 remove-node --node-id node-worker-003
```

**Note:** Removing a node triggers automatic shard redistribution.

#### Get Node Health

Check health of a specific node:

```bash
python cluster_cli.py --host localhost --port 8080 node-health --node-id node-worker-001
```

### Shard Operations

#### List All Shards

```bash
python cluster_cli.py --host localhost --port 8080 shards
```

**Output:**
```
┌──────────┬─────────────────┬─────────────┬────────────────┬──────────┐
│ Shard ID │ Primary Node    │ Replicas    │ Vector Count   │ Status   │
├──────────┼─────────────────┼─────────────┼────────────────┼──────────┤
│ shard-00 │ node-worker-001 │ 2           │ 125,432        │ HEALTHY  │
│ shard-01 │ node-worker-002 │ 2           │ 118,901        │ HEALTHY  │
│ shard-02 │ node-worker-001 │ 2           │ 131,245        │ HEALTHY  │
└──────────┴─────────────────┴─────────────┴────────────────┴──────────┘
```

#### Get Shard Status

```bash
python cluster_cli.py --host localhost --port 8080 shard-status --shard-id shard-00
```

#### Migrate a Shard

Move a shard from one node to another:

```bash
python cluster_cli.py --host localhost --port 8080 migrate-shard \
  --shard-id shard-00 \
  --from-node node-worker-001 \
  --to-node node-worker-003 \
  --strategy LIVE_COPY
```

**Migration Strategies:**
- `LIVE_COPY` - Copy data while serving queries (zero downtime)
- `SNAPSHOT` - Take snapshot then copy (brief downtime)
- `INCREMENTAL` - Incremental sync with final switchover

## Output Formats

### Table Format (Default)

Human-readable tables with aligned columns:

```bash
python cluster_cli.py status --format table
```

### JSON Format

Machine-readable JSON for automation:

```bash
python cluster_cli.py status --format json
```

```json
{
  "master_node": "node-master-001",
  "total_nodes": 3,
  "active_nodes": 3,
  "total_shards": 12,
  "status": "HEALTHY",
  "replication_factor": 2
}
```

### Compact Format

Condensed output for quick checks:

```bash
python cluster_cli.py status --format compact
```

```
HEALTHY | 3 nodes | 12 shards | master: node-master-001
```

## Examples

### Monitor Cluster Health

```bash
#!/bin/bash
# health-check.sh - Monitor cluster and alert on issues

while true; do
  STATUS=$(python cluster_cli.py status --format json | jq -r '.status')

  if [ "$STATUS" != "HEALTHY" ]; then
    echo "ALERT: Cluster is $STATUS" | mail -s "Cluster Alert" admin@example.com

    # Get diagnostics
    python cluster_cli.py diagnostics --format json > cluster-diagnostics.json
  fi

  sleep 60
done
```

### Auto-Scaling Script

```bash
#!/bin/bash
# auto-scale.sh - Add nodes based on load

LOAD=$(python cluster_cli.py metrics --format json | jq -r '.cpu_usage')

if (( $(echo "$LOAD > 80" | bc -l) )); then
  echo "High load detected: ${LOAD}%"

  # Add new worker node
  NEW_NODE="node-worker-$(date +%s)"
  python cluster_cli.py add-node \
    --node-id "$NEW_NODE" \
    --address "auto-scaled-node:8080"

  echo "Added new node: $NEW_NODE"
fi
```

### Shard Rebalancing

```bash
#!/bin/bash
# rebalance-shards.sh - Redistribute shards evenly

# Get current shard distribution
python cluster_cli.py shards --format json > current-shards.json

# Identify overloaded nodes and migrate shards
# (Custom logic based on your requirements)

python cluster_cli.py migrate-shard \
  --shard-id shard-05 \
  --from-node overloaded-node \
  --to-node underutilized-node \
  --strategy LIVE_COPY
```

### Cluster Backup Status

```bash
# Get all node health and shard status for backup verification
python cluster_cli.py diagnostics --format json > cluster-backup-$(date +%Y%m%d).json

# Verify all shards have minimum replica count
python cluster_cli.py shards --format json | jq '[.[] | select(.replicas < 2)]'
```

## API Endpoints Used

The Distributed CLI interacts with these API endpoints:

| Command | API Endpoint | Method |
|---------|--------------|--------|
| `status` | `/api/v1/cluster/status` | GET |
| `diagnostics` | `/api/v1/cluster/diagnostics` | GET |
| `metrics` | `/api/v1/cluster/metrics` | GET |
| `nodes` | `/api/v1/cluster/nodes` | GET |
| `add-node` | `/api/v1/cluster/nodes` | POST |
| `remove-node` | `/api/v1/cluster/nodes/{id}` | DELETE |
| `node-health` | `/api/v1/cluster/nodes/{id}/health` | GET |
| `shards` | `/api/v1/cluster/shards` | GET |
| `shard-status` | `/api/v1/cluster/shards/{id}` | GET |
| `migrate-shard` | `/api/v1/cluster/shards/{id}/migrate` | POST |

## Automation Integration

### Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cluster-health-check
spec:
  schedule: "*/5 * * * *"  # Every 5 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: health-checker
            image: python:3.9
            command:
            - python
            - /scripts/cluster_cli.py
            - --host
            - jadevectordb-master
            - --port
            - "8080"
            - diagnostics
            - --format
            - json
          restartPolicy: OnFailure
```

### Prometheus Exporter

```python
# prometheus_exporter.py - Export cluster metrics to Prometheus

import time
from cluster_cli import ClusterCLI
from prometheus_client import start_http_server, Gauge

# Create metrics
cluster_nodes = Gauge('jadevectordb_cluster_nodes', 'Total cluster nodes')
cluster_shards = Gauge('jadevectordb_cluster_shards', 'Total shards')

cli = ClusterCLI(host='localhost', port=8080)

def collect_metrics():
    status = cli.cluster_status()
    cluster_nodes.set(status.get('total_nodes', 0))
    cluster_shards.set(status.get('total_shards', 0))

if __name__ == '__main__':
    start_http_server(9091)
    while True:
        collect_metrics()
        time.sleep(15)
```

## Troubleshooting

### Connection Failed

```bash
# Verify master node is accessible
curl http://localhost:8080/health

# Check network connectivity
ping <master-host>
telnet <master-host> 8080
```

### Node Not Responding

```bash
# Get detailed node health
python cluster_cli.py node-health --node-id <node-id>

# Check cluster diagnostics
python cluster_cli.py diagnostics | grep <node-id>
```

### Shard Migration Stuck

```bash
# Check migration status
python cluster_cli.py shard-status --shard-id <shard-id>

# View cluster diagnostics for errors
python cluster_cli.py diagnostics --format json | jq '.errors'
```

## Performance Tips

1. **Use JSON format** for automation scripts (faster parsing)
2. **Filter specific data** instead of fetching everything
3. **Cache cluster status** for frequent health checks
4. **Use compact format** for simple monitoring
5. **Batch operations** when managing multiple nodes

## Comparison with Other CLIs

| Feature | Distributed CLI | Standard CLI |
|---------|----------------|--------------|
| **Purpose** | Cluster management | Database operations |
| **Scope** | Multi-node operations | Single-instance operations |
| **Commands** | 11 cluster-specific | 10+ database operations |
| **Output Formats** | JSON, table, compact | JSON only |
| **Target Users** | DevOps, SRE | Developers, data scientists |

## Related Documentation

- [Main CLI Documentation](../README.md)
- [Python CLI](../python/README.md)
- [Distributed Deployment Guide](../../docs/distributed_deployment_guide.md)
- [Kubernetes Deployment](../../k8s/README.md)

## Support

For cluster management issues:
- Check [Troubleshooting](#troubleshooting) section
- See [Distributed Deployment Guide](../../docs/distributed_deployment_guide.md)
- GitHub Issues: [JadeVectorDB Issues](https://github.com/Jade-biz-1/JadeVectorDB/issues)
