# Quick Start: Enable Distributed Features

## ⚠️ Prerequisites

Before enabling distributed features:
- Ensure you have 3+ nodes available
- Network connectivity between nodes on cluster port (default: 9080)
- Completed Phase 1 single-node testing
- Read `docs/distributed_services_api.md` for architecture details

## 🚀 Three Ways to Enable

### Option 1: Environment Variables (Recommended for Testing)

```bash
# Set environment variables
export JADEVECTORDB_ENABLE_SHARDING=true
export JADEVECTORDB_ENABLE_REPLICATION=true
export JADEVECTORDB_ENABLE_CLUSTERING=true
export JADEVECTORDB_NUM_SHARDS=16
export JADEVECTORDB_REPLICATION_FACTOR=3
export JADEVECTORDB_CLUSTER_HOST=0.0.0.0
export JADEVECTORDB_CLUSTER_PORT=9080
export JADEVECTORDB_SEED_NODES=node1:9080,node2:9080,node3:9080

# Run the application
cd backend/build
./jadevectordb
```

### Option 2: JSON Configuration (Recommended for Production)

Edit `backend/config/production.json`:

```json
{
  "distributed": {
    "enable_sharding": true,
    "enable_replication": true,
    "enable_clustering": true,
    "sharding_strategy": "hash",
    "num_shards": 16,
    "replication_factor": 3,
    "synchronous_replication": true,
    "cluster_host": "0.0.0.0",
    "cluster_port": 9080,
    "seed_nodes": [
      "node1:9080",
      "node2:9080",
      "node3:9080"
    ]
  }
}
```

Then run:

```bash
export JADEVECTORDB_ENV=production
cd backend/build
./jadevectordb
```

### Option 3: Docker Compose

Edit `docker-compose.yml` or use `docker-compose.distributed.yml`:

```yaml
services:
  jadevectordb-master:
    image: jadevectordb/jadevectordb:latest
    environment:
      - JADEVECTORDB_ENV=production
      - JADEVECTORDB_ENABLE_SHARDING=true
      - JADEVECTORDB_ENABLE_REPLICATION=true
      - JADEVECTORDB_ENABLE_CLUSTERING=true
      - JADEVECTORDB_NUM_SHARDS=16
      - JADEVECTORDB_SEED_NODES=jadevectordb-master:9080,jadevectordb-worker-1:9080,jadevectordb-worker-2:9080
    ports:
      - "8080:8080"
      - "9080:9080"
    networks:
      - jadevectordb-cluster

  jadevectordb-worker-1:
    image: jadevectordb/jadevectordb:latest
    environment:
      - JADEVECTORDB_ENV=production
      - JADEVECTORDB_ENABLE_SHARDING=true
      - JADEVECTORDB_ENABLE_REPLICATION=true
      - JADEVECTORDB_ENABLE_CLUSTERING=true
      - JADEVECTORDB_SEED_NODES=jadevectordb-master:9080,jadevectordb-worker-1:9080,jadevectordb-worker-2:9080
    ports:
      - "8081:8080"
      - "9081:9080"
    networks:
      - jadevectordb-cluster

  jadevectordb-worker-2:
    image: jadevectordb/jadevectordb:latest
    environment:
      - JADEVECTORDB_ENV=production
      - JADEVECTORDB_ENABLE_SHARDING=true
      - JADEVECTORDB_ENABLE_REPLICATION=true
      - JADEVECTORDB_ENABLE_CLUSTERING=true
      - JADEVECTORDB_SEED_NODES=jadevectordb-master:9080,jadevectordb-worker-1:9080,jadevectordb-worker-2:9080
    ports:
      - "8082:8080"
      - "9082:9080"
    networks:
      - jadevectordb-cluster

networks:
  jadevectordb-cluster:
    driver: bridge
```

Run:
```bash
docker-compose up --build
```

## 🔍 Verify Distributed Features Are Running

Check the logs for:

```
[INFO] Distributed config: sharding=true, replication=true, clustering=true
[INFO] Distributed services initialized successfully
[INFO] Distributed services started successfully
```

## 📊 Available Configuration Options

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| Enable Sharding | `JADEVECTORDB_ENABLE_SHARDING` | `false` | Enable data sharding |
| Enable Replication | `JADEVECTORDB_ENABLE_REPLICATION` | `false` | Enable replication |
| Enable Clustering | `JADEVECTORDB_ENABLE_CLUSTERING` | `false` | Enable clustering |
| Num Shards | `JADEVECTORDB_NUM_SHARDS` | `4` | Number of shards |
| Replication Factor | `JADEVECTORDB_REPLICATION_FACTOR` | `3` | Replicas per shard |
| Cluster Host | `JADEVECTORDB_CLUSTER_HOST` | `0.0.0.0` | Cluster bind address |
| Cluster Port | `JADEVECTORDB_CLUSTER_PORT` | `9080` | Cluster port |
| Seed Nodes | `JADEVECTORDB_SEED_NODES` | `[]` | Comma-separated nodes |
| Sharding Strategy | `JADEVECTORDB_SHARDING_STRATEGY` | `hash` | hash/range/vector/auto |
| Routing Strategy | `JADEVECTORDB_ROUTING_STRATEGY` | `round_robin` | Routing algorithm |

## 🧪 Testing Distributed Features

1. **Start 3 nodes** using one of the methods above
2. **Create a database**:
   ```bash
   curl -X POST http://localhost:8080/api/databases \
     -H "Content-Type: application/json" \
     -d '{"database_id": "test_distributed", "dimensions": 128}'
   ```

3. **Insert vectors** (will be distributed across shards):
   ```bash
   curl -X POST http://localhost:8080/api/databases/test_distributed/vectors \
     -H "Content-Type: application/json" \
     -d '{
       "vector_id": "vec1",
       "embedding": [0.1, 0.2, ...],
       "metadata": {"test": "distributed"}
     }'
   ```

4. **Search** (query will be distributed):
   ```bash
   curl -X POST http://localhost:8080/api/databases/test_distributed/search \
     -H "Content-Type: application/json" \
     -d '{
       "embedding": [0.1, 0.2, ...],
       "k": 10
     }'
   ```

5. **Check cluster status**:
   ```bash
   curl http://localhost:8080/api/cluster/status
   ```

## ⚠️ Important Notes

- **Phase 1 Default**: Distributed features are **disabled** by default
- **Phase 2 Readiness**: All code implemented but needs multi-node testing
- **Network Requirements**: Ensure ports 8080 (API) and 9080 (cluster) are open
- **Seed Nodes**: Must include all nodes for proper cluster formation
- **Production Testing**: Thoroughly test before production use

## 📖 More Information

- Full configuration guide: `docs/CONFIGURATION_GUIDE.md`
- Distributed API docs: `docs/distributed_services_api.md`
- Deployment guide: `docs/distributed_deployment_guide.md`
- Architecture: `docs/architecture.md`

## 🐛 Troubleshooting

**Issue**: Distributed services not starting
- **Check**: Look for `[WARN] Failed to initialize distributed services` in logs
- **Solution**: Verify seed nodes are reachable and ports are open

**Issue**: Shards not created
- **Check**: `enable_sharding=true` in logs
- **Solution**: Verify environment variable or JSON config is set correctly

**Issue**: Nodes not forming cluster
- **Check**: Seed nodes connectivity: `nc -zv node1 9080`
- **Solution**: Fix network connectivity or firewall rules

---

**Created**: December 19, 2025  
**Status**: Distributed features fully implemented, configuration system ready  
**Phase**: Phase 1 (single-node) by default, Phase 2 (distributed) when enabled
