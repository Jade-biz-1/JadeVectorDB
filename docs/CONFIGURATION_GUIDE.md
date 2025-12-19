# JadeVectorDB Configuration Guide

## Configuration Overview

JadeVectorDB supports flexible configuration through multiple sources with the following precedence (highest to lowest):

1. **Environment Variables** (highest priority)
2. **Docker Secrets** (for sensitive data)
3. **JSON Configuration Files**
4. **Default Values** (lowest priority)

## Configuration Files

Configuration files are located in `backend/config/` and loaded based on the `JADEVECTORDB_ENV` environment variable:

- **development.json** - Development environment (default)
- **production.json** - Production environment
- **distributed.json** - Optional distributed-specific settings
- **performance.json** - Performance tuning (optional)
- **logging.json** - Logging configuration (optional)

### Loading Order

1. Base config (`development.json` or `production.json`)
2. Performance config (if exists)
3. Logging config (if exists)
4. Distributed config (if exists)
5. Environment variable overrides
6. Docker secrets

## Environment Selection

Set the environment using:

```bash
export JADEVECTORDB_ENV=production  # or development, testing
```

## Distributed System Configuration

### Enable Distributed Features

Three ways to enable distributed features:

#### 1. Via JSON Configuration File

Edit `backend/config/production.json` or create `backend/config/distributed.json`:

```json
{
  "distributed": {
    "enable_sharding": true,
    "enable_replication": true,
    "enable_clustering": true,
    "sharding_strategy": "hash",
    "num_shards": 16,
    "replication_factor": 3,
    "seed_nodes": ["node1:9080", "node2:9080", "node3:9080"]
  }
}
```

#### 2. Via Environment Variables (Overrides JSON)

```bash
export JADEVECTORDB_ENABLE_SHARDING=true
export JADEVECTORDB_ENABLE_REPLICATION=true
export JADEVECTORDB_ENABLE_CLUSTERING=true
export JADEVECTORDB_NUM_SHARDS=16
export JADEVECTORDB_REPLICATION_FACTOR=3
export JADEVECTORDB_CLUSTER_HOST=0.0.0.0
export JADEVECTORDB_CLUSTER_PORT=9080
export JADEVECTORDB_SEED_NODES=node1:9080,node2:9080,node3:9080
```

#### 3. Via Docker Compose

```yaml
services:
  jadevectordb:
    image: jadevectordb/jadevectordb:latest
    environment:
      - JADEVECTORDB_ENV=production
      - JADEVECTORDB_ENABLE_SHARDING=true
      - JADEVECTORDB_ENABLE_REPLICATION=true
      - JADEVECTORDB_ENABLE_CLUSTERING=true
      - JADEVECTORDB_NUM_SHARDS=16
      - JADEVECTORDB_SEED_NODES=node1:9080,node2:9080
```

### Distributed Configuration Options

#### Sharding Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `enable_sharding` | `JADEVECTORDB_ENABLE_SHARDING` | `false` | Enable data sharding across nodes |
| `sharding_strategy` | `JADEVECTORDB_SHARDING_STRATEGY` | `hash` | Strategy: `hash`, `range`, `vector`, `auto` |
| `num_shards` | `JADEVECTORDB_NUM_SHARDS` | `4` | Number of shards (production: 16+) |
| `replication_factor` | `JADEVECTORDB_REPLICATION_FACTOR` | `3` | Replicas per shard |

#### Replication Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `enable_replication` | `JADEVECTORDB_ENABLE_REPLICATION` | `false` | Enable data replication |
| `default_replication_factor` | `JADEVECTORDB_DEFAULT_REPLICATION_FACTOR` | `3` | Default replicas |
| `synchronous_replication` | `JADEVECTORDB_SYNCHRONOUS_REPLICATION` | `false` | Sync vs async replication |
| `replication_timeout_ms` | `JADEVECTORDB_REPLICATION_TIMEOUT_MS` | `5000` | Replication timeout |
| `replication_strategy` | `JADEVECTORDB_REPLICATION_STRATEGY` | `simple` | Replication strategy |

#### Clustering Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `enable_clustering` | `JADEVECTORDB_ENABLE_CLUSTERING` | `false` | Enable cluster mode |
| `cluster_host` | `JADEVECTORDB_CLUSTER_HOST` | `0.0.0.0` | Cluster communication host |
| `cluster_port` | `JADEVECTORDB_CLUSTER_PORT` | `9080` | Cluster communication port |
| `seed_nodes` | `JADEVECTORDB_SEED_NODES` | `[]` | Comma-separated seed nodes |

#### Routing Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `routing_strategy` | `JADEVECTORDB_ROUTING_STRATEGY` | `round_robin` | Strategy: `round_robin`, `least_loaded`, `locality_aware` |
| `max_route_cache_size` | `JADEVECTORDB_MAX_ROUTE_CACHE_SIZE` | `1000` | Route cache size |
| `route_ttl_seconds` | `JADEVECTORDB_ROUTE_TTL_SECONDS` | `300` | Route cache TTL |
| `enable_adaptive_routing` | `JADEVECTORDB_ENABLE_ADAPTIVE_ROUTING` | `true` | Enable adaptive routing |

## Other Configuration Sections

### Server Configuration

```json
{
  "server": {
    "port": 8080,
    "host": "0.0.0.0"
  }
}
```

Environment variables:
- `JADEVECTORDB_PORT=8080`
- `JADEVECTORDB_HOST=0.0.0.0`

### Database Configuration

```json
{
  "database": {
    "connection_pool_size": 20,
    "query_timeout_seconds": 30,
    "max_retries": 3,
    "db_path": "/data/jadevectordb.db",
    "auth_db_path": "/data/jadevectordb_auth.db"
  }
}
```

Environment variables:
- `JADEVECTORDB_DB_PATH=/data/jadevectordb.db`
- `JADEVECTORDB_AUTH_DB_PATH=/data/jadevectordb_auth.db`

### Security Configuration

```json
{
  "security": {
    "enable_rate_limiting": true,
    "enable_ip_blocking": true,
    "max_failed_logins": 5,
    "block_duration_seconds": 3600
  }
}
```

### Logging Configuration

```json
{
  "logging": {
    "level": "info",
    "format": "json",
    "output": "file",
    "file_path": "/var/log/jadevectordb/app.log"
  }
}
```

Environment variables:
- `JADEVECTORDB_LOG_LEVEL=info`
- `JADEVECTORDB_LOG_FILE=/var/log/jadevectordb/app.log`

## Example Configurations

### Single-Node Deployment (Default)

```json
{
  "server": {
    "port": 8080,
    "host": "0.0.0.0"
  },
  "distributed": {
    "enable_sharding": false,
    "enable_replication": false,
    "enable_clustering": false
  }
}
```

### Multi-Node Distributed Cluster

```json
{
  "server": {
    "port": 8080,
    "host": "0.0.0.0"
  },
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
      "jadevectordb-0.jadevectordb-headless:9080",
      "jadevectordb-1.jadevectordb-headless:9080",
      "jadevectordb-2.jadevectordb-headless:9080"
    ]
  }
}
```

### Development Environment

```bash
# Use development.json (already has distributed disabled)
export JADEVECTORDB_ENV=development
./jadevectordb
```

### Production Environment with Distributed Features

```bash
# Use production.json + environment overrides
export JADEVECTORDB_ENV=production
export JADEVECTORDB_ENABLE_SHARDING=true
export JADEVECTORDB_ENABLE_REPLICATION=true
export JADEVECTORDB_ENABLE_CLUSTERING=true
export JADEVECTORDB_SEED_NODES=node1:9080,node2:9080,node3:9080
./jadevectordb
```

## Docker Secrets

For sensitive configuration (passwords, API keys), use Docker secrets:

1. Create secret files:
   ```bash
   echo "my_db_password" > /run/secrets/db_password
   echo "my_jwt_secret" > /run/secrets/jwt_secret
   ```

2. Docker Compose:
   ```yaml
   secrets:
     db_password:
       file: ./secrets/db_password
     jwt_secret:
       file: ./secrets/jwt_secret
   
   services:
     jadevectordb:
       secrets:
         - db_password
         - jwt_secret
   ```

## Configuration Validation

The system validates configuration on startup and will fail if:
- Required secrets are missing (in production)
- Invalid values (e.g., negative ports)
- Conflicting settings

Check logs for validation errors:
```
[ERROR] Configuration validation failed: Invalid port number
```

## Troubleshooting

### Distributed Features Not Starting

1. Check configuration is loaded:
   ```
   [INFO] Distributed config: sharding=true, replication=true, clustering=true
   ```

2. Check for initialization errors:
   ```
   [WARN] Failed to initialize distributed services: ...
   ```

3. Verify seed nodes are reachable:
   ```bash
   nc -zv node1 9080
   ```

### Configuration Not Loading

1. Verify file exists:
   ```bash
   ls -la backend/config/production.json
   ```

2. Check JSON syntax:
   ```bash
   jq . backend/config/production.json
   ```

3. Check environment variable:
   ```bash
   echo $JADEVECTORDB_ENV
   ```

## See Also

- [Distributed Services API](../docs/distributed_services_api.md)
- [Distributed Deployment Guide](../docs/distributed_deployment_guide.md)
- [Architecture Documentation](../docs/architecture.md)
