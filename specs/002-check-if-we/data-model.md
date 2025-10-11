# Data Model for JadeVectorDB

## Entities

### Vector
**Description:** A mathematical representation of data in N-dimensional space, including the vector values and associated metadata.

**Fields:**
- `id` (string): Unique identifier for the vector
- `values` (array<float>): The vector values as float32/float64 values, length = vector dimension
- `metadata` (object): Associated metadata for the vector
  - `source` (string, optional): Source of the vector data
  - `created_at` (timestamp): Timestamp when the vector was created
  - `updated_at` (timestamp): Timestamp when the vector was last updated
  - `tags` (array<string>, optional): Tags associated with the vector
  - `owner` (string, optional): Owner of the vector
  - `permissions` (array<string>, optional): Permissions for the vector
  - `category` (string, optional): Category of the vector
  - `score` (float, optional): Score or relevance value
  - `status` (enum: string): Status of the vector (active, archived, deleted)
  - `custom` (object, optional): Custom metadata fields
- `index` (object, optional): Index configuration for this vector
  - `type` (string): Type of index (HNSW, IVF, LSH, etc.)
  - `version` (string): Version of the index
  - `parameters` (object): Index-specific parameters
- `embedding_model` (object, optional): Embedding model used to generate this vector
  - `name` (string): Name of the embedding model
  - `version` (string): Version of the embedding model
  - `provider` (string): Provider of the embedding model
  - `input_type` (string): Type of input data (text, image, etc.)
- `shard` (string): Shard identifier for distributed storage
- `replicas` (array<string>): List of node identifiers where replicas are stored
- `version` (integer): Version number for the vector record
- `deleted` (boolean): Flag indicating if the vector is logically deleted

**Relationships:**
- Belongs to one Database
- Belongs to one Index (via index field)

**Validation Rules:**
- `id` must be unique within the database
- `values` must have consistent dimensions for the database's configuration
- `created_at` and `updated_at` must be valid ISO 8601 timestamps
- `status` must be one of the allowed enum values
- `values` length must match the database's configured vector dimension

### Database
**Description:** A collection of vectors with common configuration parameters like vector dimensions, indexing algorithm, and metadata schema.

**Fields:**
- `databaseId` (string): Unique identifier for the database
- `name` (string): Name of the database
- `description` (string, optional): Description of the database
- `vectorDimension` (integer): Dimension of vectors in this database
- `indexType` (string): Default index type for this database (HNSW, IVF, LSH, etc.)
- `indexParameters` (object): Default parameters for the index
- `sharding` (object): Sharding configuration
  - `strategy` (string): Sharding strategy (hash, range, vector-based)
  - `numShards` (integer): Number of shards
- `replication` (object): Replication configuration
  - `factor` (integer): Number of replicas
  - `sync` (boolean): Whether to use synchronous replication
- `embeddingModels` (array<object>): List of supported embedding models
- `metadataSchema` (object): Schema definition for metadata
- `retentionPolicy` (object, optional): Retention policy configuration
- `accessControl` (object): Access control configuration
- `created_at` (timestamp): Timestamp when the database was created
- `updated_at` (timestamp): Timestamp when the database was last updated

**Relationships:**
- Contains many Vectors
- Contains many Indexes

**Validation Rules:**
- `databaseId` must be unique across the cluster
- `name` must be unique across the cluster
- `vectorDimension` must be positive
- `indexType` must be one of supported index types
- `sharding.numShards` must be positive
- `replication.factor` must be positive
- `created_at` and `updated_at` must be valid ISO 8601 timestamps

### Index
**Description:** A data structure that enables fast similarity search by organizing vectors for efficient retrieval.

**Fields:**
- `indexId` (string): Unique identifier for the index
- `databaseId` (string): Reference to the parent database
- `type` (string): Type of index (HNSW, IVF, LSH, Flat Index)
- `parameters` (object): Index-specific configuration parameters
  - Varies by index type (e.g., HNSW: M, efConstruction; IVF: nlist, etc.)
- `status` (enum: string): Status of the index (building, ready, failed)
- `created_at` (timestamp): Timestamp when the index was created
- `updated_at` (timestamp): Timestamp when the index was last updated

**Relationships:**
- Belongs to one Database
- Associated with many Vectors

**Validation Rules:**
- `indexId` must be unique across the cluster
- `databaseId` must reference an existing database
- `type` must be one of supported index types
- Parameters must be valid for the index type
- `status` must be one of allowed enum values
- `created_at` and `updated_at` must be valid ISO 8601 timestamps

### EmbeddingModel
**Description:** A machine learning model that transforms raw data (text, images, etc.) into vector representations.

**Fields:**
- `modelId` (string): Unique identifier for the model
- `name` (string): Name of the model (BERT, ResNet, etc.)
- `version` (string): Version of the model
- `provider` (string): Provider of the model (huggingface, torchvision, etc.)
- `inputType` (string): Type of input data the model expects
- `outputDimension` (integer): Dimension of vectors produced by this model
- `parameters` (object): Model-specific parameters
- `status` (enum: string): Status of the model (active, inactive, failed)

**Relationships:**
- Associated with many Vectors

**Validation Rules:**
- `modelId` must be unique across the cluster
- `outputDimension` must be positive
- `status` must be one of allowed enum values

### Server
**Description:** A single instance of the vector database service that can operate as either master or worker depending on cluster state.

**Fields:**
- `serverId` (string): Unique identifier for the server
- `host` (string): Host address of the server
- `port` (integer): Port on which the server is running
- `nodeType` (enum: string): Type of node (master, worker)
- `status` (enum: string): Status of the server (active, standby, failed)
- `lastHeartbeat` (timestamp): Timestamp of last heartbeat
- `resources` (object): Resource allocation information
  - `cpu` (float): CPU usage percentage
  - `memory` (float): Memory usage percentage
  - `storage` (float): Storage usage percentage
- `created_at` (timestamp): Timestamp when the server was registered

**Relationships:**
- Manages many Shards (as master)
- Stores many Vectors (as worker)

**Validation Rules:**
- `serverId` must be unique across the cluster
- `port` must be a valid port number (1-65535)
- `nodeType` must be master or worker
- `status` must be one of allowed enum values
- `host` must be a valid IP address or domain name

### Shard
**Description:** A partition of the vector database containing a subset of the overall vector data.

**Fields:**
- `shardId` (string): Unique identifier for the shard
- `databaseId` (string): Reference to the parent database
- `serverId` (string): Reference to the server hosting this shard
- `rangeStart` (string, optional): Start of the range for range-based sharding
- `rangeEnd` (string, optional): End of the range for range-based sharding
- `hashRange` (object, optional): Hash range for hash-based sharding
  - `start` (integer): Start of hash range
  - `end` (integer): End of hash range
- `status` (enum: string): Status of the shard (active, migrating, read-only)
- `size` (integer): Size of shard in bytes
- `replicas` (array<string>): List of server IDs that have replicas
- `created_at` (timestamp): Timestamp when the shard was created

**Relationships:**
- Belongs to one Database
- Hosted on one Server
- Contains many Vectors

**Validation Rules:**
- `shardId` must be unique across the cluster
- `databaseId` must reference an existing database
- `serverId` must reference an existing server
- `status` must be one of allowed enum values
- `size` must be non-negative

## State Transitions

### Vector State Transitions
- `active` → `archived`: When retention policy expires
- `active` → `deleted`: When explicitly deleted
- `archived` → `active`: When restored from archive
- `deleted` → `active`: On undelete operation (if soft delete is enabled)

### Index State Transitions
- `building` → `ready`: When index build completes successfully
- `building` → `failed`: When index build encounters an error
- `ready` → `building`: When index is being updated or rebuilt
- `ready` → `failed`: When index encounters an error

### Server State Transitions
- `standby` → `active`: When server becomes master or active worker
- `active` → `standby`: When server transitions to standby
- `active` → `failed`: When server becomes unresponsive
- `failed` → `standby`: When server recovers

### Shard State Transitions
- `active` → `migrating`: When shard is being moved between servers
- `migrating` → `active`: When migration completes successfully
- `migrating` → `active`: When migration fails but original remains active
- `active` → `read-only`: During maintenance operations
- `read-only` → `active`: When maintenance completes

## UI and CLI Considerations

### Web UI Schema
The Web UI (Next.js + shadcn) will include components for:
- Cluster monitoring dashboard
- Database management interface
- Vector data exploration and visualization
- Index configuration and management
- User and permission management
- Performance metrics visualization
- Query builder and execution interface

### CLI Schema
The CLI tools will support:
- Python client library with full API coverage
- Shell script utilities for common operations
- Configuration management
- Batch operations for data import/export
- Monitoring and diagnostic commands
- Backup and recovery operations