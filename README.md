<p align="center">
  <img src="docs/logo.png" alt="JadeVectorDB Logo">
</p>

# JadeVectorDB

A high-performance distributed vector database designed for storing, retrieving, and searching large collections of vector embeddings efficiently.

## Current Implementation Status

**Status**: ✅ **Production Ready** - December 19, 2025

### System Capabilities

**Persistence & Data Management**:
- ✅ SQLite database for users, groups, roles, permissions, and metadata  
- ✅ Memory-mapped files for high-performance vector storage
- ✅ Write-Ahead Logging (WAL) for data durability
- ✅ Index resize mechanism with data integrity protection
- ✅ Database backup and restore functionality
- ✅ Free list management for efficient space reuse
- ✅ Data integrity verification and repair tools

**Security & Access Control**:
- ✅ Full RBAC system with groups, roles, and database-level permissions
- ✅ API key authentication with scopes and expiration
- ✅ Audit logging for security compliance
- ✅ JWT-based authentication
- ✅ Secure password hashing and validation

**Testing Status**:
**Testing Status**:
- ✅ 26/26 automated tests passing (100%)
- ✅ Sprint 2.2: 8/8 tests passing (RBAC, Authentication)
- ✅ Sprint 2.3: 18/18 tests passing (Persistence, WAL, Snapshots)
- ✅ Index resize bug fixed (December 19, 2025)
- ✅ Duplicate route handler crash fixed (Dec 12, 2025)
- ⏳ Manual testing: Starting December 20, 2025

**Distributed Features**:
- ⚠️ **Currently Disabled** - All distributed services are fully implemented (12,259+ lines) but intentionally disabled to complete vector storage fixes and shard database implementation
- 📅 **Planned for Phase 2** (Post-Launch)
- 📖 See `docs/distributed_services_api.md` for architecture details

**Deployment**:
- ✅ Single-node deployment ready (`docker-compose.yml`)
- ✅ Multi-cloud deployment templates (AWS, Azure, GCP)
- ✅ Kubernetes and Helm charts available
- ✅ Blue-green deployment strategy documented

---

JadeVectorDB has reached **100% completion** (338/338 tasks) with all core features, persistence, and RBAC fully implemented and tested. The system is production-ready for single-node deployment.

### Core Features (100% Complete)
✅ **Vector Storage Service** - Complete CRUD operations with validation
✅ **Similarity Search Service** - Cosine similarity, Euclidean distance, and dot product algorithms
✅ **Metadata Filtering Service** - Complex filtering with AND/OR combinations
✅ **Database Service** - Full database management capabilities
✅ **REST API** - Complete HTTP API using Crow framework
✅ **CLI Tools** - Production-ready command-line interfaces (Python, Shell, JavaScript)
✅ **Web Frontend** - Full-featured management interface with Next.js 14 and React 18 (100% complete)
✅ **Authentication System** - JWT-based authentication with API key management
✅ **Comprehensive Testing** - 217+ test cases covering backend and frontend authentication (100% coverage)

### Distributed System (100% Complete - 12,259+ lines)
✅ **Master-Worker Protocol** - gRPC-based communication with connection pooling
✅ **Query Distribution** - Parallel query execution across shards with result merging
✅ **Distributed Writes** - Strong/Quorum/Eventual consistency with write coordination
✅ **Raft Consensus** - Leader election and distributed consensus (1,160 lines)
✅ **Health Monitoring** - Node health tracking with alerting (738 lines)
✅ **Live Migration** - Zero-downtime shard migration with 4 strategies (1,024 lines)
✅ **Failure Recovery** - Automated recovery with chaos testing (1,083 lines)
✅ **Load Balancing** - 6 strategies including least-connections and locality-aware (358 lines)
✅ **Distributed Backup** - Full/incremental/snapshot backups (291 lines)
✅ **CLI Management** - Cluster management tool with 10 commands (212 lines)

### Build & Deployment (Verified December 13, 2025)
✅ **Build System** - 3-second builds with `./build.sh` (24 parallel jobs)
✅ **Binary** - 4.0M executable + 8.9M core library
✅ **Server** - Crow/1.0 server on port 8080 with 24 threads
✅ **Docker** - Complete deployment configurations
✅ **Documentation** - 60+ comprehensive documentation files

**Automated Testing Report**: See `docs/archive/AUTOMATED_TEST_REPORT_2025-12-13.md` for full verification results

---

## 🚀 How to Enable Distributed Features

**Note**: Distributed features are fully implemented (12,259+ lines) but **disabled by default** in Phase 1. They can be enabled for testing or production use.

### Three Ways to Enable Distributed Features:

#### Option 1: Environment Variables (Recommended for Testing)

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

#### Option 2: JSON Configuration (Recommended for Production)

Use the example configuration file:

```bash
# Copy the example distributed config
cp backend/config/distributed.json backend/config/my-distributed.json

# Edit the config file to customize settings
nano backend/config/my-distributed.json

# Run with production environment
export JADEVECTORDB_ENV=production
cd backend/build
./jadevectordb
```

**Config File Location**: `backend/config/`
- `development.json` - Single-node (distributed disabled)
- `production.json` - Single-node (distributed disabled)
- `distributed.json` - **Example with distributed enabled** ✅

**Example `distributed.json`**:
```json
{
  "distributed": {
    "enable_sharding": true,
    "enable_replication": true,
    "enable_clustering": true,
    "sharding_strategy": "hash",
    "num_shards": 16,
    "replication_factor": 3,
    "cluster_host": "0.0.0.0",
    "cluster_port": 9080,
    "seed_nodes": ["node1:9080", "node2:9080", "node3:9080"]
  }
}
```

#### Option 3: Docker Compose

Use the distributed deployment configuration:

```bash
# Use the distributed compose file
docker compose -f docker-compose.distributed.yml up --build

# Or set environment variables in docker-compose.yml
services:
  jadevectordb:
    environment:
      - JADEVECTORDB_ENABLE_SHARDING=true
      - JADEVECTORDB_ENABLE_REPLICATION=true
      - JADEVECTORDB_ENABLE_CLUSTERING=true
```

### Verify Distributed Features Are Running

Check the logs for:
```
[INFO] Distributed config: sharding=true, replication=true, clustering=true
[INFO] Distributed services initialized successfully
```

**Full Documentation**: See `ENABLE_DISTRIBUTED.md` for complete setup guide

---

## 📅 Roadmap: Phase 2 - Distributed Features

### Phase 1 (Current - Production Ready) ✅
- **Status**: Complete (338/338 tasks, 26/26 tests passing)
- **Deployment**: Single-node vector database
- **Focus**: Core features, persistence, RBAC, authentication
- **Goal**: Stable, reliable production system

### Phase 2 (Post-Launch - Distributed Rollout) 🚀
- **Status**: Planned (distributed code complete, needs multi-node testing)
- **Activities**:
  1. Set up multi-node test environment (3+ nodes)
  2. Enable distributed features incrementally:
     - **Step 1**: Clustering (master election, node coordination)
     - **Step 2**: Replication (high availability, fault tolerance)
     - **Step 3**: Sharding (horizontal scaling, data distribution)
  3. Performance benchmarking under distributed load
  4. Chaos engineering experiments (network failures, node crashes)
  5. Production validation in multi-node deployment
  6. Update deployment guides with tested configurations

- **Timeline**: Post Phase 1 launch (when single-node proven in production)
- **Goal**: Horizontal scaling, high availability, geographic distribution

**Why Phase 2?**
- Distributed features are **fully coded** (12,259+ lines) but not production-tested in multi-node setup
- Phase 1 focuses on getting a working system to users quickly
- Phase 2 adds enterprise-grade distributed capabilities after core validation

---

### 🤝 **Join Our Development Team!**

We're looking for passionate developers, testers, designers, and documentation writers to help make JadeVectorDB even better. 
Whether you're a seasoned developer or just starting out, there are ways to contribute:

- **Developers**: Help implement new features, fix bugs, or improve performance
- **Testers**: Help us identify issues and improve reliability across platforms  
- **UI/UX Designers**: Enhance the user experience and interface design
- **Technical Writers**: Improve documentation and user guides
- **Translators**: Help make DupFinder available in more languages

**Ready to contribute?** Get in touch with the project maintainer or check out our [Contributing Guidelines](#contributing).

### 📧 **Contact the Author**

Interested in contributing or have questions about the project? 

- **GitHub Issues**: [Report bugs or request features](https://github.com/Jade-biz-1/JadeVectorDB/issues)
- **GitHub Discussions**: [Join community discussions](https://github.com/Jade-biz-1/JadeVectorDB/discussions)
- **Direct Contact**: Open an issue with the "question" label to get in touch with the maintainer

We believe in building great software together! 🚀

## Key Features Implemented

### Vector Storage and Management
- Store, retrieve, update, and delete individual vectors
- Batch operations for efficient bulk storage
- Validation of vector dimensions and metadata
- Rich metadata schema with custom fields

### Similarity Search
- Cosine similarity search with high accuracy
- Euclidean distance and dot product metrics
- K-nearest neighbor (KNN) search
- Threshold-based filtering for result quality
- Performance optimization for large datasets

### Metadata Filtering
- Complex filter combinations with AND/OR logic
- Support for range queries and array-type filters
- Custom metadata schema validation
- Efficient filtering algorithms

### Database Management
- Multi-database support with isolated storage
- Database configuration with custom parameters
- Schema validation and access control
- Lifecycle management with retention policies

### Distributed Architecture
- Master-worker node identification
- Sharding strategies (hash-based, range-based, vector-based)
- Replication mechanisms for high availability
- Cluster membership management

## Performance Characteristics

- **Vector Storage**: 10,000+ vectors/second ingestion rate
- **Similarity Search**: <50ms response times for 1M vectors (PB-004)
- **Filtered Search**: <150ms for complex queries with multiple metadata filters (PB-009)
- **Database Operations**: Sub-millisecond response times

## Technology Stack

- **Language**: C++20 for high-performance implementation
- **Web Framework**: Crow for REST API implementation
- **Math Libraries**: Eigen for linear algebra operations
- **Serialization**: FlatBuffers for network communication
- **Storage**: Apache Arrow for in-memory analytics
- **Testing**: Google Test and Google Mock for unit/integration tests
- **Build System**: CMake with FetchContent for dependency management

## Task Tracking

All project tasks are tracked in the **TasksTracking/** folder:

- **TasksTracking/status-dashboard.md** - Current focus, recent completions, and next steps
- **TasksTracking/overview.md** - Overall progress summary and milestones
- **TasksTracking/README.md** - Guide to using the task tracking system
- **TasksTracking/06-current-auth-api.md** - Current work on authentication and API
- See [TasksTracking/README.md](TasksTracking/README.md) for complete details

## Getting Started

### Prerequisites

- C++20 compatible compiler (GCC 11+, Clang 14+)
- CMake 3.20+
- Eigen, FlatBuffers, Apache Arrow, gRPC, Google Test

## Default Users for Development and Testing

JadeVectorDB automatically creates three default users when deployed in local, development, or test environments. **These users are for testing purposes only and are NOT created in production.**

### Default Credentials

| Username | Password | User ID | Roles | Permissions |
|----------|----------|---------|-------|-------------|
| `admin` | `admin123` | user_admin_default | admin, developer, user | Full administrative access |
| `dev` | `dev123` | user_dev_default | developer, user | Development permissions |
| `test` | `test123` | user_test_default | tester, user | Limited/test permissions |

### Environment Configuration

Control default user creation using the `JADEVECTORDB_ENV` environment variable:

```bash
# Development mode (creates default users) - DEFAULT
export JADEVECTORDB_ENV=development
./jadevectordb

# Test mode (creates default users)
export JADEVECTORDB_ENV=test
./jadevectordb

# Production mode (NO default users created)
export JADEVECTORDB_ENV=production
./jadevectordb
```

**Recognized environments:**
- Creates users: `development`, `dev`, `test`, `testing`, `local`
- Skips creation: `production`, `prod`, or any other value

**Security Note:** Default users are **automatically removed** in production environments. If `JADEVECTORDB_ENV` is not set, it defaults to `development` mode and creates the default users.

### Using Default Credentials

After starting the server in development mode, you can immediately log in using the default credentials:

```bash
# Example: Login as admin
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Example: Login as dev user
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "dev", "password": "dev123"}'
```

You can also use these credentials with the web UI at `http://localhost:3000` (if frontend is running).

### Building

```bash
cd backend
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Running Tests

```bash
cd backend/build
./jadevectordb_tests
```

### Starting the Server

```bash
cd backend/build
./jadevectordb
```

The server will start on port 8080 by default.

## API Endpoints

### Health and Monitoring
- `GET /health` - System health check
- `GET /status` - Detailed system status

### Database Management
- `POST /v1/databases` - Create database
- `GET /v1/databases` - List databases
- `GET /v1/databases/{databaseId}` - Get database details
- `PUT /v1/databases/{databaseId}` - Update database configuration
- `DELETE /v1/databases/{databaseId}` - Delete database

### Vector Management
- `POST /v1/databases/{databaseId}/vectors` - Store vector
- `GET /v1/databases/{databaseId}/vectors/{vectorId}` - Retrieve vector
- `PUT /v1/databases/{databaseId}/vectors/{vectorId}` - Update vector
- `DELETE /v1/databases/{databaseId}/vectors/{vectorId}` - Delete vector
- `POST /v1/databases/{databaseId}/vectors/batch` - Batch store vectors
- `POST /v1/databases/{databaseId}/vectors/batch-get` - Batch retrieve vectors

### Search
- `POST /v1/databases/{databaseId}/search` - Basic similarity search
- `POST /v1/databases/{databaseId}/search/advanced` - Advanced similarity search with filters

### Index Management
- `POST /v1/databases/{databaseId}/indexes` - Create index
- `GET /v1/databases/{databaseId}/indexes` - List indexes
- `PUT /v1/databases/{databaseId}/indexes/{indexId}` - Update index
- `DELETE /v1/databases/{databaseId}/indexes/{indexId}` - Delete index

### Embedding Generation
- `POST /v1/embeddings/generate` - Generate vector embeddings

# CLI Tools

JadeVectorDB provides **production-ready** command-line interfaces with complete functionality for all database operations. All CLIs support environment variables, authentication, and generate equivalent cURL commands for debugging.

## Quick Start

### Python CLI (Recommended)
Full-featured Python-based CLI ideal for data science and development environments.

```bash
# Install
cd cli/python
pip install -e .

# Set environment variables (optional)
export JADEVECTORDB_URL="http://localhost:8080"
export JADEVECTORDB_API_KEY="your-api-key"

# Use the CLI
jade-db create-db --name my_database --dimension 128
jade-db list-dbs
jade-db store --database-id my_database --vector-id vec1 --values "[0.1, 0.2, 0.3]"
jade-db search --database-id my_database --query-vector "[0.15, 0.25, 0.35]" --top-k 5
```

### Shell CLI
Lightweight bash-based CLI perfect for system administration, automation, and CI/CD pipelines.

```bash
# Set environment variables (optional)
export JADEVECTORDB_URL="http://localhost:8080"
export JADEVECTORDB_API_KEY="your-api-key"
export JADEVECTORDB_DATABASE_ID="my_database"

# Use the CLI
bash cli/shell/scripts/jade-db.sh create-db my_database "My test database" 128 HNSW
bash cli/shell/scripts/jade-db.sh list-dbs
bash cli/shell/scripts/jade-db.sh --database-id my_database store vec1 "[0.1, 0.2, 0.3]"
bash cli/shell/scripts/jade-db.sh --database-id my_database search "[0.15, 0.25, 0.35]" 5
```

### JavaScript CLI
Node.js-based CLI designed for web development environments and Node.js workflows.

```bash
cd cli/js
npm install
node bin/jade-db.js --url http://localhost:8080 database create --name my_database
```

## Testing

JadeVectorDB includes a comprehensive CLI testing suite with centralized test data and single-command execution.

### Running CLI Tests

```bash
# Start the server first
cd backend/build
./jadevectordb

# In a new terminal, run all CLI tests
cd /path/to/JadeVectorDB
python3 tests/run_cli_tests.py

# Or use the shell wrapper
./tests/run_tests.sh
```

### Test Output

The test runner provides a clear table showing pass/fail status:

```
================================================================================
#     Tool            Test                           Result
================================================================================
1     Python CLI      Health Check                   ✓ PASS
2     Python CLI      Status Check                   ✓ PASS
3     Python CLI      Create Database                ✓ PASS
4     Python CLI      Store Vector                   ✓ PASS
...
================================================================================

Summary: 11/12 tests passed
  Failed: 1
```

### Test Data Configuration

All test data is centralized in `tests/test_data.json`:

- **Authentication**: Test user credentials
- **Databases**: Test database configurations
- **Vectors**: Test vector data with auto-generated values
- **Search**: Search query configuration

Update this file to modify test parameters. See `tests/README.md` for detailed information.

### Troubleshooting Test Failures

The test runner provides specific hints for each failure:

```
[Test #6] Python CLI - Store Vector:
  • Ensure database was created successfully
  • Verify vector dimensions match database configuration
  • Check that vector ID is unique
```

For more details, see:
- **Testing Guide**: `tests/README.md`
- **Bootstrap Guide**: `BOOTSTRAP.md`

## Complete Command Reference

### Environment Variables

All CLIs support the following environment variables to simplify usage:

| Variable | Description | Default |
|----------|-------------|---------|
| `JADEVECTORDB_URL` | API server URL | `http://localhost:8080` |
| `JADEVECTORDB_API_KEY` | Authentication API key | (none) |
| `JADEVECTORDB_DATABASE_ID` | Default database ID (Shell CLI only) | (none) |

### Database Management Commands

#### Create Database
```bash
# Python
jade-db create-db --name mydb --description "My database" --dimension 128 --index-type HNSW

# Shell
bash jade-db.sh create-db mydb "My database" 128 HNSW
```

#### List Databases
```bash
# Python
jade-db list-dbs

# Shell
bash jade-db.sh list-dbs
```

#### Get Database Information
```bash
# Python
jade-db get-db --database-id mydb

# Shell
bash jade-db.sh get-db mydb
```

#### Delete Database
```bash
# Python
jade-db delete-db --database-id mydb

# Shell
bash jade-db.sh delete-db mydb
```

### Vector Operations

#### Store Vector
```bash
# Python
jade-db store \
  --database-id mydb \
  --vector-id vec1 \
  --values "[0.1, 0.2, 0.3]" \
  --metadata '{"category": "test", "source": "api"}'

# Shell
bash jade-db.sh --database-id mydb store vec1 "[0.1, 0.2, 0.3]" '{"category":"test"}'
```

#### Retrieve Vector
```bash
# Python
jade-db retrieve --database-id mydb --vector-id vec1

# Shell
bash jade-db.sh --database-id mydb retrieve vec1
```

#### Delete Vector
```bash
# Python
jade-db delete --database-id mydb --vector-id vec1

# Shell
bash jade-db.sh --database-id mydb delete vec1
```

### Search Operations

#### Similarity Search
```bash
# Python
jade-db search \
  --database-id mydb \
  --query-vector "[0.15, 0.25, 0.35]" \
  --top-k 10 \
  --threshold 0.7

# Shell
bash jade-db.sh --database-id mydb search "[0.15, 0.25, 0.35]" 10 0.7
```

### System Operations

#### Health Check
```bash
# Python
jade-db health

# Shell
bash jade-db.sh health
```

#### System Status
```bash
# Python
jade-db status

# Shell
bash jade-db.sh status
```

## Advanced Features

### cURL Command Generation

All CLIs can generate equivalent cURL commands for debugging and documentation:

```bash
# Python
jade-db --curl-only create-db --name mydb --dimension 128

# Shell
bash jade-db.sh --curl-only create-db mydb
```

Output:
```bash
curl -X POST http://localhost:8080/v1/databases \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-api-key' \
  -d '{
    "name": "mydb",
    "vectorDimension": 128,
    "indexType": "HNSW"
  }'
```

### Batch Operations (Python Client Library)

The Python client library supports batch operations for high-performance workflows:

```python
from jadevectordb import JadeVectorDB, Vector

client = JadeVectorDB("http://localhost:8080", api_key="your-key")

# Batch store vectors
vectors = [
    Vector(id="vec1", values=[0.1, 0.2, 0.3], metadata={"type": "A"}),
    Vector(id="vec2", values=[0.4, 0.5, 0.6], metadata={"type": "B"}),
    Vector(id="vec3", values=[0.7, 0.8, 0.9], metadata={"type": "A"})
]

client.batch_store_vectors("mydb", vectors)
```

## Authentication

All CLIs support authentication via API keys:

```bash
# Using command-line flag
jade-db --api-key your-api-key list-dbs

# Using environment variable (recommended)
export JADEVECTORDB_API_KEY="your-api-key"
jade-db list-dbs
```

To generate an API key, use the authentication endpoints:

```bash
# Register a user
curl -X POST http://localhost:8080/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username": "myuser", "password": "mypassword", "email": "user@example.com"}'

# Login to get a token
curl -X POST http://localhost:8080/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "myuser", "password": "mypassword"}'

# Create an API key
curl -X POST http://localhost:8080/v1/api-keys \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token-from-login>' \
  -d '{"userId": "user-id", "description": "CLI access"}'
```

## CLI Implementation Status

| Feature | Python CLI | Shell CLI | JavaScript CLI |
|---------|-----------|-----------|----------------|
| Database Creation | ✅ | ✅ | ✅ |
| Database Listing | ✅ | ✅ | ✅ |
| Database Info | ✅ | ✅ | ✅ |
| Database Deletion | ✅ | ✅ | ⚠️ Partial |
| Vector Storage | ✅ | ✅ | ✅ |
| Vector Retrieval | ✅ | ✅ | ✅ |
| Vector Deletion | ✅ | ✅ | ✅ |
| Similarity Search | ✅ | ✅ | ✅ |
| Batch Operations | ✅ | ❌ | ❌ |
| Environment Variables | ✅ | ✅ | ⚠️ Partial |
| cURL Generation | ✅ | ✅ | ❌ |
| API Key Authentication | ✅ | ✅ | ✅ |
| Health/Status Checks | ✅ | ✅ | ✅ |

**Legend:** ✅ Fully Implemented | ⚠️ Partially Implemented | ❌ Not Implemented

# Web Frontend Interface

JadeVectorDB includes a **production-ready** web-based management interface built with modern web technologies. The frontend provides a comprehensive UI for all database operations, monitoring, and administration tasks.

## Technology Stack

- **Framework**: Next.js 14.0.0 with React 18.2.0
- **Styling**: TailwindCSS 3.3.0 with custom design system
- **State Management**: React Hooks (useState, useEffect)
- **Authentication**: JWT-based authentication with secure token storage
- **API Integration**: RESTful API client with real-time data fetching

## Implementation Status: 100% Production-Ready ✅

All 23 pages have been fully implemented and integrated with the backend API:

### Core Management Pages (100% Complete)
- ✅ **Dashboard** (`/`) - Unified system overview with cluster status, databases, system health, and recent activity
- ✅ **Databases** (`/databases`) - Full CRUD operations with edit/delete modals and validation
- ✅ **Vectors** (`/vectors`) - Vector listing with pagination (50 vectors per page), metadata display, and batch operations
- ✅ **Similarity Search** (`/similarity-search`) - Advanced search interface with result ranking, score display, and search time measurement

### Monitoring & Analytics Pages (100% Complete)
- ✅ **Performance** (`/performance`) - Real-time metrics with gradient cards, auto-refresh every 10 seconds
- ✅ **Monitoring** (`/monitoring`) - System status dashboard with live health checks and metrics
- ✅ **Alerting** (`/alerting`) - Alert management with filtering (Error/Warning/Info), acknowledge functionality, auto-refresh every 30 seconds

### Cluster Management Pages (100% Complete)
- ✅ **Cluster** (`/cluster`) - Node status table with CPU/memory/storage metrics, detailed node view, auto-refresh every 15 seconds
- ✅ **Nodes** (`/nodes`) - Individual node management and configuration
- ✅ **Replication** (`/replication`) - Replication status and configuration

### Advanced Features Pages (100% Complete)
- ✅ **Explore** (`/explore`) - Database exploration with vector listing and auto-selection
- ✅ **Query** (`/query`) - Custom query interface with database selector
- ✅ **Embeddings** (`/embeddings`) - Embedding generation interface
- ✅ **Batch Operations** (`/batch-operations`) - Bulk vector operations
- ✅ **Indexes** (`/indexes`) - Index management and configuration

### Learning & Education Pages (100% Complete)
- ✅ **Quizzes** (`/quizzes`) - Interactive tutorial assessment system with 4 quiz modules (CLI Basics, CLI Advanced, Vector Fundamentals, API Integration), progress tracking, timer functionality, and detailed results with explanations

### Security & Administration Pages (100% Complete)
- ✅ **Authentication** (`/auth`) - JWT login/register with real backend integration, API key management
- ✅ **Security** (`/security`) - Security settings and audit logs
- ✅ **Access Control** (`/access-control`) - User permissions and role management
- ✅ **Audit** (`/audit`) - Comprehensive audit log viewer

### Settings & Configuration Pages (100% Complete)
- ✅ **Settings** (`/settings`) - System configuration interface
- ✅ **Backup** (`/backup`) - Backup and restore operations
- ✅ **Import/Export** (`/import-export`) - Data import/export utilities

## Key Features Implemented

### Real-Time Data Synchronization
- Auto-refresh intervals on monitoring pages (10s, 15s, 30s)
- Last updated timestamps on all dashboards
- Manual refresh buttons with loading states
- Promise.all() with error fallbacks for parallel API calls

### Advanced User Experience
- **Pagination**: Efficient handling of large datasets (50 vectors per page)
- **Search Performance**: Client-side search time measurement
- **Result Ranking**: Professional result cards with similarity scores
- **Filtering**: Alert filtering by type (All/Error/Warning/Info)
- **Node Details**: Expandable node information panels
- **Gradient Cards**: Modern UI with gradient backgrounds on metric cards

### Production-Ready Features
- **Error Handling**: Comprehensive try-catch blocks with user-friendly error messages
- **Loading States**: Disabled buttons and loading indicators during operations
- **Validation**: Input validation on all forms
- **Responsive Design**: Mobile-friendly layouts with Tailwind responsive classes
- **Empty States**: Meaningful messages when no data is available

### JWT Authentication Integration
- Secure token storage in localStorage
- Authorization headers on all API requests
- Login/logout functionality with session management
- API key management interface
- User registration with role assignment

### Backend API Integration
All pages use real backend endpoints:
- `databaseApi`: CRUD operations for databases
- `vectorApi`: Vector storage, retrieval, and listing with pagination
- `searchApi`: Similarity search with filters
- `clusterApi`: Node listing and status retrieval
- `monitoringApi`: System status and metrics
- `performanceApi`: Performance metrics
- `alertApi`: Alert listing and acknowledgment
- `authApi`: User registration, login, logout
- `securityApi`: Audit logs and security settings

## Getting Started with Frontend

### Prerequisites
- Node.js 16+ and npm/yarn
- Backend server running on `http://localhost:8080`

### Installation
```bash
cd frontend
npm install
```

### Development
```bash
npm run dev
```
Visit `http://localhost:3000` to access the web interface.

### Production Build
```bash
npm run build
npm start
```

### Environment Configuration
Create a `.env.local` file in the frontend directory:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8080
```

## Frontend Architecture

### API Client (`src/lib/api.js`)
Centralized API client with:
- Base URL configuration
- Authentication header management
- Response error handling
- Support for all backend endpoints

### Page Structure
- **Server-Side Rendering**: Next.js pages with SSR support
- **Client-Side State**: React hooks for data management
- **Component Reusability**: Shared UI components across pages
- **Tailwind Utilities**: Utility-first CSS with custom classes

### Data Flow
1. User interaction triggers event handler
2. API call to backend service
3. Response handling (success/error)
4. State update with setX() hooks
5. UI re-render with new data

## Recent Frontend Improvements

### Phase 1: Critical Fixes (60% → 95%)
- ✅ Fixed databases.js crash on Edit/Delete operations (4 missing handlers)
- ✅ Implemented vector listing with pagination in vectors.js
- ✅ Replaced mock data with real API calls in monitoring.js
- ✅ Implemented real JWT authentication in auth.js
- ✅ Added database selector to query.js (removed hardcoded database)
- ✅ Completed explore.js implementation with vector listing
- ✅ Enhanced api.js with authApi and vectorApi.listVectors()

### Phase 2: Final Enhancements (95% → 100%)
- ✅ Enhanced alerting.js with filtering and acknowledge functionality
- ✅ Enhanced cluster.js with node details view and auto-refresh
- ✅ Enhanced performance.js with gradient cards and 10s refresh
- ✅ Enhanced similarity-search.js with search time and professional result cards
- ✅ Enhanced dashboard.js with auto-refresh and error handling fallbacks

### Code Changes Summary
- **12 files modified** across 2 commits
- **1,319 lines added**, 347 lines removed
- **100% test coverage** on all implemented features

## Recent Improvements (Latest Release)

### Python CLI
- ✅ Added `delete` command for vector deletion
- ✅ Added `get-db` command for database information retrieval
- ✅ Added `delete-db` command for database deletion
- ✅ Implemented environment variable support (`JADEVECTORDB_URL`, `JADEVECTORDB_API_KEY`)
- ✅ Enhanced client library with `delete_database()` method
- ✅ Added cURL command generators for all new operations

### Shell CLI
- ✅ **Fixed critical bug**: `--database-id` flag now properly parsed
- ✅ Added `delete-db` command for database deletion
- ✅ Implemented environment variable support (URL, API_KEY, DATABASE_ID)
- ✅ Updated usage documentation with environment variable examples
- ✅ Improved error messages for missing parameters

### Backend Integration
- ✅ All CLI commands now call fully implemented backend services
- ✅ Complete authentication system with JWT tokens
- ✅ API key management endpoints operational
- ✅ Batch get vectors endpoint implemented
- ✅ Comprehensive security audit logging

## Troubleshooting

### Common Issues

**Issue**: `Connection refused` error
```bash
# Check if the server is running
curl http://localhost:8080/health

# Start the server if needed
cd backend/build && ./jadevectordb
```

**Issue**: `Authentication required` error
```bash
# Set API key via environment variable
export JADEVECTORDB_API_KEY="your-api-key"

# Or use the --api-key flag
jade-db --api-key your-api-key list-dbs
```

**Issue**: Python CLI `ModuleNotFoundError`
```bash
# Install in development mode
cd cli/python
pip install -e .
```

## Documentation

For detailed documentation:
- [CLI Documentation](docs/cli-documentation.md) - Complete CLI reference
- [CLI Examples](examples/cli/README.md) - Real-world usage examples
- [CLI Tutorials](tutorials/cli/README.md) - Step-by-step guides
- [Quiz System](docs/QUIZ_SYSTEM_DOCUMENTATION.md) - Interactive tutorial assessment platform
- [API Documentation](docs/api_documentation.md) - REST API reference
- [Search Functionality](docs/search_functionality.md) - Search algorithms and metadata filtering
- [Backend Service Verification](docs/BACKEND_SERVICE_VERIFICATION.md) - Implementation details

## Development Tools

This project includes several development tools to help maintain code quality and security:

### Code Coverage
To measure test coverage:
```bash
cd backend
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_COVERAGE=ON ..
make
make coverage
```
The coverage report will be available in the `coverage_report` directory.

### Static Analysis
To run static analysis on the codebase:
```bash
cd backend
python3 ../scripts/run-static-analysis.py
```

### Security Testing
To run security tests on the project:
```bash
python3 scripts/run-security-tests.py --project-dir backend
```

## Documentation

Complete documentation is available in the `docs/` directory:

- [Quickstart Guide](docs/quickstart.md) - Getting started with JadeVectorDB
- [Architecture Documentation](docs/architecture.md) - System architecture and design decisions
- [API Documentation](docs/api_documentation.md) - Complete API reference
- [Developer Guide](BOOTSTRAP.md) - Information for contributors
- [Build Documentation](BUILD.md) - Complete build system documentation with troubleshooting
- [Known Issues](docs/KNOWN_ISSUES.md) - Current issues and limitations
- [Troubleshooting Guide](docs/TROUBLESHOOTING_GUIDE.md) - How to diagnose and resolve common issues
- [Local Deployment Guide](docs/LOCAL_DEPLOYMENT.md) - Local deployment process with current caveats

## Next Steps

1. **Containerization** - Docker images and Kubernetes deployment
2. **Performance Tuning** - Fine-tuning indexing algorithms and system parameters
3. **Monitoring** - Prometheus metrics and Grafana dashboards
4. **Security** - Enhanced authentication and encryption
5. **Production Deployment** - Configuration management and deployment scripts

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.