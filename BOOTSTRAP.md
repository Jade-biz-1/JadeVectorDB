# JadeVectorDB - Bootstrap Guide for New Sessions

## 🎯 Purpose
This document helps you (Claude) quickly get up to speed when starting a new session with this project.

---

## ✅ LATEST STATUS (February 12, 2026)

**Build Status**: ✅ PASSING (fast, incremental builds)
**Automated Tests**: ✅ 16/16 test suites passing (all pre-existing failures resolved)
**Overall Completion**: 100% (360/360 tasks) - Production Ready
**Current Sprint**: All phases complete (Phase 16 completed Jan 28, 2026)
**Distributed System**: ✅ Fully implemented (12,259+ lines) — disabled by default for Phase 1
**CLI Tools**: ✅ Operational (cluster_cli.py with 10 commands)
**Binary**: ✅ Functional (4.0M executable + 8.9M core library)
**Server**: ✅ Runs on port 8080 with configurable thread pool
**Status**: **Production Ready (single-node validated). Distributed features available and can be enabled via configuration (Phase 2 planned)**

### What Works:
- ✅ Main library compiles in 5 seconds
- ✅ All distributed services implemented (health monitoring, live migration, failure recovery, load balancing, backup/restore)
- ✅ Advanced persistence features complete: WAL, snapshots, statistics, integrity verification, index resize, free list
- ✅ CLI management tools functional
- ✅ 60+ documentation files complete
- ✅ Docker deployment configs ready
- ✅ Code quality standards met (Result<T> error handling, extensive logging)

### Known Issues:
- ✅ All 16/16 test suites passing - pre-existing test failures fully resolved (Feb 2026)
- ✅ **Auth test failures** - FIXED (2026-02-12) - DB/memory mismatch in AuthenticationService, deadlock in persistence layer
- ✅ **Runtime crash on startup** - FIXED (2025-12-12) - Singleton ownership issue resolved
- ✅ **Application hanging on Ctrl-C** - FIXED (2025-12-26) - Signal handler now calls immediate exit
- See `CleanupReport.md` and `TasksTracking/SPRINT_2_3_TEST_RESULTS.md` for test details

---

## 🚨 ARCHITECTURE POLICY: NO DUPLICATE SERVICES

**CRITICAL RULE: Never create duplicate or parallel systems for the same functionality!**

### What This Means:
- **One authentication system only**: Use `AuthenticationService` (services/authentication_service.h)
- **NEVER create alternatives**: Do not create `AuthManager`, `AuthSystem`, `AuthService`, etc.
- **One implementation per feature**: If a service exists, extend it - don't create a new one
- **Check before creating**: Always search for existing implementations before writing new code

### Why This Matters:
- Having two authentication systems caused significant technical debt
- Wasted development time maintaining duplicate code
- Confusion about which system to use
- Difficult migration and cleanup work

### The AuthManager Lesson (RESOLVED ✅):
In December 2025, we discovered two parallel authentication systems:
1. `lib/auth.h` / `lib/auth.cpp` - `AuthManager` class (old, **DELETED**)
2. `services/authentication_service.h` / `services/authentication_service.cpp` - `AuthenticationService` class (current, **USE THIS**)

This duplication was **completely unwanted** and required extensive cleanup work. **The cleanup is now complete** (2025-12-12):
- ✅ Deleted `lib/auth.h` and `lib/auth.cpp`
- ✅ Removed all AuthManager references from source files
- ✅ Fixed double-free crash caused by singleton ownership issue
- ✅ Valgrind clean (0 errors)

**Action Required**: Before implementing ANY new service or major feature:
1. Search the codebase for existing implementations
2. If found, extend or refactor the existing code
3. If creating new, use clear, unique names and document in BOOTSTRAP.md
4. Never create "alternative" or "parallel" implementations

---

## ⚠️ CRITICAL: Build System

**NEVER use `make` or `cmake` directly! ALWAYS use the build script!**

### The ONLY Correct Way to Build:

```bash
# Must run from backend/ directory
cd backend
./build.sh
```

### Common Build Commands:

```bash
# Standard build (Release, with tests and benchmarks)
cd backend && ./build.sh

# Clean build (removes build directory first)
cd backend && ./build.sh --clean

# Debug build
cd backend && ./build.sh --type Debug --clean

# Production build (no tests/benchmarks, optimized)
cd backend && ./build.sh --no-tests --no-benchmarks

# Fast build (limit parallel jobs)
cd backend && ./build.sh --jobs 2
```

### Build Script Must-Knows:

1. **Always run from `backend/` directory**
2. **Script location**: `backend/build.sh`
3. **Output**: `backend/build/jadevectordb` (executable)
4. **First build**: ~12 minutes (fetches dependencies)
5. **Incremental**: ~1 minute
6. **All dependencies built from source** (no apt-get needed!)


### Known Build Issues:

✅ **Tests now compile and pass** - All 16/16 test suites operational (Feb 2026):
```bash
cd backend && ./build.sh
```

✅ **Runtime crash on startup** - FIXED (2025-12-12)
- Was caused by singleton ownership issue (ConfigManager/MetricsRegistry wrapped in unique_ptr)
- Now builds and runs cleanly with proper shutdown

---

## 🧪 CLI Testing System

JadeVectorDB provides a comprehensive CLI testing suite for validating end-to-end functionality.

### Quick Start

1. **Start the server:**
  ```bash
  cd backend/build
  ./jadevectordb
  ```
2. **In a new terminal, run all CLI tests:**
  ```bash
  python3 tests/run_cli_tests.py
  # or use the shell wrapper
  ./tests/run_tests.sh
  ```

### Test Data
All test data is centralized in `tests/test_data.json`:
- Authentication: test user credentials
- Databases: test database configurations
- Vectors: test vector specifications
- Search: search query parameters

### Output Format
The test runner prints a summary table with pass/fail status for each test. Example:
```
================================================================================
#     Tool            Test                           Result
================================================================================
1     Python CLI      Health Check                   ✓ PASS
2     Python CLI      Status Check                   ✓ PASS
3     Python CLI      Create Database                ✓ PASS
...
================================================================================

Summary: 11/12 tests passed
  Failed: 1
```

### Troubleshooting
If a test fails, the runner provides hints. Common issues:
- Server not running or listening on wrong port
- Test data mismatch (see `tests/test_data.json`)
- Authentication or password requirements not met

See `tests/README.md` for full details and troubleshooting tips.

---

---

## 🧪 CLI Testing System

**IMPORTANT**: JadeVectorDB has a unified, comprehensive CLI testing suite covering all CLI functionality including Phase 16 features (user management and bulk import/export).

### Quick Start

```bash
# 1. Start the server
cd backend/build
./jadevectordb

# 2. In a new terminal, run all CLI tests
cd /path/to/JadeVectorDB

# Recommended: Use the master test runner
./tests/run_all_tests.sh

# Or run the test suite directly
python3 tests/run_cli_tests.py
```

### Test Data Configuration

All test data is centralized in `tests/test_data.json`:

- **Authentication**: Test user credentials (username, password, email)
- **Databases**: Test database configurations (name, dimension, index type)
- **Vectors**: Test vector specifications (auto-generated values)
- **Search**: Search query parameters

**Password Requirements**: Test passwords must meet security standards:
- At least 10 characters (updated from 8)
- Contains uppercase, lowercase, digit, and special character
- Example: `CliTest123@`

### Test Output Format

```
================================================================================
#     Tool            Test                           Result
================================================================================
1     Python CLI      Health Check                   ✓ PASS
2     Python CLI      Status Check                   ✓ PASS
...
21    User Mgmt       Add User                       ✓ PASS
22    User Mgmt       List Users                     ✓ PASS
27    Import/Export   Export Vectors                 ✓ PASS
28    Import/Export   Import Vectors                 ✓ PASS
================================================================================

Summary: 36/36 tests passed
```

### Troubleshooting Failed Tests

The test runner provides specific hints for failures:

```
[Test #21] User Mgmt - Add User:
  • Verify user management API endpoints are accessible
  • Check if admin privileges are required
  • Review authentication service logs
```

### Test Coverage (36 Tests)

- **Python CLI Tests** (Tests 1-7): Health, Status, List DBs, Create DB, Get DB, Store Vector, Search
- **Shell CLI Tests** (Tests 8-12): Health, Status, List DBs, Create DB, Get DB
- **Persistence Tests** (Tests 13-15): User persistence, database persistence
- **RBAC Tests** (Tests 16-20): Role-based access control
- **Phase 16 - Python User Management** (Tests 21-26): Add, list, show, activate, deactivate, delete users
- **Phase 16 - Python Import/Export** (Tests 27-28): Export and import vectors in bulk
- **Phase 16 - Shell User Management** (Tests 29-34): Add, list, show, activate, deactivate, delete users
- **Phase 16 - Shell Import/Export** (Tests 35-36): Export and import vectors in bulk

**Note**: JavaScript CLI tests use Jest and are run separately: `cd cli/js && npm test`

### Test Files

- **Master Test Runner**: `tests/run_all_tests.sh` - Runs all tests
- **Test Suite**: `tests/run_cli_tests.py` - Comprehensive test implementation
- **Test Data**: `tests/test_data.json` - Centralized test configuration
- **Documentation**: `tests/README.md` - Full testing guide

---

## 🧪 UNIT TESTING ORGANIZATION

**CRITICAL: All unit test code belongs in dedicated testing directories!**

### Unit Test Locations

- **Backend Unit Tests**: `backend/unittesting/`
  - All persistence layer unit tests (test_sprint_*.cpp)
  - Test executables (compiled binaries)
  - Comprehensive test documentation (README.md)
  - 76 tests across Sprint 1.2, 1.3, and 1.4
  
- **Backend Helper Scripts**: `backend/scripts/`
  - Python helper scripts (fix_*.py)
  - Build and maintenance utilities

### Rules for Test Code

1. ❌ **NEVER** create test files in `backend/` root directory
2. ✅ **ALWAYS** place unit tests in `backend/unittesting/`
3. ✅ **ALWAYS** place helper scripts in `backend/scripts/`
4. ✅ **UPDATE** `backend/unittesting/README.md` when adding new tests
5. ✅ **USE** consistent naming: `test_sprint_X_Y.cpp` for persistence tests

### Why This Matters

- Keeps backend root directory clean and organized
- Makes it easy to find and run all unit tests
- Separates test code from production code
- Professional project structure
- Easy onboarding for new developers

**See**: `backend/unittesting/README.md` for complete test documentation and build instructions

---

## 💾 DATA PERSISTENCE ARCHITECTURE (Implemented)

### Current Status: PERSISTENT STORAGE IMPLEMENTED

**✅ NOTE**: The system now uses a hybrid persistent storage architecture. Persistence features (WAL, snapshots, memory-mapped vector files, index resize protection, free-list management, integrity verification) have been implemented and tested (Sprint 2.3).

### Hybrid Storage Architecture

JadeVectorDB uses a **two-tier hybrid storage** system:

**Tier 1: SQLite Database** (`/var/lib/jadevectordb/system.db`)
- **Purpose**: Transactional metadata with ACID guarantees
- **Contents**: 
  - User accounts (username, password hash, email)
  - Groups and group memberships
  - Roles and permissions (RBAC system)
  - API keys and authentication tokens
  - Sessions and audit logs
  - Database metadata and configurations
  - Index metadata and build status

**Tier 2: Memory-Mapped Files** (`/var/lib/jadevectordb/databases/{db_id}/vectors.mmap`)
- **Purpose**: High-performance vector data storage
- **Format**: SIMD-aligned binary format
- **Access**: Zero-copy via OS page caching
- **Benefits**: Handles GB-TB datasets efficiently

### Directory Structure

```
/var/lib/jadevectordb/
├── system.db              # SQLite: All metadata
└── databases/
    ├── {uuid-1}/
    │   ├── vectors.mmap   # Vector embeddings
    │   └── indexes/       # Index files (HNSW, IVF, etc.)
    ├── {uuid-2}/
    │   ├── vectors.mmap
    │   └── indexes/
    └── ...
```

### Configuration

**Environment Variable**:
```bash
export JADEVECTORDB_DATA_DIR=/var/lib/jadevectordb
```

**Config File** (`backend/config/jadevectordb.conf`):
```ini
[storage]
data_directory=/var/lib/jadevectordb
sqlite_wal_mode=true
sqlite_checkpoint_interval=1000
vector_sync_interval_sec=5
vector_sync_on_write=false

[security]
enable_rbac=true
api_key_expiry_days=365
session_timeout_minutes=60
```

### Persistence Classes

**Key Components**:
- `SQLitePersistenceLayer` - Manages SQLite operations
- `HybridDatabasePersistence` - Orchestrates SQLite + mmap
- `MemoryMappedVectorStore` - Per-database vector file management
- `AuthenticationService` - Enhanced with SQLite backing

**Interface**: `DatabasePersistenceInterface` (unchanged for backward compatibility)

### Durability Guarantees

**SQLite (Metadata)**:
- Write-Ahead Logging (WAL) mode
- Atomic commits with crash recovery
- Checkpoint every 1000 transactions or 5 minutes
- **Guarantee**: No metadata loss on crash

**Memory-Mapped Files (Vectors)**:
- Periodic sync every 5 seconds (configurable)
- Sync on graceful shutdown
- **Trade-off**: May lose last few seconds on power failure
- Set `vector_sync_on_write=true` for immediate durability (slower)

### Implementation Timeline

**Phase 1+2** (Weeks 1-3): SQLite + RBAC
- Users, groups, roles, permissions
- API keys and session management
- Database metadata persistence
- Audit logging

**Phase 3** (Weeks 4-5): Vector Persistence
- Memory-mapped file implementation
- Vector serialization and SIMD alignment
- Integration with search operations

**See**: `TasksTracking/11-persistent-storage-implementation.md` for details

---

## 📚 Essential Documentation Files

**Read these in order when starting a session:**

1. **`BUILD.md`** (root) - **CRITICAL: Main build system guide - READ THIS for all build-related information**
2. **`docs/COMPLETE_BUILD_SYSTEM_SETUP.md`** - Complete build system overview
3. **`backend/BUILD_QUICK_REFERENCE.md`** - Quick command reference
4. **`docs/archive/RECOVERY_SUMMARY_2025-12-03.md`** - Last session status (if exists)
5. **`TasksTracking/status-dashboard.md`** - **Current focus and recent work - CHECK THIS FIRST**
6. **`TasksTracking/README.md`** - **Task tracking system guide - ALWAYS update status when completing tasks**
7. **`TasksTracking/overview.md`** - **Overall project progress and completion status**

### Quick Reference:

| File | Purpose |
|------|---------|
| `BUILD.md` (root) | **Main build documentation - READ THIS FIRST for builds** |
| `docs/COMPLETE_BUILD_SYSTEM_SETUP.md` | Build system overview & features |
| `backend/BUILD_QUICK_REFERENCE.md` | Quick build commands |
| `backend/build.sh` | **THE BUILD SCRIPT** |
| `docs/archive/RECOVERY_SUMMARY_2025-12-03.md` | Last session recovery summary |
| `BOOTSTRAP.md` | **THIS FILE - Complete developer onboarding and project overview** |
| `TasksTracking/status-dashboard.md` | **Current focus and status - CHECK THIS** |
| `TasksTracking/overview.md` | Overall progress and milestones |
| `TasksTracking/06-current-auth-api.md` | **Current work: Auth & API tasks** |
| `TasksTracking/README.md` | Task tracking system organization |

---

## 📐 CRITICAL: Specification & Architecture Documents

**⚠️ READ THESE BEFORE MAKING ANY DESIGN DECISIONS!**

The `specs/002-check-if-we/` folder contains **foundational documents** that define the architecture, research decisions, and implementation plan for JadeVectorDB. These documents should be consulted BEFORE:
- Adding new features
- Making architectural decisions
- Implementing distributed system components
- Choosing algorithms or data structures

### Core Specification Documents:

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `specs/002-check-if-we/spec.md` | **Master specification** (1162 lines) - Complete feature requirements, user stories, acceptance criteria | Before implementing any new feature |
| `specs/002-check-if-we/architecture/architecture.md` | **System architecture** - Master-worker pattern, component interactions, data flow diagrams | Before making any structural changes |
| `specs/002-check-if-we/plan.md` | **Implementation plan** - Tech stack, constitution check, project structure | Before adding dependencies or changing structure |
| `specs/002-check-if-we/research.md` | **Technical research decisions** - Indexing algorithms, embedding models, distributed patterns | Before implementing algorithms |
| `specs/002-check-if-we/data-model.md` | **Data model specification** - Entity definitions, relationships | Before modifying data structures |
| `specs/002-check-if-we/IMPLEMENTATION_PLAN_SUMMARY.md` | **Phase summary** - 196 tasks across 12 phases, dependencies | For understanding task relationships |

### Research Documents (specs/002-check-if-we/research/):

These contain detailed technical research. **Consult before implementing related features:**

| Research Doc | Topic | Key Decisions |
|--------------|-------|---------------|
| `001-vector-indexing-algorithms.md` | HNSW, IVF, LSH algorithms | IVF+PQ for distributed, HNSW for single-node |
| `002-embedding-models-integration.md` | Embedding providers | Hugging Face, Ollama, OpenAI integration |
| `003-distributed-systems-patterns.md` | Distributed architecture | Master-worker, Raft consensus, sharding |
| `004-performance-optimization.md` | SIMD, caching, batching | Performance best practices |
| `005-industry-comparisons.md` | Comparison with Pinecone, Weaviate, etc. | Feature parity goals |
| `006-security-implementations.md` | Auth, encryption, access control | Security requirements |
| `007-infrastructure-considerations.md` | Deployment, containerization | Docker, K8s patterns |
| `008-monitoring-and-observability.md` | Metrics, logging, alerting | Monitoring strategy |
| `009-data-migration.md` | Data import/export | Migration patterns |
| `010-cpp-implementation-considerations.md` | C++ best practices | Memory management, RAII |
| `011-advanced-data-structures-algorithms.md` | Data structures | Skip lists, B+ trees |
| `012-serialization-memory-management.md` | FlatBuffers, Arrow | Serialization choices |
| `013-cpp-testing-strategies.md` | Testing approaches | Google Test patterns |

### API Contracts:

| Contract | Purpose |
|----------|---------|
| `specs/002-check-if-we/contracts/vector-db-api.yaml` | OpenAPI specification for REST API |
| `specs/002-check-if-we/contracts/examples.json` | API request/response examples |

### Checklists:

| Checklist | Purpose |
|-----------|---------|
| `specs/002-check-if-we/checklists/requirements.md` | Feature requirements checklist |

### Key Architecture Decisions (from research):

1. **Indexing Algorithms**: 
   - Distributed: IVF with Product Quantization (PQ)
   - Single-node: HNSW for speed/accuracy
   - Configurable per database

2. **Embedding Providers** (pluggable architecture):
   - Hugging Face (direct loading)
   - Ollama (local API)
   - OpenAI/Gemini (external API)

3. **Distributed Architecture**:
   - Master-worker pattern
   - Raft consensus for leader election
   - Hash/range-based sharding
   - Eventual/strong consistency configurable

4. **Performance Goals**:
   - Sub-50ms search for 1M vectors
   - 10,000+ vectors/second ingestion
   - 99.9% availability

5. **Tech Stack**:
   - C++20 for backend (mandatory)
   - gRPC for inter-service communication
   - FlatBuffers for serialization
   - Next.js + shadcn for frontend

---

## 📁 Project Structure and Architecture

JadeVectorDB is a distributed vector database with a modular architecture consisting of several key components:

```
JadeVectorDB/
├── backend/                    # Main C++ codebase
│   ├── src/                   # Source code
│   │   ├── api/              # API layer (REST, gRPC)
│   │   ├── services/         # Core services
│   │   ├── models/           # Data models
│   │   └── utils/            # Utilities
│   ├── tests/                # Unit & integration tests
│   ├── build.sh              # ⚠️ THE BUILD SCRIPT - USE THIS!
│   ├── CMakeLists.txt        # CMake configuration
│   └── BUILD*.md             # Build documentation
├── frontend/                  # Next.js Web UI (23+ fully implemented pages)
│   ├── src/                  # React components
│   └── public/               # Static assets
├── cli/                       # CLI tools
│   ├── python/               # Python CLI (full-featured)
│   └── shell/                # Shell CLI (lightweight)
├── docs/                      # Documentation (technical docs, guides)
│   ├── COMPLETE_BUILD_SYSTEM_SETUP.md
│   └── archive/              # Archived documentation
├── docker/                    # Containerization and orchestration
├── scripts/                   # Development and deployment utilities
├── TasksTracking/             # **TASK TRACKING & SPRINT SUMMARIES - SINGLE SOURCE OF TRUTH**
│   ├── README.md             # Task system guide
│   ├── status-dashboard.md   # Current focus (CHECK THIS FIRST!)
│   ├── overview.md           # Overall progress
│   ├── SprintSummary/        # **ALL SPRINT COMPLETION DOCUMENTS GO HERE**
│   │   ├── SPRINT_1_5_SUMMARY.md
│   │   ├── SPRINT_2_1_COMPLETION_SUMMARY.md
│   │   ├── SPRINT_2_2_COMPLETION_SUMMARY.md
│   │   └── (other sprint docs)
│   ├── 01-setup-foundational.md
│   ├── 02-core-features.md
│   ├── 03-advanced-features.md
│   ├── 04-monitoring-polish.md
│   ├── 05-tutorial.md
│   ├── 06-current-auth-api.md  # Current work
│   ├── 07-backend-core.md
│   ├── 08-distributed-completion.md
│   └── 09-distributed-tasks.md
├── specs/                     # Specifications
├── BUILD.md                   # Main build guide (root)
├── docs/archive/RECOVERY_SUMMARY_2025-12-03.md        # Last session status
└── BOOTSTRAP.md               # This file
```

### Key Backend Services

The backend features several core services that handle different aspects of the system:

- **DatabaseService** - Manages database creation, configuration, and lifecycle
- **VectorStorageService** - Handles vector storage and retrieval operations
- **SimilaritySearchService** - Implements various similarity search algorithms (cosine, euclidean, dot product)
- **AuthenticationService** - JWT-based authentication with API key management
- **ClusterService** - Implements Raft-based consensus for master election
- **ShardingService** - Distributes data across cluster nodes using multiple strategies
- **ReplicationService** - Ensures data availability through configurable replication
- **DistributedServiceManager** - Coordinates all distributed services

### Frontend Features

The Next.js web interface provides 23+ fully implemented pages for:
- Database management
- Vector operations
- Similarity search
- Performance monitoring
- Cluster management
- Security and access control

---

## 🚀 Quick Start for New Sessions

### Step 1: Ask the User
```
"Should I load the bootstrap document to understand the project setup?"
```

### Step 2: Check Project Status
```bash
# Check uncommitted changes
git status

# Check recent commits
git log --oneline -5

# Check current branch
git branch
```

### Step 3: Read Recovery Summary (if exists)
```bash
cat docs/archive/RECOVERY_SUMMARY_2025-12-03.md
```

### Step 4: Build the Project (Correct Way!)
```bash
cd backend && ./build.sh --no-tests --no-benchmarks
```

### Step 5: Ask What to Work On
```
"I've reviewed the project status. What would you like to work on today?"
```

---

## 🛠️ Development Environment Setup

### Prerequisites

Before you begin, ensure you have the following tools installed:

- **Git:** For version control
- **C++ Toolchain:** Modern C++ compiler (GCC, Clang, or MSVC) supporting C++20
- **CMake:** Version 3.20 or higher
- **Node.js:** Version 18 or higher (for frontend)
- **Python:** Version 3.8 or higher (for Python CLI)
- **Docker and Docker Compose:** For containerized deployment

**Note**: All C++ dependencies (Eigen, FlatBuffers, Apache Arrow, gRPC, etc.) are automatically fetched and built from source by the build system - no manual installation needed!

You can check prerequisites by running:
```bash
sh .specify/scripts/bash/check-prerequisites.sh
```

### Cloning the Repository

```bash
git clone <repository-url>
cd JadeVectorDB
```

### Frontend Setup (Next.js)

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Access at http://localhost:3000
```

### CLI Setup (Python)

```bash
# Navigate to the Python CLI directory
cd cli/python

# Install in editable mode (recommended for development)
pip install -e .

# Verify installation
jade-db --help
```

---

## 🌐 API Configuration

**Important API Notes:**

- API endpoints may be versioned: Check if API is accessible as `/api/v1/...` instead of just `/api/...`
- Default port: 8080
- REST API uses Crow framework
- gRPC support is optional (disabled by default)

**Example API endpoints to verify:**
```bash
# May be /api/v1/databases or /v1/databases
curl http://localhost:8080/api/v1/health
curl http://localhost:8080/v1/health

# Always check actual routes in backend/src/api/rest/
```

---

## 🔐 Authentication & Security

**CRITICAL: Password Requirements**

The authentication system enforces the following password rules:
- **Minimum length**: 10 characters
- **Strong passwords required** in production mode

**Default User Credentials (Development/Test Only)**

When `JADEVECTORDB_ENV` is set to `development`, `dev`, `test`, `testing`, or `local`, the system automatically creates default users:

| Username | Password | Roles | User ID |
|----------|----------|-------|---------|
| `admin` | `admin123` | admin, developer, user | user_admin_default |
| `dev` | `dev123` | developer, user | user_dev_default |
| `test` | `test123` | tester, user | user_test_default |

**IMPORTANT**: These default users are ONLY created in non-production environments and are NOT created in production mode.

**Authentication Service**

The project uses **AuthenticationService** (located in `backend/src/services/authentication_service.h`) as the single source of truth for authentication:
- User registration and login
- Session management
- API key management
- Password hashing and validation
- Default user seeding (dev/test only)

**Common Authentication Mistakes to Avoid:**
1. ❌ **DON'T** use passwords shorter than 10 characters
2. ❌ **DON'T** use simple passwords like "admin123" (too short!)
3. ❌ **DON'T** expect default users in production mode
4. ✅ **DO** use strong passwords that meet minimum requirements (10+ chars)
5. ✅ **DO** set `JADEVECTORDB_ENV=development` for local testing to enable default users
6. ✅ **DO** use AuthenticationService (NOT AuthManager) for all auth operations

---

## 📊 Current Project Status (as of Feb 12, 2026)

### Overall Progress: 100% complete (360/360 tasks)

### Completion Status:
- ✅ **Core vector database**: 100% complete (US1-US4)
- ✅ **Advanced features**: 100% complete (US5-US9, embedding, lifecycle)
- ✅ **Monitoring & polish**: 100% complete
- ✅ **Build system**: 100% functional (self-contained, all deps from source)
- ✅ **Documentation**: Complete
- ✅ **CLI tools**: 100% complete (Python & shell with cURL generation)
- ✅ **Authentication & API (Phase 14)**: 100% complete (T219-T238 all finished)
- ✅ **Backend Core (Phase 15)**: 100% complete (T239-T253 all finished)
- ✅ **Distributed system**: 100% complete (all foundation and operational features)
- ✅ **Interactive tutorial**: 100% complete (all core features and enhancements)
- ✅ **Frontend UI**: 32 pages fully functional, production build working
- ⚠️ **Frontend testing**: 46% coverage (91/200 tests passing) - needs improvement
- ✅ **Persistence (Sprints 2.1-2.3)**: 100% complete
- ✅ **Phase 16 - Hybrid Search, Re-ranking & Analytics**: 100% complete (22/22 tasks)
  - ✅ **Feature 1: Hybrid Search**: 100% complete (T16.1-T16.8)
  - ✅ **Feature 2: Re-ranking**: 100% complete (T16.9-T16.14)
  - ✅ **Feature 3: Query Analytics**: 100% complete (T16.15-T16.22)
- ✅ **Test Infrastructure**: 16/16 test suites passing (all pre-existing failures resolved Feb 2026)

### Current Branch:
- `main`

### Phase 14: Authentication & API Completion
**Status**: ✅ 100% COMPLETE (20/20 tasks)

All authentication and API tasks completed:
1. **Authentication handlers** - ✅ Complete (T219-T222)
2. **User management** - ✅ Complete (T220)
3. **API key management** - ✅ Complete (T221)
4. **Security audit routes** - ✅ Complete (T222)
5. **Alert/Cluster/Performance routes** - ✅ Complete (T223-T225)
6. **Backend tests** - ✅ Complete (T230-T232)
7. **Frontend tests** - ✅ Complete (T233)
8. **Documentation** - ✅ Complete (T229, T235)
9. **Default user seeding** - ✅ Complete (T236-T237)

### Phase 16: Hybrid Search, Re-ranking, and Query Analytics
**Status**: ✅ **100% COMPLETE (22/22 tasks)** 🎉
**Completion Date**: January 28, 2026
**Last Updated**: January 28, 2026

#### Feature 1: Hybrid Search ✅ COMPLETE (T16.1-T16.8)
Combines vector similarity with BM25 keyword search for improved retrieval quality.

**Completed Components**:
- ✅ **BM25 Scoring Engine** (T16.1) - Tokenization, IDF calculation, configurable parameters
- ✅ **Inverted Index** (T16.2) - In-memory index with fast term lookup (<1ms)
- ✅ **Index Persistence** (T16.3) - SQLite storage with incremental updates
- ✅ **Score Fusion** (T16.4) - RRF and weighted linear fusion algorithms
- ✅ **HybridSearchEngine** (T16.5) - Service orchestration and integration
- ✅ **REST API Endpoints** (T16.6) - 4 new endpoints for hybrid search
- ✅ **CLI Support** (T16.7) - Full command-line interface
- ✅ **Testing & Documentation** (T16.8) - 59/59 unit tests passing (100%)

**Test Results**: 59/59 unit tests + all integration tests passing
**Documentation**: API docs, user guide, architecture - all complete

#### Feature 2: Re-ranking ✅ COMPLETE (T16.9-T16.14)
Cross-encoder models boost search result precision through intelligent re-ranking.

**Completed Components**:
- ✅ **Python Reranking Server** (T16.9) - Subprocess with sentence-transformers
- ✅ **Subprocess Management** (T16.10) - C++/Python IPC communication
- ✅ **RerankingService** (T16.11) - Batch inference and score normalization
- ✅ **Service Integration** (T16.12) - Integrated with hybrid and vector search
- ✅ **REST API Endpoints** (T16.13) - 4 new reranking endpoints
- ✅ **Testing & Documentation** (T16.14) - Tests created, docs complete

**Architecture**: Python subprocess (Phase 1), future: Microservice + ONNX Runtime
**Performance**: ~150-300ms for 100 documents
**Model**: cross-encoder/ms-marco-MiniLM-L-6-v2

#### Feature 3: Query Analytics ✅ **100% COMPLETE (8/8 tasks)**
Track and analyze search queries for optimization.

**All Components Complete** (T16.15-T16.22):
- ✅ **QueryLogger** (T16.15) - Data collection with 15/15 tests passing
- ✅ **Analytics Database Schema** (T16.16) - Complete SQLite schema
- ✅ **Query Interception** (T16.17) - QueryAnalyticsManager integration (10/10 tests)
- ✅ **AnalyticsEngine** (T16.18) - Insights generation with 15/15 tests passing
- ✅ **Batch Processor** (T16.19) - Background jobs with 15/15 tests passing
- ✅ **REST API Endpoints** (T16.20) - 7 new analytics endpoints
- ✅ **Analytics Dashboard** (T16.21) - Full web UI with Recharts visualization
- ✅ **Testing & Documentation** (T16.22) - Integration tests (7/7) + 2,900 lines docs

**Dashboard Features**:
- Key metrics cards (total queries, success rate, avg latency, QPS)
- Time-series charts (queries over time, latency distribution)
- Tabbed interface (Overview, Query Explorer, Patterns, Insights)
- Automated insights with color-coded recommendations
- Query patterns analysis, slow queries detection, trending queries
- Database selector, time range picker (1h, 24h, 7d, 30d)
- Auto-refresh every 30 seconds

**Backend**: All 7 REST API endpoints implemented and functional
**Frontend**: Analytics dashboard page created (1048 lines) with Recharts library
**Testing**: 121/121 analytics tests passing (100% coverage)
**Documentation**: 4 comprehensive guides (2,900+ lines total)
  - API Reference (930 lines)
  - Dashboard User Guide (690 lines)
  - Metrics Interpretation Guide (630 lines)
  - Privacy & Retention Policy (650 lines)

### Frontend Status (Updated Jan 13, 2026):
**Build Status**: ✅ Production build working (all 32 pages compile successfully)
**Dependencies**: ✅ All installed (949 packages)
**Test Status**: ⚠️ 46% coverage (91/200 tests passing)

**Recent Frontend Fixes (Jan 13, 2026):**
1. ✅ Fixed JSX syntax errors in 5 pages (advanced-search, embeddings, indexes, query, similarity-search)
2. ✅ Installed lucide-react icon library
3. ✅ Created missing badge.js UI component
4. ✅ Fixed tsconfig.json path alias configuration (@/* now resolves to ./src/*)
5. ✅ Reduced npm vulnerabilities from 7 to 3 (remaining: glob transitive dependency)
6. ✅ All 32 pages building successfully in production mode

**Frontend Testing Gaps:**
- Only 2/30 pages have dedicated unit tests (6.7% page coverage)
- 28 pages need unit tests: dashboard, databases, vectors, users, monitoring, etc.
- Integration tests need fixes (element selection issues)
- Missing: accessibility tests, performance tests
- See `frontend/TESTING_IMPLEMENTATION_PLAN.md` for detailed improvement plan

**Frontend Known Issues:**
1. **LocalStorage SSR warnings** - tutorials page uses localStorage during SSR (non-critical)
2. **Test failures** - 109/200 tests failing (mainly integration test issues)
3. **npm vulnerabilities** - 3 high severity (glob via eslint-config-next, cannot auto-fix)

### Backend Known Issues:
**No critical issues** - Backend is production-ready!

**Minor Notes:**
1. **Backend Tests**: 16/16 test suites passing (all pre-existing failures resolved Feb 2026)
2. **Backend Build**: Tests now compile and pass; use `--no-tests --no-benchmarks` only for fastest builds
3. ~~**Database ID mismatch**~~ - May need verification
4. ~~**AuthManager cleanup**~~ - ✅ COMPLETE (2025-12-12)
5. ~~**distributed_worker_service.cpp stubs**~~ - ✅ COMPLETE (T259)
6. ~~**Auth test failures (14 tests)**~~ - ✅ FIXED (2026-02-12) - DB/memory mismatch + deadlock in persistence layer

### Recent Work (February 2026):

**Test Infrastructure Fixes** (February 12, 2026):
- ✅ Fixed 14 authentication test failures (DB/memory map mismatch in AuthenticationService)
- ✅ Added `get_role_id_by_name()` and `create_role()` to SQLitePersistenceLayer
- ✅ Fixed deadlock in `assign_role_to_user()` / `revoke_role_from_user()` (non-recursive mutex)
- ✅ Fixed QueryLogger writer thread wake condition (batch timeout regression)
- ✅ Fixed RerankingService and RerankingIntegration tests
- ✅ Isolated test fixtures with unique temp directories
- ✅ All 16/16 test suites now passing

**Cross-Platform Build Support** (February 2026):
- ✅ Added macOS and Windows build support

### Recent Work (January 2026):

**Phase 16 - Hybrid Search Implementation** (T16.1-T16.8):
- ✅ Implemented BM25 scoring engine with configurable parameters (T16.1)
- ✅ Built inverted index with fast term lookup (T16.2)
- ✅ Added SQLite-based index persistence (T16.3)
- ✅ Implemented RRF and weighted linear score fusion (T16.4)
- ✅ Created HybridSearchEngine service (T16.5)
- ✅ Added 4 REST API endpoints for hybrid search (T16.6)
- ✅ Implemented CLI commands (T16.7)
- ✅ Completed testing: 59/59 unit tests passing (T16.8)
- ✅ Fixed min-max normalization edge case (identical scores → 1.0)

**Phase 16 - Re-ranking Implementation** (T16.9-T16.14):
- ✅ Created Python reranking server with sentence-transformers (T16.9)
- ✅ Implemented C++ subprocess management (T16.10)
- ✅ Built RerankingService with batch inference (T16.11)
- ✅ Integrated with hybrid and vector search (T16.12)
- ✅ Added 4 REST API endpoints for re-ranking (T16.13)
- ✅ Completed documentation and testing (T16.14)

**Frontend Improvements** (Jan 13, 2026):
- ✅ **Fixed production build** - All 32 pages now compile successfully
- ✅ **Fixed JSX syntax errors** - 5 pages corrected (advanced-search, embeddings, indexes, query, similarity-search)
- ✅ **Installed dependencies** - lucide-react icon library added
- ✅ **Created missing component** - badge.js UI component implemented
- ✅ **Fixed path aliases** - tsconfig.json corrected for proper module resolution
- ✅ **Security improvements** - Reduced npm vulnerabilities from 7 to 3
- ✅ **Test suite verification** - 91/200 tests passing (46% coverage)
- ✅ **Created testing plan** - Comprehensive 7-week testing implementation roadmap

**Backend** (December 2025):
- ✅ Completed all authentication endpoints (T219-T222)
- ✅ Fixed ShardState enum comparison errors
- ✅ **AuthManager cleanup COMPLETE** - Deleted lib/auth.h, lib/auth.cpp (2025-12-12)
- ✅ **Fixed double-free crash** - Singleton ownership issue in main.cpp (2025-12-12)
- ✅ **Valgrind clean** - 0 errors on shutdown (2025-12-12)
- ✅ **Fixed Ctrl-C hanging** - Signal handler now exits immediately (2025-12-26)

---

## 🔧 Development Patterns and Best Practices

### Error Handling
```cpp
// Return errors using Result<T> type
return tl::make_unexpected(ErrorHandler::create_error(
    ErrorCode::INTERNAL_ERROR,
    "Error message"
));
```

### Logging
```cpp
// Initialize logger
logger_ = logging::LoggerManager::get_logger("ServiceName");

// Use logger
logger_->info("message");
logger_->error("error message");
```

### Type Conventions
- Use `SearchResults` (plural) for result structs
- Use `vector_count` not `record_count`
- Use `primary` not `is_primary`

### Code Organization

The project follows a modular architecture pattern:
- Each service is implemented in its own file with clear interfaces
- Service dependencies are managed through dependency injection
- Configuration is centralized with environment variable support
- Error handling follows a consistent Result<T> pattern

### Testing Strategy

The project maintains high test coverage with:
- Unit tests for individual components (80%+ coverage target)
- Integration tests for service interactions
- End-to-end tests for complete workflows
- Chaos tests for failure scenarios
- Performance benchmarks for critical operations

**Running Tests:**
```bash
# Backend tests (when test compilation is fixed)
cd backend/build
./jadevectordb_tests

# Code coverage
cd backend
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_COVERAGE=ON ..
make
make coverage
# Coverage report available in coverage_report/ directory
```

### Performance Considerations

When developing for JadeVectorDB:
- Use efficient data structures (Eigen for linear algebra operations)
- Implement connection pooling for distributed communication
- Optimize query execution plans
- Consider memory usage for large vector operations
- Implement proper caching strategies
- Target sub-50ms search for 1M vectors
- Target 10,000+ vectors/second ingestion

### Git Workflow

For contributing to the project:
1. Fork the repository
2. Create a feature branch (`feature/description-of-change`)
3. Make your changes following the existing code style
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request with a clear description

### Continuous Integration

The project uses CI/CD pipelines that:
- Run all tests on each commit
- Build Docker images automatically
- Perform static analysis and security checks
- Generate code coverage reports
- Run performance benchmarks

---

## 📋 DOCUMENTATION ORGANIZATION POLICY

**CRITICAL: All sprint completion documents MUST be in TasksTracking/SprintSummary/**

### Sprint Documentation Rules:

1. **ALL sprint completion summaries** go in `TasksTracking/SprintSummary/`
   - ✅ CORRECT: `TasksTracking/SprintSummary/SPRINT_2_3_COMPLETION_SUMMARY.md`
   - ❌ WRONG: `SPRINT_2_3_COMPLETION_SUMMARY.md` (root)
   - ❌ WRONG: `backend/SPRINT_2_3_COMPLETION_SUMMARY.md`
   - ❌ WRONG: `docs/SPRINT_2_3_COMPLETION_SUMMARY.md`
   - ❌ WRONG: `frontend/SPRINT_2_3_COMPLETION_SUMMARY.md`

2. **Task tracking documents** go in `TasksTracking/`
   - Status dashboards, progress trackers, overview documents
   - FR compliance analyses
   - Implementation summaries for specific features (T236, etc.)

3. **Technical documentation** goes in `docs/`
   - API references, architecture docs, guides
   - NOT sprint summaries or task tracking

4. **Build documentation** stays in `backend/`
   - BUILD.md, BUILD_QUICK_REFERENCE.md, README_BUILD.md
   - Build-specific configurations

### Why This Matters:

- **Single source of truth**: All sprint docs in one place
- **Easy to find**: No searching across multiple directories
- **Consistent organization**: Clear separation of concerns
- **Version control**: Track sprint history systematically

### When Creating Sprint Documents:

```bash
# ✅ CORRECT - Create in SprintSummary directory
TasksTracking/SprintSummary/SPRINT_X_Y_COMPLETION_SUMMARY.md

# ❌ NEVER create sprint docs in these locations:
# - Root directory
# - backend/
# - docs/
# - frontend/
# - Any other directory
```

**Action Required**: If you find sprint documents outside `TasksTracking/SprintSummary/`, move them there immediately.

---

## ⚠️ MANDATORY: Documentation Update Protocol

**CRITICAL REQUIREMENT - MUST DO AFTER EVERY CHANGE:**

After **ANY** of the following activities, you **MUST** update documentation:
- ✅ Code implementation
- ✅ Defect fixing
- ✅ Feature completion
- ✅ Task completion
- ✅ Configuration changes
- ✅ Architecture decisions

**Required Updates (in order):**

1. **BOOTSTRAP.md** (this file)
   - Update "Current Project Status" section with new progress
   - Update "Known Issues" if bugs were fixed
   - Add new sections for significant features
   - Update "Last Updated" date at bottom
   - Add new password requirements, auth changes, etc.

2. **TasksTracking/** (Task tracking system)
   - Mark tasks as completed in appropriate file (e.g., `06-current-auth-api.md`)
   - Update `status-dashboard.md` with recent completions
   - Update `overview.md` with new completion percentages
   - Move completed items from "In Progress" to "Completed"

3. **README.md** (if needed)
   - Update feature list if new features added
   - Update installation steps if build process changed
   - Update API examples if endpoints changed
   - Update version number if applicable

**Example Workflow:**
```
1. Fix authentication bug
2. Update BOOTSTRAP.md: Remove from "Known Issues"
3. Update TasksTracking/06-current-auth-api.md: Mark T231 as completed
4. Update TasksTracking/status-dashboard.md: Add to recent completions
5. Update TasksTracking/overview.md: Increment completion count
6. Check README.md: Update authentication section if needed
```

**This is NOT optional - it's a critical part of completing any task!**

---

## ❌ Common Mistakes to Avoid

1. ❌ **DON'T** run `make` directly
2. ❌ **DON'T** run `cmake` directly (except through build.sh)
3. ❌ **DON'T** forget to run from `backend/` directory
4. ❌ **DON'T** run `cmake` directly (except for incremental rebuilds from build/)
5. ❌ **DON'T** use `docker-compose` (obsolete) - use `docker compose` instead
6. ❌ **DON'T** create unnecessary summaries - ask user first if summary is needed
7. ❌ **DON'T** forget to update TasksTracking files when completing tasks
8. ❌ **DON'T** update tasks in old locations (specs/002-check-if-we/tasks.md is archived)
9. ❌ **DON'T** skip documentation updates after completing work (see Mandatory Documentation Update Protocol above)
10. ✅ **DO** use `cd backend && ./build.sh`
11. ✅ **DO** read `BUILD.md` (root) for ALL build-related information
12. ✅ **DO** use `TodoWrite` for complex tasks
13. ✅ **DO** check `docs/archive/RECOVERY_SUMMARY_2025-12-03.md` when resuming work
14. ✅ **DO** verify API endpoint versions (/api/v1 vs /v1)
15. ✅ **DO** update BOOTSTRAP.md, TasksTracking, and README.md after EVERY change
16. ✅ **DO** update TasksTracking/status-dashboard.md when completing tasks
17. ✅ **DO** check TasksTracking/06-current-auth-api.md for current work

---

## 🔄 Session Workflow

1. **Greet user** and ask to load bootstrap
2. **Read** `TasksTracking/status-dashboard.md` to understand current focus
3. **Read** `docs/archive/RECOVERY_SUMMARY_2025-12-03.md` if it exists
4. **Read** `BUILD.md` for build-related information (if needed)
5. **Check** git status and recent commits
6. **Ask** user what they want to work on
7. **Create** todo list with `TodoWrite` for complex tasks
8. **Build** using `cd backend && ./build.sh`
9. **Test** changes: `cd backend/build && ctest -R "TestName" --output-on-failure`
10. **⚠️ MANDATORY: Update documentation** after completing work:
    - Update `BOOTSTRAP.md`: Status, known issues, new features
    - Update `TasksTracking/` files: Mark tasks complete, update progress
    - Update `README.md`: If features/API/build process changed
    - **This is REQUIRED, not optional!**
11. **Ask before creating summaries** - don't create unnecessary summaries unless requested
12. **Commit** when user asks

---

## 🐳 Docker Deployment

### Local Development Deployment

For local development, use the standard Docker Compose setup:

```bash
# Build and run all services
docker compose up --build

# Run in detached mode
docker compose up -d

# Stop all services
docker compose down
```

This creates a single-node setup with:
- JadeVectorDB API server on port 8081
- Web UI on port 3000
- Prometheus monitoring on port 9090
- Grafana dashboard on port 3001

### Distributed Deployment

For production and scaling, use the distributed architecture with master-worker nodes:

```bash
# Start distributed cluster
docker compose -f docker-compose.distributed.yml up --build

# Run in detached mode
docker compose -f docker-compose.distributed.yml up -d
```

This creates a 3-node cluster with:
- 1 Master node (jadevectordb-master)
- 2 Worker nodes (jadevectordb-worker-1, jadevectordb-worker-2)
- Web UI connected to the master node
- Monitoring services (Prometheus and Grafana)

The distributed setup includes:
- **ClusterService**: Raft-based consensus for master election
- **ShardingService**: Multiple data distribution strategies (hash, range, vector-based, auto)
- **ReplicationService**: Configurable replication (synchronous/asynchronous)
- **Load balancing**: Health-aware request routing

### Kubernetes Deployment

For cloud deployments, JadeVectorDB can be deployed to Kubernetes using StatefulSets:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: jadevectordb-master
spec:
  serviceName: jadevectordb-master
  replicas: 3
  selector:
    matchLabels:
      app: jadevectordb
      role: master
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: jadevectordb-worker
spec:
  serviceName: jadevectordb-worker
  replicas: 5
  selector:
    matchLabels:
      app: jadevectordb
      role: worker
```

**Important Notes:**
- Docker uses the same `build.sh` internally
- Use `docker compose` (with space), NOT `docker-compose` (with hyphen - it's obsolete)
- All services run containerized with proper networking and volume management

---

## 📈 Build Performance

| Configuration | First Build | Incremental |
|--------------|-------------|-------------|
| Standard (no gRPC) | ~12 min | ~1 min |
| Minimal (no tests/benchmarks) | ~8 min | ~30 sec |
| With gRPC | ~40 min | ~1 min |

**Why first build is slow**: All dependencies (Eigen, FlatBuffers, Crow, Google Test, Benchmark, Arrow) are fetched and compiled from source!

---

## 📊 Monitoring and Observability

JadeVectorDB includes comprehensive monitoring and observability capabilities to help you understand system performance and health.

### Built-in Metrics

The system exposes various metrics through API endpoints:
- **Health checks**: `/health` endpoint for system status
- **Performance metrics**: `/status` endpoint for detailed system status
- **API endpoints**:
  - `GET /health` - System health check
  - `GET /status` - Detailed system status with cluster information
  - `GET /v1/databases` - List all databases and their status
  - `GET /v1/monitoring/performance` - Performance metrics

### Prometheus Integration

The system is configured to work with Prometheus for metric collection:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'jadevectordb'
    static_configs:
      - targets: ['jadevectordb:8080']
    scrape_interval: 15s
    metrics_path: /metrics
```

### Grafana Dashboards

Grafana dashboards provide visualization of key metrics:
- Cluster status and node health
- Query performance (p50, p95, p99 latencies)
- Throughput metrics
- Resource utilization (CPU, memory, disk, network)
- Replication lag and consistency metrics

Access Grafana at `http://localhost:3001` when running with Docker Compose.

### Key Metrics to Monitor

| Metric Category | Key Metrics |
|-----------------|-------------|
| **Cluster Health** | Node status, master status, connectivity |
| **Performance** | Query latency (p50, p95, p99), throughput |
| **Data Distribution** | Shard count per node, data balance |
| **Replication** | Replication lag, consistency level |
| **Resources** | CPU, memory, disk, network per node |

### Distributed Tracing

The distributed system components support distributed tracing across:
- Master-worker communication
- Query execution across shards
- Replication operations
- Cluster management operations

---

## 🌐 Scaling Strategies

### Horizontal Scaling

JadeVectorDB scales horizontally by adding worker nodes to the cluster:

1. **Sharding-based scaling**: Data is automatically distributed across nodes based on sharding strategy
2. **Query distribution**: Search requests are distributed across relevant shards in parallel
3. **Load balancing**: Queries are distributed across nodes based on their load and health status

### Vertical Scaling

For single-node deployments, you can scale vertically by increasing:
- CPU and memory resources
- Storage capacity
- Network bandwidth

### Auto-scaling Considerations

While auto-scaling is not yet implemented in the current version, the architecture supports:
- Dynamic node joining/leaving
- Automatic shard rebalancing
- Load-based scaling decisions

---

## 🔧 Distributed System Management

### Cluster Management Commands

Use the command-line interface to manage your distributed cluster:

**Python CLI:**
```bash
# Check cluster status
jade-db cluster status

# List all nodes
jade-db nodes list

# Add a new node to the cluster
jade-db cluster add-node --node-id node-id --node-type worker

# Remove a node from the cluster
jade-db cluster remove-node --node-id node-id
```

**Shell CLI:**
```bash
# Check cluster health
bash cli/shell/scripts/jade-db.sh cluster-status

# List nodes
bash cli/shell/scripts/jade-db.sh list-nodes
```

### Master Election and Failover

The system uses Raft-based consensus for master election:
- Automatic master failover in case of master node failure
- Election timeout: 5-10 seconds for failover
- Consistent leader election with no split-brain scenarios
- Quorum-based decision making

### Data Migration and Rebalancing

The system supports live data migration between nodes:
- Zero-downtime migration for queries
- Automated shard rebalancing
- Configurable migration strategies (LIVE_COPY, SNAPSHOT, INCREMENTAL)
- Progress tracking and rollback capability

### Backup and Restore

Distributed backup and restore capabilities include:
- Cluster-wide snapshots
- Incremental backup strategies
- Point-in-time restore functionality
- Backup verification and integrity checks

### Security in Distributed Mode

Security measures in distributed mode:
- Node authentication within the cluster
- Secure RPC communication (to be implemented with TLS)
- Role-based access control across all nodes
- Secure API key distribution

---

## 🎯 Next Session Checklist

When starting a new session, verify:

- [ ] Did I ask user about loading bootstrap?
- [ ] Did I read `TasksTracking/status-dashboard.md` for current focus?
- [ ] Did I read `docs/archive/RECOVERY_SUMMARY_2025-12-03.md`?
- [ ] Did I check git status?
- [ ] Did I read `BUILD.md` (root) for build-related information?
- [ ] Am I in the `backend/` directory for building?
- [ ] Am I using `./build.sh` (NOT make or cmake)?
- [ ] Am I using `--no-tests --no-benchmarks` to avoid test errors?
- [ ] Did I create a todo list for complex tasks?
- [ ] **⚠️ Did I update BOOTSTRAP.md after completing work?**
- [ ] **⚠️ Did I update TasksTracking files when completing tasks?**
- [ ] **⚠️ Did I check if README.md needs updates?**
- [ ] Did I avoid creating unnecessary summaries (ask user first)?
- [ ] Did I verify API endpoint versions (/api/v1 vs /v1)?
- [ ] Am I using `docker compose` (NOT `docker-compose`)?

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `./build.sh --jobs 2` |
| Slow build | `./build.sh --no-tests --no-benchmarks` |
| Build fails | `./build.sh --clean` |
| Test compilation errors | Use `--no-tests --no-benchmarks` |
| Runtime crash | Known issue: duplicate route handlers |

---

## 📞 Key Reminders

1. **⚠️ MANDATORY**: Update BOOTSTRAP.md, TasksTracking, and README.md after EVERY change!
2. **Build System**: Self-contained, fetches all dependencies from source
3. **Build Script**: `cd backend && ./build.sh`
4. **Build Documentation**: Read `BUILD.md` (root) for ALL build information
5. **Tests**: All 16/16 test suites passing; use `--no-tests --no-benchmarks` only for fastest builds
6. **Recovery**: Check `docs/archive/RECOVERY_SUMMARY_2025-12-03.md` when resuming
7. **Current Focus**: Check `TasksTracking/status-dashboard.md` FIRST
8. **Task Tracking**: Update TasksTracking files when completing tasks (NOT old specs/002-check-if-we/tasks.md)
9. **Task System**: TasksTracking is the SINGLE SOURCE OF TRUTH for all tasks
10. **Todo List**: Use `TodoWrite` for complex, multi-step tasks
11. **Docker**: Use `docker compose` (NOT `docker-compose` - it's obsolete)
12. **API Versioning**: Check if endpoints are /api/v1 or /v1
13. **Summaries**: Don't create unnecessary summaries - ask user first
14. **Git Branch**: Currently on `main`
15. **Project Progress**: 100% complete (360/360 tasks)
16. **Authentication**: Use AuthenticationService (NOT AuthManager)
17. **Passwords**: Minimum 10 characters (see Authentication section)

---

## 📝 Dependencies (All Auto-Fetched)

The build system automatically fetches and compiles:
- Eigen 3.4.0 (linear algebra)
- FlatBuffers v23.5.26 (serialization)
- Crow v1.0+5 (REST API framework)
- Google Test v1.14.0 (testing)
- Google Benchmark v1.8.3 (benchmarks)
- Apache Arrow 14.0.0 (columnar format)
- gRPC v1.60.0 (optional, disabled by default)

**No manual installation needed!**

---

## 📚 Additional Resources and Documentation

For further information about JadeVectorDB, consult these additional resources:

- **[API Documentation](docs/api_documentation.md)** - Complete REST API reference
- **[CLI Documentation](docs/cli-documentation.md)** - Command-line interface reference
- **[Architecture Documentation](docs/architecture.md)** - Detailed system architecture
- **[Search Functionality](docs/search_functionality.md)** - Search algorithms and metadata filtering
- **[Quiz System Documentation](docs/QUIZ_SYSTEM_DOCUMENTATION.md)** - Interactive tutorial assessment platform
- **[Distributed Implementation Plan](DISTRIBUTED_SYSTEM_IMPLEMENTATION_PLAN.md)** - Detailed distributed system architecture
- **[Deployment Guide](docs/DOCKER_DEPLOYMENT.md)** - Production deployment instructions
- **[Contributing Guidelines](CONTRIBUTING.md)** - Guidelines for contributing to the project
- **[Frontend Testing Implementation Plan](frontend/TESTING_IMPLEMENTATION_PLAN.md)** - Comprehensive 7-week testing strategy (Jan 2026)
- **[Frontend Code Review Report](frontendreport.md)** - Frontend status assessment and testing gaps analysis
- **[Consistency Report](docs/archive/CONSISTENCY_REPORT_2025-12-03.md)** - Archived report on code consistency and implementation status

---

**Last Updated**: February 12, 2026
**Current Branch**: main
**Build System Version**: Self-contained FetchContent-based (cross-platform: macOS, Linux, Windows)
**Backend Test Status**: 16/16 test suites passing (100%)
**Frontend Status**: Production build working, 46% test coverage

**Critical Reminders**:
- ⚠️ **MANDATORY**: Update BOOTSTRAP.md, TasksTracking, and README.md after EVERY change!
- **Backend Build**: Use `cd backend && ./build.sh`, NOT `make` or `cmake` directly!
- **Frontend Build**: Use `npm run build` from frontend/ directory (all 32 pages compile)
- Read `BUILD.md` (root) for ALL build-related information
- Update `TasksTracking/` files when completing tasks (NOT old specs/002-check-if-we/tasks.md)
- Use `docker compose` (NOT `docker-compose`)
- Check API versioning (/api/v1 vs /v1)
- Don't create unnecessary summaries - ask user first!
- Use AuthenticationService (NOT AuthManager) - single source of truth for auth
- Passwords must be minimum 10 characters (see Authentication section)
- **Frontend Testing**: 28 pages need unit tests - see `frontend/TESTING_IMPLEMENTATION_PLAN.md`
- **Frontend Path Aliases**: Use `@/` for imports (resolves to `./src/`)
