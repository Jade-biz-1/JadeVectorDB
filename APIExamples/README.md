# JadeVectorDB Python API Examples

A comprehensive collection of runnable examples demonstrating every API available in the JadeVectorDB Python client library. Each file is self-contained and heavily commented for learning purposes.

## Quick Start

```bash
# 1. Navigate to the project root
cd /path/to/JadeVectorDB

# 2. Make sure the backend server is running (in a separate terminal)
cd backend/build && ./jadevectordb

# 3. (Optional) Edit .env to set your API key, or leave empty for auto-registration
vim APIExamples/.env

# 4. Make the runner script executable and run it
chmod +x APIExamples/run_all_examples.sh
./APIExamples/run_all_examples.sh
```

That's it. The script handles everything: creates a Python virtual environment, installs the client library, checks the server, obtains an auth token, and runs all 15 examples in sequence with pass/fail reporting.

## What `run_all_examples.sh` Does

The runner script performs the following steps automatically:

1. **Loads `.env`** — reads `JADEVECTORDB_URL`, `JADEVECTORDB_API_KEY`, and `JADEVECTORDB_USER_ID` from `APIExamples/.env`
2. **Creates a Python virtual environment** — in `APIExamples/.venv/` (isolated from your system Python)
3. **Installs the `jadevectordb` client library** — from the local source at `cli/python/` using `pip install -e cli/python/` (this also installs the `requests` dependency)
4. **Checks server connectivity** — hits the `/health` endpoint; exits with instructions if the server is not running
5. **Auto-registers a demo user** — if `JADEVECTORDB_API_KEY` is left empty in `.env`, it registers a temporary user and logs in to obtain an auth token
6. **Runs all 15 examples sequentially** — each with a 60-second timeout, printing output and a colored pass/fail result
7. **Prints a summary** — total/passed/failed count with a list of any failures

### Run a single example by number

```bash
# Only run example 04 (Similarity Search):
./APIExamples/run_all_examples.sh 04
```

## Prerequisites

### 1. Start the JadeVectorDB Server

Ensure the JadeVectorDB backend is compiled and running:

```bash
cd backend/build && ./jadevectordb
# Server starts on http://localhost:8080 by default
```

### 2. Install the Python Client Library

> **Important:** The `jadevectordb` package is **not published to PyPI**. Running `pip install jadevectordb` will fail. It is shipped as part of this repository under `cli/python/`.

**Option A — Use the runner script (recommended):**

`run_all_examples.sh` creates its own virtual environment and installs the library automatically. You do not need to install anything manually.

**Option B — Manual install into your own environment:**

```bash
# From the project root:
cd /path/to/JadeVectorDB

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the client library from the local source tree
pip install -e cli/python/

# This installs the jadevectordb package and its dependency (requests).
# Verify it works:
python3 -c "from jadevectordb import JadeVectorDB; print('OK')"
```

### 3. Configure Environment

Edit `APIExamples/.env` with your server details before running:

```bash
# Server URL
JADEVECTORDB_URL=http://localhost:8080

# Auth token or API key (leave empty to let the script auto-register a demo user)
JADEVECTORDB_API_KEY=

# User ID — needed by API key and audit examples (leave empty to auto-create)
JADEVECTORDB_USER_ID=
```

## Running Examples Individually

If you have already installed the library (Option B above), you can run any example directly:

```bash
cd /path/to/JadeVectorDB
python3 APIExamples/01_getting_started.py

# Override the server URL:
JADEVECTORDB_URL=http://my-server:8080 python3 APIExamples/04_similarity_search.py
```

> **Note:** When running examples directly (not via the script), make sure you `cd` to the project root (`JadeVectorDB/`) first, or the `sys.path` adjustment inside each example may not find the client library.

## Examples

| #  | File | APIs Covered |
|----|------|--------------|
| 01 | [Getting Started](01_getting_started.py) | Client initialization, `get_health()`, `get_status()` |
| 02 | [Database Management](02_database_management.py) | `create_database()`, `list_databases()`, `get_database()`, `update_database()`, `delete_database()` |
| 03 | [Vector Operations](03_vector_operations.py) | `store_vector()`, `batch_store_vectors()`, `retrieve_vector()`, `list_vectors()`, `update_vector()`, `batch_get_vectors()`, `delete_vector()` |
| 04 | [Similarity Search](04_similarity_search.py) | `search()`, `advanced_search()` |
| 05 | [Hybrid Search](05_hybrid_search.py) | `hybrid_search()`, `build_bm25_index()`, `get_bm25_index_status()`, `rebuild_bm25_index()`, `add_bm25_documents()` |
| 06 | [Reranking](06_reranking.py) | `rerank_search()`, `rerank()`, `get_reranking_config()`, `update_reranking_config()` |
| 07 | [Index Management](07_index_management.py) | `create_index()`, `list_indexes()`, `update_index()`, `delete_index()` |
| 08 | [Embeddings](08_embeddings.py) | `generate_embeddings()` |
| 09 | [User Management](09_user_management.py) | `create_user()`, `list_users()`, `get_user()`, `update_user()`, `activate_user()`, `deactivate_user()`, `delete_user()` |
| 10 | [API Key Management](10_api_key_management.py) | `create_api_key()`, `list_api_keys()`, `revoke_api_key()` |
| 11 | [Security & Audit](11_security_audit.py) | `get_audit_log()`, `get_sessions()`, `get_audit_stats()` |
| 12 | [Analytics](12_analytics.py) | `get_analytics_stats()`, `get_analytics_queries()`, `get_analytics_patterns()`, `get_analytics_insights()`, `get_analytics_trending()`, `submit_analytics_feedback()`, `export_analytics()` |
| 13 | [Password Management](13_password_management.py) | `change_password()`, `admin_reset_password()` |
| 14 | [Import & Export](14_import_export.py) | `VectorImporter`, `VectorExporter` (JSON/CSV bulk operations) |
| 15 | [Error Handling](15_error_handling.py) | `JadeVectorDBError`, retry patterns, graceful degradation |

## Sample Data

The `sample_data/` directory contains pre-built datasets for testing:

- **`products.json`** — 5 product vectors (128 dimensions) with metadata (name, category, price, brand, text)
- **`products.csv`** — 3 product vectors in CSV format

These files are used by the import/export example and can also be used for your own experimentation.

## API Coverage

These examples cover all **50+ methods** in the JadeVectorDB Python client library across 10 API categories:

- **System** — Health checks, server status
- **Databases** — Full CRUD lifecycle
- **Vectors** — Store, retrieve, update, delete, batch operations
- **Search** — Similarity search, advanced filtered search, hybrid (BM25 + vector), reranking
- **Indexes** — HNSW, IVF, LSH, FLAT index management
- **Embeddings** — Text-to-vector generation
- **Users** — User CRUD, activation, role management
- **API Keys** — Key creation, listing, revocation
- **Security** — Audit logs, sessions, audit statistics
- **Analytics** — Stats, query logs, patterns, insights, trending, feedback, export

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'jadevectordb'` | Install the library: `pip install -e cli/python/` from the project root |
| `ConnectionError` or `Server not reachable` | Start the backend: `cd backend/build && ./jadevectordb` |
| `401 Unauthorized` | Set a valid API key in `.env` or leave it empty for auto-registration |
| Examples fail with `timeout` | The server may be under load — increase the timeout in the script |
