<CLI_INFORMATION>
# CLI Tools

JadeVectorDB provides three command-line interface implementations to suit different environments and user preferences:

### 1. Python CLI (`/cli/python/`)
Full-featured Python-based CLI ideal for data science environments.

```bash
pip install -e cli/python
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name my_database
```

### 2. Shell CLI (`/cli/shell/`)
Lightweight bash-based CLI perfect for system administration and automation.

```bash
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 create-db my_database
```

### 3. JavaScript CLI (`/cli/js/`)
Node.js-based CLI designed for web development environments.

```bash
cd cli/js
npm install
node bin/jade-db.js --url http://localhost:8080 --api-key mykey123 database create --name my_database
```

All implementations provide core functionality:
- Database management (create, list, get, update, delete)
- Vector operations (store, retrieve, update, delete, list, batch get)
- Search operations (similarity, advanced filtered, hybrid, rerank)
- Index management (create, list, delete)
- User management (create, list, update, delete, activate/deactivate)
- API key management (create, list, revoke)
- System operations (health, status, audit log, analytics)
- Embedding generation
- Import/export (JSON, CSV)

The **Python CLI** additionally supports:
- `--curl-only` flag to generate equivalent cURL commands
- Multiple output formats (`--format json|yaml|table|csv`)
- Password management and analytics commands

For detailed documentation, see [CLI Documentation](docs/cli-documentation.md).
For examples, see [CLI Examples](examples/cli/README.md).
For tutorials, see [CLI Tutorials](tutorials/cli/README.md).
</CLI_INFORMATION>