# JadeVectorDB CLI Tutorial

Welcome to the JadeVectorDB CLI tutorial! Master the command-line interface through hands-on exercises and practical examples.

## Overview

This tutorial teaches you to use JadeVectorDB through its command-line interface (CLI), covering everything from basic operations to advanced production workflows.

## Tutorial Structure

```
cli/
├── docs/              # Tutorial documentation
│   ├── basics.md      # Basic concepts and operations
│   └── advanced.md    # Advanced features and patterns
├── exercises/         # Hands-on exercises
│   ├── 01-basics/
│   ├── 02-batch-operations/
│   ├── 03-metadata-filtering/
│   ├── 04-index-management/
│   └── 05-advanced-workflows/
└── sample-data/       # Sample datasets
    ├── products.json  # E-commerce product data
    └── ...
```

## Getting Started

### Prerequisites

1. **JadeVectorDB Instance**
   - Running at `http://localhost:8080` (or your configured URL)
   - Accessible and healthy

2. **CLI Tool** (choose one):
   - **Python CLI:** `pip install -e cli/python` (recommended)
   - **Shell CLI:** Available in `cli/shell/scripts/jade-db.sh`
   - **JavaScript CLI:** Available in `cli/js/bin/jade-db.js`

3. **Optional Tools:**
   - `jq` for JSON parsing (batch operations)
   - `curl` for debugging API calls

### Quick Start

1. **Verify Installation**
   ```bash
   # Python CLI
   jade-db --version

   # Shell CLI
   bash cli/shell/scripts/jade-db.sh --help
   ```

2. **Set Environment Variables**
   ```bash
   export JADE_DB_URL=http://localhost:8080
   export JADE_DB_API_KEY=mykey123  # Replace with your key
   ```

3. **Check System Health**
   ```bash
   jade-db health
   ```

4. **Start with Exercise 1**
   ```bash
   cd exercises/01-basics
   cat README.md
   ```

## Learning Path

### Beginner Path (2-3 hours)
1. Read [`docs/basics.md`](./docs/basics.md)
2. Complete Exercise 1: Basics
3. Read [`docs/advanced.md`](./docs/advanced.md) (overview)

### Intermediate Path (3-4 hours)
1. Complete Exercise 1: Basics
2. Complete Exercise 2: Batch Operations
3. Complete Exercise 3: Metadata Filtering
4. Read advanced documentation thoroughly

### Advanced Path (5-6 hours)
1. Complete all exercises (1-5)
2. Study sample scripts and patterns
3. Implement custom workflows for your use case

## Exercise Guide

### Exercise 1: CLI Basics
**Time:** 30-45 minutes
**Difficulty:** ⭐ Beginner

Learn fundamental operations:
- Creating databases
- Storing and retrieving vectors
- Basic similarity search
- System health checks

**Start:** [`exercises/01-basics/README.md`](./exercises/01-basics/README.md)

### Exercise 2: Batch Operations
**Time:** 45-60 minutes
**Difficulty:** ⭐⭐ Intermediate

Master efficient data import:
- Importing multiple vectors
- Error handling and logging
- Progress monitoring
- Performance measurement

**Start:** [`exercises/02-batch-operations/README.md`](./exercises/02-batch-operations/README.md)

### Exercise 3: Metadata Filtering
**Time:** 45-60 minutes
**Difficulty:** ⭐⭐ Intermediate

Advanced search capabilities:
- Complex metadata filters
- Combining vector similarity with filters
- Range queries and array operations
- Search optimization

**Start:** [`exercises/03-metadata-filtering/README.md`](./exercises/03-metadata-filtering/README.md)

### Exercise 4: Index Management
**Time:** 30-45 minutes
**Difficulty:** ⭐⭐⭐ Advanced

Performance optimization:
- Understanding index types (HNSW, IVF, LSH, FLAT)
- Choosing the right index for your use case
- Index configuration parameters
- Performance benchmarking

**Start:** [`exercises/04-index-management/README.md`](./exercises/04-index-management/README.md)

### Exercise 5: Advanced Workflows
**Time:** 60-90 minutes
**Difficulty:** ⭐⭐⭐ Advanced

Production-ready patterns:
- Monitoring and health checks
- Backup and restore procedures
- Lifecycle management
- Automation scripts

**Start:** [`exercises/05-advanced-workflows/README.md`](./exercises/05-advanced-workflows/README.md)

## Sample Data

### Available Datasets

| File | Description | Records | Dimensions |
|------|-------------|---------|------------|
| `products.json` | E-commerce products | 8 | 8 |
| `documents.json` | Document embeddings | 20 | 16 |
| `images.json` | Image embeddings | 15 | 32 |

### Using Sample Data

```bash
# Example: Import products using sample data
cd exercises/02-batch-operations
cat ../../sample-data/products.json | jq -c '.[]' | while read product; do
  # Extract and import
  # ...
done
```

## CLI Command Reference

### Database Operations
```bash
# Create database
jade-db create-db --name DB_NAME --dimension DIM --index-type TYPE

# List databases
jade-db list-db

# Get database info
jade-db get-db --database-id DB_ID

# Delete database
jade-db delete-db --database-id DB_ID
```

### Vector Operations
```bash
# Store vector
jade-db store \
  --database-id DB_ID \
  --vector-id VEC_ID \
  --values "[0.1, 0.2, ...]" \
  --metadata '{"key": "value"}'

# Retrieve vector
jade-db retrieve --database-id DB_ID --vector-id VEC_ID

# Delete vector
jade-db delete --database-id DB_ID --vector-id VEC_ID
```

### Search Operations
```bash
# Similarity search
jade-db search \
  --database-id DB_ID \
  --query-vector "[0.1, 0.2, ...]" \
  --top-k 10

# Search with threshold
jade-db search \
  --database-id DB_ID \
  --query-vector "[0.1, 0.2, ...]" \
  --top-k 10 \
  --threshold 0.8

# Search with metadata filters (if supported)
jade-db search \
  --database-id DB_ID \
  --query-vector "[0.1, 0.2, ...]" \
  --top-k 10 \
  --filter '{"category": "laptop"}'
```

### System Operations
```bash
# Health check
jade-db health

# System status
jade-db status

# Database statistics
jade-db stats --database-id DB_ID
```

## Best Practices

### 1. Environment Configuration
✅ **Do:**
- Use environment variables for URL and API keys
- Store credentials securely
- Use different configs for dev/staging/prod

❌ **Don't:**
- Hardcode credentials in scripts
- Commit API keys to version control
- Use production credentials in development

### 2. Error Handling
✅ **Do:**
- Check command exit codes
- Log errors to files
- Implement retry logic for transient failures
- Provide helpful error messages

❌ **Don't:**
- Ignore errors silently
- Stop entire batch on first failure
- Retry indefinitely without backoff

### 3. Performance
✅ **Do:**
- Use batch operations for multiple vectors
- Add appropriate delays to avoid rate limiting
- Monitor resource usage
- Benchmark your workflows

❌ **Don't:**
- Import one vector at a time in loops
- Hammer the API without rate limiting
- Ignore performance metrics

### 4. Data Management
✅ **Do:**
- Validate data before import
- Use meaningful vector IDs
- Include comprehensive metadata
- Document your schema

❌ **Don't:**
- Import without validation
- Use random/meaningless IDs
- Skip metadata
- Leave data undocumented

## Troubleshooting

### Common Issues

#### Connection Refused
```
Error: Connection refused to http://localhost:8080
```
**Solution:**
- Verify JadeVectorDB is running
- Check the URL and port
- Ensure firewall allows connections

#### Authentication Failed
```
Error: 401 Unauthorized
```
**Solution:**
- Verify API key is correct
- Check key has necessary permissions
- Ensure key is not expired

#### Dimension Mismatch
```
Error: Vector dimension mismatch. Expected 8, got 4
```
**Solution:**
- Check database dimension configuration
- Verify vector arrays have correct length
- Ensure no parsing errors in vector values

#### Rate Limiting
```
Error: 429 Too Many Requests
```
**Solution:**
- Add delays between requests: `sleep 0.1`
- Reduce batch size
- Implement exponential backoff

### Getting Help

1. **Check Documentation:** Read [docs/basics.md](./docs/basics.md) and [docs/advanced.md](./docs/advanced.md)
2. **Review Examples:** Look at solution scripts in exercises
3. **Test Connectivity:** Run `jade-db health`
4. **Enable Debug Mode:** Add `--verbose` flag (if supported)
5. **Check Logs:** Review JadeVectorDB server logs

## CLI Comparison

### Python CLI
**Pros:**
- Full-featured
- Best documentation
- Easy to extend
- Cross-platform

**Cons:**
- Requires Python installation
- Slightly slower startup

**Best for:** General use, automation, integration

### Shell CLI
**Pros:**
- No dependencies (just bash)
- Fast startup
- Native to Unix/Linux
- Easy to integrate with other shell commands

**Cons:**
- Platform-dependent (Unix/Linux)
- Limited error handling

**Best for:** Quick tasks, system administration, scripts

### JavaScript CLI
**Pros:**
- Node.js ecosystem integration
- Familiar for web developers
- Good for Node.js applications

**Cons:**
- Requires Node.js
- Less common for system automation

**Best for:** Web developers, Node.js integration

## Next Steps

After completing the CLI tutorial:

1. **Production Deployment**
   - Set up monitoring
   - Configure backups
   - Implement security best practices

2. **Integration**
   - Integrate CLI scripts with your CI/CD pipeline
   - Automate database maintenance tasks
   - Create custom workflows for your use case

3. **Advanced Features**
   - Explore the Web Tutorial for visual understanding
   - Study the API documentation for programmatic access
   - Review production deployment guides

## Contributing

Found an issue or want to improve the tutorials?

- **Report Issues:** GitHub issues with `cli-tutorial` label
- **Suggest Improvements:** Submit pull requests
- **Share Scripts:** Contribute useful automation scripts

## Resources

- **Main Documentation:** [`/docs/`](../../docs/)
- **API Reference:** [`/docs/api_documentation.md`](../../docs/api_documentation.md)
- **CLI Tool Documentation:** [`/cli/README.md`](../../cli/README.md)
- **Examples:** [`/examples/cli/`](../../examples/cli/)

---

**Ready to start? Begin with [Exercise 1: CLI Basics](./exercises/01-basics/README.md)!**
