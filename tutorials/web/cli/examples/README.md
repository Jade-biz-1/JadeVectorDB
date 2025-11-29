# JadeVectorDB CLI Tutorial Examples

This directory contains ready-to-run example scripts that demonstrate various features and use cases of JadeVectorDB. Each script is fully executable and includes detailed explanations.

## Prerequisites

Before running these examples, ensure you have:

1. **JadeVectorDB server running** at `http://localhost:8080` (or set `JADEVECTORDB_URL`)
2. **API key** for authentication (or set `JADEVECTORDB_API_KEY`)
3. **Python CLI installed** (for Python examples):
   ```bash
   cd cli/python
   pip install -e .
   ```

## Environment Setup

Set these environment variables to avoid typing them repeatedly:

```bash
export JADEVECTORDB_URL="http://localhost:8080"
export JADEVECTORDB_API_KEY="your-api-key"
```

## Available Examples

### 1. Quick Start (`quick-start.sh`)

**Difficulty:** Beginner
**Duration:** ~2 minutes
**Purpose:** Learn the basics of JadeVectorDB in a simple, guided script

A beginner-friendly script that covers all fundamental operations:
- Health checks
- Database creation
- Vector storage with metadata
- Vector retrieval
- Similarity search
- Database listing

**Run it:**
```bash
./quick-start.sh
```

**What you'll learn:**
- Basic CLI commands
- Creating your first database
- Storing and retrieving vectors
- Performing similarity searches
- Understanding vector dimensions and metadata

---

### 2. Batch Import (`batch-import.py`)

**Difficulty:** Intermediate
**Duration:** ~5 minutes
**Purpose:** Learn how to efficiently import large datasets

A Python script demonstrating high-performance batch vector imports:
- Generates 1,000 random vectors with metadata
- Imports in batches of 100 vectors
- Measures import performance (vectors/second)
- Performs sample searches on the imported data

**Run it:**
```bash
./batch-import.py
```

**What you'll learn:**
- Batch operation patterns
- Performance optimization techniques
- Generating synthetic test data
- Import rate monitoring
- Working with large datasets

**Customization:**
Edit the script to change:
- `NUM_VECTORS` - Number of vectors to import (default: 1000)
- `VECTOR_DIMENSION` - Vector dimension size (default: 128)
- `BATCH_SIZE` - Batch size for imports (default: 100)

---

### 3. Workflow Demo (`workflow-demo.sh`)

**Difficulty:** Intermediate
**Duration:** ~5-10 minutes
**Purpose:** Understand complete real-world workflows

An interactive demonstration of multi-database management:
- Creates 3 separate databases (Electronics, Clothing, Books)
- Populates each with relevant product data
- Demonstrates cross-database operations
- Shows database information retrieval
- Includes cleanup options

**Run it:**
```bash
./workflow-demo.sh
```

**What you'll learn:**
- Multi-tenant database architecture
- Organizing vectors by domain/category
- Cross-database queries
- Complete CRUD workflows
- Database lifecycle management

**Interactive features:**
- Pauses between steps for learning
- Optional cleanup at the end
- Step-by-step explanations

---

### 4. Product Search Demo (`product-search-demo.sh`)

**Difficulty:** Advanced
**Duration:** ~10 minutes
**Purpose:** See real-world product recommendation in action

A practical demonstration of e-commerce product recommendations:
- Creates a product catalog with diverse items
- Demonstrates similarity-based recommendations
- Shows category-aware suggestions
- Includes budget-conscious search examples

**Run it:**
```bash
./product-search-demo.sh
```

**What you'll learn:**
- Real-world recommendation systems
- Vector similarity for product matching
- Category-specific searches
- Price-aware recommendations
- E-commerce use cases

**Demos included:**
1. Premium laptop recommendations
2. Gaming product suggestions
3. Mobile device recommendations
4. Budget-friendly alternatives

---

## Usage Patterns

### Running All Examples

To run through all examples sequentially:

```bash
# 1. Quick start
./quick-start.sh

# 2. Batch import
./batch-import.py

# 3. Full workflow
./workflow-demo.sh

# 4. Product recommendations
./product-search-demo.sh
```

### Custom Configuration

All scripts respect environment variables:

```bash
# Custom server URL
export JADEVECTORDB_URL="http://production-server:8080"

# Custom API key
export JADEVECTORDB_API_KEY="prod-api-key-123"

# Run with custom config
./quick-start.sh
```

### Cleanup

After running examples, clean up test databases:

```bash
# List all databases
jade-db --url $JADEVECTORDB_URL --api-key $JADEVECTORDB_API_KEY list-dbs

# Delete specific database
jade-db --url $JADEVECTORDB_URL --api-key $JADEVECTORDB_API_KEY delete-db --database-id <database-name>
```

## Troubleshooting

### Error: Connection Refused

```
Error: Failed to connect to http://localhost:8080
```

**Solution:** Start the JadeVectorDB server:
```bash
cd backend/build
./jadevectordb
```

### Error: Authentication Failed

```
Error: Unauthorized (401)
```

**Solution:** Set a valid API key:
```bash
export JADEVECTORDB_API_KEY="your-valid-api-key"
```

Or create a new API key through the authentication endpoints.

### Error: ModuleNotFoundError (Python)

```
ModuleNotFoundError: No module named 'jadevectordb'
```

**Solution:** Install the Python CLI:
```bash
cd cli/python
pip install -e .
```

### Error: Database Already Exists

```
Error: Database 'quickstart_db' already exists
```

**Solution:** Delete the existing database first:
```bash
jade-db --api-key $JADEVECTORDB_API_KEY delete-db --database-id quickstart_db
```

Or modify the script to use a different database name.

## Learning Path

**Recommended order for learning:**

1. **Start with Quick Start** (`quick-start.sh`)
   - Understand basic concepts
   - Get familiar with CLI commands
   - See simple workflows

2. **Progress to Batch Import** (`batch-import.py`)
   - Learn performance optimization
   - Work with larger datasets
   - Understand batch operations

3. **Explore Workflow Demo** (`workflow-demo.sh`)
   - Multi-database management
   - Real-world patterns
   - Complete workflows

4. **Master Product Search** (`product-search-demo.sh`)
   - Production use cases
   - Advanced similarity search
   - Recommendation systems

## Customizing Examples

All scripts are designed to be modified. Common customizations:

### Change Vector Dimensions
Edit the `--dimension` parameter in database creation:
```bash
jade-db create-db --name mydb --dimension 256  # Instead of 4
```

### Adjust Search Parameters
Modify `--top-k` and `--threshold`:
```bash
jade-db search --top-k 10 --threshold 0.8  # More results, higher threshold
```

### Use Different Index Types
Try different index types for performance:
```bash
jade-db create-db --name mydb --dimension 128 --index-type FLAT   # Exact search
jade-db create-db --name mydb --dimension 128 --index-type HNSW   # Approximate (fast)
jade-db create-db --name mydb --dimension 128 --index-type IVF    # Large datasets
```

## Next Steps

After completing these examples:

1. **Read the tutorials:**
   - [Basics Tutorial](../basics.md)
   - [Advanced Tutorial](../advanced.md)

2. **Explore the CLI documentation:**
   - [CLI Documentation](../../../docs/cli-documentation.md)
   - [API Documentation](../../../docs/api_documentation.md)

3. **Check out more examples:**
   - [Python CLI Examples](../../../examples/cli/python-examples.md)
   - [Shell CLI Examples](../../../examples/cli/shell-examples.md)
   - [JavaScript CLI Examples](../../../examples/cli/javascript-examples.md)

4. **Build your own applications:**
   - Integrate JadeVectorDB into your projects
   - Create custom similarity search systems
   - Build recommendation engines
   - Develop semantic search applications

## Contributing

Found an issue or want to add more examples? Contributions are welcome!

1. Fork the repository
2. Create your example script
3. Add documentation to this README
4. Submit a pull request

## Support

If you encounter issues with these examples:

1. Check that JadeVectorDB server is running
2. Verify your API key and URL settings
3. Review the troubleshooting section above
4. Check the main [CLI Documentation](../../../docs/cli-documentation.md)
5. Open an issue on GitHub

---

**Happy Learning!** 🚀

These examples are designed to give you hands-on experience with JadeVectorDB. Take your time exploring each one, and don't hesitate to modify them to fit your learning style.
