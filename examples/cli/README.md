# JadeVectorDB CLI Examples

This directory contains comprehensive examples for using the JadeVectorDB CLI tools in all three supported implementations: Python, Shell, and JavaScript.

## Available CLI Implementations

JadeVectorDB provides three different CLI implementations to suit various environments and user preferences:

1. **Python CLI** - Located in `cli/python/`
   - Implemented using Python and the `argparse` library
   - Good for data science environments where Python is prevalent
   - Uses the `jade-db` command

2. **Shell CLI** - Located in `cli/shell/`
   - Implemented as bash scripts for maximum portability
   - Good for system administration and CI/CD environments
   - Uses shell scripts like `jade-db.sh`

3. **JavaScript/Node.js CLI** - Located in `cli/js/`
   - Implemented using Node.js with the `commander` library
   - Good for web development environments
   - Uses the `jade-db` command via Node.js

## Getting Started

### Python CLI

```bash
# Install the package
pip install -e cli/python

# Run commands
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name my_database
```

### Shell CLI

```bash
# Run directly
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 create-db my_database
```

### JavaScript CLI

```bash
# Install dependencies
cd cli/js
npm install

# Run commands
node bin/jade-db.js --url http://localhost:8080 --api-key mykey123 database create --name my_database
```

## Example Use Cases

The following examples demonstrate common operations in all three implementations:

- [Python CLI Examples](./python-examples.md)
- [Shell CLI Examples](./shell-examples.md)
- [JavaScript CLI Examples](./javascript-examples.md)

## Core Operations

All CLI implementations support the following core operations:

1. **Database Management**
   - Create databases
   - List databases
   - Get database details

2. **Vector Operations**
   - Store vectors
   - Retrieve vectors
   - Delete vectors

3. **Search Operations**
   - Similarity search
   - Search with filters and thresholds

4. **System Operations**
   - Health checks
   - Status information

## Common Parameters

All implementations use the following common parameters:

- `--url`: JadeVectorDB API URL (default: http://localhost:8080)
- `--api-key`: API key for authentication (if required by the server)

## Choosing the Right CLI

- Use **Python CLI** if:
  - You're in a Python-heavy environment
  - You're already using Python for data science tasks
  - You want to integrate with Python-based data workflows

- Use **Shell CLI** if:
  - You need maximum portability
  - You're integrating into shell scripts
  - You're working in CI/CD environments

- Use **JavaScript CLI** if:
  - You're in a Node.js environment
  - You prefer JavaScript-based tooling
  - You want to integrate with JavaScript-based workflows

## Next Steps

For more detailed examples and advanced usage scenarios, see the implementation-specific examples in this directory.