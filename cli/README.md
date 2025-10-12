# JadeVectorDB CLI Tools

Command-line interface tools for interacting with the JadeVectorDB vector database system.

## Installation

The CLI tools are included with the Python client library:

```bash
pip install jadevectordb
```

After installation, you can use the `jade-db` command.

## Usage

All commands require the `--url` parameter to specify the JadeVectorDB API endpoint:

```bash
jade-db --url http://localhost:8080 [command] [options]
```

If authentication is required, use the `--api-key` parameter:

```bash
jade-db --url http://localhost:8080 --api-key my-api-key [command] [options]
```

## Commands

### Database Management

#### Create a Database
```bash
jade-db --url http://localhost:8080 create-db --name my-documents --description "Document embeddings" --dimension 768 --index-type HNSW
```

#### List Databases
```bash
jade-db --url http://localhost:8080 list-dbs
```

### Vector Operations

#### Store a Vector
```bash
jade-db --url http://localhost:8080 --database-id my-documents store --vector-id doc1 --values "[0.1, 0.2, 0.3]" --metadata '{"category":"tech","title":"Example"}'
```

#### Retrieve a Vector
```bash
jade-db --url http://localhost:8080 --database-id my-documents retrieve --vector-id doc1
```

#### Delete a Vector
```bash
jade-db --url http://localhost:8080 --database-id my-documents delete --vector-id doc1
```

### Search Operations

#### Perform Similarity Search
```bash
jade-db --url http://localhost:8080 --database-id my-documents search --query-vector "[0.15, 0.25, 0.35]" --top-k 5 --threshold 0.7
```

### System Monitoring

#### Get System Status
```bash
jade-db --url http://localhost:8080 status
```

#### Get System Health
```bash
jade-db --url http://localhost:8080 health
```

## Shell Script CLI

In addition to the Python CLI, a shell script interface is available for systems where Python is not preferred:

```bash
# Make the script executable
chmod +x /path/to/jade-db.sh

# Use the shell CLI
./jade-db.sh --url http://localhost:8080 list-dbs
```

The shell script supports the same commands as the Python CLI with similar parameters.

## Examples

### Complete Workflow Example

1. Create a new database:
   ```bash
   jade-db --url http://localhost:8080 create-db --name documents --description "Document embeddings" --dimension 768
   ```

2. Store a vector:
   ```bash
   jade-db --url http://localhost:8080 store --database-id documents --vector-id doc1 --values "[0.1, 0.2, 0.3, 0.4, 0.5]"
   ```

3. Perform a search:
   ```bash
   jade-db --url http://localhost:8080 search --database-id documents --query-vector "[0.15, 0.25, 0.35, 0.45, 0.55]" --top-k 3
   ```

4. View system health:
   ```bash
   jade-db --url http://localhost:8080 health
   ```

## Configuration

Instead of specifying the URL for every command, you can set an environment variable:

```bash
export JDB_URL=http://localhost:8080
export JDB_API_KEY=your-api-key
```

Then run commands without the `--url` and `--api-key` parameters.

## Troubleshooting

- If you get "command not found" errors, ensure the installation completed successfully and the Python scripts are in your PATH.
- If you receive authentication errors, verify your API key is correct.
- If you have connection issues, ensure the JadeVectorDB server is running and accessible at the specified URL.