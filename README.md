<p align="center">
  <img src="docs/logo.png" alt="JadeVectorDB Logo">
</p>

# JadeVectorDB

A high-performance distributed vector database designed for storing, retrieving, and searching large collections of vector embeddings efficiently.

## Current Implementation Status

This project is in early stage of development and still a work in progress. It is published in order to invite volunteers. 

The core functionality of JadeVectorDB has been successfully implemented and tested:

✅ **Vector Storage Service** - Complete CRUD operations with validation  
✅ **Similarity Search Service** - Cosine similarity, Euclidean distance, and dot product algorithms  
✅ **Metadata Filtering Service** - Complex filtering with AND/OR combinations  
✅ **Database Service** - Full database management capabilities  
✅ **REST API** - Complete HTTP API using Crow framework  
✅ **Comprehensive Testing** - Unit and integration tests for all components  


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

## Getting Started

### Prerequisites

- C++20 compatible compiler (GCC 11+, Clang 14+)
- CMake 3.20+
- Dependencies: Eigen, FlatBuffers, Apache Arrow, gRPC, Google Test

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

All implementations provide the same core functionality:
- Database management (create, list, get, delete)
- Vector operations (store, retrieve, delete)
- Search operations (similarity search with filters)
- System operations (health and status checks)

For detailed documentation, see [CLI Documentation](docs/cli-documentation.md).
For examples, see [CLI Examples](examples/cli/README.md).
For tutorials, see [CLI Tutorials](tutorials/cli/README.md).

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
- [Developer Guide](DEVELOPER_GUIDE.md) - Information for contributors

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