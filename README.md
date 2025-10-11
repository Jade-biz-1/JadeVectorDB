<p align="center">
  <img src="docs/logo.png" alt="JadeVectorDB Logo">
</p>

# JadeVectorDB: High-Performance Distributed Vector Database

## Executive Summary

JadeVectorDB is a new high-performance, distributed vector database designed to be built from scratch. The primary goal of this project is to deliver a scalable and resilient solution for storing, indexing, and searching large volumes of vector embeddings with high efficiency.

The core features of the database include:
- **High-Performance Search:** Support for fast and accurate similarity searches (e.g., cosine similarity, Euclidean distance) using advanced indexing algorithms like HNSW, IVF, and LSH.
- **Distributed Architecture:** A master-worker architecture that supports horizontal scaling, data sharding, and automatic failover to ensure high availability and resilience.
- **Comprehensive API:** A complete set of APIs (REST and gRPC) for all database operations, including database creation, vector management, and complex search queries.
- **Flexible Data Model:** Support for rich metadata associated with vectors, enabling powerful filtered searches that combine semantic and structural criteria.
- **Integrated Embedding Generation:** Optional capabilities to generate vector embeddings from raw data (text, images) using various popular models.

The project will be developed in C++ for maximum performance and will follow a microservices architecture. It is designed for deployment on modern server platforms, including cloud and on-premises environments, with support for containerization and orchestration.

## Key Features

*   **Vector Storage and Retrieval:** Store vector embeddings with associated metadata and retrieve them for similarity searches.
*   **Similarity Search:** Perform fast and accurate similarity searches (cosine similarity, Euclidean distance, dot product).
*   **Advanced Similarity Search with Filters:** Combine vector similarity search with metadata filtering for precise results.
*   **Database Creation and Configuration:** Create and manage vector database instances with configurable parameters.
*   **Vector Embedding Integration:** Integrate with various vector embedding models (BERT, Word2Vec, GloVe, etc.) for text, image, and other data types.
*   **Distributed Deployment and Scaling:** Deploy in a distributed master-worker architecture with horizontal scaling and automatic failover.
*   **Vector Index Management:** Manage vector indexes (HNSW, IVF, LSH) with configurable parameters.
*   **Monitoring and Health Status:** Monitor the health and performance of the vector database.
*   **Vector Data Lifecycle Management:** Manage vector data lifecycle including archival, cleanup, and retention policies.

## Technical Stack

*   **Backend**: C++20 for high-performance vector operations with SIMD optimizations
*   **Libraries**: Eigen (linear algebra), OpenBLAS/BLIS (vector operations), FlatBuffers (serialization), Apache Arrow (in-memory analytics), gRPC (service communication)
*   **Storage**: Custom binary format optimized for vector operations with memory-mapped files; supports Apache Arrow for in-memory operations and FlatBuffers for network serialization
*   **Frontend**: Next.js framework with shadcn UI components
*   **CLI**: Python and shell script integration
*   **Testing**: Google Test with Google Mock for unit and integration testing; Google Benchmark for performance testing
*   **Platform**: Linux server environments (Ubuntu 20.04+, CentOS 8+, RHEL 8+) with containerization support (Docker, Kubernetes)

## Performance Goals

* Similarity searches return results for 1 million vectors in under 50ms with 95% accuracy
* Handle 10,000+ vectors per second ingestion
* 99.9% availability with automatic failover under 30 seconds
* Support for vector dimensions up to 4096

## Architecture

JadeVectorDB uses a microservices architecture with the following components:

- **Backend**: C++20 services for high-performance vector operations
- **Frontend**: Next.js web UI with shadcn components
- **CLI**: Python and shell-based command-line tools

The system supports multiple indexing algorithms configurable per database:
- **HNSW**: Recommended for single-node deployments or where maximum accuracy is the priority
- **IVF with PQ**: Recommended for large-scale distributed deployments due to superior query routing and scalability
- **LSH**: Available for specialized use cases where build time and memory usage are primary constraints

## Implementation Phases

The implementation follows a phased approach with 12 distinct phases:

1. **Setup**: Project initialization and environment setup
2. **Foundational**: Prerequisites blocking all user stories (core C++ libraries, build system, basic data structures)
3. **Vector Storage & Retrieval**: Core vector storage and retrieval capabilities (P1 priority)
4. **Similarity Search**: Similarity search capabilities (P1 priority)
5. **Advanced Search**: Advanced search with metadata filtering (P2 priority)
6. **Database Management**: Database creation and configuration (P2 priority)
7. **Embedding Management**: Embedding model integration and generation (P2 priority)
8. **Distributed System**: Distributed deployment and scaling (P2 priority)
9. **Index Management**: Configurable indexing algorithms (P3 priority)
10. **Data Lifecycle**: Data archival, cleanup, and retention policies (P3 priority)
11. **Monitoring**: Monitoring and health status capabilities (P2 priority)
12. **Polish & Cross-Cutting**: Final refinement and cross-cutting concerns

## Quick Start

See the [Quickstart Guide](specs/002-check-if-we/quickstart.md) for getting started with JadeVectorDB.

## Documentation

Complete documentation is available in the `specs/002-check-if-we/` directory and includes:

- [Specification](specs/002-check-if-we/spec.md): Detailed feature requirements and design
- [Implementation Plan](specs/002-check-if-we/plan.md): Technical approach and architecture
- [Research Findings](specs/002-check-if-we/research.md): Technical decisions and evaluations
- [Task Execution Plan](specs/002-check-if-we/tasks.md): Detailed implementation tasks
- [Data Model](specs/002-check-if-we/data-model.md): Entity definitions and relationships
- [API Contracts](specs/002-check-if-we/contracts/): API specification in OpenAPI format

## Contributing

We welcome contributions to JadeVectorDB! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the terms specified in [LICENSE](LICENSE).