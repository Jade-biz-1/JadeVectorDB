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

## Work in Progress

This `README.md` is a work in progress and will be updated soon with more detailed information, including installation instructions, usage examples, and contribution guidelines.

## Getting Started

Further details on functional and non-functional requirements, API design, data persistence, and implementation roadmap can be found in the `spec.md` document located in the `specs/002-check-if-we/` directory.