# JadeVectorDB Architecture Document

**Status:** Draft
**Version:** 0.1

**Note:** This document provides a detailed overview of the system architecture for JadeVectorDB. It is a living document and should be updated as design decisions are made and the system evolves. This document and the main `spec.md` are tightly linked; changes in one should be reflected in the other.

## 1. Architectural Goals and Constraints

The architecture is designed to meet the following key goals:
- **High Performance:** Achieve low-latency search and high-throughput ingestion.
- **Scalability:** Horizontally scale to handle billions of vectors and high concurrent loads.
- **Resilience:** Ensure high availability and fault tolerance with no single point of failure.
- **Maintainability:** Promote a clean separation of concerns to allow for independent development and deployment of components.

The key constraints are detailed in the `spec.md` document under the "Technical Constraints" section.

## 2. System Architecture Overview

JadeVectorDB will be implemented as a distributed, microservices-based system. The architecture follows a master-worker pattern for coordination and data distribution.

### High-Level Diagram

This diagram illustrates the main components of the system and their interactions.

```mermaid
graph TD
    subgraph "User/Client"
        API_Client[API Client]
    end

    subgraph "JadeVectorDB Cluster"
        Load_Balancer[Load Balancer]
        subgraph "Control Plane"
            Master_Node[Master Node]
        end
        subgraph "Data Plane"
            Worker_Node_1[Worker Node 1]
            Worker_Node_2[Worker Node 2]
            Worker_Node_3[Worker Node 3]
        end
    end

    subgraph "External Services"
        Embedding_Service[Embedding Service]
        Persistent_Storage[(Persistent Storage)]
    end

    API_Client --> Load_Balancer
    Load_Balancer --> Master_Node
    Load_Balancer --> Worker_Node_1
    Load_Balancer --> Worker_Node_2
    Load_Balancer --> Worker_Node_3
    Master_Node --- Worker_Node_1
    Master_Node --- Worker_Node_2
    Master_Node --- Worker_Node_3
    Worker_Node_1 --- Persistent_Storage
    Worker_Node_2 --- Persistent_Storage
    Worker_Node_3 --- Persistent_Storage
    Worker_Node_1 --> Embedding_Service
```
*Diagram: High-level overview of JadeVectorDB components.*

### Component Descriptions

- **API Client:** Any application that interacts with the database via the REST or gRPC API.
- **Load Balancer:** Distributes incoming API requests across the appropriate nodes in the cluster.
- **Master Node:** The brain of the cluster. It does not store any vector data itself. Its responsibilities include:
    - Managing cluster state (e.g., list of active workers).
    - Handling database and index metadata.
    - Coordinating distributed operations like data sharding and query planning.
    - Performing leader election for high availability.
- **Worker Nodes:** The workhorses of the cluster. Responsibilities include:
    - Storing and managing a subset (shard) of the vector data.
    - Performing local similarity searches on its shard.
    - Handling data ingestion, indexing, and retrieval for its shard.
- **Persistent Storage:** A durable storage layer (e.g., a distributed file system or cloud object storage) where vectors and indexes are persisted.
- **Embedding Service:** An optional external or internal service for generating vector embeddings from raw data.

## 3. Data Flow

### Data Ingestion Flow

```mermaid
sequenceDiagram
    participant Client
    participant Master
    participant Worker
    participant Storage

    Client->>Master: POST /vectors (with data)
    Master->>Master: Determine target shard/worker
    Master->>Worker: Ingest data for shard
    Worker->>Worker: Index vector and metadata
    Worker->>Storage: Persist vector and index
    Storage-->>Worker: Acknowledge persistence
    Worker-->>Master: Acknowledge ingestion
    Master-->>Client: Return success
```
*Diagram: Sequence of events during data ingestion.*

### Search Query Flow

```mermaid
sequenceDiagram
    participant Client
    participant Master
    participant Workers

    Client->>Master: POST /search (with query vector)
    Master->>Master: Plan distributed query
    Master->>Workers: Fan out query to relevant workers/shards
    Workers->>Workers: Perform local ANN search
    Workers-->>Master: Return top-K results from each shard
    Master->>Master: Merge results and select global top-K
    Master-->>Client: Return final results
```
*Diagram: Sequence of events during a distributed search query.*

## 4. Future Considerations

This section will be expanded to include:
- Detailed C4 models (Context, Containers, Components, Code).
- Network architecture and security group considerations.
- Data replication and consistency models in more detail.
- Deployment architecture for Kubernetes and major cloud providers.
