# Research: Vector Indexing Algorithms

This document outlines the research on vector indexing algorithms for the JadeVectorDB project.

## 1. Research Need

Research current state-of-the-art implementations and performance benchmarks for:
- HNSW (Hierarchical Navigable Small World) algorithms
- IVF (Inverted File) indexing techniques
- LSH (Locality Sensitive Hashing) approaches
- Comparison of Approximate Nearest Neighbor (ANN) algorithms and trade-offs between accuracy and speed

## 2. Research Steps

- [x] Research HNSW algorithms.
- [x] Research IVF indexing techniques.
- [x] Research LSH approaches.
- [x] Research and compare ANN algorithms, focusing on accuracy vs. speed trade-offs.
- [x] Summarize findings and provide references.

## 3. HNSW (Hierarchical Navigable Small World)

### Overview

HNSW is a graph-based algorithm for approximate nearest neighbor (ANN) search [1]. It builds a hierarchical graph of the data points, with each layer being a "small world" graph. The top layers are sparse and have long-range connections, while the lower layers are dense and have short-range connections. This allows for efficient searching by starting at the top layer and navigating down to the desired data point.

### How it Works

1.  **Graph Construction**: HNSW builds a multi-layered graph. Each data point is inserted one by one. For each new point, it is connected to its nearest neighbors in the graph. The layer for each new point is chosen randomly with an exponentially decaying probability.
2.  **Search**: The search starts at a predefined entry point in the top layer. The algorithm greedily traverses the graph, moving to the neighbor closest to the query vector. This is done for each layer, until the bottom layer is reached. The search is then refined in the bottom layer to find the nearest neighbors.

### Key Parameters

*   `M`: The maximum number of connections for each node.
*   `efConstruction`: The size of the dynamic list of nearest neighbors during construction.
*   `efSearch`: The size of the dynamic list of nearest neighbors during search.

### Advantages

*   **High Performance**: One of the fastest ANN algorithms.
*   **High Recall**: Achieves high accuracy.
*   **Dynamic**: Supports adding new data points without rebuilding the entire index.

### Disadvantages

*   **Memory Usage**: Can have a high memory footprint.
*   **Parameter Tuning**: Performance is sensitive to the choice of parameters.

## 4. IVF (Inverted File)

### Overview

The Inverted File (IVF) index is a popular method for ANN search that works by partitioning the dataset into clusters [2]. The search is then performed in two steps: first, a coarse search to find the most relevant clusters, and then a fine search within those clusters.

### How it Works

1.  **Clustering**: The data points are clustered using an algorithm like k-means. Each cluster is represented by a centroid.
2.  **Inverted List**: An inverted list is created that maps each centroid to the data points in its cluster.
3.  **Search**: When a query vector is given, it is first compared to all the centroids to find the nearest clusters. Then, the search is narrowed down to the data points within those selected clusters.

### Key Parameters

*   `nlist`: The number of clusters to create.
*   `nprobe`: The number of clusters to search during the query.

### Advantages

*   **Fast**: Significantly faster than brute-force search.
*   **Scalable**: Can handle large datasets.
*   **Good Accuracy**: Provides a good trade-off between speed and accuracy.

### Disadvantages

*   **Static**: The index needs to be rebuilt if new data is added.
*   **Parameter Tuning**: Performance depends on the choice of `nlist` and `nprobe`.

## 5. LSH (Locality Sensitive Hashing)

### Overview

Locality Sensitive Hashing (LSH) is a technique that uses a family of hash functions to hash similar items into the same "buckets" with high probability [3]. It is one of the earliest ANN algorithms.

### How it Works

1.  **Hashing**: A set of LSH functions is chosen. Each function maps data points to a hash value.
2.  **Bucketing**: Data points are stored in buckets based on their hash values. Multiple hash tables are often used to increase the probability of finding the nearest neighbors.
3.  **Search**: For a query vector, the algorithm hashes it using the same LSH functions and retrieves the items in the corresponding buckets. These items are then compared to the query vector to find the nearest neighbors.

### Key Parameters

*   The choice of LSH family of functions.
*   The number of hash functions and hash tables.

### Advantages

*   **Probabilistic Guarantees**: Provides theoretical guarantees on the probability of finding the nearest neighbors.
*   **Sub-linear Query Time**: Can achieve sub-linear query time.

### Disadvantages

*   **Lower Accuracy**: Often has lower accuracy compared to other modern ANN algorithms.
*   **Memory Usage**: Can have high memory usage due to multiple hash tables.
*   **Parameter Tuning**: Can be difficult to tune.

## 6. Comparison of ANN Algorithms

### Overview

The choice of an ANN algorithm is a trade-off between search speed, accuracy (recall), memory usage, and build time. There is no single best algorithm for all use cases.

### Trade-offs

*   **HNSW**: Generally offers the best query performance (speed and accuracy) for most datasets. However, it has a high memory footprint and long build times.
*   **IVF**: A good all-around choice, offering a balance between performance, memory usage, and build time. Its performance is highly dependent on the choice of parameters (`nlist` and `nprobe`). It is often combined with Product Quantization (PQ) to reduce memory usage.
*   **LSH**: The fastest to build and has the lowest memory usage. However, it generally has the lowest accuracy and can be difficult to tune.

### Benchmarks

A comprehensive benchmark of ANN algorithms can be found at [ann-benchmarks.com](http://ann-benchmarks.com/). This website provides a standardized way to evaluate and compare the performance of different ANN algorithms on various datasets.

### Summary

| Algorithm | Speed | Accuracy | Memory Usage | Build Time | Dynamic |
|---|---|---|---|---|---|
| HNSW | Very Fast | Very High | High | Slow | Yes |
| IVF | Fast | High | Medium | Medium | No |
| LSH | Medium | Medium | Low | Very Fast | Yes |

**Conclusion:** For JadeVectorDB, which prioritizes high performance, **HNSW** is the recommended primary indexing algorithm. **IVF** should also be supported as a more memory-efficient option, especially when combined with Product Quantization (PQ). LSH could be considered for scenarios where build time and memory are the primary constraints, but it should not be the default choice.

## 7. References

[1] Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *42*(4), 824-836. ([https://arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320))

[2] Sivic, J., & Zisserman, A. (2003). Video Google: A text retrieval approach to object matching in videos. *Proceedings of the Ninth IEEE International Conference on Computer Vision*, 1470-1477. ([https://www.robots.ox.ac.uk/~vgg/publications/2003/sivic03/sivic03.pdf](https://www.robots.ox.ac.uk/~vgg/publications/2003/sivic03/sivic03.pdf))

[3] Gionis, A., Indyk, P., & Motwani, R. (1999). Similarity search in high dimensions via hashing. *Proceedings of the 25th Very Large Database Conference*, 518-529. ([http://www.vldb.org/conf/1999/P49.pdf](http://www.vldb.org/conf/1999/P49.pdf))
