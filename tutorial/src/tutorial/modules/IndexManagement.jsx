import React, { useState, useEffect } from 'react';
import { useTutorialState } from '../../hooks/useTutorialState';
import CodeEditor from '../../components/CodeEditor';
import VisualDashboard from '../../components/VisualDashboard';
import LivePreviewPanel from '../../components/LivePreviewPanel';

const IndexManagement = () => {
  const { state, actions } = useTutorialState();
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState([]);

  // Tutorial steps for Index Management module
  const tutorialSteps = [
    {
      id: 0,
      title: "Understanding Indexing in Vector Databases",
      description: "Learn about indexing algorithms and their role in vector search performance.",
      content: `
        <h3>Why Indexing Matters</h3>
        <p>Vector databases like JadeVectorDB store massive collections of high-dimensional vectors. Without indexing, every search would require:</p>
        <ul>
          <li><strong>Brute Force</strong>: Compare the query vector with every stored vector</li>
          <li><strong>Performance Degradation</strong>: Linear degradation with database size</li>
          <li><strong>Resource Waste</strong>: Inefficient CPU and memory usage</li>
          <li><strong>Cost Increase</strong>: Higher infrastructure requirements</li>
        </ul>
        
        <h3>Trade-offs in Indexing</h3>
        <p>Indexing algorithms balance several factors:</p>
        <ul>
          <li><strong>Accuracy</strong>: How close search results are to brute force</li>
          <li><strong>Speed</strong>: How quickly searches return results</li>
          <li><strong>Memory</strong>: How much RAM the index consumes</li>
          <li><strong>Build Time</strong>: How long it takes to create the index</li>
          <li><strong>Update Cost</strong>: How expensive it is to maintain the index</li>
        </ul>
        
        <h3>Indexing Taxonomy</h3>
        <p>JadeVectorDB supports several indexing algorithms:</p>
        <ul>
          <li><strong>HNSW</strong>: Hierarchical Navigable Small World - for high accuracy</li>
          <li><strong>IVF</strong>: Inverted File Index - for high speed</li>
          <li><strong>LSH</strong>: Locality Sensitive Hashing - for large-scale datasets</li>
          <li><strong>FLAT</strong>: No Indexing - for small datasets requiring exact search</li>
          <li><strong>Composite</strong>: Combination of multiple indexes for complex scenarios</li>
        </ul>
        
        <h3>Index Selection Criteria</h3>
        <p>Choosing the right index depends on your requirements:</p>
        <ul>
          <li><strong>Dataset Size</strong>: Number of vectors to index</li>
          <li><strong>Query Volume</strong>: Number of queries per second</li>
          <li><strong>Accuracy Requirements</strong>: How close results must be to ground truth</li>
          <li><strong>Resource Constraints</strong>: Available memory, CPU, and disk</li>
          <li><strong>Update Frequency</strong>: How often vectors are added/modified</li>
        </ul>
        
        <h3>JadeVectorDB Indexing Philosophy</h3>
        <p>JadeVectorDB embraces flexible indexing:</p>
        <ul>
          <li><strong>Multiple Indexes Per Database</strong>: Different indexes for different use cases</li>
          <li><strong>Runtime Index Switching</strong>: Change indexes without rebuilding</li>
          <li><strong>Automatic Index Tuning</strong>: Smart defaults with manual overrides</li>
          <li><strong>Index Monitoring</strong>: Real-time performance metrics</li>
          <li><strong>Index Evolution</strong>: Seamlessly upgrade indexes</li>
        </ul>
      `,
      codeExample: `
        // HNSW INDEX: High accuracy for quality-critical applications
        const hnswIndex = await client.createIndex(db.id, {
          type: "HNSW",
          parameters: {
            M: 16,              // Connectivity parameter
            efConstruction: 200, // Construction accuracy vs speed
            efSearch: 64        // Query accuracy vs speed
          }
        });
        
        console.log(\`HNSW index created: \${hnswIndex.id}\`);
        
        // IVF INDEX: High speed for performance-critical applications
        const ivfIndex = await client.createIndex(db.id, {
          type: "IVF",
          parameters: {
            nlist: 1000,        // Number of clusters
            nprobe: 10          // Number of clusters to search
          }
        });
        
        console.log(\`IVF index created: \${ivfIndex.id}\`);
        
        // LSH INDEX: Large scale for massive datasets
        const lshIndex = await client.createIndex(db.id, {
          type: "LSH",
          parameters: {
            hashTables: 32,     // Number of hash tables
            hashBits: 64        // Bits per hash table
          }
        });
        
        console.log(\`LSH index created: \${lshIndex.id}\`);
        
        // FLAT INDEX: Exact search for small datasets
        const flatIndex = await client.createIndex(db.id, {
          type: "FLAT",
          parameters: {
            exact: true          // Brute force search
          }
        });
        
        console.log(\`FLAT index created: \${flatIndex.id}\`);
        
        // COMPOSITE INDEX: Combination for complex scenarios
        const compositeIndex = await client.createIndex(db.id, {
          type: "COMPOSITE",
          parameters: {
            indexes: [
              {
                name: "hnsw-precise",
                type: "HNSW",
                parameters: { M: 16, efConstruction: 200, efSearch: 64 }
              },
              {
                name: "ivf-fast",
                type: "IVF",
                parameters: { nlist: 1000, nprobe: 10 }
              }
            ],
            strategy: "HYBRID"    // Combine results from both indexes
          }
        });
        
        console.log(\`Composite index created: \${compositeIndex.id}\`);
        
        // VIEW ALL INDEXES: List existing indexes
        const indexes = await client.listIndexes(db.id);
        console.log(\`Database has \${indexes.length} indexes:\`);
        indexes.forEach(index => {
          console.log(\` - \${index.id}: \${index.type} with \${index.size} vectors\`);
        });
      `,
      expectedOutcome: "Understanding of indexing algorithms and their trade-offs in vector databases"
    },
    {
      id: 1,
      title: "HNSW - Hierarchical Navigable Small World",
      description: "Learn about HNSW, the most popular indexing algorithm for high accuracy.",
      content: `
        <h3>What is HNSW?</h3>
        <p>HNSW (Hierarchical Navigable Small World) organizes vectors in a graph with multiple hierarchical layers:</p>
        <ul>
          <li><strong>Graph Structure</strong>: Each vector connects to nearby vectors</li>
          <li><strong>Hierarchical Layers</strong>: Top layers for coarse search, bottom for refinement</li>
          <li><strong>Greedy Navigation</strong>: Follow edges towards the closest vectors</li>
          <li><strong>Probabilistic Layers</strong>: Higher-degree nodes in upper layers</li>
        </ul>
        
        <h3>HNSW Parameters</h3>
        <p>Configure HNSW for your specific needs:</p>
        <ul>
          <li><strong>M</strong>: Maximum number of connections per node (12-48 typical)</li>
          <li><strong>efConstruction</strong>: Size of candidate list during index building (100-500 typical)</li>
          <li><strong>efSearch</strong>: Size of candidate list during search (10-128 typical)</li>
        </ul>
        
        <h3>HNSW Benefits</h3>
        <p>Why choose HNSW:</p>
        <ul>
          <li><strong>High Accuracy</strong>: Often matches brute force results</li>
          <li><strong>Good Speed</strong>: Much faster than brute force</li>
          <li><strong>Memory Efficient</strong>: Graph structure uses little extra memory</li>
          <li><strong>Scalable</strong>: Works well with millions of vectors</li>
          <li><strong>Mature</strong>: Well-studied with predictable behavior</li>
        </ul>
        
        <h3>HNSW Limitations</h3>
        <p>When HNSW might not be ideal:</p>
        <ul>
          <li><strong>Slow Builds</strong>: Can take hours for very large datasets</li>
          <li>High Memory for very large indexes</li>
          <li>Difficult to update incrementally</li>
        </ul>
        
        <h3>HNSW Best Practices</h3>
        <p>Maximize HNSW performance:</p>
        <ul>
          <li><strong>Choose M Carefully</strong>: Higher M = more accuracy but slower search</li>
          <li><strong>Balance efConstruction</strong>: Higher = better accuracy but longer build time</li>
          <li><strong>Tune efSearch</strong>: Higher = more accuracy but slower queries</li>
          <li><strong>Monitor Memory</strong>: Large M and ef values increase memory usage</li>
          <li><strong>Regular Rebuilding</strong>: For frequently updated datasets</li>
        </ul>
      `,
      codeExample: `
        // CREATE HNSW INDEX: With optimized parameters
        const hnswIndex = await client.createIndex(db.id, {
          type: "HNSW",
          parameters: {
            M: 16,              // Moderate connectivity
            efConstruction: 200, // Good accuracy vs build time balance
            efSearch: 64        // Good accuracy vs query time balance
          }
        });
        
        console.log(\`HNSW index created: \${hnswIndex.id}\`);
        
        // OPTIMIZE HNSW PARAMETERS: Based on use case requirements
        const hnswIndexHighAccuracy = await client.createIndex(db.id, {
          type: "HNSW",
          parameters: {
            M: 32,              // High connectivity for better accuracy
            efConstruction: 500, // Longer build time for better accuracy
            efSearch: 128       // Slower queries for better accuracy
          },
          name: "hnsw-high-accuracy"
        });
        
        const hnswIndexHighSpeed = await client.createIndex(db.id, {
          type: "HNSW",
          parameters: {
            M: 12,              // Lower connectivity for faster search
            efConstruction: 100,  // Faster build time for quicker deployment
            efSearch: 32        // Faster queries at expense of accuracy
          },
          name: "hnsw-high-speed"
        });
        
        // MONITOR HNSW PERFORMANCE: Check recall and latency
        const hnswMetrics = await client.getIndexMetrics(hnswIndex.id);
        console.log(\`HNSW Recall: \${(hnswMetrics.recall * 100).toFixed(2)}%\`);
        console.log(\`HNSW Latency: \${hnswMetrics.avgQueryLatencyMs.toFixed(2)}ms\`);
        console.log(\`HNSW Memory: \${(hnswMetrics.memoryBytes / 1024 / 1024).toFixed(2)}MB\`);
        
        // TUNE HNSW AT RUNTIME: Adjust parameters without rebuilding
        const tunedHnswIndex = await client.updateIndex(hnswIndex.id, {
          parameters: {
            efSearch: 96       // Increase search accuracy dynamically
          }
        });
        
        console.log(\`HNSW index tuned: \${tunedHnswIndex.id}\`);
        
        // COMPARE HNSW INDEXES: Evaluate different configurations
        const hnswIndexes = await client.listIndexes(db.id, { type: "HNSW" });
        const comparisonResults = await Promise.all(hnswIndexes.map(async (index) => {
          const metrics = await client.getIndexMetrics(index.id);
          return {
            index: index.name,
            recall: metrics.recall,
            latency: metrics.avgQueryLatencyMs,
            memory: metrics.memoryBytes
          };
        }));
        
        console.table(comparisonResults);
      `,
      expectedOutcome: "Ability to configure, optimize, and tune HNSW indexes for specific accuracy/speed requirements"
    },
    {
      id: 2,
      title: "IVF - Inverted File Index",
      description: "Learn about IVF, the indexing algorithm optimized for high-speed queries.",
      content: `
        <h3>What is IVF?</h3>
        <p>IVF (Inverted File Index) divides vectors into clusters and searches only promising clusters:</p>
        <ul>
          <li><strong>Clustering</strong>: Group similar vectors together</li>
          <li><strong>Centroids</strong>: Representative vector of each cluster</li>
          <li><strong>Inverted File</strong>: List of vectors in each cluster</li>
          <li><strong>Coarse Quantization</strong>: Assign query to nearest clusters</li>
        </ul>
        
        <h3>IVF Parameters</h3>
        <p>Configure IVF for your specific needs:</p>
        <ul>
          <li><strong>nlist</strong>: Number of clusters (100-10000 typical)</li>
          <li><strong>nprobe</strong>: Number of clusters to search (1-100 typical)</li>
          <li><strong>k-means Iterations</strong>: Number of clustering iterations (10-50 typical)</li>
        </ul>
        
        <h3>IVF Benefits</h3>
        <p>Why choose IVF:</p>
        <ul>
          <li><strong>High Speed</strong>: Searches only promising clusters</li>
          <li><strong>Good Scalability</strong>: Works with billions of vectors</li>
          <li><strong>Memory Efficient</strong>: Only stores centroids and inverted files</li>
          <li><strong>Easy to Update</strong>: Add vectors to existing clusters</li>
        </ul>
        
        <h3>IVF Limitations</h3>
        <p>When IVF might not be ideal:</p>
        <ul>
          <li><strong>Lower Accuracy</strong>: May miss relevant vectors in distant clusters</li>
          <li><strong>Clustering Quality</strong>: Poor clusters reduce effectiveness</li>
          <li><strong>Parameter Sensitivity</strong>: Requires tuning for optimal performance</li>
        </ul>
        
        <h3>IVF Best Practices</h3>
        <p>Maximize IVF performance:</p>
        <ul>
          <li><strong>Choose nlist Carefully</strong>: Roughly sqrt(n) where n is number of vectors</li>
          <li><strong>Balance nprobe</strong>: Higher = better accuracy but slower search</li>
          <li><strong>Monitor Cluster Quality</strong>: Poorly formed clusters hurt performance</li>
          <li><strong>Re-cluster Periodically</strong>: As datasets evolve</li>
          <li><strong>Combine with Quantization</strong>: For additional performance gains</li>
        </ul>
      `,
      codeExample: `
        // CREATE IVF INDEX: With balanced parameters
        const ivfIndex = await client.createIndex(db.id, {
          type: "IVF",
          parameters: {
            nlist: 1000,        // Moderate number of clusters
            nprobe: 10          // Search 10 clusters for good balance
          }
        });
        
        console.log(\`IVF index created: \${ivfIndex.id}\`);
        
        // OPTIMIZE IVF PARAMETERS: For different scenarios
        const ivfIndexHighAccuracy = await client.createIndex(db.id, {
          type: "IVF",
          parameters: {
            nlist: 2000,        // More clusters for finer granularity
            nprobe: 50          // Search more clusters for better accuracy
          },
          name: "ivf-high-accuracy"
        });
        
        const ivfIndexHighSpeed = await client.createIndex(db.id, {
          type: "IVF",
          parameters: {
            nlist: 500,         // Fewer clusters for faster clustering
            nprobe: 5          // Search fewer clusters for faster search
          },
          name: "ivf-high-speed"
        });
        
        // MONITOR IVF PERFORMANCE: Check precision and speed
        const ivfMetrics = await client.getIndexMetrics(ivfIndex.id);
        console.log(\`IVF Precision: \${(ivfMetrics.precision * 100).toFixed(2)}%\`);
        console.log(\`IVF QPS: \${ivfMetrics.queriesPerSecond.toFixed(2)} queries/second\`);
        console.log(\`IVF Memory: \${(ivfMetrics.memoryBytes / 1024 / 1024).toFixed(2)}MB\`);
        
        // TUNE IVF AT RUNTIME: Adjust parameters without rebuilding
        const tunedIvfIndex = await client.updateIndex(ivfIndex.id, {
          parameters: {
            nprobe: 25          // Increase search thoroughness dynamically
          }
        });
        
        console.log(\`IVF index tuned: \${tunedIvfIndex.id}\`);
        
        // COMPARE IVF INDEXES: Evaluate different configurations
        const ivfIndexes = await client.listIndexes(db.id, { type: "IVF" });
        const ivfComparisonResults = await Promise.all(ivfIndexes.map(async (index) => {
          const metrics = await client.getIndexMetrics(index.id);
          return {
            index: index.name,
            precision: metrics.precision,
            qps: metrics.queriesPerSecond,
            memory: metrics.memoryBytes
          };
        }));
        
        console.table(ivfComparisonResults);
      `,
      expectedOutcome: "Ability to configure, optimize, and tune IVF indexes for specific speed/accuracy requirements"
    },
    {
      id: 3,
      title: "LSH - Locality Sensitive Hashing",
      description: "Learn about LSH, the indexing algorithm optimized for massive scale.",
      content: `
        <h3>What is LSH?</h3>
        <p>LSH (Locality Sensitive Hashing) hashes similar vectors to the same buckets with high probability:</p>
        <ul>
          <li><strong>Hash Functions</strong>: Deterministic functions that map similar vectors to same buckets</li>
          <li><strong>Collision Probability</strong>: Higher for similar vectors</li>
          <li><strong>Multiple Tables</strong>: Several hash tables reduce false negatives</li>
          <li><strong>Candidate Verification</strong>: Check all vectors in candidate buckets</li>
        </ul>
        
        <h3>LSH Parameters</h3>
        <p>Configure LSH for your specific needs:</p>
        <ul>
          <li><strong>Hash Tables</strong>: Number of hash tables (16-128 typical)</li>
          <li><strong>Hash Bits</strong>: Bits per hash table (32-256 typical)</li>
          <li><strong>Hash Functions</strong>: Family of hash functions (MinHash, SimHash, etc.)</li>
        </ul>
        
        <h3>LSH Benefits</h3>
        <p>Why choose LSH:</p>
        <ul>
          <li><strong>Massive Scale</strong>: Works with billions of vectors</li>
          <li><strong>Low Memory</strong>: Only stores hashes, not vectors</li>
          <li><strong>Fast Builds</strong>: Nearly instantaneous indexing</li>
          <li><strong>Easy Updates</strong>: Simply hash new vectors</li>
        </ul>
        
        <h3>LSH Limitations</h3>
        <p>When LSH might not be ideal:</p>
        <ul>
          <li><strong>Lower Accuracy</strong>: May miss similar vectors due to hash collisions</li>
          <li><strong>Parameter Complexity</strong>: Difficult to tune for optimal performance</li>
          <li><strong>Hash Quality</strong>: Poor hash functions reduce effectiveness</li>
        </ul>
        
        <h3>LSH Best Practices</h3>
        <p>Maximize LSH performance:</p>
        <ul>
          <li><strong>Choose Hash Functions Carefully</strong>: Match your distance metric</li>
          <li><strong>Balance Tables and Bits</strong>: More tables = better recall, more bits = better precision</li>
          <li><strong>Monitor False Positives</strong>: Tune parameters to reduce irrelevant candidates</li>
          <li><strong>Use with PQ</strong>: Combine with product quantization for memory efficiency</li>
          <li><strong>Regular Rehashing</strong>: As datasets evolve</li>
        </ul>
      `,
      codeExample: `
        // CREATE LSH INDEX: With balanced parameters
        const lshIndex = await client.createIndex(db.id, {
          type: "LSH",
          parameters: {
            hashTables: 32,     // Moderate number of hash tables
            hashBits: 128       // Moderate number of bits per table
          }
        });
        
        console.log(\`LSH index created: \${lshIndex.id}\`);
        
        // OPTIMIZE LSH PARAMETERS: For different scenarios
        const lshIndexHighAccuracy = await client.createIndex(db.id, {
          type: "LSH",
          parameters: {
            hashTables: 64,     // More tables for better recall
            hashBits: 256       // More bits for better precision
          },
          name: "lsh-high-accuracy"
        });
        
        const lshIndexHighScale = await client.createIndex(db.id, {
          type: "LSH",
          parameters: {
            hashTables: 16,     // Fewer tables for faster hashing
            hashBits: 64        // Fewer bits for lower memory
          },
          name: "lsh-high-scale"
        });
        
        // MONITOR LSH PERFORMANCE: Check recall and memory usage
        const lshMetrics = await client.getIndexMetrics(lshIndex.id);
        console.log(\`LSH Recall: \${(lshMetrics.recall * 100).toFixed(2)}%\`);
        console.log(\`LSH Memory: \${(lshMetrics.memoryBytes / 1024 / 1024).toFixed(2)}MB\`);
        console.log(\`LSH Build Time: \${lshMetrics.buildTimeMs}ms\`);
        
        // TUNE LSH AT RUNTIME: Adjust parameters without rebuilding
        const tunedLshIndex = await client.updateIndex(lshIndex.id, {
          parameters: {
            hashTables: 48       // Increase recall dynamically
          }
        });
        
        console.log(\`LSH index tuned: \${tunedLshIndex.id}\`);
        
        // COMPARE LSH INDEXES: Evaluate different configurations
        const lshIndexes = await client.listIndexes(db.id, { type: "LSH" });
        const lshComparisonResults = await Promise.all(lshIndexes.map(async (index) => {
          const metrics = await client.getIndexMetrics(index.id);
          return {
            index: index.name,
            recall: metrics.recall,
            memory: metrics.memoryBytes,
            buildTime: metrics.buildTimeMs
          };
        }));
        
        console.table(lshComparisonResults);
      `,
      expectedOutcome: "Ability to configure, optimize, and tune LSH indexes for massive scale applications"
    },
    {
      id: 4,
      title: "Advanced Indexing Configuration",
      description: "Learn to create complex indexing strategies with composite and custom indexes.",
      content: `
        <h3>Composite Indexes</h3>
        <p>Combine multiple indexes for different stages of search:</p>
        <ul>
          <li><strong>Two-Stage Search</strong>: Coarse filter with fast index, refine with accurate index</li>
          <li><strong>Multi-Modal Indexes</strong>: Different indexes for different data types</li>
          <li><strong>Ensemble Methods</strong>: Combine results from multiple indexes</li>
        </ul>
        
        <h3>Custom Indexes</h3>
        <p>Create domain-specific indexing strategies:</p>
        <ul>
          <li><strong>Domain Knowledge</strong>: Leverage specific characteristics of your data</li>
          <li><strong>Custom Metrics</strong>: Implement specialized distance functions</li>
          <li><strong>Hybrid Approaches</strong>: Combine vector and classical indexing</li>
        </ul>
        
        <h3>Index Routing</h3>
        <p>Direct queries to the most appropriate index:</p>
        <ul>
          <li><strong>Query-Based Routing</strong>: Route based on query characteristics</li>
          <li><strong>User-Based Routing</strong>: Route based on user preferences</li>
          <li><strong>Data-Based Routing</strong>: Route based on stored vector characteristics</li>
        </ul>
        
        <h3>Index Lifecycle Management</h3>
        <p>Efficiently manage indexes throughout their lifecycle:</p>
        <ul>
          <li><strong>Creation Strategies</strong>: Batch vs. incremental builds</li>
          <li><strong>Update Policies</strong>: When and how to update indexes</li>
          <li><strong>Deletion Strategies</strong>: Clean up unused indexes</li>
          <li><strong>Rolling Updates</strong>: Replace indexes without downtime</li>
        </ul>
        
        <h3>Performance Monitoring</h3>
        <p>Track index performance over time:</p>
        <ul>
          <li><strong>Key Metrics</strong>: Recall, latency, memory, build time</li>
          <li><strong>Trend Analysis</strong>: Identify performance degradation</li>
          <li><strong>A/B Testing</strong>: Compare index configurations</li>
          <li><strong>Automated Tuning</strong>: Adjust parameters based on performance</li>
        </ul>
      `,
      codeExample: `
        // CREATE COMPOSITE INDEX: Two-stage search
        const compositeIndex = await client.createIndex(db.id, {
          type: "COMPOSITE",
          parameters: {
            strategy: "TWO_STAGE",
            indexes: [
              {
                name: "coarse-filter",
                type: "IVF",
                parameters: {
                  nlist: 100,     // Few clusters for fast coarse filter
                  nprobe: 5       // Few probes for ultra-fast first stage
                }
              },
              {
                name: "fine-refinement",
                type: "HNSW",
                parameters: {
                  M: 16,          // Moderate connectivity
                  efSearch: 64    // Good accuracy for final results
                }
              }
            ],
            routing: {
              // First stage filters candidates
              // Second stage refines results
              stages: ["coarse-filter", "fine-refinement"]
            }
          }
        });
        
        console.log(\`Composite index created: \${compositeIndex.id}\`);
        
        // CREATE CUSTOM INDEX: Domain-specific approach
        const customIndex = await client.createIndex(db.id, {
          type: "CUSTOM",
          parameters: {
            algorithm: "TIME_SERIES_HNSW",
            parameters: {
              windowSize: 10,     // Time window for grouping
              M: 24,              // Higher connectivity for temporal relationships
              efConstruction: 300, // Better accuracy for complex temporal patterns
              efSearch: 96        // Good accuracy for temporal search
            },
            distanceFunction: "DTW"  // Dynamic Time Warping for temporal similarity
          }
        });
        
        console.log(\`Custom index created: \${customIndex.id}\`);
        
        // MONITOR COMPOSITE INDEX PERFORMANCE: End-to-end metrics
        const compositeMetrics = await client.getIndexMetrics(compositeIndex.id);
        console.log(\`Composite Recall: \${(compositeMetrics.recall * 100).toFixed(2)}%\`);
        console.log(\`Composite Latency: \${compositeMetrics.avgQueryLatencyMs.toFixed(2)}ms\`);
        console.log(\`Composite Memory: \${(compositeMetrics.memoryBytes / 1024 / 1024).toFixed(2)}MB\`);
        
        // A/B TESTING INDEXES: Compare performance
        const abTestResult = await client.compareIndexes(db.id, {
          indexes: [hnswIndex.id, ivfIndex.id, lshIndex.id],
          queries: testQueries,
          metrics: ["recall", "latency", "memory"]
        });
        
        console.log("A/B Test Results:");
        abTestResult.results.forEach(result => {
          console.log(\`\${result.index}: \${(result.metrics.recall * 100).toFixed(2)}% recall, \${result.metrics.latency.toFixed(2)}ms latency\`);
        });
        
        // AUTOMATED INDEX TUNING: Adjust based on performance
        const tunedIndexes = await client.autoTuneIndexes(db.id, {
          targetRecall: 0.95,     // 95% recall target
          maxLatencyMs: 50,       // Max 50ms latency
          optimizeFor: "balanced"  // Balance recall and latency
        });
        
        console.log(\`Auto-tuned \${tunedIndexes.length} indexes\`);
        tunedIndexes.forEach(index => {
          console.log(\` - \${index.name}: \${index.changes.join(", ")}\`);
        });
        
        // ROLLING INDEX UPDATE: Replace without downtime
        const newHnswIndex = await client.createIndex(db.id, {
          type: "HNSW",
          parameters: {
            M: 20,              // Increased connectivity
            efConstruction: 250, // Better accuracy
            efSearch: 80        // Better query accuracy
          }
        });
        
        const replacedIndex = await client.replaceIndex(hnswIndex.id, newHnswIndex.id, {
          strategy: "blue-green", // Zero-downtime replacement
          validationQueries: testQueries.slice(0, 100) // Validate with sample queries
        });
        
        console.log(\`Index replaced: \${replacedIndex.oldId} → \${replacedIndex.newId}\`);
      `,
      expectedOutcome: "Ability to create and manage complex indexing strategies including composite and custom indexes"
    }
  ];

  // Mark step as completed
  const markStepCompleted = (stepId) => {
    if (!completedSteps.includes(stepId)) {
      setCompletedSteps([...completedSteps, stepId]);
      actions.saveAssessmentResult(4, stepId, { 
        passed: true, 
        score: 100,
        completedAt: new Date().toISOString()
      });
    }
  };

  // Move to next step
  const goToNextStep = () => {
    if (currentStep < tutorialSteps.length - 1) {
      setCurrentStep(currentStep + 1);
      markStepCompleted(currentStep);
    }
  };

  // Move to previous step
  const goToPreviousStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  // Complete module
  const completeModule = () => {
    actions.updateModuleProgress(4, tutorialSteps.length);
    // In a real app, this would navigate to the next module
  };

  // Effect to mark first step as visited
  useEffect(() => {
    if (currentStep === 0 && !completedSteps.includes(0)) {
      // Mark intro step as visited after a short delay
      const timer = setTimeout(() => {
        markStepCompleted(0);
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [currentStep]);

  const currentTutorialStep = tutorialSteps[currentStep];

  return (
    <div className="tutorial-module">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Module 5: Index Management</h1>
        <p className="text-gray-600">Learn to configure, optimize, and manage indexes for optimal search performance</p>
      </div>

      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">
            Step {currentStep + 1} of {tutorialSteps.length}
          </span>
          <span className="text-sm font-medium text-gray-700">
            {Math.round(((currentStep + 1) / tutorialSteps.length) * 100)}% Complete
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-500 ease-out" 
            style={{ width: `${((currentStep + 1) / tutorialSteps.length) * 100}%` }}
          ></div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Content Panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* Step Content */}
          <div className="module-card">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h2 className="text-xl font-semibold text-gray-800">{currentTutorialStep.title}</h2>
                <p className="text-gray-600 mt-1">{currentTutorialStep.description}</p>
              </div>
              {completedSteps.includes(currentTutorialStep.id) && (
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                  <svg className="mr-1.5 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Completed
                </span>
              )}
            </div>

            <div className="prose max-w-none mb-6">
              <div dangerouslySetInnerHTML={{ __html: currentTutorialStep.content }} />
            </div>

            <div className="bg-blue-50 border border-blue-100 rounded-lg p-4 mb-6">
              <h3 className="font-medium text-blue-800 mb-2">Expected Outcome</h3>
              <p className="text-blue-700 text-sm">{currentTutorialStep.expectedOutcome}</p>
            </div>
          </div>

          {/* Code Editor */}
          <div className="module-card">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Try It Yourself</h3>
            <CodeEditor 
              initialCode={currentTutorialStep.codeExample}
              moduleId={4}
              stepId={currentStep}
            />
          </div>

          {/* Visualization */}
          <div className="module-card">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Visualization</h3>
            <VisualDashboard />
          </div>

          {/* Live Preview */}
          <div className="module-card">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Live Preview</h3>
            <LivePreviewPanel />
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Step Navigation */}
          <div className="module-card">
            <h3 className="font-semibold text-gray-800 mb-4">Module Steps</h3>
            <div className="space-y-2">
              {tutorialSteps.map((step, index) => (
                <button
                  key={step.id}
                  onClick={() => setCurrentStep(index)}
                  className={`w-full text-left p-3 rounded-lg transition-all ${
                    currentStep === index
                      ? 'bg-blue-50 border border-blue-200 text-blue-800'
                      : completedSteps.includes(step.id)
                      ? 'bg-green-50 border border-green-200 text-green-800 hover:bg-green-100'
                      : 'bg-gray-50 border border-gray-200 text-gray-800 hover:bg-gray-100'
                  }`}
                >
                  <div className="flex items-center">
                    <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center mr-3 ${
                      currentStep === index
                        ? 'bg-blue-500 text-white'
                        : completedSteps.includes(step.id)
                        ? 'bg-green-500 text-white'
                        : 'bg-gray-300 text-gray-700'
                    }`}>
                      {completedSteps.includes(step.id) ? (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <span className="text-xs font-medium">{index + 1}</span>
                      )}
                    </div>
                    <div>
                      <div className="font-medium text-sm">{step.title}</div>
                      <div className="text-xs opacity-75">{step.description}</div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="module-card">
            <h3 className="font-semibold text-gray-800 mb-4">Quick Actions</h3>
            <div className="space-y-3">
              <button
                onClick={goToPreviousStep}
                disabled={currentStep === 0}
                className="w-full btn-secondary flex items-center justify-center"
              >
                <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Previous Step
              </button>
              
              {currentStep < tutorialSteps.length - 1 ? (
                <button
                  onClick={goToNextStep}
                  className="w-full btn-primary flex items-center justify-center"
                >
                  Next Step
                  <svg className="w-4 h-4 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              ) : (
                <button
                  onClick={completeModule}
                  className="w-full btn-success flex items-center justify-center"
                >
                  <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Complete Module
                </button>
              )}
            </div>
          </div>

          {/* Help Resources */}
          <div className="module-card">
            <h3 className="font-semibold text-gray-800 mb-4">Help Resources</h3>
            <div className="space-y-3">
              <a href="#" className="block p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition-colors">
                <div className="flex items-center">
                  <svg className="w-5 h-5 text-gray-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  <span className="text-sm font-medium text-gray-700">Documentation</span>
                </13>
              </a>
              
              <a href="#" className="block p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition-colors">
                <div className="flex items-center">
                  <svg className="w-5 h-5 text-gray-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                  <span className="text-sm font-medium text-gray-700">Community Forum</span>
                </div>
              </a>
              
              <a href="#" className="block p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition-colors">
                <div className="flex items-center">
                  <svg className="w-5 h-5 text-gray-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                  <span className="text-sm font-medium text-gray-700">Video Tutorial</span>
                </div>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IndexManagement;