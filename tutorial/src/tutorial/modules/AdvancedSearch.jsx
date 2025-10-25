import React, { useState, useEffect } from 'react';
import { useTutorialState } from '../../hooks/useTutorialState';
import CodeEditor from '../../components/CodeEditor';
import VisualDashboard from '../../components/VisualDashboard';
import LivePreviewPanel from '../../components/LivePreviewPanel';

const AdvancedSearch = () => {
  const { state, actions } = useTutorialState();
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState([]);

  // Tutorial steps for Advanced Search module
  const tutorialSteps = [
    {
      id: 0,
      title: "Understanding Similarity Metrics",
      description: "Learn about different similarity metrics and when to use each one.",
      content: `
        <h3>Similarity Metrics</h3>
        <p>JadeVectorDB supports several similarity metrics to measure vector relationships:</p>
        <ul>
          <li><strong>Cosine Similarity</strong>: Measures the angle between vectors (most common)</li>
          <li><strong>Euclidean Distance</strong>: Measures straight-line distance</li>
          <li><strong>Dot Product</strong>: Measures vector alignment</li>
          <li><strong>Manhattan Distance</strong>: Measures distance along axes at right angles</li>
        </ul>
        
        <h3>Cosine Similarity</h3>
        <p>Cosine similarity is most commonly used for semantic search because:</p>
        <ul>
          <li>Considers the orientation (angle) of vectors, not their magnitude</li>
          <li>Works well with high-dimensional data</li>
          <li>Values range from -1 to 1 (-1 = opposite, 0 = orthogonal, 1 = identical)</li>
        </ul>
        
        <h3>Euclidean Distance</h3>
        <p>Euclidean distance is useful when:</p>
        <ul>
          <li>Vector magnitude matters in your use case</li>
          <li>Working with lower-dimensional vectors</li>
          <li>Values range from 0 to ∞ (0 = identical, ∞ = very different)</li>
        </ul>
        
        <h3>Dot Product</h3>
        <p>Dot product is effective when:</p>
        <ul>
          <li>Vectors are unit normalized (length = 1)</li>
          <li>Used with certain neural network embeddings</li>
          <li>Values range from -∞ to ∞ (higher = more similar)</li>
        </ul>
      `,
      codeExample: `
        // COSINE SIMILARITY: Most common for semantic search
        const cosineSearchResults = await client.search(db.id, queryVector, {
          topK: 10,
          threshold: 0.7,
          metric: "cosine"
        });
        
        console.log(\`Cosine similarity results: \${cosineSearchResults.length} vectors\`);
        
        // EUCLIDEAN DISTANCE: Useful when magnitude matters
        const euclideanSearchResults = await client.search(db.id, queryVector, {
          topK: 10,
          threshold: 0.7,
          metric: "euclidean"
        });
        
        console.log(\`Euclidean distance results: \${euclideanSearchResults.length} vectors\`);
        
        // DOT PRODUCT: Effective with unit vectors
        const dotProductSearchResults = await client.search(db.id, queryVector, {
          topK: 10,
          threshold: 0.7,
          metric: "dot"
        });
        
        console.log(\`Dot product results: \${dotProductSearchResults.length} vectors\`);
        
        // CUSTOM METRIC: JadeVectorDB allows custom similarity functions
        const customSearchResults = await client.search(db.id, queryVector, {
          topK: 10,
          threshold: 0.7,
          metric: "custom",
          customMetric: (a, b) => {
            // Example: Weighted combination of cosine and euclidean
            const cosine = computeCosineSimilarity(a, b);
            const euclidean = 1 / (1 + computeEuclideanDistance(a, b));
            return 0.7 * cosine + 0.3 * euclidean;
          }
        });
        
        console.log(\`Custom metric results: \${customSearchResults.length} vectors\`);
      `,
      expectedOutcome: "Understanding of various similarity metrics and their applications"
    },
    {
      id: 1,
      title: "Metadata Filtering",
      description: "Learn how to combine semantic and structural search with metadata filtering.",
      content: `
        <h3>Metadata Filtering</h3>
        <p>Filtering by metadata enhances search precision by combining semantic similarity with structured data:</p>
        <ul>
          <li><strong>Exact Matches</strong>: Boolean, categorical, or string equality</li>
          <li><strong>Range Queries</strong>: Numeric ranges, dates, times</li>
          <li><strong>Full-Text Search</strong>: Text content within metadata fields</li>
          <li><strong>Geospatial Queries</strong>: Location-based filtering</li>
        </ul>
        
        <h3>Filter Operators</h3>
        <p>JadeVectorDB supports various operators:</p>
        <ul>
          <li><strong>EQUALS</strong> (=): Exact value matches</li>
          <li><strong>NOT_EQUALS</strong> (≠): Not equal to a value</li>
          <li><strong>GREATER_THAN</strong> (>): Greater than a value</li>
          <li><strong>LESS_THAN</strong> (<): Less than a value</li>
          <li><strong>RANGE</strong> (≥ ≤): Within a range</li>
          <li><strong>CONTAINS</strong> (∋): Array contains a value</li>
          <li><strong>IN</strong> (∈): Value in a list</li>
        </ul>
        
        <h3>Boolean Logic</h3>
        <p>Combine filters with boolean operations:</p>
        <ul>
          <li><strong>AND</strong>: All conditions must be true</li>
          <li><strong>OR</strong>: Any condition must be true</li>
          <li><strong>NOT</strong>: Negate a condition</li>
          <li><strong>Nested Conditions</strong>: Complex combinations with parentheses</li>
        </ul>
      `,
      codeExample: `
        // BASIC FILTERING: Simple exact match filters
        const filteredSearchResults = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            category: "product",
            status: "active"
          }
        });
        
        console.log(\`Filtered results: \${filteredSearchResults.length} vectors\`);
        
        // RANGE FILTERING: Numeric range queries
        const rangeFilterResults = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            price: { $gte: 10, $lte: 100 },
            rating: { $gt: 4.0 }
          }
        });
        
        console.log(\`Range filtered results: \${rangeFilterResults.length} vectors\`);
        
        // ARRAY CONTAINS FILTERING: Tags or categories
        const arrayFilterResults = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            tags: { $in: ["electronics", "smart-home"] },
            colors: { $contains: "black" }
          }
        });
        
        console.log(\`Array filtered results: \${arrayFilterResults.length} vectors\`);
        
        // COMPLEX BOOLEAN LOGIC: AND/OR combinations
        const complexFilterResults = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            $and: [
              { category: "electronics" },
              { $or: [
                  { brand: "Samsung" },
                  { brand: "Apple" }
                ]
              },
              { price: { $lte: 1000 } }
            ]
          }
        });
        
        console.log(\`Complex filtered results: \${complexFilterResults.length} vectors\`);
        
        // DATE FILTERING: Time-based queries
        const dateFilterResults = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            created_at: { 
              $gte: "2023-01-01T00:00:00Z",
              $lt: "2024-01-01T00:00:00Z"
            }
          }
        });
        
        console.log(\`Date filtered results: \${dateFilterResults.length} vectors\`);
      `,
      expectedOutcome: "Ability to combine semantic similarity with metadata filtering for precise search results"
    },
    {
      id: 2,
      title: "Advanced Filtering Techniques",
      description: "Learn advanced filtering techniques including geospatial and nested object queries.",
      content: `
        <h3>Geospatial Filtering</h3>
        <p>JadeVectorDB supports spatial queries for location-based search:</p>
        <ul>
          <li><strong>RADIUS</strong>: Find vectors within a circular radius</li>
          <li><strong>BOUNDING_BOX</strong>: Find vectors within rectangular bounds</li>
          <li><strong>POLYGON</strong>: Find vectors within complex polygonal areas</li>
        </ul>
        
        <h3>Temporal Filtering</h3>
        <p>Time-aware filtering for temporal data:</p>
        <ul>
          <li><strong>RANGE</strong>: Between specific dates/times</li>
          <li><strong>RELATIVE</strong>: Recent data within a time frame</li>
          <li><strong>PERIODIC</strong>: Recurring time patterns</li>
        </ul>
        
        <h3>Nested Object Queries</h3>
        <p>Query deeply nested metadata structures:</p>
        <ul>
          <li><strong>PATH</strong>: Navigate nested object paths</li>
          <li><strong>ARRAY_ELEMENTS</strong>: Query array element properties</li>
          <li><strong>EXISTS</strong>: Check existence of nested fields</li>
        </ul>
        
        <h3>Full-Text Search</h3>
        <p>Combine vector similarity with text search capabilities:</p>
        <ul>
          <li><strong>MATCH_PHRASE</strong>: Exact phrase matching</li>
          <li><strong>MATCH_ANY</strong>: Match any terms</li>
          <li><strong>MATCH_ALL</strong>: Match all terms</li>
          <li><strong>FUZZY</strong>: Approximate matching with typos</li>
        </ul>
        
        <h3>Performance Considerations</h3>
        <p>Optimize complex filtering:</p>
        <ul>
          <li>Use indexes on frequently filtered fields</li>
          <li>Limit complex nested queries</li>
          <li>Consider filter selectivity (high-selectivity filters first)</li>
          <li>Use filter pushdown when possible</li>
        </ul>
      `,
      codeExample: `
        // GEOSPATIAL FILTERING: Radius-based location queries
        const geoFilterResults = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            location: {
              $withinRadius: {
                center: { lat: 37.7749, lng: -122.4194 }, // San Francisco
                radiusMeters: 5000 // 5 km radius
              }
            }
          }
        });
        
        console.log(\`Geo-filtered results: \${geoFilterResults.length} vectors\`);
        
        // TEMPORAL FILTERING: Time-based queries with relative ranges
        const temporalFilterResults = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            created_at: {
              $recent: {
                value: 30,
                unit: "days"
              }
            }
          }
        });
        
        console.log(\`Recent results: \${temporalFilterResults.length} vectors\`);
        
        // NESTED OBJECT QUERIES: Deep metadata traversal
        const nestedFilterResults = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            "user.profile.age": { $gte: 18, $lte: 65 },
            "user.preferences.notifications.email": true,
            "product.specs.dimensions.weight": { $lt: 5.0 }
          }
        });
        
        console.log(\`Nested filtered results: \${nestedFilterResults.length} vectors\`);
        
        // FULL-TEXT SEARCH: Text content within metadata
        const fullTextResults = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            "product.description": {
              $text: {
                query: "wireless headphones",
                fuzzy: true,
                fuzziness: 2 // Allow up to 2 character differences
              }
            }
          }
        });
        
        console.log(\`Full-text results: \${fullTextResults.length} vectors\`);
        
        // COMPLEX NESTED COMBINATIONS: Multiple filtering techniques
        const complexNestedResults = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            $and: [
              { "product.category": "electronics" },
              { "product.price": { $lte: 500 } },
              { "location": {
                  $withinBoundingBox: {
                    bottomLeft: { lat: 37.7, lng: -122.5 },
                    topRight: { lat: 37.8, lng: -122.3 }
                  }
                }
              },
              { "created_at": {
                  $recent: { value: 7, unit: "days" }
                }
              },
              { $or: [
                  { "tags": { $contains: "sale" } },
                  { "product.rating": { $gte: 4.5 } }
                ]
              }
            ]
          }
        });
        
        console.log(\`Complex nested results: \${complexNestedResults.length} vectors\`);
      `,
      expectedOutcome: "Ability to perform advanced filtering with geospatial, temporal, and nested object queries"
    },
    {
      id: 3,
      title: "Hybrid Search Techniques",
      description: "Learn how to combine vector similarity with traditional search techniques.",
      content: `
        <h3>Hybrid Search</h3>
        <p>Combine multiple ranking signals for better results:</p>
        <ul>
          <li><strong>Vector Similarity</strong>: Semantic relevance</li>
          <li><strong>Text Relevance</strong>: Keyword matching</li>
          <li><strong>Popularity</strong>: User engagement metrics</li>
          <li><strong>Freshness</strong>: Recency of content</li>
          <li><strong>Personalization</strong>: User preferences</li>
        </ul>
        
        <h3>Weighted Combination</h3>
        <p>Balance different signals with weights:</p>
        <ul>
          <li><strong>Linear Combination</strong>: Weights for each signal</li>
          <li><strong>Machine Learning</strong>: Learned ranking models</li>
          <li><strong>Reciprocal Rank Fusion</strong>: Combines rankings from different sources</li>
          <li><strong>Custom Ranking</strong>: Domain-specific ranking functions</li>
        </ul>
        
        <h3>Multi-Modal Search</h3>
        <p>Search across different modalities:</p>
        <ul>
          <li><strong>Text + Images</strong>: Cross-modal similarity</li>
          <li><strong>Audio + Text</strong>: Speech-to-text search</li>
          <li><strong>Video + Metadata</strong>: Content-aware video search</li>
        </ul>
        
        <h3>Faceted Search</h3>
        <p>Enable exploratory search with facets:</p>
        <ul>
          <li><strong>Aggregations</strong>: Count by categories</li>
          <li><strong>Drill-down</strong>: Navigate through result facets</li>
          <li><strong>Dynamic Filtering</strong>: Update results based on facet selections</li>
        </ul>
      `,
      codeExample: `
        // HYBRID SEARCH: Combining vector similarity with text relevance
        const hybridSearchResults = await client.hybridSearch(db.id, {
          vector: queryVector,
          text: "wireless headphones with noise cancellation",
          vectorWeight: 0.7,
          textWeight: 0.3,
          topK: 20
        });
        
        console.log(\`Hybrid search results: \${hybridSearchResults.length} vectors\`);
        
        // RECIPROCAL RANK FUSION: Combining multiple ranking sources
        const rrfSearchResults = await client.reciprocalRankFusion([
          {
            results: vectorSearchResults,
            k: 60 // RRF parameter
          },
          {
            results: textSearchResults,
            k: 60
          },
          {
            results: popularityRanking,
            k: 60
          }
        ], {
          topK: 10,
          threshold: 0.5
        });
        
        console.log(\`RRF combined results: \${rrfSearchResults.length} vectors\`);
        
        // MULTI-MODAL SEARCH: Cross-modal similarity
        const multimodalResults = await client.multimodalSearch(db.id, {
          modalities: [
            {
              type: "text",
              value: "beach sunset"
            },
            {
              type: "image",
              value: imageVector
            }
          ],
          weights: [0.5, 0.5],
          topK: 10
        });
        
        console.log(\`Multimodal results: \${multimodalResults.length} vectors\`);
        
        // FACETED SEARCH: Aggregated results for exploration
        const facetedSearchResults = await client.facetedSearch(db.id, {
          vector: queryVector,
          facets: ["category", "brand", "price_range"],
          aggregations: {
            category: { $count: true },
            price_range: { 
              $ranges: [
                { name: "budget", from: 0, to: 100 },
                { name: "mid-range", from: 100, to: 500 },
                { name: "premium", from: 500, to: Infinity }
              ]
            },
            avg_rating: { $avg: "rating" }
          },
          topK: 10
        });
        
        console.log(\`Faceted results: \${facetedSearchResults.results.length} vectors\`);
        console.log(\`Aggregations: \${JSON.stringify(facetedSearchResults.aggregations)}\`);
        
        // PERSONALIZED SEARCH: Incorporating user preferences
        const personalizedResults = await client.personalizedSearch(db.id, {
          vector: queryVector,
          userId: "user-123",
          personalization: {
            weights: {
              "user.interests": 0.3,
              "user.purchase_history": 0.4,
              "popularity": 0.2,
              "freshness": 0.1
            }
          },
          topK: 10
        });
        
        console.log(\`Personalized results: \${personalizedResults.length} vectors\`);
      `,
      expectedOutcome: "Ability to combine multiple search techniques for enhanced search experiences"
    },
    {
      id: 4,
      title: "Performance Optimization",
      description: "Learn how to optimize advanced search performance for large-scale applications.",
      content: `
        <h3>Indexing Strategies</h3>
        <p>Optimize performance with appropriate indexing:</p>
        <ul>
          <li><strong>HNSW</strong>: Best for high accuracy and moderate query speed</li>
          <li><strong>IVF</strong>: Best for high-speed queries with approximate accuracy</li>
          <li><strong>LSH</strong>: Best for very large datasets with lower accuracy needs</li>
          <li><strong>FLAT</strong>: Best for small datasets requiring exact search</li>
        </ul>
        
        <h3>Caching Strategies</h3>
        <p>Improve response times with intelligent caching:</p>
        <ul>
          <li><strong>Query Caching</strong>: Cache frequent query results</li>
          <li><strong>Vector Caching</strong>: Cache frequently accessed vectors</li>
          <li><strong>Filter Caching</strong>: Cache common filter combinations</li>
          <li><strong>Result Caching</strong>: Cache aggregated results</li>
        </ul>
        
        <h3>Query Optimization</h3>
        <p>Optimize individual queries:</p>
        <ul>
          <li><strong>Prefiltering</strong>: Apply simple filters before vector search</li>
          <li><strong>Early Termination</strong>: Stop search when enough results found</li>
          <li><strong>Approximate Search</strong>: Trade accuracy for speed</li>
          <li><strong>Parallel Processing</strong>: Distribute workload across cores</li>
        </ul>
        
        <h3>Scalability Techniques</h3>
        <p>Scale for large workloads:</p>
        <ul>
          <li><strong>Sharding</strong>: Distribute vectors across multiple nodes</li>
          <li><strong>Replication</strong>: Create redundant copies for availability</li>
          <li><strong>Load Balancing</strong>: Distribute queries evenly</li>
          <li><strong>Horizontal Scaling</strong>: Add nodes to increase capacity</li>
        </ul>
      `,
      codeExample: `
        // INDEX CONFIGURATION: Optimizing for specific use cases
        const hnswIndex = await client.createIndex(db.id, {
          type: "HNSW",
          parameters: {
            M: 16,              // Connectivity parameter
            efConstruction: 200, // Construction time/accuracy tradeoff
            efSearch: 64        // Query time/accuracy tradeoff
          }
        });
        
        const ivfIndex = await client.createIndex(db.id, {
          type: "IVF",
          parameters: {
            nlist: 1000,        // Number of clusters
            nprobe: 10          // Number of clusters to probe
          }
        });
        
        console.log(\`Indices created: HNSW (\${hnswIndex.id}), IVF (\${ivfIndex.id})\`);
        
        // QUERY OPTIMIZATION: Prefiltering for better performance
        const optimizedSearchResults = await client.search(db.id, queryVector, {
          topK: 10,
          threshold: 0.7,
          // Apply simple filters first to reduce search space
          prefilter: {
            category: "electronics",
            status: "active",
            price: { $lte: 1000 }
          },
          // Early termination - stop when 10 results above threshold found
          earlyTermination: {
            enabled: true,
            threshold: 0.8
          }
        });
        
        console.log(\`Optimized search results: \${optimizedSearchResults.length} vectors\`);
        
        // CACHING CONFIGURATION: Enabling result caching
        const cachedResults = await client.search(db.id, queryVector, {
          topK: 10,
          threshold: 0.7,
          cache: {
            enabled: true,
            ttl: 300, // Cache for 5 minutes
            key: "search-query-123" // Custom cache key
          }
        });
        
        console.log(\`Cached results: \${cachedResults.length} vectors\`);
        
        // BATCH SEARCH: Optimizing multiple queries
        const queries = [
          { vector: queryVector1, topK: 10, threshold: 0.7 },
          { vector: queryVector2, topK: 10, threshold: 0.7 },
          { vector: queryVector3, topK: 10, threshold: 0.7 }
        ];
        
        const batchResults = await client.batchSearch(db.id, queries, {
          parallel: true, // Process queries in parallel
          maxConcurrency: 5 // Limit concurrent requests
        });
        
        console.log(\`Batch search completed: \${batchResults.length} result sets\`);
        
        // PERFORMANCE MONITORING: Track query performance
        const performanceMetrics = await client.getMetrics(db.id);
        console.log(\`Query latency: \${performanceMetrics.avgQueryLatencyMs}ms\`);
        console.log(\`Throughput: \${performanceMetrics.queriesPerSecond} QPS\`);
        console.log(\`Cache hit ratio: \${performanceMetrics.cacheHitRatio}%\`);
        
        // SCALABILITY CONFIGURATION: Sharding and replication
        const shardingConfig = {
          strategy: "hash", // or "range" or "vector"
          numShards: 8,
          replication: {
            factor: 3, // 3 copies of each shard
            sync: true // Synchronous replication
          }
        };
        
        await client.configureSharding(db.id, shardingConfig);
        console.log("Sharding and replication configured");
      `,
      expectedOutcome: "Ability to optimize search performance and scale for production workloads"
    }
  ];

  // Mark step as completed
  const markStepCompleted = (stepId) => {
    if (!completedSteps.includes(stepId)) {
      setCompletedSteps([...completedSteps, stepId]);
      actions.saveAssessmentResult(2, stepId, { 
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
    actions.updateModuleProgress(2, tutorialSteps.length);
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
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Module 3: Advanced Search</h1>
        <p className="text-gray-600">Master similarity search with filters, hybrid techniques, and performance optimization</p>
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
              moduleId={2}
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
                </div>
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

export default AdvancedSearch;