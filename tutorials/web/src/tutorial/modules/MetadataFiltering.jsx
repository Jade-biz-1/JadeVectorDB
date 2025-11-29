import React, { useState, useEffect } from 'react';
import { useTutorialState } from '../../hooks/useTutorialState';
import CodeEditor from '../../components/CodeEditor';
import VisualDashboard from '../../components/VisualDashboard';
import LivePreviewPanel from '../../components/LivePreviewPanel';

const MetadataFiltering = () => {
  const { state, actions } = useTutorialState();
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState([]);

  // Tutorial steps for Metadata Filtering module
  const tutorialSteps = [
    {
      id: 0,
      title: "Understanding Metadata in Vector Databases",
      description: "Learn about metadata and its role in vector search.",
      content: `
        <h3>What is Metadata?</h3>
        <p>Metadata is descriptive information that provides context for your vector data:</p>
        <ul>
          <li><strong>Structural Information</strong>: Format, size, creation date</li>
          <li><strong>Descriptive Information</strong>: Title, description, author</li>
          <li><strong>Administrative Information</strong>: Permissions, status, owner</li>
          <li><strong>Preservation Information</strong>: Version, backup, migration</li>
        </ul>
        
        <h3>Benefits of Metadata</h3>
        <p>Incorporating metadata with vector data enables:</p>
        <ul>
          <li><strong>Structured Filtering</strong>: Restrict search to specific categories</li>
          <li><strong>Multi-Dimensional Search</strong>: Combine semantic and structural criteria</li>
          <li><strong>Efficient Organization</strong>: Group related vectors</li>
          <li><strong>Enhanced Discoverability</strong>: Locate vectors with familiar filters</li>
          <li><strong>Access Control</strong>: Restrict who can access what data</li>
        </ul>
        
        <h3>Metadata Design Principles</h3>
        <p>To maximize effectiveness:</p>
        <ul>
          <li><strong>Consistency</strong>: Use the same field names and types</li>
          <li><strong>Normalization</strong>: Organize related information in nested objects</li>
          <li><strong>Indexing</strong>: Identify frequently filtered fields for indexing</li>
          <li><strong>Validation</strong>: Define constraints to maintain quality</li>
          <li><strong>Extensibility</strong>: Design for future expansion</li>
        </ul>
        
        <h3>JadeVectorDB Metadata Capabilities</h3>
        <p>JadeVectorDB supports flexible metadata schemas:</p>
        <ul>
          <li><strong>No Schema Required</strong>: Add any fields without prior definition</li>
          <li><strong>Type Inference</strong>: Automatically recognize common types</li>
          <li><strong>Schema Validation</strong>: Optionally enforce field constraints</li>
          <li><strong>Rich Querying</strong>: Filter with complex boolean expressions</li>
          <li><strong>Indexing</strong>: Speed up frequently used filters</li>
        </ul>
      `,
      codeExample: `
        // EXAMPLE METADATA STRUCTURE: Storing vectors with diverse metadata
        const vectorWithMetadata = {
          id: "document-001",
          values: [0.1, 0.2, 0.3, /* ... */],
          metadata: {
            // Descriptive information
            title: "Annual Financial Report",
            description: "2023 financial statements and projections",
            category: "finance",
            tags: ["annual-report", "financial", "2023"],
            
            // Administrative information
            owner: "finance-team",
            status: "published",
            permissions: ["read", "search"],
            access_level: "confidential",
            
            // Structural information
            file_type: "pdf",
            size_bytes: 2567890,
            created_at: new Date("2023-03-15").toISOString(),
            updated_at: new Date().toISOString(),
            
            // Nested objects
            author: {
              name: "Jane Smith",
              department: "Finance",
              email: "jane.smith@example.com"
            },
            metrics: {
              page_views: 1247,
              downloads: 342,
              shares: 89
            },
            versions: [
              { version: "1.0", date: "2023-03-15" },
              { version: "1.1", date: "2023-03-20" }
            ]
          }
        };
        
        // STORE VECTOR WITH METADATA
        const storeResult = await client.storeVector(db.id, vectorWithMetadata);
        console.log(\`Vector stored with metadata: \${storeResult.id}\`);
        
        // RETRIEVE VECTOR AND VERIFY METADATA
        const retrievedVector = await client.getVector(db.id, "document-001");
        console.log(\`Title: \${retrievedVector.metadata.title}\`);
        console.log(\`Tags: \${retrievedVector.metadata.tags.join(", ")}\`);
        console.log(\`Author: \${retrievedVector.metadata.author.name}\`);
        
        // BATCH STORE MULTIPLE VECTORS WITH METADATA
        const vectorsWithMetadata = [
          {
            id: "doc-001",
            values: [0.1, 0.2, 0.3],
            metadata: { category: "finance", status: "published", year: 2023 }
          },
          {
            id: "doc-002",
            values: [0.2, 0.3, 0.4],
            metadata: { category: "marketing", status: "draft", year: 2023 }
          },
          {
            id: "doc-003",
            values: [0.3, 0.4, 0.5],
            metadata: { category: "hr", status: "published", year: 2022 }
          }
        ];
        
        const batchResult = await client.storeVectorsBatch(db.id, vectorsWithMetadata);
        console.log(\`Batch stored \${batchResult.length} vectors with metadata\`);
      `,
      expectedOutcome: "Understanding of metadata in vector databases and how to incorporate it with vector data"
    },
    {
      id: 1,
      title: "Basic Filtering Operations",
      description: "Learn to filter vectors by metadata values to narrow search results.",
      content: `
        <h3>Basic Metadata Filtering</h3>
        <p>Filtering by metadata values is fundamental to precision search:</p>
        <ul>
          <li><strong>Equality</strong>: Match specific values</li>
          <li><strong>Existence</strong>: Check for presence of fields</li>
          <li><strong>Membership</strong>: Check if values belong to sets</li>
          <li><strong>Range</strong>: Filter by numerical ranges</li>
          <li><strong>Text Matching</strong>: Search within text fields</li>
        </ul>
        
        <h3>Filtering Performance</h3>
        <p>Efficient filtering requires strategic planning:</p>
        <ul>
          <li><strong>Indexing</strong>: Speed up frequently filtered fields</li>
          <li><strong>Selectivity</strong>: Filter order affects performance</li>
          <li><strong>Caching</strong>: Reuse results of common filters</li>
          <li><strong>Prefiltering</strong>: Apply simple filters before vector search</li>
        </ul>
        
        <h3>Filter Pushdown</h3>
        <p>JadeVectorDB optimizes performance by pushing filters down to storage:</p>
        <ul>
          <li><strong>Vector Filtering</strong>: Applied before similarity search</li>
          <li><strong>Index Filtering</strong>: Uses indexes when available</li>
          <li><strong>Memory Filtering</strong>: Optimized for in-memory operations</li>
          <li><strong>Distributed Filtering</strong>: Coordinated across cluster nodes</li>
        </ul>
        
        <h3>Common Filter Patterns</h3>
        <p>Typical metadata filtering scenarios:</p>
        <ul>
          <li><strong>Multi-category</strong>: Filter by one or more categories</li>
          <li><strong>Date Range</strong>: Filter by time periods</li>
          <li><strong>Permission-based</strong>: Restrict by user roles or ownership</li>
          <li><strong>Status-based</strong>: Filter by document or content states</li>
        </ul>
      `,
      codeExample: `
        // EQUALITY FILTERING: Match exact metadata values
        const financeDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            category: "finance",
            status: "published"
          }
        });
        
        console.log(\`Found \${financeDocs.length} published finance documents\`);
        
        // EXISTS FILTERING: Check for presence of specific fields
        const taggedDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            "metadata.tags": { $exists: true } // Check if tags field exists
          }
        });
        
        console.log(\`Found \${taggedDocs.length} documents with tags\`);
        
        // MEMBERSHIP FILTERING: Check if values belong to sets
        const importantDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            "metadata.tags": { $in: ["important", "urgent", "critical"] }
          }
        });
        
        console.log(\`Found \${importantDocs.length} important documents\`);
        
        // RANGE FILTERING: Filter by numerical ranges
        const recentLargeDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            "metadata.year": { $gte: 2022 },
            "metadata.size_bytes": { $gte: 1000000 } // 1MB minimum
          }
        });
        
        console.log(\`Found \${recentLargeDocs.length} large recent documents\`);
        
        // TEXT MATCHING: Search within text fields
        const textFilteredDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            "metadata.description": { $text: "financial projection" }
          }
        });
        
        console.log(\`Found \${textFilteredDocs.length} documents mentioning financial projections\`);
        
        // COMBINING FILTERS: Multiple filter types together
        const complexFilteredDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            $and: [
              { "metadata.category": "finance" },
              { "metadata.year": { $gte: 2022 } },
              { "metadata.tags": { $in: ["annual-report", "quarterly"] } },
              { "metadata.status": "published" }
            ]
          }
        });
        
        console.log(\`Found \${complexFilteredDocs.length} complex filtered documents\`);
      `,
      expectedOutcome: "Ability to filter vector search results using basic metadata filtering operations"
    },
    {
      id: 2,
      title: "Advanced Filtering Techniques",
      description: "Learn advanced filtering techniques including nested objects and complex boolean logic.",
      content: `
        <h3>Nested Object Filtering</h3>
        <p>Filtering within nested metadata structures:</p>
        <ul>
          <li><strong>Path-based</strong>: Navigate nested object hierarchies</li>
          <li><strong>Array Elements</strong>: Filter array element properties</li>
          <li><strong>Object Arrays</strong>: Query collections of objects</li>
          <li><strong>Nested Aggregations</strong>: Compute statistics on nested data</li>
        </ul>
        
        <h3>Complex Boolean Logic</h3>
        <p>Combine multiple filter conditions:</p>
        <ul>
          <li><strong>AND</strong>: All conditions must be true</li>
          <li><strong>OR</strong>: Any condition must be true</li>
          <li><strong>NOT</strong>: Negate a condition</li>
          <li><strong>Nested Logic</strong>: Complex combinations with parentheses</li>
        </ul>
        
        <h3>Geospatial Filtering</h3>
        <p>Filter by geographic coordinates:</p>
        <ul>
          <li><strong>Radius</strong>: Find vectors within a circular area</li>
          <li><strong>Bounding Box</strong>: Find vectors within rectangular bounds</li>
          <li><strong>Polygon</strong>: Find vectors within complex shapes</li>
          <li><strong>Distance</strong>: Filter by distance to a point</li>
        </ul>
        
        <h3>Temporal Filtering</h3>
        <p>Filter by time-based criteria:</p>
        <ul>
          <li><strong>Absolute Dates</strong>: Specific date ranges</li>
          <li><strong>Relative Dates</strong>: Recent or upcoming periods</li>
          <li><strong>Periodic</strong>: Recurring time patterns</li>
          <li><strong>Duration</strong>: Age of data</li>
        </ul>
        
        <h3>Full-Text Search</h3>
        <p>Combine vector similarity with text search:</p>
        <ul>
          <li><strong>Phrase Matching</strong>: Exact text sequences</li>
          <li><strong>Term Matching</strong>: Individual word matching</li>
          <li><strong>Fuzzy Matching</strong>: Approximate matches with typos</li>
          <li><strong>Proximity Matching</strong>: Words near each other</li>
        </ul>
      `,
      codeExample: `
        // NESTED OBJECT FILTERING: Navigate nested metadata structures
        const nestedFilteredDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            "metadata.author.department": "Finance",
            "metadata.metrics.page_views": { $gte: 1000 },
            "metadata.versions.version": { $in: ["1.0", "1.1"] }
          }
        });
        
        console.log(\`Found \${nestedFilteredDocs.length} documents with nested filtering\`);
        
        // COMPLEX BOOLEAN LOGIC: Combine multiple filter conditions
        const booleanFilteredDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            $and: [
              { "metadata.category": { $in: ["finance", "marketing"] } },
              { $or: [
                  { "metadata.status": "published" },
                  { "metadata.status": "archived" }
                ]
              },
              { $not: { "metadata.tags": { $in: ["deprecated", "obsolete"] } } }
            ]
          }
        });
        
        console.log(\`Found \${booleanFilteredDocs.length} documents with boolean logic\`);
        
        // GEOSPATIAL FILTERING: Filter by geographic coordinates
        const geospatialFilteredDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            "metadata.location": {
              $withinRadius: {
                center: { lat: 37.7749, lng: -122.4194 }, // San Francisco
                radiusMeters: 10000 // 10km radius
              }
            }
          }
        });
        
        console.log(\`Found \${geospatialFilteredDocs.length} nearby documents\`);
        
        // TEMPORAL FILTERING: Filter by time-based criteria
        const temporalFilteredDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            $and: [
              { "metadata.created_at": {
                  $gte: new Date("2023-01-01").toISOString(),
                  $lt: new Date("2024-01-01").toISOString()
                }
              },
              { "metadata.updated_at": {
                  $recent: { value: 30, unit: "days" } // Updated in last 30 days
                }
              }
            ]
          }
        });
        
        console.log(\`Found \${temporalFilteredDocs.length} temporally filtered documents\`);
        
        // FULL-TEXT SEARCH: Combine with text-based search
        const fullTextFilteredDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            "metadata.description": {
              $text: {
                query: "annual financial report 2023",
                operator: "and", // All terms must be present
                fuzziness: 1, // Allow 1 character difference
                proximity: 5 // Terms must be within 5 words of each other
              }
            }
          }
        });
        
        console.log(\`Found \${fullTextFilteredDocs.length} full-text filtered documents\`);
        
        // COMBINING ALL ADVANCED TECHNIQUES: Complex filtering example
        const complexAdvancedFilteredDocs = await client.search(db.id, queryVector, {
          topK: 10,
          filters: {
            $and: [
              // Basic filtering
              { "metadata.category": { $in: ["finance", "marketing"] } },
              
              // Temporal filtering
              { "metadata.created_at": {
                  $gte: new Date("2023-01-01").toISOString()
                }
              },
              
              // Nested object filtering
              { "metadata.author.department": { $in: ["Finance", "Marketing"] } },
              
              // Geospatial filtering
              { "metadata.location": {
                  $withinBoundingBox: {
                    bottomLeft: { lat: 37.0, lng: -123.0 },
                    topRight: { lat: 38.0, lng: -122.0 }
                  }
                }
              },
              
              // Complex boolean logic
              { $or: [
                  { "metadata.status": "published" },
                  { 
                    "metadata.status": "draft",
                    "metadata.author.name": "Jane Smith"
                  }
                ]
              },
              
              // Full-text search
              { "metadata.description": {
                  $text: {
                    query: "important quarterly",
                    operator: "or"
                  }
                }
              }
            ]
          }
        });
        
        console.log(\`Found \${complexAdvancedFilteredDocs.length} complex advanced filtered documents\`);
      `,
      expectedOutcome: "Ability to filter vector search results using advanced techniques including nested objects and complex boolean logic"
    },
    {
      id: 3,
      title: "Schema Validation and Best Practices",
      description: "Learn to validate metadata schemas and implement best practices for maintainable filters.",
      content: `
        <h3>Schema Validation</h3>
        <p>Ensure data consistency with schema validation:</p>
        <ul>
          <li><strong>Field Validation</strong>: Type checking and constraints</li>
          <li><strong>Required Fields</strong>: Mandatory metadata properties</li>
          <li><strong>Pattern Matching</strong>: Validate string formats</li>
          <li><strong>Range Validation</strong>: Ensure numerical constraints</li>
          <li><strong>Cross-field Validation</strong>: Complex relationships between fields</li>
        </ul>
        
        <h3>Indexing Strategies</h3>
        <p>Optimize performance with strategic indexing:</p>
        <ul>
          <li><strong>Frequency Analysis</strong>: Identify commonly filtered fields</li>
          <li><strong>Selectivity</strong>: Prioritize high-selectivity filters</li>
          <li><strong>Compound Indexes</strong>: Combine frequently used field pairs</li>
          <li><strong>Partial Indexes</strong>: Index only specific value ranges</li>
          <li><strong>Multi-Key Indexes</strong>: Index array elements efficiently</li>
        </ul>
        
        <h3>Best Practices</h3>
        <p>Maintainable filtering with best practices:</p>
        <ul>
          <li><strong>Consistent Naming</strong>: Standardize field names across documents</li>
          <li><strong>Data Normalization</strong>: Reduce redundancy and improve consistency</li>
          <li><strong>Versioning</strong>: Manage schema changes gracefully</li>
          <li><strong>Documentation</strong>: Maintain clear field definitions</li>
          <li><strong>Testing</strong>: Validate filters with diverse data sets</li>
        </ul>
        
        <h3>Performance Optimization</h3>
        <p>Maximize filtering efficiency:</p>
        <ul>
          <li><strong>Filter Pushdown</strong>: Apply filters at the data source</li>
          <li><strong>Short-circuiting</strong>: Stop evaluation when results determined</li>
          <li><strong>Caching</strong>: Store results of common filter combinations</li>
          <li><strong>Prefiltering</strong>: Apply inexpensive filters first</li>
          <li><strong>Parallel Processing</strong>: Distribute filter workload</li>
        </ul>
        
        <h3>Troubleshooting</h3>
        <p>Debugging and resolving filtering issues:</p>
        <ul>
          <li><strong>Query Analysis</strong>: Examine filter execution plans</li>
          <li><strong>Index Diagnostics</strong>: Verify index usage and performance</li>
          <li><strong>Performance Profiling</strong>: Measure filter execution times</li>
          <li><strong>Query Rewriting</strong>: Optimize complex filters</li>
          <li><strong>Error Handling</strong>: Gracefully manage invalid filters</li>
        </ul>
      `,
      codeExample: `
        // SCHEMA VALIDATION: Define and enforce metadata schemas
        const metadataSchema = {
          title: {
            type: "string",
            required: true,
            maxLength: 255
          },
          category: {
            type: "string",
            required: true,
            enum: ["finance", "marketing", "hr", "it", "legal"]
          },
          status: {
            type: "string",
            required: true,
            enum: ["draft", "published", "archived", "deprecated"]
          },
          tags: {
            type: "array",
            items: { type: "string" },
            maxItems: 20
          },
          author: {
            type: "object",
            properties: {
              name: { type: "string", required: true },
              department: { type: "string", required: true },
              email: { type: "string", format: "email" }
            }
          },
          created_at: {
            type: "string",
            format: "date-time"
          },
          size_bytes: {
            type: "integer",
            minimum: 0
          }
        };
        
        // VALIDATE METADATA AGAINST SCHEMA
        function validateMetadata(metadata, schema) {
          const errors = [];
          
          // Check required fields
          for (const [field, definition] of Object.entries(schema)) {
            if (definition.required && metadata[field] === undefined) {
              errors.push(\`Required field "\${field}" is missing\`);
            }
          }
          
          // Check field types
          for (const [field, value] of Object.entries(metadata)) {
            const definition = schema[field];
            if (definition) {
              if (definition.type === "string" && typeof value !== "string") {
                errors.push(\`Field "\${field}" should be a string\`);
              }
              if (definition.type === "integer" && !Number.isInteger(value)) {
                errors.push(\`Field "\${field}" should be an integer\`);
              }
              if (definition.type === "array" && !Array.isArray(value)) {
                errors.push(\`Field "\${field}" should be an array\`);
              }
              if (definition.type === "object" && typeof value !== "object") {
                errors.push(\`Field "\${field}" should be an object\`);
              }
            }
          }
          
          // Check enums
          for (const [field, definition] of Object.entries(schema)) {
            if (definition.enum && !definition.enum.includes(metadata[field])) {
              errors.push(\`Field "\${field}" value "\${metadata[field]}" not in allowed values: \${definition.enum.join(", ")}\`);
            }
          }
          
          return errors.length === 0 ? { valid: true } : { valid: false, errors };
        }
        
        // APPLY SCHEMA VALIDATION
        const docMetadata = {
          title: "Annual Report",
          category: "finance",
          status: "published",
          tags: ["2023", "important"],
          author: {
            name: "John Doe",
            department: "Finance",
            email: "john.doe@example.com"
          },
          created_at: new Date().toISOString(),
          size_bytes: 2567890
        };
        
        const validation = validateMetadata(docMetadata, metadataSchema);
        if (!validation.valid) {
          console.error("Metadata validation failed:", validation.errors);
        } else {
          console.log("Metadata validation passed");
        }
        
        // INDEXING STRATEGIES: Create and manage indexes for filtering
        const indexingStrategies = [
          {
            // High-frequency filter index
            fields: ["category", "status"],
            type: "compound",
            description: "Index for common category/status combinations"
          },
          {
            // High-selectivity filter index
            fields: ["tags"],
            type: "multikey",
            description: "Index for tag-based filtering"
          },
          {
            // Range query index
            fields: ["created_at"],
            type: "range",
            description: "Index for temporal filtering"
          },
          {
            // Text search index
            fields: ["title", "description"],
            type: "text",
            description: "Index for full-text search"
          }
        ];
        
        // CREATE INDEXES
        for (const strategy of indexingStrategies) {
          const index = await client.createIndex(db.id, {
            name: \`\${strategy.fields.join("-")}-index\`,
            fields: strategy.fields,
            type: strategy.type
          });
          console.log(\`Created \${strategy.type} index: \${index.id}\`);
        }
        
        // BEST PRACTICES: Consistent naming and normalization
        const consistentMetadata = {
          // Use consistent naming conventions
          document_title: "Annual Report", // snake_case
          created_date: new Date().toISOString(),
          department_code: "FIN", // Abbreviations when appropriate
          
          // Normalize categorical values
          categories: ["FINANCE", "REPORT"], // UPPERCASE for consistency
          
          // Use standard formats
          iso_date_created: new Date().toISOString(), // ISO 8601
          iso_timestamp: new Date().toISOString(), // Consistent timestamp format
          
          // Organize related fields in nested objects
          author: {
            full_name: "John Doe",
            department_name: "Finance",
            employee_id: "EMP001"
          },
          
          // Use arrays for multi-valued fields
          tags: ["annual", "financial", "2023"],
          related_documents: ["doc-001", "doc-002"]
        };
        
        console.log("Consistent metadata applied:", consistentMetadata);
        
        // PERFORMANCE MONITORING: Analyze filter performance
        const performanceMetrics = await client.analyzeFilters(db.id, {
          filters: {
            "metadata.category": "finance",
            "metadata.status": "published",
            "metadata.created_at": {
              $gte: new Date("2023-01-01").toISOString()
            }
          },
          analyze: ["execution_time", "index_usage", "query_plan"]
        });
        
        console.log("Filter performance:", performanceMetrics);
        
        // TROUBLESHOOTING: Debug filter issues
        const filterDebugging = await client.debugFilter(db.id, {
          filters: {
            "metadata.complex_nested.field": "value"
          },
          debug: true
        });
        
        console.log("Filter debugging info:", filterDebugging);
        console.log("Query plan:", filterDebugging.query_plan);
        console.log("Used indexes:", filterDebugging.used_indexes);
      `,
      expectedOutcome: "Ability to validate metadata schemas and implement best practices for maintainable, performant filtering"
    }
  ];

  // Mark step as completed
  const markStepCompleted = (stepId) => {
    if (!completedSteps.includes(stepId)) {
      setCompletedSteps([...completedSteps, stepId]);
      actions.saveAssessmentResult(3, stepId, { 
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
    actions.updateModuleProgress(3, tutorialSteps.length);
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
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Module 4: Metadata Filtering</h1>
        <p className="text-gray-600">Learn to combine semantic similarity with metadata filtering for precision search</p>
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
              moduleId={3}
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

export default MetadataFiltering;