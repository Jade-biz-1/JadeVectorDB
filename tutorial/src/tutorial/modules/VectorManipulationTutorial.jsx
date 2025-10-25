import React, { useState, useEffect } from 'react';
import { useTutorialState } from '../hooks/useTutorialState';
import CodeEditor from '../../components/CodeEditor';
import VisualDashboard from '../../components/VisualDashboard';
import LivePreviewPanel from '../../components/LivePreviewPanel';

const VectorManipulationTutorial = () => {
  const { state, actions } = useTutorialState();
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState([]);
  const [code, setCode] = useState('');

  // Tutorial steps for Vector Manipulation module
  const tutorialSteps = [
    {
      id: 0,
      title: "Understanding CRUD Operations",
      description: "Learn how to Create, Read, Update, and Delete vectors in JadeVectorDB.",
      content: `
        <h3>CRUD Operations</h3>
        <p>Vector manipulation in JadeVectorDB follows standard CRUD operations:</p>
        <ul>
          <li><strong>Create</strong>: Store new vectors with unique IDs</li>
          <li><strong>Read</strong>: Retrieve existing vectors by ID</li>
          <li><strong>Update</strong>: Modify vector values or metadata</li>
          <li><strong>Delete</strong>: Remove vectors from the database</li>
        </ul>
        
        <h3>Best Practices</h3>
        <p>When manipulating vectors, consider:</p>
        <ul>
          <li>Always use consistent ID formats</li>
          <li>Validate vector dimensions before storage</li>
          <li>Handle errors gracefully (duplicate IDs, missing vectors, etc.)</li>
          <li>Use batch operations for efficiency when storing multiple vectors</li>
        </ul>
        
        <h3>Data Integrity</h3>
        <p>Maintaining data integrity is crucial:</p>
        <ul>
          <li>Vectors in the same database must have matching dimensions</li>
          <li>Metadata should follow consistent schemas</li>
          <li>Regular validation prevents data corruption</li>
        </ul>
      `,
      codeExample: `
        // CREATE: Store a new vector
        const newVector = {
          id: "vector-001",
          values: [0.1, 0.2, 0.3, /* ... more values ... */],
          metadata: {
            category: "example",
            tags: ["tutorial", "vector"],
            score: 0.95
          }
        };
        
        const result = await client.storeVector(db.id, newVector);
        console.log("Vector created:", result.id);
        
        // READ: Retrieve a vector by ID
        const retrievedVector = await client.getVector(db.id, "vector-001");
        console.log("Vector retrieved:", retrievedVector);
        
        // UPDATE: Modify vector metadata
        const updatedVector = {
          ...retrievedVector,
          metadata: {
            ...retrievedVector.metadata,
            category: "updated-example",
            updated_at: new Date().toISOString()
          }
        };
        
        const updateResult = await client.updateVector(db.id, updatedVector);
        console.log("Vector updated:", updateResult.id);
        
        // DELETE: Remove a vector
        const deleteResult = await client.deleteVector(db.id, "vector-001");
        console.log("Vector deleted:", deleteResult.success);
      `,
      expectedOutcome: "Understanding of basic CRUD operations for vector manipulation"
    },
    {
      id: 1,
      title: "Batch Operations",
      description: "Learn how to efficiently manipulate multiple vectors simultaneously.",
      content: `
        <h3>Batch Operations</h3>
        <p>Working with individual vectors is fine for small datasets, but for large datasets, batch operations are essential for efficiency:</p>
        <ul>
          <li><strong>StoreVectorsBatch</strong>: Store multiple vectors in a single operation</li>
          <li><strong>UpdateVectorsBatch</strong>: Update multiple vectors simultaneously</li>
          <li><strong>DeleteVectorsBatch</strong>: Remove multiple vectors at once</li>
        </ul>
        
        <h3>Performance Benefits</h3>
        <p>Batch operations provide significant performance improvements:</p>
        <ul>
          <li>Reduced network round trips</li>
          <li>Optimized database writes</li>
          <li>Better resource utilization</li>
          <li>Improved throughput for large datasets</li>
        </ul>
        
        <h3>Batch Size Considerations</h3>
        <p>While batches are efficient, consider:</p>
        <ul>
          <li>Optimal batch size varies by use case (typically 100-1000 vectors)</li>
          <li>Large batches may timeout or consume excessive memory</li>
          <li>Error handling becomes more complex with batches</li>
        </ul>
      `,
      codeExample: `
        // BATCH STORE: Store multiple vectors efficiently
        const vectorsToStore = [
          {
            id: "vec-001",
            values: [0.1, 0.2, 0.3, /* ... */],
            metadata: { category: "batch-example", batch: 1 }
          },
          {
            id: "vec-002",
            values: [0.2, 0.3, 0.4, /* ... */],
            metadata: { category: "batch-example", batch: 1 }
          },
          {
            id: "vec-003",
            values: [0.3, 0.4, 0.5, /* ... */],
            metadata: { category: "batch-example", batch: 1 }
          }
        ];
        
        try {
          const batchResults = await client.storeVectorsBatch(db.id, vectorsToStore);
          console.log(\`Stored \${batchResults.length} vectors successfully\`);
        } catch (error) {
          console.error("Batch store failed:", error);
          // Handle individual vector failures within batch
        }
        
        // BATCH UPDATE: Update multiple vectors
        const vectorsToUpdate = vectorsToStore.map(vector => ({
          ...vector,
          metadata: {
            ...vector.metadata,
            updated_at: new Date().toISOString(),
            updated_by: "batch-process"
          }
        }));
        
        const updateResults = await client.updateVectorsBatch(db.id, vectorsToUpdate);
        console.log(\`Updated \${updateResults.length} vectors successfully\`);
        
        // BATCH DELETE: Remove multiple vectors
        const vectorIdsToDelete = ["vec-001", "vec-002", "vec-003"];
        const deleteResults = await client.deleteVectorsBatch(db.id, vectorIdsToDelete);
        console.log(\`Deleted \${deleteResults.deleted_count} vectors successfully\`);
      `,
      expectedOutcome: "Ability to efficiently manipulate multiple vectors using batch operations"
    },
    {
      id: 2,
      title: "Vector Validation and Error Handling",
      description: "Learn how to validate vectors and handle common errors in manipulation operations.",
      content: `
        <h3>Vector Validation</h3>
        <p>Proper validation prevents data corruption and runtime errors:</p>
        <ul>
          <li><strong>Dimension Validation</strong>: Ensure vectors match database dimensions</li>
          <li><strong>ID Uniqueness</strong>: Prevent duplicate IDs in same database</li>
          <li><strong>Value Validation</strong>: Check for NaN or infinite values</li>
          <li><strong>Metadata Validation</strong>: Validate metadata against schema</li>
        </ul>
        
        <h3>Common Errors and Solutions</h3>
        <p>Anticipating errors improves application reliability:</p>
        <ul>
          <li><strong>Duplicate IDs</strong>: Handle gracefully with update logic</li>
          <li><strong>Vector Dimension Mismatch</strong>: Validate before storage</li>
          <li><strong>Network Timeouts</strong>: Implement retry logic</li>
          <li><strong>Rate Limiting</strong>: Back off on API calls when throttled</li>
        </ul>
        
        <h3>Graceful Error Handling</h3>
        <p>Robust error handling ensures system stability:</p>
        <ul>
          <li>Use try-catch blocks for operations</li>
          <li>Implement retry mechanisms with exponential backoff</li>
          <li>Log errors for debugging and monitoring</li>
          <li>Provide user-friendly error messages</li>
        </ul>
      `,
      codeExample: `
        // VALIDATION: Validate vectors before storage
        function validateVector(vector, expectedDimension) {
          // Check ID
          if (!vector.id || typeof vector.id !== 'string') {
            throw new Error("Vector must have a valid ID");
          }
          
          // Check values
          if (!Array.isArray(vector.values)) {
            throw new Error("Vector values must be an array");
          }
          
          // Check dimension
          if (vector.values.length !== expectedDimension) {
            throw new Error(\`Vector dimension mismatch: expected \${expectedDimension}, got \${vector.values.length}\`);
          }
          
          // Check for invalid values
          for (let i = 0; i < vector.values.length; i++) {
            if (!isFinite(vector.values[i])) {
              throw new Error(\`Invalid value at index \${i}: \${vector.values[i]}\`);
            }
          }
          
          return true;
        }
        
        // ERROR HANDLING: Robust error handling with retries
        async function storeVectorWithRetry(dbId, vector, maxRetries = 3) {
          let lastError;
          
          for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
              // Validate vector
              validateVector(vector, 128); // Assuming 128 dimensions
              
              // Attempt to store vector
              const result = await client.storeVector(dbId, vector);
              console.log(\`Vector stored successfully on attempt \${attempt + 1}\`);
              return result;
            } catch (error) {
              lastError = error;
              console.warn(\`Attempt \${attempt + 1} failed: \${error.message}\`);
              
              // Don't retry on validation errors
              if (error.message.includes("dimension mismatch") || error.message.includes("Invalid value")) {
                throw error;
              }
              
              // Wait before retry with exponential backoff
              if (attempt < maxRetries) {
                const delay = Math.pow(2, attempt) * 1000; // 1s, 2s, 4s, etc.
                console.log(\`Retrying in \${delay}ms...\`);
                await new Promise(resolve => setTimeout(resolve, delay));
              }
            }
          }
          
          // If we get here, all attempts failed
          throw new Error(\`Failed to store vector after \${maxRetries + 1} attempts: \${lastError.message}\`);
        }
        
        // USAGE EXAMPLE
        const vector = {
          id: "validated-vector",
          values: [0.1, 0.2, 0.3, 0.4, 0.5], // Too short for 128-dim database
          metadata: { timestamp: new Date().toISOString() }
        };
        
        try {
          await storeVectorWithRetry("my-database", vector);
        } catch (error) {
          console.error("Vector storage failed:", error.message);
        }
      `,
      expectedOutcome: "Ability to validate vectors and implement robust error handling"
    },
    {
      id: 3,
      title: "Advanced Vector Manipulation Techniques",
      description: "Learn advanced techniques for efficient vector manipulation and management.",
      content: `
        <h3>Advanced Techniques</h3>
        <p>Beyond basic CRUD, advanced manipulation techniques include:</p>
        <ul>
          <li><strong>Partial Updates</strong>: Update only specific metadata fields</li>
          <li><strong>Conditional Updates</strong>: Update vectors based on conditions</li>
          <li><strong>Bulk Imports</strong>: Efficiently import large datasets</li>
          <li><strong>Vector Transformations</strong>: Modify vector values programmatically</li>
        </ul>
        
        <h3>Performance Optimization</h3>
        <p>Optimize manipulation performance with:</p>
        <ul>
          <li>Connection pooling for concurrent operations</li>
          <li>Pipelining requests to reduce latency</li>
          <li>Asynchronous processing for non-blocking operations</li>
          <li>Efficient memory management for large batches</li>
        </ul>
        
        <h3>Data Migration Strategies</h3>
        <p>When updating schemas or moving data:</p>
        <ul>
          <li>Use versioned metadata for gradual migrations</li>
          <li>Implement blue-green deployment for zero-downtime updates</li>
          <li>Backup data before major changes</li>
          <li>Test migration processes with sample data</li>
        </ul>
      `,
      codeExample: `
        // PARTIAL UPDATES: Update only specific metadata fields
        async function updateVectorMetadata(dbId, vectorId, metadataUpdates) {
          try {
            // Get current vector
            const currentVector = await client.getVector(dbId, vectorId);
            
            // Merge updates with existing metadata
            const updatedVector = {
              ...currentVector,
              metadata: {
                ...currentVector.metadata,
                ...metadataUpdates,
                updated_at: new Date().toISOString()
              }
            };
            
            // Store updated vector
            return await client.updateVector(dbId, updatedVector);
          } catch (error) {
            console.error(\`Failed to update vector \${vectorId}: \${error.message}\`);
            throw error;
          }
        }
        
        // CONDITIONAL UPDATES: Update only if conditions are met
        async function conditionalUpdate(dbId, vectorId, updates, conditionFn) {
          try {
            // Get current vector
            const currentVector = await client.getVector(dbId, vectorId);
            
            // Check condition
            if (!conditionFn(currentVector)) {
              console.log("Condition not met, skipping update");
              return { success: false, reason: "Condition not met" };
            }
            
            // Apply updates
            const updatedVector = {
              ...currentVector,
              ...updates,
              metadata: {
                ...currentVector.metadata,
                ...updates.metadata,
                updated_at: new Date().toISOString()
              }
            };
            
            // Store updated vector
            const result = await client.updateVector(dbId, updatedVector);
            return { success: true, result };
          } catch (error) {
            console.error(\`Failed to conditionally update vector \${vectorId}: \${error.message}\`);
            throw error;
          }
        }
        
        // BULK IMPORT: Efficiently import large datasets
        async function bulkImportVectors(dbId, vectors, batchSize = 100) {
          console.log(\`Starting bulk import of \${vectors.length} vectors in batches of \${batchSize}\`);
          
          const results = [];
          let totalProcessed = 0;
          
          for (let i = 0; i < vectors.length; i += batchSize) {
            const batch = vectors.slice(i, i + batchSize);
            console.log(\`Processing batch \${Math.floor(i/batchSize) + 1}/\${Math.ceil(vectors.length/batchSize)}\`);
            
            try {
              const batchResult = await client.storeVectorsBatch(dbId, batch);
              results.push(...batchResult);
              totalProcessed += batch.length;
              
              console.log(\`Batch \${Math.floor(i/batchSize) + 1} completed: \${batch.length} vectors\`);
            } catch (error) {
              console.error(\`Batch \${Math.floor(i/batchSize) + 1} failed: \${error.message}\`);
              // Depending on requirements, you might want to continue with other batches
              throw error;
            }
          }
          
          console.log(\`Bulk import completed. Total vectors processed: \${totalProcessed}\`);
          return results;
        }
        
        // USAGE EXAMPLES
        // Partial update
        await updateVectorMetadata("my-database", "vector-001", {
          category: "updated-category",
          tags: ["tag1", "tag2", "tag3"]
        });
        
        // Conditional update
        await conditionalUpdate(
          "my-database", 
          "vector-001", 
          { metadata: { status: "processed" } },
          (vector) => vector.metadata.status === "pending"
        );
        
        // Bulk import
        const largeDataset = Array.from({ length: 5000 }, (_, i) => ({
          id: \`bulk-vector-\${i}\`,
          values: Array.from({ length: 128 }, () => Math.random()),
          metadata: { batch: "bulk-import", index: i }
        }));
        
        await bulkImportVectors("my-database", largeDataset, 500);
      `,
      expectedOutcome: "Ability to implement advanced vector manipulation techniques for complex scenarios"
    }
  ];

  // Update code when step changes
  useEffect(() => {
    setCode(tutorialSteps[currentStep].codeExample);
  }, [currentStep]);

  // Mark step as completed
  const markStepCompleted = (stepId) => {
    if (!completedSteps.includes(stepId)) {
      setCompletedSteps([...completedSteps, stepId]);
      actions.saveAssessmentResult(1, stepId, { 
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
    actions.updateModuleProgress(1, tutorialSteps.length);
    // Unlock next module
    actions.unlockModule(2);
    // Add achievement
    actions.addAchievementIfNotExists({
      id: "vector_manipulation_complete",
      title: "Vector Manipulation Master",
      description: "Completed the Vector Manipulation tutorial module",
      icon: "🔧",
      earnedAt: new Date().toISOString()
    });
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
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Module 2: Vector Manipulation</h1>
        <p className="text-gray-600">Learn CRUD operations and advanced techniques for vector manipulation</p>
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
              initialCode={code}
              moduleId={1}
              stepId={currentStep}
              onCodeChange={setCode}
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
                  className={`w-full text-left p-3 rounded-lg border text-left transition-all ${
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

export default VectorManipulationTutorial;