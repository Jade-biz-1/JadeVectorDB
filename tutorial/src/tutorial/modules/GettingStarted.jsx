import React, { useState, useEffect } from 'react';
import { useTutorialState } from '../../hooks/useTutorialState';
import CodeEditor from '../../components/CodeEditor';
import VisualDashboard from '../../components/VisualDashboard';
import LivePreviewPanel from '../../components/LivePreviewPanel';

const GettingStarted = () => {
  const { state, actions } = useTutorialState();
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState([]);

  // Tutorial steps for Getting Started module
  const tutorialSteps = [
    {
      id: 0,
      title: "Introduction to Vector Databases",
      description: "Learn what vector databases are and why they're important for AI applications.",
      content: `
        <h3>What is a Vector Database?</h3>
        <p>A vector database is a specialized database designed to store, manage, and search high-dimensional vectors efficiently. These vectors are mathematical representations of data like text, images, or audio that capture their semantic meaning.</p>
        
        <h3>Why Vector Databases Matter</h3>
        <p>Traditional databases excel at exact matches, but they struggle with semantic search - finding similar items based on meaning rather than exact keywords. Vector databases solve this by:</p>
        <ul>
          <li>Converting data into numerical vectors using AI models</li>
          <li>Using similarity algorithms to find related items</li>
          <li>Providing fast search through specialized indexing</li>
        </ul>
        
        <h3>JadeVectorDB Features</h3>
        <p>JadeVectorDB offers:</p>
        <ul>
          <li>High-performance similarity search</li>
          <li>Distributed architecture for scalability</li>
          <li>Multiple indexing algorithms (HNSW, IVF, LSH)</li>
          <li>Rich metadata filtering capabilities</li>
          <li>Integration with popular embedding models</li>
        </ul>
      `,
      codeExample: `
        // JadeVectorDB stores vectors like this:
        const vectorExample = {
          id: "doc-123",
          values: [0.1, 0.8, -0.3, /* ... 125 more dimensions */],
          metadata: {
            title: "Introduction to AI",
            category: "technology",
            author: "Jane Doe",
            created_at: "2023-01-15"
          }
        };
      `,
      expectedOutcome: "Understanding of vector databases and JadeVectorDB capabilities"
    },
    {
      id: 1,
      title: "Creating Your First Database",
      description: "Learn how to create a new vector database with JadeVectorDB.",
      content: `
        <h3>Database Creation</h3>
        <p>Before storing vectors, you need to create a database. Each database has specific configurations:</p>
        <ul>
          <li><strong>Name</strong>: A human-readable identifier</li>
          <li><strong>Vector Dimension</strong>: The size of vectors (e.g., 128, 768, 1536)</li>
          <li><strong>Index Type</strong>: Algorithm for fast search (HNSW, IVF, LSH, etc.)</li>
        </ul>
        
        <h3>Best Practices</h3>
        <p>When creating databases:</p>
        <ul>
          <li>Choose vector dimensions that match your embedding model</li>
          <li>Select an index type based on your performance needs</li>
          <li>Use descriptive names for easy identification</li>
        </ul>
      `,
      codeExample: `
        // Create a new vector database
        const db = await client.createDatabase({
          name: "my-first-database",
          vectorDimension: 128,
          indexType: "HNSW",
          indexParameters: {
            M: 16,
            efConstruction: 200,
            efSearch: 64
          }
        });
        
        console.log("Database created:", db.id);
      `,
      expectedOutcome: "Successfully create a vector database with proper configuration"
    },
    {
      id: 2,
      title: "Storing Your First Vector",
      description: "Learn how to store vector embeddings with metadata in your database.",
      content: `
        <h3>Vector Storage</h3>
        <p>Vectors consist of:</p>
        <ul>
          <li><strong>ID</strong>: Unique identifier for the vector</li>
          <li><strong>Values</strong>: Array of numerical values (the embedding)</li>
          <li><strong>Metadata</strong>: Additional information (tags, categories, etc.)</li>
        </ul>
        
        <h3>Metadata Best Practices</h3>
        <p>Effective metadata enhances search capabilities:</p>
        <ul>
          <li>Use consistent field names across vectors</li>
          <li>Include relevant categorical information</li>
          <li>Add timestamps for temporal filtering</li>
          <li>Include ownership/access control information</li>
        </ul>
      `,
      codeExample: `
        // Store a vector with metadata
        const vector = {
          id: "vector-001",
          values: [0.1, 0.2, 0.3, 0.4, 0.5], // Simplified for example
          metadata: {
            title: "Sample Document",
            category: "example",
            tags: ["tutorial", "getting-started"],
            author: "Tutorial User",
            created_at: new Date().toISOString()
          }
        };
        
        const result = await client.storeVector(db.id, vector);
        console.log("Vector stored:", result.id);
      `,
      expectedOutcome: "Successfully store a vector with metadata in the database"
    },
    {
      id: 3,
      title: "Performing Your First Search",
      description: "Learn how to perform similarity search to find related vectors.",
      content: `
        <h3>Similarity Search</h3>
        <p>Similarity search finds vectors similar to a query vector:</p>
        <ul>
          <li><strong>Query Vector</strong>: The vector you're comparing against</li>
          <li><strong>Top K</strong>: Number of similar results to return</li>
          <li><strong>Threshold</strong>: Minimum similarity score (0.0 to 1.0)</li>
        </ul>
        
        <h3>Similarity Metrics</h3>
        <p>JadeVectorDB supports multiple metrics:</p>
        <ul>
          <li><strong>Cosine Similarity</strong>: Measures angle between vectors</li>
          <li><strong>Euclidean Distance</strong>: Measures straight-line distance</li>
          <li><strong>Dot Product</strong>: Measures vector alignment</li>
        </ul>
      `,
      codeExample: `
        // Perform similarity search
        const queryVector = {
          values: [0.15, 0.25, 0.35, 0.45, 0.55] // Similar to stored vector
        };
        
        const searchResults = await client.search(db.id, queryVector, {
          topK: 5,
          threshold: 0.7,
          includeMetadata: true
        });
        
        console.log("Search results:", searchResults);
      `,
      expectedOutcome: "Successfully perform a similarity search and interpret results"
    }
  ];

  // Mark step as completed
  const markStepCompleted = (stepId) => {
    if (!completedSteps.includes(stepId)) {
      setCompletedSteps([...completedSteps, stepId]);
      actions.saveAssessmentResult(0, stepId, { 
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
    actions.updateModuleProgress(0, tutorialSteps.length);
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
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Module 1: Getting Started</h1>
        <p className="text-gray-600">Learn basic concepts and create your first vector database</p>
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
              moduleId={0}
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

export default GettingStarted;