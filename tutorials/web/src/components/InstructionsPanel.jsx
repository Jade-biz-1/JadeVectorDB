import React from 'react';
import { useTutorial } from '../contexts/TutorialContext';

const InstructionsPanel = () => {
  const { currentModule, setCurrentModule, currentStep, setCurrentStep } = useTutorial();
  
  // Mock module data - in a real implementation, this would come from the tutorial state
  const modules = [
    {
      id: 0,
      title: "Getting Started",
      description: "Learn basic concepts and create your first vector database",
      steps: 3,
      completed: 2
    },
    {
      id: 1,
      title: "Vector Manipulation",
      description: "Learn CRUD operations for vectors",
      steps: 4,
      completed: 0
    },
    {
      id: 2,
      title: "Advanced Search",
      description: "Master similarity search with various metrics",
      steps: 5,
      completed: 0
    },
    {
      id: 3,
      title: "Metadata Filtering",
      description: "Combine semantic and structural search",
      steps: 3,
      completed: 0
    },
    {
      id: 4,
      title: "Index Management",
      description: "Understand and configure indexing strategies",
      steps: 4,
      completed: 0
    },
    {
      id: 5,
      title: "Advanced Features",
      description: "Explore advanced capabilities",
      steps: 3,
      completed: 0
    }
  ];
  
  const currentModuleData = modules[currentModule] || modules[0];
  
  // Step descriptions for each module
  const getStepDescriptions = (moduleId) => {
    switch (moduleId) {
      case 0: // Getting Started
        return [
          "Create a database",
          "Store a vector",
          "Perform similarity search"
        ];
      case 1: // Vector Manipulation
        return [
          "Create multiple vectors",
          "Update vector metadata",
          "Delete vectors",
          "Batch operations"
        ];
      case 2: // Advanced Search
        return [
          "Search with filters",
          "Adjust similarity thresholds",
          "Use different metrics",
          "Pagination",
          "Advanced query options"
        ];
      case 3: // Metadata Filtering
        return [
          "Filter by metadata fields",
          "Combine multiple filters",
          "Range queries"
        ];
      case 4: // Index Management
        return [
          "Create custom indexes",
          "Configure index parameters",
          "Monitor index performance",
          "Delete indexes"
        ];
      case 5: // Advanced Features
        return [
          "Lifecycle management",
          "Analytics and monitoring",
          "Performance tuning"
        ];
      default:
        return Array(modules[moduleId]?.steps || 3).fill("Complete the task");
    }
  };
  
  const stepDescriptions = getStepDescriptions(currentModule);
  
  return (
    <aside className="tutorial-sidebar">
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Tutorial Progress</h2>
        <div className="space-y-2">
          {modules.map((module, index) => (
            <div 
              key={module.id}
              className={`p-3 rounded-lg cursor-pointer transition-colors ${
                index === currentModule 
                  ? 'bg-blue-50 border border-blue-200' 
                  : 'hover:bg-gray-100'
              }`}
              onClick={() => setCurrentModule(index)}
            >
              <div className="flex justify-between items-center">
                <span className="font-medium text-sm">{module.title}</span>
                {module.completed === module.steps ? (
                  <span className="text-green-600 text-xs">✓</span>
                ) : (
                  <span className="text-gray-500 text-xs">{module.completed}/{module.steps}</span>
                )}
              </div>
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-1.5">
                  <div 
                    className="bg-blue-600 h-1.5 rounded-full" 
                    style={{ width: `${(module.completed / module.steps) * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div>
        <h2 className="text-lg font-semibold text-gray-800 mb-3">Current Module: {currentModuleData.title}</h2>
        <p className="text-sm text-gray-600 mb-4">{currentModuleData.description}</p>
        
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
          <h3 className="font-medium text-yellow-800 mb-1">Learning Objective</h3>
          <p className="text-sm text-yellow-700">
            {currentModule === 0 
              ? "Understand basic concepts (vectors, dimensions, similarity) and create your first vector database"
              : currentModule === 1 
              ? "Learn how to perform CRUD operations on vectors"
              : currentModule === 2
              ? "Master advanced search techniques with filters and thresholds"
              : currentModule === 3
              ? "Use metadata filtering to refine search results"
              : currentModule === 4
              ? "Configure and manage indexing strategies for optimal performance"
              : "Explore advanced features for production use"}
          </p>
        </div>
        
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="font-medium text-gray-800">Steps</h3>
            <div className="flex space-x-1">
              <button 
                onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                disabled={currentStep === 0}
                className={`p-1 rounded ${currentStep === 0 ? 'text-gray-400 cursor-not-allowed' : 'text-gray-600 hover:text-gray-800'}`}
              >
                ←
              </button>
              <button 
                onClick={() => setCurrentStep(Math.min(currentModuleData.steps - 1, currentStep + 1))}
                disabled={currentStep === currentModuleData.steps - 1}
                className={`p-1 rounded ${currentStep === currentModuleData.steps - 1 ? 'text-gray-400 cursor-not-allowed' : 'text-gray-600 hover:text-gray-800'}`}
              >
                →
              </button>
            </div>
          </div>
          {[...Array(currentModuleData.steps)].map((_, i) => (
            <div 
              key={i} 
              className={`flex items-center p-2 rounded cursor-pointer ${
                i === currentStep ? 'bg-blue-100' : i < currentStep ? 'bg-green-100 hover:bg-green-200' : 'bg-gray-100 hover:bg-gray-200'
              }`}
              onClick={() => setCurrentStep(i)}
            >
              <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-2 text-xs ${
                i < currentStep ? 'bg-green-500 text-white' : 
                i === currentStep ? 'bg-blue-500 text-white' : 
                'bg-gray-300 text-gray-700'
              }`}>
                {i < currentStep ? '✓' : i + 1}
              </div>
              <span className="text-sm">
                {stepDescriptions[i] || `Step ${i + 1}`}
              </span>
            </div>
          ))}
        </div>
        
        <div className="mt-6 p-3 bg-blue-50 rounded-lg border border-blue-100">
          <h3 className="font-medium text-blue-800 mb-2">Navigation Tips</h3>
          <ul className="text-xs text-blue-700 space-y-1">
            <li>• Click on any module in the sidebar to switch modules</li>
            <li>• Click on any step to jump to that step</li>
            <li>• Use arrow buttons to navigate between steps</li>
            <li>• Progress is automatically saved</li>
          </ul>
        </div>
      </div>
    </aside>
  );
};

export default InstructionsPanel;