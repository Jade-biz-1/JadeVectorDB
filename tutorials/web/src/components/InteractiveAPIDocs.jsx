import React, { useState, useEffect } from 'react';

const InteractiveAPIDocs = ({ onRunCode }) => {
  const [activeEndpoint, setActiveEndpoint] = useState('createDatabase');
  const [expandedSections, setExpandedSections] = useState({});
  const [copiedCode, setCopiedCode] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('javascript');

  // Sample API documentation data
  const apiDocs = {
    createDatabase: {
      title: 'Create Database',
      description: 'Create a new vector database with specified configuration',
      method: 'POST',
      endpoint: '/v1/databases',
      parameters: [
        { name: 'name', type: 'string', required: true, description: 'Unique name for the database' },
        { name: 'dimensions', type: 'number', required: true, description: 'Vector dimension size' },
        { name: 'description', type: 'string', required: false, description: 'Optional description' },
      ],
      examples: {
        javascript: `// Create a new database
const response = await fetch('/v1/databases', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
  },
  body: JSON.stringify({
    name: 'my_vectors',
    dimensions: 128,
    description: 'My vector database'
  })
});

const result = await response.json();
console.log(result);`,
        python: `# Create a new database
import requests

response = requests.post(
    "http://localhost:8000/v1/databases",
    json={
        "name": "my_vectors",
        "dimensions": 128,
        "description": "My vector database"
    },
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"
    }
)

result = response.json()
print(result)`,
        curl: `curl -X POST http://localhost:8000/v1/databases \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
    "name": "my_vectors",
    "dimensions": 128,
    "description": "My vector database"
  }'`
      },
      response: {
        success: {
          code: 201,
          body: {
            id: "db_123456789",
            name: "my_vectors",
            dimensions: 128,
            description: "My vector database",
            created_at: "2023-10-10T12:00:00Z"
          }
        },
        error: {
          code: 400,
          message: "Invalid request parameters"
        }
      }
    },
    addVector: {
      title: 'Add Vector',
      description: 'Add a vector with optional metadata to a database',
      method: 'POST',
      endpoint: '/v1/databases/{databaseId}/vectors',
      parameters: [
        { name: 'databaseId', type: 'string', required: true, description: 'ID of the target database' },
        { name: 'vector', type: 'array', required: true, description: 'Array of vector values' },
        { name: 'id', type: 'string', required: false, description: 'Optional vector ID' },
        { name: 'metadata', type: 'object', required: false, description: 'Optional metadata object' },
      ],
      examples: {
        javascript: `// Add a vector to the database
const response = await fetch('/v1/databases/my_vectors/vectors', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
  },
  body: JSON.stringify({
    id: 'vector_1',
    vector: [0.1, 0.2, 0.3, 0.4],
    metadata: {
      category: 'image',
      tags: ['red', 'car']
    }
  })
});

const result = await response.json();
console.log(result);`,
        python: `# Add a vector to the database
import requests

response = requests.post(
    "http://localhost:8000/v1/databases/my_vectors/vectors",
    json={
        "id": "vector_1",
        "vector": [0.1, 0.2, 0.3, 0.4],
        "metadata": {
            "category": "image",
            "tags": ["red", "car"]
        }
    },
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"
    }
)

result = response.json()
print(result)`,
        curl: `curl -X POST http://localhost:8000/v1/databases/my_vectors/vectors \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
    "id": "vector_1",
    "vector": [0.1, 0.2, 0.3, 0.4],
    "metadata": {
      "category": "image",
      "tags": ["red", "car"]
    }
  }'`
      },
      response: {
        success: {
          code: 201,
          body: {
            id: "vector_1",
            database_id: "db_123456789",
            created_at: "2023-10-10T12:05:00Z"
          }
        },
        error: {
          code: 400,
          message: "Invalid vector dimensions"
        }
      }
    },
    similaritySearch: {
      title: 'Similarity Search',
      description: 'Find vectors most similar to the query vector',
      method: 'POST',
      endpoint: '/v1/databases/{databaseId}/search',
      parameters: [
        { name: 'databaseId', type: 'string', required: true, description: 'ID of the target database' },
        { name: 'vector', type: 'array', required: true, description: 'Query vector to find similar vectors' },
        { name: 'top_k', type: 'number', required: false, default: 10, description: 'Number of results to return' },
        { name: 'threshold', type: 'number', required: false, description: 'Similarity threshold filter' },
      ],
      examples: {
        javascript: `// Perform similarity search
const response = await fetch('/v1/databases/my_vectors/search', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
  },
  body: JSON.stringify({
    vector: [0.5, 0.6, 0.7, 0.8],
    top_k: 5,
    threshold: 0.7
  })
});

const result = await response.json();
console.log(result);`,
        python: `# Perform similarity search
import requests

response = requests.post(
    "http://localhost:8000/v1/databases/my_vectors/search",
    json={
        "vector": [0.5, 0.6, 0.7, 0.8],
        "top_k": 5,
        "threshold": 0.7
    },
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"
    }
)

result = response.json()
print(result)`,
        curl: `curl -X POST http://localhost:8000/v1/databases/my_vectors/search \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
    "vector": [0.5, 0.6, 0.7, 0.8],
    "top_k": 5,
    "threshold": 0.7
  }'`
      },
      response: {
        success: {
          code: 200,
          body: {
            results: [
              {
                id: "vector_3",
                similarity: 0.92,
                metadata: { category: "image", tags: ["blue", "sky"] }
              },
              {
                id: "vector_1",
                similarity: 0.87,
                metadata: { category: "image", tags: ["red", "car"] }
              }
            ]
          }
        },
        error: {
          code: 400,
          message: "Invalid query vector dimensions"
        }
      }
    }
  };

  const toggleSection = (sectionId) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  const copyCode = (code) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const runCode = (code) => {
    if (onRunCode) {
      onRunCode(code);
    }
  };

  const currentDoc = apiDocs[activeEndpoint];

  return (
    <div className="border border-gray-200 rounded-lg p-4 h-full overflow-hidden flex flex-col">
      <div className="pb-3 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
          <span>📖</span>
          API Reference
        </h3>
        <p className="text-sm text-gray-600 mt-1">
          Interactive documentation with runnable examples
        </p>
      </div>
      
      <div className="flex-1 flex flex-col gap-4 pt-4 overflow-hidden">
        <div className="flex border-b border-gray-200">
          <button
            className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
              activeEndpoint === 'createDatabase' 
                ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveEndpoint('createDatabase')}
          >
            Create DB
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
              activeEndpoint === 'addVector' 
                ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveEndpoint('addVector')}
          >
            Add Vector
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
              activeEndpoint === 'similaritySearch' 
                ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveEndpoint('similaritySearch')}
          >
            Search
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto">
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-2">
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                currentDoc.method === 'POST' ? 'bg-blue-100 text-blue-800' :
                currentDoc.method === 'GET' ? 'bg-green-100 text-green-800' :
                currentDoc.method === 'PUT' ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {currentDoc.method}
              </span>
              <span className="font-mono text-sm text-gray-700">{currentDoc.endpoint}</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-800">{currentDoc.title}</h3>
            <p className="text-gray-600 mt-2">{currentDoc.description}</p>
          </div>
          
          <div className="space-y-4 mb-6">
            <div className="space-y-2">
              <div 
                className="flex items-center gap-2 cursor-pointer"
                onClick={() => toggleSection(`params-${activeEndpoint}`)}
              >
                <span>▶</span>
                <h4 className="font-semibold">Parameters</h4>
              </div>
              {expandedSections[`params-${activeEndpoint}`] && (
                <div className="ml-5 space-y-2 pl-4 border-l-2 border-gray-200">
                  {currentDoc.parameters.map((param, idx) => (
                    <div key={idx} className="text-sm">
                      <div className="font-medium">
                        {param.name}
                        {param.required && <span className="text-red-500 ml-1">*</span>}
                        <span className="text-gray-500 ml-2">({param.type})</span>
                        {param.default !== undefined && (
                          <span className="text-gray-500 ml-2">= {param.default}</span>
                        )}
                      </div>
                      <div className="ml-2 text-gray-600">{param.description}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            <div className="border-t border-gray-200 pt-4">
              <h4 className="font-semibold mb-3">Examples</h4>
              <div className="border-b border-gray-200 mb-2">
                <div className="flex">
                  <button
                    className={`px-3 py-1 text-sm ${
                      selectedLanguage === 'javascript' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    onClick={() => setSelectedLanguage('javascript')}
                  >
                    JavaScript
                  </button>
                  <button
                    className={`px-3 py-1 text-sm ${
                      selectedLanguage === 'python' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    onClick={() => setSelectedLanguage('python')}
                  >
                    Python
                  </button>
                  <button
                    className={`px-3 py-1 text-sm ${
                      selectedLanguage === 'curl' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    onClick={() => setSelectedLanguage('curl')}
                  >
                    cURL
                  </button>
                </div>
              </div>
              
              <div className="relative">
                <pre className="bg-gray-100 rounded-md p-4 text-sm overflow-x-auto">
                  <code>{currentDoc.examples[selectedLanguage]}</code>
                </pre>
                <div className="absolute top-2 right-2 flex gap-1">
                  <button
                    className="px-2 py-1 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded text-sm"
                    onClick={() => copyCode(currentDoc.examples[selectedLanguage])}
                  >
                    {copiedCode === currentDoc.examples[selectedLanguage] ? '✓ Copied' : '📋 Copy'}
                  </button>
                  <button
                    className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm flex items-center gap-1"
                    onClick={() => runCode(currentDoc.examples[selectedLanguage])}
                  >
                    <span>▶</span> Run
                  </button>
                </div>
              </div>
            </div>
            
            <div className="border-t border-gray-200 pt-4">
              <div 
                className="flex items-center gap-2 cursor-pointer"
                onClick={() => toggleSection(`response-${activeEndpoint}`)}
              >
                <span>▶</span>
                <h4 className="font-semibold">Response Format</h4>
              </div>
              {expandedSections[`response-${activeEndpoint}`] && (
                <div className="ml-5 space-y-4 pl-4 border-l-2 border-gray-200">
                  <div>
                    <h5 className="font-medium text-green-700">Success Response ({currentDoc.response.success.code})</h5>
                    <pre className="bg-gray-100 rounded-md p-4 text-sm overflow-x-auto">
                      <code>{JSON.stringify(currentDoc.response.success.body, null, 2)}</code>
                    </pre>
                  </div>
                  <div>
                    <h5 className="font-medium text-red-700">Error Response ({currentDoc.response.error.code})</h5>
                    <pre className="bg-gray-100 rounded-md p-4 text-sm overflow-x-auto">
                      <code>{JSON.stringify({ error: currentDoc.response.error.message }, null, 2)}</code>
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InteractiveAPIDocs;