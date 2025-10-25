import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { 
  BookOpen, 
  Play, 
  Copy, 
  Check, 
  ChevronDown, 
  ChevronRight,
  Code,
  Terminal
} from "lucide-react";

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
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const runCode = (code) => {
    if (onRunCode) {
      onRunCode(code);
    }
  };

  const currentDoc = apiDocs[activeEndpoint];

  return (
    <Card className="h-full overflow-hidden flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <BookOpen className="h-5 w-5" />
          API Reference
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Interactive documentation with runnable examples
        </p>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-4 overflow-hidden">
        <Tabs 
          value={activeEndpoint} 
          onValueChange={setActiveEndpoint}
          className="flex-1 flex flex-col overflow-hidden h-full"
        >
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="createDatabase">
              <Code className="h-3 w-3 mr-1" />
              Create DB
            </TabsTrigger>
            <TabsTrigger value="addVector">
              <Code className="h-3 w-3 mr-1" />
              Add Vector
            </TabsTrigger>
            <TabsTrigger value="similaritySearch">
              <Code className="h-3 w-3 mr-1" />
              Search
            </TabsTrigger>
          </TabsList>
          
          {Object.entries(apiDocs).map(([key, doc]) => (
            <TabsContent key={key} value={key} className="flex-1 flex flex-col overflow-hidden h-full">
              <div className="flex-1 overflow-y-auto">
                <div className="mb-6">
                  <div className="flex items-center gap-2 mb-2">
                    <Badge variant="secondary">{doc.method}</Badge>
                    <span className="font-mono text-sm">{doc.endpoint}</span>
                  </div>
                  <h3 className="text-xl font-semibold">{doc.title}</h3>
                  <p className="text-muted-foreground mt-2">{doc.description}</p>
                </div>
                
                <div className="space-y-4 mb-6">
                  <div className="space-y-2">
                    <h4 className="font-semibold flex items-center">
                      <button 
                        onClick={() => toggleSection(`params-${key}`)}
                        className="flex items-center gap-1 w-full text-left"
                      >
                        {expandedSections[`params-${key}`] ? 
                          <ChevronDown className="h-4 w-4" /> : 
                          <ChevronRight className="h-4 w-4" />
                        }
                        Parameters
                      </button>
                    </h4>
                    {expandedSections[`params-${key}`] && (
                      <div className="ml-5 space-y-2">
                        {doc.parameters.map((param, idx) => (
                          <div key={idx} className="text-sm">
                            <span className="font-medium">{param.name}</span> 
                            {param.required && <span className="text-red-500 ml-1">*</span>}
                            <span className="text-muted-foreground ml-2">
                              ({param.type})
                            </span>
                            {param.default && (
                              <span className="text-muted-foreground ml-2">
                                = {param.default}
                              </span>
                            )}
                            <div className="ml-2">{param.description}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <Separator />
                  
                  <div className="space-y-2">
                    <h4 className="font-semibold">Examples</h4>
                    <Tabs value={selectedLanguage} onValueChange={setSelectedLanguage}>
                      <TabsList className="grid w-full grid-cols-3">
                        <TabsTrigger value="javascript">JavaScript</TabsTrigger>
                        <TabsTrigger value="python">Python</TabsTrigger>
                        <TabsTrigger value="curl">cURL</TabsTrigger>
                      </TabsList>
                      <TabsContent value="javascript" className="mt-2">
                        <div className="relative">
                          <pre className="bg-muted rounded-md p-4 text-sm overflow-x-auto">
                            <code>{doc.examples.javascript}</code>
                          </pre>
                          <div className="absolute top-2 right-2 flex gap-1">
                            <Button 
                              size="sm" 
                              variant="secondary" 
                              onClick={() => copyCode(doc.examples.javascript)}
                            >
                              {copiedCode === doc.examples.javascript ? (
                                <Check className="h-4 w-4" />
                              ) : (
                                <Copy className="h-4 w-4" />
                              )}
                            </Button>
                            <Button 
                              size="sm" 
                              variant="default" 
                              onClick={() => runCode(doc.examples.javascript)}
                            >
                              <Play className="h-4 w-4 mr-1" />
                              Run
                            </Button>
                          </div>
                        </div>
                      </TabsContent>
                      <TabsContent value="python" className="mt-2">
                        <div className="relative">
                          <pre className="bg-muted rounded-md p-4 text-sm overflow-x-auto">
                            <code>{doc.examples.python}</code>
                          </pre>
                          <div className="absolute top-2 right-2 flex gap-1">
                            <Button 
                              size="sm" 
                              variant="secondary" 
                              onClick={() => copyCode(doc.examples.python)}
                            >
                              {copiedCode === doc.examples.python ? (
                                <Check className="h-4 w-4" />
                              ) : (
                                <Copy className="h-4 w-4" />
                              )}
                            </Button>
                            <Button 
                              size="sm" 
                              variant="default" 
                              onClick={() => runCode(doc.examples.python)}
                            >
                              <Play className="h-4 w-4 mr-1" />
                              Run
                            </Button>
                          </div>
                        </div>
                      </TabsContent>
                      <TabsContent value="curl" className="mt-2">
                        <div className="relative">
                          <pre className="bg-muted rounded-md p-4 text-sm overflow-x-auto">
                            <code>{doc.examples.curl}</code>
                          </pre>
                          <div className="absolute top-2 right-2 flex gap-1">
                            <Button 
                              size="sm" 
                              variant="secondary" 
                              onClick={() => copyCode(doc.examples.curl)}
                            >
                              {copiedCode === doc.examples.curl ? (
                                <Check className="h-4 w-4" />
                              ) : (
                                <Copy className="h-4 w-4" />
                              )}
                            </Button>
                            <Button 
                              size="sm" 
                              variant="default" 
                              onClick={() => runCode(doc.examples.curl)}
                            >
                              <Play className="h-4 w-4 mr-1" />
                              Run
                            </Button>
                          </div>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </div>
                  
                  <Separator />
                  
                  <div className="space-y-2">
                    <h4 className="font-semibold flex items-center">
                      <button 
                        onClick={() => toggleSection(`response-${key}`)}
                        className="flex items-center gap-1 w-full text-left"
                      >
                        {expandedSections[`response-${key}`] ? 
                          <ChevronDown className="h-4 w-4" /> : 
                          <ChevronRight className="h-4 w-4" />
                        }
                        Response Format
                      </button>
                    </h4>
                    {expandedSections[`response-${key}`] && (
                      <div className="ml-5 space-y-4">
                        <div>
                          <h5 className="font-medium">Success Response ({doc.response.success.code})</h5>
                          <pre className="bg-muted rounded-md p-4 text-sm overflow-x-auto">
                            <code>{JSON.stringify(doc.response.success.body, null, 2)}</code>
                          </pre>
                        </div>
                        <div>
                          <h5 className="font-medium">Error Response ({doc.response.error.code})</h5>
                          <pre className="bg-muted rounded-md p-4 text-sm overflow-x-auto">
                            <code>{JSON.stringify({ error: doc.response.error.message }, null, 2)}</code>
                          </pre>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default InteractiveAPIDocs;