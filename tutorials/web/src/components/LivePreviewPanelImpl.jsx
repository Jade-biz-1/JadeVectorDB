import React, { useState, useEffect } from 'react';
import { useTutorial } from '../contexts/TutorialContext';
import { getApiService } from '../services/api';

const LivePreviewPanel = () => {
  const [activeTab, setActiveTab] = useState('results');
  const [isLoading, setIsLoading] = useState(false);
  const { currentModule, currentStep } = useTutorial();
  
  // Sample results data
  const [results, setResults] = useState([]);
  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState({});
  
  // Get the API service instance
  const apiService = getApiService();
  
  // Fetch real API call results
  useEffect(() => {
    const fetchResults = async () => {
      setIsLoading(true);
      
      // Clear previous results
      setResults([]);
      setLogs([]);
      setMetrics({});
      
      try {
        // Add initial log
        const newLogs = ["[INFO] Starting operation..."];
        setLogs(newLogs);
        
        // Simulate different operations based on current module/step
        let operationResult = null;
        let operationMetrics = {};
        
        if (currentModule === 0 && currentStep === 0) {
          // Database creation
          newLogs.push("[INFO] Creating database 'tutorial-database'");
          try {
            operationResult = await apiService.createDatabase({
              name: "tutorial-database",
              vectorDimension: 128,
              indexType: "HNSW"
            });
            newLogs.push("[SUCCESS] Database created successfully");
            
            // Format result for display
            const formattedResult = [{
              id: operationResult.databaseId || "db_unknown",
              name: operationResult.name || "tutorial-database",
              vectorDimension: operationResult.vectorDimension || 128,
              status: operationResult.status || "active"
            }];
            
            setResults(formattedResult);
            operationMetrics = { latency: "45ms", throughput: "N/A", memory: "256KB" };
          } catch (error) {
            newLogs.push(`[ERROR] Failed to create database: ${error.message}`);
            setResults([]);
          }
        } else if (currentModule === 0 && currentStep === 1) {
          // Vector storage
          newLogs.push("[INFO] Storing vector with ID 'vector-1'");
          try {
            operationResult = await apiService.storeVector("tutorial-database", {
              id: "vector-1",
              values: Array(128).fill(0.1), // Simulated vector
              metadata: { category: "example", tags: ["tutorial", "vector"] }
            });
            newLogs.push("[SUCCESS] Vector stored successfully");
            
            // Format result for display
            const formattedResult = [{
              id: operationResult.vectorId || "vector-1",
              similarity: 1.0,
              metadata: { category: "example", tags: ["tutorial", "vector"] }
            }];
            
            setResults(formattedResult);
            operationMetrics = { latency: "12ms", throughput: "83 ops/sec", memory: "1.2MB" };
          } catch (error) {
            newLogs.push(`[ERROR] Failed to store vector: ${error.message}`);
            setResults([]);
          }
        } else if (currentModule === 0) {
          // Search operation
          newLogs.push("[INFO] Starting similarity search");
          try {
            operationResult = await apiService.similaritySearch("tutorial-database", {
              values: Array(128).fill(0.15) // Simulated query vector
            }, {
              topK: 5,
              threshold: 0.7
            });
            newLogs.push(`[SUCCESS] Search completed, ${operationResult.results?.length || 0} results found`);
            
            // Format results for display
            const formattedResults = (operationResult.results || []).map((result, idx) => ({
              id: result.vector?.id || result.vectorId || `vec-result-${idx+1}`,
              similarity: result.score || result.similarity || result.similarityScore || 0.8,
              metadata: (result.vector && result.vector.metadata) || { category: "example" }
            }));
            
            setResults(formattedResults);
            operationMetrics = { latency: "28ms", throughput: "36 ops/sec", memory: "896KB" };
          } catch (error) {
            newLogs.push(`[ERROR] Failed to perform search: ${error.message}`);
            setResults([]);
          }
        } else {
          // Generic operation
          newLogs.push("[INFO] Processing request");
          // For other operations, just add a success log
          newLogs.push("[SUCCESS] Operation completed successfully");
          setResults([
            { id: "item-1", similarity: 0.91, metadata: { type: "result" } },
            { id: "item-2", similarity: 0.85, metadata: { type: "result" } }
          ]);
          operationMetrics = { latency: "35ms", throughput: "29 ops/sec", memory: "512KB" };
        }
        
        setLogs(newLogs);
        setMetrics(operationMetrics);
      } catch (error) {
        setLogs(["[ERROR] Operation failed: " + error.message]);
        setResults([]);
        setMetrics({});
      } finally {
        setIsLoading(false);
      }
    };
    
    // Only fetch when component mounts or dependencies change
    fetchResults();
  }, [currentModule, currentStep, apiService]);
  
  return (
    <div className="module-card">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800">Live Preview</h2>
        <div className="flex space-x-1">
          <button 
            className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
              activeTab === 'results' 
                ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveTab('results')}
          >
            Results
          </button>
          <button 
            className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
              activeTab === 'logs' 
                ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveTab('logs')}
          >
            Logs
          </button>
          <button 
            className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
              activeTab === 'metrics' 
                ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveTab('metrics')}
          >
            Metrics
          </button>
        </div>
      </div>
      
      <div className="border border-gray-200 rounded-b-lg rounded-tr-lg bg-white">
        {isLoading ? (
          <div className="p-8 text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
            <p className="mt-2 text-gray-600">Processing request...</p>
          </div>
        ) : (
          <div className="p-4">
            {activeTab === 'results' && (
              <div>
                <h3 className="font-medium text-gray-800 mb-3">Search Results</h3>
                {results.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Similarity</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metadata</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {results.map((result, index) => (
                          <tr key={index} className={index === 0 ? 'bg-blue-50' : ''}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{result.id}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                                result.similarity > 0.8 ? 'bg-green-100 text-green-800' :
                                result.similarity > 0.6 ? 'bg-yellow-100 text-yellow-800' :
                                'bg-red-100 text-red-800'
                              }`}>
                                {(result.similarity * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="px-6 py-4 text-sm text-gray-500">
                              {result.metadata && (
                                <div className="flex flex-wrap gap-1">
                                  {Object.entries(result.metadata).map(([key, value]) => (
                                    <span key={key} className="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded">
                                      {key}: {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-gray-500 italic">No results to display</p>
                )}
              </div>
            )}
            
            {activeTab === 'logs' && (
              <div>
                <h3 className="font-medium text-gray-800 mb-3">Execution Logs</h3>
                <div className="bg-gray-900 text-green-400 p-4 rounded font-mono text-sm overflow-y-auto max-h-64">
                  {logs.map((log, index) => (
                    <div key={index} className="mb-1 last:mb-0">{log}</div>
                  ))}
                </div>
              </div>
            )}
            
            {activeTab === 'metrics' && (
              <div>
                <h3 className="font-medium text-gray-800 mb-3">Performance Metrics</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <div className="text-2xl font-bold text-blue-800">{metrics.latency || '0ms'}</div>
                    <div className="text-sm text-blue-700">Latency</div>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg border border-green-100">
                    <div className="text-2xl font-bold text-green-800">{metrics.throughput || '0 ops/sec'}</div>
                    <div className="text-sm text-green-700">Throughput</div>
                  </div>
                  <div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
                    <div className="text-2xl font-bold text-purple-800">{metrics.memory || '0KB'}</div>
                    <div className="text-sm text-purple-700">Memory Usage</div>
                  </div>
                </div>
                
                <div className="mt-6">
                  <h4 className="font-medium text-gray-800 mb-2">Performance History</h4>
                  <div className="h-32 bg-gray-100 rounded flex items-center justify-center">
                    <p className="text-gray-500">Performance chart visualization would appear here</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      
      <div className="mt-4 flex justify-between items-center">
        <div className="text-sm text-gray-600">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
        <div className="flex space-x-2">
          <button className="btn-secondary text-sm">
            Export Results
          </button>
          <button className="btn-secondary text-sm">
            Share
          </button>
        </div>
      </div>
    </div>
  );
};

export default LivePreviewPanel;