import React, { useState, useEffect } from 'react';

const ResourceUsageMonitor = ({ sessionId }) => {
  const [resourceUsage, setResourceUsage] = useState({
    api_calls_made: 0,
    vectors_stored: 0,
    databases_created: 0,
    memory_used_bytes: 0
  });
  
  const [limits, setLimits] = useState({
    max_api_calls_per_minute: 60,
    max_vectors_per_session: 1000,
    max_databases_per_session: 10,
    max_memory_per_session_bytes: 104857600 // 100 MB
  });
  
  const [isOverLimit, setIsOverLimit] = useState(false);
  
  // Simulate getting resource usage from backend
  useEffect(() => {
    const fetchResourceUsage = async () => {
      // In a real implementation, this would fetch from the backend API
      // For the tutorial, we'll simulate this with dummy data
      
      // Simulate usage
      const simulatedUsage = {
        api_calls_made: Math.floor(Math.random() * 60),
        vectors_stored: Math.floor(Math.random() * 1000),
        databases_created: Math.floor(Math.random() * 10),
        memory_used_bytes: Math.floor(Math.random() * 104857600) // 0-100MB
      };
      
      setResourceUsage(simulatedUsage);
      
      // Check if any limits are exceeded
      const overLimit = 
        simulatedUsage.api_calls_made >= limits.max_api_calls_per_minute ||
        simulatedUsage.vectors_stored >= limits.max_vectors_per_session ||
        simulatedUsage.databases_created >= limits.max_databases_per_session ||
        simulatedUsage.memory_used_bytes >= limits.max_memory_per_session_bytes;
      
      setIsOverLimit(overLimit);
    };
    
    // Initial fetch
    fetchResourceUsage();
    
    // Set up interval to update resource usage
    const interval = setInterval(fetchResourceUsage, 5000);
    
    return () => clearInterval(interval);
  }, [sessionId, limits]);
  
  const resetSession = async () => {
    // In a real implementation, this would call the API to reset the session
    setResourceUsage({
      api_calls_made: 0,
      vectors_stored: 0,
      databases_created: 0,
      memory_used_bytes: 0
    });
  };
  
  // Calculate percentages for progress bars
  const apiCallPercentage = Math.min(100, 
    (resourceUsage.api_calls_made / limits.max_api_calls_per_minute) * 100
  );
  
  const vectorStoragePercentage = Math.min(100, 
    (resourceUsage.vectors_stored / limits.max_vectors_per_session) * 100
  );
  
  const databaseStoragePercentage = Math.min(100, 
    (resourceUsage.databases_created / limits.max_databases_per_session) * 100
  );
  
  const memoryUsagePercentage = Math.min(100, 
    (resourceUsage.memory_used_bytes / limits.max_memory_per_session_bytes) * 100
  );
  
  // Convert bytes to MB for display
  const memoryUsedMB = (resourceUsage.memory_used_bytes / (1024 * 1024)).toFixed(2);
  const maxMemoryMB = (limits.max_memory_per_session_bytes / (1024 * 1024)).toFixed(2);
  
  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
          <span>📊</span>
          Resource Usage
        </h3>
        <button 
          className="text-sm px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded flex items-center gap-1"
          onClick={resetSession}
        >
          <span>↺</span> Reset
        </button>
      </div>
      
      {isOverLimit && (
        <div className="flex items-center gap-2 mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded-md">
          <span className="text-yellow-600">⚠️</span>
          <span className="text-sm text-yellow-700">
            You've reached resource limits. Some operations may be restricted.
          </span>
        </div>
      )}
      
      <div className="space-y-4 mt-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="flex items-center gap-1">
              <span>⏱️</span>
              API Calls (per minute)
            </span>
            <span>{resourceUsage.api_calls_made} / {limits.max_api_calls_per_minute}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full ${
                apiCallPercentage > 80 ? 'bg-red-500' : 'bg-blue-500'
              }`} 
              style={{ width: `${apiCallPercentage}%` }}
            ></div>
          </div>
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="flex items-center gap-1">
              <span>📦</span>
              Vectors Stored
            </span>
            <span>{resourceUsage.vectors_stored} / {limits.max_vectors_per_session}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full ${
                vectorStoragePercentage > 80 ? 'bg-red-500' : 'bg-green-500'
              }`} 
              style={{ width: `${vectorStoragePercentage}%` }}
            ></div>
          </div>
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="flex items-center gap-1">
              <span>🗄️</span>
              Databases Created
            </span>
            <span>{resourceUsage.databases_created} / {limits.max_databases_per_session}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full ${
                databaseStoragePercentage > 80 ? 'bg-red-500' : 'bg-purple-500'
              }`} 
              style={{ width: `${databaseStoragePercentage}%` }}
            ></div>
          </div>
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="flex items-center gap-1">
              <span>💾</span>
              Memory Usage
            </span>
            <span>{memoryUsedMB}MB / {maxMemoryMB}MB</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full ${
                memoryUsagePercentage > 80 ? 'bg-red-500' : 'bg-yellow-500'
              }`} 
              style={{ width: `${memoryUsagePercentage}%` }}
            ></div>
          </div>
        </div>
      </div>
      
      <div className="pt-4">
        <p className="text-xs text-gray-600">
          Resource limits help ensure fair usage of the tutorial environment.
          Reset your session to clear usage counters if needed.
        </p>
      </div>
    </div>
  );
};

export default ResourceUsageMonitor;