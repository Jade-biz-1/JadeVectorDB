import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Cpu, 
  HardDrive, 
  Database, 
  Activity,
  RotateCcw,
  AlertTriangle
} from "lucide-react";

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
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Activity className="h-5 w-5" />
            Resource Usage
          </CardTitle>
          <Button size="sm" variant="outline" onClick={resetSession}>
            <RotateCcw className="h-4 w-4 mr-1" />
            Reset
          </Button>
        </div>
        {isOverLimit && (
          <div className="flex items-center gap-2 mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded-md">
            <AlertTriangle className="h-4 w-4 text-yellow-600" />
            <span className="text-sm text-yellow-700">
              You've reached resource limits. Some operations may be restricted.
            </span>
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="flex items-center gap-1">
              <Cpu className="h-3 w-3" />
              API Calls (per minute)
            </span>
            <span>{resourceUsage.api_calls_made} / {limits.max_api_calls_per_minute}</span>
          </div>
          <Progress value={apiCallPercentage} className="h-2" />
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="flex items-center gap-1">
              <Database className="h-3 w-3" />
              Vectors Stored
            </span>
            <span>{resourceUsage.vectors_stored} / {limits.max_vectors_per_session}</span>
          </div>
          <Progress value={vectorStoragePercentage} className="h-2" />
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="flex items-center gap-1">
              <Database className="h-3 w-3" />
              Databases Created
            </span>
            <span>{resourceUsage.databases_created} / {limits.max_databases_per_session}</span>
          </div>
          <Progress value={databaseStoragePercentage} className="h-2" />
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="flex items-center gap-1">
              <HardDrive className="h-3 w-3" />
              Memory Usage
            </span>
            <span>{memoryUsedMB}MB / {maxMemoryMB}MB</span>
          </div>
          <Progress value={memoryUsagePercentage} className="h-2" />
        </div>
        
        <div className="pt-2">
          <div className="text-xs text-muted-foreground">
            <p className="mb-1">Resource limits help ensure fair usage of the tutorial environment.</p>
            <p>Reset your session to clear usage counters if needed.</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ResourceUsageMonitor;