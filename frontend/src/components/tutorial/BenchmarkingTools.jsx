import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  BarChart3, 
  Play, 
  Pause, 
  RotateCcw,
  TrendingUp,
  Clock,
  Database,
  Zap,
  Activity
} from "lucide-react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const BenchmarkingTools = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [activeBenchmark, setActiveBenchmark] = useState('vectorSearch');
  const [benchmarkResults, setBenchmarkResults] = useState({
    vectorSearch: {
      avgLatency: 0,
      throughput: 0,
      memoryUsage: 0,
      progress: 0,
      dataPoints: []
    },
    databaseOperations: {
      avgLatency: 0,
      throughput: 0,
      memoryUsage: 0,
      progress: 0,
      dataPoints: []
    },
    indexOperations: {
      avgLatency: 0,
      throughput: 0,
      memoryUsage: 0,
      progress: 0,
      dataPoints: []
    }
  });

  // Chart options
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Performance Metrics Over Time',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  // Initialize chart data
  const getChartData = (type) => {
    const results = benchmarkResults[type];
    return {
      labels: results.dataPoints.map((_, i) => `Test ${i + 1}`),
      datasets: [
        {
          label: 'Latency (ms)',
          data: results.dataPoints.map(d => d.latency),
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
        },
        {
          label: 'Throughput (ops/s)',
          data: results.dataPoints.map(d => d.throughput),
          borderColor: 'rgb(153, 102, 255)',
          backgroundColor: 'rgba(153, 102, 255, 0.5)',
        },
      ],
    };
  };

  // Simulate benchmarking process
  useEffect(() => {
    let interval;
    if (isRunning) {
      interval = setInterval(() => {
        setBenchmarkResults(prev => {
          const updated = { ...prev };
          const currentBenchmark = updated[activeBenchmark];
          
          // Simulate progress increase
          const newProgress = Math.min(100, currentBenchmark.progress + 5);
          currentBenchmark.progress = newProgress;
          
          // Generate random performance data
          const newDataPoint = {
            latency: Math.floor(Math.random() * 100) + 50, // 50-150ms
            throughput: Math.floor(Math.random() * 500) + 100, // 100-600 ops/s
            memoryUsage: Math.floor(Math.random() * 50) + 100 // 100-150 MB
          };
          
          currentBenchmark.dataPoints = [...currentBenchmark.dataPoints, newDataPoint];
          
          // Update averages
          const allData = currentBenchmark.dataPoints;
          if (allData.length > 0) {
            const avgLatency = allData.reduce((sum, dp) => sum + dp.latency, 0) / allData.length;
            const avgThroughput = allData.reduce((sum, dp) => sum + dp.throughput, 0) / allData.length;
            const avgMemory = allData.reduce((sum, dp) => sum + dp.memoryUsage, 0) / allData.length;
            
            currentBenchmark.avgLatency = Math.round(avgLatency);
            currentBenchmark.throughput = Math.round(avgThroughput);
            currentBenchmark.memoryUsage = Math.round(avgMemory);
          }
          
          return updated;
        });
      }, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRunning, activeBenchmark]);

  const runBenchmark = () => {
    setIsRunning(true);
  };

  const stopBenchmark = () => {
    setIsRunning(false);
  };

  const resetBenchmark = () => {
    setIsRunning(false);
    setBenchmarkResults({
      vectorSearch: {
        avgLatency: 0,
        throughput: 0,
        memoryUsage: 0,
        progress: 0,
        dataPoints: []
      },
      databaseOperations: {
        avgLatency: 0,
        throughput: 0,
        memoryUsage: 0,
        progress: 0,
        dataPoints: []
      },
      indexOperations: {
        avgLatency: 0,
        throughput: 0,
        memoryUsage: 0,
        progress: 0,
        dataPoints: []
      }
    });
  };

  const currentResults = benchmarkResults[activeBenchmark];

  return (
    <Card className="h-full overflow-hidden flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Performance Benchmarks
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Compare performance metrics across different operations
        </p>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-4 overflow-hidden">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button 
              size="sm" 
              variant={isRunning ? "secondary" : "default"}
              onClick={runBenchmark}
              disabled={isRunning}
            >
              <Play className="h-4 w-4 mr-1" />
              Run
            </Button>
            <Button 
              size="sm" 
              variant={isRunning ? "default" : "secondary"}
              onClick={stopBenchmark}
              disabled={!isRunning}
            >
              <Pause className="h-4 w-4 mr-1" />
              Stop
            </Button>
            <Button 
              size="sm" 
              variant="outline"
              onClick={resetBenchmark}
            >
              <RotateCcw className="h-4 w-4 mr-1" />
              Reset
            </Button>
          </div>
          <Badge variant="outline" className="text-xs">
            {isRunning ? "Running" : "Ready"}
          </Badge>
        </div>
        
        <Progress value={currentResults.progress} className="h-2" />
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          <div className="flex flex-col items-center justify-center p-4 bg-muted rounded-lg">
            <TrendingUp className="h-5 w-5 text-primary mb-1" />
            <div className="text-2xl font-bold">{currentResults.avgLatency}ms</div>
            <div className="text-xs text-muted-foreground">Avg Latency</div>
          </div>
          <div className="flex flex-col items-center justify-center p-4 bg-muted rounded-lg">
            <Zap className="h-5 w-5 text-primary mb-1" />
            <div className="text-2xl font-bold">{currentResults.throughput} ops/s</div>
            <div className="text-xs text-muted-foreground">Throughput</div>
          </div>
          <div className="flex flex-col items-center justify-center p-4 bg-muted rounded-lg">
            <Activity className="h-5 w-5 text-primary mb-1" />
            <div className="text-2xl font-bold">{currentResults.memoryUsage} MB</div>
            <div className="text-xs text-muted-foreground">Memory Usage</div>
          </div>
        </div>
        
        <Tabs 
          value={activeBenchmark} 
          onValueChange={setActiveBenchmark}
          className="flex-1 flex flex-col overflow-hidden"
        >
          <TabsList className="grid w-full grid-cols-3 mb-2">
            <TabsTrigger value="vectorSearch">
              <TrendingUp className="h-3 w-3 mr-1" />
              Vector Search
            </TabsTrigger>
            <TabsTrigger value="databaseOperations">
              <Database className="h-3 w-3 mr-1" />
              DB Operations
            </TabsTrigger>
            <TabsTrigger value="indexOperations">
              <BarChart3 className="h-3 w-3 mr-1" />
              Index Ops
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="vectorSearch" className="flex-1 flex flex-col overflow-hidden">
            <div className="text-sm mb-2">
              <p className="font-medium">Vector Search Benchmark</p>
              <p className="text-muted-foreground">Measures query response times and throughput for similarity searches.</p>
            </div>
            <div className="flex-1">
              <Line options={chartOptions} data={getChartData('vectorSearch')} />
            </div>
          </TabsContent>
          
          <TabsContent value="databaseOperations" className="flex-1 flex flex-col overflow-hidden">
            <div className="text-sm mb-2">
              <p className="font-medium">Database Operations Benchmark</p>
              <p className="text-muted-foreground">Measures performance of CRUD operations on vector databases.</p>
            </div>
            <div className="flex-1">
              <Line options={chartOptions} data={getChartData('databaseOperations')} />
            </div>
          </TabsContent>
          
          <TabsContent value="indexOperations" className="flex-1 flex flex-col overflow-hidden">
            <div className="text-sm mb-2">
              <p className="font-medium">Index Operations Benchmark</p>
              <p className="text-muted-foreground">Measures performance of index creation and maintenance.</p>
            </div>
            <div className="flex-1">
              <Line options={chartOptions} data={getChartData('indexOperations')} />
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default BenchmarkingTools;