import React, { useState, useEffect } from 'react';
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
    <div className="border border-gray-200 rounded-lg p-4 h-full overflow-hidden flex flex-col">
      <div className="pb-3 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
          <span className="text-blue-600">📊</span>
          Performance Benchmarks
        </h3>
        <p className="text-sm text-gray-600 mt-1">
          Compare performance metrics across different operations
        </p>
      </div>
      
      <div className="flex-1 flex flex-col gap-4 pt-4 overflow-hidden">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button 
              className={`px-3 py-1.5 text-sm rounded-md ${
                isRunning 
                  ? 'bg-gray-100 text-gray-500 cursor-not-allowed' 
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
              onClick={runBenchmark}
              disabled={isRunning}
            >
              <span className="flex items-center gap-1">
                <span>▶</span> Run
              </span>
            </button>
            <button 
              className={`px-3 py-1.5 text-sm rounded-md ${
                !isRunning 
                  ? 'bg-gray-100 text-gray-500 cursor-not-allowed' 
                  : 'bg-red-600 text-white hover:bg-red-700'
              }`}
              onClick={stopBenchmark}
              disabled={!isRunning}
            >
              <span className="flex items-center gap-1">
                <span>⏸</span> Stop
              </span>
            </button>
            <button 
              className="px-3 py-1.5 text-sm rounded-md border border-gray-300 text-gray-700 hover:bg-gray-50"
              onClick={resetBenchmark}
            >
              <span className="flex items-center gap-1">
                <span>↺</span> Reset
              </span>
            </button>
          </div>
          <span className={`px-2 py-1 text-xs rounded-full ${
            isRunning ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
          }`}>
            {isRunning ? "Running" : "Ready"}
          </span>
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full" 
            style={{ width: `${currentResults.progress}%` }}
          ></div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          <div className="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-lg">
            <div className="text-blue-600 text-lg">📈</div>
            <div className="text-2xl font-bold">{currentResults.avgLatency}ms</div>
            <div className="text-xs text-gray-600">Avg Latency</div>
          </div>
          <div className="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-lg">
            <div className="text-yellow-600 text-lg">⚡</div>
            <div className="text-2xl font-bold">{currentResults.throughput} ops/s</div>
            <div className="text-xs text-gray-600">Throughput</div>
          </div>
          <div className="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-lg">
            <div className="text-purple-600 text-lg">💾</div>
            <div className="text-2xl font-bold">{currentResults.memoryUsage} MB</div>
            <div className="text-xs text-gray-600">Memory Usage</div>
          </div>
        </div>
        
        <div className="border-b border-gray-200">
          <div className="flex">
            <button
              className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
                activeBenchmark === 'vectorSearch' 
                  ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                  : 'text-gray-600 hover:text-gray-800'
              }`}
              onClick={() => setActiveBenchmark('vectorSearch')}
            >
              <span className="flex items-center gap-1">
                <span>📈</span> Vector Search
              </span>
            </button>
            <button
              className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
                activeBenchmark === 'databaseOperations' 
                  ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                  : 'text-gray-600 hover:text-gray-800'
              }`}
              onClick={() => setActiveBenchmark('databaseOperations')}
            >
              <span className="flex items-center gap-1">
                <span>🗄️</span> DB Operations
              </span>
            </button>
            <button
              className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
                activeBenchmark === 'indexOperations' 
                  ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                  : 'text-gray-600 hover:text-gray-800'
              }`}
              onClick={() => setActiveBenchmark('indexOperations')}
            >
              <span className="flex items-center gap-1">
                <span>📊</span> Index Ops
              </span>
            </button>
          </div>
        </div>
        
        <div className="flex-1 overflow-y-auto">
          {activeBenchmark === 'vectorSearch' && (
            <div>
              <div className="text-sm mb-2">
                <p className="font-medium">Vector Search Benchmark</p>
                <p className="text-gray-600">Measures query response times and throughput for similarity searches.</p>
              </div>
              <div className="h-64">
                <Line options={chartOptions} data={getChartData('vectorSearch')} />
              </div>
            </div>
          )}
          
          {activeBenchmark === 'databaseOperations' && (
            <div>
              <div className="text-sm mb-2">
                <p className="font-medium">Database Operations Benchmark</p>
                <p className="text-gray-600">Measures performance of CRUD operations on vector databases.</p>
              </div>
              <div className="h-64">
                <Line options={chartOptions} data={getChartData('databaseOperations')} />
              </div>
            </div>
          )}
          
          {activeBenchmark === 'indexOperations' && (
            <div>
              <div className="text-sm mb-2">
                <p className="font-medium">Index Operations Benchmark</p>
                <p className="text-gray-600">Measures performance of index creation and maintenance.</p>
              </div>
              <div className="h-64">
                <Line options={chartOptions} data={getChartData('indexOperations')} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BenchmarkingTools;