import Head from 'next/head';
import { useState, useEffect } from 'react';
import { monitoringApi, databaseApi } from '../lib/api';

export default function MonitoringDashboard() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  // Mock data for demonstration
  const [metrics, setMetrics] = useState({
    totalDatabases: 3,
    totalVectors: 63900,
    qps: 1250,
    avgQueryTime: 2.5,
    storageUsed: 85.4,
    uptime: '7 days, 3 hours, 15 minutes'
  });

  const fetchSystemStatus = async () => {
    setLoading(true);
    try {
      // In real implementation, this would call the API to get system status
      // const response = await fetch('/api/status');
      // const data = await response.json();
      // setSystemStatus(data);
      
      // Mock status data
      setSystemStatus({
        status: 'operational',
        checks: {
          database: 'ok',
          storage: 'ok',
          network: 'ok',
          memory: 'ok',
          cpu: 'ok'
        },
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Error fetching system status:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    // Refresh status every 30 seconds
    const interval = setInterval(fetchSystemStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Monitoring Dashboard - JadeVectorDB</title>
        <meta name="description" content="Monitor JadeVectorDB system performance" />
      </Head>

      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">System Monitoring</h1>
        </div>
      </header>

      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {/* System Status Card */}
          <div className="bg-white shadow rounded-lg p-6 mb-8">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-gray-900">System Status</h2>
              <button
                onClick={fetchSystemStatus}
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
              >
                {loading ? 'Refreshing...' : 'Refresh Status'}
              </button>
            </div>
            
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-green-600">Operational</div>
                <div className="text-sm text-gray-500">Overall Status</div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{metrics.totalDatabases}</div>
                <div className="text-sm text-gray-500">Databases</div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{metrics.totalVectors.toLocaleString()}</div>
                <div className="text-sm text-gray-500">Total Vectors</div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{metrics.uptime}</div>
                <div className="text-sm text-gray-500">Uptime</div>
              </div>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            <div className="bg-white shadow rounded-lg p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Performance Metrics</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium text-gray-900">Queries Per Second</span>
                    <span className="text-sm font-bold text-indigo-600">{metrics.qps}</span>
                  </div>
                  <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-green-600 h-2 rounded-full" style={{ width: `${Math.min(100, metrics.qps / 20)}%` }}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium text-gray-900">Avg Query Time (ms)</span>
                    <span className="text-sm font-bold text-indigo-600">{metrics.avgQueryTime}</span>
                  </div>
                  <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${Math.min(100, 100 - (metrics.avgQueryTime / 10) * 100)}%` }}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium text-gray-900">Storage Utilization (%)</span>
                    <span className="text-sm font-bold text-indigo-600">{metrics.storageUsed}%</span>
                  </div>
                  <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-yellow-600 h-2 rounded-full" style={{ width: `${metrics.storageUsed}%` }}></div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Health Checks */}
            <div className="bg-white shadow rounded-lg p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Health Checks</h3>
              <div className="space-y-3">
                {systemStatus && Object.entries(systemStatus.checks).map(([service, status]) => (
                  <div key={service} className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700 capitalize">{service}</span>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      status === 'ok' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Recent Activity */}
            <div className="bg-white shadow rounded-lg p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h3>
              <div className="space-y-4">
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    <div className="w-3 h-3 rounded-full bg-green-500 mt-1"></div>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900">Database created</p>
                    <p className="text-sm text-gray-500">New database 'Products' created</p>
                    <p className="text-xs text-gray-400 mt-1">2 hours ago</p>
                  </div>
                </div>
                
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    <div className="w-3 h-3 rounded-full bg-blue-500 mt-1"></div>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900">Index built</p>
                    <p className="text-sm text-gray-500">HNSW index built for 'Documents' DB</p>
                    <p className="text-xs text-gray-400 mt-1">5 hours ago</p>
                  </div>
                </div>
                
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    <div className="w-3 h-3 rounded-full bg-yellow-500 mt-1"></div>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900">Performance alert</p>
                    <p className="text-sm text-gray-500">Query response time increased</p>
                    <p className="text-xs text-gray-400 mt-1">1 day ago</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Database Status Table */}
          <div className="bg-white shadow overflow-hidden sm:rounded-md">
            <div className="px-4 py-5 border-b border-gray-200 sm:px-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900">Database Status</h3>
              <p className="mt-1 text-sm text-gray-500">
                Status of all vector databases in the system
              </p>
            </div>
            <ul className="divide-y divide-gray-200">
              {[
                { id: 'db1', name: 'Documents', status: 'online', vectors: 12500, indexes: 3, storage: '12.4 GB' },
                { id: 'db2', name: 'Images', status: 'online', vectors: 8900, indexes: 2, storage: '8.7 GB' },
                { id: 'db3', name: 'Products', status: 'warning', vectors: 42500, indexes: 5, storage: '45.2 GB' }
              ].map((db) => (
                <li key={db.id}>
                  <div className="px-4 py-4 sm:px-6">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-medium text-indigo-600 truncate">{db.name}</div>
                      <div className="ml-2 flex-shrink-0 flex">
                        <span className={`inline-flex px-2 text-xs leading-5 font-semibold rounded-full ${
                          db.status === 'online' 
                            ? 'bg-green-100 text-green-800' 
                            : db.status === 'warning'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-red-100 text-red-800'
                        }`}>
                          {db.status}
                        </span>
                      </div>
                    </div>
                    <div className="mt-2 sm:flex sm:justify-between">
                      <div className="sm:flex">
                        <div className="mr-6 text-sm text-gray-500">
                          Vectors: {db.vectors.toLocaleString()}
                        </div>
                        <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                          Indexes: {db.indexes}
                        </div>
                      </div>
                      <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                        Storage: {db.storage}
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}