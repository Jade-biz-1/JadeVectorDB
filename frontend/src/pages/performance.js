import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { performanceApi } from '../lib/api';

export default function PerformanceDashboard() {
  const [stats, setStats] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    fetchStats();
    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    setLoading(true);
    try {
      const response = await performanceApi.getMetrics();
      const metricsData = response.metrics || [];

      // Transform metrics if needed
      const formattedStats = Array.isArray(metricsData)
        ? metricsData
        : Object.entries(metricsData).map(([label, value]) => ({ label, value }));

      setStats(formattedStats);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
      setStats([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Performance Dashboard - JadeVectorDB</title>
        <meta name="description" content="Monitor system performance" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Performance Dashboard</h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-semibold text-gray-800">System Performance Metrics</h2>
                {lastUpdated && (
                  <p className="text-sm text-gray-500 mt-1">
                    Last updated: {lastUpdated.toLocaleTimeString()}
                  </p>
                )}
              </div>
              <button
                onClick={fetchStats}
                disabled={loading}
                className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 rounded-md disabled:opacity-50"
              >
                {loading ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {loading && stats.length === 0 ? (
                <div className="col-span-3 text-center py-8 text-gray-500">Loading metrics...</div>
              ) : stats.length === 0 ? (
                <div className="col-span-3 text-center py-8 text-gray-500">No metrics found.</div>
              ) : (
                stats.map((stat, index) => (
                  <div key={index} className="bg-gradient-to-br from-indigo-50 to-blue-50 p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600 uppercase tracking-wide">
                          {stat.label || stat.name || `Metric ${index + 1}`}
                        </p>
                        <p className="text-3xl font-bold text-gray-900 mt-2">
                          {typeof stat.value === 'number' ? stat.value.toLocaleString() : stat.value}
                        </p>
                        {stat.unit && (
                          <p className="text-xs text-gray-500 mt-1">{stat.unit}</p>
                        )}
                      </div>
                      {stat.trend && (
                        <div className={`flex items-center ${
                          stat.trend === 'up' ? 'text-green-600' : 'text-red-600'
                        }`}>
                          <span className="text-2xl">
                            {stat.trend === 'up' ? '↑' : '↓'}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Info panel */}
            <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-md">
              <div className="flex items-start">
                <svg className="h-5 w-5 text-blue-400 mt-0.5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                <div>
                  <h3 className="text-sm font-medium text-blue-800">Performance Monitoring</h3>
                  <div className="mt-2 text-sm text-blue-700">
                    <p>Metrics are automatically refreshed every 10 seconds. Use the Refresh button to manually update.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
