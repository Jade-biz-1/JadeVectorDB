import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { performanceApi } from '../lib/api';

export default function PerformanceDashboard() {
  const [stats, setStats] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchStats = async () => {
      setLoading(true);
      try {
        const response = await performanceApi.getMetrics();
        setStats(response.metrics || []);
      } catch (error) {
        console.error('Error fetching performance metrics:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

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
            <h2 className="text-xl font-semibold text-gray-800 mb-4">System Stats</h2>
            <div className="grid grid-cols-2 gap-6">
              {loading ? (
                <div className="col-span-2 text-center py-4">Loading...</div>
              ) : stats.length === 0 ? (
                <div className="col-span-2 text-center py-4">No metrics found.</div>
              ) : (
                stats.map(stat => (
                  <div key={stat.label} className="bg-green-50 p-4 rounded-lg text-center">
                    <div className="text-lg font-bold text-green-700">{stat.label}</div>
                    <div className="text-2xl text-gray-900">{stat.value}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
