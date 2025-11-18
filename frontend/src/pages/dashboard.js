import Head from 'next/head';
import { useEffect, useState } from 'react';
import { clusterApi, databaseApi, monitoringApi, securityApi } from '../lib/api';

export default function Dashboard() {
  const [nodes, setNodes] = useState([]);
  const [databases, setDatabases] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [recentLogs, setRecentLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    fetchDashboardData();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const [nodesRes, dbRes, statusRes, logsRes] = await Promise.all([
        clusterApi.listNodes().catch(err => ({ nodes: [] })),
        databaseApi.listDatabases(5, 0).catch(err => ({ databases: [] })),
        monitoringApi.systemStatus().catch(err => null),
        securityApi.listAuditLogs(5, 0).catch(err => ({ logs: [] }))
      ]);

      setNodes(nodesRes.nodes || []);
      setDatabases(dbRes.databases || []);
      setSystemStatus(statusRes || null);
      setRecentLogs(logsRes.logs || []);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Dashboard - JadeVectorDB</title>
        <meta name="description" content="Unified system overview" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">JadeVectorDB Dashboard</h1>
              {lastUpdated && (
                <p className="text-sm text-gray-500 mt-1">
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </p>
              )}
            </div>
            <button
              onClick={fetchDashboardData}
              disabled={loading}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 rounded-md disabled:opacity-50"
            >
              {loading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Cluster Status */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Cluster Status ({nodes.length} nodes)
            </h2>
            {loading && nodes.length === 0 ? (
              <div className="text-center py-4 text-gray-500">Loading...</div>
            ) : nodes.length === 0 ? (
              <div className="text-center py-4 text-gray-500">No cluster nodes found</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Node ID</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {nodes.map(node => (
                      <tr key={node.id}>
                        <td className="px-4 py-2 text-sm text-gray-900">{node.id}</td>
                        <td className="px-4 py-2 text-sm text-gray-500">{node.role || 'worker'}</td>
                        <td className="px-4 py-2">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            node.status === 'active'
                              ? 'bg-green-100 text-green-800'
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {node.status || 'unknown'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Database Overview */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Databases ({databases.length})
            </h2>
            {loading && databases.length === 0 ? (
              <div className="text-center py-4 text-gray-500">Loading...</div>
            ) : databases.length === 0 ? (
              <div className="text-center py-4 text-gray-500">No databases found</div>
            ) : (
              <ul className="divide-y divide-gray-200">
                {databases.map(db => (
                  <li key={db.databaseId || db.id} className="py-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-semibold text-indigo-700">{db.name}</p>
                        <p className="text-sm text-gray-500">
                          {db.vectorDimension}D, {db.indexType || 'FLAT'}
                        </p>
                      </div>
                      {db.stats && (
                        <div className="text-right text-sm text-gray-500">
                          <p>{db.stats.vectorCount || 0} vectors</p>
                        </div>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* System Health */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">System Health</h2>
            {loading && !systemStatus ? (
              <div className="text-center py-4 text-gray-500">Loading...</div>
            ) : systemStatus ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Overall Status</span>
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    systemStatus.status === 'operational'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {systemStatus.status || 'Unknown'}
                  </span>
                </div>
                {systemStatus.checks && Object.entries(systemStatus.checks).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 capitalize">{key}</span>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      value === 'ok'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {value}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-4 text-gray-500">No status available</div>
            )}
          </div>

          {/* Recent Activity */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Recent Activity</h2>
            {loading && recentLogs.length === 0 ? (
              <div className="text-center py-4 text-gray-500">Loading...</div>
            ) : recentLogs.length === 0 ? (
              <div className="text-center py-4 text-gray-500">No recent activity</div>
            ) : (
              <ul className="divide-y divide-gray-200">
                {recentLogs.map((log, idx) => (
                  <li key={log.id || idx} className="py-3">
                    <div className="flex items-start">
                      <div className="flex-shrink-0">
                        <div className="w-2 h-2 rounded-full bg-blue-500 mt-1.5"></div>
                      </div>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-gray-900">
                          {log.user || 'System'}: {log.event || log.action}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          {log.timestamp ? new Date(log.timestamp).toLocaleString() : 'Recently'}
                        </p>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
