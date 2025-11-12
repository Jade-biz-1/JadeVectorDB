import Head from 'next/head';
import { useEffect, useState } from 'react';
import { clusterApi, databaseApi, monitoringApi, securityApi } from '../lib/api';

export default function Dashboard() {
  const [nodes, setNodes] = useState([]);
  const [databases, setDatabases] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [recentLogs, setRecentLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoading(true);
      try {
        const [nodesRes, dbRes, statusRes, logsRes] = await Promise.all([
          clusterApi.listNodes(),
          databaseApi.listDatabases(5, 0),
          monitoringApi.systemStatus(),
          securityApi.listAuditLogs(5, 0)
        ]);
        setNodes(nodesRes.nodes || []);
        setDatabases(dbRes.databases || []);
        setSystemStatus(statusRes || null);
        setRecentLogs(logsRes.logs || []);
      } catch (error) {
        console.error('Error loading dashboard:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchDashboardData();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Dashboard - JadeVectorDB</title>
        <meta name="description" content="Unified system overview" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">JadeVectorDB Dashboard</h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Cluster Status */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Cluster Status</h2>
            {loading ? <div>Loading...</div> : (
              <table className="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Node ID</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {nodes.map(node => (
                    <tr key={node.id}>
                      <td className="px-4 py-2">{node.id}</td>
                      <td className="px-4 py-2">{node.role}</td>
                      <td className="px-4 py-2">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${node.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>{node.status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
          {/* Database Overview */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Databases</h2>
            {loading ? <div>Loading...</div> : (
              <ul className="divide-y divide-gray-200">
                {databases.map(db => (
                  <li key={db.databaseId} className="py-2">
                    <span className="font-bold text-indigo-700">{db.name}</span> <span className="text-gray-500">({db.vectorDimension}D, {db.indexType})</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          {/* System Health */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">System Health</h2>
            {loading ? <div>Loading...</div> : systemStatus ? (
              <pre className="bg-gray-100 p-2 rounded text-xs overflow-x-auto">{JSON.stringify(systemStatus, null, 2)}</pre>
            ) : <div>No status available.</div>}
          </div>
          {/* Recent Activity */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Recent Activity</h2>
            {loading ? <div>Loading...</div> : (
              <ul className="divide-y divide-gray-200">
                {recentLogs.map(log => (
                  <li key={log.id} className="py-2">
                    <span className="font-bold">{log.user}</span>: {log.event} <span className="text-gray-500">({log.timestamp})</span>
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
