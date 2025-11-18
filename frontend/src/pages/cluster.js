import Head from 'next/head';
import { useState, useEffect } from 'react';
import { clusterApi } from '../lib/api';

export default function ClusterManagement() {
  const [nodes, setNodes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeDetails, setNodeDetails] = useState(null);

  useEffect(() => {
    fetchNodes();
    // Auto-refresh every 15 seconds
    const interval = setInterval(fetchNodes, 15000);
    return () => clearInterval(interval);
  }, []);

  const fetchNodes = async () => {
    setLoading(true);
    try {
      const response = await clusterApi.listNodes();
      setNodes(response.nodes || []);
    } catch (error) {
      console.error('Error fetching cluster nodes:', error);
      setNodes([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchNodeDetails = async (nodeId) => {
    try {
      const response = await clusterApi.getNodeStatus(nodeId);
      setNodeDetails(response);
      setSelectedNode(nodeId);
    } catch (error) {
      console.error('Error fetching node details:', error);
      alert(`Error fetching node details: ${error.message}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Cluster Management - JadeVectorDB</title>
        <meta name="description" content="Cluster management dashboard" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Cluster Management</h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="bg-white p-6 rounded-lg shadow mb-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-800">Node Status ({nodes.length} nodes)</h2>
              <button
                onClick={fetchNodes}
                disabled={loading}
                className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 rounded-md disabled:opacity-50"
              >
                {loading ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Node ID</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Role</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">CPU (cores)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Memory (GB)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Storage (GB)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Network (Mbps)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {loading ? (
                    <tr><td colSpan={8} className="text-center py-4 text-gray-500">Loading nodes...</td></tr>
                  ) : nodes.length === 0 ? (
                    <tr><td colSpan={8} className="text-center py-4 text-gray-500">No nodes found.</td></tr>
                  ) : (
                    nodes.map(node => (
                      <tr key={node.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{node.id}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                            {node.role || 'worker'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            node.status === 'active'
                              ? 'bg-green-100 text-green-800'
                              : node.status === 'inactive'
                                ? 'bg-gray-100 text-gray-800'
                                : 'bg-red-100 text-red-800'
                          }`}>
                            {node.status || 'unknown'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{node.cpu || 'N/A'}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{node.memory || 'N/A'}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{node.storage || 'N/A'}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{node.network || 'N/A'}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button
                            onClick={() => fetchNodeDetails(node.id)}
                            className="text-indigo-600 hover:text-indigo-900"
                          >
                            View Details
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Node Details Panel */}
          {selectedNode && nodeDetails && (
            <div className="bg-white p-6 rounded-lg shadow">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-800">Node Details: {selectedNode}</h3>
                <button
                  onClick={() => {
                    setSelectedNode(null);
                    setNodeDetails(null);
                  }}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕ Close
                </button>
              </div>
              <pre className="bg-gray-50 p-4 rounded-md text-sm overflow-x-auto">
                {JSON.stringify(nodeDetails, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
