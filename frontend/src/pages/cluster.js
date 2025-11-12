import Head from 'next/head';
import { useState, useEffect } from 'react';
import { clusterApi } from '../lib/api';

export default function ClusterManagement() {
  const [nodes, setNodes] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchNodes = async () => {
      setLoading(true);
      try {
        const response = await clusterApi.listNodes();
        setNodes(response.nodes || []);
      } catch (error) {
        console.error('Error fetching cluster nodes:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchNodes();
  }, []);

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
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Node Status</h2>
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Node ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">CPU (cores)</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Memory (GB)</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Storage (GB)</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Network (Mbps)</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {loading ? (
                  <tr><td colSpan={7} className="text-center py-4">Loading...</td></tr>
                ) : nodes.length === 0 ? (
                  <tr><td colSpan={7} className="text-center py-4">No nodes found.</td></tr>
                ) : (
                  nodes.map(node => (
                    <tr key={node.id}>
                      <td className="px-6 py-4 whitespace-nowrap">{node.id}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{node.role}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${node.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>{node.status}</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">{node.cpu}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{node.memory}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{node.storage}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{node.network}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
