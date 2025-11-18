import Head from 'next/head';
import { useState, useEffect } from 'react';
import { vectorApi, databaseApi } from '../lib/api';

export default function DataExploration() {
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [vectors, setVectors] = useState([]);
  const [loading, setLoading] = useState(false);
  const [limit, setLimit] = useState(20);

  useEffect(() => {
    fetchDatabases();
  }, []);

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      const dbList = response.databases || [];
      setDatabases(dbList);

      // Auto-select first database if available
      if (dbList.length > 0) {
        const firstDbId = dbList[0].databaseId || dbList[0].id;
        setSelectedDatabase(firstDbId);
      }
    } catch (error) {
      console.error('Error fetching databases:', error);
      alert('Error fetching databases: ' + error.message);
    }
  };

  useEffect(() => {
    if (selectedDatabase) {
      fetchVectors();
    }
  }, [selectedDatabase, limit]);

  const fetchVectors = async () => {
    if (!selectedDatabase) return;

    setLoading(true);
    try {
      const response = await vectorApi.listVectors(selectedDatabase, limit, 0);
      const vectorsData = response.vectors || [];

      // Transform vectors for display
      const formattedVectors = vectorsData.map(v => ({
        id: v.vectorId || v.id,
        values: Array.isArray(v.values) ? v.values : [],
        metadata: v.metadata || {},
        label: v.metadata?.label || v.metadata?.name || 'N/A'
      }));

      setVectors(formattedVectors);
    } catch (error) {
      console.error('Error fetching vectors:', error);
      setVectors([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Data Exploration - JadeVectorDB</title>
        <meta name="description" content="Explore vector data and projections" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Data Exploration</h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="bg-white p-6 rounded-lg shadow mb-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {/* Database Selector */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Database
                </label>
                <select
                  value={selectedDatabase}
                  onChange={e => setSelectedDatabase(e.target.value)}
                  className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="">-- Select a Database --</option>
                  {databases.map(db => (
                    <option key={db.databaseId || db.id} value={db.databaseId || db.id}>
                      {db.name} (Dimension: {db.vectorDimension})
                    </option>
                  ))}
                </select>
              </div>

              {/* Limit Selector */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Vectors
                </label>
                <select
                  value={limit}
                  onChange={e => setLimit(parseInt(e.target.value))}
                  className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                </select>
              </div>
            </div>

            <button
              onClick={fetchVectors}
              disabled={!selectedDatabase || loading}
              className="mb-6 bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Loading...' : 'Refresh Vectors'}
            </button>
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Vector Data ({vectors.length} vectors)</h2>
            <table className="min-w-full divide-y divide-gray-200 mb-8">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Vector ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Values</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Label</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {loading ? (
                  <tr><td colSpan={3} className="text-center py-4">Loading...</td></tr>
                ) : vectors.length === 0 ? (
                  <tr><td colSpan={3} className="text-center py-4">No vectors found.</td></tr>
                ) : (
                  vectors.map(vec => (
                    <tr key={vec.id}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{vec.id}</td>
                      <td className="px-6 py-4 text-sm text-gray-500">
                        <div className="max-w-md truncate">
                          [{vec.values.slice(0, 5).join(', ')}{vec.values.length > 5 ? ', ...' : ''}]
                        </div>
                        <div className="text-xs text-gray-400">Dimension: {vec.values.length}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{vec.label}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>

            <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-md">
              <div className="flex items-start">
                <svg className="h-5 w-5 text-yellow-400 mt-0.5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                <div>
                  <h3 className="text-sm font-medium text-yellow-800">Feature Note: Vector Projections</h3>
                  <div className="mt-2 text-sm text-yellow-700">
                    <p>2D/3D vector projection visualization (t-SNE/UMAP) is not yet implemented in the backend. This feature would allow you to visualize high-dimensional vectors in a 2D or 3D space for easier exploration and pattern recognition.</p>
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
