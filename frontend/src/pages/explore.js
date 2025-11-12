import Head from 'next/head';
import { useState, useEffect } from 'react';
import { vectorApi } from '../lib/api';

export default function DataExploration() {
  const [vectors, setVectors] = useState([]);
  const [projection, setProjection] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchVectors = async () => {
      setLoading(true);
      try {
        // Example: fetch vectors for a default database (replace with actual databaseId)
        const databaseId = 'default';
        // This would be a backend endpoint to list vectors, not just get one
        // For demo, fetch a single vector (should be replaced with batch/list API)
        // const response = await vectorApi.getVector(databaseId, 'vec1');
        // setVectors([response]);
        // For now, keep as empty or mock until batch/list endpoint is available
        setVectors([]);
        // For projection, you would call a backend endpoint for t-SNE/UMAP
        setProjection([]);
      } catch (error) {
        console.error('Error fetching vectors:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchVectors();
  }, []);

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
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Vector Data</h2>
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
                      <td className="px-6 py-4 whitespace-nowrap">{vec.id}</td>
                      <td className="px-6 py-4 whitespace-nowrap">[{vec.values.join(', ')}]</td>
                      <td className="px-6 py-4 whitespace-nowrap">{vec.label}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">2D Projection (t-SNE/UMAP)</h2>
            <div className="grid grid-cols-3 gap-4">
              {loading ? (
                <div className="col-span-3 text-center py-4">Loading...</div>
              ) : projection.length === 0 ? (
                <div className="col-span-3 text-center py-4">No projection data.</div>
              ) : (
                projection.map(point => (
                  <div key={point.id} className="bg-blue-50 p-4 rounded-lg text-center">
                    <div className="text-lg font-bold text-blue-600">{point.label}</div>
                    <div className="text-sm text-gray-500">x: {point.x}, y: {point.y}</div>
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
