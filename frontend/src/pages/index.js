import Head from 'next/head';
import { useState, useEffect } from 'react';
import { databaseApi } from '../lib/api';

export default function Dashboard() {
  const [databases, setDatabases] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchDatabases = async () => {
    setLoading(true);
    try {
      const response = await databaseApi.listDatabases();
      setDatabases(response.databases.map(db => ({
        id: db.databaseId,
        name: db.name,
        description: db.description,
        vectors: db.stats?.vectorCount || 0,
        indexes: db.stats?.indexCount || 0
      })));
    } catch (error) {
      console.error('Error fetching databases:', error);
      alert(`Error fetching databases: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatabases();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>JadeVectorDB Dashboard</title>
        <meta name="description" content="JadeVectorDB Management Dashboard" />
      </Head>

      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-900">JadeVectorDB Dashboard</h1>
          <button
            onClick={fetchDatabases}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Refresh Databases'}
          </button>
        </div>
      </header>

      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="px-4 py-6 sm:px-0">
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">System Overview</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">{databases.length}</div>
                  <div className="text-sm text-gray-500">Total Databases</div>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {databases.reduce((sum, db) => sum + db.vectors, 0).toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-500">Total Vectors</div>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {databases.reduce((sum, db) => sum + db.indexes, 0)}
                  </div>
                  <div className="text-sm text-gray-500">Total Indexes</div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Databases</h2>
            {databases.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {databases.map((db) => (
                  <div key={db.id} className="bg-white overflow-hidden shadow rounded-lg">
                    <div className="px-4 py-5 sm:p-6">
                      <h3 className="text-lg font-medium text-gray-900">{db.name}</h3>
                      <p className="mt-1 text-sm text-gray-500">{db.description}</p>
                      <div className="mt-4 grid grid-cols-2 gap-2">
                        <div className="text-sm">
                          <span className="font-medium">Vectors:</span> {db.vectors.toLocaleString()}
                        </div>
                        <div className="text-sm">
                          <span className="font-medium">Indexes:</span> {db.indexes}
                        </div>
                      </div>
                      <div className="mt-4">
                        <a 
                          href={`/databases?id=${db.id}`}
                          className="text-blue-600 hover:text-blue-900 text-sm font-medium"
                        >
                          View details →
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-gray-500">No databases found. Create one to get started.</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}