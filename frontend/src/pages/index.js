import Head from 'next/head';
import { useState } from 'react';

export default function Dashboard() {
  const [databases, setDatabases] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchDatabases = async () => {
    setLoading(true);
    try {
      // This would connect to the JadeVectorDB API
      // const response = await fetch('/api/databases');
      // const data = await response.json();
      // setDatabases(data);
      
      // Mock data for now
      setDatabases([
        { id: 'db1', name: 'Documents', description: 'Vector database for document embeddings', vectors: 12500, indexes: 3 },
        { id: 'db2', name: 'Images', description: 'Vector database for image embeddings', vectors: 8900, indexes: 2 },
        { id: 'db3', name: 'Products', description: 'Vector database for product recommendations', vectors: 42500, indexes: 5 }
      ]);
    } catch (error) {
      console.error('Error fetching databases:', error);
    } finally {
      setLoading(false);
    }
  };

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
            <div className="border-4 border-dashed border-gray-200 rounded-lg h-96 flex items-center justify-center">
              <div className="text-center">
                <h2 className="text-xl font-semibold text-gray-700">Welcome to JadeVectorDB</h2>
                <p className="mt-2 text-gray-500">
                  Manage your vector databases, perform searches, and monitor system performance.
                </p>
              </div>
            </div>
          </div>
          
          <div className="mt-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Databases</h2>
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
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}