import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { searchApi, databaseApi } from '../lib/api';

export default function QueryInterface() {
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(10);

  useEffect(() => {
    fetchDatabases();
  }, []);

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      const dbList = response.databases || [];
      setDatabases(dbList);

      // Auto-select first database if available
      if (dbList.length > 0 && !selectedDatabase) {
        setSelectedDatabase(dbList[0].databaseId || dbList[0].id);
      }
    } catch (error) {
      console.error('Error fetching databases:', error);
      alert('Error fetching databases: ' + error.message);
    }
  };

  const handleQuery = async () => {
    if (!selectedDatabase) {
      alert('Please select a database first');
      return;
    }

    if (!query.trim()) {
      alert('Please enter a query');
      return;
    }

    setLoading(true);
    try {
      const searchRequest = { query, topK };
      const response = await searchApi.similaritySearch(selectedDatabase, searchRequest);
      setResult(response);
    } catch (error) {
      setResult({ status: 'error', message: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Query Interface - JadeVectorDB</title>
        <meta name="description" content="Run vector database queries" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Query Interface</h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Run Query</h2>

            {/* Database Selector */}
            <div className="mb-4">
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

            {/* Query Input and Top-K */}
            <div className="grid grid-cols-12 gap-4 mb-4">
              <div className="col-span-10">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Query
                </label>
                <input
                  type="text"
                  className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter your search query..."
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyPress={e => e.key === 'Enter' && handleQuery()}
                />
              </div>
              <div className="col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Top-K
                </label>
                <input
                  type="number"
                  className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  value={topK}
                  onChange={e => setTopK(parseInt(e.target.value) || 10)}
                  min="1"
                  max="100"
                />
              </div>
            </div>

            <div className="mb-4">
              <button
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-6 py-3 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={handleQuery}
                disabled={loading || !selectedDatabase}
              >
                {loading ? 'Running Query...' : 'Run Query'}
              </button>
            </div>
            {/* Results */}
            {result && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Results</h3>
                {result.status === 'error' ? (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-md">
                    <div className="font-semibold text-red-800">Error:</div>
                    <div className="text-red-700">{result.message}</div>
                  </div>
                ) : (
                  <div className="bg-gray-50 p-4 rounded-md">
                    <div className="mb-2 text-sm text-gray-600">
                      Found {result.results?.length || 0} results
                    </div>
                    <pre className="text-sm overflow-x-auto">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </Layout>
  );
}
