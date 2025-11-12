import Head from 'next/head';
import { useState } from 'react';
import { searchApi } from '../lib/api';

export default function QueryInterface() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const handleQuery = async () => {
    setLoading(true);
    try {
      // Example: use a default databaseId (replace with actual selection)
      const databaseId = 'default';
      const searchRequest = { query, topK: 10 };
      const response = await searchApi.similaritySearch(databaseId, searchRequest);
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
            <div className="flex mb-4">
              <input
                type="text"
                className="border border-gray-300 rounded-l px-4 py-2 w-full"
                placeholder="Enter query..."
                value={query}
                onChange={e => setQuery(e.target.value)}
              />
              <button
                className="bg-blue-600 text-white px-4 py-2 rounded-r"
                onClick={handleQuery}
                disabled={loading}
              >
                {loading ? 'Running...' : 'Run'}
              </button>
            </div>
            {result && (
              <div className="mt-4 p-4 bg-blue-50 rounded">
                <div className="font-bold">Result:</div>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
