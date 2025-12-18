import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { searchApi, databaseApi } from '../lib/api';

export default function SimilaritySearchPage() {
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [queryVector, setQueryVector] = useState('');
  const [topK, setTopK] = useState(10);
  const [threshold, setThreshold] = useState(0.0);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTime, setSearchTime] = useState(null);

  useEffect(() => {
    fetchDatabases();
  }, []);

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      const dbList = response.databases || [];
      setDatabases(dbList);

      // Auto-select first database
      if (dbList.length > 0 && !selectedDatabase) {
        setSelectedDatabase(dbList[0].databaseId || dbList[0].id);
      }
    } catch (error) {
      console.error('Error fetching databases:', error);
      alert('Error fetching databases: ' + error.message);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!selectedDatabase) return alert('Select a database first');

    setLoading(true);
    const startTime = performance.now();

    try {
      let parsedVector = [];
      if (queryVector.trim().startsWith('[')) {
        parsedVector = JSON.parse(queryVector);
      } else {
        parsedVector = queryVector.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
      }

      if (parsedVector.length === 0) {
        alert('Please enter a valid vector');
        setLoading(false);
        return;
      }

      const searchRequest = {
        queryVector: parsedVector,
        topK: parseInt(topK),
        threshold: parseFloat(threshold)
      };

      const response = await searchApi.similaritySearch(selectedDatabase, searchRequest);
      setResults(response.results || []);

      const endTime = performance.now();
      setSearchTime(((endTime - startTime) / 1000).toFixed(3));
    } catch (error) {
      console.error('Error performing search:', error);
      alert('Error performing search: ' + error.message);
      setResults([]);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Similarity Search - JadeVectorDB</title>
        <meta name="description" content="Perform similarity search in JadeVectorDB" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Similarity Search</h1>
        </div>
      </header>
      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          {/* Database Selection */}
          <div className="mb-6">
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

          {/* Search Form */}
          <form onSubmit={handleSearch}>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Query Vector
              </label>
              <textarea
                placeholder="Enter comma-separated values or JSON array, e.g.: [0.1, 0.2, 0.3] or 0.1, 0.2, 0.3"
                value={queryVector}
                onChange={e => setQueryVector(e.target.value)}
                className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                rows="3"
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Top K Results
                </label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={topK}
                  onChange={e => setTopK(e.target.value)}
                  className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Similarity Threshold (0.0 - 1.0)
                </label>
                <input
                  type="number"
                  step={0.01}
                  min={0}
                  max={1}
                  value={threshold}
                  onChange={e => setThreshold(e.target.value)}
                  className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !selectedDatabase}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-6 py-3 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </form>
        </div>

        {/* Results Section */}
        {(results.length > 0 || searchTime) && (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-800">
                Search Results ({results.length})
              </h2>
              {searchTime && (
                <span className="text-sm text-gray-500">
                  Search completed in {searchTime}s
                </span>
              )}
            </div>

            {results.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No results found matching your criteria.</p>
            ) : (
              <div className="space-y-4">
                {results.map((result, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 rounded-lg p-4 hover:border-indigo-300 hover:shadow-sm transition-all"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800">
                            Rank #{idx + 1}
                          </span>
                          {result.vectorId && (
                            <span className="text-sm text-gray-500">
                              ID: {result.vectorId || result.id}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold text-gray-900">
                          {typeof result.score === 'number'
                            ? result.score.toFixed(4)
                            : result.score}
                        </div>
                        <div className="text-xs text-gray-500">Similarity Score</div>
                      </div>
                    </div>

                    {result.values && (
                      <div className="mb-2">
                        <div className="text-xs font-medium text-gray-500 uppercase mb-1">
                          Vector Values
                        </div>
                        <div className="text-sm text-gray-700 bg-gray-50 px-3 py-2 rounded font-mono overflow-x-auto">
                          [{result.values.slice(0, 10).map(v =>
                            typeof v === 'number' ? v.toFixed(3) : v
                          ).join(', ')}
                          {result.values.length > 10 && ', ...'}]
                        </div>
                      </div>
                    )}

                    {result.metadata && Object.keys(result.metadata).length > 0 && (
                      <div>
                        <div className="text-xs font-medium text-gray-500 uppercase mb-1">
                          Metadata
                        </div>
                        <div className="text-sm text-gray-700 bg-gray-50 px-3 py-2 rounded">
                          {JSON.stringify(result.metadata, null, 2)}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </Layout>
  );
}
