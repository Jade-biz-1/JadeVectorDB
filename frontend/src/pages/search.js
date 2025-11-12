import Head from 'next/head';
import { useState, useEffect } from 'react';
import { searchApi, databaseApi } from '../lib/api';

export default function SearchInterface() {
  const [queryVector, setQueryVector] = useState('');
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [topK, setTopK] = useState(10);
  const [threshold, setThreshold] = useState(0.0);
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [includeValues, setIncludeValues] = useState(false);
  const [results, setResults] = useState([]);
  const [databases, setDatabases] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetchingDatabases, setFetchingDatabases] = useState(false);
  
  // Extract database ID from query parameters if available
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const dbId = urlParams.get('databaseId');
    if (dbId) {
      setSelectedDatabase(dbId);
    }
  }, []);

  useEffect(() => {
    fetchDatabases();
  }, []);

  const fetchDatabases = async () => {
    setFetchingDatabases(true);
    try {
      const response = await databaseApi.listDatabases();
      const dbList = response.databases || [];
      setDatabases(dbList.map(db => ({
        id: db.databaseId,
        name: db.name
      })));
    } catch (error) {
      console.error('Error fetching databases:', error);
      alert(`Error fetching databases: ${error.message}`);
    } finally {
      setFetchingDatabases(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      // Parse query vector - handle both JSON array and comma-separated string
      let parsedVector = [];
      
      // If it looks like a JSON array, parse it as JSON
      if (queryVector.trim().startsWith('[')) {
        parsedVector = JSON.parse(queryVector);
      } else {
        // Otherwise, split by comma and convert to numbers
        parsedVector = queryVector.split(',')
          .map(s => parseFloat(s.trim()))
          .filter(v => !isNaN(v));
      }
      
      const searchRequest = {
        queryVector: parsedVector,
        topK: parseInt(topK),
        threshold: parseFloat(threshold),
        includeMetadata: includeMetadata,
        includeVectorData: includeValues,
        includeValues: includeValues
      };
      
  const response = await searchApi.similaritySearch(selectedDatabase, searchRequest);
  setResults(response.results || []);
    } catch (error) {
      console.error('Error performing search:', error);
      alert(`Error performing search: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Vector Search - JadeVectorDB</title>
        <meta name="description" content="Perform similarity searches in JadeVectorDB" />
      </Head>

      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Vector Search</h1>
        </div>
      </header>

      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {/* Search Form */}
          <div className="bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6 mb-8">
            <div className="md:grid md:grid-cols-3 md:gap-6">
              <div className="md:col-span-1">
                <h3 className="text-lg font-medium leading-6 text-gray-900">Similarity Search</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Find vectors similar to your query vector.
                </p>
              </div>
              <div className="mt-5 md:mt-0 md:col-span-2">
                <form onSubmit={handleSearch}>
                  <div className="grid grid-cols-6 gap-6">
                    <div className="col-span-6">
                      <label htmlFor="database" className="block text-sm font-medium text-gray-700">
                        Database
                      </label>
                      <select
                        id="database"
                        name="database"
                        value={selectedDatabase}
                        onChange={(e) => setSelectedDatabase(e.target.value)}
                        required
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      >
                        <option value="">Select a database</option>
                        {databases.map((db) => (
                          <option key={db.id} value={db.id}>{db.name}</option>
                        ))}
                      </select>
                    </div>

                    <div className="col-span-6">
                      <label htmlFor="queryVector" className="block text-sm font-medium text-gray-700">
                        Query Vector
                      </label>
                      <textarea
                        id="queryVector"
                        name="queryVector"
                        rows={4}
                        value={queryVector}
                        onChange={(e) => setQueryVector(e.target.value)}
                        placeholder="Enter vector values as comma-separated numbers or JSON array"
                        required
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      />
                      <p className="mt-1 text-sm text-gray-500">
                        Example: [0.1, 0.2, 0.3, ...] or 0.1, 0.2, 0.3, ...
                      </p>
                    </div>

                    <div className="col-span-6 sm:col-span-3">
                      <label htmlFor="topK" className="block text-sm font-medium text-gray-700">
                        Top K Results
                      </label>
                      <input
                        type="number"
                        name="topK"
                        id="topK"
                        min="1"
                        max="1000"
                        value={topK}
                        onChange={(e) => setTopK(parseInt(e.target.value))}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      />
                    </div>

                    <div className="col-span-6 sm:col-span-3">
                      <label htmlFor="threshold" className="block text-sm font-medium text-gray-700">
                        Similarity Threshold
                      </label>
                      <input
                        type="number"
                        name="threshold"
                        id="threshold"
                        min="0.0"
                        max="1.0"
                        step="0.01"
                        value={threshold}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      />
                      <p className="mt-1 text-sm text-gray-500">
                        Minimum similarity score (0.0 to 1.0)
                      </p>
                    </div>

                    <div className="col-span-6">
                      <div className="flex items-start">
                        <div className="flex items-center h-5">
                          <input
                            id="includeMetadata"
                            name="includeMetadata"
                            type="checkbox"
                            checked={includeMetadata}
                            onChange={(e) => setIncludeMetadata(e.target.checked)}
                            className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded"
                          />
                        </div>
                        <div className="ml-3 text-sm">
                          <label htmlFor="includeMetadata" className="font-medium text-gray-700">
                            Include metadata in results
                          </label>
                        </div>
                      </div>
                    </div>

                    <div className="col-span-6">
                      <div className="flex items-start">
                        <div className="flex items-center h-5">
                          <input
                            id="includeValues"
                            name="includeValues"
                            type="checkbox"
                            checked={includeValues}
                            onChange={(e) => setIncludeValues(e.target.checked)}
                            className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded"
                          />
                        </div>
                        <div className="ml-3 text-sm">
                          <label htmlFor="includeValues" className="font-medium text-gray-700">
                            Include vector values in results
                          </label>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6">
                    <button
                      type="submit"
                      disabled={loading || !selectedDatabase || !queryVector}
                      className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                    >
                      {loading ? 'Searching...' : 'Search Similar Vectors'}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>

          {/* Search Results */}
          {results.length > 0 && (
            <div className="bg-white shadow overflow-hidden sm:rounded-md">
              <div className="px-4 py-5 border-b border-gray-200 sm:px-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900">Search Results</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Top {results.length} most similar vectors
                </p>
              </div>
              <ul className="divide-y divide-gray-200">
                {results.map((result, index) => (
                  <li key={result.vector?.id || result.vectorId || index} className={index === 0 ? 'border-t border-gray-200' : ''}>
                    <div className="px-4 py-4 sm:px-6">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-medium text-indigo-600 truncate">
                          Vector ID: {result.vector?.id || result.vectorId || 'Unknown'}
                        </div>
                        <div className="ml-2 flex-shrink-0 flex">
                          <span className="inline-flex px-2 text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                            {(result.score || result.similarity || result.similarityScore || 0).toFixed(4)} similarity
                          </span>
                        </div>
                      </div>
                      
                      {result.vector && result.vector.metadata && (
                        <div className="mt-2">
                          <div className="text-sm text-gray-900">
                            <span className="font-medium">Metadata:</span>
                          </div>
                          <div className="mt-1 flex flex-col space-y-1">
                            {Object.entries(result.vector.metadata).map(([key, value]) => (
                              <div key={key} className="text-sm text-gray-500">
                                <span className="font-medium">{key}:</span> {typeof value === 'string' ? value : JSON.stringify(value)}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {result.vector && result.vector.values && includeValues && (
                        <div className="mt-2">
                          <div className="text-sm text-gray-900">
                            <span className="font-medium">Values (first 10):</span>
                          </div>
                          <div className="mt-1 text-sm text-gray-500">
                            {(() => {
                              const values = result.vector.values || [];
                              const displayValues = values.slice(0, 10).map((v) => {
                                const numeric = Number(v);
                                return Number.isFinite(numeric) ? numeric.toFixed(4) : String(v);
                              }).join(', ');
                              return `[${displayValues}${values.length > 10 ? '...' : ''}]`;
                            })()}
                          </div>
                        </div>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}