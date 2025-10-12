import Head from 'next/head';
import { useState } from 'react';

export default function SearchInterface() {
  const [queryVector, setQueryVector] = useState('');
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  
  const databases = [
    { id: 'db1', name: 'Documents' },
    { id: 'db2', name: 'Images' },
    { id: 'db3', name: 'Products' }
  ];

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      // In real implementation, this would call the API to perform search
      console.log('Searching with:', { queryVector, selectedDatabase, topK });
      
      // Mock search results
      setResults([
        { id: 'vec1', similarity: 0.95, metadata: { category: 'document', title: 'Introduction to ML' } },
        { id: 'vec2', similarity: 0.89, metadata: { category: 'document', title: 'Deep Learning Basics' } },
        { id: 'vec3', similarity: 0.87, metadata: { category: 'document', title: 'Neural Networks' } },
        { id: 'vec4', similarity: 0.85, metadata: { category: 'document', title: 'AI Concepts' } },
        { id: 'vec5', similarity: 0.82, metadata: { category: 'document', title: 'Machine Learning Models' } }
      ]);
    } catch (error) {
      console.error('Error performing search:', error);
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
                        max="100"
                        value={topK}
                        onChange={(e) => setTopK(parseInt(e.target.value))}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      />
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
                  <li key={result.id}>
                    <div className="px-4 py-4 sm:px-6">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-medium text-indigo-600 truncate">Vector ID: {result.id}</div>
                        <div className="ml-2 flex-shrink-0 flex">
                          <span className="inline-flex px-2 text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                            {result.similarity.toFixed(4)} similarity
                          </span>
                        </div>
                      </div>
                      <div className="mt-2">
                        <div className="text-sm text-gray-900">
                          <span className="font-medium">Metadata:</span>
                        </div>
                        <div className="mt-1 flex flex-col space-y-1">
                          {Object.entries(result.metadata).map(([key, value]) => (
                            <div key={key} className="text-sm text-gray-500">
                              <span className="font-medium">{key}:</span> {value}
                            </div>
                          ))}
                        </div>
                      </div>
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