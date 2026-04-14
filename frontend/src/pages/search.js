import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { searchApi } from '../lib/api';
import { useDatabases } from '../hooks/useDatabases';
import {
  Alert, AlertDescription,
  Button,
  Card, CardHeader, CardTitle, CardDescription, CardContent,
  EmptyState,
  FormField,
  StatusBadge,
} from '../components/ui';

export default function SearchInterface() {
  const [queryVector, setQueryVector] = useState('');
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [topK, setTopK] = useState(10);
  const [threshold, setThreshold] = useState(0.0);
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [includeValues, setIncludeValues] = useState(false);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const { databases, loading: fetchingDatabases, error: dbError } = useDatabases();

  // Surface database-fetch errors through the page error state
  useEffect(() => {
    if (dbError) setError(dbError);
  }, [dbError]);

  // Extract database ID from query parameters if available
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      const dbId = urlParams.get('databaseId');
      if (dbId) {
        setSelectedDatabase(dbId);
      }
    }
  }, []);

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // Parse query vector - handle both JSON array and comma-separated string
      let parsedVector = [];

      if (queryVector.trim().startsWith('[')) {
        parsedVector = JSON.parse(queryVector);
      } else {
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
      setError(`Error performing search: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const inputCls = 'w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-500 transition';

  return (
    <Layout title="Vector Search - JadeVectorDB">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-1">Vector Search</h1>
        <p className="text-gray-500">Find vectors similar to your query vector</p>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-6 bg-red-50 border-red-200 text-red-800">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-xl">Similarity Search</CardTitle>
          <CardDescription>Find vectors similar to your query vector</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSearch}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

              <FormField label="Database" htmlFor="database" required className="md:col-span-2">
                <select
                  id="database"
                  className={inputCls}
                  value={selectedDatabase}
                  onChange={(e) => setSelectedDatabase(e.target.value)}
                  required
                  disabled={fetchingDatabases}
                >
                  <option value="">Select a database</option>
                  {databases.map((db) => (
                    <option key={db.id} value={db.id}>{db.name}</option>
                  ))}
                </select>
              </FormField>

              <FormField
                label="Query Vector"
                htmlFor="queryVector"
                required
                hint="Example: [0.1, 0.2, 0.3, ...] or 0.1, 0.2, 0.3, ..."
                className="md:col-span-2"
              >
                <textarea
                  id="queryVector"
                  className={`${inputCls} resize-y min-h-[100px] font-mono`}
                  value={queryVector}
                  onChange={(e) => setQueryVector(e.target.value)}
                  placeholder="Enter vector values as comma-separated numbers or JSON array"
                  required
                />
              </FormField>

              <FormField label="Top K Results" htmlFor="topK">
                <input
                  type="number"
                  id="topK"
                  className={inputCls}
                  min="1"
                  max="1000"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                />
              </FormField>

              <FormField
                label="Similarity Threshold"
                htmlFor="threshold"
                hint="Minimum similarity score (0.0 to 1.0)"
              >
                <input
                  type="number"
                  id="threshold"
                  className={inputCls}
                  min="0.0"
                  max="1.0"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                />
              </FormField>

              <div className="md:col-span-2 flex flex-col gap-3">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    id="includeMetadata"
                    type="checkbox"
                    className="w-4 h-4 rounded border-gray-300 text-indigo-600"
                    checked={includeMetadata}
                    onChange={(e) => setIncludeMetadata(e.target.checked)}
                  />
                  <span className="text-sm text-gray-700">Include metadata in results</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    id="includeValues"
                    type="checkbox"
                    className="w-4 h-4 rounded border-gray-300 text-indigo-600"
                    checked={includeValues}
                    onChange={(e) => setIncludeValues(e.target.checked)}
                  />
                  <span className="text-sm text-gray-700">Include vector values in results</span>
                </label>
              </div>

              <div className="md:col-span-2">
                <Button
                  type="submit"
                  disabled={loading || !selectedDatabase || !queryVector}
                >
                  {loading ? 'Searching…' : 'Search Similar Vectors'}
                </Button>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>

      {results.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Search Results</CardTitle>
            <CardDescription>Top {results.length} most similar vectors</CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="divide-y divide-gray-100">
              {results.map((result, index) => (
                <div
                  key={result.vector?.id || result.vectorId || index}
                  className="px-6 py-5 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium text-indigo-600 font-mono">
                      Vector ID: {result.vector?.id || result.vectorId || 'Unknown'}
                    </span>
                    <StatusBadge
                      status="success"
                      label={`${(result.score || result.similarity || result.similarityScore || 0).toFixed(4)} similarity`}
                    />
                  </div>

                  {result.vector && result.vector.metadata && (
                    <div className="mt-2">
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Metadata</p>
                      <div className="space-y-0.5">
                        {Object.entries(result.vector.metadata).map(([key, value]) => (
                          <div key={key} className="text-sm text-gray-600">
                            <span className="font-medium">{key}:</span>{' '}
                            {typeof value === 'string' ? value : JSON.stringify(value)}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {result.vector && result.vector.values && includeValues && (
                    <div className="mt-2">
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                        Values (first 10)
                      </p>
                      <div className="font-mono text-xs text-gray-600 bg-gray-50 px-3 py-2 rounded overflow-x-auto">
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
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {!loading && results.length === 0 && queryVector && (
        <Card>
          <CardContent>
            <EmptyState
              icon="🔍"
              title="No results found"
              description="Try adjusting your query vector or similarity threshold"
            />
          </CardContent>
        </Card>
      )}
    </Layout>
  );
}
