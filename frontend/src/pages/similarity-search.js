import Head from 'next/head';
import { useState, useEffect } from 'react';
import { searchApi, databaseApi } from '../lib/api';

export default function SimilaritySearchPage() {
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [queryVector, setQueryVector] = useState('');
  const [topK, setTopK] = useState(10);
  const [threshold, setThreshold] = useState(0.0);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchDatabases();
  }, []);

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      setDatabases(response.databases || []);
    } catch (error) {
      alert('Error fetching databases: ' + error.message);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!selectedDatabase) return alert('Select a database first');
    setLoading(true);
    try {
      let parsedVector = [];
      if (queryVector.trim().startsWith('[')) {
        parsedVector = JSON.parse(queryVector);
      } else {
        parsedVector = queryVector.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
      }
      const searchRequest = {
        queryVector: parsedVector,
        topK: parseInt(topK),
        threshold: parseFloat(threshold)
      };
      const response = await searchApi.similaritySearch(selectedDatabase, searchRequest);
      setResults(response.results || []);
    } catch (error) {
      alert('Error performing search: ' + error.message);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Similarity Search - JadeVectorDB</title>
        <meta name="description" content="Perform similarity search in JadeVectorDB" />
      </Head>
      <main className="max-w-4xl mx-auto py-8">
        <h1 className="text-2xl font-bold mb-6">Similarity Search</h1>
        <div className="mb-4">
          <label className="block mb-2">Select Database:</label>
          <select value={selectedDatabase} onChange={e => setSelectedDatabase(e.target.value)} className="border rounded px-2 py-1">
            <option value="">-- Select --</option>
            {databases.map(db => (
              <option key={db.databaseId || db.id} value={db.databaseId || db.id}>{db.name}</option>
            ))}
          </select>
        </div>
        <form onSubmit={handleSearch} className="mb-6">
          <h2 className="text-lg font-semibold mb-2">Query Vector</h2>
          <input type="text" placeholder="Comma-separated values or JSON array" value={queryVector} onChange={e => setQueryVector(e.target.value)} className="border rounded px-2 py-1 mr-2 w-full" required />
          <div className="flex space-x-2 mt-2">
            <input type="number" min={1} max={100} value={topK} onChange={e => setTopK(e.target.value)} className="border rounded px-2 py-1" placeholder="Top K" />
            <input type="number" step={0.01} value={threshold} onChange={e => setThreshold(e.target.value)} className="border rounded px-2 py-1" placeholder="Threshold" />
            <button type="submit" className="bg-indigo-600 text-white px-4 py-1 rounded">Search</button>
          </div>
        </form>
        <div>
          <h2 className="text-lg font-semibold mb-2">Results</h2>
          <ul className="divide-y">
            {results.map((result, idx) => (
              <li key={idx} className="py-2">
                <div>Score: {result.score}</div>
                <div>Vector: {result.values ? result.values.join(', ') : ''}</div>
                <div>Metadata: {result.metadata ? JSON.stringify(result.metadata) : ''}</div>
              </li>
            ))}
          </ul>
        </div>
      </main>
    </div>
  );
}
