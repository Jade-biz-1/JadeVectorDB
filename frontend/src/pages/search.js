import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
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
  const [error, setError] = useState('');

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
      setError(`Error fetching databases: ${error.message}`);
    } finally {
      setFetchingDatabases(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

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
      setError(`Error performing search: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout title="Vector Search - JadeVectorDB">
      <style jsx>{`
        .search-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
        }

        .page-header {
          margin-bottom: 30px;
        }

        .page-title {
          font-size: 32px;
          font-weight: 700;
          color: #2c3e50;
          margin: 0 0 10px 0;
        }

        .page-description {
          color: #7f8c8d;
          font-size: 16px;
        }

        .alert {
          padding: 15px;
          border-radius: 8px;
          margin-bottom: 20px;
          font-size: 14px;
        }

        .alert-error {
          background: #fee2e2;
          border: 1px solid #fecaca;
          color: #991b1b;
        }

        .card {
          background: white;
          border-radius: 8px;
          padding: 30px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          margin-bottom: 30px;
        }

        .card-title {
          font-size: 20px;
          font-weight: 600;
          color: #2c3e50;
          margin: 0 0 10px 0;
        }

        .card-subtitle {
          font-size: 14px;
          color: #7f8c8d;
          margin-bottom: 25px;
        }

        .form-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 20px;
        }

        @media (min-width: 768px) {
          .form-grid {
            grid-template-columns: 1fr 1fr;
          }
        }

        .form-group {
          display: flex;
          flex-direction: column;
        }

        .form-group.full-width {
          grid-column: 1 / -1;
        }

        .form-label {
          font-weight: 500;
          color: #2c3e50;
          margin-bottom: 8px;
          font-size: 14px;
        }

        .form-input,
        .form-select,
        .form-textarea {
          padding: 10px 12px;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          font-size: 14px;
          transition: all 0.2s;
          font-family: inherit;
        }

        .form-input:focus,
        .form-select:focus,
        .form-textarea:focus {
          outline: none;
          border-color: #3498db;
          box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }

        .form-textarea {
          resize: vertical;
          min-height: 100px;
          font-family: monospace;
        }

        .form-hint {
          font-size: 12px;
          color: #7f8c8d;
          margin-top: 5px;
        }

        .checkbox-group {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .checkbox {
          width: 18px;
          height: 18px;
          cursor: pointer;
        }

        .checkbox-label {
          font-size: 14px;
          color: #2c3e50;
          cursor: pointer;
        }

        .btn {
          padding: 12px 24px;
          border-radius: 6px;
          font-weight: 500;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
          border: none;
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }

        .btn-primary {
          background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
          color: white;
        }

        .btn-primary:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
        }

        .btn-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .results-card {
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          overflow: hidden;
        }

        .results-header {
          padding: 20px 25px;
          border-bottom: 1px solid #ecf0f1;
        }

        .results-title {
          font-size: 18px;
          font-weight: 600;
          color: #2c3e50;
          margin: 0 0 5px 0;
        }

        .results-subtitle {
          font-size: 14px;
          color: #7f8c8d;
        }

        .results-list {
          display: flex;
          flex-direction: column;
        }

        .result-item {
          padding: 20px 25px;
          border-bottom: 1px solid #ecf0f1;
          transition: background 0.2s;
        }

        .result-item:hover {
          background: #f8f9fa;
        }

        .result-item:last-child {
          border-bottom: none;
        }

        .result-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }

        .result-id {
          font-size: 14px;
          font-weight: 500;
          color: #3498db;
          font-family: monospace;
        }

        .score-badge {
          display: inline-flex;
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 600;
          background: #d4edda;
          color: #155724;
        }

        .metadata-section,
        .values-section {
          margin-top: 12px;
        }

        .section-label {
          font-size: 13px;
          font-weight: 600;
          color: #2c3e50;
          margin-bottom: 5px;
        }

        .metadata-grid {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .metadata-item {
          font-size: 13px;
          color: #555;
        }

        .metadata-key {
          font-weight: 500;
        }

        .values-display {
          font-family: monospace;
          font-size: 12px;
          color: #555;
          background: #f8f9fa;
          padding: 10px;
          border-radius: 4px;
          overflow-x: auto;
        }

        .empty-state {
          text-align: center;
          padding: 60px 20px;
          color: #7f8c8d;
        }

        .empty-icon {
          font-size: 48px;
          margin-bottom: 15px;
        }
      `}</style>

      <div className="search-container">
        <div className="page-header">
          <h1 className="page-title">Vector Search</h1>
          <p className="page-description">Find vectors similar to your query vector</p>
        </div>

        {error && (
          <div className="alert alert-error">
            {error}
          </div>
        )}

        <div className="card">
          <h2 className="card-title">Similarity Search</h2>
          <p className="card-subtitle">Find vectors similar to your query vector</p>

          <form onSubmit={handleSearch}>
            <div className="form-grid">
              <div className="form-group full-width">
                <label htmlFor="database" className="form-label">Database *</label>
                <select
                  id="database"
                  className="form-select"
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
              </div>

              <div className="form-group full-width">
                <label htmlFor="queryVector" className="form-label">Query Vector *</label>
                <textarea
                  id="queryVector"
                  className="form-textarea"
                  value={queryVector}
                  onChange={(e) => setQueryVector(e.target.value)}
                  placeholder="Enter vector values as comma-separated numbers or JSON array"
                  required
                />
                <p className="form-hint">
                  Example: [0.1, 0.2, 0.3, ...] or 0.1, 0.2, 0.3, ...
                </p>
              </div>

              <div className="form-group">
                <label htmlFor="topK" className="form-label">Top K Results</label>
                <input
                  type="number"
                  id="topK"
                  className="form-input"
                  min="1"
                  max="1000"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                />
              </div>

              <div className="form-group">
                <label htmlFor="threshold" className="form-label">Similarity Threshold</label>
                <input
                  type="number"
                  id="threshold"
                  className="form-input"
                  min="0.0"
                  max="1.0"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                />
                <p className="form-hint">Minimum similarity score (0.0 to 1.0)</p>
              </div>

              <div className="form-group full-width">
                <div className="checkbox-group">
                  <input
                    id="includeMetadata"
                    type="checkbox"
                    className="checkbox"
                    checked={includeMetadata}
                    onChange={(e) => setIncludeMetadata(e.target.checked)}
                  />
                  <label htmlFor="includeMetadata" className="checkbox-label">
                    Include metadata in results
                  </label>
                </div>
              </div>

              <div className="form-group full-width">
                <div className="checkbox-group">
                  <input
                    id="includeValues"
                    type="checkbox"
                    className="checkbox"
                    checked={includeValues}
                    onChange={(e) => setIncludeValues(e.target.checked)}
                  />
                  <label htmlFor="includeValues" className="checkbox-label">
                    Include vector values in results
                  </label>
                </div>
              </div>

              <div className="form-group full-width">
                <button
                  type="submit"
                  disabled={loading || !selectedDatabase || !queryVector}
                  className="btn btn-primary"
                >
                  {loading ? 'Searching...' : 'Search Similar Vectors'}
                </button>
              </div>
            </div>
          </form>
        </div>

        {results.length > 0 && (
          <div className="results-card">
            <div className="results-header">
              <h3 className="results-title">Search Results</h3>
              <p className="results-subtitle">Top {results.length} most similar vectors</p>
            </div>
            <div className="results-list">
              {results.map((result, index) => (
                <div key={result.vector?.id || result.vectorId || index} className="result-item">
                  <div className="result-header">
                    <div className="result-id">
                      Vector ID: {result.vector?.id || result.vectorId || 'Unknown'}
                    </div>
                    <span className="score-badge">
                      {(result.score || result.similarity || result.similarityScore || 0).toFixed(4)} similarity
                    </span>
                  </div>

                  {result.vector && result.vector.metadata && (
                    <div className="metadata-section">
                      <div className="section-label">Metadata:</div>
                      <div className="metadata-grid">
                        {Object.entries(result.vector.metadata).map(([key, value]) => (
                          <div key={key} className="metadata-item">
                            <span className="metadata-key">{key}:</span> {typeof value === 'string' ? value : JSON.stringify(value)}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {result.vector && result.vector.values && includeValues && (
                    <div className="values-section">
                      <div className="section-label">Values (first 10):</div>
                      <div className="values-display">
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
          </div>
        )}

        {!loading && results.length === 0 && queryVector && (
          <div className="card">
            <div className="empty-state">
              <div className="empty-icon">🔍</div>
              <h3>No results found</h3>
              <p>Try adjusting your query vector or similarity threshold</p>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
