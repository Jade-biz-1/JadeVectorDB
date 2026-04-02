import { useState, useEffect } from 'react';
import { queryAPI } from '../services/api';
import QueryForm from '../components/QueryForm';
import QueryResponse from '../components/QueryResponse';
import SystemStats from '../components/SystemStats';
import '../styles/QueryPage.css';

function QueryPage() {
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const data = await queryAPI.getStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  };

  const handleQuery = async (question, deviceType, topK) => {
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const data = await queryAPI.query(question, deviceType, topK);
      setResponse(data);
      // Refresh stats after query
      loadStats();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process query');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="query-page">
      <div className="query-container">
        <header className="page-header">
          <h2>Maintenance Documentation Q&A</h2>
          <p>Ask questions about your equipment maintenance procedures</p>
        </header>

        <QueryForm onSubmit={handleQuery} loading={loading} />

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {loading && (
          <div className="loading-message">
            <div className="spinner"></div>
            <p>Processing your question...</p>
          </div>
        )}

        {response && <QueryResponse response={response} />}

        {stats && <SystemStats stats={stats} />}
      </div>
    </div>
  );
}

export default QueryPage;
