import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Layout from '../../components/Layout';
import { databaseApi, vectorApi } from '../../lib/api';

export default function DatabaseDetails() {
  const router = useRouter();
  const { id } = router.query;

  const [database, setDatabase] = useState(null);
  const [vectors, setVectors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (id) {
      fetchDatabaseDetails();
      // Temporarily disable vector fetching until backend supports it
      // fetchVectors();
    }
  }, [id]);

  const fetchDatabaseDetails = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await databaseApi.getDatabase(id);
      setDatabase(response);
    } catch (error) {
      console.error('Error fetching database details:', error);
      setError(`Error fetching database details: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchVectors = async () => {
    try {
      const response = await vectorApi.listVectors(id, 10, 0);
      setVectors(response.vectors || []);
    } catch (error) {
      console.error('Error fetching vectors:', error);
    }
  };

  if (loading) {
    return (
      <Layout title="Loading... - JadeVectorDB">
        <div className="loading">Loading database details...</div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="Error - JadeVectorDB">
        <div className="error-container">
          <h2>Error</h2>
          <p>{error}</p>
          <button onClick={() => router.push('/databases')} className="btn btn-primary">
            Back to Databases
          </button>
        </div>
      </Layout>
    );
  }

  if (!database) {
    return (
      <Layout title="Not Found - JadeVectorDB">
        <div className="error-container">
          <h2>Database Not Found</h2>
          <p>The database you're looking for doesn't exist.</p>
          <button onClick={() => router.push('/databases')} className="btn btn-primary">
            Back to Databases
          </button>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title={`${database.name} - JadeVectorDB`}>
      <style jsx>{`
        .details-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
        }

        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
        }

        .header h1 {
          margin: 0;
          font-size: 32px;
          color: #2c3e50;
        }

        .actions {
          display: flex;
          gap: 10px;
        }

        .btn {
          padding: 10px 20px;
          border: none;
          border-radius: 6px;
          font-size: 14px;
          cursor: pointer;
          text-decoration: none;
          display: inline-block;
          transition: all 0.3s ease;
        }

        .btn-primary {
          background: #3498db;
          color: white;
        }

        .btn-primary:hover {
          background: #2980b9;
        }

        .btn-secondary {
          background: #95a5a6;
          color: white;
        }

        .btn-secondary:hover {
          background: #7f8c8d;
        }

        .info-section {
          background: white;
          border-radius: 8px;
          padding: 30px;
          margin-bottom: 30px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .info-section h2 {
          margin: 0 0 20px 0;
          font-size: 24px;
          color: #2c3e50;
        }

        .info-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
        }

        .info-item {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .info-label {
          font-size: 12px;
          font-weight: 600;
          color: #7f8c8d;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .info-value {
          font-size: 16px;
          color: #2c3e50;
          font-weight: 500;
        }

        .badge {
          display: inline-block;
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 600;
        }

        .badge-success {
          background: #d4edda;
          color: #155724;
        }

        .badge-info {
          background: #d1ecf1;
          color: #0c5460;
        }

        .vectors-section {
          background: white;
          border-radius: 8px;
          padding: 30px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .vectors-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .vectors-header h2 {
          margin: 0;
          font-size: 24px;
          color: #2c3e50;
        }

        .vectors-list {
          display: flex;
          flex-direction: column;
          gap: 15px;
        }

        .vector-item {
          padding: 15px;
          border: 1px solid #e1e8ed;
          border-radius: 6px;
          background: #f8f9fa;
        }

        .vector-id {
          font-family: monospace;
          font-size: 14px;
          color: #3498db;
          margin-bottom: 10px;
        }

        .vector-dims {
          font-size: 12px;
          color: #7f8c8d;
        }

        .empty-state {
          text-align: center;
          padding: 40px;
          color: #7f8c8d;
        }

        .loading, .error-container {
          text-align: center;
          padding: 40px;
        }

        .error-container {
          color: #e74c3c;
        }
      `}</style>

      <div className="details-container">
        <div className="header">
          <h1>{database.name}</h1>
          <div className="actions">
            <button onClick={() => router.push('/databases')} className="btn btn-secondary">
              Back to Databases
            </button>
            <button onClick={() => router.push(`/search?databaseId=${id}`)} className="btn btn-primary">
              Search Vectors
            </button>
          </div>
        </div>

        <div className="info-section">
          <h2>Database Information</h2>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Database ID</span>
              <span className="info-value">{database.databaseId || id}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Name</span>
              <span className="info-value">{database.name}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Description</span>
              <span className="info-value">{database.description || 'No description'}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Vector Dimension</span>
              <span className="info-value">{database.vectorDimension}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Index Type</span>
              <span className="info-value">
                <span className="badge badge-info">{database.indexType}</span>
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">Status</span>
              <span className="info-value">
                <span className="badge badge-success">{database.status || 'active'}</span>
              </span>
            </div>
            {database.created_at && (
              <div className="info-item">
                <span className="info-label">Created</span>
                <span className="info-value">{new Date(database.created_at).toLocaleString()}</span>
              </div>
            )}
            {database.updated_at && (
              <div className="info-item">
                <span className="info-label">Updated</span>
                <span className="info-value">{new Date(database.updated_at).toLocaleString()}</span>
              </div>
            )}
          </div>
        </div>

        <div className="vectors-section">
          <div className="vectors-header">
            <h2>Recent Vectors</h2>
            <button onClick={() => router.push(`/vectors?databaseId=${id}`)} className="btn btn-primary">
              Manage Vectors
            </button>
          </div>

          {vectors.length > 0 ? (
            <div className="vectors-list">
              {vectors.map((vector, index) => (
                <div key={vector.id || index} className="vector-item">
                  <div className="vector-id">ID: {vector.id}</div>
                  <div className="vector-dims">
                    Dimensions: {vector.vector?.length || database.vectorDimension}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <p>No vectors in this database yet.</p>
              <button onClick={() => router.push(`/vectors?databaseId=${id}`)} className="btn btn-primary">
                Add Vectors
              </button>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
