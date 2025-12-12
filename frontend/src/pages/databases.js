import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { databaseApi } from '../lib/api';

export default function DatabaseManagement() {
  const [databases, setDatabases] = useState([]);
  const [newDatabase, setNewDatabase] = useState({
    name: '',
    description: '',
    vectorDimension: 128,
    indexType: 'FLAT'
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const fetchDatabases = async () => {
    try {
      setError('');
      const response = await databaseApi.listDatabases();
      setDatabases(response.databases || []);
    } catch (error) {
      console.error('Error fetching databases:', error);
      setError(`Error fetching databases: ${error.message}`);
    }
  };

  const handleCreateDatabase = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const databaseData = {
        name: newDatabase.name,
        description: newDatabase.description,
        vectorDimension: parseInt(newDatabase.vectorDimension),
        indexType: newDatabase.indexType
      };

      await databaseApi.createDatabase(databaseData);
      setNewDatabase({ name: '', description: '', vectorDimension: 128, indexType: 'FLAT' });
      setSuccess('Database created successfully!');
      fetchDatabases();
    } catch (error) {
      console.error('Error creating database:', error);
      setError(`Error creating database: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatabases();
  }, []);

  return (
    <Layout title="Database Management - JadeVectorDB">
      <style jsx>{`
        .page-header {
          margin-bottom: 2rem;
        }

        .page-title {
          font-size: 2rem;
          font-weight: 700;
          color: #111827;
          margin-bottom: 0.5rem;
        }

        .page-description {
          color: #6b7280;
          font-size: 1rem;
        }

        .alert {
          padding: 1rem;
          border-radius: 8px;
          margin-bottom: 1.5rem;
          font-size: 0.875rem;
        }

        .alert-error {
          background: #fee2e2;
          border: 1px solid #fecaca;
          color: #991b1b;
        }

        .alert-success {
          background: #dcfce7;
          border: 1px solid #bbf7d0;
          color: #166534;
        }

        .card {
          background: white;
          border-radius: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          padding: 2rem;
          margin-bottom: 2rem;
        }

        .card-header {
          margin-bottom: 1.5rem;
        }

        .card-title {
          font-size: 1.25rem;
          font-weight: 600;
          color: #111827;
          margin-bottom: 0.5rem;
        }

        .card-subtitle {
          color: #6b7280;
          font-size: 0.875rem;
        }

        .form-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 1.5rem;
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
          color: #374151;
          margin-bottom: 0.5rem;
          font-size: 0.875rem;
        }

        .form-input,
        .form-select,
        .form-textarea {
          padding: 0.75rem;
          border: 1px solid #d1d5db;
          border-radius: 8px;
          font-size: 0.875rem;
          transition: all 0.2s;
        }

        .form-input:focus,
        .form-select:focus,
        .form-textarea:focus {
          outline: none;
          border-color: #667eea;
          box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .form-textarea {
          resize: vertical;
          min-height: 80px;
        }

        .btn {
          padding: 0.75rem 1.5rem;
          border-radius: 8px;
          font-weight: 500;
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.2s;
          border: none;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
        }

        .btn-primary {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
        }

        .btn-primary:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .btn-primary:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .btn-secondary {
          background: #f3f4f6;
          color: #374151;
        }

        .btn-secondary:hover {
          background: #e5e7eb;
        }

        .btn-danger {
          background: #ef4444;
          color: white;
        }

        .btn-danger:hover {
          background: #dc2626;
        }

        .empty-state {
          text-align: center;
          padding: 3rem 1rem;
          color: #6b7280;
        }

        .empty-state-icon {
          font-size: 4rem;
          margin-bottom: 1rem;
        }

        .empty-state-title {
          font-size: 1.25rem;
          font-weight: 600;
          color: #374151;
          margin-bottom: 0.5rem;
        }

        .database-list {
          display: grid;
          gap: 1rem;
        }

        .database-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          padding: 1.5rem;
          transition: all 0.2s;
        }

        .database-card:hover {
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
          transform: translateY(-2px);
        }

        .database-header {
          display: flex;
          justify-content: space-between;
          align-items: start;
          margin-bottom: 1rem;
        }

        .database-name {
          font-size: 1.25rem;
          font-weight: 600;
          color: #111827;
          text-decoration: none;
          margin-bottom: 0.5rem;
          display: block;
        }

        .database-name:hover {
          color: #667eea;
        }

        .database-description {
          color: #6b7280;
          font-size: 0.875rem;
          margin-bottom: 1rem;
        }

        .database-stats {
          display: flex;
          gap: 2rem;
          margin-bottom: 1rem;
        }

        .stat {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: #6b7280;
          font-size: 0.875rem;
        }

        .stat-icon {
          color: #9ca3af;
        }

        .badge {
          display: inline-flex;
          align-items: center;
          padding: 0.25rem 0.75rem;
          border-radius: 9999px;
          font-size: 0.75rem;
          font-weight: 500;
        }

        .badge-success {
          background: #dcfce7;
          color: #166534;
        }

        .badge-error {
          background: #fee2e2;
          color: #991b1b;
        }

        .database-actions {
          display: flex;
          gap: 0.75rem;
          flex-wrap: wrap;
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #e5e7eb;
        }

        .btn-small {
          padding: 0.5rem 1rem;
          font-size: 0.813rem;
        }
      `}</style>

      <div className="page-header">
        <h1 className="page-title">Database Management</h1>
        <p className="page-description">Create and manage your vector databases</p>
      </div>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      {success && (
        <div className="alert alert-success">
          {success}
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Create New Database</h2>
          <p className="card-subtitle">Configure a new vector database with specific settings</p>
        </div>

        <form onSubmit={handleCreateDatabase}>
          <div className="form-grid">
            <div className="form-group full-width">
              <label htmlFor="name" className="form-label">Database Name *</label>
              <input
                type="text"
                id="name"
                className="form-input"
                value={newDatabase.name}
                onChange={(e) => setNewDatabase({...newDatabase, name: e.target.value})}
                required
                placeholder="my-vector-db"
              />
            </div>

            <div className="form-group full-width">
              <label htmlFor="description" className="form-label">Description</label>
              <textarea
                id="description"
                className="form-textarea"
                value={newDatabase.description}
                onChange={(e) => setNewDatabase({...newDatabase, description: e.target.value})}
                placeholder="A brief description of this database"
              />
            </div>

            <div className="form-group">
              <label htmlFor="vectorDimension" className="form-label">Vector Dimension *</label>
              <select
                id="vectorDimension"
                className="form-select"
                value={newDatabase.vectorDimension}
                onChange={(e) => setNewDatabase({...newDatabase, vectorDimension: parseInt(e.target.value)})}
              >
                <option value={128}>128</option>
                <option value={256}>256</option>
                <option value={512}>512</option>
                <option value={768}>768 (BERT)</option>
                <option value={1024}>1024</option>
                <option value={1536}>1536 (OpenAI)</option>
                <option value={2048}>2048</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="indexType" className="form-label">Index Type *</label>
              <select
                id="indexType"
                className="form-select"
                value={newDatabase.indexType}
                onChange={(e) => setNewDatabase({...newDatabase, indexType: e.target.value})}
              >
                <option value="FLAT">FLAT (Linear Search)</option>
                <option value="HNSW">HNSW (Fast Approximate)</option>
                <option value="IVF">IVF (Inverted File)</option>
                <option value="LSH">LSH (Locality Sensitive)</option>
              </select>
            </div>

            <div className="form-group full-width">
              <button type="submit" className="btn btn-primary" disabled={loading}>
                {loading ? 'Creating...' : 'Create Database'}
              </button>
            </div>
          </div>
        </form>
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Your Databases</h2>
          <p className="card-subtitle">Manage and access your vector databases</p>
        </div>

        {databases.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📊</div>
            <h3 className="empty-state-title">No databases yet</h3>
            <p>Create your first database to get started</p>
          </div>
        ) : (
          <div className="database-list">
            {databases.map((database) => (
              <div key={database.id || database.databaseId} className="database-card">
                <div className="database-header">
                  <div>
                    <a
                      href={`/databases/${database.id || database.databaseId}`}
                      className="database-name"
                    >
                      {database.name}
                    </a>
                    <span className={`badge ${database.status === 'active' ? 'badge-success' : 'badge-error'}`}>
                      {database.status || 'active'}
                    </span>
                  </div>
                </div>

                {database.description && (
                  <p className="database-description">{database.description}</p>
                )}

                <div className="database-stats">
                  <div className="stat">
                    <span className="stat-icon">📊</span>
                    <span>{(database.vectors || database.stats?.vectorCount || 0).toLocaleString()} vectors</span>
                  </div>
                  <div className="stat">
                    <span className="stat-icon">🔍</span>
                    <span>{(database.indexes || database.stats?.indexCount || 0)} indexes</span>
                  </div>
                  <div className="stat">
                    <span className="stat-icon">📏</span>
                    <span>{database.vectorDimension}D</span>
                  </div>
                </div>

                <div className="database-actions">
                  <a
                    href={`/databases/${database.id || database.databaseId}`}
                    className="btn btn-secondary btn-small"
                  >
                    View Details
                  </a>
                  <a
                    href={`/search?databaseId=${database.id || database.databaseId}`}
                    className="btn btn-primary btn-small"
                  >
                    Search Vectors
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Layout>
  );
}
