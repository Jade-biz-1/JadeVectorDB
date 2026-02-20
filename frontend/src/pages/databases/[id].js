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
  const [expandedVectorId, setExpandedVectorId] = useState(null);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deleteConfirmName, setDeleteConfirmName] = useState('');
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (id) {
      fetchDatabaseDetails();
      fetchVectors();
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
      const response = await vectorApi.listVectors(id, 5, 0);
      setVectors(response.vectors || []);
    } catch (error) {
      console.error('Error fetching vectors:', error);
    }
  };

  const handleDeleteDatabase = async () => {
    setDeleting(true);
    try {
      await databaseApi.deleteDatabase(id);
      router.push('/databases');
    } catch (error) {
      alert('Failed to delete database: ' + error.message);
    } finally {
      setDeleting(false);
    }
  };

  const toggleVectorExpand = (vectorId) => {
    setExpandedVectorId(expandedVectorId === vectorId ? null : vectorId);
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

        .btn-danger {
          background: #e74c3c;
          color: white;
        }

        .btn-danger:hover {
          background: #c0392b;
        }

        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .modal {
          background: white;
          border-radius: 8px;
          padding: 30px;
          max-width: 480px;
          width: 90%;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .modal-header {
          font-size: 20px;
          font-weight: 600;
          color: #e74c3c;
          margin: 0 0 15px 0;
        }

        .delete-warning {
          color: #2c3e50;
          margin-bottom: 20px;
          line-height: 1.5;
        }

        .delete-input {
          width: 100%;
          padding: 10px 12px;
          border: 1px solid #ddd;
          border-radius: 6px;
          font-size: 14px;
          margin-bottom: 20px;
          box-sizing: border-box;
        }

        .delete-input:focus {
          outline: none;
          border-color: #e74c3c;
        }

        .modal-actions {
          display: flex;
          justify-content: flex-end;
          gap: 10px;
        }

        .btn-danger:disabled {
          background: #f5b7b1;
          cursor: not-allowed;
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
          border: 1px solid #e1e8ed;
          border-radius: 6px;
          background: #ffffff;
          overflow: hidden;
          transition: box-shadow 0.2s;
        }

        .vector-item:hover {
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .vector-item-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 15px;
          background: #f8f9fa;
        }

        .vector-id-link {
          display: flex;
          align-items: center;
          gap: 10px;
          transition: opacity 0.2s;
        }

        .vector-id-link:hover {
          opacity: 0.7;
        }

        .vector-id-badge {
          font-family: monospace;
          font-size: 14px;
          color: #3498db;
          font-weight: 600;
        }

        .expand-icon {
          color: #7f8c8d;
          font-size: 12px;
        }

        .vector-dims {
          font-size: 12px;
          color: #7f8c8d;
          background: #e1e8ed;
          padding: 4px 12px;
          border-radius: 12px;
        }

        .vector-details {
          padding: 15px;
          border-top: 1px solid #e1e8ed;
          animation: slideDown 0.2s ease-out;
        }

        @keyframes slideDown {
          from {
            opacity: 0;
            max-height: 0;
          }
          to {
            opacity: 1;
            max-height: 500px;
          }
        }

        .vector-details-section {
          margin-bottom: 15px;
        }

        .vector-details-section:last-child {
          margin-bottom: 0;
        }

        .detail-label {
          font-size: 12px;
          font-weight: 600;
          color: #7f8c8d;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 8px;
        }

        .values-preview {
          display: flex;
          align-items: start;
          gap: 10px;
        }

        .values-text {
          font-family: monospace;
          font-size: 12px;
          color: #2c3e50;
          background: #f8f9fa;
          padding: 10px;
          border-radius: 4px;
          flex: 1;
          word-break: break-all;
        }

        .btn-copy {
          padding: 6px 12px;
          background: #3498db;
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 11px;
          cursor: pointer;
          white-space: nowrap;
          transition: background 0.2s;
        }

        .btn-copy:hover {
          background: #2980b9;
        }

        .metadata-content {
          font-family: monospace;
          font-size: 12px;
          color: #2c3e50;
          background: #f8f9fa;
          padding: 10px;
          border-radius: 4px;
          overflow-x: auto;
          margin: 0;
        }

        .no-data {
          font-size: 12px;
          color: #95a5a6;
          font-style: italic;
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
            <button onClick={() => setDeleteModalOpen(true)} className="btn btn-danger">
              Delete Database
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
              {vectors.map((vector, index) => {
                const isExpanded = expandedVectorId === vector.id;
                const vectorLength = vector.values?.length || vector.vector?.length || database.vectorDimension;

                return (
                  <div key={vector.id || index} className="vector-item">
                    <div className="vector-item-header">
                      <div
                        className="vector-id-link"
                        onClick={() => toggleVectorExpand(vector.id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <span className="vector-id-badge">ID: {vector.id}</span>
                        <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
                      </div>
                      <div className="vector-dims">
                        {vectorLength} dimensions
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="vector-details">
                        <div className="vector-details-section">
                          <div className="detail-label">Vector Values:</div>
                          <div className="vector-values">
                            {vector.values ? (
                              <div className="values-preview">
                                <span className="values-text">
                                  [{vector.values.slice(0, 10).join(', ')}
                                  {vector.values.length > 10 ? `, ... (${vector.values.length - 10} more)` : ''}]
                                </span>
                                <button
                                  className="btn-copy"
                                  onClick={() => {
                                    navigator.clipboard.writeText(JSON.stringify(vector.values));
                                    alert('Vector values copied to clipboard!');
                                  }}
                                >
                                  Copy All
                                </button>
                              </div>
                            ) : (
                              <span className="no-data">No vector data available</span>
                            )}
                          </div>
                        </div>

                        {vector.metadata && Object.keys(vector.metadata).length > 0 && (
                          <div className="vector-details-section">
                            <div className="detail-label">Metadata:</div>
                            <pre className="metadata-content">
                              {JSON.stringify(vector.metadata, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
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

      {deleteModalOpen && (
        <div className="modal-overlay" onClick={() => setDeleteModalOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3 className="modal-header">Delete Database</h3>
            <p className="delete-warning">
              This will permanently delete <strong>{database.name}</strong> and all its vectors. This action cannot be undone.
            </p>
            <p className="delete-warning">
              Type <strong>{database.name}</strong> to confirm:
            </p>
            <input
              type="text"
              className="delete-input"
              placeholder="Enter database name"
              value={deleteConfirmName}
              onChange={(e) => setDeleteConfirmName(e.target.value)}
              autoFocus
            />
            <div className="modal-actions">
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setDeleteModalOpen(false);
                  setDeleteConfirmName('');
                }}
              >
                Cancel
              </button>
              <button
                className="btn btn-danger"
                disabled={deleteConfirmName !== database.name || deleting}
                onClick={handleDeleteDatabase}
              >
                {deleting ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </Layout>
  );
}
