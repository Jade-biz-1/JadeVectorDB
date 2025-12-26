import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { vectorApi, databaseApi } from '../lib/api';

export default function VectorManagement() {
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [databaseDetails, setDatabaseDetails] = useState(null);
  const [vectors, setVectors] = useState([]);
  const [newVector, setNewVector] = useState({ values: '', metadata: '' });
  const [editVector, setEditVector] = useState(null);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);
  const [totalVectors, setTotalVectors] = useState(0);

  useEffect(() => {
    fetchDatabases();
  }, []);

  useEffect(() => {
    if (selectedDatabase) {
      fetchDatabaseDetails();
      setCurrentPage(1); // Reset to page 1 when database changes
    }
  }, [selectedDatabase]);

  useEffect(() => {
    if (selectedDatabase) {
      fetchVectors();
    }
  }, [selectedDatabase, currentPage]);

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      setDatabases(response.databases || []);
    } catch (error) {
      alert('Error fetching databases: ' + error.message);
    }
  };

  const fetchDatabaseDetails = async () => {
    if (!selectedDatabase) return;
    try {
      const response = await databaseApi.getDatabase(selectedDatabase);
      setDatabaseDetails(response);
    } catch (error) {
      console.error('Error fetching database details:', error);
    }
  };

  const fetchVectors = async () => {
    if (!selectedDatabase) return;
    setLoading(true);
    try {
      const offset = (currentPage - 1) * pageSize;
      const response = await vectorApi.listVectors(selectedDatabase, pageSize, offset);
      setVectors(response.vectors || []);
      setTotalVectors(response.total || 0);
    } catch (error) {
      console.error('Error fetching vectors:', error);
      alert('Error fetching vectors: ' + error.message);
      setVectors([]);
      setTotalVectors(0);
    }
    setLoading(false);
  };

  const totalPages = Math.ceil(totalVectors / pageSize);

  const goToPage = (page) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  const handleCreateVector = async (e) => {
    e.preventDefault();
    if (!selectedDatabase) return alert('Select a database first');

    // Parse and validate values
    const values = newVector.values.trim().split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));

    // Validate dimension
    if (databaseDetails && databaseDetails.vectorDimension) {
      const expectedDim = databaseDetails.vectorDimension;
      if (values.length !== expectedDim) {
        return alert(
          `Dimension mismatch!\n\n` +
          `Database "${databaseDetails.name}" expects ${expectedDim} dimensions.\n` +
          `You provided ${values.length} dimension${values.length !== 1 ? 's' : ''}.\n\n` +
          `Please provide exactly ${expectedDim} comma-separated values.`
        );
      }
    }

    setLoading(true);
    try {
      const metadata = newVector.metadata ? JSON.parse(newVector.metadata) : {};
      const id = `vec_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
      await vectorApi.storeVector(selectedDatabase, { id, values, metadata });
      setNewVector({ values: '', metadata: '' });
      fetchVectors();
    } catch (error) {
      alert('Error creating vector: ' + error.message);
    }
    setLoading(false);
  };

  const handleEditVector = (vector) => {
    setEditVector({
      id: vector.id,
      values: vector.values.join(','),
      metadata: JSON.stringify(vector.metadata)
    });
    setEditModalOpen(true);
  };

  const handleUpdateVector = async (e) => {
    e.preventDefault();
    if (!selectedDatabase) return;
    setLoading(true);
    try {
      const values = editVector.values.split(',').map(Number);
      const metadata = editVector.metadata ? JSON.parse(editVector.metadata) : {};
      await vectorApi.updateVector(selectedDatabase, editVector.id, { values, metadata });
      setEditModalOpen(false);
      fetchVectors();
    } catch (error) {
      alert('Error updating vector: ' + error.message);
    }
    setLoading(false);
  };

  const handleDeleteVector = async (vectorId) => {
    if (!selectedDatabase) return;
    if (!confirm('Are you sure you want to delete this vector?')) return;
    setLoading(true);
    try {
      await vectorApi.deleteVector(selectedDatabase, vectorId);
      fetchVectors();
    } catch (error) {
      alert('Error deleting vector: ' + error.message);
    }
    setLoading(false);
  };

  return (
    <Layout title="Vector Management - JadeVectorDB">
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

        .card {
          background: white;
          border-radius: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          padding: 2rem;
          margin-bottom: 2rem;
        }

        .card-title {
          font-size: 1.25rem;
          font-weight: 600;
          color: #111827;
          margin-bottom: 1.5rem;
        }

        .form-group {
          margin-bottom: 1.5rem;
        }

        .form-label {
          display: block;
          font-size: 0.875rem;
          font-weight: 500;
          color: #374151;
          margin-bottom: 0.5rem;
        }

        .form-input, .form-select {
          width: 100%;
          padding: 0.75rem;
          border: 1px solid #d1d5db;
          border-radius: 8px;
          font-size: 1rem;
          transition: all 0.2s;
        }

        .form-input:focus, .form-select:focus {
          outline: none;
          border-color: #667eea;
          box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }

        .btn {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 8px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          font-size: 1rem;
        }

        .btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-primary {
          background: #667eea;
          color: white;
        }

        .btn-primary:hover:not(:disabled) {
          background: #5568d3;
        }

        .btn-warning {
          background: #f59e0b;
          color: white;
        }

        .btn-warning:hover {
          background: #d97706;
        }

        .btn-danger {
          background: #ef4444;
          color: white;
        }

        .btn-danger:hover {
          background: #dc2626;
        }

        .btn-secondary {
          background: #f3f4f6;
          color: #374151;
        }

        .btn-secondary:hover {
          background: #e5e7eb;
        }

        .vector-list {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .vector-item {
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 1.5rem;
          transition: box-shadow 0.2s;
        }

        .vector-item:hover {
          box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .vector-header {
          display: flex;
          justify-content: space-between;
          align-items: start;
          margin-bottom: 1rem;
        }

        .vector-id {
          display: inline-block;
          padding: 0.25rem 0.75rem;
          background: #ddd6fe;
          color: #5b21b6;
          border-radius: 9999px;
          font-size: 0.875rem;
          font-weight: 500;
        }

        .vector-actions {
          display: flex;
          gap: 0.5rem;
        }

        .vector-actions button {
          padding: 0.5rem 1rem;
          font-size: 0.875rem;
        }

        .vector-content {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .vector-field {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .vector-field-label {
          font-size: 0.75rem;
          font-weight: 500;
          color: #6b7280;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .vector-field-value {
          font-family: 'Courier New', monospace;
          font-size: 0.875rem;
          color: #111827;
          background: #f9fafb;
          padding: 0.75rem;
          border-radius: 6px;
          word-break: break-all;
        }

        .empty-state {
          text-align: center;
          padding: 3rem 2rem;
          color: #6b7280;
        }

        .loading-state {
          text-align: center;
          padding: 2rem;
          color: #6b7280;
        }

        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0,0,0,0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .modal {
          background: white;
          border-radius: 12px;
          box-shadow: 0 20px 25px rgba(0,0,0,0.15);
          max-width: 600px;
          width: 90%;
          padding: 2rem;
        }

        .modal-header {
          font-size: 1.5rem;
          font-weight: 700;
          color: #111827;
          margin-bottom: 1.5rem;
        }

        .modal-actions {
          display: flex;
          justify-content: flex-end;
          gap: 1rem;
          margin-top: 2rem;
        }

        .dimension-info {
          background: #eff6ff;
          border: 1px solid #bfdbfe;
          border-radius: 8px;
          padding: 1rem;
          margin-bottom: 1.5rem;
          color: #1e40af;
          font-size: 0.875rem;
        }

        .value-count {
          font-size: 0.875rem;
          font-weight: 600;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          margin-left: 0.5rem;
        }

        .value-count.valid {
          color: #059669;
          background: #d1fae5;
        }

        .value-count.invalid {
          color: #dc2626;
          background: #fee2e2;
        }

        .card-header-with-info {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
        }

        .vector-count-info {
          font-size: 0.875rem;
          color: #6b7280;
          font-weight: 500;
        }

        .pagination {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 0.5rem;
          margin-top: 2rem;
          padding-top: 1.5rem;
          border-top: 1px solid #e5e7eb;
        }

        .pagination-btn {
          padding: 0.5rem 1rem;
          background: #667eea;
          color: white;
          border: none;
          border-radius: 6px;
          font-size: 0.875rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .pagination-btn:hover:not(:disabled) {
          background: #5568d3;
        }

        .pagination-btn:disabled {
          background: #d1d5db;
          cursor: not-allowed;
        }

        .pagination-pages {
          display: flex;
          align-items: center;
          gap: 0.25rem;
        }

        .pagination-page {
          min-width: 2.5rem;
          height: 2.5rem;
          padding: 0.5rem;
          background: white;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          font-size: 0.875rem;
          font-weight: 500;
          color: #374151;
          cursor: pointer;
          transition: all 0.2s;
        }

        .pagination-page:hover {
          background: #f3f4f6;
          border-color: #667eea;
        }

        .pagination-page.active {
          background: #667eea;
          color: white;
          border-color: #667eea;
          cursor: default;
        }

        .pagination-ellipsis {
          padding: 0 0.5rem;
          color: #6b7280;
        }
      `}</style>

      <div className="page-header">
        <h1 className="page-title">Vector Management</h1>
        <p className="page-description">Store, manage, and search vector embeddings</p>
      </div>

      <div className="card">
        <h2 className="card-title">Select Database</h2>
        <div className="form-group">
          <label className="form-label">Database</label>
          <select
            className="form-select"
            value={selectedDatabase}
            onChange={e => { setSelectedDatabase(e.target.value); fetchVectors(); }}
          >
            <option value="">-- Select a database --</option>
            {databases.map(db => (
              <option key={db.databaseId || db.id} value={db.databaseId || db.id}>
                {db.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="card">
        <h2 className="card-title">Create New Vector</h2>
        {databaseDetails && (
          <div className="dimension-info">
            <strong>Required Dimension:</strong> {databaseDetails.vectorDimension} values
            {' | '}
            <strong>Database:</strong> {databaseDetails.name}
            {' | '}
            <strong>Index Type:</strong> {databaseDetails.indexType}
          </div>
        )}
        <form onSubmit={handleCreateVector}>
          <div className="form-group">
            <label className="form-label">
              Vector Values (comma-separated numbers)
              {newVector.values && (() => {
                const count = newVector.values.trim().split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v)).length;
                const expected = databaseDetails?.vectorDimension || 0;
                const isValid = count === expected;
                return (
                  <span className={`value-count ${isValid ? 'valid' : 'invalid'}`}>
                    {' '}[{count} / {expected} values]
                  </span>
                );
              })()}
            </label>
            <input
              type="text"
              className="form-input"
              placeholder="0.1, 0.2, 0.3, 0.4, ..."
              value={newVector.values}
              onChange={e => setNewVector({ ...newVector, values: e.target.value })}
              required
            />
          </div>
          <div className="form-group">
            <label className="form-label">Metadata (JSON format, optional)</label>
            <input
              type="text"
              className="form-input"
              placeholder='{"label": "example", "category": "test"}'
              value={newVector.metadata}
              onChange={e => setNewVector({ ...newVector, metadata: e.target.value })}
            />
          </div>
          <button type="submit" className="btn btn-primary" disabled={!selectedDatabase || loading}>
            {loading ? 'Creating...' : 'Create Vector'}
          </button>
        </form>
      </div>

      <div className="card">
        <div className="card-header-with-info">
          <h2 className="card-title">Vectors</h2>
          {totalVectors > 0 && (
            <div className="vector-count-info">
              Showing {(currentPage - 1) * pageSize + 1}-{Math.min(currentPage * pageSize, totalVectors)} of {totalVectors} vectors
            </div>
          )}
        </div>

        {loading && <div className="loading-state">Loading vectors...</div>}
        {!loading && vectors.length === 0 && selectedDatabase && (
          <div className="empty-state">No vectors found in this database</div>
        )}
        {!loading && vectors.length === 0 && !selectedDatabase && (
          <div className="empty-state">Select a database to view vectors</div>
        )}
        {!loading && vectors.length > 0 && (
          <>
            <div className="vector-list">
              {vectors.map(vector => (
                <div key={vector.id} className="vector-item">
                  <div className="vector-header">
                    <span className="vector-id">ID: {vector.id}</span>
                    <div className="vector-actions">
                      <button className="btn btn-warning" onClick={() => handleEditVector(vector)}>
                        Edit
                      </button>
                      <button className="btn btn-danger" onClick={() => handleDeleteVector(vector.id)}>
                        Delete
                      </button>
                    </div>
                  </div>
                  <div className="vector-content">
                    <div className="vector-field">
                      <div className="vector-field-label">Values ({vector.values.length} dimensions)</div>
                      <div className="vector-field-value">
                        [{vector.values.slice(0, 10).join(', ')}
                        {vector.values.length > 10 ? `, ... (${vector.values.length - 10} more)` : ''}]
                      </div>
                    </div>
                    <div className="vector-field">
                      <div className="vector-field-label">Metadata</div>
                      <div className="vector-field-value">{JSON.stringify(vector.metadata)}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {totalPages > 1 && (
              <div className="pagination">
                <button
                  className="pagination-btn"
                  onClick={() => goToPage(currentPage - 1)}
                  disabled={currentPage === 1}
                >
                  Previous
                </button>

                <div className="pagination-pages">
                  {currentPage > 2 && (
                    <>
                      <button className="pagination-page" onClick={() => goToPage(1)}>1</button>
                      {currentPage > 3 && <span className="pagination-ellipsis">...</span>}
                    </>
                  )}

                  {currentPage > 1 && (
                    <button className="pagination-page" onClick={() => goToPage(currentPage - 1)}>
                      {currentPage - 1}
                    </button>
                  )}

                  <button className="pagination-page active">{currentPage}</button>

                  {currentPage < totalPages && (
                    <button className="pagination-page" onClick={() => goToPage(currentPage + 1)}>
                      {currentPage + 1}
                    </button>
                  )}

                  {currentPage < totalPages - 1 && (
                    <>
                      {currentPage < totalPages - 2 && <span className="pagination-ellipsis">...</span>}
                      <button className="pagination-page" onClick={() => goToPage(totalPages)}>
                        {totalPages}
                      </button>
                    </>
                  )}
                </div>

                <button
                  className="pagination-btn"
                  onClick={() => goToPage(currentPage + 1)}
                  disabled={currentPage === totalPages}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {editModalOpen && (
        <div className="modal-overlay" onClick={() => setEditModalOpen(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h3 className="modal-header">Edit Vector</h3>
            <form onSubmit={handleUpdateVector}>
              <div className="form-group">
                <label className="form-label">Vector Values (comma-separated)</label>
                <input
                  type="text"
                  className="form-input"
                  value={editVector.values}
                  onChange={e => setEditVector({ ...editVector, values: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">Metadata (JSON format)</label>
                <input
                  type="text"
                  className="form-input"
                  value={editVector.metadata}
                  onChange={e => setEditVector({ ...editVector, metadata: e.target.value })}
                />
              </div>
              <div className="modal-actions">
                <button type="button" className="btn btn-secondary" onClick={() => setEditModalOpen(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Update Vector
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </Layout>
  );
}
