import { useState, useEffect } from 'react';
import { adminAPI } from '../services/api';
import DocumentUpload from '../components/DocumentUpload';
import DocumentList from '../components/DocumentList';
import '../styles/AdminPage.css';

function AdminPage() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await adminAPI.listDocuments();
      setDocuments(data.documents);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const handleUploadSuccess = () => {
    loadDocuments();
  };

  const handleDelete = async (docId) => {
    if (!confirm('Are you sure you want to delete this document? This will remove all associated vectors.')) {
      return;
    }

    try {
      await adminAPI.deleteDocument(docId);
      loadDocuments();
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to delete document');
    }
  };

  return (
    <div className="admin-page">
      <div className="admin-container">
        <header className="page-header">
          <h2>Document Management</h2>
          <p>Upload and manage maintenance documentation</p>
        </header>

        <DocumentUpload onSuccess={handleUploadSuccess} />

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {loading ? (
          <div className="loading-message">
            <div className="spinner"></div>
            <p>Loading documents...</p>
          </div>
        ) : (
          <DocumentList
            documents={documents}
            onDelete={handleDelete}
            onRefresh={loadDocuments}
          />
        )}
      </div>
    </div>
  );
}

export default AdminPage;
