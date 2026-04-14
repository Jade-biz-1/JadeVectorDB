function DocumentList({ documents, onDelete, onRefresh }) {
  const getStatusColor = (status) => {
    const colors = {
      complete: 'green',
      processing: 'blue',
      pending: 'orange',
      failed: 'red',
    };
    return colors[status] || 'gray';
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  if (documents.length === 0) {
    return (
      <div className="document-list empty">
        <p>No documents uploaded yet. Upload your first document above!</p>
      </div>
    );
  }

  return (
    <div className="document-list">
      <div className="list-header">
        <h3>Uploaded Documents ({documents.length})</h3>
        <button onClick={onRefresh} className="refresh-button">
          Refresh
        </button>
      </div>

      <div className="documents-grid">
        {documents.map((doc) => (
          <div key={doc.id} className="document-card">
            <div className="doc-header">
              <h4 className="doc-filename">{doc.filename}</h4>
              <span
                className="doc-status"
                style={{ backgroundColor: getStatusColor(doc.status) }}
              >
                {doc.status}
              </span>
            </div>

            <div className="doc-details">
              <div className="doc-detail">
                <span className="detail-label">Category:</span>
                <span className="detail-value">{doc.category}</span>
              </div>

              <div className="doc-detail">
                <span className="detail-label">Uploaded:</span>
                <span className="detail-value">{formatDate(doc.uploaded_at)}</span>
              </div>

              {doc.processed_at && (
                <div className="doc-detail">
                  <span className="detail-label">Processed:</span>
                  <span className="detail-value">{formatDate(doc.processed_at)}</span>
                </div>
              )}

              {doc.chunk_count !== null && (
                <div className="doc-detail">
                  <span className="detail-label">Chunks:</span>
                  <span className="detail-value">{doc.chunk_count}</span>
                </div>
              )}

              {doc.error && (
                <div className="doc-error">
                  <strong>Error:</strong> {doc.error}
                </div>
              )}
            </div>

            <div className="doc-actions">
              <button
                onClick={() => onDelete(doc.id)}
                className="delete-button"
                disabled={doc.status === 'processing'}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default DocumentList;
