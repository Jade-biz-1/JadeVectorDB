import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Layout from '../components/Layout';
import { apiKeysApi, authApi } from '../lib/api';

export default function ApiKeyManagement() {
  const router = useRouter();
  const [apiKeys, setApiKeys] = useState([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [newKeyData, setNewKeyData] = useState({
    description: '',
    permissions: '',
    validity_days: 30
  });
  const [generatedKey, setGeneratedKey] = useState(null);
  const [currentUser, setCurrentUser] = useState(null);

  useEffect(() => {
    // Get current user
    if (typeof window !== 'undefined') {
      const user = authApi.getCurrentUser();
      setCurrentUser(user);
      if (user && user.user_id) {
        fetchApiKeys(user.user_id);
      }
    }
  }, []);

  const fetchApiKeys = async (userId) => {
    setLoading(true);
    setError('');
    try {
      const response = await apiKeysApi.listApiKeys(userId);
      setApiKeys(response.api_keys || []);
    } catch (err) {
      console.error('Error fetching API keys:', err);
      setError(`Error fetching API keys: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewKeyData({ ...newKeyData, [name]: value });
  };

  const handleCreateKey = async (e) => {
    e.preventDefault();
    setCreating(true);
    setError('');
    setSuccess('');
    setGeneratedKey(null);

    try {
      if (!currentUser || !currentUser.user_id) {
        setError('You must be logged in to create API keys');
        return;
      }

      const permissionsArray = newKeyData.permissions
        ? newKeyData.permissions.split(',').map(p => p.trim())
        : [];

      const response = await apiKeysApi.createApiKey(
        currentUser.user_id,
        permissionsArray,
        newKeyData.description,
        parseInt(newKeyData.validity_days)
      );

      setGeneratedKey(response);
      setSuccess('API key created successfully! Save it now - you won\'t be able to see it again.');
      setNewKeyData({ description: '', permissions: '', validity_days: 30 });

      // Refresh the list
      fetchApiKeys(currentUser.user_id);
    } catch (err) {
      console.error('Error creating API key:', err);
      setError(`Error creating API key: ${err.message}`);
    } finally {
      setCreating(false);
    }
  };

  const handleRevokeKey = async (keyId) => {
    if (!window.confirm('Are you sure you want to revoke this API key? This action cannot be undone.')) {
      return;
    }

    setError('');
    setSuccess('');

    try {
      await apiKeysApi.revokeApiKey(keyId);
      setSuccess('API key revoked successfully!');

      if (currentUser && currentUser.user_id) {
        fetchApiKeys(currentUser.user_id);
      }
    } catch (err) {
      console.error('Error revoking API key:', err);
      setError(`Error revoking API key: ${err.message}`);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setSuccess('Copied to clipboard!');
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  if (!currentUser || !currentUser.user_id) {
    return (
      <Layout title="API Key Management - JadeVectorDB">
        <style jsx>{`
          .auth-required {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 400px;
          }

          .auth-card {
            background: white;
            border-radius: 8px;
            padding: 40px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 400px;
          }

          .auth-title {
            font-size: 24px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
          }

          .auth-description {
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 25px;
          }

          .btn-login {
            padding: 12px 24px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
          }

          .btn-login:hover {
            background: #2980b9;
          }
        `}</style>
        <div className="auth-required">
          <div className="auth-card">
            <h2 className="auth-title">Authentication Required</h2>
            <p className="auth-description">You must be logged in to manage API keys</p>
            <button onClick={() => router.push('/')} className="btn-login">
              Go to Login
            </button>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="API Key Management - JadeVectorDB">
      <style jsx>{`
        .apikeys-container {
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

        .user-info {
          background: #e3f2fd;
          border: 1px solid #90caf9;
          border-radius: 8px;
          padding: 15px;
          margin-bottom: 20px;
          font-size: 14px;
          color: #1565c0;
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

        .alert-success {
          background: #dcfce7;
          border: 1px solid #bbf7d0;
          color: #166534;
        }

        .card {
          background: white;
          border-radius: 8px;
          padding: 30px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          margin-bottom: 30px;
        }

        .card.highlight {
          border: 2px solid #27ae60;
        }

        .card-title {
          font-size: 20px;
          font-weight: 600;
          color: #2c3e50;
          margin: 0 0 10px 0;
        }

        .card-title.success {
          color: #27ae60;
        }

        .card-subtitle {
          font-size: 14px;
          color: #7f8c8d;
          margin-bottom: 25px;
        }

        .generated-key-display {
          background: #f8f9fa;
          padding: 15px;
          border-radius: 6px;
          font-family: monospace;
          font-size: 13px;
          word-break: break-all;
          margin-bottom: 15px;
        }

        .form-group {
          display: flex;
          flex-direction: column;
          margin-bottom: 20px;
        }

        .form-label {
          font-weight: 500;
          color: #2c3e50;
          margin-bottom: 8px;
          font-size: 14px;
        }

        .form-input {
          padding: 10px 12px;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          font-size: 14px;
          transition: all 0.2s;
        }

        .form-input:focus {
          outline: none;
          border-color: #3498db;
          box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }

        .form-hint {
          font-size: 12px;
          color: #7f8c8d;
          margin-top: 5px;
        }

        .btn {
          padding: 10px 20px;
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

        .btn-danger {
          background: #e74c3c;
          color: white;
          padding: 6px 12px;
          font-size: 12px;
        }

        .btn-danger:hover:not(:disabled) {
          background: #c0392b;
        }

        .keys-list {
          display: flex;
          flex-direction: column;
          gap: 15px;
        }

        .key-item {
          border: 1px solid #e1e8ed;
          border-radius: 8px;
          padding: 20px;
          display: flex;
          justify-content: space-between;
          align-items: start;
        }

        .key-info {
          flex: 1;
        }

        .key-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 15px;
        }

        .key-title {
          font-size: 16px;
          font-weight: 600;
          color: #2c3e50;
        }

        .badge {
          display: inline-flex;
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 600;
        }

        .badge-success {
          background: #d4edda;
          color: #155724;
        }

        .badge-error {
          background: #f8d7da;
          color: #721c24;
        }

        .key-details {
          display: flex;
          flex-direction: column;
          gap: 5px;
          font-size: 13px;
        }

        .detail-row {
          color: #555;
        }

        .detail-label {
          color: #7f8c8d;
          margin-right: 8px;
        }

        .detail-value {
          font-family: monospace;
        }

        .empty-state {
          text-align: center;
          padding: 40px;
          color: #7f8c8d;
        }

        .loading-state {
          text-align: center;
          padding: 40px;
          color: #7f8c8d;
        }
      `}</style>

      <div className="apikeys-container">
        <div className="page-header">
          <h1 className="page-title">API Key Management</h1>
          <p className="page-description">Generate and manage your API keys for programmatic access</p>
        </div>

        <div className="user-info">
          <strong>Logged in as:</strong> {currentUser.username} (User ID: {currentUser.user_id})
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

        {generatedKey && (
          <div className="card highlight">
            <h2 className="card-title success">New API Key Created</h2>
            <p className="card-subtitle">
              Save this key securely - you won't be able to see it again!
            </p>
            <div className="generated-key-display">
              {generatedKey.api_key}
            </div>
            <button onClick={() => copyToClipboard(generatedKey.api_key)} className="btn btn-primary">
              Copy to Clipboard
            </button>
          </div>
        )}

        <div className="card">
          <h2 className="card-title">Create New API Key</h2>
          <p className="card-subtitle">Generate a new API key for programmatic access</p>

          <form onSubmit={handleCreateKey}>
            <div className="form-group">
              <label htmlFor="description" className="form-label">Description</label>
              <input
                id="description"
                name="description"
                type="text"
                className="form-input"
                placeholder="e.g., Production API key"
                value={newKeyData.description}
                onChange={handleInputChange}
                disabled={creating}
              />
            </div>

            <div className="form-group">
              <label htmlFor="permissions" className="form-label">Permissions (comma-separated)</label>
              <input
                id="permissions"
                name="permissions"
                type="text"
                className="form-input"
                placeholder="e.g., read, write, delete"
                value={newKeyData.permissions}
                onChange={handleInputChange}
                disabled={creating}
              />
              <p className="form-hint">Optional. Leave empty for default permissions.</p>
            </div>

            <div className="form-group">
              <label htmlFor="validity_days" className="form-label">Validity Period (days)</label>
              <input
                id="validity_days"
                name="validity_days"
                type="number"
                className="form-input"
                min="1"
                max="365"
                value={newKeyData.validity_days}
                onChange={handleInputChange}
                disabled={creating}
              />
            </div>

            <button type="submit" disabled={creating} className="btn btn-primary">
              {creating ? 'Creating...' : 'Create API Key'}
            </button>
          </form>
        </div>

        <div className="card">
          <h2 className="card-title">Your API Keys</h2>
          <p className="card-subtitle">Manage your existing API keys</p>

          {loading ? (
            <div className="loading-state">Loading API keys...</div>
          ) : apiKeys.length === 0 ? (
            <div className="empty-state">No API keys found. Create one above to get started.</div>
          ) : (
            <div className="keys-list">
              {apiKeys.map((key) => (
                <div key={key.key_id} className="key-item">
                  <div className="key-info">
                    <div className="key-header">
                      <h3 className="key-title">
                        {key.description || 'Unnamed Key'}
                      </h3>
                      <span className={`badge ${key.is_active ? 'badge-success' : 'badge-error'}`}>
                        {key.is_active ? 'Active' : 'Revoked'}
                      </span>
                    </div>
                    <div className="key-details">
                      <div className="detail-row">
                        <span className="detail-label">Key ID:</span>
                        <span className="detail-value">{key.key_id}</span>
                      </div>
                      <div className="detail-row">
                        <span className="detail-label">Created:</span>
                        <span>{formatDate(key.created_at)}</span>
                      </div>
                      <div className="detail-row">
                        <span className="detail-label">Expires:</span>
                        <span>{formatDate(key.expires_at)}</span>
                      </div>
                      {key.permissions && key.permissions.length > 0 && (
                        <div className="detail-row">
                          <span className="detail-label">Permissions:</span>
                          <span>{key.permissions.join(', ')}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  {key.is_active && (
                    <button
                      onClick={() => handleRevokeKey(key.key_id)}
                      className="btn btn-danger"
                    >
                      Revoke
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
