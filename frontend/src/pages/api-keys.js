import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Layout from '../components/Layout';
import { apiKeysApi, authApi } from '../lib/api';
import {
  Alert, AlertDescription,
  Button,
  Card, CardHeader, CardTitle, CardDescription, CardContent,
  EmptyState,
  FormField,
  LoadingSpinner,
  StatusBadge,
} from '../components/ui';

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

  const formatDate = (value) => {
    if (!value && value !== 0) return 'N/A';
    try {
      if (typeof value === 'number') {
        if (value === 0) return 'Never';
        const ms = value < 1e12 ? value * 1000 : value;
        return new Date(ms).toLocaleString();
      }
      return new Date(value).toLocaleString();
    } catch {
      return String(value);
    }
  };

  const inputCls = 'w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-500 transition';

  if (!currentUser || !currentUser.user_id) {
    return (
      <Layout title="API Key Management - JadeVectorDB">
        <div className="flex items-center justify-center min-h-[400px]">
          <Card className="max-w-sm w-full text-center">
            <CardHeader>
              <CardTitle>Authentication Required</CardTitle>
              <CardDescription>You must be logged in to manage API keys</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={() => router.push('/')}>Go to Login</Button>
            </CardContent>
          </Card>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="API Key Management - JadeVectorDB">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-1">API Key Management</h1>
        <p className="text-gray-500">Generate and manage your API keys for programmatic access</p>
      </div>

      <Alert className="mb-6 bg-blue-50 border-blue-200 text-blue-800">
        <AlertDescription>
          <strong>Logged in as:</strong> {currentUser.username} (User ID: {currentUser.user_id})
        </AlertDescription>
      </Alert>

      {error && (
        <Alert variant="destructive" className="mb-6 bg-red-50 border-red-200 text-red-800">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {success && (
        <Alert className="mb-6 bg-green-50 border-green-200 text-green-800">
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      {/* ── Generated key display ── */}
      {generatedKey && (
        <Card className="mb-6 border-2 border-green-400">
          <CardHeader>
            <CardTitle className="text-green-700">New API Key Created</CardTitle>
            <CardDescription>Save this key securely - you won't be able to see it again!</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-gray-50 rounded-lg px-4 py-3 font-mono text-sm break-all mb-4">
              {generatedKey.api_key}
            </div>
            <Button onClick={() => copyToClipboard(generatedKey.api_key)}>
              Copy to Clipboard
            </Button>
          </CardContent>
        </Card>
      )}

      {/* ── Create form ── */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-xl">Create New API Key</CardTitle>
          <CardDescription>Generate a new API key for programmatic access</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleCreateKey}>
            <div className="space-y-4">
              <FormField label="Description" htmlFor="description">
                <input
                  id="description"
                  name="description"
                  type="text"
                  className={inputCls}
                  placeholder="e.g., Production API key"
                  value={newKeyData.description}
                  onChange={handleInputChange}
                  disabled={creating}
                />
              </FormField>

              <FormField
                label="Permissions (comma-separated)"
                htmlFor="permissions"
                hint="Optional. Leave empty for default permissions."
              >
                <input
                  id="permissions"
                  name="permissions"
                  type="text"
                  className={inputCls}
                  placeholder="e.g., read, write, delete"
                  value={newKeyData.permissions}
                  onChange={handleInputChange}
                  disabled={creating}
                />
              </FormField>

              <FormField label="Validity Period (days)" htmlFor="validity_days">
                <input
                  id="validity_days"
                  name="validity_days"
                  type="number"
                  className={inputCls}
                  min="1"
                  max="365"
                  value={newKeyData.validity_days}
                  onChange={handleInputChange}
                  disabled={creating}
                />
              </FormField>

              <Button type="submit" disabled={creating}>
                {creating ? 'Creating…' : 'Create API Key'}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* ── API key list ── */}
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">Your API Keys</CardTitle>
          <CardDescription>Manage your existing API keys</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <LoadingSpinner label="Loading API keys…" />
          ) : apiKeys.length === 0 ? (
            <EmptyState
              icon="🔑"
              title="No API keys found"
              description="Create one above to get started"
            />
          ) : (
            <div className="space-y-4">
              {apiKeys.map((key) => (
                <div key={key.key_id} className="border border-gray-200 rounded-xl p-5 flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-3">
                      <h3 className="text-base font-semibold text-gray-900">
                        {key.description || 'Unnamed Key'}
                      </h3>
                      <StatusBadge status={key.is_active ? 'active' : 'inactive'} label={key.is_active ? 'Active' : 'Revoked'} />
                    </div>
                    <div className="space-y-1 text-sm text-gray-600">
                      <div>
                        <span className="text-gray-400 mr-2">Key ID:</span>
                        <span className="font-mono">{key.key_id}</span>
                      </div>
                      <div>
                        <span className="text-gray-400 mr-2">Created:</span>
                        {formatDate(key.created_at)}
                      </div>
                      <div>
                        <span className="text-gray-400 mr-2">Expires:</span>
                        {formatDate(key.expires_at)}
                      </div>
                      {key.permissions && key.permissions.length > 0 && (
                        <div>
                          <span className="text-gray-400 mr-2">Permissions:</span>
                          {key.permissions.join(', ')}
                        </div>
                      )}
                    </div>
                  </div>
                  {key.is_active && (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => handleRevokeKey(key.key_id)}
                    >
                      Revoke
                    </Button>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </Layout>
  );
}
