import Head from 'next/head';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Alert, AlertTitle, AlertDescription } from '../components/ui/alert';
import { apiKeysApi, authApi } from '../lib/api';

export default function ApiKeyManagement() {
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
    const user = authApi.getCurrentUser();
    setCurrentUser(user);
    if (user.user_id) {
      fetchApiKeys(user.user_id);
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
    alert('Copied to clipboard!');
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
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <Card className="max-w-md">
          <CardHeader>
            <CardTitle>Authentication Required</CardTitle>
            <CardDescription>You must be logged in to manage API keys</CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/login">
              <Button className="w-full">Go to Login</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>API Key Management - JadeVectorDB</title>
        <meta name="description" content="Manage your JadeVectorDB API keys" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        <header className="bg-white shadow">
          <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center">
              <h1 className="text-3xl font-bold text-gray-900">API Key Management</h1>
              <Link href="/dashboard">
                <Button variant="outline">← Dashboard</Button>
              </Link>
            </div>
          </div>
        </header>

        <main>
          <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            {/* User Info */}
            <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
              <p className="text-sm text-blue-800">
                <strong>Logged in as:</strong> {currentUser.username} (User ID: {currentUser.user_id})
              </p>
            </div>

            {/* Status Messages */}
            {error && (
              <Alert variant="destructive" className="mb-6">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {success && (
              <Alert variant="success" className="mb-6">
                <AlertTitle>Success</AlertTitle>
                <AlertDescription>{success}</AlertDescription>
              </Alert>
            )}

            {/* Generated Key Display */}
            {generatedKey && (
              <Card className="mb-6 border-green-500">
                <CardHeader>
                  <CardTitle className="text-green-700">New API Key Created</CardTitle>
                  <CardDescription>
                    Save this key securely - you won't be able to see it again!
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="bg-gray-100 p-3 rounded font-mono text-sm break-all">
                    {generatedKey.api_key}
                  </div>
                  <Button
                    onClick={() => copyToClipboard(generatedKey.api_key)}
                    className="mt-3"
                  >
                    Copy to Clipboard
                  </Button>
                </CardContent>
              </Card>
            )}

            {/* Create New API Key */}
            <Card className="mb-8">
              <CardHeader>
                <CardTitle>Create New API Key</CardTitle>
                <CardDescription>
                  Generate a new API key for programmatic access
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleCreateKey} className="space-y-4">
                  <div>
                    <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
                      Description
                    </label>
                    <Input
                      id="description"
                      name="description"
                      type="text"
                      placeholder="e.g., Production API key"
                      value={newKeyData.description}
                      onChange={handleInputChange}
                      disabled={creating}
                    />
                  </div>

                  <div>
                    <label htmlFor="permissions" className="block text-sm font-medium text-gray-700 mb-1">
                      Permissions (comma-separated)
                    </label>
                    <Input
                      id="permissions"
                      name="permissions"
                      type="text"
                      placeholder="e.g., read, write, delete"
                      value={newKeyData.permissions}
                      onChange={handleInputChange}
                      disabled={creating}
                    />
                    <p className="text-xs text-gray-500 mt-1">Optional. Leave empty for default permissions.</p>
                  </div>

                  <div>
                    <label htmlFor="validity_days" className="block text-sm font-medium text-gray-700 mb-1">
                      Validity Period (days)
                    </label>
                    <Input
                      id="validity_days"
                      name="validity_days"
                      type="number"
                      min="1"
                      max="365"
                      value={newKeyData.validity_days}
                      onChange={handleInputChange}
                      disabled={creating}
                    />
                  </div>

                  <Button type="submit" disabled={creating}>
                    {creating ? 'Creating...' : 'Create API Key'}
                  </Button>
                </form>
              </CardContent>
            </Card>

            {/* Existing API Keys */}
            <Card>
              <CardHeader>
                <CardTitle>Your API Keys</CardTitle>
                <CardDescription>
                  Manage your existing API keys
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <p className="text-gray-500">Loading...</p>
                ) : apiKeys.length === 0 ? (
                  <p className="text-gray-500">No API keys found. Create one above to get started.</p>
                ) : (
                  <div className="space-y-4">
                    {apiKeys.map((key) => (
                      <div
                        key={key.key_id}
                        className="border rounded-lg p-4 flex justify-between items-start"
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h3 className="font-semibold">
                              {key.description || 'Unnamed Key'}
                            </h3>
                            <span className={`px-2 py-1 rounded text-xs ${
                              key.is_active
                                ? 'bg-green-100 text-green-800'
                                : 'bg-red-100 text-red-800'
                            }`}>
                              {key.is_active ? 'Active' : 'Revoked'}
                            </span>
                          </div>
                          <dl className="text-sm space-y-1">
                            <div>
                              <dt className="inline text-gray-600">Key ID:</dt>
                              <dd className="inline ml-2 font-mono">{key.key_id}</dd>
                            </div>
                            <div>
                              <dt className="inline text-gray-600">Created:</dt>
                              <dd className="inline ml-2">{formatDate(key.created_at)}</dd>
                            </div>
                            <div>
                              <dt className="inline text-gray-600">Expires:</dt>
                              <dd className="inline ml-2">{formatDate(key.expires_at)}</dd>
                            </div>
                            {key.permissions && key.permissions.length > 0 && (
                              <div>
                                <dt className="inline text-gray-600">Permissions:</dt>
                                <dd className="inline ml-2">{key.permissions.join(', ')}</dd>
                              </div>
                            )}
                          </dl>
                        </div>
                        {key.is_active && (
                          <Button
                            onClick={() => handleRevokeKey(key.key_id)}
                            variant="destructive"
                            size="sm"
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
          </div>
        </main>
      </div>
    </>
  );
}
