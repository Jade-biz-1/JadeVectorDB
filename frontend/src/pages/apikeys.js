import Head from 'next/head';
import { useState, useEffect } from 'react';
import { apiKeyApi } from '../lib/api';

export default function ApiKeys() {
  const [keys, setKeys] = useState([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);

  const fetchKeys = async () => {
    setLoading(true);
    try {
      const response = await apiKeyApi.listKeys();
      setKeys(response.keys || []);
    } catch (error) {
      console.error('Error fetching API keys:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchKeys();
  }, []);

  const handleGenerate = async () => {
    setCreating(true);
    try {
      await apiKeyApi.createKey({});
      fetchKeys();
    } catch (error) {
      console.error('Error creating API key:', error);
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (keyId) => {
    setLoading(true);
    try {
      await apiKeyApi.revokeKey(keyId);
      fetchKeys();
    } catch (error) {
      console.error('Error revoking API key:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>API Key Management - JadeVectorDB</title>
        <meta name="description" content="Manage API keys for access control" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">API Key Management</h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">API Keys</h2>
            <button className="bg-green-600 text-white px-4 py-2 rounded mb-4" onClick={handleGenerate} disabled={creating}>
              {creating ? 'Generating...' : 'Generate New Key'}
            </button>
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Key ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Value</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {loading ? (
                  <tr><td colSpan={4} className="text-center py-4">Loading...</td></tr>
                ) : keys.length === 0 ? (
                  <tr><td colSpan={4} className="text-center py-4">No API keys found.</td></tr>
                ) : (
                  keys.map(key => (
                    <tr key={key.id}>
                      <td className="px-6 py-4 whitespace-nowrap">{key.id}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{key.value}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{key.status}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <button className="bg-red-600 text-white px-2 py-1 rounded" onClick={() => handleRevoke(key.id)} disabled={loading}>
                          Revoke
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
