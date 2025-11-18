import Head from 'next/head';
import { useState, useEffect } from 'react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Select } from '../components/ui/select';
import { Alert, AlertTitle, AlertDescription } from '../components/ui/alert';
import { authApi, apiKeyApi } from '../lib/api';

export default function AuthManagement() {
  const [activeTab, setActiveTab] = useState('apikey'); // 'apikey' or 'auth'
  const [apiKeys, setApiKeys] = useState([]);
  const [newKeyName, setNewKeyName] = useState('');
  const [newKeyPermissions, setNewKeyPermissions] = useState(['read']);
  const [showApiKey, setShowApiKey] = useState(null);
  const [generatedApiKey, setGeneratedApiKey] = useState('');
  const [authStatus, setAuthStatus] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);

  // Load auth status from localStorage and fetch API keys
  useEffect(() => {
    const storedAuth = localStorage.getItem('jadevectordb_authenticated');
    const storedApiKey = localStorage.getItem('jadevectordb_api_key');
    if (storedAuth === 'true' && storedApiKey) {
      setAuthStatus(true);
      fetchApiKeys();
    }
  }, []);

  const fetchApiKeys = async () => {
    try {
      const response = await apiKeyApi.listKeys();
      const keysData = response.apiKeys || [];

      // Transform API key data
      const formattedKeys = keysData.map(k => ({
        id: k.keyId || k.id,
        name: k.name || 'Unnamed Key',
        createdAt: new Date(k.createdAt || Date.now()),
        lastUsed: k.lastUsed ? new Date(k.lastUsed) : null,
        permissions: k.permissions || ['read']
      }));

      setApiKeys(formattedKeys);
    } catch (error) {
      console.error('Error fetching API keys:', error);
      setApiKeys([]);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await authApi.login(username, password);

      // Store authentication token/API key
      if (response.token) {
        localStorage.setItem('jadevectordb_api_key', response.token);
        localStorage.setItem('jadevectordb_authenticated', 'true');
        localStorage.setItem('jadevectordb_username', username);
        setAuthStatus(true);
        fetchApiKeys();
        alert(`Successfully logged in as ${username}`);
      } else {
        alert('Login failed: No token received');
      }
    } catch (error) {
      console.error('Login error:', error);
      alert(`Login failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    setLoading(true);

    try {
      await authApi.logout();
    } catch (error) {
      console.error('Logout error:', error);
      // Continue with logout even if API call fails
    } finally {
      // Clear local storage
      localStorage.removeItem('jadevectordb_authenticated');
      localStorage.removeItem('jadevectordb_api_key');
      localStorage.removeItem('jadevectordb_username');
      setAuthStatus(false);
      setUsername('');
      setPassword('');
      setApiKeys([]);
      setLoading(false);
      alert('Logged out successfully');
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      alert('Passwords do not match');
      return;
    }

    setLoading(true);

    try {
      const response = await authApi.register(username, password);

      alert(`User ${username} registered successfully! You can now log in.`);
      setUsername('');
      setPassword('');
      setConfirmPassword('');
    } catch (error) {
      console.error('Registration error:', error);
      alert(`Registration failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateApiKey = async (e) => {
    e.preventDefault();
    if (!newKeyName) {
      alert('Please enter a name for the API key');
      return;
    }

    setLoading(true);

    try {
      const keyData = {
        name: newKeyName,
        permissions: newKeyPermissions
      };

      const response = await apiKeyApi.createKey(keyData);

      // The backend should return the generated API key
      const newApiKey = response.apiKey || response.key;
      setGeneratedApiKey(newApiKey);

      // Refresh the API keys list
      await fetchApiKeys();

      // Clear form
      setNewKeyName('');
      setNewKeyPermissions(['read']);

      // Optionally save to localStorage if this is the first/primary key
      if (!localStorage.getItem('jadevectordb_api_key')) {
        localStorage.setItem('jadevectordb_api_key', newApiKey);
      }
    } catch (error) {
      console.error('Error creating API key:', error);
      alert(`Error creating API key: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteApiKey = async (keyId) => {
    if (!window.confirm('Are you sure you want to delete this API key? This action cannot be undone.')) {
      return;
    }

    setLoading(true);

    try {
      await apiKeyApi.revokeKey(keyId);

      // Refresh the API keys list
      await fetchApiKeys();

      if (showApiKey === keyId) {
        setShowApiKey(null);
        setGeneratedApiKey('');
      }

      alert('API key revoked successfully');
    } catch (error) {
      console.error('Error revoking API key:', error);
      alert(`Error revoking API key: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handlePermissionChange = (permission) => {
    if (newKeyPermissions.includes(permission)) {
      setNewKeyPermissions(newKeyPermissions.filter(p => p !== permission));
    } else {
      setNewKeyPermissions([...newKeyPermissions, permission]);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert('API Key copied to clipboard!');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Authentication & API Keys - JadeVectorDB</title>
        <meta name="description" content="Manage authentication and API keys in JadeVectorDB" />
      </Head>

      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Authentication & API Key Management</h1>
        </div>
      </header>

      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {/* Tabs */}
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('auth')}
                className={`${
                  activeTab === 'auth'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
              >
                Authentication
              </button>
              <button
                onClick={() => setActiveTab('apikey')}
                className={`${
                  activeTab === 'apikey'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
              >
                API Keys
              </button>
            </nav>
          </div>

          {/* Authentication Tab */}
          {activeTab === 'auth' && (
            <div className="mt-6 bg-white shadow px-4 py-5 sm:p-6">
              <div className="md:grid md:grid-cols-3 md:gap-6">
                <div className="md:col-span-1">
                  <h3 className="text-lg font-medium leading-6 text-gray-900">Authentication</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    {authStatus ? 'You are currently logged in' : 'Log in to access the system'}
                  </p>
                </div>
                <div className="mt-5 md:mt-0 md:col-span-2">
                  {authStatus ? (
                    <div className="space-y-6">
                      <Alert variant="default">
                        <svg className="h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                        <AlertTitle>Authenticated</AlertTitle>
                        <AlertDescription>
                          You are successfully authenticated.
                        </AlertDescription>
                      </Alert>
                      <Button onClick={handleLogout} variant="destructive">
                        Log Out
                      </Button>
                    </div>
                  ) : (
                    <form onSubmit={handleLogin}>
                      <div className="grid grid-cols-6 gap-6">
                        <div className="col-span-6">
                          <label htmlFor="username" className="block text-sm font-medium text-gray-700">
                            Username
                          </label>
                          <Input
                            type="text"
                            name="username"
                            id="username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                            className="mt-1"
                          />
                        </div>

                        <div className="col-span-6">
                          <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                            Password
                          </label>
                          <Input
                            type="password"
                            name="password"
                            id="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            className="mt-1"
                          />
                        </div>
                      </div>

                      <div className="mt-6">
                        <Button type="submit" className="w-full">
                          Log In
                        </Button>
                      </div>
                    </form>
                  )}
                </div>
              </div>

              {/* Registration Section */}
              <div className="mt-8 md:grid md:grid-cols-3 md:gap-6">
                <div className="md:col-span-1">
                  <h3 className="text-lg font-medium leading-6 text-gray-900">Create Account</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Register a new account for the system
                  </p>
                </div>
                <div className="mt-5 md:mt-0 md:col-span-2">
                  <form onSubmit={handleRegister}>
                    <div className="grid grid-cols-6 gap-6">
                      <div className="col-span-6">
                        <label htmlFor="newUsername" className="block text-sm font-medium text-gray-700">
                          Username
                        </label>
                        <Input
                          type="text"
                          name="newUsername"
                          id="newUsername"
                          value={username}
                          onChange={(e) => setUsername(e.target.value)}
                          required
                          className="mt-1"
                        />
                      </div>

                      <div className="col-span-6">
                        <label htmlFor="newPassword" className="block text-sm font-medium text-gray-700">
                          Password
                        </label>
                        <Input
                          type="password"
                          name="newPassword"
                          id="newPassword"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          required
                          className="mt-1"
                        />
                      </div>

                      <div className="col-span-6">
                        <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                          Confirm Password
                        </label>
                        <Input
                          type="password"
                          name="confirmPassword"
                          id="confirmPassword"
                          value={confirmPassword}
                          onChange={(e) => setConfirmPassword(e.target.value)}
                          required
                          className="mt-1"
                        />
                      </div>
                    </div>

                    <div className="mt-6">
                      <Button type="submit" className="w-full">
                        Register
                      </Button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          )}

          {/* API Keys Tab */}
          {activeTab === 'apikey' && (
            <div className="mt-6 bg-white shadow px-4 py-5 sm:p-6">
              <div className="md:grid md:grid-cols-3 md:gap-6">
                <div className="md:col-span-1">
                  <h3 className="text-lg font-medium leading-6 text-gray-900">API Keys</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Manage API keys for programmatic access
                  </p>
                </div>
                <div className="mt-5 md:mt-0 md:col-span-2">
                  {/* Create API Key Form */}
                  <form onSubmit={handleCreateApiKey} className="mb-8">
                    <div className="grid grid-cols-6 gap-6">
                      <div className="col-span-6">
                        <label htmlFor="keyName" className="block text-sm font-medium text-gray-700">
                          Key Name
                        </label>
                        <Input
                          type="text"
                          name="keyName"
                          id="keyName"
                          value={newKeyName}
                          onChange={(e) => setNewKeyName(e.target.value)}
                          placeholder="e.g., Production API Key"
                          className="mt-1"
                        />
                        <p className="mt-1 text-sm text-gray-500">
                          A descriptive name for your API key
                        </p>
                      </div>

                      <div className="col-span-6">
                        <fieldset>
                          <legend className="text-sm font-medium text-gray-700">Permissions</legend>
                          <div className="mt-2 space-y-2">
                            <div className="flex items-center">
                              <input
                                id="read-permission"
                                name="permissions"
                                type="checkbox"
                                checked={newKeyPermissions.includes('read')}
                                onChange={() => handlePermissionChange('read')}
                                className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                              />
                              <label htmlFor="read-permission" className="ml-2 block text-sm text-gray-700">
                                Read
                              </label>
                            </div>
                            <div className="flex items-center">
                              <input
                                id="write-permission"
                                name="permissions"
                                type="checkbox"
                                checked={newKeyPermissions.includes('write')}
                                onChange={() => handlePermissionChange('write')}
                                className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                              />
                              <label htmlFor="write-permission" className="ml-2 block text-sm text-gray-700">
                                Write
                              </label>
                            </div>
                            <div className="flex items-center">
                              <input
                                id="delete-permission"
                                name="permissions"
                                type="checkbox"
                                checked={newKeyPermissions.includes('delete')}
                                onChange={() => handlePermissionChange('delete')}
                                className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                              />
                              <label htmlFor="delete-permission" className="ml-2 block text-sm text-gray-700">
                                Delete
                              </label>
                            </div>
                          </div>
                        </fieldset>
                      </div>
                    </div>

                    <div className="mt-6">
                      <Button type="submit" className="w-full">
                        Create API Key
                      </Button>
                    </div>
                  </form>

                  {/* Generated API Key Display */}
                  {generatedApiKey && (
                    <Alert variant="default" className="mb-6">
                      <svg className="h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                      <AlertTitle>Your new API Key</AlertTitle>
                      <AlertDescription>
                        <p className="mt-2 break-all font-mono">
                          {generatedApiKey}
                        </p>
                        <p className="mt-2">
                          Make sure to save this key, as you won't see it again.
                        </p>
                        <Button 
                          onClick={() => copyToClipboard(generatedApiKey)}
                          variant="secondary"
                          className="mt-2"
                        >
                          Copy Key
                        </Button>
                      </AlertDescription>
                    </Alert>
                  )}

                  {/* API Key List */}
                  <div>
                    <h4 className="text-md font-medium text-gray-900 mb-4">Existing API Keys</h4>
                    {apiKeys.length > 0 ? (
                      <ul className="border border-gray-200 rounded-md divide-y divide-gray-200">
                        {apiKeys.map((key) => (
                          <li key={key.id} className="pl-3 pr-4 py-3 flex items-center justify-between text-sm">
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium text-gray-900 truncate">{key.name}</p>
                              <p className="text-sm text-gray-500">
                                Created: {key.createdAt.toLocaleDateString()} 
                                {key.lastUsed && `, Last used: ${key.lastUsed.toLocaleDateString()}`}
                              </p>
                              <div className="mt-1">
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                                  Permissions: {key.permissions.join(', ')}
                                </span>
                              </div>
                            </div>
                            <div className="ml-4 flex-shrink-0 flex space-x-4">
                              <button
                                onClick={() => {
                                  if (showApiKey === key.id) {
                                    setShowApiKey(null);
                                    setGeneratedApiKey('');
                                  } else {
                                    setShowApiKey(key.id);
                                    setGeneratedApiKey('••••••••••••••••••••••••••••••••'); // Masked view
                                  }
                                }}
                                className="font-medium text-indigo-600 hover:text-indigo-900"
                              >
                                {showApiKey === key.id ? 'Hide' : 'Show'}
                              </button>
                              <button
                                onClick={() => handleDeleteApiKey(key.id)}
                                className="font-medium text-red-600 hover:text-red-900"
                              >
                                Delete
                              </button>
                            </div>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-gray-500">No API keys created yet.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}