import Head from 'next/head';
import { useState, useEffect } from 'react';
import { databaseApi, lifecycleApi } from '../lib/api';

export default function LifecycleManagement() {
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [retentionPolicy, setRetentionPolicy] = useState({
    maxAgeDays: 30,
    archiveOnExpire: false,
    deleteOnExpire: false,
    autoScale: false
  });
  const [currentPolicy, setCurrentPolicy] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(false);
  const [message, setMessage] = useState('');

  // Fetch databases on component mount
  useEffect(() => {
    fetchDatabases();
  }, []);

  // Fetch current policy when database is selected
  useEffect(() => {
    if (selectedDatabase) {
      fetchCurrentPolicy();
    } else {
      setCurrentPolicy(null);
    }
  }, [selectedDatabase]);

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      setDatabases(response.databases.map(db => ({
        id: db.databaseId,
        name: db.name
      })));
    } catch (error) {
      console.error('Error fetching databases:', error);
      alert(`Error fetching databases: ${error.message}`);
    }
  };

  const fetchCurrentPolicy = async () => {
    setFetching(true);
    try {
      // In a real implementation, this would fetch the current lifecycle policy
      // For now, we'll simulate with an API call that might retrieve the policy
      const response = await lifecycleApi.lifecycleStatus(selectedDatabase);
      setCurrentPolicy(response);
      if (response.retentionPolicy) {
        setRetentionPolicy(response.retentionPolicy);
      }
    } catch (error) {
      console.error('Error fetching lifecycle status:', error);
      // Set default values if no policy exists yet
      setRetentionPolicy({
        maxAgeDays: 30,
        archiveOnExpire: false,
        deleteOnExpire: false,
        autoScale: false
      });
      setCurrentPolicy(null);
    } finally {
      setFetching(false);
    }
  };

  const handlePolicyChange = (e) => {
    const { name, value, type, checked } = e.target;
    setRetentionPolicy(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSavePolicy = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage('');

    try {
      // Validate inputs
      if (retentionPolicy.maxAgeDays < 1) {
        throw new Error('Maximum age must be at least 1 day');
      }

      if (retentionPolicy.archiveOnExpire && retentionPolicy.deleteOnExpire) {
        throw new Error('Cannot archive and delete on expiration simultaneously');
      }

      // In a real implementation, we would save the policy
      // For now, simulating an API call to configure retention
      await lifecycleApi.configureRetention(selectedDatabase, retentionPolicy);
      
      setMessage({
        type: 'success',
        text: 'Retention policy saved successfully'
      });

      // Refresh the policy display
      fetchCurrentPolicy();
    } catch (error) {
      console.error('Error saving retention policy:', error);
      setMessage({
        type: 'error',
        text: `Error saving retention policy: ${error.message}`
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Lifecycle Management - JadeVectorDB</title>
        <meta name="description" content="Manage data lifecycle in JadeVectorDB" />
      </Head>

      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Lifecycle Management</h1>
        </div>
      </header>

      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {/* Database Selection */}
          <div className="bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6 mb-8">
            <div className="md:grid md:grid-cols-3 md:gap-6">
              <div className="md:col-span-1">
                <h3 className="text-lg font-medium leading-6 text-gray-900">Select Database</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Choose a database to configure its lifecycle settings.
                </p>
              </div>
              <div className="mt-5 md:mt-0 md:col-span-2">
                <select
                  value={selectedDatabase}
                  onChange={(e) => setSelectedDatabase(e.target.value)}
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                >
                  <option value="">Select a database</option>
                  {databases.map((db) => (
                    <option key={db.id} value={db.id}>{db.name}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {selectedDatabase && (
            <>
              {/* Retention Policy Form */}
              <div className="bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6 mb-8">
                <div className="md:grid md:grid-cols-3 md:gap-6">
                  <div className="md:col-span-1">
                    <h3 className="text-lg font-medium leading-6 text-gray-900">Retention Policy</h3>
                    <p className="mt-1 text-sm text-gray-500">
                      Configure how long data should be retained in the database.
                    </p>
                  </div>
                  <div className="mt-5 md:mt-0 md:col-span-2">
                    <form onSubmit={handleSavePolicy}>
                      <div className="grid grid-cols-6 gap-6">
                        <div className="col-span-6">
                          <label htmlFor="maxAgeDays" className="block text-sm font-medium text-gray-700">
                            Maximum Age (Days)
                          </label>
                          <input
                            type="number"
                            name="maxAgeDays"
                            id="maxAgeDays"
                            min="1"
                            value={retentionPolicy.maxAgeDays}
                            onChange={handlePolicyChange}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                          />
                          <p className="mt-1 text-sm text-gray-500">
                            Maximum number of days to retain data before archiving or deletion
                          </p>
                        </div>

                        <div className="col-span-6">
                          <div className="flex items-start">
                            <div className="flex items-center h-5">
                              <input
                                id="archiveOnExpire"
                                name="archiveOnExpire"
                                type="checkbox"
                                checked={retentionPolicy.archiveOnExpire}
                                onChange={handlePolicyChange}
                                className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded"
                              />
                            </div>
                            <div className="ml-3 text-sm">
                              <label htmlFor="archiveOnExpire" className="font-medium text-gray-700">
                                Archive on expiration
                              </label>
                              <p className="text-gray-500">
                                Move expired data to long-term storage instead of deleting
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="col-span-6">
                          <div className="flex items-start">
                            <div className="flex items-center h-5">
                              <input
                                id="deleteOnExpire"
                                name="deleteOnExpire"
                                type="checkbox"
                                checked={retentionPolicy.deleteOnExpire}
                                onChange={handlePolicyChange}
                                className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded"
                              />
                            </div>
                            <div className="ml-3 text-sm">
                              <label htmlFor="deleteOnExpire" className="font-medium text-gray-700">
                                Delete on expiration
                              </label>
                              <p className="text-gray-500">
                                Permanently delete expired data (cannot be recovered)
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="col-span-6">
                          <div className="flex items-start">
                            <div className="flex items-center h-5">
                              <input
                                id="autoScale"
                                name="autoScale"
                                type="checkbox"
                                checked={retentionPolicy.autoScale}
                                onChange={handlePolicyChange}
                                className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded"
                              />
                            </div>
                            <div className="ml-3 text-sm">
                              <label htmlFor="autoScale" className="font-medium text-gray-700">
                                Auto-scale on retention changes
                              </label>
                              <p className="text-gray-500">
                                Automatically adjust resources based on data retention
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="mt-6">
                        <button
                          type="submit"
                          disabled={loading}
                          className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                        >
                          {loading ? 'Saving...' : 'Save Retention Policy'}
                        </button>
                      </div>
                    </form>
                  </div>
                </div>
              </div>

              {/* Lifecycle Status */}
              <div className="bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6 mb-8">
                <div className="md:grid md:grid-cols-3 md:gap-6">
                  <div className="md:col-span-1">
                    <h3 className="text-lg font-medium leading-6 text-gray-900">Lifecycle Status</h3>
                    <p className="mt-1 text-sm text-gray-500">
                      Current lifecycle settings for the database.
                    </p>
                  </div>
                  <div className="mt-5 md:mt-0 md:col-span-2">
                    {fetching ? (
                      <p className="text-gray-500">Loading lifecycle status...</p>
                    ) : currentPolicy ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="text-sm font-medium text-gray-500">Max Age</p>
                            <p className="text-sm text-gray-900">{currentPolicy.retentionPolicy?.maxAgeDays || retentionPolicy.maxAgeDays} days</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-500">Archive on Expire</p>
                            <p className="text-sm text-gray-900">{currentPolicy.retentionPolicy?.archiveOnExpire ? 'Yes' : 'No'}</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-500">Delete on Expire</p>
                            <p className="text-sm text-gray-900">{currentPolicy.retentionPolicy?.deleteOnExpire ? 'Yes' : 'No'}</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-500">Auto Scale</p>
                            <p className="text-sm text-gray-900">{currentPolicy.retentionPolicy?.autoScale ? 'Yes' : 'No'}</p>
                          </div>
                        </div>
                        
                        <div className="mt-6">
                          <h4 className="text-md font-medium text-gray-900">Archival Information</h4>
                          <div className="mt-2 grid grid-cols-2 gap-4">
                            <div>
                              <p className="text-sm font-medium text-gray-500">Next Archival Run</p>
                              <p className="text-sm text-gray-900">
                                {new Date(Date.now() + 24 * 60 * 60 * 1000).toLocaleString()} {/* Tomorrow */}
                              </p>
                            </div>
                            <div>
                              <p className="text-sm font-medium text-gray-500">Archived Vectors</p>
                              <p className="text-sm text-gray-900">{currentPolicy.archivedCount || 0}</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-gray-500">No lifecycle policy configured yet.</p>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Message Display */}
              {message && (
                <div className={`rounded-md p-4 mb-6 ${
                  message.type === 'success' ? 'bg-green-50' : 'bg-red-50'
                }`}>
                  <div className="flex">
                    <div className="flex-shrink-0">
                      {message.type === 'success' ? (
                        <svg className="h-5 w-5 text-green-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      ) : (
                        <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                      )}
                    </div>
                    <div className="ml-3">
                      <h3 className={`text-sm font-medium ${
                        message.type === 'success' ? 'text-green-800' : 'text-red-800'
                      }`}>
                        {message.text}
                      </h3>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}