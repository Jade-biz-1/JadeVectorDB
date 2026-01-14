import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { indexApi, databaseApi } from '../lib/api';

export default function IndexManagement() {
  const [databases, setDatabases] = useState([]);
  const [indexes, setIndexes] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(false);
  
  // Form state for creating new indexes
  const [newIndex, setNewIndex] = useState({
    type: 'HNSW',
    parameters: {},
  });
  
  // Fetch all databases on component mount
  useEffect(() => {
    fetchDatabases();
  }, []);

  // Fetch indexes when a database is selected
  useEffect(() => {
    if (selectedDatabase) {
      fetchIndexes();
    } else {
      setIndexes([]);
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

  const fetchIndexes = async () => {
    setFetching(true);
    try {
      const response = await indexApi.listIndexes(selectedDatabase);
      setIndexes(response.indexes);
    } catch (error) {
      console.error('Error fetching indexes:', error);
      alert(`Error fetching indexes: ${error.message}`);
    } finally {
      setFetching(false);
    }
  };

  const handleCreateIndex = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const indexData = {
        type: newIndex.type,
        parameters: newIndex.parameters
      };
      
      await indexApi.createIndex(selectedDatabase, indexData);
      setNewIndex({ type: 'HNSW', parameters: {} });
      fetchIndexes(); // Refresh the list
    } catch (error) {
      console.error('Error creating index:', error);
      alert(`Error creating index: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteIndex = async (indexId) => {
    if (!window.confirm('Are you sure you want to delete this index? This action cannot be undone.')) {
      return;
    }
    
    try {
      await indexApi.deleteIndex(selectedDatabase, indexId);
      fetchIndexes(); // Refresh the list
    } catch (error) {
      console.error('Error deleting index:', error);
      alert(`Error deleting index: ${error.message}`);
    }
  };

  // Format parameters for display
  const formatParameters = (params) => {
    if (!params || typeof params !== 'object') return 'None';
    
    try {
      return JSON.stringify(params, null, 2);
    } catch (e) {
      return 'Invalid parameters';
    }
  };

  return (
    <Layout title="Index Management - JadeVectorDB">
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {/* Database Selection */}
          <div className="bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6 mb-8">
            <div className="md:grid md:grid-cols-3 md:gap-6">
              <div className="md:col-span-1">
                <h3 className="text-lg font-medium leading-6 text-gray-900">Select Database</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Choose a database to manage its indexes.
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
              {/* Create Index Form */}
              <div className="bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6 mb-8">
                <div className="md:grid md:grid-cols-3 md:gap-6">
                  <div className="md:col-span-1">
                    <h3 className="text-lg font-medium leading-6 text-gray-900">Create Index</h3>
                    <p className="mt-1 text-sm text-gray-500">
                      Create a new index for the selected database.
                    </p>
                  </div>
                  <div className="mt-5 md:mt-0 md:col-span-2">
                    <form onSubmit={handleCreateIndex}>
                      <div className="grid grid-cols-6 gap-6">
                        <div className="col-span-6 sm:col-span-3">
                          <label htmlFor="indexType" className="block text-sm font-medium text-gray-700">
                            Index Type
                          </label>
                          <select
                            id="indexType"
                            name="indexType"
                            value={newIndex.type}
                            onChange={(e) => setNewIndex({...newIndex, type: e.target.value})}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                          >
                            <option value="FLAT">FLAT (Linear)</option>
                            <option value="HNSW">HNSW (Hierarchical Navigable Small World)</option>
                            <option value="IVF">IVF (Inverted File)</option>
                            <option value="LSH">LSH (Locality Sensitive Hashing)</option>
                          </select>
                        </div>

                        <div className="col-span-6">
                          <label htmlFor="parameters" className="block text-sm font-medium text-gray-700">
                            Parameters (JSON)
                          </label>
                          <textarea
                            id="parameters"
                            name="parameters"
                            rows={4}
                            value={JSON.stringify(newIndex.parameters, null, 2)}
                            onChange={(e) => {
                              try {
                                const params = JSON.parse(e.target.value);
                                setNewIndex({...newIndex, parameters: params});
                              } catch (error) {
                                // Handle JSON parsing error gracefully
                              }
                            }}
                            placeholder='{"M": 16, "ef_construction": 200}'
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm font-mono text-sm"
                          />
                          <p className="mt-1 text-sm text-gray-500">
                            Index-specific configuration parameters in JSON format
                          </p>
                        </div>
                      </div>

                      <div className="mt-6">
                        <button
                          type="submit"
                          disabled={loading}
                          className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                        >
                          {loading ? 'Creating...' : 'Create Index'}
                        </button>
                      </div>
                    </form>
                  </div>
                </div>
              </div>

              {/* Index List */}
              <div className="bg-white shadow overflow-hidden sm:rounded-md">
                <div className="px-4 py-5 border-b border-gray-200 sm:px-6">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">Indexes</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Indexes in database: {databases.find(db => db.id === selectedDatabase)?.name}
                  </p>
                </div>
                
                {fetching ? (
                  <div className="px-4 py-5 sm:px-6 text-center">
                    <p className="text-gray-500">Loading indexes...</p>
                  </div>
                ) : indexes.length === 0 ? (
                  <div className="px-4 py-5 sm:px-6 text-center">
                    <p className="text-gray-500">No indexes found in this database</p>
                  </div>
                ) : (
                  <ul className="divide-y divide-gray-200">
                    {indexes.map((index) => (
                      <li key={index.indexId} className="bg-white hover:bg-gray-50">
                        <div className="px-4 py-4 sm:px-6">
                          <div className="flex items-center justify-between">
                            <div className="text-sm font-medium text-indigo-600 truncate">{index.type} Index</div>
                            <div className="ml-2 flex-shrink-0 flex">
                              <span className={`inline-flex px-2 text-xs leading-5 font-semibold rounded-full ${
                                index.status === 'ready' 
                                  ? 'bg-green-100 text-green-800' 
                                  : index.status === 'building'
                                    ? 'bg-yellow-100 text-yellow-800'
                                    : 'bg-red-100 text-red-800'
                              }`}>
                                {index.status}
                              </span>
                            </div>
                          </div>
                          <div className="mt-2 sm:flex sm:justify-between">
                            <div className="sm:flex">
                              <div className="mr-6 text-sm text-gray-500">
                                <span className="font-medium">ID:</span> {index.indexId}
                              </div>
                              <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                                <span className="font-medium">Vectors:</span> {index.stats?.vectorCount || 0}
                              </div>
                            </div>
                            <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                              <span className="font-medium">Size:</span> {(index.stats?.sizeBytes ? (index.stats.sizeBytes / (1024 * 1024)).toFixed(2) : 0)} MB
                            </div>
                          </div>
                          {index.parameters && Object.keys(index.parameters).length > 0 && (
                            <div className="mt-2">
                              <div className="text-sm text-gray-900">
                                <span className="font-medium">Parameters:</span>
                              </div>
                              <pre className="mt-1 text-xs text-gray-500 bg-gray-100 p-2 rounded overflow-x-auto">
                                {formatParameters(index.parameters)}
                              </pre>
                            </div>
                          )}
                          <div className="mt-3 flex justify-end">
                            <button
                              onClick={() => handleDeleteIndex(index.indexId)}
                              className="inline-flex items-center px-3 py-1 border border-transparent text-xs leading-4 font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                            >
                              Delete
                            </button>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          )}
        </div>
      </main>
    </Layout>
  );
}