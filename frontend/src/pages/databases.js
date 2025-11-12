import Head from 'next/head';
import { useState, useEffect } from 'react';
import { databaseApi } from '../lib/api';

export default function DatabaseManagement() {
  const [databases, setDatabases] = useState([]);
  const [newDatabase, setNewDatabase] = useState({ name: '', description: '', vectorDimension: 128, indexType: 'FLAT' });
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(false);
  // Add missing state for modals and edit/delete database
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [editDatabase, setEditDatabase] = useState({});
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deleteDatabase, setDeleteDatabase] = useState(null);

  const fetchDatabases = async () => {
    setFetching(true);
    try {
      const response = await databaseApi.listDatabases();
      setDatabases(response.databases.map(db => ({
        id: db.databaseId,
        name: db.name,
        description: db.description,
        vectors: db.stats?.vectorCount || 0,
        indexes: db.stats?.indexCount || 0,
        status: db.status || 'active',
        vectorDimension: db.vectorDimension
      })));
    } catch (error) {
      console.error('Error fetching databases:', error);
      alert(`Error fetching databases: ${error.message}`);
    } finally {
      setFetching(false);
    }
  };

  const handleCreateDatabase = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const databaseData = {
        name: newDatabase.name,
        description: newDatabase.description,
        vectorDimension: parseInt(newDatabase.vectorDimension),
        indexType: newDatabase.indexType
      };
      
      await databaseApi.createDatabase(databaseData);
      setNewDatabase({ name: '', description: '', vectorDimension: 128, indexType: 'FLAT' });
      fetchDatabases(); // Refresh the list
    } catch (error) {
      console.error('Error creating database:', error);
      alert(`Error creating database: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatabases();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Database Management - JadeVectorDB</title>
        <meta name="description" content="Manage JadeVectorDB databases" />
      </Head>

      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Database Management</h1>
        </div>
      </header>

      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {/* Create Database Form */}
          <div className="bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6 mb-8">
            <div className="md:grid md:grid-cols-3 md:gap-6">
              <div className="md:col-span-1">
                <h3 className="text-lg font-medium leading-6 text-gray-900">Create Database</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Create a new vector database with specific configurations.
                </p>
              </div>
              <div className="mt-5 md:mt-0 md:col-span-2">
                <form onSubmit={handleCreateDatabase}>
                  <div className="grid grid-cols-6 gap-6">
                    <div className="col-span-6">
                      <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                        Database Name
                      </label>
                      <input
                        type="text"
                        name="name"
                        id="name"
                        value={newDatabase.name}
                        onChange={(e) => setNewDatabase({...newDatabase, name: e.target.value})}
                        required
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      />
                    </div>

                    <div className="col-span-6">
                      <label htmlFor="description" className="block text-sm font-medium text-gray-700">
                        Description
                      </label>
                      <textarea
                        id="description"
                        name="description"
                        rows={3}
                        value={newDatabase.description}
                        onChange={(e) => setNewDatabase({...newDatabase, description: e.target.value})}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      />
                    </div>

                    <div className="col-span-6 sm:col-span-3">
                      <label htmlFor="vectorDimension" className="block text-sm font-medium text-gray-700">
                        Vector Dimension
                      </label>
                      <select
                        id="vectorDimension"
                        name="vectorDimension"
                        value={newDatabase.vectorDimension}
                        onChange={(e) => setNewDatabase({...newDatabase, vectorDimension: parseInt(e.target.value)})}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      >
                        <option value={128}>128</option>
                        <option value={256}>256</option>
                        <option value={512}>512</option>
                        <option value={768}>768 (BERT)</option>
                        <option value={1024}>1024</option>
                        <option value={1536}>1536 (OpenAI)</option>
                        <option value={2048}>2048</option>
                      </select>
                    </div>

                    <div className="col-span-6 sm:col-span-3">
                      <label htmlFor="indexType" className="block text-sm font-medium text-gray-700">
                        Index Type
                      </label>
                      <select
                        id="indexType"
                        name="indexType"
                        value={newDatabase.indexType}
                        onChange={(e) => setNewDatabase({...newDatabase, indexType: e.target.value})}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      >
                        <option value="FLAT">FLAT (Linear)</option>
                        <option value="HNSW">HNSW (Hierarchical Navigable Small World)</option>
                        <option value="IVF">IVF (Inverted File)</option>
                        <option value="LSH">LSH (Locality Sensitive Hashing)</option>
                      </select>
                    </div>
                  </div>

                  <div className="mt-6">
                    <button
                      type="submit"
                      disabled={loading}
                      className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                    >
                      {loading ? 'Creating...' : 'Create Database'}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>

          {/* Database List with Edit/Delete */}
          <div className="bg-white shadow overflow-hidden sm:rounded-md">
            <ul className="divide-y divide-gray-200">
              {databases.map((database) => (
                <li key={database.id}>
                  <div className="px-4 py-4 sm:px-6">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-medium text-indigo-600 truncate">
                        <a href={`/databases/${database.id}`}>{database.name}</a>
                      </div>
                      <div className="ml-2 flex-shrink-0 flex">
                        <span className={`inline-flex px-2 text-xs leading-5 font-semibold rounded-full ${
                          database.status === 'active' || database.status === 'online'
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {database.status}
                        </span>
                      </div>
                    </div>
                    <div className="mt-2 sm:flex sm:justify-between">
                      <div className="sm:flex">
                        <div className="mr-6 text-sm text-gray-500">
                          {database.description}
                        </div>
                        <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                          <svg className="flex-shrink-0 mr-1.5 h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                          </svg>
                          {database.vectors.toLocaleString()} vectors
                        </div>
                      </div>
                      <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                        <svg className="flex-shrink-0 mr-1.5 h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M4 4a2 2 0 00-2 2v4a2 2 0 002 2V6h10a2 2 0 00-2-2H4zm2 6a2 2 0 012-2h8a2 2 0 012 2v4a2 2 0 01-2 2H8a2 2 0 01-2-2v-4zm6 4a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                        </svg>
                        {database.indexes} indexes
                      </div>
                    </div>
                  </div>
                  <div className="bg-gray-50 px-4 py-4 sm:px-6">
                    <div className="flex justify-end space-x-3">
                      <a
                        href={`/databases/${database.id}`}
                        className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      >
                        View Details
                      </a>
                      <a
                        href={`/search?databaseId=${database.id}`}
                        className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      >
                        Search
                      </a>
                      <button
                        onClick={() => handleEditDatabase(database)}
                        className="inline-flex items-center px-3 py-2 border border-yellow-300 shadow-sm text-sm leading-4 font-medium rounded-md text-yellow-700 bg-yellow-100 hover:bg-yellow-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteDatabase(database)}
                        className="inline-flex items-center px-3 py-2 border border-red-300 shadow-sm text-sm leading-4 font-medium rounded-md text-red-700 bg-red-100 hover:bg-red-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          {/* Edit Database Modal */}
          {editModalOpen && (
            <div className="fixed z-10 inset-0 overflow-y-auto">
              <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                <div className="fixed inset-0 transition-opacity" aria-hidden="true">
                  <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
                </div>
                <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
                <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
                  <div>
                    <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Edit Database</h3>
                    <form onSubmit={handleUpdateDatabase}>
                      <div className="grid grid-cols-6 gap-6">
                        <div className="col-span-6 sm:col-span-3">
                          <label htmlFor="editName" className="block text-sm font-medium text-gray-700">Name</label>
                          <input
                            type="text"
                            id="editName"
                            name="editName"
                            value={editDatabase.name}
                            onChange={(e) => setEditDatabase({...editDatabase, name: e.target.value})}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                            required
                          />
                        </div>
                        <div className="col-span-6 sm:col-span-3">
                          <label htmlFor="editDescription" className="block text-sm font-medium text-gray-700">Description</label>
                          <input
                            type="text"
                            id="editDescription"
                            name="editDescription"
                            value={editDatabase.description}
                            onChange={(e) => setEditDatabase({...editDatabase, description: e.target.value})}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                          />
                        </div>
                        <div className="col-span-6 sm:col-span-3">
                          <label htmlFor="editVectorDimension" className="block text-sm font-medium text-gray-700">Vector Dimension</label>
                          <select
                            id="editVectorDimension"
                            name="editVectorDimension"
                            value={editDatabase.vectorDimension}
                            onChange={(e) => setEditDatabase({...editDatabase, vectorDimension: parseInt(e.target.value)})}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                          >
                            <option value={128}>128</option>
                            <option value={256}>256</option>
                            <option value={512}>512</option>
                            <option value={768}>768 (BERT)</option>
                            <option value={1024}>1024</option>
                            <option value={1536}>1536 (OpenAI)</option>
                            <option value={2048}>2048</option>
                          </select>
                        </div>
                        <div className="col-span-6 sm:col-span-3">
                          <label htmlFor="editIndexType" className="block text-sm font-medium text-gray-700">Index Type</label>
                          <select
                            id="editIndexType"
                            name="editIndexType"
                            value={editDatabase.indexType}
                            onChange={(e) => setEditDatabase({...editDatabase, indexType: e.target.value})}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                          >
                            <option value="FLAT">FLAT (Linear)</option>
                            <option value="HNSW">HNSW (Hierarchical Navigable Small World)</option>
                            <option value="IVF">IVF (Inverted File)</option>
                            <option value="LSH">LSH (Locality Sensitive Hashing)</option>
                          </select>
                        </div>
                      </div>
                      <div className="mt-6 flex justify-end space-x-3">
                        <button
                          type="button"
                          onClick={() => setEditModalOpen(false)}
                          className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                        >
                          Update
                        </button>
                      </div>
                    </form>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Delete Confirmation Modal */}
          {deleteModalOpen && (
            <div className="fixed z-10 inset-0 overflow-y-auto">
              <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                <div className="fixed inset-0 transition-opacity" aria-hidden="true">
                  <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
                </div>
                <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
                <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
                  <div>
                    <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Delete Database</h3>
                    <p>Are you sure you want to delete <span className="font-semibold">{deleteDatabase?.name}</span>? This action cannot be undone.</p>
                    <div className="mt-6 flex justify-end space-x-3">
                      <button
                        type="button"
                        onClick={() => setDeleteModalOpen(false)}
                        className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onClick={confirmDeleteDatabase}
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );

// --- CRUD Logic ---
}