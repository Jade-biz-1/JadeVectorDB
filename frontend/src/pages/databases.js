import Head from 'next/head';
import { useState } from 'react';

export default function DatabaseManagement() {
  const [databases, setDatabases] = useState([
    { id: 'db1', name: 'Documents', description: 'Vector database for document embeddings', vectors: 12500, indexes: 3, status: 'online' },
    { id: 'db2', name: 'Images', description: 'Vector database for image embeddings', vectors: 8900, indexes: 2, status: 'online' },
    { id: 'db3', name: 'Products', description: 'Vector database for product recommendations', vectors: 42500, indexes: 5, status: 'offline' }
  ]);
  
  const [newDatabase, setNewDatabase] = useState({ name: '', description: '', dimension: 128 });
  const [loading, setLoading] = useState(false);

  const handleCreateDatabase = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      // In real implementation, this would call the API to create a database
      console.log('Creating database:', newDatabase);
      
      // Mock creation
      const newDb = {
        id: `db${databases.length + 1}`,
        name: newDatabase.name,
        description: newDatabase.description,
        vectors: 0,
        indexes: 0,
        status: 'online'
      };
      
      setDatabases([...databases, newDb]);
      setNewDatabase({ name: '', description: '', dimension: 128 });
    } catch (error) {
      console.error('Error creating database:', error);
    } finally {
      setLoading(false);
    }
  };

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
                      <label htmlFor="dimension" className="block text-sm font-medium text-gray-700">
                        Vector Dimension
                      </label>
                      <select
                        id="dimension"
                        name="dimension"
                        value={newDatabase.dimension}
                        onChange={(e) => setNewDatabase({...newDatabase, dimension: parseInt(e.target.value)})}
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

          {/* Database List */}
          <div className="bg-white shadow overflow-hidden sm:rounded-md">
            <ul className="divide-y divide-gray-200">
              {databases.map((database) => (
                <li key={database.id}>
                  <div className="px-4 py-4 sm:px-6">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-medium text-indigo-600 truncate">{database.name}</div>
                      <div className="ml-2 flex-shrink-0 flex">
                        <span className={`inline-flex px-2 text-xs leading-5 font-semibold rounded-full ${
                          database.status === 'online' 
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
                      <button
                        type="button"
                        className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      >
                        View Details
                      </button>
                      <button
                        type="button"
                        className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      >
                        Search
                      </button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}