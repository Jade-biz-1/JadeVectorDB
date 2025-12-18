import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { vectorApi, databaseApi } from '../lib/api';

export default function VectorManagement() {
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [vectors, setVectors] = useState([]);
  const [newVector, setNewVector] = useState({ values: '', metadata: '' });
  const [editVector, setEditVector] = useState(null);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [pagination, setPagination] = useState({ limit: 50, offset: 0, total: 0 });

  useEffect(() => {
    fetchDatabases();
  }, []);

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      setDatabases(response.databases || []);
    } catch (error) {
      alert('Error fetching databases: ' + error.message);
    }
  };

  const fetchVectors = async (limit = pagination.limit, offset = pagination.offset) => {
    if (!selectedDatabase) return;
    setLoading(true);
    try {
      const response = await vectorApi.listVectors(selectedDatabase, limit, offset);
      const vectorsData = response.vectors || [];

      // Transform vectors to include ID and format values for display
      const formattedVectors = vectorsData.map(v => ({
        id: v.vectorId || v.id,
        values: Array.isArray(v.values) ? v.values : [],
        metadata: v.metadata || {},
        valuesString: Array.isArray(v.values) ? v.values.join(', ') : '',
        metadataString: JSON.stringify(v.metadata || {})
      }));

      setVectors(formattedVectors);
      setPagination({
        limit,
        offset,
        total: response.total || vectorsData.length
      });
    } catch (error) {
      console.error('Error fetching vectors:', error);
      alert('Error fetching vectors: ' + error.message);
      setVectors([]);
    }
    setLoading(false);
  };

  const handleCreateVector = async (e) => {
    e.preventDefault();
    if (!selectedDatabase) return alert('Select a database first');
    setLoading(true);
    try {
      const values = newVector.values.split(',').map(Number);
      const metadata = newVector.metadata ? JSON.parse(newVector.metadata) : {};
      await vectorApi.storeVector(selectedDatabase, { values, metadata });
      setNewVector({ values: '', metadata: '' });
      fetchVectors();
    } catch (error) {
      alert('Error creating vector: ' + error.message);
    }
    setLoading(false);
  };

  const handleEditVector = (vector) => {
    setEditVector({
      id: vector.id,
      values: vector.valuesString,
      metadata: vector.metadataString
    });
    setEditModalOpen(true);
  };

  const handleUpdateVector = async (e) => {
    e.preventDefault();
    if (!selectedDatabase || !editVector) return;
    setLoading(true);
    try {
      const values = editVector.values.split(',').map(Number);
      const metadata = editVector.metadata ? JSON.parse(editVector.metadata) : {};
      await vectorApi.updateVector(selectedDatabase, editVector.id, { values, metadata });
      setEditModalOpen(false);
      fetchVectors();
    } catch (error) {
      alert('Error updating vector: ' + error.message);
    }
    setLoading(false);
  };

  const handleDeleteVector = async (vectorId) => {
    if (!selectedDatabase) return;
    if (!confirm('Are you sure you want to delete this vector?')) return;

    setLoading(true);
    try {
      await vectorApi.deleteVector(selectedDatabase, vectorId);
      fetchVectors();
    } catch (error) {
      alert('Error deleting vector: ' + error.message);
    }
    setLoading(false);
  };

  const handleNextPage = () => {
    const newOffset = pagination.offset + pagination.limit;
    if (newOffset < pagination.total) {
      fetchVectors(pagination.limit, newOffset);
    }
  };

  const handlePrevPage = () => {
    const newOffset = Math.max(0, pagination.offset - pagination.limit);
    fetchVectors(pagination.limit, newOffset);
  };

  return (
    <Layout title="Vector Management - JadeVectorDB">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8 text-gray-900">Vector Management</h1>
        <div className="mb-4">
          <label className="block mb-2">Select Database:</label>
          <select value={selectedDatabase} onChange={e => { setSelectedDatabase(e.target.value); fetchVectors(); }} className="border rounded px-2 py-1">
            <option value="">-- Select --</option>
        {/* Database Selection */}
        <div className="mb-6 bg-white rounded-lg shadow p-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Select Database:</label>
          <select 
            value={selectedDatabase} 
            onChange={e => { setSelectedDatabase(e.target.value); fetchVectors(); }} 
            className="w-full md:w-auto border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="">-- Select Database --</option>
            {databases.map(db => (
              <option key={db.databaseId || db.id} value={db.databaseId || db.id}>{db.name}</option>
        {/* Vectors List */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">Vectors</h2>
          </div>
          <div className="p-6">
            {loading && (
            <>
              <div className="space-y-4">
                {vectors.map(vector => (
                  <div key={vector.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center mb-2">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800">
                            ID: {vector.id}
                          </span>
                        </div>
                        <div className="mb-2">
                          <p className="text-xs font-medium text-gray-500 mb-1">Values:</p>
                          <p className="text-sm text-gray-900 font-mono bg-gray-50 p-2 rounded break-all">
                            [{vector.valuesString.substring(0, 150)}{vector.valuesString.length > 150 ? '...' : ''}]
                          </p>
                        </div>
                        <div>
                          <p className="text-xs font-medium text-gray-500 mb-1">Metadata:</p>
                          <p className="text-sm text-gray-900 font-mono bg-gray-50 p-2 rounded break-all">
                            {vector.metadataString}
                          </p>
                        </div>
                      </div>
                      <div className="ml-4 flex-shrink-0 flex flex-col space-y-2">
                        <button
                          onClick={() => handleEditVector(vector)}
                          className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-yellow-500 hover:bg-yellow-600 transition-colors"
                        >
                          <svg className="mr-1.5 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                          </svg>
                          Edit
                        </button>
                        <button
                          onClick={() => handleDeleteVector(vector.id)}
                          className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-red-500 hover:bg-red-600 transition-colors"
                        >
                          <svg className="mr-1.5 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>data (JSON format)
              </label>
              <input 
                type="text" 
                placeholder='{"key": "value", "label": "example"}' 
                value={newVector.metadata} 
                onChange={e => setNewVector({ ...newVector, metadata: e.target.value })} 
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500" 
              />
            </div>
            <button 
              type="submit" 
              disabled={!selectedDatabase || loading}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-6 py-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Creating...' : 'Create Vector'}
            </button>
          </form>
        </div>ding && vectors.length === 0 && selectedDatabase && (
            <p className="text-gray-500">No vectors found in this database.</p>
          )}
          {!loading && vectors.length > 0 && (
            <>
              <div className="bg-white shadow overflow-hidden sm:rounded-md mb-4">
                <ul className="divide-y divide-gray-200">
                  {vectors.map(vector => (
                    <li key={vector.id} className="px-4 py-4">
                      <div className="flex items-center justify-between">
              {/* Pagination Controls */}
              <div className="mt-6 flex items-center justify-between border-t border-gray-200 bg-white px-4 py-4 sm:px-6 rounded-lg">
                <div className="flex-1 flex justify-between items-center">
                  <div>
                    <p className="text-sm text-gray-700">
                      Showing <span className="font-medium">{pagination.offset + 1}</span> to{' '}
                      <span className="font-medium">
                        {Math.min(pagination.offset + pagination.limit, pagination.total)}
                      </span>{' '}
                      of <span className="font-medium">{pagination.total}</span> vectors
                    </p>
                  </div>
                  <div className="flex space-x-3">
                    <button
                      onClick={handlePrevPage}
                      disabled={pagination.offset === 0}
                      className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      <svg className="mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                      </svg>
                      Previous
                    </button>
                    <button
                      onClick={handleNextPage}
                      disabled={pagination.offset + pagination.limit >= pagination.total}
                      className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      Next
                      <svg className="ml-2 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </button>
        {/* Edit Modal */}
        {editModalOpen && (
          <div className="fixed inset-0 bg-gray-600 bg-opacity-75 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-900">Edit Vector</h3>
                <button
                  onClick={() => setEditModalOpen(false)}
                  className="text-gray-400 hover:text-gray-500"
                >
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <form onSubmit={handleUpdateVector} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Vector Values (comma-separated)
                  </label>
                  <input 
                    type="text" 
                    value={editVector.values} 
                    onChange={e => setEditVector({ ...editVector, values: e.target.value })} 
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500" 
                    required 
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Metadata (JSON)
                  </label>
                  <input 
                    type="text" 
                    value={editVector.metadata} 
                    onChange={e => setEditVector({ ...editVector, metadata: e.target.value })} 
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500" 
                  />
                </div>
                <div className="flex justify-end space-x-3 pt-4">
                  <button 
                    type="button" 
                    onClick={() => setEditModalOpen(false)} 
                    className="px-6 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium transition-colors"
                  >
                    Cancel
                  </button>
                  <button 
                    type="submit" 
                    disabled={loading}
                    className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-6 py-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {loading ? 'Updating...' : 'Update Vector'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}                   <button
                      onClick={handlePrevPage}
                      disabled={pagination.offset === 0}
                      className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Previous
                    </button>
                    <button
                      onClick={handleNextPage}
                      disabled={pagination.offset + pagination.limit >= pagination.total}
                      className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
        {editModalOpen && (
          <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded shadow-lg">
              <h3 className="text-lg font-bold mb-4">Edit Vector</h3>
              <form onSubmit={handleUpdateVector}>
                <input type="text" value={editVector.values} onChange={e => setEditVector({ ...editVector, values: e.target.value })} className="border rounded px-2 py-1 mb-2 w-full" required />
                <input type="text" value={editVector.metadata} onChange={e => setEditVector({ ...editVector, metadata: e.target.value })} className="border rounded px-2 py-1 mb-2 w-full" />
                <div className="flex justify-end space-x-2">
                  <button type="button" onClick={() => setEditModalOpen(false)} className="px-4 py-1 rounded border">Cancel</button>
                  <button type="submit" className="bg-indigo-600 text-white px-4 py-1 rounded">Update</button>
                </div>
              </form>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
