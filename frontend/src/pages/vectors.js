import Head from 'next/head';
import { useState, useEffect } from 'react';
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
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Vector Management - JadeVectorDB</title>
        <meta name="description" content="Manage vectors in JadeVectorDB" />
      </Head>
      <main className="max-w-4xl mx-auto py-8">
        <h1 className="text-2xl font-bold mb-6">Vector Management</h1>
        <div className="mb-4">
          <label className="block mb-2">Select Database:</label>
          <select value={selectedDatabase} onChange={e => { setSelectedDatabase(e.target.value); fetchVectors(); }} className="border rounded px-2 py-1">
            <option value="">-- Select --</option>
            {databases.map(db => (
              <option key={db.databaseId || db.id} value={db.databaseId || db.id}>{db.name}</option>
            ))}
          </select>
        </div>
        <form onSubmit={handleCreateVector} className="mb-6">
          <h2 className="text-lg font-semibold mb-2">Create Vector</h2>
          <input type="text" placeholder="Comma-separated values" value={newVector.values} onChange={e => setNewVector({ ...newVector, values: e.target.value })} className="border rounded px-2 py-1 mr-2" required />
          <input type="text" placeholder="Metadata (JSON)" value={newVector.metadata} onChange={e => setNewVector({ ...newVector, metadata: e.target.value })} className="border rounded px-2 py-1 mr-2" />
          <button type="submit" className="bg-indigo-600 text-white px-4 py-1 rounded">Create</button>
        </form>
        <div>
          <h2 className="text-lg font-semibold mb-2">Vectors</h2>
          {loading && <p className="text-gray-500">Loading vectors...</p>}
          {!loading && vectors.length === 0 && selectedDatabase && (
            <p className="text-gray-500">No vectors found in this database.</p>
          )}
          {!loading && vectors.length > 0 && (
            <>
              <div className="bg-white shadow overflow-hidden sm:rounded-md mb-4">
                <ul className="divide-y divide-gray-200">
                  {vectors.map(vector => (
                    <li key={vector.id} className="px-4 py-4">
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            ID: {vector.id}
                          </p>
                          <p className="text-sm text-gray-500 truncate">
                            Values: [{vector.valuesString.substring(0, 100)}{vector.valuesString.length > 100 ? '...' : ''}]
                          </p>
                          <p className="text-sm text-gray-500">
                            Metadata: {vector.metadataString}
                          </p>
                        </div>
                        <div className="ml-4 flex-shrink-0 flex space-x-2">
                          <button
                            onClick={() => handleEditVector(vector)}
                            className="inline-flex items-center px-3 py-1 border border-yellow-300 text-sm font-medium rounded-md text-yellow-700 bg-yellow-100 hover:bg-yellow-200"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => handleDeleteVector(vector.id)}
                            className="inline-flex items-center px-3 py-1 border border-red-300 text-sm font-medium rounded-md text-red-700 bg-red-100 hover:bg-red-200"
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Pagination Controls */}
              <div className="flex items-center justify-between bg-white px-4 py-3 sm:px-6 rounded-md shadow">
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
                  <div className="flex space-x-2">
                    <button
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
