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

  const fetchVectors = async () => {
    if (!selectedDatabase) return;
    setLoading(true);
    try {
      // You may want to implement pagination here
      // For now, assume API endpoint exists: GET /databases/:id/vectors
      // This is a placeholder, update as per backend
      // const response = await vectorApi.listVectors(selectedDatabase);
      // setVectors(response.vectors || []);
    } catch (error) {
      alert('Error fetching vectors: ' + error.message);
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
    setEditVector(vector);
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
    setLoading(true);
    try {
      await vectorApi.deleteVector(selectedDatabase, vectorId);
      fetchVectors();
    } catch (error) {
      alert('Error deleting vector: ' + error.message);
    }
    setLoading(false);
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
          <ul className="divide-y">
            {vectors.map(vector => (
              <li key={vector.id} className="py-2 flex justify-between items-center">
                <span>{vector.values.join(', ')}</span>
                <span>{JSON.stringify(vector.metadata)}</span>
                <button onClick={() => handleEditVector(vector)} className="text-yellow-600 mr-2">Edit</button>
                <button onClick={() => handleDeleteVector(vector.id)} className="text-red-600">Delete</button>
              </li>
            ))}
          </ul>
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
