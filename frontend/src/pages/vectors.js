import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { vectorApi, databaseApi } from '../lib/api';
import { useDatabases } from '../hooks/useDatabases';
import {
  Button,
  Card, CardHeader, CardTitle, CardContent,
  EmptyState,
  FormField,
  LoadingSpinner,
  Modal,
} from '../components/ui';

export default function VectorManagement() {
  const { databases } = useDatabases();
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [databaseDetails, setDatabaseDetails] = useState(null);
  const [vectors, setVectors] = useState([]);
  const [newVector, setNewVector] = useState({ values: '', metadata: '' });
  const [editVector, setEditVector] = useState(null);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);
  const [totalVectors, setTotalVectors] = useState(0);

  useEffect(() => {
    if (selectedDatabase) {
      fetchDatabaseDetails();
      setCurrentPage(1);
    }
  }, [selectedDatabase]);

  useEffect(() => {
    if (selectedDatabase) {
      fetchVectors();
    }
  }, [selectedDatabase, currentPage]);

  const fetchDatabaseDetails = async () => {
    if (!selectedDatabase) return;
    try {
      const response = await databaseApi.getDatabase(selectedDatabase);
      setDatabaseDetails(response);
    } catch (error) {
      console.error('Error fetching database details:', error);
    }
  };

  const fetchVectors = async () => {
    if (!selectedDatabase) return;
    setLoading(true);
    try {
      const offset = (currentPage - 1) * pageSize;
      const response = await vectorApi.listVectors(selectedDatabase, pageSize, offset);
      setVectors(response.vectors || []);
      setTotalVectors(response.total || 0);
    } catch (error) {
      console.error('Error fetching vectors:', error);
      alert('Error fetching vectors: ' + error.message);
      setVectors([]);
      setTotalVectors(0);
    }
    setLoading(false);
  };

  const totalPages = Math.ceil(totalVectors / pageSize);

  const goToPage = (page) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  const handleCreateVector = async (e) => {
    e.preventDefault();
    if (!selectedDatabase) return alert('Select a database first');

    const values = newVector.values.trim().split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));

    if (databaseDetails && databaseDetails.vectorDimension) {
      const expectedDim = databaseDetails.vectorDimension;
      if (values.length !== expectedDim) {
        return alert(
          `Dimension mismatch!\n\n` +
          `Database "${databaseDetails.name}" expects ${expectedDim} dimensions.\n` +
          `You provided ${values.length} dimension${values.length !== 1 ? 's' : ''}.\n\n` +
          `Please provide exactly ${expectedDim} comma-separated values.`
        );
      }
    }

    setLoading(true);
    try {
      const metadata = newVector.metadata ? JSON.parse(newVector.metadata) : {};
      const id = `vec_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
      await vectorApi.storeVector(selectedDatabase, { id, values, metadata });
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
      values: vector.values.join(','),
      metadata: JSON.stringify(vector.metadata)
    });
    setEditModalOpen(true);
  };

  const handleUpdateVector = async (e) => {
    e.preventDefault();
    if (!selectedDatabase) return;
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

  const inputCls = 'w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-500 transition';

  return (
    <Layout title="Vector Management - JadeVectorDB">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-1">Vector Management</h1>
        <p className="text-gray-500">Store, manage, and search vector embeddings</p>
      </div>

      {/* ── Database Selector ── */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-xl">Select Database</CardTitle>
        </CardHeader>
        <CardContent>
          <FormField label="Database" htmlFor="dbSelect">
            <select
              id="dbSelect"
              className={inputCls}
              value={selectedDatabase}
              onChange={e => { setSelectedDatabase(e.target.value); fetchVectors(); }}
            >
              <option value="">-- Select a database --</option>
              {databases.map(db => (
                <option key={db.databaseId || db.id} value={db.databaseId || db.id}>
                  {db.name}
                </option>
              ))}
            </select>
          </FormField>
        </CardContent>
      </Card>

      {/* ── Create Vector ── */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-xl">Create New Vector</CardTitle>
        </CardHeader>
        <CardContent>
          {databaseDetails && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 text-sm text-blue-800 mb-5">
              <strong>Required Dimension:</strong> {databaseDetails.vectorDimension} values
              {' | '}
              <strong>Database:</strong> {databaseDetails.name}
              {' | '}
              <strong>Index Type:</strong> {databaseDetails.indexType}
            </div>
          )}

          <form onSubmit={handleCreateVector}>
            <FormField
              label={
                <>
                  Vector Values (comma-separated numbers)
                  {newVector.values && (() => {
                    const count = newVector.values.trim().split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v)).length;
                    const expected = databaseDetails?.vectorDimension || 0;
                    const isValid = count === expected;
                    return (
                      <span className={`ml-2 text-xs font-semibold px-1.5 py-0.5 rounded ${isValid ? 'text-emerald-700 bg-emerald-100' : 'text-red-700 bg-red-100'}`}>
                        [{count} / {expected} values]
                      </span>
                    );
                  })()}
                </>
              }
              htmlFor="vectorValues"
            >
              <input
                id="vectorValues"
                type="text"
                className={inputCls}
                placeholder="0.1, 0.2, 0.3, 0.4, ..."
                value={newVector.values}
                onChange={e => setNewVector({ ...newVector, values: e.target.value })}
                required
              />
            </FormField>

            <FormField label="Metadata (JSON format, optional)" htmlFor="vectorMetadata">
              <input
                id="vectorMetadata"
                type="text"
                className={inputCls}
                placeholder='{"label": "example", "category": "test"}'
                value={newVector.metadata}
                onChange={e => setNewVector({ ...newVector, metadata: e.target.value })}
              />
            </FormField>

            <Button type="submit" disabled={!selectedDatabase || loading}>
              {loading ? 'Creating…' : 'Create Vector'}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* ── Vector List ── */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl">Vectors</CardTitle>
            {totalVectors > 0 && (
              <span className="text-sm text-gray-500">
                Showing {(currentPage - 1) * pageSize + 1}-{Math.min(currentPage * pageSize, totalVectors)} of {totalVectors} vectors
              </span>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {loading && <LoadingSpinner label="Loading vectors…" />}

          {!loading && vectors.length === 0 && selectedDatabase && (
            <EmptyState icon="📦" title="No vectors found in this database" />
          )}

          {!loading && vectors.length === 0 && !selectedDatabase && (
            <EmptyState icon="🗄️" title="Select a database to view vectors" />
          )}

          {!loading && vectors.length > 0 && (
            <>
              <div className="space-y-3">
                {vectors.map(vector => (
                  <div
                    key={vector.id}
                    className="border border-gray-200 rounded-xl p-5 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <span className="inline-block px-3 py-1 bg-violet-100 text-violet-700 rounded-full text-sm font-medium">
                        ID: {vector.id}
                      </span>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline" onClick={() => handleEditVector(vector)}>
                          Edit
                        </Button>
                        <Button size="sm" variant="destructive" onClick={() => handleDeleteVector(vector.id)}>
                          Delete
                        </Button>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div>
                        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                          Values ({vector.values.length} dimensions)
                        </p>
                        <div className="font-mono text-sm text-gray-700 bg-gray-50 px-3 py-2 rounded break-all">
                          [{vector.values.slice(0, 10).join(', ')}
                          {vector.values.length > 10 ? `, ... (${vector.values.length - 10} more)` : ''}]
                        </div>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                          Metadata
                        </p>
                        <div className="font-mono text-sm text-gray-700 bg-gray-50 px-3 py-2 rounded break-all">
                          {JSON.stringify(vector.metadata)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {totalPages > 1 && (
                <div className="flex items-center justify-center gap-2 mt-6 pt-5 border-t border-gray-100">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => goToPage(currentPage - 1)}
                    disabled={currentPage === 1}
                  >
                    Previous
                  </Button>

                  <div className="flex items-center gap-1">
                    {currentPage > 2 && (
                      <>
                        <button
                          className="min-w-[2.25rem] h-9 px-2 border border-gray-200 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
                          onClick={() => goToPage(1)}
                        >
                          1
                        </button>
                        {currentPage > 3 && <span className="px-2 text-gray-400">…</span>}
                      </>
                    )}
                    {currentPage > 1 && (
                      <button
                        className="min-w-[2.25rem] h-9 px-2 border border-gray-200 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
                        onClick={() => goToPage(currentPage - 1)}
                      >
                        {currentPage - 1}
                      </button>
                    )}
                    <button className="min-w-[2.25rem] h-9 px-2 bg-indigo-600 text-white rounded-md text-sm font-medium cursor-default">
                      {currentPage}
                    </button>
                    {currentPage < totalPages && (
                      <button
                        className="min-w-[2.25rem] h-9 px-2 border border-gray-200 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
                        onClick={() => goToPage(currentPage + 1)}
                      >
                        {currentPage + 1}
                      </button>
                    )}
                    {currentPage < totalPages - 1 && (
                      <>
                        {currentPage < totalPages - 2 && <span className="px-2 text-gray-400">…</span>}
                        <button
                          className="min-w-[2.25rem] h-9 px-2 border border-gray-200 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
                          onClick={() => goToPage(totalPages)}
                        >
                          {totalPages}
                        </button>
                      </>
                    )}
                  </div>

                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => goToPage(currentPage + 1)}
                    disabled={currentPage === totalPages}
                  >
                    Next
                  </Button>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* ── Edit Vector Modal ── */}
      <Modal
        open={editModalOpen}
        onClose={() => setEditModalOpen(false)}
        title="Edit Vector"
      >
        <form onSubmit={handleUpdateVector}>
          <FormField label="Vector Values (comma-separated)" htmlFor="editValues">
            <input
              id="editValues"
              type="text"
              className={inputCls}
              value={editVector?.values || ''}
              onChange={e => setEditVector({ ...editVector, values: e.target.value })}
              required
            />
          </FormField>

          <FormField label="Metadata (JSON format)" htmlFor="editMetadata">
            <input
              id="editMetadata"
              type="text"
              className={inputCls}
              value={editVector?.metadata || ''}
              onChange={e => setEditVector({ ...editVector, metadata: e.target.value })}
            />
          </FormField>

          <div className="flex justify-end gap-3 mt-5">
            <Button type="button" variant="secondary" onClick={() => setEditModalOpen(false)}>
              Cancel
            </Button>
            <Button type="submit">
              Update Vector
            </Button>
          </div>
        </form>
      </Modal>
    </Layout>
  );
}
