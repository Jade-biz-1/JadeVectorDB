import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Layout from '../../components/Layout';
import { databaseApi, vectorApi } from '../../lib/api';
import {
  Alert, AlertDescription,
  Button,
  Card, CardHeader, CardTitle, CardContent,
  EmptyState,
  LoadingSpinner,
  Modal,
  StatusBadge,
} from '../../components/ui';

export default function DatabaseDetails() {
  const router = useRouter();
  const { id } = router.query;

  const [database, setDatabase] = useState(null);
  const [vectors, setVectors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expandedVectorId, setExpandedVectorId] = useState(null);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deleteConfirmName, setDeleteConfirmName] = useState('');
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (id) {
      fetchDatabaseDetails();
      fetchVectors();
    }
  }, [id]);

  const fetchDatabaseDetails = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await databaseApi.getDatabase(id);
      setDatabase(response);
    } catch (error) {
      console.error('Error fetching database details:', error);
      setError(`Error fetching database details: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchVectors = async () => {
    try {
      const response = await vectorApi.listVectors(id, 5, 0);
      setVectors(response.vectors || []);
    } catch (error) {
      console.error('Error fetching vectors:', error);
    }
  };

  const handleDeleteDatabase = async () => {
    setDeleting(true);
    try {
      await databaseApi.deleteDatabase(id);
      router.push('/databases');
    } catch (error) {
      alert('Failed to delete database: ' + error.message);
    } finally {
      setDeleting(false);
    }
  };

  const toggleVectorExpand = (vectorId) => {
    setExpandedVectorId(expandedVectorId === vectorId ? null : vectorId);
  };

  const inputCls = 'w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-red-300 focus:border-red-400 transition';

  if (loading) {
    return (
      <Layout title="Loading... - JadeVectorDB">
        <LoadingSpinner label="Loading database details…" />
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="Error - JadeVectorDB">
        <Alert variant="destructive" className="mb-6 bg-red-50 border-red-200 text-red-800">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <Button variant="secondary" onClick={() => router.push('/databases')}>
          Back to Databases
        </Button>
      </Layout>
    );
  }

  if (!database) {
    return (
      <Layout title="Not Found - JadeVectorDB">
        <EmptyState
          icon="🔍"
          title="Database Not Found"
          description="The database you're looking for doesn't exist."
          action={
            <Button variant="secondary" onClick={() => router.push('/databases')}>
              Back to Databases
            </Button>
          }
        />
      </Layout>
    );
  }

  return (
    <Layout title={`${database.name} - JadeVectorDB`}>
      {/* ── Page header ── */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
        <h1 className="text-3xl font-bold text-gray-900">{database.name}</h1>
        <div className="flex gap-3 flex-wrap">
          <Button variant="secondary" onClick={() => router.push('/databases')}>
            Back to Databases
          </Button>
          <Button onClick={() => router.push(`/search?databaseId=${id}`)}>
            Search Vectors
          </Button>
          <Button variant="destructive" onClick={() => setDeleteModalOpen(true)}>
            Delete Database
          </Button>
        </div>
      </div>

      {/* ── Database Info ── */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-xl">Database Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {[
              { label: 'Database ID', value: database.databaseId || id },
              { label: 'Name', value: database.name },
              { label: 'Description', value: database.description || 'No description' },
              { label: 'Vector Dimension', value: database.vectorDimension },
              { label: 'Index Type', value: <StatusBadge status="info" label={database.indexType} /> },
              { label: 'Status', value: <StatusBadge status={database.status || 'active'} /> },
              ...(database.created_at ? [{ label: 'Created', value: new Date(database.created_at).toLocaleString() }] : []),
              ...(database.updated_at ? [{ label: 'Updated', value: new Date(database.updated_at).toLocaleString() }] : []),
            ].map(({ label, value }) => (
              <div key={label}>
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">{label}</p>
                <p className="text-sm font-medium text-gray-800">{value}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* ── Vectors section ── */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-3">
            <CardTitle className="text-xl">Recent Vectors</CardTitle>
            <Button size="sm" onClick={() => router.push(`/vectors?databaseId=${id}`)}>
              Manage Vectors
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {vectors.length > 0 ? (
            <div className="space-y-3">
              {vectors.map((vector, index) => {
                const isExpanded = expandedVectorId === vector.id;
                const vectorLength = vector.values?.length || vector.vector?.length || database.vectorDimension;

                return (
                  <div
                    key={vector.id || index}
                    className="border border-gray-200 rounded-xl overflow-hidden"
                  >
                    <div
                      className="flex items-center justify-between px-4 py-3 bg-gray-50 cursor-pointer hover:bg-gray-100 transition-colors"
                      onClick={() => toggleVectorExpand(vector.id)}
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-sm font-semibold text-indigo-600">
                          ID: {vector.id}
                        </span>
                        <span className="text-gray-400 text-xs">{isExpanded ? '▼' : '▶'}</span>
                      </div>
                      <span className="text-xs text-gray-500 bg-gray-200 px-2.5 py-1 rounded-full">
                        {vectorLength} dimensions
                      </span>
                    </div>

                    {isExpanded && (
                      <div className="px-4 py-4 border-t border-gray-200 space-y-3">
                        <div>
                          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                            Vector Values
                          </p>
                          {vector.values ? (
                            <div className="flex items-start gap-3">
                              <div className="flex-1 font-mono text-xs text-gray-700 bg-gray-50 px-3 py-2 rounded break-all">
                                [{vector.values.slice(0, 10).join(', ')}
                                {vector.values.length > 10 ? `, ... (${vector.values.length - 10} more)` : ''}]
                              </div>
                              <button
                                className="px-2.5 py-1.5 text-xs font-medium bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors whitespace-nowrap"
                                onClick={() => {
                                  navigator.clipboard.writeText(JSON.stringify(vector.values));
                                  alert('Vector values copied to clipboard!');
                                }}
                              >
                                Copy All
                              </button>
                            </div>
                          ) : (
                            <p className="text-xs text-gray-400 italic">No vector data available</p>
                          )}
                        </div>

                        {vector.metadata && Object.keys(vector.metadata).length > 0 && (
                          <div>
                            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                              Metadata
                            </p>
                            <pre className="font-mono text-xs text-gray-700 bg-gray-50 px-3 py-2 rounded overflow-x-auto m-0">
                              {JSON.stringify(vector.metadata, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <EmptyState
              icon="📦"
              title="No vectors in this database yet"
              action={
                <Button size="sm" onClick={() => router.push(`/vectors?databaseId=${id}`)}>
                  Add Vectors
                </Button>
              }
            />
          )}
        </CardContent>
      </Card>

      {/* ── Delete Confirmation Modal ── */}
      <Modal
        open={deleteModalOpen}
        onClose={() => { setDeleteModalOpen(false); setDeleteConfirmName(''); }}
        title="Delete Database"
      >
        <p className="text-sm text-gray-600 mb-3">
          This will permanently delete <strong className="text-gray-900">{database.name}</strong> and
          all its vectors. This action cannot be undone.
        </p>
        <p className="text-sm text-gray-600 mb-3">
          Type <strong className="text-gray-900">{database.name}</strong> to confirm:
        </p>
        <input
          type="text"
          className={inputCls}
          placeholder="Enter database name"
          value={deleteConfirmName}
          onChange={(e) => setDeleteConfirmName(e.target.value)}
          autoFocus
        />
        <div className="flex justify-end gap-3 mt-5">
          <Button
            variant="secondary"
            onClick={() => { setDeleteModalOpen(false); setDeleteConfirmName(''); }}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            disabled={deleteConfirmName !== database.name || deleting}
            onClick={handleDeleteDatabase}
          >
            {deleting ? 'Deleting…' : 'Delete'}
          </Button>
        </div>
      </Modal>
    </Layout>
  );
}
