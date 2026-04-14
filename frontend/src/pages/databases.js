import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { databaseApi } from '../lib/api';
import {
  Alert, AlertDescription,
  Button,
  Card, CardHeader, CardTitle, CardDescription, CardContent,
  EmptyState,
  FormField,
  StatusBadge,
} from '../components/ui';

export default function DatabaseManagement() {
  const [databases, setDatabases] = useState([]);
  const [newDatabase, setNewDatabase] = useState({
    name: '',
    description: '',
    vectorDimension: 128,
    indexType: 'FLAT'
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const fetchDatabases = async () => {
    try {
      setError('');
      const response = await databaseApi.listDatabases();
      setDatabases(response.databases || []);
    } catch (err) {
      console.error('Error fetching databases:', err);
      setError(`Error fetching databases: ${err.message}`);
    }
  };

  const handleCreateDatabase = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');
    try {
      await databaseApi.createDatabase({
        name: newDatabase.name,
        description: newDatabase.description,
        vectorDimension: parseInt(newDatabase.vectorDimension),
        indexType: newDatabase.indexType,
      });
      setNewDatabase({ name: '', description: '', vectorDimension: 128, indexType: 'FLAT' });
      setSuccess('Database created successfully!');
      fetchDatabases();
    } catch (err) {
      console.error('Error creating database:', err);
      setError(`Error creating database: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchDatabases(); }, []);

  const inputCls = 'w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-500 transition';

  return (
    <Layout title="Database Management - JadeVectorDB">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-1">Database Management</h1>
        <p className="text-gray-500">Create and manage your vector databases</p>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-6 bg-red-50 border-red-200 text-red-800">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {success && (
        <Alert className="mb-6 bg-green-50 border-green-200 text-green-800">
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      {/* ── Create form ── */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-xl">Create New Database</CardTitle>
          <CardDescription>Configure a new vector database with specific settings</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleCreateDatabase}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

              <FormField label="Database Name" htmlFor="name" required className="md:col-span-2">
                <input
                  id="name"
                  type="text"
                  className={inputCls}
                  value={newDatabase.name}
                  onChange={(e) => setNewDatabase({ ...newDatabase, name: e.target.value })}
                  required
                  placeholder="my-vector-db"
                />
              </FormField>

              <FormField label="Description" htmlFor="description" className="md:col-span-2">
                <textarea
                  id="description"
                  className={`${inputCls} resize-y min-h-[80px]`}
                  value={newDatabase.description}
                  onChange={(e) => setNewDatabase({ ...newDatabase, description: e.target.value })}
                  placeholder="A brief description of this database"
                />
              </FormField>

              <FormField label="Vector Dimension" htmlFor="vectorDimension" required>
                <select
                  id="vectorDimension"
                  className={inputCls}
                  value={newDatabase.vectorDimension}
                  onChange={(e) => setNewDatabase({ ...newDatabase, vectorDimension: parseInt(e.target.value) })}
                >
                  <option value={128}>128</option>
                  <option value={256}>256</option>
                  <option value={512}>512</option>
                  <option value={768}>768 (BERT)</option>
                  <option value={1024}>1024</option>
                  <option value={1536}>1536 (OpenAI)</option>
                  <option value={2048}>2048</option>
                </select>
              </FormField>

              <FormField label="Index Type" htmlFor="indexType" required>
                <select
                  id="indexType"
                  className={inputCls}
                  value={newDatabase.indexType}
                  onChange={(e) => setNewDatabase({ ...newDatabase, indexType: e.target.value })}
                >
                  <option value="FLAT">FLAT (Linear Search)</option>
                  <option value="HNSW">HNSW (Fast Approximate)</option>
                  <option value="IVF">IVF (Inverted File)</option>
                  <option value="LSH">LSH (Locality Sensitive)</option>
                </select>
              </FormField>

              <div className="md:col-span-2">
                <Button type="submit" disabled={loading} className="w-full sm:w-auto">
                  {loading ? 'Creating…' : 'Create Database'}
                </Button>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* ── Database list ── */}
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">Your Databases</CardTitle>
          <CardDescription>Manage and access your vector databases</CardDescription>
        </CardHeader>
        <CardContent>
          {databases.length === 0 ? (
            <EmptyState
              icon="📊"
              title="No databases yet"
              description="Create your first database to get started"
            />
          ) : (
            <div className="grid gap-4">
              {databases.map((db) => {
                const id = db.id || db.databaseId;
                return (
                  <div
                    key={id}
                    className="border border-gray-200 rounded-xl p-5 hover:shadow-md hover:-translate-y-0.5 transition-all"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <a
                          href={`/databases/${id}`}
                          className="text-lg font-semibold text-gray-900 hover:text-indigo-600 transition-colors block mb-1"
                        >
                          {db.name}
                        </a>
                        <StatusBadge status={db.status || 'active'} />
                      </div>
                    </div>

                    {db.description && (
                      <p className="text-sm text-gray-500 mb-3">{db.description}</p>
                    )}

                    <div className="flex gap-6 text-sm text-gray-500 mb-4">
                      <span>📊 {(db.vectors ?? db.stats?.vectorCount ?? 0).toLocaleString()} vectors</span>
                      <span>🔍 {db.indexes ?? db.stats?.indexCount ?? 0} indexes</span>
                      <span>📏 {db.vectorDimension}D</span>
                    </div>

                    <div className="flex gap-3 flex-wrap pt-3 border-t border-gray-100">
                      <Button variant="secondary" size="sm" asChild>
                        <a href={`/databases/${id}`}>View Details</a>
                      </Button>
                      <Button size="sm" asChild>
                        <a href={`/search?databaseId=${id}`}>Search Vectors</a>
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </Layout>
  );
}
