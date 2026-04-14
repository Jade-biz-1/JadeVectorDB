import { useEffect, useState } from 'react';
import Layout from '../components/Layout';
import { databaseApi, monitoringApi, securityApi, adminApi, authApi, usersApi } from '../lib/api';
import {
  Button,
  Card, CardHeader, CardTitle, CardContent,
  EmptyState,
  LoadingSpinner,
  StatusBadge,
} from '../components/ui';

export default function Dashboard() {
  const [databases, setDatabases] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [recentLogs, setRecentLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [userRoles, setUserRoles] = useState([]);
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    fetchDashboardData();
    fetchUserRoles();
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchUserRoles = async () => {
    try {
      const currentUser = authApi.getCurrentUser();
      if (currentUser && currentUser.user_id) {
        const userDetails = await usersApi.getUser(currentUser.user_id);
        if (userDetails && userDetails.roles) {
          setUserRoles(userDetails.roles);
          setIsAdmin(userDetails.roles.includes('admin'));
        }
      }
    } catch (error) {
      console.error('Error fetching user roles:', error);
    }
  };

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const [dbRes, statusRes, logsRes] = await Promise.all([
        databaseApi.listDatabases(5, 0).catch(() => ({ databases: [] })),
        monitoringApi.systemStatus().catch(() => null),
        securityApi.listAuditLogs(5, 0).catch(() => ({ events: [] }))
      ]);

      setDatabases(dbRes.databases || []);
      setSystemStatus(statusRes || null);
      setRecentLogs(logsRes.events || []);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleShutdown = async () => {
    if (!window.confirm('Are you sure you want to shut down the server? This will stop all operations and disconnect all clients.')) {
      return;
    }

    try {
      await adminApi.shutdownServer();
      alert('Server shutdown initiated successfully. The server will stop shortly.');
      setTimeout(() => {
        window.location.href = '/';
      }, 2000);
    } catch (error) {
      alert('Failed to shutdown server: ' + error.message);
      console.error('Error shutting down server:', error);
    }
  };

  const thCls = 'text-left px-3 py-3 border-b-2 border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wide bg-gray-50';
  const tdCls = 'px-3 py-3 border-b border-gray-100 text-sm text-gray-700';

  return (
    <Layout title="Dashboard - JadeVectorDB">
      {/* ── Header ── */}
      <Card className="mb-6">
        <CardContent className="py-5">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">JadeVectorDB Dashboard</h1>
              {lastUpdated && (
                <p className="text-sm text-gray-500 mt-0.5">
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </p>
              )}
            </div>
            <div className="flex gap-3">
              <Button
                onClick={fetchDashboardData}
                disabled={loading}
                variant="secondary"
              >
                {loading ? 'Refreshing…' : 'Refresh'}
              </Button>
              {isAdmin && (
                <Button variant="destructive" onClick={handleShutdown}>
                  Shutdown Server
                </Button>
              )}
              {!isAdmin && (
                <Button
                  variant="destructive"
                  onClick={handleShutdown}
                  title="Temporary testing button - normally requires admin role"
                >
                  Shutdown Server (Test)
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
        {/* ── Server Info ── */}
        <Card>
          <CardHeader>
            <CardTitle>Server Info</CardTitle>
          </CardHeader>
          <CardContent>
            {systemStatus ? (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <StatusBadge status={systemStatus.status || 'unknown'} className="mb-1" />
                  <p className="text-xs text-gray-500 mt-1">Status</p>
                </div>
                <div>
                  <p className="text-xl font-bold text-indigo-600">{systemStatus.uptime || 'N/A'}</p>
                  <p className="text-xs text-gray-500">Uptime</p>
                </div>
                <div>
                  <p className="text-xl font-bold text-indigo-600">{systemStatus.version || 'N/A'}</p>
                  <p className="text-xs text-gray-500">Version</p>
                </div>
                <div>
                  <p className="text-xl font-bold text-indigo-600">{systemStatus.performance?.database_count ?? 0}</p>
                  <p className="text-xs text-gray-500">Databases</p>
                </div>
              </div>
            ) : (
              loading
                ? <LoadingSpinner size="sm" label="Loading status…" />
                : <EmptyState icon="⚠️" title="Server status unavailable" />
            )}
          </CardContent>
        </Card>

        {/* ── Recent Databases ── */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Databases ({databases.length})</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {loading && databases.length === 0 ? (
              <div className="px-6 py-4"><LoadingSpinner size="sm" label="Loading…" /></div>
            ) : databases.length === 0 ? (
              <div className="px-6 py-4">
                <EmptyState icon="🗄️" title="No databases found" />
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr>
                      <th className={thCls}>Name</th>
                      <th className={thCls}>Dimension</th>
                      <th className={thCls}>Index</th>
                    </tr>
                  </thead>
                  <tbody>
                    {databases.map(db => (
                      <tr key={db.id || db.databaseId} className="hover:bg-gray-50 transition-colors">
                        <td className={tdCls}>
                          <a
                            href={`/databases/${db.id || db.databaseId}`}
                            className="text-indigo-600 font-medium hover:underline"
                          >
                            {db.name}
                          </a>
                        </td>
                        <td className={tdCls}>{db.vectorDimension}</td>
                        <td className={tdCls}>
                          <StatusBadge status="success" label={db.indexType} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── System Resources ── */}
      {systemStatus && (
        <Card className="mb-5">
          <CardHeader>
            <CardTitle>System Resources</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-2xl font-bold text-indigo-600">
                  {systemStatus.system?.cpu_usage_percent != null
                    ? `${Math.round(systemStatus.system.cpu_usage_percent)}%`
                    : 'N/A'}
                </p>
                <p className="text-sm text-gray-500">CPU Usage</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-indigo-600">
                  {systemStatus.system?.memory_usage_percent != null
                    ? `${Math.round(systemStatus.system.memory_usage_percent)}%`
                    : 'N/A'}
                </p>
                <p className="text-sm text-gray-500">Memory Usage</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-indigo-600">
                  {systemStatus.system?.disk_usage_percent != null
                    ? `${Math.round(systemStatus.system.disk_usage_percent)}%`
                    : 'N/A'}
                </p>
                <p className="text-sm text-gray-500">Disk Usage</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-indigo-600">
                  {systemStatus.performance?.total_vectors ?? 0}
                </p>
                <p className="text-sm text-gray-500">Total Vectors</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Recent Audit Logs ── */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Audit Logs</CardTitle>
        </CardHeader>
        <CardContent>
          {loading && recentLogs.length === 0 ? (
            <LoadingSpinner size="sm" label="Loading logs…" />
          ) : recentLogs.length === 0 ? (
            <EmptyState icon="📋" title="No recent logs" />
          ) : (
            <div className="space-y-2">
              {recentLogs.map((log, index) => (
                <div
                  key={index}
                  className="pl-4 py-3 pr-3 bg-gray-50 rounded-lg border-l-4 border-indigo-400"
                >
                  <p className="text-xs text-gray-500 mb-0.5">
                    {log.timestamp ? new Date(log.timestamp).toLocaleString() : new Date().toLocaleString()}
                    {log.user_id ? ` — ${log.user_id}` : ''}
                  </p>
                  <p className="text-sm text-gray-800">
                    {log.event_type || log.action || 'Log entry'}
                    {log.details ? `: ${log.details}` : ''}
                  </p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </Layout>
  );
}
