import { useEffect, useState } from 'react';
import Layout from '../components/Layout';
import { databaseApi, monitoringApi, securityApi, adminApi, authApi, usersApi } from '../lib/api';

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
      // Optionally redirect to a shutdown confirmation page
      setTimeout(() => {
        window.location.href = '/';
      }, 2000);
    } catch (error) {
      alert('Failed to shutdown server: ' + error.message);
      console.error('Error shutting down server:', error);
    }
  };

  return (
    <Layout title="Dashboard - JadeVectorDB">
      <style jsx>{`
        .dashboard-container {
          max-width: 1400px;
          margin: 0 auto;
          padding: 20px;
        }
        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .dashboard-header h1 {
          font-size: 32px;
          color: #2c3e50;
          margin: 0;
        }
        .last-updated {
          font-size: 14px;
          color: #7f8c8d;
          margin-top: 5px;
        }
        .header-buttons {
          display: flex;
          gap: 10px;
        }
        .btn-refresh {
          padding: 10px 20px;
          background: #3498db;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: all 0.3s ease;
        }
        .btn-refresh:hover:not(:disabled) {
          background: #2980b9;
        }
        .btn-refresh:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .btn-shutdown {
          padding: 10px 20px;
          background: #e74c3c;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: all 0.3s ease;
        }
        .btn-shutdown:hover {
          background: #c0392b;
        }
        .grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
          gap: 20px;
          margin-bottom: 20px;
        }
        .card {
          background: white;
          border-radius: 8px;
          padding: 25px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .card h2 {
          font-size: 20px;
          color: #2c3e50;
          margin: 0 0 20px 0;
          font-weight: 600;
        }
        .empty-state {
          text-align: center;
          padding: 40px;
          color: #7f8c8d;
        }
        table {
          width: 100%;
          border-collapse: collapse;
        }
        thead {
          background: #f8f9fa;
        }
        th {
          text-align: left;
          padding: 12px 10px;
          border-bottom: 2px solid #ecf0f1;
          font-size: 12px;
          color: #7f8c8d;
          text-transform: uppercase;
          font-weight: 600;
        }
        td {
          padding: 12px 10px;
          border-bottom: 1px solid #ecf0f1;
          font-size: 14px;
          color: #2c3e50;
        }
        tr:hover {
          background: #f8f9fa;
        }
        .badge {
          display: inline-block;
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 600;
        }
        .badge-success {
          background: #d4edda;
          color: #155724;
        }
        .badge-error {
          background: #f8d7da;
          color: #721c24;
        }
        .db-link {
          color: #3498db;
          text-decoration: none;
          font-weight: 500;
        }
        .db-link:hover {
          text-decoration: underline;
        }
        .log-entry {
          padding: 12px;
          border-left: 3px solid #3498db;
          background: #f8f9fa;
          margin-bottom: 10px;
          border-radius: 4px;
        }
        .log-time {
          font-size: 12px;
          color: #7f8c8d;
          margin-bottom: 5px;
        }
        .log-message {
          font-size: 14px;
          color: #2c3e50;
        }
        .stat-value {
          font-size: 28px;
          font-weight: 700;
          color: #3498db;
          margin-bottom: 5px;
        }
        .stat-label {
          font-size: 14px;
          color: #7f8c8d;
        }
      `}</style>

      <div className="dashboard-container">
        <div className="dashboard-header">
          <div>
            <h1>JadeVectorDB Dashboard</h1>
            {lastUpdated && (
              <div className="last-updated">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </div>
            )}
          </div>
          <div className="header-buttons">
            <button
              onClick={fetchDashboardData}
              disabled={loading}
              className="btn-refresh"
            >
              {loading ? 'Refreshing...' : 'Refresh'}
            </button>
            {isAdmin && (
              <button
                onClick={handleShutdown}
                className="btn-shutdown"
              >
                Shutdown Server
              </button>
            )}
            {/* Temporary: Show shutdown button for testing */}
            {!isAdmin && (
              <button
                onClick={handleShutdown}
                className="btn-shutdown"
                title="Temporary testing button - normally requires admin role"
              >
                Shutdown Server (Test)
              </button>
            )}
          </div>
        </div>

        <div className="grid">
          <div className="card">
            <h2>Server Info</h2>
            {systemStatus ? (
              <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px' }}>
                <div>
                  <div className="stat-value" style={{ fontSize: '22px' }}>
                    <span className="badge badge-success">{systemStatus.status || 'unknown'}</span>
                  </div>
                  <div className="stat-label">Status</div>
                </div>
                <div>
                  <div className="stat-value" style={{ fontSize: '22px' }}>{systemStatus.uptime || 'N/A'}</div>
                  <div className="stat-label">Uptime</div>
                </div>
                <div>
                  <div className="stat-value" style={{ fontSize: '22px' }}>{systemStatus.version || 'N/A'}</div>
                  <div className="stat-label">Version</div>
                </div>
                <div>
                  <div className="stat-value" style={{ fontSize: '22px' }}>{systemStatus.performance?.database_count ?? 0}</div>
                  <div className="stat-label">Databases</div>
                </div>
              </div>
            ) : (
              <div className="empty-state">{loading ? 'Loading...' : 'Server status unavailable'}</div>
            )}
          </div>

          <div className="card">
            <h2>Recent Databases ({databases.length})</h2>
            {loading && databases.length === 0 ? (
              <div className="empty-state">Loading...</div>
            ) : databases.length === 0 ? (
              <div className="empty-state">No databases found</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Dimension</th>
                    <th>Index</th>
                  </tr>
                </thead>
                <tbody>
                  {databases.map(db => (
                    <tr key={db.id || db.databaseId}>
                      <td>
                        <a href={`/databases/${db.id || db.databaseId}`} className="db-link">
                          {db.name}
                        </a>
                      </td>
                      <td>{db.vectorDimension}</td>
                      <td>
                        <span className="badge badge-success">{db.indexType}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {systemStatus && (
          <div className="card">
            <h2>System Resources</h2>
            <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
              <div>
                <div className="stat-value">{systemStatus.system?.cpu_usage_percent != null ? `${Math.round(systemStatus.system.cpu_usage_percent)}%` : 'N/A'}</div>
                <div className="stat-label">CPU Usage</div>
              </div>
              <div>
                <div className="stat-value">{systemStatus.system?.memory_usage_percent != null ? `${Math.round(systemStatus.system.memory_usage_percent)}%` : 'N/A'}</div>
                <div className="stat-label">Memory Usage</div>
              </div>
              <div>
                <div className="stat-value">{systemStatus.system?.disk_usage_percent != null ? `${Math.round(systemStatus.system.disk_usage_percent)}%` : 'N/A'}</div>
                <div className="stat-label">Disk Usage</div>
              </div>
              <div>
                <div className="stat-value">{systemStatus.performance?.total_vectors ?? 0}</div>
                <div className="stat-label">Total Vectors</div>
              </div>
            </div>
          </div>
        )}

        <div className="card">
          <h2>Recent Audit Logs</h2>
          {loading && recentLogs.length === 0 ? (
            <div className="empty-state">Loading...</div>
          ) : recentLogs.length === 0 ? (
            <div className="empty-state">No recent logs</div>
          ) : (
            <div>
              {recentLogs.map((log, index) => (
                <div key={index} className="log-entry">
                  <div className="log-time">
                    {log.timestamp ? new Date(log.timestamp).toLocaleString() : new Date().toLocaleString()}
                    {log.user_id ? ` — ${log.user_id}` : ''}
                  </div>
                  <div className="log-message">
                    {log.event_type || log.action || 'Log entry'}
                    {log.details ? `: ${log.details}` : ''}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
