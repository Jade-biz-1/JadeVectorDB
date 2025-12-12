import { useEffect, useState } from 'react';
import Layout from '../components/Layout';
import { clusterApi, databaseApi, monitoringApi, securityApi } from '../lib/api';

export default function Dashboard() {
  const [nodes, setNodes] = useState([]);
  const [databases, setDatabases] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [recentLogs, setRecentLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const [nodesRes, dbRes, statusRes, logsRes] = await Promise.all([
        clusterApi.listNodes().catch(() => ({ nodes: [] })),
        databaseApi.listDatabases(5, 0).catch(() => ({ databases: [] })),
        monitoringApi.systemStatus().catch(() => null),
        securityApi.listAuditLogs(5, 0).catch(() => ({ logs: [] }))
      ]);

      setNodes(nodesRes.nodes || []);
      setDatabases(dbRes.databases || []);
      setSystemStatus(statusRes || null);
      setRecentLogs(logsRes.logs || []);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setLoading(false);
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
          <button
            onClick={fetchDashboardData}
            disabled={loading}
            className="btn-refresh"
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>

        <div className="grid">
          <div className="card">
            <h2>Cluster Status ({nodes.length} nodes)</h2>
            {loading && nodes.length === 0 ? (
              <div className="empty-state">Loading...</div>
            ) : nodes.length === 0 ? (
              <div className="empty-state">No cluster nodes found</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Node ID</th>
                    <th>Role</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {nodes.map(node => (
                    <tr key={node.id}>
                      <td>{node.id}</td>
                      <td>{node.role || 'worker'}</td>
                      <td>
                        <span className={`badge ${node.status === 'active' ? 'badge-success' : 'badge-error'}`}>
                          {node.status || 'unknown'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
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
            <h2>System Status</h2>
            <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
              <div>
                <div className="stat-value">{systemStatus.uptime || 'N/A'}</div>
                <div className="stat-label">Uptime</div>
              </div>
              <div>
                <div className="stat-value">{systemStatus.cpu || 'N/A'}</div>
                <div className="stat-label">CPU Usage</div>
              </div>
              <div>
                <div className="stat-value">{systemStatus.memory || 'N/A'}</div>
                <div className="stat-label">Memory Usage</div>
              </div>
              <div>
                <div className="stat-value">{systemStatus.requests || '0'}</div>
                <div className="stat-label">Total Requests</div>
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
                  <div className="log-time">{log.timestamp || new Date().toLocaleString()}</div>
                  <div className="log-message">{log.message || log.action || 'Log entry'}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
