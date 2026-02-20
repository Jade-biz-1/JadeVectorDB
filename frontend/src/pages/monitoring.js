import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { monitoringApi, databaseApi, vectorApi } from '../lib/api';

export default function MonitoringDashboard() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [databases, setDatabases] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [metrics, setMetrics] = useState({
    totalDatabases: 0,
    totalVectors: 0,
    qps: 0,
    avgQueryTime: 0,
    storageUsed: 0,
    uptime: '0 minutes'
  });

  const fetchSystemStatus = async () => {
    setLoading(true);
    try {
      // Fetch real system status from backend
      const response = await monitoringApi.systemStatus();

      // Set system status
      setSystemStatus({
        status: response.status || 'operational',
        checks: response.checks || {
          database: 'ok',
          storage: 'ok',
          network: 'ok',
          memory: 'ok',
          cpu: 'ok'
        },
        timestamp: response.timestamp || new Date().toISOString()
      });

      // Extract metrics from response - map backend field names to frontend state
      const perf = response.performance || {};
      const sys = response.system || {};
      setMetrics({
        totalDatabases: perf.database_count || 0,
        totalVectors: perf.total_vectors || 0,
        qps: perf.active_connections || 0,
        avgQueryTime: perf.avg_query_time_ms || 0,
        storageUsed: Math.round(sys.disk_usage_percent || 0),
        uptime: response.uptime || '0 minutes'
      });

      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching system status:', error);
      // Fallback to basic status
      setSystemStatus({
        status: 'unknown',
        checks: {},
        timestamp: new Date().toISOString()
      });
    } finally {
      setLoading(false);
    }
  };

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      const databasesData = response.databases || [];

      // Fetch per-database vector counts in parallel
      const formattedDbs = await Promise.all(databasesData.map(async (db) => {
        const dbId = db.databaseId || db.id;
        let vectorCount = db.stats?.vectorCount || 0;
        let indexCount = db.stats?.indexCount || 0;
        let storageSize = db.stats?.storageSize || '0 KB';

        // If stats not included in list response, fetch vector count directly
        if (!db.stats) {
          try {
            const vecResponse = await vectorApi.listVectors(dbId, 1, 0);
            vectorCount = vecResponse.total || 0;
            indexCount = 1; // Each database has its configured index
            // Estimate storage: vectors * dimension * 4 bytes (float32)
            const dim = db.vectorDimension || 0;
            const bytes = vectorCount * dim * 4;
            if (bytes >= 1024 * 1024) {
              storageSize = (bytes / (1024 * 1024)).toFixed(1) + ' MB';
            } else if (bytes >= 1024) {
              storageSize = (bytes / 1024).toFixed(1) + ' KB';
            } else {
              storageSize = bytes + ' B';
            }
          } catch (e) {
            // Silently fall back to defaults
          }
        }

        return {
          id: dbId,
          name: db.name,
          status: db.status || 'online',
          vectors: vectorCount,
          indexes: indexCount,
          storage: storageSize
        };
      }));

      setDatabases(formattedDbs);
    } catch (error) {
      console.error('Error fetching databases:', error);
      setDatabases([]);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    fetchDatabases();
    // Refresh status every 30 seconds
    const interval = setInterval(() => {
      fetchSystemStatus();
      fetchDatabases();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Layout title="Monitoring Dashboard - JadeVectorDB">
      <style jsx>{`
        .monitoring-container {
          max-width: 1400px;
          margin: 0 auto;
          padding: 20px;
        }

        .page-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
        }

        .header-left h1 {
          font-size: 32px;
          font-weight: 700;
          color: #2c3e50;
          margin: 0 0 5px 0;
        }

        .last-updated {
          font-size: 14px;
          color: #7f8c8d;
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

        .card {
          background: white;
          border-radius: 8px;
          padding: 25px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          margin-bottom: 25px;
        }

        .card-title {
          font-size: 20px;
          font-weight: 600;
          color: #2c3e50;
          margin: 0 0 20px 0;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px;
        }

        .metric-box {
          background: #f8f9fa;
          padding: 20px;
          border-radius: 6px;
        }

        .metric-value {
          font-size: 28px;
          font-weight: 700;
          color: #3498db;
          margin-bottom: 5px;
        }

        .metric-value.status-ok {
          color: #27ae60;
        }

        .metric-label {
          font-size: 14px;
          color: #7f8c8d;
        }

        .grid-3 {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
          gap: 25px;
          margin-bottom: 25px;
        }

        .health-checks {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .health-check-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .health-check-label {
          font-size: 14px;
          font-weight: 500;
          color: #2c3e50;
          text-transform: capitalize;
        }

        .status-badge {
          display: inline-flex;
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 600;
        }

        .status-badge.ok {
          background: #d4edda;
          color: #155724;
        }

        .status-badge.error {
          background: #f8d7da;
          color: #721c24;
        }

        .status-badge.warning {
          background: #fff3cd;
          color: #856404;
        }

        .performance-metrics {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .metric-row {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .metric-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .metric-name {
          font-size: 14px;
          font-weight: 500;
          color: #2c3e50;
        }

        .metric-number {
          font-size: 14px;
          font-weight: 600;
          color: #3498db;
        }

        .progress-bar {
          width: 100%;
          height: 8px;
          background: #ecf0f1;
          border-radius: 4px;
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          background: #3498db;
          border-radius: 4px;
          transition: width 0.3s ease;
        }

        .progress-fill.green {
          background: #27ae60;
        }

        .progress-fill.yellow {
          background: #f39c12;
        }

        .progress-fill.red {
          background: #e74c3c;
        }

        .activity-list {
          display: flex;
          flex-direction: column;
          gap: 15px;
        }

        .activity-item {
          display: flex;
          gap: 12px;
        }

        .activity-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          margin-top: 5px;
          flex-shrink: 0;
        }

        .activity-dot.green {
          background: #27ae60;
        }

        .activity-dot.blue {
          background: #3498db;
        }

        .activity-dot.yellow {
          background: #f39c12;
        }

        .activity-content {
          flex: 1;
        }

        .activity-title {
          font-size: 14px;
          font-weight: 500;
          color: #2c3e50;
          margin-bottom: 2px;
        }

        .activity-description {
          font-size: 13px;
          color: #7f8c8d;
          margin-bottom: 3px;
        }

        .activity-time {
          font-size: 12px;
          color: #95a5a6;
        }

        .table-container {
          overflow-x: auto;
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
          padding: 12px 15px;
          border-bottom: 2px solid #ecf0f1;
          font-size: 12px;
          color: #7f8c8d;
          text-transform: uppercase;
          font-weight: 600;
        }

        td {
          padding: 15px;
          border-bottom: 1px solid #ecf0f1;
          font-size: 14px;
          color: #2c3e50;
        }

        tbody tr:hover {
          background: #f8f9fa;
        }

        .empty-state {
          text-align: center;
          padding: 40px;
          color: #7f8c8d;
        }
      `}</style>

      <div className="monitoring-container">
        <div className="page-header">
          <div className="header-left">
            <h1>System Monitoring</h1>
            {lastUpdated && (
              <div className="last-updated">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </div>
            )}
          </div>
          <button
            onClick={() => {
              fetchSystemStatus();
              fetchDatabases();
            }}
            disabled={loading}
            className="btn-refresh"
          >
            {loading ? 'Refreshing...' : 'Refresh Status'}
          </button>
        </div>

        <div className="card">
          <h2 className="card-title">System Overview</h2>
          <div className="metrics-grid">
            <div className="metric-box">
              <div className="metric-value status-ok">
                {systemStatus?.status === 'operational' ? 'Operational' : systemStatus?.status || 'Unknown'}
              </div>
              <div className="metric-label">Overall Status</div>
            </div>
            <div className="metric-box">
              <div className="metric-value">{metrics.totalDatabases}</div>
              <div className="metric-label">Total Databases</div>
            </div>
            <div className="metric-box">
              <div className="metric-value">{metrics.totalVectors.toLocaleString()}</div>
              <div className="metric-label">Total Vectors</div>
            </div>
            <div className="metric-box">
              <div className="metric-value">{metrics.uptime}</div>
              <div className="metric-label">System Uptime</div>
            </div>
          </div>
        </div>

        <div className="grid-3">
          <div className="card">
            <h2 className="card-title">Performance Metrics</h2>
            <div className="performance-metrics">
              <div className="metric-row">
                <div className="metric-header">
                  <span className="metric-name">Queries Per Second</span>
                  <span className="metric-number">{metrics.qps}</span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill green"
                    style={{ width: `${Math.min(100, (metrics.qps / 100) * 100)}%` }}
                  />
                </div>
              </div>

              <div className="metric-row">
                <div className="metric-header">
                  <span className="metric-name">Avg Query Time (ms)</span>
                  <span className="metric-number">{metrics.avgQueryTime}</span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.min(100, 100 - (metrics.avgQueryTime / 1000) * 100)}%` }}
                  />
                </div>
              </div>

              <div className="metric-row">
                <div className="metric-header">
                  <span className="metric-name">Storage Utilization</span>
                  <span className="metric-number">{metrics.storageUsed}%</span>
                </div>
                <div className="progress-bar">
                  <div
                    className={`progress-fill ${metrics.storageUsed > 80 ? 'red' : metrics.storageUsed > 60 ? 'yellow' : 'green'}`}
                    style={{ width: `${metrics.storageUsed}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <h2 className="card-title">Health Checks</h2>
            <div className="health-checks">
              {systemStatus && Object.keys(systemStatus.checks).length > 0 ? (
                Object.entries(systemStatus.checks).map(([service, status]) => (
                  <div key={service} className="health-check-item">
                    <span className="health-check-label">{service}</span>
                    <span className={`status-badge ${status === 'ok' ? 'ok' : 'error'}`}>
                      {status}
                    </span>
                  </div>
                ))
              ) : (
                <div className="empty-state">No health check data available</div>
              )}
            </div>
          </div>

          <div className="card">
            <h2 className="card-title">Recent Activity</h2>
            <div className="activity-list">
              <div className="activity-item">
                <div className="activity-dot green" />
                <div className="activity-content">
                  <div className="activity-title">System Started</div>
                  <div className="activity-description">JadeVectorDB initialized successfully</div>
                  <div className="activity-time">Today</div>
                </div>
              </div>
              <div className="activity-item">
                <div className="activity-dot blue" />
                <div className="activity-content">
                  <div className="activity-title">Database Activity</div>
                  <div className="activity-description">Processing vector operations</div>
                  <div className="activity-time">Ongoing</div>
                </div>
              </div>
              <div className="activity-item">
                <div className="activity-dot yellow" />
                <div className="activity-content">
                  <div className="activity-title">Monitoring Active</div>
                  <div className="activity-description">Real-time metrics collection enabled</div>
                  <div className="activity-time">Active</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <h2 className="card-title">Database Status</h2>
          <div className="table-container">
            {databases.length === 0 ? (
              <div className="empty-state">No databases found or unable to fetch database information</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Database Name</th>
                    <th>Status</th>
                    <th>Vectors</th>
                    <th>Indexes</th>
                    <th>Storage</th>
                  </tr>
                </thead>
                <tbody>
                  {databases.map((db) => (
                    <tr key={db.id}>
                      <td>{db.name}</td>
                      <td>
                        <span className={`status-badge ${db.status === 'online' ? 'ok' : db.status === 'warning' ? 'warning' : 'error'}`}>
                          {db.status}
                        </span>
                      </td>
                      <td>{db.vectors.toLocaleString()}</td>
                      <td>{db.indexes}</td>
                      <td>{db.storage}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}
