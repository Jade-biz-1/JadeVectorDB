import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { monitoringApi, databaseApi, vectorApi } from '../lib/api';
import {
  Button,
  Card, CardHeader, CardTitle, CardContent,
  EmptyState,
  StatusBadge,
} from '../components/ui';

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
      const response = await monitoringApi.systemStatus();

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

      const formattedDbs = await Promise.all(databasesData.map(async (db) => {
        const dbId = db.databaseId || db.id;
        let vectorCount = db.stats?.vectorCount || 0;
        let indexCount = db.stats?.indexCount || 0;
        let storageSize = db.stats?.storageSize || '0 KB';

        if (!db.stats) {
          try {
            const vecResponse = await vectorApi.listVectors(dbId, 1, 0);
            vectorCount = vecResponse.total || 0;
            indexCount = 1;
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
    const interval = setInterval(() => {
      fetchSystemStatus();
      fetchDatabases();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const thCls = 'text-left px-4 py-3 border-b-2 border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wide bg-gray-50';
  const tdCls = 'px-4 py-3 border-b border-gray-100 text-sm text-gray-700';

  // Determine progress bar color
  const progressColor = (value) => {
    if (value > 80) return 'bg-red-500';
    if (value > 60) return 'bg-amber-400';
    return 'bg-emerald-500';
  };

  return (
    <Layout title="Monitoring Dashboard - JadeVectorDB">
      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-1">System Monitoring</h1>
          {lastUpdated && (
            <p className="text-sm text-gray-500">Last updated: {lastUpdated.toLocaleTimeString()}</p>
          )}
        </div>
        <Button
          onClick={() => { fetchSystemStatus(); fetchDatabases(); }}
          disabled={loading}
          variant="secondary"
        >
          {loading ? 'Refreshing…' : 'Refresh Status'}
        </Button>
      </div>

      {/* ── System Overview ── */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>System Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { value: systemStatus?.status === 'operational' ? 'Operational' : systemStatus?.status || 'Unknown', label: 'Overall Status', isStatus: true },
              { value: metrics.totalDatabases, label: 'Total Databases' },
              { value: metrics.totalVectors.toLocaleString(), label: 'Total Vectors' },
              { value: metrics.uptime, label: 'System Uptime' },
            ].map(({ value, label, isStatus }) => (
              <div key={label} className="bg-gray-50 rounded-xl p-4">
                {isStatus
                  ? <StatusBadge status={systemStatus?.status === 'operational' ? 'active' : 'warning'} label={String(value)} className="mb-1" />
                  : <p className="text-2xl font-bold text-indigo-600 mb-1">{value}</p>
                }
                <p className="text-sm text-gray-500">{label}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-6">
        {/* ── Performance Metrics ── */}
        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-5">
              {[
                { name: 'Queries Per Second', value: metrics.qps, max: 100, pct: Math.min(100, (metrics.qps / 100) * 100) },
                { name: 'Avg Query Time (ms)', value: metrics.avgQueryTime, max: 1000, pct: Math.min(100, 100 - (metrics.avgQueryTime / 1000) * 100) },
                { name: 'Storage Utilization', value: `${metrics.storageUsed}%`, max: 100, pct: metrics.storageUsed, colored: true },
              ].map(({ name, value, pct, colored }) => (
                <div key={name}>
                  <div className="flex justify-between text-sm mb-1.5">
                    <span className="font-medium text-gray-700">{name}</span>
                    <span className="font-semibold text-indigo-600">{value}</span>
                  </div>
                  <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${colored ? progressColor(pct) : 'bg-indigo-500'}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* ── Health Checks ── */}
        <Card>
          <CardHeader>
            <CardTitle>Health Checks</CardTitle>
          </CardHeader>
          <CardContent>
            {systemStatus && Object.keys(systemStatus.checks).length > 0 ? (
              <div className="space-y-3">
                {Object.entries(systemStatus.checks).map(([service, status]) => (
                  <div key={service} className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700 capitalize">{service}</span>
                    <StatusBadge status={status === 'ok' ? 'active' : 'error'} label={status} />
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState title="No health check data available" />
            )}
          </CardContent>
        </Card>

        {/* ── Recent Activity ── */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { dot: 'bg-emerald-500', title: 'System Started', desc: 'JadeVectorDB initialized successfully', time: 'Today' },
                { dot: 'bg-indigo-500', title: 'Database Activity', desc: 'Processing vector operations', time: 'Ongoing' },
                { dot: 'bg-amber-400', title: 'Monitoring Active', desc: 'Real-time metrics collection enabled', time: 'Active' },
              ].map(({ dot, title, desc, time }) => (
                <div key={title} className="flex gap-3">
                  <span className={`w-2.5 h-2.5 rounded-full mt-1.5 flex-shrink-0 ${dot}`} />
                  <div>
                    <p className="text-sm font-medium text-gray-800">{title}</p>
                    <p className="text-sm text-gray-500">{desc}</p>
                    <p className="text-xs text-gray-400 mt-0.5">{time}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ── Database Status Table ── */}
      <Card>
        <CardHeader>
          <CardTitle>Database Status</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {databases.length === 0 ? (
            <div className="px-6 py-4">
              <EmptyState icon="🗄️" title="No databases found" description="Unable to fetch database information" />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    <th className={thCls}>Database Name</th>
                    <th className={thCls}>Status</th>
                    <th className={thCls}>Vectors</th>
                    <th className={thCls}>Indexes</th>
                    <th className={thCls}>Storage</th>
                  </tr>
                </thead>
                <tbody>
                  {databases.map((db) => (
                    <tr key={db.id} className="hover:bg-gray-50 transition-colors">
                      <td className={tdCls}>{db.name}</td>
                      <td className={tdCls}>
                        <StatusBadge
                          status={db.status === 'online' ? 'active' : db.status === 'warning' ? 'warning' : 'error'}
                          label={db.status}
                        />
                      </td>
                      <td className={tdCls}>{db.vectors.toLocaleString()}</td>
                      <td className={tdCls}>{db.indexes}</td>
                      <td className={tdCls}>{db.storage}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </Layout>
  );
}
