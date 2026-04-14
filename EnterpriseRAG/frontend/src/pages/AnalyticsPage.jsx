import { useState, useEffect } from 'react';
import { queryAPI } from '../services/api';
import '../styles/AnalyticsPage.css';

function StatCard({ label, value, sub }) {
  return (
    <div className="stat-card">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export default function AnalyticsPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    queryAPI.getAnalytics(20)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="analytics-loading">Loading analytics…</div>;
  if (error)   return <div className="analytics-error">Error: {error}</div>;
  if (!data)   return null;

  const successPct = (data.success_rate * 100).toFixed(1);
  const avgConf    = (data.avg_confidence * 100).toFixed(1);
  const avgMs      = Math.round(data.avg_processing_time_ms);

  return (
    <div className="analytics-page">
      <h2 className="analytics-title">Query Analytics</h2>

      {/* Summary cards */}
      <div className="stat-cards">
        <StatCard label="Total Queries"      value={data.total_queries.toLocaleString()} />
        <StatCard label="Success Rate"       value={`${successPct}%`} />
        <StatCard label="Avg Confidence"     value={`${avgConf}%`} />
        <StatCard label="Avg Response Time"  value={`${avgMs} ms`} />
      </div>

      {/* Category breakdown */}
      <section className="analytics-section">
        <h3>Queries by Category</h3>
        {Object.keys(data.category_breakdown ?? {}).length === 0 ? (
          <p className="empty-msg">No queries yet.</p>
        ) : (
          <table className="analytics-table">
            <thead>
              <tr><th>Category</th><th>Queries</th></tr>
            </thead>
            <tbody>
              {Object.entries(data.category_breakdown ?? {})
                .sort((a, b) => b[1] - a[1])
                .map(([cat, count]) => (
                  <tr key={cat}>
                    <td>{cat}</td>
                    <td>{count}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Recent queries */}
      <section className="analytics-section">
        <h3>Recent Queries</h3>
        {data.recent_queries.length === 0 ? (
          <p className="empty-msg">No queries recorded yet.</p>
        ) : (
          <table className="analytics-table recent-table">
            <thead>
              <tr>
                <th>Question</th>
                <th>Category</th>
                <th>Confidence</th>
                <th>Time (ms)</th>
                <th>Sources</th>
                <th>Status</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {data.recent_queries.map(q => (
                <tr key={q.id} className={q.success ? '' : 'row-failed'}>
                  <td className="col-question" title={q.question}>
                    {q.question.length > 60 ? q.question.slice(0, 60) + '…' : q.question}
                  </td>
                  <td>{q.category}</td>
                  <td>{(q.confidence * 100).toFixed(0)}%</td>
                  <td>{q.processing_time_ms}</td>
                  <td>{q.sources_count}</td>
                  <td>
                    <span className={`badge ${q.success ? 'badge-ok' : 'badge-fail'}`}>
                      {q.success ? 'OK' : 'FAIL'}
                    </span>
                  </td>
                  <td className="col-ts">{q.timestamp.replace('T', ' ').slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
