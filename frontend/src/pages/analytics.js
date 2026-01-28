import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { analyticsApi, databaseApi } from '../lib/api';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function AnalyticsDashboard() {
  const [selectedDatabase, setSelectedDatabase] = useState(null);
  const [databases, setDatabases] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Analytics data
  const [statistics, setStatistics] = useState([]);
  const [insights, setInsights] = useState(null);
  const [queryPatterns, setQueryPatterns] = useState([]);
  const [recentQueries, setRecentQueries] = useState([]);
  const [trendingQueries, setTrendingQueries] = useState([]);

  // UI state
  const [timeRange, setTimeRange] = useState('24h');
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch databases on component mount
  useEffect(() => {
    fetchDatabases();
  }, []);

  // Fetch analytics when database is selected
  useEffect(() => {
    if (selectedDatabase) {
      fetchAnalytics();
      // Auto-refresh every 30 seconds
      const interval = setInterval(fetchAnalytics, 30000);
      return () => clearInterval(interval);
    }
  }, [selectedDatabase, timeRange]);

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      const databasesData = response.databases || [];
      setDatabases(databasesData);

      // Auto-select first database if available
      if (databasesData.length > 0 && !selectedDatabase) {
        setSelectedDatabase(databasesData[0].databaseId || databasesData[0].id);
      }
    } catch (error) {
      console.error('Error fetching databases:', error);
    }
  };

  const getTimeRangeMs = () => {
    const now = Date.now();
    const ranges = {
      '1h': 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000,
    };
    return { startTime: now - ranges[timeRange], endTime: now };
  };

  const fetchAnalytics = async () => {
    if (!selectedDatabase) return;

    setLoading(true);
    try {
      const { startTime, endTime } = getTimeRangeMs();
      const granularity = timeRange === '1h' ? 'hourly' : timeRange === '24h' ? 'hourly' : 'daily';

      // Fetch all analytics data in parallel
      const [statsData, insightsData, patternsData, queriesData, trendingData] = await Promise.all([
        analyticsApi.getStatistics(selectedDatabase, { startTime, endTime, granularity }),
        analyticsApi.getInsights(selectedDatabase, { startTime, endTime }),
        analyticsApi.getQueryPatterns(selectedDatabase, { minCount: 2, limit: 10 }),
        analyticsApi.getRecentQueries(selectedDatabase, { limit: 50, startTime, endTime }),
        analyticsApi.getTrendingQueries(selectedDatabase, { timeBucket: 'daily', minGrowthRate: 50 }),
      ]);

      setStatistics(statsData.statistics || []);
      setInsights(insightsData);
      setQueryPatterns(patternsData.patterns || []);
      setRecentQueries(queriesData.queries || []);
      setTrendingQueries(trendingData.trending || []);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatTimestamp = (ts) => {
    return new Date(ts).toLocaleString();
  };

  // Calculate summary metrics from insights
  const summaryMetrics = insights ? {
    totalQueries: insights.summary?.total_queries || 0,
    successRate: insights.summary?.success_rate ? `${(insights.summary.success_rate * 100).toFixed(1)}%` : '0%',
    avgLatency: insights.summary?.avg_latency_ms ? formatDuration(insights.summary.avg_latency_ms) : '0ms',
    qps: insights.summary?.queries_per_second ? insights.summary.queries_per_second.toFixed(2) : '0',
  } : {
    totalQueries: 0,
    successRate: '0%',
    avgLatency: '0ms',
    qps: '0',
  };

  return (
    <Layout title="Query Analytics - JadeVectorDB">
      <style jsx>{`
        .analytics-container {
          max-width: 1600px;
          margin: 0 auto;
          padding: 20px;
        }

        .page-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
          flex-wrap: wrap;
          gap: 15px;
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

        .header-controls {
          display: flex;
          gap: 15px;
          align-items: center;
          flex-wrap: wrap;
        }

        .db-select {
          padding: 10px 15px;
          border: 1px solid #ddd;
          border-radius: 6px;
          font-size: 14px;
          background: white;
          cursor: pointer;
          min-width: 200px;
        }

        .time-range-buttons {
          display: flex;
          gap: 5px;
          background: #f8f9fa;
          padding: 4px;
          border-radius: 6px;
        }

        .time-btn {
          padding: 8px 16px;
          background: transparent;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          color: #555;
          transition: all 0.2s;
        }

        .time-btn:hover {
          background: #e9ecef;
        }

        .time-btn.active {
          background: #3498db;
          color: white;
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

        .tabs {
          display: flex;
          gap: 10px;
          margin-bottom: 25px;
          border-bottom: 2px solid #ecf0f1;
        }

        .tab-btn {
          padding: 12px 24px;
          background: transparent;
          border: none;
          border-bottom: 3px solid transparent;
          cursor: pointer;
          font-size: 15px;
          font-weight: 500;
          color: #7f8c8d;
          transition: all 0.2s;
        }

        .tab-btn:hover {
          color: #3498db;
        }

        .tab-btn.active {
          color: #3498db;
          border-bottom-color: #3498db;
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
          grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
          gap: 20px;
          margin-bottom: 25px;
        }

        .metric-card {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          padding: 25px;
          border-radius: 12px;
          color: white;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .metric-card:nth-child(2) {
          background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .metric-card:nth-child(3) {
          background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }

        .metric-card:nth-child(4) {
          background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }

        .metric-value {
          font-size: 36px;
          font-weight: 700;
          margin-bottom: 8px;
        }

        .metric-label {
          font-size: 14px;
          opacity: 0.9;
        }

        .chart-container {
          background: white;
          border-radius: 8px;
          padding: 25px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          margin-bottom: 25px;
          min-height: 400px;
        }

        .insights-panel {
          background: #f8f9fa;
          border-radius: 8px;
          padding: 20px;
        }

        .insight-item {
          padding: 15px;
          background: white;
          border-radius: 6px;
          margin-bottom: 12px;
          border-left: 4px solid #3498db;
        }

        .insight-item.warning {
          border-left-color: #f39c12;
        }

        .insight-item.error {
          border-left-color: #e74c3c;
        }

        .insight-item.success {
          border-left-color: #27ae60;
        }

        .insight-title {
          font-size: 14px;
          font-weight: 600;
          color: #2c3e50;
          margin-bottom: 5px;
        }

        .insight-description {
          font-size: 13px;
          color: #7f8c8d;
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
          padding: 60px 20px;
          color: #7f8c8d;
        }

        .empty-state-icon {
          font-size: 48px;
          margin-bottom: 15px;
          opacity: 0.5;
        }

        .grid-2 {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
          gap: 25px;
        }

        .query-text {
          font-family: 'Courier New', monospace;
          background: #f8f9fa;
          padding: 8px 12px;
          border-radius: 4px;
          font-size: 13px;
          max-width: 400px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .badge {
          display: inline-block;
          padding: 4px 10px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 600;
        }

        .badge.success {
          background: #d4edda;
          color: #155724;
        }

        .badge.error {
          background: #f8d7da;
          color: #721c24;
        }

        .badge.warning {
          background: #fff3cd;
          color: #856404;
        }

        .badge.info {
          background: #d1ecf1;
          color: #0c5460;
        }
      `}</style>

      <div className="analytics-container">
        {/* Header */}
        <div className="page-header">
          <div className="header-left">
            <h1>Query Analytics</h1>
            {lastUpdated && (
              <div className="last-updated">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </div>
            )}
          </div>
          <div className="header-controls">
            <select
              className="db-select"
              value={selectedDatabase || ''}
              onChange={(e) => setSelectedDatabase(e.target.value)}
            >
              <option value="">Select Database</option>
              {databases.map((db) => (
                <option key={db.databaseId || db.id} value={db.databaseId || db.id}>
                  {db.name}
                </option>
              ))}
            </select>

            <div className="time-range-buttons">
              {['1h', '24h', '7d', '30d'].map((range) => (
                <button
                  key={range}
                  className={`time-btn ${timeRange === range ? 'active' : ''}`}
                  onClick={() => setTimeRange(range)}
                >
                  {range}
                </button>
              ))}
            </div>

            <button
              onClick={fetchAnalytics}
              disabled={loading || !selectedDatabase}
              className="btn-refresh"
            >
              {loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>

        {!selectedDatabase ? (
          <div className="empty-state">
            <div className="empty-state-icon">📊</div>
            <h2>No Database Selected</h2>
            <p>Please select a database to view analytics</p>
          </div>
        ) : (
          <>
            {/* Tabs */}
            <div className="tabs">
              <button
                className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveTab('overview')}
              >
                Overview
              </button>
              <button
                className={`tab-btn ${activeTab === 'queries' ? 'active' : ''}`}
                onClick={() => setActiveTab('queries')}
              >
                Query Explorer
              </button>
              <button
                className={`tab-btn ${activeTab === 'patterns' ? 'active' : ''}`}
                onClick={() => setActiveTab('patterns')}
              >
                Patterns
              </button>
              <button
                className={`tab-btn ${activeTab === 'insights' ? 'active' : ''}`}
                onClick={() => setActiveTab('insights')}
              >
                Insights
              </button>
            </div>

            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <>
                {/* Key Metrics */}
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-value">{summaryMetrics.totalQueries.toLocaleString()}</div>
                    <div className="metric-label">Total Queries</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{summaryMetrics.successRate}</div>
                    <div className="metric-label">Success Rate</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{summaryMetrics.avgLatency}</div>
                    <div className="metric-label">Avg Latency</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{summaryMetrics.qps}</div>
                    <div className="metric-label">Queries Per Second</div>
                  </div>
                </div>

                {/* Time Series Chart */}
                <div className="chart-container">
                  <h2 className="card-title">Queries Over Time</h2>
                  {statistics.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={statistics}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="time_bucket"
                          tickFormatter={(value) => {
                            const date = new Date(value);
                            return timeRange === '1h' || timeRange === '24h'
                              ? date.toLocaleTimeString()
                              : date.toLocaleDateString();
                          }}
                        />
                        <YAxis />
                        <Tooltip
                          labelFormatter={(value) => new Date(value).toLocaleString()}
                        />
                        <Legend />
                        <Line type="monotone" dataKey="total_queries" stroke="#3498db" name="Total Queries" />
                        <Line type="monotone" dataKey="successful_queries" stroke="#27ae60" name="Successful" />
                        <Line type="monotone" dataKey="failed_queries" stroke="#e74c3c" name="Failed" />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty-state">No query data available for this time range</div>
                  )}
                </div>

                {/* Latency Distribution */}
                {statistics.length > 0 && (
                  <div className="chart-container">
                    <h2 className="card-title">Latency Distribution</h2>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={statistics}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="time_bucket"
                          tickFormatter={(value) => {
                            const date = new Date(value);
                            return timeRange === '1h' || timeRange === '24h'
                              ? date.toLocaleTimeString()
                              : date.toLocaleDateString();
                          }}
                        />
                        <YAxis label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip labelFormatter={(value) => new Date(value).toLocaleString()} />
                        <Legend />
                        <Bar dataKey="avg_latency_ms" fill="#3498db" name="Avg" />
                        <Bar dataKey="p95_latency_ms" fill="#f39c12" name="P95" />
                        <Bar dataKey="p99_latency_ms" fill="#e74c3c" name="P99" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Top Patterns and Slow Queries Grid */}
                <div className="grid-2">
                  <div className="card">
                    <h2 className="card-title">Top Query Patterns</h2>
                    <div className="table-container">
                      {queryPatterns.length > 0 ? (
                        <table>
                          <thead>
                            <tr>
                              <th>Pattern</th>
                              <th>Count</th>
                              <th>Avg Latency</th>
                            </tr>
                          </thead>
                          <tbody>
                            {queryPatterns.slice(0, 5).map((pattern, idx) => (
                              <tr key={idx}>
                                <td>
                                  <div className="query-text">{pattern.normalized_text || pattern.pattern}</div>
                                </td>
                                <td>{pattern.count}</td>
                                <td>{formatDuration(pattern.avg_latency_ms)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <div className="empty-state">No query patterns found</div>
                      )}
                    </div>
                  </div>

                  <div className="card">
                    <h2 className="card-title">Slow Queries</h2>
                    <div className="table-container">
                      {insights?.slow_queries && insights.slow_queries.length > 0 ? (
                        <table>
                          <thead>
                            <tr>
                              <th>Query</th>
                              <th>Latency</th>
                              <th>Time</th>
                            </tr>
                          </thead>
                          <tbody>
                            {insights.slow_queries.slice(0, 5).map((query, idx) => (
                              <tr key={idx}>
                                <td>
                                  <div className="query-text">{query.query_text || 'N/A'}</div>
                                </td>
                                <td>
                                  <span className="badge warning">{formatDuration(query.total_time_ms)}</span>
                                </td>
                                <td>{new Date(query.timestamp).toLocaleTimeString()}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <div className="empty-state">No slow queries detected</div>
                      )}
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Query Explorer Tab */}
            {activeTab === 'queries' && (
              <div className="card">
                <h2 className="card-title">Recent Queries</h2>
                <div className="table-container">
                  {recentQueries.length > 0 ? (
                    <table>
                      <thead>
                        <tr>
                          <th>Timestamp</th>
                          <th>Query Text</th>
                          <th>Type</th>
                          <th>Results</th>
                          <th>Latency</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {recentQueries.map((query, idx) => (
                          <tr key={idx}>
                            <td>{formatTimestamp(query.timestamp)}</td>
                            <td>
                              <div className="query-text">{query.query_text || 'N/A'}</div>
                            </td>
                            <td>
                              <span className="badge info">{query.query_type || 'vector'}</span>
                            </td>
                            <td>{query.num_results || 0}</td>
                            <td>{formatDuration(query.total_time_ms)}</td>
                            <td>
                              <span className={`badge ${query.has_error ? 'error' : 'success'}`}>
                                {query.has_error ? 'Failed' : 'Success'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="empty-state">No recent queries found</div>
                  )}
                </div>
              </div>
            )}

            {/* Patterns Tab */}
            {activeTab === 'patterns' && (
              <>
                <div className="card">
                  <h2 className="card-title">Common Query Patterns</h2>
                  <div className="table-container">
                    {queryPatterns.length > 0 ? (
                      <table>
                        <thead>
                          <tr>
                            <th>Pattern</th>
                            <th>Count</th>
                            <th>Avg Latency</th>
                            <th>Avg Results</th>
                            <th>First Seen</th>
                            <th>Last Seen</th>
                          </tr>
                        </thead>
                        <tbody>
                          {queryPatterns.map((pattern, idx) => (
                            <tr key={idx}>
                              <td>
                                <div className="query-text">{pattern.normalized_text || pattern.pattern}</div>
                              </td>
                              <td>{pattern.count}</td>
                              <td>{formatDuration(pattern.avg_latency_ms)}</td>
                              <td>{pattern.avg_results?.toFixed(1) || '0'}</td>
                              <td>{formatTimestamp(pattern.first_seen)}</td>
                              <td>{formatTimestamp(pattern.last_seen)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <div className="empty-state">No query patterns found</div>
                    )}
                  </div>
                </div>

                {trendingQueries.length > 0 && (
                  <div className="card">
                    <h2 className="card-title">Trending Queries</h2>
                    <div className="table-container">
                      <table>
                        <thead>
                          <tr>
                            <th>Query Pattern</th>
                            <th>Current Count</th>
                            <th>Previous Count</th>
                            <th>Growth Rate</th>
                          </tr>
                        </thead>
                        <tbody>
                          {trendingQueries.map((query, idx) => (
                            <tr key={idx}>
                              <td>
                                <div className="query-text">{query.normalized_text || query.query_text}</div>
                              </td>
                              <td>{query.current_count}</td>
                              <td>{query.previous_count}</td>
                              <td>
                                <span className="badge success">
                                  +{query.growth_rate?.toFixed(1) || '0'}%
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Insights Tab */}
            {activeTab === 'insights' && (
              <>
                {insights && (
                  <div className="card">
                    <h2 className="card-title">Automated Insights & Recommendations</h2>
                    <div className="insights-panel">
                      {/* Summary Insights */}
                      <div className="insight-item success">
                        <div className="insight-title">Query Performance</div>
                        <div className="insight-description">
                          {insights.summary?.total_queries || 0} queries processed with{' '}
                          {((insights.summary?.success_rate || 0) * 100).toFixed(1)}% success rate.
                          Average latency: {formatDuration(insights.summary?.avg_latency_ms || 0)}
                        </div>
                      </div>

                      {/* Peak Hour */}
                      {insights.summary?.peak_hour && (
                        <div className="insight-item info">
                          <div className="insight-title">Peak Usage</div>
                          <div className="insight-description">
                            Highest traffic at {new Date(insights.summary.peak_hour).toLocaleTimeString()}{' '}
                            with {insights.summary.peak_hour_queries} queries
                          </div>
                        </div>
                      )}

                      {/* Slow Queries Warning */}
                      {insights.slow_queries && insights.slow_queries.length > 0 && (
                        <div className="insight-item warning">
                          <div className="insight-title">Performance Alert</div>
                          <div className="insight-description">
                            {insights.slow_queries.length} slow queries detected (&gt;{' '}
                            {formatDuration(1000)}). Consider optimizing indexes or query patterns.
                          </div>
                        </div>
                      )}

                      {/* Zero Results Warning */}
                      {insights.zero_result_queries && insights.zero_result_queries.length > 0 && (
                        <div className="insight-item error">
                          <div className="insight-title">Content Gaps</div>
                          <div className="insight-description">
                            {insights.zero_result_queries.length} query patterns returned zero results.
                            Consider adding relevant content or improving search quality.
                          </div>
                        </div>
                      )}

                      {/* Trending Queries */}
                      {insights.trending_queries && insights.trending_queries.length > 0 && (
                        <div className="insight-item success">
                          <div className="insight-title">Trending Topics</div>
                          <div className="insight-description">
                            {insights.trending_queries.length} queries showing significant growth.
                            Monitor these patterns for content opportunities.
                          </div>
                        </div>
                      )}

                      {/* No Insights */}
                      {(!insights.slow_queries || insights.slow_queries.length === 0) &&
                       (!insights.zero_result_queries || insights.zero_result_queries.length === 0) &&
                       (!insights.trending_queries || insights.trending_queries.length === 0) && (
                        <div className="insight-item info">
                          <div className="insight-title">All Good!</div>
                          <div className="insight-description">
                            No significant issues detected. Your search system is performing well.
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Zero Result Queries */}
                {insights?.zero_result_queries && insights.zero_result_queries.length > 0 && (
                  <div className="card">
                    <h2 className="card-title">Zero-Result Queries</h2>
                    <div className="table-container">
                      <table>
                        <thead>
                          <tr>
                            <th>Query Pattern</th>
                            <th>Occurrence Count</th>
                          </tr>
                        </thead>
                        <tbody>
                          {insights.zero_result_queries.map((query, idx) => (
                            <tr key={idx}>
                              <td>
                                <div className="query-text">{query.normalized_text || query.query_text}</div>
                              </td>
                              <td>{query.count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </Layout>
  );
}
