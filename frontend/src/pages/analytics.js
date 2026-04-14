import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { analyticsApi, databaseApi } from '../lib/api';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import {
  Button,
  Card, CardHeader, CardTitle, CardContent,
  EmptyState,
  StatusBadge,
} from '../components/ui';

export default function AnalyticsDashboard() {
  const [selectedDatabase, setSelectedDatabase] = useState(null);
  const [databases, setDatabases] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  const [statistics, setStatistics] = useState([]);
  const [insights, setInsights] = useState(null);
  const [queryPatterns, setQueryPatterns] = useState([]);
  const [recentQueries, setRecentQueries] = useState([]);
  const [trendingQueries, setTrendingQueries] = useState([]);

  const [timeRange, setTimeRange] = useState('24h');
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchDatabases();
  }, []);

  useEffect(() => {
    if (selectedDatabase) {
      fetchAnalytics();
      const interval = setInterval(fetchAnalytics, 30000);
      return () => clearInterval(interval);
    }
  }, [selectedDatabase, timeRange]);

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      const databasesData = response.databases || [];
      setDatabases(databasesData);

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

  const formatTimestamp = (ts) => new Date(ts).toLocaleString();

  const summaryMetrics = insights ? {
    totalQueries: insights.summary?.total_queries || 0,
    successRate: insights.summary?.success_rate ? `${(insights.summary.success_rate * 100).toFixed(1)}%` : '0%',
    avgLatency: insights.summary?.avg_latency_ms ? formatDuration(insights.summary.avg_latency_ms) : '0ms',
    qps: insights.summary?.queries_per_second ? insights.summary.queries_per_second.toFixed(2) : '0',
  } : { totalQueries: 0, successRate: '0%', avgLatency: '0ms', qps: '0' };

  const inputCls = 'px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-500 transition';

  const thCls = 'text-left px-4 py-3 border-b-2 border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wide bg-gray-50';
  const tdCls = 'px-4 py-3 border-b border-gray-100 text-sm text-gray-700';

  const METRIC_GRADIENTS = [
    'from-violet-500 to-purple-700',
    'from-pink-400 to-rose-500',
    'from-sky-400 to-cyan-400',
    'from-emerald-400 to-teal-400',
  ];

  return (
    <Layout title="Query Analytics - JadeVectorDB">
      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-1">Query Analytics</h1>
          {lastUpdated && (
            <p className="text-sm text-gray-500">Last updated: {lastUpdated.toLocaleTimeString()}</p>
          )}
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <select
            className={`${inputCls} min-w-[180px]`}
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

          <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
            {['1h', '24h', '7d', '30d'].map((range) => (
              <button
                key={range}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  timeRange === range
                    ? 'bg-indigo-600 text-white'
                    : 'text-gray-600 hover:bg-gray-200'
                }`}
                onClick={() => setTimeRange(range)}
              >
                {range}
              </button>
            ))}
          </div>

          <Button
            onClick={fetchAnalytics}
            disabled={loading || !selectedDatabase}
            variant="secondary"
          >
            {loading ? 'Loading…' : 'Refresh'}
          </Button>
        </div>
      </div>

      {!selectedDatabase ? (
        <EmptyState
          icon="📊"
          title="No Database Selected"
          description="Please select a database to view analytics"
        />
      ) : (
        <>
          {/* ── Tabs ── */}
          <div className="flex gap-1 border-b-2 border-gray-200 mb-6">
            {[
              { id: 'overview', label: 'Overview' },
              { id: 'queries', label: 'Query Explorer' },
              { id: 'patterns', label: 'Patterns' },
              { id: 'insights', label: 'Insights' },
            ].map(({ id, label }) => (
              <button
                key={id}
                className={`px-5 py-3 text-sm font-medium border-b-2 -mb-0.5 transition-colors ${
                  activeTab === id
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-indigo-500'
                }`}
                onClick={() => setActiveTab(id)}
              >
                {label}
              </button>
            ))}
          </div>

          {/* ── Overview Tab ── */}
          {activeTab === 'overview' && (
            <>
              {/* Key Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                {[
                  { value: summaryMetrics.totalQueries.toLocaleString(), label: 'Total Queries' },
                  { value: summaryMetrics.successRate, label: 'Success Rate' },
                  { value: summaryMetrics.avgLatency, label: 'Avg Latency' },
                  { value: summaryMetrics.qps, label: 'Queries Per Second' },
                ].map(({ value, label }, i) => (
                  <div
                    key={label}
                    className={`bg-gradient-to-br ${METRIC_GRADIENTS[i]} rounded-xl p-5 text-white shadow-md`}
                  >
                    <p className="text-3xl font-bold mb-1">{value}</p>
                    <p className="text-sm opacity-90">{label}</p>
                  </div>
                ))}
              </div>

              {/* Time Series Chart */}
              <Card className="mb-6">
                <CardHeader>
                  <CardTitle>Queries Over Time</CardTitle>
                </CardHeader>
                <CardContent>
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
                        <Tooltip labelFormatter={(value) => new Date(value).toLocaleString()} />
                        <Legend />
                        <Line type="monotone" dataKey="total_queries" stroke="#6366f1" name="Total Queries" />
                        <Line type="monotone" dataKey="successful_queries" stroke="#10b981" name="Successful" />
                        <Line type="monotone" dataKey="failed_queries" stroke="#ef4444" name="Failed" />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <EmptyState title="No query data available for this time range" />
                  )}
                </CardContent>
              </Card>

              {/* Latency Distribution */}
              {statistics.length > 0 && (
                <Card className="mb-6">
                  <CardHeader>
                    <CardTitle>Latency Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
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
                        <Bar dataKey="avg_latency_ms" fill="#6366f1" name="Avg" />
                        <Bar dataKey="p95_latency_ms" fill="#f59e0b" name="P95" />
                        <Bar dataKey="p99_latency_ms" fill="#ef4444" name="P99" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {/* Top Patterns + Slow Queries */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <Card>
                  <CardHeader><CardTitle>Top Query Patterns</CardTitle></CardHeader>
                  <CardContent className="p-0">
                    {queryPatterns.length > 0 ? (
                      <div className="overflow-x-auto">
                        <table className="w-full border-collapse">
                          <thead>
                            <tr>
                              <th className={thCls}>Pattern</th>
                              <th className={thCls}>Count</th>
                              <th className={thCls}>Avg Latency</th>
                            </tr>
                          </thead>
                          <tbody>
                            {queryPatterns.slice(0, 5).map((pattern, idx) => (
                              <tr key={idx} className="hover:bg-gray-50">
                                <td className={tdCls}>
                                  <div className="font-mono text-xs bg-gray-50 px-2 py-1 rounded max-w-[300px] truncate">
                                    {pattern.normalized_text || pattern.pattern}
                                  </div>
                                </td>
                                <td className={tdCls}>{pattern.count}</td>
                                <td className={tdCls}>{formatDuration(pattern.avg_latency_ms)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="px-6 py-4"><EmptyState title="No query patterns found" /></div>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader><CardTitle>Slow Queries</CardTitle></CardHeader>
                  <CardContent className="p-0">
                    {insights?.slow_queries && insights.slow_queries.length > 0 ? (
                      <div className="overflow-x-auto">
                        <table className="w-full border-collapse">
                          <thead>
                            <tr>
                              <th className={thCls}>Query</th>
                              <th className={thCls}>Latency</th>
                              <th className={thCls}>Time</th>
                            </tr>
                          </thead>
                          <tbody>
                            {insights.slow_queries.slice(0, 5).map((query, idx) => (
                              <tr key={idx} className="hover:bg-gray-50">
                                <td className={tdCls}>
                                  <div className="font-mono text-xs bg-gray-50 px-2 py-1 rounded max-w-[300px] truncate">
                                    {query.query_text || 'N/A'}
                                  </div>
                                </td>
                                <td className={tdCls}>
                                  <StatusBadge status="warning" label={formatDuration(query.total_time_ms)} />
                                </td>
                                <td className={tdCls}>{new Date(query.timestamp).toLocaleTimeString()}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="px-6 py-4"><EmptyState title="No slow queries detected" /></div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </>
          )}

          {/* ── Query Explorer Tab ── */}
          {activeTab === 'queries' && (
            <Card>
              <CardHeader><CardTitle>Recent Queries</CardTitle></CardHeader>
              <CardContent className="p-0">
                {recentQueries.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr>
                          <th className={thCls}>Timestamp</th>
                          <th className={thCls}>Query Text</th>
                          <th className={thCls}>Type</th>
                          <th className={thCls}>Results</th>
                          <th className={thCls}>Latency</th>
                          <th className={thCls}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {recentQueries.map((query, idx) => (
                          <tr key={idx} className="hover:bg-gray-50">
                            <td className={tdCls}>{formatTimestamp(query.timestamp)}</td>
                            <td className={tdCls}>
                              <div className="font-mono text-xs bg-gray-50 px-2 py-1 rounded max-w-[300px] truncate">
                                {query.query_text || 'N/A'}
                              </div>
                            </td>
                            <td className={tdCls}>
                              <StatusBadge status="info" label={query.query_type || 'vector'} />
                            </td>
                            <td className={tdCls}>{query.num_results || 0}</td>
                            <td className={tdCls}>{formatDuration(query.total_time_ms)}</td>
                            <td className={tdCls}>
                              <StatusBadge
                                status={query.has_error ? 'error' : 'success'}
                                label={query.has_error ? 'Failed' : 'Success'}
                              />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="px-6 py-4"><EmptyState title="No recent queries found" /></div>
                )}
              </CardContent>
            </Card>
          )}

          {/* ── Patterns Tab ── */}
          {activeTab === 'patterns' && (
            <>
              <Card className="mb-5">
                <CardHeader><CardTitle>Common Query Patterns</CardTitle></CardHeader>
                <CardContent className="p-0">
                  {queryPatterns.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse">
                        <thead>
                          <tr>
                            <th className={thCls}>Pattern</th>
                            <th className={thCls}>Count</th>
                            <th className={thCls}>Avg Latency</th>
                            <th className={thCls}>Avg Results</th>
                            <th className={thCls}>First Seen</th>
                            <th className={thCls}>Last Seen</th>
                          </tr>
                        </thead>
                        <tbody>
                          {queryPatterns.map((pattern, idx) => (
                            <tr key={idx} className="hover:bg-gray-50">
                              <td className={tdCls}>
                                <div className="font-mono text-xs bg-gray-50 px-2 py-1 rounded max-w-[300px] truncate">
                                  {pattern.normalized_text || pattern.pattern}
                                </div>
                              </td>
                              <td className={tdCls}>{pattern.count}</td>
                              <td className={tdCls}>{formatDuration(pattern.avg_latency_ms)}</td>
                              <td className={tdCls}>{pattern.avg_results?.toFixed(1) || '0'}</td>
                              <td className={tdCls}>{formatTimestamp(pattern.first_seen)}</td>
                              <td className={tdCls}>{formatTimestamp(pattern.last_seen)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="px-6 py-4"><EmptyState title="No query patterns found" /></div>
                  )}
                </CardContent>
              </Card>

              {trendingQueries.length > 0 && (
                <Card>
                  <CardHeader><CardTitle>Trending Queries</CardTitle></CardHeader>
                  <CardContent className="p-0">
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse">
                        <thead>
                          <tr>
                            <th className={thCls}>Query Pattern</th>
                            <th className={thCls}>Current Count</th>
                            <th className={thCls}>Previous Count</th>
                            <th className={thCls}>Growth Rate</th>
                          </tr>
                        </thead>
                        <tbody>
                          {trendingQueries.map((query, idx) => (
                            <tr key={idx} className="hover:bg-gray-50">
                              <td className={tdCls}>
                                <div className="font-mono text-xs bg-gray-50 px-2 py-1 rounded max-w-[300px] truncate">
                                  {query.normalized_text || query.query_text}
                                </div>
                              </td>
                              <td className={tdCls}>{query.current_count}</td>
                              <td className={tdCls}>{query.previous_count}</td>
                              <td className={tdCls}>
                                <StatusBadge status="success" label={`+${query.growth_rate?.toFixed(1) || '0'}%`} />
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}

          {/* ── Insights Tab ── */}
          {activeTab === 'insights' && (
            <>
              {insights && (
                <Card className="mb-5">
                  <CardHeader><CardTitle>Automated Insights & Recommendations</CardTitle></CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {[
                        {
                          type: 'success',
                          title: 'Query Performance',
                          desc: `${insights.summary?.total_queries || 0} queries processed with ${((insights.summary?.success_rate || 0) * 100).toFixed(1)}% success rate. Average latency: ${formatDuration(insights.summary?.avg_latency_ms || 0)}`
                        },
                        ...(insights.summary?.peak_hour ? [{
                          type: 'info',
                          title: 'Peak Usage',
                          desc: `Highest traffic at ${new Date(insights.summary.peak_hour).toLocaleTimeString()} with ${insights.summary.peak_hour_queries} queries`
                        }] : []),
                        ...(insights.slow_queries?.length > 0 ? [{
                          type: 'warning',
                          title: 'Performance Alert',
                          desc: `${insights.slow_queries.length} slow queries detected (> ${formatDuration(1000)}). Consider optimizing indexes or query patterns.`
                        }] : []),
                        ...(insights.zero_result_queries?.length > 0 ? [{
                          type: 'error',
                          title: 'Content Gaps',
                          desc: `${insights.zero_result_queries.length} query patterns returned zero results. Consider adding relevant content or improving search quality.`
                        }] : []),
                        ...(insights.trending_queries?.length > 0 ? [{
                          type: 'success',
                          title: 'Trending Topics',
                          desc: `${insights.trending_queries.length} queries showing significant growth. Monitor these patterns for content opportunities.`
                        }] : []),
                        ...(!insights.slow_queries?.length && !insights.zero_result_queries?.length && !insights.trending_queries?.length ? [{
                          type: 'info',
                          title: 'All Good!',
                          desc: 'No significant issues detected. Your search system is performing well.'
                        }] : []),
                      ].map(({ type, title, desc }) => {
                        const borderColor = { success: 'border-emerald-400', warning: 'border-amber-400', error: 'border-red-400', info: 'border-indigo-400' }[type];
                        return (
                          <div key={title} className={`pl-4 py-3 pr-3 bg-gray-50 rounded-lg border-l-4 ${borderColor}`}>
                            <p className="text-sm font-semibold text-gray-800 mb-0.5">{title}</p>
                            <p className="text-sm text-gray-600">{desc}</p>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              )}

              {insights?.zero_result_queries && insights.zero_result_queries.length > 0 && (
                <Card>
                  <CardHeader><CardTitle>Zero-Result Queries</CardTitle></CardHeader>
                  <CardContent className="p-0">
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse">
                        <thead>
                          <tr>
                            <th className={thCls}>Query Pattern</th>
                            <th className={thCls}>Occurrence Count</th>
                          </tr>
                        </thead>
                        <tbody>
                          {insights.zero_result_queries.map((query, idx) => (
                            <tr key={idx} className="hover:bg-gray-50">
                              <td className={tdCls}>
                                <div className="font-mono text-xs bg-gray-50 px-2 py-1 rounded max-w-[500px] truncate">
                                  {query.normalized_text || query.query_text}
                                </div>
                              </td>
                              <td className={tdCls}>{query.count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </>
      )}
    </Layout>
  );
}
