import Head from 'next/head';
import { useState, useEffect } from 'react';
import { alertApi } from '../lib/api';

export default function Alerting() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('all'); // 'all', 'warning', 'error', 'info'

  useEffect(() => {
    fetchAlerts();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchAlerts = async () => {
    setLoading(true);
    try {
      const response = await alertApi.listAlerts();
      setAlerts(response.alerts || []);
    } catch (error) {
      console.error('Error fetching alerts:', error);
      setAlerts([]);
    } finally {
      setLoading(false);
    }
  };

  const handleAcknowledge = async (alertId) => {
    try {
      await alertApi.acknowledgeAlert(alertId);
      await fetchAlerts(); // Refresh list
      alert('Alert acknowledged successfully');
    } catch (error) {
      console.error('Error acknowledging alert:', error);
      alert(`Error acknowledging alert: ${error.message}`);
    }
  };

  const filteredAlerts = filter === 'all'
    ? alerts
    : alerts.filter(alert => alert.type?.toLowerCase() === filter.toLowerCase());

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Alerting - JadeVectorDB</title>
        <meta name="description" content="System alerts and notifications" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Alerting</h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-800">System Alerts ({filteredAlerts.length})</h2>
              <div className="flex space-x-2">
                {/* Filter buttons */}
                <select
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  className="border border-gray-300 rounded-md px-3 py-1 text-sm"
                >
                  <option value="all">All Types</option>
                  <option value="error">Error</option>
                  <option value="warning">Warning</option>
                  <option value="info">Info</option>
                </select>
                <button
                  onClick={fetchAlerts}
                  disabled={loading}
                  className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-1 rounded-md disabled:opacity-50"
                >
                  {loading ? 'Refreshing...' : 'Refresh'}
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Message</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {loading ? (
                    <tr><td colSpan={4} className="text-center py-4 text-gray-500">Loading alerts...</td></tr>
                  ) : filteredAlerts.length === 0 ? (
                    <tr><td colSpan={4} className="text-center py-4 text-gray-500">No alerts found.</td></tr>
                  ) : (
                    filteredAlerts.map(alert => (
                      <tr key={alert.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            alert.type?.toLowerCase() === 'error'
                              ? 'bg-red-100 text-red-800'
                              : alert.type?.toLowerCase() === 'warning'
                                ? 'bg-yellow-100 text-yellow-800'
                                : 'bg-blue-100 text-blue-800'
                          }`}>
                            {alert.type || 'Info'}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900">{alert.message || 'No message'}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {alert.time || alert.timestamp || 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          {!alert.acknowledged && (
                            <button
                              onClick={() => handleAcknowledge(alert.id)}
                              className="text-indigo-600 hover:text-indigo-900 font-medium"
                            >
                              Acknowledge
                            </button>
                          )}
                          {alert.acknowledged && (
                            <span className="text-green-600 font-medium">✓ Acknowledged</span>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
