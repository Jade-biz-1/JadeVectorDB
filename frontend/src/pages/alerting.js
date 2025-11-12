import Head from 'next/head';
import { useState, useEffect } from 'react';
import { alertApi } from '../lib/api';

export default function Alerting() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchAlerts = async () => {
      setLoading(true);
      try {
        const response = await alertApi.listAlerts();
        setAlerts(response.alerts || []);
      } catch (error) {
        console.error('Error fetching alerts:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchAlerts();
  }, []);

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
            <h2 className="text-xl font-semibold text-gray-800 mb-4">System Alerts</h2>
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Message</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {loading ? (
                  <tr><td colSpan={3} className="text-center py-4">Loading...</td></tr>
                ) : alerts.length === 0 ? (
                  <tr><td colSpan={3} className="text-center py-4">No alerts found.</td></tr>
                ) : (
                  alerts.map(alert => (
                    <tr key={alert.id}>
                      <td className="px-6 py-4 whitespace-nowrap">{alert.type}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{alert.message}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{alert.time}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
