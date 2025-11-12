import Head from 'next/head';
import { useState, useEffect } from 'react';
import { securityApi } from '../lib/api';

export default function SecurityMonitoring() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchLogs = async () => {
      setLoading(true);
      try {
        const response = await securityApi.listAuditLogs();
        setLogs(response.logs || []);
      } catch (error) {
        console.error('Error fetching audit logs:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchLogs();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Security Monitoring - JadeVectorDB</title>
        <meta name="description" content="Security monitoring dashboard" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Security Monitoring</h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Audit Logs</h2>
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Timestamp</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">User</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Event</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {loading ? (
                  <tr><td colSpan={4} className="text-center py-4">Loading...</td></tr>
                ) : logs.length === 0 ? (
                  <tr><td colSpan={4} className="text-center py-4">No audit logs found.</td></tr>
                ) : (
                  logs.map(log => (
                    <tr key={log.id}>
                      <td className="px-6 py-4 whitespace-nowrap">{log.timestamp}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{log.user}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{log.event}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${log.status === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>{log.status}</span>
                      </td>
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
