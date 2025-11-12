import Head from 'next/head';

export default function IntegrationGuide() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Integration Guide - JadeVectorDB</title>
        <meta name="description" content="How to integrate JadeVectorDB" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Integration Guide</h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">How to Integrate</h2>
            <ul className="list-disc ml-6 text-gray-700">
              <li>Use REST API endpoints for database and vector operations</li>
              <li>Authenticate using API keys from the API Key Management page</li>
              <li>Refer to the <a href="/docs/api_documentation.md" className="text-blue-600 underline">API documentation</a> for endpoint details</li>
              <li>See example code in <a href="/examples/cli" className="text-blue-600 underline">/examples/cli</a> and <a href="/examples/frontend" className="text-blue-600 underline">/examples/frontend</a></li>
              <li>Contact support for advanced integration scenarios</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}
