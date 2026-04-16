/** @type {import('next').NextConfig} */
const JADEVECTORDB_URL = process.env.JADEVECTORDB_URL || 'http://localhost:8080';

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/health',
        destination: `${JADEVECTORDB_URL}/health`,
      },
      {
        source: '/api/status',
        destination: `${JADEVECTORDB_URL}/status`,
      },
      {
        source: '/api/:path*',
        destination: `${JADEVECTORDB_URL}/v1/:path*`,
      },
    ]
  },
}

module.exports = nextConfig
