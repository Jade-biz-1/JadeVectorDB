/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/health',
        destination: 'http://localhost:8080/health',
      },
      {
        source: '/api/status',
        destination: 'http://localhost:8080/status',
      },
      {
        source: '/api/:path*',
        destination: 'http://localhost:8080/v1/:path*',
      },
    ]
  },
}

module.exports = nextConfig
