/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export', // Export as a static application
  trailingSlash: true, // Generate /about/index.html from /about
  images: {
    unoptimized: true // Since we're using output: 'export'
  },
  webpack: (config, { isServer }) => {
    // Important: Add WASM support for potential future vector operations
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      };
    }

    // Allow importing WASM files
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };

    return config;
  }
};

module.exports = nextConfig;