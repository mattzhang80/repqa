/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*",
      },
      {
        source: "/data/processed/:path*",
        destination: "http://localhost:8000/data/processed/:path*",
      },
      {
        source: "/data/reports/:path*",
        destination: "http://localhost:8000/data/reports/:path*",
      },
    ];
  },
};

export default nextConfig;
