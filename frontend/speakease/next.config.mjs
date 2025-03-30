/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    formats: ['image/avif', 'image/webp'],
    // This is crucial for GIFs to animate in Next.js
    unoptimized: true,
  },
}

export default nextConfig;
