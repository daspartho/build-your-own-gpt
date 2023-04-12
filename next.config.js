/** @type {import('next').NextConfig} */
const nextConfig = {
  //  reactStrictMode: true, -- because of the 'bug' in react-beautiful-dnd
  swcMinify: true,
};

module.exports = nextConfig;

// module.exports = {
//   async rewrites() {
//     return [
//       {
//         source: "/api/:path*",
//         destination: "http://localhost:3001/",
//       },
//     ];
//   },
// };
