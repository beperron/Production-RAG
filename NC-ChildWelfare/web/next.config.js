/** @type {import('next').NextConfig} */
// Served at the root of its own subdomain (nc-policy.parallel42.ai).
// If ever mounted under a path instead, set NEXT_PUBLIC_BASE_PATH=/nc-policy.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || undefined;
module.exports = { basePath, reactStrictMode: true };
