import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite config — outputs into ../app/static so FastAPI serves the bundle
// alongside its existing /static mount. Dev server proxies /api to the
// FastAPI backend on :8000.
export default defineConfig({
  plugins: [react()],
  base: '/static/',
  build: {
    outDir: '../app/static',
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
