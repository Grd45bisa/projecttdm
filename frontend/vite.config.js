import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Konfigurasi proxy untuk API calls
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
        // Opsional: rewrite path
        // rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
});