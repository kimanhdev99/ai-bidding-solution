import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../api/www'
  },
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
