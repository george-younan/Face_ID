import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'

const backend = 'http://localhost:5000'
const routes = [
  '/process_frame', '/faces_count', '/toggle_detection', '/register_face',
  '/known_faces_list', '/static_unknown_faces', '/update_unknown_snapshot',
  '/detected_faces', '/refresh_faces', '/camera_status', '/test_camera',
  '/delete_face', '/stats',
]

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), tailwindcss()],
  server: {
    proxy: Object.fromEntries(routes.map(r => [r, { target: backend, changeOrigin: true }])),
  },
})