import adapter from '@sveltejs/adapter-node';
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			// Node adapter: the dashboard ships as a standalone Node service
			// sitting in front of the Python bridge (see aether/dashboard/api.py).
			adapter: adapter()
		})
	],
	server: {
		// Dev-time proxy to the Aether Python bridge. In production the
		// SvelteKit server route /api/[...path] performs the same proxying.
		proxy: {
			'/api': { target: 'http://localhost:8600', changeOrigin: true },
			'/ws': { target: 'ws://localhost:8600', ws: true }
		}
	}
});
