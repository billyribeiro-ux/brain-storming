/**
 * Production API proxy: forwards /api/* to the Aether Python bridge.
 * In dev, Vite's server.proxy handles this; this route makes the built
 * Node server self-contained (single origin, no CORS anywhere).
 */
import { env } from '$env/dynamic/private';
import type { RequestHandler } from './$types';

const BRIDGE = env.AETHER_BRIDGE_URL ?? 'http://localhost:8600';

const forward: RequestHandler = async ({ params, request, url }) => {
	const target = `${BRIDGE}/api/${params.path}${url.search}`;
	const init: RequestInit = {
		method: request.method,
		headers: { 'content-type': request.headers.get('content-type') ?? 'application/json' },
		body: request.method === 'GET' || request.method === 'HEAD' ? undefined : await request.arrayBuffer()
	};
	try {
		const res = await fetch(target, init);
		return new Response(res.body, {
			status: res.status,
			headers: { 'content-type': res.headers.get('content-type') ?? 'application/json' }
		});
	} catch {
		return new Response(
			JSON.stringify({ error: 'bridge_unreachable', bridge: BRIDGE }),
			{ status: 502, headers: { 'content-type': 'application/json' } }
		);
	}
};

export const GET = forward;
export const POST = forward;
