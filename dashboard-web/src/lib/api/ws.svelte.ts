/**
 * Real-time channel to the Aether bridge — a Svelte 5 runes store.
 *
 * Auto-reconnects with exponential backoff, Zod-parses every frame, and
 * fans events out to bounded in-memory feeds the panels consume. Unknown
 * or malformed frames are counted, never thrown — a telemetry stream must
 * not take down the cockpit.
 */
import { browser } from '$app/environment';
import { WsEvent, type Bar, type Signal, type Trade } from './schemas';

const MAX_FEED = 500;

export type ConnState = 'connecting' | 'live' | 'down';

class LiveFeed {
	conn = $state<ConnState>('down');
	lastHeartbeat = $state<string | null>(null);
	signals = $state<Signal[]>([]);
	trades = $state<Trade[]>([]);
	/** Latest replayed/live bar per ticker (the chart appends from here). */
	lastBar = $state<Record<string, Bar>>({});
	replay = $state<{ running: boolean; ticker: string | null; date: string | null; speed: number }>({
		running: false,
		ticker: null,
		date: null,
		speed: 1
	});
	badFrames = $state(0);

	#ws: WebSocket | null = null;
	#attempt = 0;
	#closedByUs = false;

	connect(): void {
		if (!browser || this.#ws) return;
		this.#closedByUs = false;
		this.conn = 'connecting';
		const proto = location.protocol === 'https:' ? 'wss' : 'ws';
		const ws = new WebSocket(`${proto}://${location.host}/ws`);
		this.#ws = ws;

		ws.onopen = () => {
			this.conn = 'live';
			this.#attempt = 0;
		};
		ws.onmessage = (msg) => {
			let frame: unknown;
			try {
				frame = JSON.parse(String(msg.data));
			} catch {
				this.badFrames += 1;
				return;
			}
			const parsed = WsEvent.safeParse(frame);
			if (!parsed.success) {
				this.badFrames += 1;
				return;
			}
			const ev = parsed.data;
			switch (ev.type) {
				case 'hello':
				case 'heartbeat':
					this.lastHeartbeat = ev.server_time;
					break;
				case 'signal':
					this.signals = [ev.payload, ...this.signals].slice(0, MAX_FEED);
					break;
				case 'trade':
					this.trades = [ev.payload, ...this.trades].slice(0, MAX_FEED);
					break;
				case 'bar':
					this.lastBar = { ...this.lastBar, [ev.ticker]: ev.payload };
					break;
				case 'replay_state':
					this.replay = {
						running: ev.payload.running,
						ticker: ev.payload.ticker,
						date: ev.payload.date,
						speed: ev.payload.speed
					};
					break;
			}
		};
		ws.onclose = () => {
			this.#ws = null;
			this.conn = 'down';
			if (this.#closedByUs) return;
			const delay = Math.min(15_000, 500 * 2 ** this.#attempt++);
			setTimeout(() => this.connect(), delay);
		};
		ws.onerror = () => ws.close();
	}

	disconnect(): void {
		this.#closedByUs = true;
		this.#ws?.close();
		this.#ws = null;
		this.conn = 'down';
	}

	clearFeeds(): void {
		this.signals = [];
		this.trades = [];
	}
}

/** Singleton — one socket per tab. */
export const live = new LiveFeed();
