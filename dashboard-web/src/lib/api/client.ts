/**
 * Typed API client. Every response is Zod-parsed; failures throw an
 * `ApiError` carrying the endpoint and status so error boundaries can show
 * something actionable instead of a blank panel.
 */
import type { z } from 'zod';
import {
	Autopsy,
	BacktestSummary,
	Bar,
	BrainStatus,
	CausalSnapshot,
	EquityPoint,
	ImaginationFan,
	Signal,
	TickerInfo,
	Trade,
	UncertaintyPoint
} from './schemas';

export class ApiError extends Error {
	constructor(
		public endpoint: string,
		public status: number,
		message: string
	) {
		super(`${endpoint} → ${status}: ${message}`);
	}
}

async function get<T>(endpoint: string, schema: z.ZodType<T>, fetcher = fetch): Promise<T> {
	const res = await fetcher(endpoint, { headers: { accept: 'application/json' } });
	if (!res.ok) throw new ApiError(endpoint, res.status, await res.text().catch(() => ''));
	return schema.parse(await res.json());
}

async function post<T>(
	endpoint: string,
	body: unknown,
	schema: z.ZodType<T>,
	fetcher = fetch
): Promise<T> {
	const res = await fetcher(endpoint, {
		method: 'POST',
		headers: { 'content-type': 'application/json', accept: 'application/json' },
		body: JSON.stringify(body)
	});
	if (!res.ok) throw new ApiError(endpoint, res.status, await res.text().catch(() => ''));
	return schema.parse(await res.json());
}

const qs = (params: Record<string, string | number | undefined>) => {
	const p = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') p.set(k, String(v));
	const s = p.toString();
	return s ? `?${s}` : '';
};

export const api = {
	tickers: (fetcher = fetch) => get('/api/tickers', TickerInfo.array(), fetcher),

	bars: (ticker: string, date: string, tf: '1min' | '5min', fetcher = fetch) =>
		get(`/api/bars${qs({ ticker, date, tf })}`, Bar.array(), fetcher),

	dates: (ticker: string, fetcher = fetch) =>
		get(`/api/dates${qs({ ticker })}`, DateList, fetcher),

	uncertainty: (ticker: string, date: string, fetcher = fetch) =>
		get(`/api/uncertainty${qs({ ticker, date })}`, UncertaintyPoint.array(), fetcher),

	backtests: (fetcher = fetch) => get('/api/backtests', BacktestSummary.array(), fetcher),

	signals: (opts: { backtest?: string; ticker?: string; from?: string; to?: string; limit?: number }, fetcher = fetch) =>
		get(`/api/signals${qs(opts)}`, Signal.array(), fetcher),

	trades: (opts: { backtest?: string; ticker?: string; from?: string; to?: string }, fetcher = fetch) =>
		get(`/api/trades${qs(opts)}`, Trade.array(), fetcher),

	equity: (backtest: string, fetcher = fetch) =>
		get(`/api/equity${qs({ backtest })}`, EquityPoint.array(), fetcher),

	autopsies: (opts: { ticker?: string; verdict?: string; limit?: number } = {}, fetcher = fetch) =>
		get(`/api/autopsies${qs(opts)}`, Autopsy.array(), fetcher),

	causal: (fetcher = fetch) => get('/api/causal/latest', CausalSnapshot.nullable(), fetcher),

	status: (fetcher = fetch) => get('/api/status', BrainStatus, fetcher),

	imagine: (body: { ticker: string; anchor_ts: number; n?: number; horizon?: number }, fetcher = fetch) =>
		post('/api/playground/imagine', body, ImaginationFan, fetcher),

	replay: (body: { action: 'start' | 'stop'; ticker?: string; date?: string; speed?: number }, fetcher = fetch) =>
		post('/api/playground/replay', body, Ack, fetcher),

	feedback: (
		body: { context: string; ticker?: string; signal_id?: string; rating: number; comment: string },
		fetcher = fetch
	) => post('/api/feedback', body, Ack, fetcher)
};

import { z as zz } from 'zod';
const DateList = zz.array(zz.string());
const Ack = zz.object({ ok: zz.boolean() });
