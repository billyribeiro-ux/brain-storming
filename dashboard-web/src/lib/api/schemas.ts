/**
 * Zod schemas for every payload crossing the Python bridge.
 *
 * These mirror the dataclass contracts in `aether/*/interfaces.py` — the
 * Python side is the source of truth; anything the bridge emits is parsed
 * (never blindly trusted) so a contract drift fails loudly in dev instead
 * of rendering NaNs in production.
 */
import { z } from 'zod';

export const TickerInfo = z.object({
	symbol: z.string(),
	alias: z.string(),
	kind: z.enum(['equity', 'etf', 'index']),
	last_px: z.number().nullable(),
	day_ret: z.number().nullable(), // fraction, e.g. 0.0132
	regime: z.string(), // e.g. "trending-up" | "ranging" | "volatile" | "unknown"
	signal_prob: z.number().min(0).max(1).nullable(),
	last_bar_ts: z.number().int().nullable()
});
export type TickerInfo = z.infer<typeof TickerInfo>;

export const Bar = z.object({
	t: z.number().int(), // epoch seconds, naive-ET convention of the lake
	o: z.number(),
	h: z.number(),
	l: z.number(),
	c: z.number(),
	v: z.number()
});
export type Bar = z.infer<typeof Bar>;

export const UncertaintyPoint = z.object({
	t: z.number().int(),
	aleatoric: z.number(),
	epistemic: z.number(),
	anomaly: z.number()
});
export type UncertaintyPoint = z.infer<typeof UncertaintyPoint>;

export const DriverContribution = z.object({
	name: z.string(),
	attribution: z.number() // signed strength; UI ranks by |value|
});
export type DriverContribution = z.infer<typeof DriverContribution>;

export const Signal = z.object({
	signal_id: z.string(),
	ts: z.number().int(),
	ticker: z.string(),
	side: z.enum(['long', 'short']),
	conviction: z.number().min(0).max(1),
	entry_px: z.number(),
	stop_px: z.number(),
	target_px: z.number(),
	horizon_bars: z.number().int(),
	size_frac: z.number(),
	rationale: z.string().default(''),
	drivers: z.array(DriverContribution).default([]),
	evidence: z
		.object({
			policy_prob: z.number().nullable(),
			imagination_agreement: z.number().nullable(),
			analog_winrate: z.number().nullable(),
			aleatoric: z.number().nullable(),
			epistemic: z.number().nullable(),
			anomaly: z.number().nullable()
		})
		.partial()
		.nullable()
		.default(null)
});
export type Signal = z.infer<typeof Signal>;

export const Trade = z.object({
	trade_id: z.string(),
	ticker: z.string(),
	side: z.enum(['long', 'short']),
	entry_ts: z.number().int(),
	exit_ts: z.number().int(),
	entry_px: z.number(),
	exit_px: z.number(),
	qty: z.number(),
	pnl: z.number(),
	fees: z.number(),
	stop_px: z.number(),
	target_px: z.number(),
	exit_reason: z.string(),
	conviction: z.number().default(0),
	has_autopsy: z.boolean().default(false)
});
export type Trade = z.infer<typeof Trade>;

export const Counterfactual = z.object({
	description: z.string(),
	pnl: z.number(),
	delta: z.number()
});

export const Autopsy = z.object({
	trade: Trade.partial().extend({ trade_id: z.string() }),
	verdict: z.enum(['good_loss', 'bad_loss', 'good_win', 'lucky_win']),
	narrative: z.string(),
	drivers: z.array(z.record(z.string(), z.unknown())).default([]),
	causal_context: z.array(z.record(z.string(), z.unknown())).default([]),
	counterfactuals: z.array(Counterfactual).default([]),
	lessons: z.array(z.record(z.string(), z.unknown())).default([])
});
export type Autopsy = z.infer<typeof Autopsy>;

export const CausalEdge = z.object({
	src: z.string(),
	dst: z.string(),
	lag: z.number().int(),
	weight: z.number(),
	confidence: z.number().min(0).max(1)
});
export type CausalEdge = z.infer<typeof CausalEdge>;

export const CausalSnapshot = z.object({
	fitted_start: z.string(),
	fitted_end: z.string(),
	nodes: z.array(z.string()),
	edges: z.array(CausalEdge)
});
export type CausalSnapshot = z.infer<typeof CausalSnapshot>;

export const BacktestSummary = z.object({
	name: z.string(),
	stats: z.record(z.string(), z.unknown()),
	n_trades: z.number().int(),
	n_signals: z.number().int(),
	start: z.string().nullable(),
	end: z.string().nullable()
});
export type BacktestSummary = z.infer<typeof BacktestSummary>;

export const EquityPoint = z.object({ t: z.number().int(), equity: z.number() });
export type EquityPoint = z.infer<typeof EquityPoint>;

export const BrainStatus = z.object({
	generated_at: z.string(),
	lake: z.object({
		datasets: z.number().int(),
		rows: z.number().int(),
		newest_bar: z.string().nullable()
	}),
	checkpoints: z.array(
		z.object({ layer: z.string(), path: z.string(), mtime: z.string(), steps: z.number().nullable() })
	),
	training: z.record(
		z.string(),
		z.object({
			last_step: z.number().nullable(),
			last_total: z.number().nullable(),
			best_val: z.number().nullable(),
			series: z.array(z.object({ step: z.number(), total: z.number() })).default([])
		})
	),
	diagnosis: z.object({
		healthy: z.boolean(),
		findings: z.array(
			z.object({ kind: z.string(), severity: z.string(), message: z.string() })
		)
	}),
	regime: z.record(z.string(), z.string()).default({})
});
export type BrainStatus = z.infer<typeof BrainStatus>;

export const ImaginationFan = z.object({
	ticker: z.string(),
	anchor_ts: z.number().int(),
	horizon: z.number().int(),
	/** Per-rollout latent-divergence paths [n_rollouts][horizon] — a proxy,
	 *  not prices; the UI must label it as such. */
	paths: z.array(z.array(z.number())),
	quantiles: z.object({
		p10: z.array(z.number()),
		p50: z.array(z.number()),
		p90: z.array(z.number())
	})
});
export type ImaginationFan = z.infer<typeof ImaginationFan>;

/** Envelope for every WebSocket push. */
export const WsEvent = z.discriminatedUnion('type', [
	z.object({ type: z.literal('hello'), server_time: z.string() }),
	z.object({ type: z.literal('heartbeat'), server_time: z.string() }),
	z.object({ type: z.literal('signal'), payload: Signal }),
	z.object({ type: z.literal('trade'), payload: Trade }),
	z.object({ type: z.literal('bar'), ticker: z.string(), payload: Bar }),
	z.object({
		type: z.literal('replay_state'),
		payload: z.object({
			running: z.boolean(),
			ticker: z.string().nullable(),
			date: z.string().nullable(),
			cursor: z.number().int().nullable(),
			speed: z.number()
		})
	})
]);
export type WsEvent = z.infer<typeof WsEvent>;
