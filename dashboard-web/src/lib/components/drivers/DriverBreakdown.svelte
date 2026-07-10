<script lang="ts">
	/**
	 * What is moving the focused ticker right now: the active signal's driver
	 * attributions as signed diverging bars, plus a radial "causal compass" —
	 * the strongest fitted causal edges flowing into `${ticker}.ret_1m`, laid
	 * out on a circle around the ticker. Line width encodes |weight|·confidence,
	 * color encodes sign, opacity encodes bootstrap confidence.
	 */
	import { createQuery } from '@tanstack/svelte-query';
	import { api } from '$lib/api/client';
	import type { CausalEdge, Signal } from '$lib/api/schemas';
	import Panel from '$lib/components/layout/Panel.svelte';

	let { signal, ticker }: { signal: Signal | null; ticker: string } = $props();

	const causalQ = createQuery(() => ({
		queryKey: ['causal'],
		queryFn: () => api.causal()
	}));

	/* ---------------- (a) signed driver bars ---------------- */

	const drivers = $derived(
		[...(signal?.drivers ?? [])].sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution))
	);
	const maxAbs = $derived(drivers.reduce((m, d) => Math.max(m, Math.abs(d.attribution)), 0) || 1);

	/* ---------------- (b) radial causal compass ---------------- */

	const CX = 130;
	const CY = 122;
	const R = 90; // rim radius
	const INNER = 30; // stop lines short of the center chip

	const SUFFIX_SHORT: Record<string, string> = {
		ret_1m: 'ret 1m',
		ret_5m: 'ret 5m',
		vol_z: 'vol',
		range_z: 'range',
		news_rate: 'news',
		treasury_10y_chg: '10y yield'
	};

	/** 'MKT.news_rate' → 'news', 'AAPL.vol_z' → 'AAPL vol'. */
	function shortNode(name: string): string {
		const i = name.indexOf('.');
		if (i < 0) return name;
		const prefix = name.slice(0, i);
		const short = SUFFIX_SHORT[name.slice(i + 1)] ?? name.slice(i + 1).replace(/_/g, ' ');
		return prefix === 'MKT' ? short : `${prefix} ${short}`;
	}

	const strength = (e: CausalEdge): number => Math.abs(e.weight) * e.confidence;

	const inbound = $derived.by((): CausalEdge[] => {
		const snap = causalQ.data;
		if (!snap) return [];
		const dst = `${ticker}.ret_1m`;
		return snap.edges
			.filter((e) => e.dst === dst && e.src !== dst)
			.sort((a, b) => strength(b) - strength(a))
			.slice(0, 8);
	});

	type Spoke = {
		edge: CausalEdge;
		x: number; // rim point
		y: number;
		ex: number; // inner endpoint near the chip
		ey: number;
		lx: number; // label anchor
		ly: number;
		anchor: 'start' | 'middle' | 'end';
		width: number;
		label: string;
	};

	const spokes = $derived.by((): Spoke[] => {
		const max = inbound.reduce((m, e) => Math.max(m, strength(e)), 0) || 1;
		return inbound.map((edge, i) => {
			const ang = ((-90 + (i * 360) / inbound.length) * Math.PI) / 180;
			const cos = Math.cos(ang);
			const sin = Math.sin(ang);
			return {
				edge,
				x: CX + R * cos,
				y: CY + R * sin,
				ex: CX + INNER * cos,
				ey: CY + INNER * sin,
				lx: CX + (R + 10) * cos,
				ly: CY + (R + 10) * sin + sin * 4,
				anchor: cos > 0.25 ? 'start' : cos < -0.25 ? 'end' : 'middle',
				width: 1 + 4 * (strength(edge) / max),
				label: shortNode(edge.src)
			};
		});
	});
</script>

<Panel title="Driver Breakdown" subtitle={`what is moving ${ticker} now`}>
	{#if drivers.length > 0}
		<div class="mb-4 flex flex-col gap-1.5" aria-label="Signal driver attributions">
			{#each drivers as d, i (`${d.name}:${i}`)}
				<div class="flex items-center gap-2">
					<span class="w-24 shrink-0 truncate font-mono text-[11px] text-ink-dim" title={d.name}>
						{d.name}
					</span>
					<div class="relative h-3 min-w-0 flex-1 overflow-hidden rounded-sm bg-raised/40">
						<span class="absolute inset-y-0 left-1/2 w-px bg-hairline-strong" aria-hidden="true"></span>
						{#if d.attribution >= 0}
							<span
								class="absolute inset-y-0 left-1/2 rounded-r-sm bg-long/70"
								style={`width:${((Math.abs(d.attribution) / maxAbs) * 50).toFixed(2)}%`}
								aria-hidden="true"
							></span>
						{:else}
							<span
								class="absolute inset-y-0 right-1/2 rounded-l-sm bg-short/70"
								style={`width:${((Math.abs(d.attribution) / maxAbs) * 50).toFixed(2)}%`}
								aria-hidden="true"
							></span>
						{/if}
					</div>
					<span
						class={`num w-14 shrink-0 text-right text-[11px] ${d.attribution >= 0 ? 'text-long' : 'text-short'}`}
					>
						{d.attribution >= 0 ? '+' : '−'}{Math.abs(d.attribution).toFixed(2)}
					</span>
				</div>
			{/each}
		</div>
	{/if}

	{#if causalQ.isPending}
		<p class="py-6 text-center text-xs text-ink-faint" role="status">Loading causal snapshot…</p>
	{:else if causalQ.error}
		<p class="py-6 text-center text-xs text-short" role="alert">{causalQ.error.message}</p>
	{:else if !causalQ.data}
		<p class="py-6 text-center text-xs text-ink-faint">No causal snapshot yet — run the causal fit.</p>
	{:else if inbound.length === 0}
		<p class="py-6 text-center text-xs text-ink-faint">
			No fitted causal edges into {ticker}.ret_1m.
		</p>
	{:else}
		<svg
			viewBox="0 0 260 244"
			class="mx-auto block w-full max-w-72"
			role="img"
			aria-label={`Causal compass: ${inbound.length} strongest causal drivers into ${ticker} 1-minute returns`}
		>
			<circle
				cx={CX}
				cy={CY}
				r={R}
				fill="none"
				stroke="var(--color-hairline)"
				stroke-dasharray="2 4"
			/>
			{#each spokes as s (`${s.edge.src}:${s.edge.lag}`)}
				<line
					x1={s.x}
					y1={s.y}
					x2={s.ex}
					y2={s.ey}
					stroke={s.edge.weight >= 0 ? 'var(--color-long)' : 'var(--color-short)'}
					stroke-width={s.width}
					stroke-linecap="round"
					opacity={s.edge.confidence}
				/>
				<circle cx={s.x} cy={s.y} r="3" fill="var(--color-raised)" stroke="var(--color-hairline-strong)" />
				<text
					x={s.lx}
					y={s.ly}
					text-anchor={s.anchor}
					dominant-baseline="middle"
					class="font-mono"
					font-size="9"
					fill="var(--color-ink-dim)"
				>
					{s.label}{s.edge.lag > 0 ? ` ·L${s.edge.lag}` : ''}
				</text>
			{/each}
			<!-- center chip -->
			<circle cx={CX} cy={CY} r="24" fill="var(--color-raised)" stroke="var(--color-hairline-strong)" />
			<text
				x={CX}
				y={CY}
				text-anchor="middle"
				dominant-baseline="central"
				class="font-mono"
				font-size="11"
				font-weight="600"
				fill="var(--color-ink)"
			>
				{ticker}
			</text>
		</svg>
		<div class="mt-2 flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[10px] text-ink-faint">
			<span class="flex items-center gap-1">
				<span class="inline-block h-0.5 w-3 rounded-full bg-long" aria-hidden="true"></span>
				positive weight
			</span>
			<span class="flex items-center gap-1">
				<span class="inline-block h-0.5 w-3 rounded-full bg-short" aria-hidden="true"></span>
				negative weight
			</span>
			<span>edge opacity = bootstrap confidence</span>
		</div>
	{/if}
</Panel>
