<script lang="ts">
	/**
	 * Expanded signal anatomy: rationale, evidence grid, ranked driver
	 * attributions, price levels with bps distances, and a feedback hook.
	 */
	import type { Signal } from '$lib/api/schemas';
	import FeedbackWidget from '$lib/components/feedback/FeedbackWidget.svelte';
	import { fmtPx } from '$lib/utils/format';

	let { signal }: { signal: Signal } = $props();

	const num = (v: number | null | undefined, dp = 2): string => (v == null ? '—' : v.toFixed(dp));

	const evidence = $derived([
		{ label: 'policy', value: num(signal.evidence?.policy_prob) },
		{ label: 'imagination', value: num(signal.evidence?.imagination_agreement) },
		{ label: 'analogs', value: num(signal.evidence?.analog_winrate) },
		{ label: 'aleatoric', value: num(signal.evidence?.aleatoric) },
		{ label: 'epistemic', value: num(signal.evidence?.epistemic) },
		{ label: 'anomaly', value: num(signal.evidence?.anomaly) }
	]);

	// Rank drivers by |attribution|; bar widths normalize within this signal.
	const drivers = $derived.by(() => {
		const ranked = [...signal.drivers].sort(
			(a, b) => Math.abs(b.attribution) - Math.abs(a.attribution)
		);
		const max = ranked.reduce((m, d) => Math.max(m, Math.abs(d.attribution)), 0);
		return ranked.map((d) => ({
			name: d.name,
			attribution: d.attribution,
			width: max > 0 ? (Math.abs(d.attribution) / max) * 100 : 0,
			positive: d.attribution >= 0
		}));
	});

	const bps = (px: number): string => {
		const v = ((px - signal.entry_px) / signal.entry_px) * 10_000;
		return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)} bps`;
	};
</script>

<div class="flex flex-col gap-3">
	{#if signal.rationale}
		<p class="text-xs leading-relaxed text-ink-dim">{signal.rationale}</p>
	{/if}

	<div class="grid grid-cols-3 gap-2 sm:grid-cols-6" aria-label="Evidence">
		{#each evidence as ev (ev.label)}
			<div class="rounded border border-hairline bg-raised/40 px-2 py-1.5">
				<div class="text-[9px] tracking-wider text-ink-faint uppercase">{ev.label}</div>
				<div class="num mt-0.5 text-[11px] text-ink">{ev.value}</div>
			</div>
		{/each}
	</div>

	{#if drivers.length > 0}
		<div aria-label="Ranked drivers">
			<div class="mb-1 text-[9px] tracking-wider text-ink-faint uppercase">drivers</div>
			<ul class="flex flex-col gap-1">
				{#each drivers as d, i (`${d.name}-${i}`)}
					<li class="flex items-center gap-2 text-[10px]">
						<span class="w-28 truncate text-ink-dim" title={d.name}>{d.name}</span>
						<span class="h-1.5 flex-1 overflow-hidden rounded-full bg-raised">
							<span
								class={`block h-full rounded-full ${d.positive ? 'bg-long' : 'bg-short'}`}
								style={`width: ${d.width}%`}
							></span>
						</span>
						<span class={`num w-14 text-right ${d.positive ? 'text-long' : 'text-short'}`}>
							{d.positive ? '+' : '−'}{Math.abs(d.attribution).toFixed(3)}
						</span>
					</li>
				{/each}
			</ul>
		</div>
	{/if}

	<div class="grid grid-cols-2 gap-2 sm:grid-cols-4" aria-label="Levels and horizon">
		<div class="rounded border border-hairline bg-raised/40 px-2 py-1.5">
			<div class="text-[9px] tracking-wider text-ink-faint uppercase">stop</div>
			<div class="num text-[11px] text-short">{fmtPx(signal.stop_px)}</div>
			<div class="num text-[9px] text-ink-faint">{bps(signal.stop_px)}</div>
		</div>
		<div class="rounded border border-hairline bg-raised/40 px-2 py-1.5">
			<div class="text-[9px] tracking-wider text-ink-faint uppercase">entry</div>
			<div class="num text-[11px] text-ink">{fmtPx(signal.entry_px)}</div>
			<div class="num text-[9px] text-ink-faint">anchor</div>
		</div>
		<div class="rounded border border-hairline bg-raised/40 px-2 py-1.5">
			<div class="text-[9px] tracking-wider text-ink-faint uppercase">target</div>
			<div class="num text-[11px] text-long">{fmtPx(signal.target_px)}</div>
			<div class="num text-[9px] text-ink-faint">{bps(signal.target_px)}</div>
		</div>
		<div class="rounded border border-hairline bg-raised/40 px-2 py-1.5">
			<div class="text-[9px] tracking-wider text-ink-faint uppercase">horizon</div>
			<div class="num text-[11px] text-ink">{signal.horizon_bars} bars</div>
			<div class="num text-[9px] text-ink-faint">size {(signal.size_frac * 100).toFixed(1)}%</div>
		</div>
	</div>

	<div class="border-t border-hairline pt-2">
		<FeedbackWidget context="signal" ticker={signal.ticker} signalId={signal.signal_id} />
	</div>
</div>
