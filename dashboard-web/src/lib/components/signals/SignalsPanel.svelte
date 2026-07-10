<script lang="ts">
	/**
	 * Live signal tape. The parent owns the merge (history + live feed) and
	 * row order; this panel renders rows, flashes newcomers, and expands an
	 * inline anatomy view on focus.
	 */
	import { untrack } from 'svelte';
	import type { Signal } from '$lib/api/schemas';
	import Panel from '$lib/components/layout/Panel.svelte';
	import { fmtPx, fmtTimeShort } from '$lib/utils/format';
	import ConvictionBadge from './ConvictionBadge.svelte';
	import SignalDetail from './SignalDetail.svelte';

	let { signals, onFocus }: { signals: Signal[]; onFocus?: (s: Signal | null) => void } = $props();

	let expandedId = $state<string | null>(null);

	// Rows present at mount render quietly; rows arriving later (live feed)
	// flash in via .rise-in (which already respects prefers-reduced-motion).
	// Deliberately a one-time snapshot — hence untrack.
	const initialIds = new Set(untrack(() => signals).map((s) => s.signal_id));

	function toggle(s: Signal): void {
		if (expandedId === s.signal_id) {
			expandedId = null;
			onFocus?.(null);
		} else {
			expandedId = s.signal_id;
			onFocus?.(s);
		}
	}

	const expMove = (s: Signal): string =>
		`${((Math.abs(s.target_px - s.entry_px) / s.entry_px) * 100).toFixed(2)}%`;

	const GRID =
		'grid grid-cols-[2.5rem_3.25rem_3rem_2.5rem_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)_3.25rem] items-center gap-2';
</script>

<Panel
	title="Live Signals"
	subtitle="consensus-gated reversal calls"
	empty={signals.length === 0 ? 'No signals in this window — the consensus gate is holding.' : null}
>
	<div class={`${GRID} px-2 pb-1.5 text-[9px] tracking-wider text-ink-faint uppercase`} aria-hidden="true">
		<span>time</span>
		<span>ticker</span>
		<span>side</span>
		<span>conv</span>
		<span class="text-right">entry</span>
		<span class="text-right">stop</span>
		<span class="text-right">target</span>
		<span class="text-right">move</span>
	</div>

	<ul class="flex flex-col">
		{#each signals as s (s.signal_id)}
			{@const expanded = expandedId === s.signal_id}
			<li class={initialIds.has(s.signal_id) ? '' : 'rise-in'}>
				<button
					type="button"
					onclick={() => toggle(s)}
					aria-expanded={expanded}
					aria-controls={`signal-detail-${s.signal_id}`}
					aria-label={`${s.side} ${s.ticker} at ${fmtTimeShort(s.ts)}, ${expanded ? 'collapse' : 'expand'} detail`}
					class={`${GRID} w-full rounded px-2 py-1.5 text-left text-xs transition-colors focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-inset focus-visible:outline-none ${
						expanded ? 'bg-accent-soft' : 'hover:bg-raised'
					}`}
				>
					<span class="num text-[11px] text-ink-faint">{fmtTimeShort(s.ts)}</span>
					<span class="font-mono text-[11px] font-bold text-ink">{s.ticker}</span>
					<span
						class={`inline-flex w-fit items-center rounded px-1.5 py-px text-[10px] font-semibold uppercase ${
							s.side === 'long' ? 'bg-long-soft text-long' : 'bg-short-soft text-short'
						}`}>{s.side}</span
					>
					<ConvictionBadge conviction={s.conviction} />
					<span class="num text-right text-[11px] text-ink">{fmtPx(s.entry_px)}</span>
					<span class="num text-right text-[11px] text-ink-dim">{fmtPx(s.stop_px)}</span>
					<span class="num text-right text-[11px] text-ink-dim">{fmtPx(s.target_px)}</span>
					<span class="num text-right text-[11px] text-accent">{expMove(s)}</span>
				</button>
				{#if expanded}
					<div
						id={`signal-detail-${s.signal_id}`}
						class="mx-2 mb-2 rounded border border-hairline bg-surface p-3"
					>
						<SignalDetail signal={s} />
					</div>
				{/if}
			</li>
		{/each}
	</ul>
</Panel>
