<script lang="ts">
	/**
	 * Dense institutional trade blotter. Sortable by entry time / pnl /
	 * ticker, sticky header, capped at 400 rendered rows (cheap
	 * virtualization — the footer says how many are hidden). Rows are
	 * keyboard-operable (Tab + Enter/Space) and report selection state.
	 */
	import type { Trade } from '$lib/api/schemas';
	import { fmtDateTime, fmtPct, fmtPnl, fmtPx, pnlClass, sideClass } from '$lib/utils/format';
	import { CaretDown, CaretUp, Skull } from 'phosphor-svelte';

	let {
		trades,
		onInspect,
		selectedId = null
	}: {
		trades: Trade[];
		onInspect?: (t: Trade) => void;
		selectedId?: string | null;
	} = $props();

	type SortKey = 'entry_ts' | 'pnl' | 'ticker';
	let sortKey = $state<SortKey>('entry_ts');
	let sortDir = $state<'asc' | 'desc'>('desc');

	function toggleSort(key: SortKey): void {
		if (sortKey === key) {
			sortDir = sortDir === 'asc' ? 'desc' : 'asc';
		} else {
			sortKey = key;
			sortDir = key === 'ticker' ? 'asc' : 'desc';
		}
	}

	const MAX_ROWS = 400;

	const sorted = $derived.by(() => {
		const dir = sortDir === 'asc' ? 1 : -1;
		return [...trades].sort((a, b) =>
			sortKey === 'ticker' ? dir * a.ticker.localeCompare(b.ticker) : dir * (a[sortKey] - b[sortKey])
		);
	});
	const visible = $derived(sorted.slice(0, MAX_ROWS));

	const durationMin = (t: Trade): number => Math.round((t.exit_ts - t.entry_ts) / 60);

	const reasonChip = (reason: string): string => {
		switch (reason) {
			case 'stop':
				return 'bg-short-soft text-short';
			case 'target':
				return 'bg-long-soft text-long';
			case 'eod':
				return 'bg-raised text-ink-dim';
			case 'policy_exit':
				return 'bg-accent-soft text-accent';
			default:
				return 'bg-raised text-ink-faint';
		}
	};

	const ariaSort = (key: SortKey): 'ascending' | 'descending' | undefined =>
		sortKey === key ? (sortDir === 'asc' ? 'ascending' : 'descending') : undefined;

	const fmtQty = (q: number): string =>
		q.toLocaleString('en-US', { maximumFractionDigits: Number.isInteger(q) ? 0 : 2 });
</script>

{#snippet sortHeader(label: string, key: SortKey)}
	<button
		type="button"
		class="inline-flex items-center gap-0.5 rounded text-left tracking-wide uppercase hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-accent"
		onclick={() => toggleSort(key)}
		aria-label={`Sort by ${label}`}
	>
		{label}
		{#if sortKey === key}
			{#if sortDir === 'asc'}
				<CaretUp size={10} weight="bold" aria-hidden="true" />
			{:else}
				<CaretDown size={10} weight="bold" aria-hidden="true" />
			{/if}
		{/if}
	</button>
{/snippet}

{#if trades.length === 0}
	<div class="flex min-h-24 items-center justify-center text-xs text-ink-faint">
		No trades in this range.
	</div>
{:else}
	<table class="w-full border-collapse text-[11px]" aria-label="Trades" aria-rowcount={sorted.length}>
		<thead class="sticky top-0 z-10 bg-surface">
			<tr class="text-left text-[10px] text-ink-faint">
				<th class="px-2 py-1.5 font-medium" aria-sort={ariaSort('entry_ts')}
					>{@render sortHeader('Entry', 'entry_ts')}</th
				>
				<th class="px-2 py-1.5 font-medium" aria-sort={ariaSort('ticker')}
					>{@render sortHeader('Ticker', 'ticker')}</th
				>
				<th class="px-2 py-1.5 font-medium tracking-wide uppercase">Side</th>
				<th class="px-2 py-1.5 text-right font-medium tracking-wide uppercase">Qty</th>
				<th class="px-2 py-1.5 text-right font-medium tracking-wide uppercase">Entry</th>
				<th class="px-2 py-1.5 text-right font-medium tracking-wide uppercase">Stop</th>
				<th class="px-2 py-1.5 text-right font-medium tracking-wide uppercase">Target</th>
				<th class="px-2 py-1.5 font-medium tracking-wide uppercase">Exit</th>
				<th class="px-2 py-1.5 text-right font-medium tracking-wide uppercase">Exit px</th>
				<th class="px-2 py-1.5 font-medium tracking-wide uppercase">Reason</th>
				<th class="px-2 py-1.5 text-right font-medium tracking-wide uppercase">Dur (m)</th>
				<th class="px-2 py-1.5 text-right font-medium" aria-sort={ariaSort('pnl')}
					>{@render sortHeader('PnL', 'pnl')}</th
				>
				<th class="px-2 py-1.5 text-right font-medium tracking-wide uppercase">Conv</th>
				<th class="px-2 py-1.5 font-medium tracking-wide uppercase">
					<span class="sr-only">Autopsy</span>
					<Skull size={12} aria-hidden="true" />
				</th>
			</tr>
		</thead>
		<tbody>
			{#each visible as t (t.trade_id)}
				<tr
					class={`cursor-pointer border-t border-hairline transition-colors hover:bg-raised focus-visible:outline focus-visible:-outline-offset-1 focus-visible:outline-accent ${
						selectedId === t.trade_id ? 'bg-accent-soft' : ''
					}`}
					tabindex="0"
					role="button"
					aria-label={`Inspect trade ${t.ticker} ${t.side} ${fmtDateTime(t.entry_ts)}`}
					onclick={() => onInspect?.(t)}
					onkeydown={(e) => {
						if (e.key === 'Enter' || e.key === ' ') {
							e.preventDefault();
							onInspect?.(t);
						}
					}}
				>
					<td class="num px-2 py-1 whitespace-nowrap text-ink-dim">{fmtDateTime(t.entry_ts)}</td>
					<td class="num px-2 py-1 font-medium text-ink">{t.ticker}</td>
					<td class="px-2 py-1">
						<span
							class={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ${
								t.side === 'long' ? 'bg-long-soft' : 'bg-short-soft'
							} ${sideClass(t.side)}`}>{t.side}</span
						>
					</td>
					<td class="num px-2 py-1 text-right text-ink-dim">{fmtQty(t.qty)}</td>
					<td class="num px-2 py-1 text-right">{fmtPx(t.entry_px)}</td>
					<td class="num px-2 py-1 text-right text-ink-dim">{fmtPx(t.stop_px)}</td>
					<td class="num px-2 py-1 text-right text-ink-dim">{fmtPx(t.target_px)}</td>
					<td class="num px-2 py-1 whitespace-nowrap text-ink-dim">{fmtDateTime(t.exit_ts)}</td>
					<td class="num px-2 py-1 text-right">{fmtPx(t.exit_px)}</td>
					<td class="px-2 py-1">
						<span class={`rounded px-1.5 py-0.5 text-[10px] whitespace-nowrap ${reasonChip(t.exit_reason)}`}
							>{t.exit_reason}</span
						>
					</td>
					<td class="num px-2 py-1 text-right text-ink-dim">{durationMin(t)}</td>
					<td class={`num px-2 py-1 text-right ${pnlClass(t.pnl)}`}>{fmtPnl(t.pnl)}</td>
					<td class="num px-2 py-1 text-right text-ink-dim">{fmtPct(t.conviction, 0)}</td>
					<td class="px-2 py-1">
						{#if t.has_autopsy}
							<Skull size={12} class="text-warn" aria-label="Has autopsy" />
						{/if}
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
	<p class="num px-2 py-1.5 text-[10px] text-ink-faint">
		showing {visible.length.toLocaleString('en-US')} of {sorted.length.toLocaleString('en-US')} trades{sorted.length >
		MAX_ROWS
			? ' — narrow the filters to see the rest'
			: ''}
	</p>
{/if}
