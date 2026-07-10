<script lang="ts">
	/**
	 * Instrument rail: one row per ticker (price, day return, regime, signal
	 * probability), polled every 60s. Collapsible to a 56px strip. Footer
	 * shows a mini brain-status readout.
	 */
	import { createQuery } from '@tanstack/svelte-query';
	import { CaretLeft } from 'phosphor-svelte';
	import { api } from '$lib/api/client';
	import type { TickerInfo } from '$lib/api/schemas';
	import { app } from '$lib/stores/app.svelte';
	import { fmtPct, fmtPx } from '$lib/utils/format';

	const tickersQ = createQuery(() => ({
		queryKey: ['tickers'],
		queryFn: () => api.tickers(),
		refetchInterval: 60_000
	}));

	const statusQ = createQuery(() => ({
		queryKey: ['status'],
		queryFn: () => api.status(),
		staleTime: 120_000
	}));

	const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });

	const regimeClass = (regime: string): string =>
		regime === 'trending-up'
			? 'text-long'
			: regime === 'trending-down'
				? 'text-short'
				: regime === 'volatile'
					? 'text-warn'
					: 'text-ink-faint';

	const retPill = (r: number | null): string =>
		r == null ? 'text-ink-faint' : r >= 0 ? 'bg-long-soft text-long' : 'bg-short-soft text-short';

	const retDot = (r: number | null): string =>
		r == null ? 'bg-ink-faint' : r >= 0 ? 'bg-long' : 'bg-short';

	const fmtRet = (r: number | null): string => (r == null ? '—' : `${r >= 0 ? '+' : ''}${fmtPct(r)}`);

	function select(t: TickerInfo): void {
		app.ticker = t.symbol;
		app.date = '';
		app.focusedSignalId = null;
	}
</script>

<aside
	class={`flex shrink-0 flex-col border-r border-hairline bg-abyss transition-[width] duration-200 motion-reduce:transition-none ${
		app.sidebarCollapsed ? 'w-14' : 'w-60'
	}`}
	aria-label="Instruments"
>
	<div
		class={`flex h-9 shrink-0 items-center border-b border-hairline ${
			app.sidebarCollapsed ? 'justify-center' : 'justify-between px-3'
		}`}
	>
		{#if !app.sidebarCollapsed}
			<span class="text-[10px] font-semibold tracking-[0.18em] text-ink-faint">INSTRUMENTS</span>
		{/if}
		<button
			type="button"
			class="rounded p-1 text-ink-faint hover:bg-raised hover:text-ink focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none"
			onclick={() => (app.sidebarCollapsed = !app.sidebarCollapsed)}
			aria-label={app.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
			aria-expanded={!app.sidebarCollapsed}
		>
			<CaretLeft
				size={14}
				aria-hidden="true"
				class={`transition-transform duration-200 motion-reduce:transition-none ${
					app.sidebarCollapsed ? 'rotate-180' : ''
				}`}
			/>
		</button>
	</div>

	<div class="scroll-thin min-h-0 flex-1 overflow-y-auto overflow-x-hidden py-1">
		{#if tickersQ.isPending}
			<p class="px-3 py-2 text-[11px] text-ink-faint" role="status">
				{app.sidebarCollapsed ? '…' : 'Loading instruments…'}
			</p>
		{:else if tickersQ.isError}
			<p class="px-3 py-2 text-[11px] text-short" role="alert">
				{app.sidebarCollapsed ? '!' : 'Instrument feed unavailable.'}
			</p>
		{:else if tickersQ.data}
			{#each tickersQ.data as t (t.symbol)}
				{@const selected = app.ticker === t.symbol}
				{#if app.sidebarCollapsed}
					<button
						type="button"
						class={`flex w-full flex-col items-center gap-1 border-l-2 py-2 focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-inset focus-visible:outline-none ${
							selected
								? 'border-accent bg-accent-soft'
								: 'border-transparent hover:bg-raised'
						}`}
						onclick={() => select(t)}
						aria-pressed={selected}
						aria-label={`Select ${t.alias}, day return ${fmtRet(t.day_ret)}`}
						title={`${t.alias} · ${fmtPx(t.last_px)} · ${fmtRet(t.day_ret)}`}
					>
						<span class={`font-mono text-[10px] font-bold ${selected ? 'text-ink' : 'text-ink-dim'}`}
							>{t.alias}</span
						>
						<span class={`h-1.5 w-1.5 rounded-full ${retDot(t.day_ret)}`} aria-hidden="true"></span>
					</button>
				{:else}
					<button
						type="button"
						class={`block w-full border-l-2 px-3 py-2 text-left focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-inset focus-visible:outline-none ${
							selected
								? 'border-accent bg-accent-soft'
								: 'border-transparent hover:bg-raised'
						}`}
						onclick={() => select(t)}
						aria-pressed={selected}
						aria-label={`Select ${t.alias}`}
					>
						<span class="flex items-baseline justify-between gap-2">
							<span class="font-mono text-xs font-bold text-ink">{t.alias}</span>
							<span class="num text-xs text-ink-dim">{fmtPx(t.last_px)}</span>
						</span>
						<span class="mt-0.5 flex items-center justify-between gap-2">
							<span class={`text-[10px] tracking-wide ${regimeClass(t.regime)}`}>{t.regime}</span>
							<span class={`num rounded-full px-1.5 py-px text-[10px] ${retPill(t.day_ret)}`}
								>{fmtRet(t.day_ret)}</span
							>
						</span>
						<span
							class="mt-1.5 block h-0.5 overflow-hidden rounded-full bg-raised"
							role="progressbar"
							aria-label={`Signal probability ${fmtPct(t.signal_prob ?? 0, 0)}`}
							aria-valuenow={Math.round((t.signal_prob ?? 0) * 100)}
							aria-valuemin={0}
							aria-valuemax={100}
						>
							<span
								class="block h-full rounded-full bg-accent"
								style={`width: ${(t.signal_prob ?? 0) * 100}%`}
							></span>
						</span>
					</button>
				{/if}
			{/each}
		{/if}
	</div>

	<footer class={`shrink-0 border-t border-hairline py-2 ${app.sidebarCollapsed ? 'px-0' : 'px-3'}`}>
		{#if statusQ.data}
			{@const healthy = statusQ.data.diagnosis.healthy}
			{#if app.sidebarCollapsed}
				<div class="flex justify-center">
					<span
						class={`h-1.5 w-1.5 rounded-full ${healthy ? 'bg-long' : 'bg-short'}`}
						role="status"
						aria-label={`Brain ${healthy ? 'healthy' : 'degraded'}`}
						title={`Brain ${healthy ? 'healthy' : 'degraded'}`}
					></span>
				</div>
			{:else}
				<div class="flex items-center gap-1.5" role="status">
					<span class={`h-1.5 w-1.5 rounded-full ${healthy ? 'bg-long' : 'bg-short'}`} aria-hidden="true"
					></span>
					<span class="text-[10px] font-semibold tracking-wider text-ink-faint">
						BRAIN {healthy ? 'HEALTHY' : 'DEGRADED'}
					</span>
				</div>
				<p class="num mt-1 truncate text-[10px] text-ink-faint">
					lake {statusQ.data.lake.newest_bar?.slice(0, 10) ?? '—'} · {compact.format(
						statusQ.data.lake.rows
					)} rows
				</p>
			{/if}
		{:else if statusQ.isError && !app.sidebarCollapsed}
			<p class="text-[10px] text-short" role="alert">brain status unavailable</p>
		{/if}
	</footer>
</aside>
