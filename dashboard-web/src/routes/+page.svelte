<script lang="ts">
	/**
	 * Command Deck — the main cockpit view.
	 *
	 * Left: the market chart (toolbar + candles + uncertainty strip) and a
	 * slim session stats row. Right: live signals feed and the driver
	 * breakdown for the focused signal.
	 *
	 * Server data flows through TanStack Query keyed on the app-store
	 * selection; live pushes (signals, replay bars) come from the WebSocket
	 * feed and are merged ahead of fetched data.
	 */
	import { createQuery, keepPreviousData } from '@tanstack/svelte-query';
	import Panel from '$lib/components/layout/Panel.svelte';
	import ChartToolbar from '$lib/components/chart/ChartToolbar.svelte';
	import PriceChart from '$lib/components/chart/PriceChart.svelte';
	import UncertaintyStrip from '$lib/components/chart/UncertaintyStrip.svelte';
	import SignalsPanel from '$lib/components/signals/SignalsPanel.svelte';
	import DriverBreakdown from '$lib/components/drivers/DriverBreakdown.svelte';
	import { api } from '$lib/api/client';
	import { live } from '$lib/api/ws.svelte';
	import { app } from '$lib/stores/app.svelte';
	import { fmtPx } from '$lib/utils/format';
	import type { Signal } from '$lib/api/schemas';

	const datesQ = createQuery(() => ({
		queryKey: ['dates', app.ticker],
		queryFn: () => api.dates(app.ticker)
	}));

	/** '' in the store means "newest available session". */
	const sessionDate = $derived(app.date || (datesQ.data?.at(-1) ?? ''));

	const barsQ = createQuery(() => ({
		queryKey: ['bars', app.ticker, sessionDate, app.timeframe],
		queryFn: () => api.bars(app.ticker, sessionDate, app.timeframe),
		enabled: sessionDate !== '',
		placeholderData: keepPreviousData
	}));
	const uncertaintyQ = createQuery(() => ({
		queryKey: ['uncertainty', app.ticker, sessionDate],
		queryFn: () => api.uncertainty(app.ticker, sessionDate),
		enabled: sessionDate !== '',
		placeholderData: keepPreviousData
	}));
	const signalsQ = createQuery(() => ({
		queryKey: ['signals', app.ticker, sessionDate],
		queryFn: () => api.signals({ ticker: app.ticker, from: sessionDate, to: sessionDate }),
		enabled: sessionDate !== '',
		placeholderData: keepPreviousData
	}));
	const tradesQ = createQuery(() => ({
		queryKey: ['trades', app.ticker, sessionDate],
		queryFn: () => api.trades({ ticker: app.ticker, from: sessionDate, to: sessionDate }),
		enabled: sessionDate !== '',
		placeholderData: keepPreviousData
	}));

	const bars = $derived(barsQ.data ?? []);
	const uncertainty = $derived(uncertaintyQ.data ?? []);
	const trades = $derived(tradesQ.data ?? []);

	/** Live signals for this ticker merged ahead of fetched ones (deduped). */
	const signals = $derived.by<Signal[]>(() => {
		const fromFeed = live.signals.filter((s) => s.ticker === app.ticker);
		const seen = new Set(fromFeed.map((s) => s.signal_id));
		const fetched = (signalsQ.data ?? []).filter((s) => !seen.has(s.signal_id));
		return [...fromFeed, ...fetched];
	});

	const liveBar = $derived(live.lastBar[app.ticker] ?? null);
	const focusedSignal = $derived(
		signals.find((s) => s.signal_id === app.focusedSignalId) ?? null
	);

	// Chart panel state: dates gate everything, then bars drive the view.
	const chartLoading = $derived(
		sessionDate === '' ? datesQ.isPending : barsQ.isPending
	);
	const chartError = $derived(datesQ.error?.message ?? barsQ.error?.message ?? null);

	// ---- slim session stats ------------------------------------------------
	const lastPx = $derived(liveBar?.c ?? bars.at(-1)?.c ?? null);
	const dayRange = $derived.by<{ lo: number; hi: number } | null>(() => {
		if (bars.length === 0) return null;
		let lo = Infinity;
		let hi = -Infinity;
		for (const b of bars) {
			if (b.l < lo) lo = b.l;
			if (b.h > hi) hi = b.h;
		}
		return { lo, hi };
	});
</script>

<div class="grid h-full min-h-0 grid-cols-12 gap-3">
	<!-- Left: market chart + session stats -->
	<div class="col-span-8 flex min-h-0 flex-col gap-3">
		<Panel
			title={`Market — ${app.ticker}`}
			subtitle={sessionDate ? `${sessionDate} · ${app.timeframe} · ET` : 'resolving session…'}
			loading={chartLoading}
			error={chartError}
			empty={bars.length === 0 ? 'No bars for this session.' : null}
		>
			<div class="flex flex-col gap-2">
				<ChartToolbar dates={datesQ.data ?? []} />
				<div class="h-[62vh] min-h-64">
					<PriceChart
						{bars}
						signals={app.showSignals ? signals : []}
						trades={app.showTrades ? trades : []}
						{uncertainty}
						focusedSignal={app.showLevels ? focusedSignal : null}
						{liveBar}
					/>
				</div>
				{#if app.showAttention}
					<UncertaintyStrip {uncertainty} />
				{/if}
			</div>
		</Panel>

		<div class="glass flex shrink-0 items-center gap-6 px-4 py-2 text-[11px]" aria-label="Session stats">
			<div class="flex items-baseline gap-1.5">
				<span class="tracking-wider text-ink-faint">LAST</span>
				<span class="num text-ink">{fmtPx(lastPx)}</span>
			</div>
			<div class="flex items-baseline gap-1.5">
				<span class="tracking-wider text-ink-faint">DAY RANGE</span>
				<span class="num text-ink-dim">
					{#if dayRange}
						<span class="text-short">{fmtPx(dayRange.lo)}</span>
						–
						<span class="text-long">{fmtPx(dayRange.hi)}</span>
					{:else}
						—
					{/if}
				</span>
			</div>
			<div class="flex items-baseline gap-1.5">
				<span class="tracking-wider text-ink-faint">BARS</span>
				<span class="num text-ink-dim">{bars.length}</span>
			</div>
			<div class="num ml-auto text-ink-faint">{app.timeframe}</div>
		</div>
	</div>

	<!-- Right: signals feed + driver breakdown -->
	<div class="col-span-4 flex min-h-0 flex-col gap-3">
		<div class="min-h-0 basis-[55%] *:h-full">
			<SignalsPanel
				{signals}
				onFocus={(s: Signal | null) => (app.focusedSignalId = s?.signal_id ?? null)}
			/>
		</div>
		<div class="min-h-0 flex-1 *:h-full">
			<DriverBreakdown signal={focusedSignal} ticker={app.ticker} />
		</div>
	</div>
</div>
