<script lang="ts">
	/**
	 * Trade history + autopsies — the period-filterable trade log.
	 * Toolbar filters (backtest / ticker / date range / verdict) drive a
	 * reactive trades query; summary tiles, CSV export and the equity curve
	 * all derive from the same filtered set. Inspecting a row with an
	 * autopsy loads the (cached) autopsy list and shows the post-mortem.
	 */
	import { createQuery } from '@tanstack/svelte-query';
	import { api } from '$lib/api/client';
	import type { Trade } from '$lib/api/schemas';
	import Panel from '$lib/components/layout/Panel.svelte';
	import TradesTable from '$lib/components/history/TradesTable.svelte';
	import AutopsyDetail from '$lib/components/history/AutopsyDetail.svelte';
	import EquityChart from '$lib/components/history/EquityChart.svelte';
	import { fmtPct, fmtPnl, pnlClass } from '$lib/utils/format';
	import { DownloadSimple } from 'phosphor-svelte';

	// ------------------------------------------------------------------ //
	// Filters
	// ------------------------------------------------------------------ //
	let backtest = $state('');
	let ticker = $state(''); // '' = all tickers
	let from = $state('');
	let to = $state('');
	let verdict = $state(''); // '' = all verdicts
	let inspected = $state<Trade | null>(null);

	const backtestsQ = createQuery(() => ({
		queryKey: ['backtests'],
		queryFn: () => api.backtests()
	}));
	const tickersQ = createQuery(() => ({ queryKey: ['tickers'], queryFn: () => api.tickers() }));

	// Default to the newest backtest once the list arrives.
	$effect(() => {
		const list = backtestsQ.data;
		if (list && list.length > 0 && backtest === '') backtest = list[0].name;
	});

	const selectedBt = $derived(backtestsQ.data?.find((b) => b.name === backtest) ?? null);

	// Date range defaults to the selected backtest's own span (re-applied on
	// backtest change so the log always opens on a meaningful window). The
	// guard keeps background refetches from clobbering a hand-edited range.
	let appliedBacktest = '';
	$effect(() => {
		if (selectedBt && selectedBt.name !== appliedBacktest) {
			appliedBacktest = selectedBt.name;
			from = selectedBt.start ?? '';
			to = selectedBt.end ?? '';
			inspected = null;
		}
	});

	const tradesQ = createQuery(() => ({
		queryKey: ['trades', backtest, ticker, from, to],
		queryFn: () =>
			api.trades({
				backtest: backtest || undefined,
				ticker: ticker || undefined,
				from: from || undefined,
				to: to || undefined
			}),
		enabled: backtest !== ''
	}));

	// Autopsies are only fetched when actually needed (an inspected trade
	// has one, or the verdict filter is active) and then cached for good.
	const needAutopsies = $derived(verdict !== '' || (inspected?.has_autopsy ?? false));
	const autopsiesQ = createQuery(() => ({
		queryKey: ['autopsies', 500],
		queryFn: () => api.autopsies({ limit: 500 }),
		staleTime: Infinity,
		enabled: needAutopsies
	}));

	const filteredTrades = $derived.by(() => {
		const all = tradesQ.data ?? [];
		if (verdict === '') return all;
		const autopsies = autopsiesQ.data;
		if (!autopsies) return all; // still loading — show unfiltered rather than nothing
		const ids = new Set(
			autopsies.filter((a) => a.verdict === verdict).map((a) => a.trade.trade_id)
		);
		return all.filter((t) => ids.has(t.trade_id));
	});

	// ------------------------------------------------------------------ //
	// Summary metrics (client-side, over the filtered set)
	// ------------------------------------------------------------------ //
	const stats = $derived.by(() => {
		const ts = filteredTrades;
		const n = ts.length;
		let pnl = 0;
		let fees = 0;
		let wins = 0;
		for (const t of ts) {
			pnl += t.pnl;
			fees += t.fees;
			if (t.pnl > 0) wins += 1;
		}
		return { n, pnl, fees, winRate: n > 0 ? wins / n : null, avgPnl: n > 0 ? pnl / n : null };
	});

	const inspectedAutopsy = $derived.by(() => {
		if (!inspected?.has_autopsy) return null;
		const id = inspected.trade_id;
		return autopsiesQ.data?.find((a) => a.trade.trade_id === id) ?? null;
	});

	const equityQ = createQuery(() => ({
		queryKey: ['equity', backtest],
		queryFn: () => api.equity(backtest),
		enabled: backtest !== ''
	}));

	// ------------------------------------------------------------------ //
	// CSV export of the filtered trades
	// ------------------------------------------------------------------ //
	const csvEscape = (v: string): string => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);

	function exportCsv(): void {
		const cols = [
			'trade_id',
			'ticker',
			'side',
			'entry_ts',
			'exit_ts',
			'entry_px',
			'exit_px',
			'qty',
			'pnl',
			'fees',
			'stop_px',
			'target_px',
			'exit_reason',
			'conviction',
			'has_autopsy'
		] as const;
		const lines = [
			cols.join(','),
			...filteredTrades.map((t) => cols.map((c) => csvEscape(String(t[c]))).join(','))
		];
		const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `aether_trades_${backtest || 'all'}_${from || 'start'}_${to || 'end'}.csv`;
		a.click();
		URL.revokeObjectURL(url);
	}

	const selectClass =
		'rounded-md border border-hairline bg-raised px-2 py-1 text-xs text-ink outline-none focus-visible:border-accent';

	const VERDICT_OPTIONS = [
		{ value: '', label: 'all verdicts' },
		{ value: 'good_loss', label: 'good loss' },
		{ value: 'bad_loss', label: 'bad loss' },
		{ value: 'good_win', label: 'good win' },
		{ value: 'lucky_win', label: 'lucky win' }
	] as const;
</script>

<svelte:head>
	<title>Aether — Trade History</title>
</svelte:head>

<div class="flex flex-col gap-3">
	<!-- toolbar -->
	<div class="glass flex flex-wrap items-end gap-3 px-3 py-2">
		<label class="flex flex-col gap-0.5 text-[10px] tracking-wide text-ink-faint uppercase">
			Backtest
			<select class={selectClass} bind:value={backtest} disabled={backtestsQ.isPending}>
				{#if backtestsQ.isPending}
					<option value="">loading…</option>
				{:else}
					{#each backtestsQ.data ?? [] as b (b.name)}
						<option value={b.name}>{b.name}</option>
					{/each}
				{/if}
			</select>
		</label>
		<label class="flex flex-col gap-0.5 text-[10px] tracking-wide text-ink-faint uppercase">
			Ticker
			<select class={selectClass} bind:value={ticker}>
				<option value="">all</option>
				{#each tickersQ.data ?? [] as t (t.symbol)}
					<option value={t.symbol}>{t.alias}</option>
				{/each}
			</select>
		</label>
		<label class="flex flex-col gap-0.5 text-[10px] tracking-wide text-ink-faint uppercase">
			From
			<input type="date" class={`num ${selectClass}`} bind:value={from} max={to || undefined} />
		</label>
		<label class="flex flex-col gap-0.5 text-[10px] tracking-wide text-ink-faint uppercase">
			To
			<input type="date" class={`num ${selectClass}`} bind:value={to} min={from || undefined} />
		</label>
		<label class="flex flex-col gap-0.5 text-[10px] tracking-wide text-ink-faint uppercase">
			Verdict
			<select class={selectClass} bind:value={verdict}>
				{#each VERDICT_OPTIONS as v (v.value)}
					<option value={v.value}>{v.label}</option>
				{/each}
			</select>
		</label>
		<button
			type="button"
			class="ml-auto inline-flex items-center gap-1.5 rounded-md border border-hairline bg-raised px-2.5 py-1.5 text-xs text-ink transition-colors hover:border-hairline-strong focus-visible:outline focus-visible:outline-1 focus-visible:outline-accent disabled:opacity-40"
			onclick={exportCsv}
			disabled={filteredTrades.length === 0}
			aria-label="Export filtered trades as CSV"
		>
			<DownloadSimple size={14} aria-hidden="true" />
			CSV
		</button>
	</div>

	<!-- summary tiles -->
	<div class="grid grid-cols-2 gap-3 sm:grid-cols-5" aria-label="Summary metrics">
		<div class="glass px-3 py-2">
			<p class="text-[10px] tracking-wide text-ink-faint uppercase">Trades</p>
			<p class="num text-lg text-ink">{stats.n.toLocaleString('en-US')}</p>
		</div>
		<div class="glass px-3 py-2">
			<p class="text-[10px] tracking-wide text-ink-faint uppercase">Net PnL</p>
			<p class={`num text-lg ${pnlClass(stats.pnl)}`}>{fmtPnl(stats.pnl)}</p>
		</div>
		<div class="glass px-3 py-2">
			<p class="text-[10px] tracking-wide text-ink-faint uppercase">Win rate</p>
			<p class="num text-lg text-ink">{fmtPct(stats.winRate)}</p>
		</div>
		<div class="glass px-3 py-2">
			<p class="text-[10px] tracking-wide text-ink-faint uppercase">Avg PnL</p>
			<p class={`num text-lg ${stats.avgPnl == null ? 'text-ink' : pnlClass(stats.avgPnl)}`}>
				{fmtPnl(stats.avgPnl)}
			</p>
		</div>
		<div class="glass px-3 py-2">
			<p class="text-[10px] tracking-wide text-ink-faint uppercase">Fees</p>
			<p class="num text-lg text-ink">{fmtPnl(-stats.fees)}</p>
		</div>
	</div>

	<!-- main grid -->
	<div class="grid grid-cols-12 gap-3">
		<Panel
			title="Trades"
			subtitle={`${backtest || '—'} · ${from || '…'} → ${to || '…'}`}
			class="col-span-12 h-[34rem] lg:col-span-7"
			loading={tradesQ.isPending && backtest !== ''}
			error={tradesQ.error ? tradesQ.error.message : null}
		>
			<TradesTable
				trades={filteredTrades}
				selectedId={inspected?.trade_id ?? null}
				onInspect={(t) => (inspected = t)}
			/>
		</Panel>

		<Panel
			title="Autopsy"
			subtitle={inspected ? inspected.trade_id : 'select a trade'}
			class="col-span-12 h-[34rem] lg:col-span-5"
			loading={inspected?.has_autopsy === true && autopsiesQ.isPending}
			error={inspected?.has_autopsy && autopsiesQ.error ? autopsiesQ.error.message : null}
			empty={!inspected
				? 'Click a trade row to inspect it.'
				: !inspected.has_autopsy
					? 'No autopsy for this trade.'
					: inspectedAutopsy === null && !autopsiesQ.isPending
						? 'Autopsy not found in the latest 500.'
						: null}
		>
			{#if inspectedAutopsy}
				<AutopsyDetail autopsy={inspectedAutopsy} />
			{/if}
		</Panel>

		<Panel
			title="Equity"
			subtitle={backtest || 'no backtest selected'}
			class="col-span-12"
			loading={equityQ.isPending && backtest !== ''}
			error={equityQ.error ? equityQ.error.message : null}
			empty={equityQ.data && equityQ.data.length === 0 ? 'No equity points.' : null}
		>
			<EquityChart points={equityQ.data ?? []} height={220} />
		</Panel>
	</div>
</div>
