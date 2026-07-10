<script lang="ts">
	/**
	 * Session replay control deck: pick a ticker / recorded session date /
	 * speed and stream it through the cockpit via the bridge's replay
	 * engine. The status line mirrors the WebSocket `replay_state`; the
	 * cursor is read from the latest replayed bar (`live.lastBar`) since the
	 * feed store keeps only the freshest bar per ticker.
	 *
	 * COUPLING: while a replay is running this component sets
	 * `app.ticker = live.replay.ticker` so the main deck chart follows the
	 * replayed instrument — the deck subscribes to `app.ticker` and
	 * `live.lastBar`, so pointing the global selection at the replay ticker
	 * is all it takes for the whole cockpit to ride along.
	 */
	import { createQuery } from '@tanstack/svelte-query';
	import { api } from '$lib/api/client';
	import { live } from '$lib/api/ws.svelte';
	import { app } from '$lib/stores/app.svelte';
	import { fmtTime } from '$lib/utils/format';
	import Panel from '$lib/components/layout/Panel.svelte';
	import { Play, Stop } from 'phosphor-svelte';

	/** Fallback aliases if /api/tickers is unreachable. */
	const FALLBACK_TICKERS = [
		'AAPL',
		'NVDA',
		'TSLA',
		'AMZN',
		'NFLX',
		'CSCO',
		'SPY',
		'QQQ',
		'IWM',
		'SPX'
	] as const;

	const SPEEDS = [30, 60, 120, 300] as const;
	const SESSION_SECONDS = 390 * 60; // 09:30 → 16:00 ET

	let ticker = $state('AAPL');
	let speed = $state(120);
	let busy = $state(false);
	let actionError = $state<string | null>(null);

	const tickersQ = createQuery(() => ({ queryKey: ['tickers'], queryFn: () => api.tickers() }));
	const aliases = $derived(
		tickersQ.data && tickersQ.data.length > 0
			? tickersQ.data.map((t) => t.symbol)
			: [...FALLBACK_TICKERS]
	);

	const datesQ = createQuery(() => ({
		queryKey: ['dates', ticker],
		queryFn: () => api.dates(ticker),
		staleTime: Infinity
	}));

	// Date defaults to the newest recorded session; a user pick overrides it
	// but silently falls back when it isn't in the current ticker's list.
	let dateOverride = $state<string | null>(null);
	const date = $derived.by(() => {
		const dates = datesQ.data;
		if (!dates || dates.length === 0) return '';
		if (dateOverride !== null && dates.includes(dateOverride)) return dateOverride;
		return dates[dates.length - 1];
	});
	const minDate = $derived(datesQ.data?.[0]);
	const maxDate = $derived(datesQ.data?.at(-1));

	async function send(action: 'start' | 'stop'): Promise<void> {
		busy = true;
		actionError = null;
		try {
			await api.replay(
				action === 'start' ? { action, ticker, date, speed } : { action }
			);
		} catch (e) {
			actionError = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	// COUPLING (see header comment): follow the replayed instrument so the
	// deck chart streams the same session the user just started. This is an
	// intentional cross-store side effect — the global selection must track
	// the replay — so $effect (not $derived) is the right tool here.
	$effect(() => {
		if (live.replay.running && live.replay.ticker) app.ticker = live.replay.ticker;
	});

	/** Session start (09:30 naive-ET epoch) for the replayed date. */
	const sessionStart = $derived.by(() => {
		const d = live.replay.date;
		if (!d) return null;
		const [y, m, day] = d.split('-').map(Number);
		if (!y || !m || !day) return null;
		return Date.UTC(y, m - 1, day, 9, 30) / 1000;
	});

	/** Replay cursor ≈ timestamp of the latest replayed bar for the ticker. */
	const cursor = $derived.by(() => {
		const tk = live.replay.ticker;
		return tk ? (live.lastBar[tk]?.t ?? null) : null;
	});

	const progress = $derived.by(() => {
		if (cursor === null || sessionStart === null) return 0;
		return Math.min(1, Math.max(0, (cursor - sessionStart) / SESSION_SECONDS));
	});

	const selectClass =
		'rounded-md border border-hairline bg-raised px-2 py-1 text-xs text-ink outline-none focus-visible:border-accent';
</script>

<Panel
	title="Session Replay"
	subtitle="stream a recorded session through the cockpit — real lake bars, real recorded signals"
>
	<div class="flex flex-col gap-3">
		<div class="flex flex-wrap items-end gap-3">
			<label class="flex flex-col gap-0.5 text-[10px] tracking-wide text-ink-faint uppercase">
				Ticker
				<select class={selectClass} bind:value={ticker} disabled={live.replay.running}>
					{#each aliases as a (a)}
						<option value={a}>{a}</option>
					{/each}
				</select>
			</label>
			<label class="flex flex-col gap-0.5 text-[10px] tracking-wide text-ink-faint uppercase">
				Session date
				<input
					type="date"
					class={`num ${selectClass}`}
					value={date}
					onchange={(e) => (dateOverride = e.currentTarget.value)}
					min={minDate}
					max={maxDate}
					disabled={datesQ.isPending || live.replay.running}
				/>
			</label>
			<fieldset class="flex flex-col gap-0.5">
				<legend class="text-[10px] tracking-wide text-ink-faint uppercase">Speed</legend>
				<div class="flex overflow-hidden rounded-md border border-hairline" role="group">
					{#each SPEEDS as s (s)}
						<button
							type="button"
							class={`num px-2 py-1 text-xs transition-colors focus-visible:outline focus-visible:-outline-offset-1 focus-visible:outline-accent ${
								speed === s ? 'bg-accent-soft text-accent' : 'bg-raised text-ink-dim hover:text-ink'
							}`}
							aria-pressed={speed === s}
							onclick={() => (speed = s)}
							disabled={live.replay.running}>{s}x</button
						>
					{/each}
				</div>
			</fieldset>
			<div class="flex gap-1.5">
				<button
					type="button"
					class="inline-flex items-center gap-1.5 rounded-md bg-accent-soft px-3 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent/25 focus-visible:outline focus-visible:outline-1 focus-visible:outline-accent disabled:opacity-40"
					onclick={() => send('start')}
					disabled={busy || live.replay.running || date === ''}
				>
					<Play size={13} weight="fill" aria-hidden="true" />
					Start
				</button>
				<button
					type="button"
					class="inline-flex items-center gap-1.5 rounded-md bg-short-soft px-3 py-1.5 text-xs font-medium text-short transition-colors hover:bg-short/25 focus-visible:outline focus-visible:outline-1 focus-visible:outline-short disabled:opacity-40"
					onclick={() => send('stop')}
					disabled={busy || !live.replay.running}
				>
					<Stop size={13} weight="fill" aria-hidden="true" />
					Stop
				</button>
			</div>
		</div>

		{#if actionError}
			<p class="text-xs text-short" role="alert">{actionError}</p>
		{/if}

		<!-- live status line -->
		<div class="flex items-center gap-2 text-xs" role="status" aria-live="polite">
			{#if live.replay.running}
				<span class="relative flex h-2 w-2" aria-hidden="true">
					<span class="absolute inline-flex h-full w-full animate-ping rounded-full bg-long opacity-60"
					></span>
					<span class="relative inline-flex h-2 w-2 rounded-full bg-long"></span>
				</span>
				<span class="num text-ink">
					{live.replay.ticker}
					{live.replay.date} · {cursor !== null ? fmtTime(cursor) : '--:--:--'} · {live.replay
						.speed}x
				</span>
				<div
					class="h-1 flex-1 overflow-hidden rounded bg-raised"
					role="progressbar"
					aria-label="Session progress"
					aria-valuemin={0}
					aria-valuemax={100}
					aria-valuenow={Math.round(progress * 100)}
				>
					<div class="h-full bg-long" style:width={`${(progress * 100).toFixed(1)}%`}></div>
				</div>
			{:else}
				<span class="h-2 w-2 rounded-full bg-ink-faint" aria-hidden="true"></span>
				<span class="text-ink-faint">replay idle</span>
			{/if}
		</div>
	</div>
</Panel>
