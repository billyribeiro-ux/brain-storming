<script lang="ts">
	/**
	 * Simulation playground: replay a recorded session through the cockpit
	 * and poke the world model's imagination at any anchor minute. The
	 * right column mirrors the live feed so replayed signals/trades are
	 * observable without leaving the page.
	 */
	import { createQuery } from '@tanstack/svelte-query';
	import { api } from '$lib/api/client';
	import type { ImaginationFan as Fan, Signal, Trade } from '$lib/api/schemas';
	import { live } from '$lib/api/ws.svelte';
	import { app } from '$lib/stores/app.svelte';
	import { fmtPct, fmtPnl, fmtPx, fmtTime, pnlClass, sideClass } from '$lib/utils/format';
	import Panel from '$lib/components/layout/Panel.svelte';
	import ReplayControls from '$lib/components/playground/ReplayControls.svelte';
	import ImaginationFan from '$lib/components/playground/ImaginationFan.svelte';
	import { ArrowUpRight, CircleNotch, Sparkle } from 'phosphor-svelte';

	// ------------------------------------------------------------------ //
	// Imagination anchor picker — ticker follows the global app selection.
	// ------------------------------------------------------------------ //
	const N_OPTIONS = [8, 16, 32] as const;
	const HORIZON_OPTIONS = [15, 30, 60] as const;

	let date = $state('');
	let minute = $state(180); // minutes after the 09:30 open, 0..389
	let n = $state(8);
	let horizon = $state(15);

	const datesQ = createQuery(() => ({
		queryKey: ['dates', app.ticker],
		queryFn: () => api.dates(app.ticker),
		staleTime: Infinity
	}));
	$effect(() => {
		const dates = datesQ.data;
		if (dates && dates.length > 0 && !dates.includes(date)) date = dates[dates.length - 1];
	});

	/** date @ 09:30 + minute, as a naive-ET epoch (the lake's convention). */
	const anchorTs = $derived.by(() => {
		if (!date) return null;
		const [y, m, d] = date.split('-').map(Number);
		if (!y || !m || !d) return null;
		return Date.UTC(y, m - 1, d, 9, 30) / 1000 + minute * 60;
	});

	let fan = $state<Fan | null>(null);
	let imagining = $state(false);
	let imagineError = $state<string | null>(null);

	async function imagine(): Promise<void> {
		if (anchorTs === null || imagining) return;
		imagining = true;
		imagineError = null;
		try {
			fan = await api.imagine({ ticker: app.ticker, anchor_ts: anchorTs, n, horizon });
		} catch (e) {
			// e.g. 409 when the world-model checkpoint is missing — surface
			// the bridge's own message inline.
			imagineError = e instanceof Error ? e.message : String(e);
		} finally {
			imagining = false;
		}
	}

	// ------------------------------------------------------------------ //
	// Cockpit mirror — most recent replayed signals + trades, merged.
	// ------------------------------------------------------------------ //
	type FeedRow =
		| { key: string; ts: number; kind: 'signal'; s: Signal }
		| { key: string; ts: number; kind: 'trade'; t: Trade };

	const feed = $derived.by((): FeedRow[] => {
		const rows: FeedRow[] = [
			...live.signals.map(
				(s): FeedRow => ({ key: `s-${s.signal_id}`, ts: s.ts, kind: 'signal', s })
			),
			...live.trades.map((t): FeedRow => ({ key: `t-${t.trade_id}`, ts: t.exit_ts, kind: 'trade', t }))
		];
		return rows.sort((a, b) => b.ts - a.ts).slice(0, 20);
	});

	const selectClass =
		'rounded-md border border-hairline bg-raised px-2 py-1 text-xs text-ink outline-none focus-visible:border-accent';

	const segBtn = (active: boolean): string =>
		`num px-2 py-1 text-xs transition-colors focus-visible:outline focus-visible:-outline-offset-1 focus-visible:outline-accent ${
			active ? 'bg-accent-soft text-accent' : 'bg-raised text-ink-dim hover:text-ink'
		}`;
</script>

<svelte:head>
	<title>Aether — Playground</title>
</svelte:head>

<div class="grid grid-cols-12 items-start gap-3">
	<div class="col-span-12 flex flex-col gap-3 lg:col-span-5">
		<ReplayControls />

		<Panel title="Imagination" subtitle="roll the world model forward from any anchor minute">
			<div class="flex flex-col gap-3">
				<div class="flex flex-wrap items-end gap-3">
					<label class="flex flex-col gap-0.5 text-[10px] tracking-wide text-ink-faint uppercase">
						Ticker
						<span class="num rounded-md border border-hairline bg-raised px-2 py-1 text-xs text-ink"
							>{app.ticker}</span
						>
					</label>
					<label class="flex flex-col gap-0.5 text-[10px] tracking-wide text-ink-faint uppercase">
						Date
						<input
							type="date"
							class={`num ${selectClass}`}
							bind:value={date}
							min={datesQ.data?.[0]}
							max={datesQ.data?.at(-1)}
							disabled={datesQ.isPending}
						/>
					</label>
				</div>

				<label class="flex flex-col gap-1 text-[10px] tracking-wide text-ink-faint uppercase">
					<span>
						Anchor minute —
						<span class="num text-ink normal-case">
							{anchorTs !== null ? fmtTime(anchorTs) : '—'}
						</span>
					</span>
					<input
						type="range"
						min="0"
						max="389"
						step="1"
						bind:value={minute}
						class="accent-accent"
						aria-label="Anchor minute within the session"
					/>
				</label>

				<div class="flex flex-wrap items-end gap-3">
					<fieldset class="flex flex-col gap-0.5">
						<legend class="text-[10px] tracking-wide text-ink-faint uppercase">Rollouts</legend>
						<div class="flex overflow-hidden rounded-md border border-hairline" role="group">
							{#each N_OPTIONS as v (v)}
								<button type="button" class={segBtn(n === v)} aria-pressed={n === v} onclick={() => (n = v)}
									>{v}</button
								>
							{/each}
						</div>
					</fieldset>
					<fieldset class="flex flex-col gap-0.5">
						<legend class="text-[10px] tracking-wide text-ink-faint uppercase">Horizon</legend>
						<div class="flex overflow-hidden rounded-md border border-hairline" role="group">
							{#each HORIZON_OPTIONS as v (v)}
								<button
									type="button"
									class={segBtn(horizon === v)}
									aria-pressed={horizon === v}
									onclick={() => (horizon = v)}>{v}</button
								>
							{/each}
						</div>
					</fieldset>
					<button
						type="button"
						class="inline-flex items-center gap-1.5 rounded-md bg-accent-soft px-3 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent/25 focus-visible:outline focus-visible:outline-1 focus-visible:outline-accent disabled:opacity-40"
						onclick={imagine}
						disabled={imagining || anchorTs === null}
					>
						{#if imagining}
							<CircleNotch size={13} class="animate-spin" aria-hidden="true" />
						{:else}
							<Sparkle size={13} weight="fill" aria-hidden="true" />
						{/if}
						Imagine
					</button>
				</div>

				{#if imagineError}
					<p class="text-xs break-words text-short" role="alert">{imagineError}</p>
				{/if}

				{#if imagining}
					<div class="flex min-h-32 items-center justify-center text-ink-faint" role="status">
						<CircleNotch size={18} class="animate-spin" aria-hidden="true" />
						<span class="ml-2 text-xs">imagining {n} futures…</span>
					</div>
				{:else if fan}
					<ImaginationFan {fan} />
				{:else if !imagineError}
					<p class="min-h-16 text-xs text-ink-faint">
						Pick an anchor and press Imagine to roll the world model forward.
					</p>
				{/if}
			</div>
		</Panel>
	</div>

	<Panel
		title="Cockpit Mirror"
		subtitle="replayed signals and trades, as the deck sees them"
		class="col-span-12 min-h-[24rem] lg:col-span-7"
	>
		{#snippet actions()}
			<a
				href="/"
				class="inline-flex items-center gap-1 rounded-full bg-accent-soft px-2 py-0.5 text-[11px] text-accent transition-colors hover:bg-accent/25 focus-visible:outline focus-visible:outline-1 focus-visible:outline-accent"
			>
				open Deck
				<ArrowUpRight size={11} aria-hidden="true" />
			</a>
		{/snippet}

		<div class="flex flex-col gap-2">
			<p class="text-xs leading-relaxed text-ink-dim">
				While a replay runs, the bridge re-emits that session's recorded signals and fills over the
				same WebSocket the live cockpit uses — the deck chart follows the replayed ticker
				automatically. The 20 most recent events land here.
			</p>

			{#if feed.length === 0}
				<div class="flex min-h-24 items-center justify-center text-xs text-ink-faint">
					No live events yet — start a replay to populate the feed.
				</div>
			{:else}
				<ul class="flex flex-col" aria-label="Recent replayed events">
					{#each feed as row (row.key)}
						<li class="num flex items-center gap-2 border-t border-hairline px-1 py-1 text-[11px] first:border-t-0">
							{#if row.kind === 'signal'}
								<span class="w-14 text-ink-faint">{fmtTime(row.ts)}</span>
								<span class="w-9 rounded bg-accent-soft px-1 text-center text-[9px] tracking-wide text-accent uppercase"
									>sig</span
								>
								<span class="w-12 font-medium text-ink">{row.s.ticker}</span>
								<span class={`w-12 uppercase ${sideClass(row.s.side)}`}>{row.s.side}</span>
								<span class="text-ink-dim">@ {fmtPx(row.s.entry_px)}</span>
								<span class="ml-auto text-ink-faint">conv {fmtPct(row.s.conviction, 0)}</span>
							{:else}
								<span class="w-14 text-ink-faint">{fmtTime(row.ts)}</span>
								<span class="w-9 rounded bg-raised px-1 text-center text-[9px] tracking-wide text-ink-dim uppercase"
									>trd</span
								>
								<span class="w-12 font-medium text-ink">{row.t.ticker}</span>
								<span class={`w-12 uppercase ${sideClass(row.t.side)}`}>{row.t.side}</span>
								<span class="text-ink-dim">{row.t.exit_reason} @ {fmtPx(row.t.exit_px)}</span>
								<span class={`ml-auto ${pnlClass(row.t.pnl)}`}>{fmtPnl(row.t.pnl)}</span>
							{/if}
						</li>
					{/each}
				</ul>
			{/if}
		</div>
	</Panel>
</div>
