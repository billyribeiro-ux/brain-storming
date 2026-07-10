<script lang="ts">
	/**
	 * Compact chart toolbar (28px controls): ticker badge, timeframe
	 * segmented control, overlay toggle pills, and the session date picker.
	 * Selection state lives in the global app store; the available session
	 * dates come in as a prop (fetched by the deck).
	 */
	import { CrosshairIcon, LightningIcon, TargetIcon, WavesIcon } from 'phosphor-svelte';
	import { app, type Timeframe } from '$lib/stores/app.svelte';

	let { dates = [] }: { dates?: string[] } = $props();

	const timeframes: Timeframe[] = ['1min', '5min'];
	const minDate = $derived(dates.length > 0 ? dates[0] : '');
	const maxDate = $derived(dates.length > 0 ? dates[dates.length - 1] : '');
	/** '' in the store means "newest session" — show it explicitly. */
	const shownDate = $derived(app.date || maxDate);

	const pillClass = (on: boolean): string =>
		`flex h-7 items-center gap-1 rounded-md border px-2 text-[10px] font-medium transition-colors ${
			on
				? 'border-hairline-strong bg-accent-soft text-accent'
				: 'border-hairline text-ink-faint hover:bg-raised hover:text-ink'
		}`;
</script>

<div class="flex flex-wrap items-center gap-2">
	<span
		class="num flex h-7 items-center rounded-md border border-hairline bg-accent-soft px-2 text-[11px] font-semibold text-accent"
		aria-label={`Ticker ${app.ticker}`}
	>
		{app.ticker}
	</span>

	<div
		class="flex h-7 overflow-hidden rounded-md border border-hairline"
		role="group"
		aria-label="Timeframe"
	>
		{#each timeframes as tf (tf)}
			<button
				type="button"
				class={`num px-2 text-[11px] transition-colors ${
					app.timeframe === tf
						? 'bg-accent-soft text-accent'
						: 'text-ink-dim hover:bg-raised hover:text-ink'
				}`}
				aria-pressed={app.timeframe === tf}
				aria-label={`Timeframe ${tf}`}
				onclick={() => (app.timeframe = tf)}
			>
				{tf}
			</button>
		{/each}
	</div>

	<div class="flex items-center gap-1" role="group" aria-label="Chart overlays">
		<button
			type="button"
			class={pillClass(app.showSignals)}
			aria-pressed={app.showSignals}
			aria-label="Toggle signal markers"
			onclick={() => (app.showSignals = !app.showSignals)}
		>
			<LightningIcon size={12} aria-hidden="true" />
			Signals
		</button>
		<button
			type="button"
			class={pillClass(app.showTrades)}
			aria-pressed={app.showTrades}
			aria-label="Toggle trade exit markers"
			onclick={() => (app.showTrades = !app.showTrades)}
		>
			<TargetIcon size={12} aria-hidden="true" />
			Trades
		</button>
		<button
			type="button"
			class={pillClass(app.showLevels)}
			aria-pressed={app.showLevels}
			aria-label="Toggle entry, stop and target level lines"
			onclick={() => (app.showLevels = !app.showLevels)}
		>
			<CrosshairIcon size={12} aria-hidden="true" />
			Levels
		</button>
		<button
			type="button"
			class={pillClass(app.showAttention)}
			aria-pressed={app.showAttention}
			aria-label="Toggle uncertainty attention strip"
			onclick={() => (app.showAttention = !app.showAttention)}
		>
			<WavesIcon size={12} aria-hidden="true" />
			Attention
		</button>
	</div>

	<input
		type="date"
		class="num h-7 rounded-md border border-hairline bg-surface px-2 text-[11px] text-ink-dim [color-scheme:dark] hover:text-ink focus:border-hairline-strong focus:outline-none"
		value={shownDate}
		min={minDate}
		max={maxDate}
		aria-label="Session date"
		onchange={(e) => (app.date = e.currentTarget.value)}
	/>
</div>
