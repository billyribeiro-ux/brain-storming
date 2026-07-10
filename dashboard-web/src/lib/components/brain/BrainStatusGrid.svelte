<script lang="ts">
	/**
	 * Brain vitals at a glance: lake size, checkpoint freshness, training
	 * curves, self-diagnosis and the per-ticker regime map. Polls /api/status
	 * once a minute — this is telemetry, not a trading surface.
	 */
	import { createQuery } from '@tanstack/svelte-query';
	import { api } from '$lib/api/client';
	import TrainingSpark from './TrainingSpark.svelte';

	const statusQ = createQuery(() => ({
		queryKey: ['status'],
		queryFn: () => api.status(),
		staleTime: 60_000,
		refetchInterval: 60_000
	}));

	const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 });
	const fmtVal = (v: number | null): string =>
		v == null ? '—' : Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toFixed(4);

	/** Regime → chip colors; mirrors the sidebar's regime coloring. */
	const REGIME_CLASS: Record<string, string> = {
		'trending-up': 'bg-long-soft text-long',
		'trending-down': 'bg-short-soft text-short',
		volatile: 'bg-warn-soft text-warn',
		ranging: 'bg-accent-soft text-accent'
	};
	const regimeClass = (r: string): string => REGIME_CLASS[r] ?? 'bg-raised text-ink-faint';

	const SEVERITY_CLASS: Record<string, string> = {
		critical: 'text-short',
		error: 'text-short',
		warn: 'text-warn',
		warning: 'text-warn',
		info: 'text-ink-faint'
	};
	const severityClass = (s: string): string => SEVERITY_CLASS[s.toLowerCase()] ?? 'text-ink-faint';

	const training = $derived(Object.entries(statusQ.data?.training ?? {}));
	const regimes = $derived(Object.entries(statusQ.data?.regime ?? {}));
</script>

{#if statusQ.isPending}
	<div class="glass flex min-h-24 items-center justify-center p-4 text-xs text-ink-faint" role="status">
		Loading brain status…
	</div>
{:else if statusQ.error}
	<div class="glass flex min-h-24 items-center justify-center p-4 text-xs text-short" role="alert">
		{statusQ.error.message}
	</div>
{:else if statusQ.data}
	{@const s = statusQ.data}
	<div class="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
		<!-- Lake -->
		<div class="glass glass-hover rise-in p-3">
			<h3 class="mb-2 text-[10px] font-semibold tracking-widest text-ink-faint uppercase">Lake</h3>
			<p class="num text-xl font-semibold text-ink" title={s.lake.rows.toLocaleString('en-US')}>
				{compact.format(s.lake.rows)}
				<span class="text-[11px] font-normal text-ink-dim">rows</span>
			</p>
			<dl class="mt-2 flex flex-col gap-1 text-[11px]">
				<div class="flex justify-between gap-2">
					<dt class="text-ink-faint">datasets</dt>
					<dd class="num text-ink-dim">{s.lake.datasets}</dd>
				</div>
				<div class="flex justify-between gap-2">
					<dt class="text-ink-faint">newest bar</dt>
					<dd class="num text-ink-dim">{s.lake.newest_bar ?? '—'}</dd>
				</div>
			</dl>
		</div>

		<!-- Checkpoints -->
		<div class="glass glass-hover rise-in p-3">
			<h3 class="mb-2 text-[10px] font-semibold tracking-widest text-ink-faint uppercase">Checkpoints</h3>
			{#if s.checkpoints.length === 0}
				<p class="text-[11px] text-ink-faint">no checkpoints yet</p>
			{:else}
				<ul class="flex flex-col gap-1.5 text-[11px]">
					{#each s.checkpoints as cp (cp.path)}
						<li class="flex items-baseline justify-between gap-2">
							<span class="truncate font-mono text-ink" title={cp.path}>{cp.layer}</span>
							<span class="num shrink-0 text-ink-dim">
								{#if cp.steps != null}<span class="text-ink-faint">{cp.steps} steps · </span>{/if}{cp.mtime}
							</span>
						</li>
					{/each}
				</ul>
			{/if}
		</div>

		<!-- Training (spans two columns on wide screens) -->
		<div class="glass glass-hover rise-in p-3 md:col-span-2">
			<h3 class="mb-2 text-[10px] font-semibold tracking-widest text-ink-faint uppercase">Training</h3>
			{#if training.length === 0}
				<p class="text-[11px] text-ink-faint">no training runs found</p>
			{:else}
				<ul class="flex flex-col gap-1.5">
					{#each training as [run, t] (run)}
						<li class="flex items-center justify-between gap-3 text-[11px]">
							<span class="w-32 truncate font-mono text-ink" title={run}>{run}</span>
							<span class="num text-ink-faint">
								step <span class="text-ink-dim">{t.last_step == null ? '—' : t.last_step}</span>
							</span>
							<span class="num text-ink-faint">
								total <span class="text-ink-dim">{fmtVal(t.last_total)}</span>
							</span>
							<span class="num text-ink-faint">
								best val <span class="text-ink-dim">{fmtVal(t.best_val)}</span>
							</span>
							<TrainingSpark series={t.series} width={110} height={24} />
						</li>
					{/each}
				</ul>
			{/if}
		</div>

		<!-- Diagnosis -->
		<div class="glass glass-hover rise-in p-3 md:col-span-2 xl:col-span-2">
			<h3 class="mb-2 text-[10px] font-semibold tracking-widest text-ink-faint uppercase">Diagnosis</h3>
			{#if s.diagnosis.healthy}
				<p class="flex items-center gap-2 text-[12px] text-long">
					<span class="pulse-dot" aria-hidden="true"></span>
					all systems healthy
				</p>
			{/if}
			{#if s.diagnosis.findings.length > 0}
				<ul class="mt-1 flex flex-col gap-1 text-[11px]">
					{#each s.diagnosis.findings as f, i (`${f.kind}:${i}`)}
						<li class={severityClass(f.severity)}>
							<span class="font-mono text-[10px] uppercase opacity-70">{f.kind}</span>
							{f.message}
						</li>
					{/each}
				</ul>
			{:else if !s.diagnosis.healthy}
				<p class="text-[11px] text-warn">unhealthy — no findings reported</p>
			{/if}
		</div>

		<!-- Regime map -->
		<div class="glass glass-hover rise-in p-3 md:col-span-2 xl:col-span-2">
			<h3 class="mb-2 text-[10px] font-semibold tracking-widest text-ink-faint uppercase">Regime map</h3>
			{#if regimes.length === 0}
				<p class="text-[11px] text-ink-faint">no regime data</p>
			{:else}
				<div class="flex flex-wrap gap-1.5">
					{#each regimes as [ticker, regime] (ticker)}
						<span
							class={`rounded-full px-2 py-0.5 text-[10px] font-medium ${regimeClass(regime)}`}
							title={regime}
						>
							<span class="font-mono">{ticker}</span>
							<span class="opacity-75">· {regime}</span>
						</span>
					{/each}
				</div>
			{/if}
		</div>
	</div>
{/if}

<style>
	.pulse-dot {
		position: relative;
		display: inline-block;
		width: 8px;
		height: 8px;
		border-radius: 9999px;
		background: var(--color-long);
	}
	.pulse-dot::after {
		content: '';
		position: absolute;
		inset: -4px;
		border-radius: 9999px;
		border: 1px solid var(--color-long);
		animation: pulse-ring 2s ease-out infinite;
	}
	@keyframes pulse-ring {
		0% {
			transform: scale(0.5);
			opacity: 0.9;
		}
		100% {
			transform: scale(1.4);
			opacity: 0;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.pulse-dot::after {
			animation: none;
			opacity: 0.35;
		}
	}
</style>
