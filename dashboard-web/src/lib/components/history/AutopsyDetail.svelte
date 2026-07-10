<script lang="ts">
	/**
	 * Post-mortem view of a single trade: verdict banner, trade facts,
	 * narrative, counterfactual ladder (best alternative ringed), ranked
	 * driver bars and recorded lessons. `autopsy.trade` is a *partial*
	 * Trade — every fact is guarded so a sparse payload degrades to '—'.
	 */
	import type { Autopsy } from '$lib/api/schemas';
	import { fmtDateTime, fmtPnl, fmtPx, pnlClass } from '$lib/utils/format';
	import { CloudRain, Clover, Trophy, Wrench, type IconComponentProps } from 'phosphor-svelte';
	import type { Component } from 'svelte';

	let { autopsy }: { autopsy: Autopsy } = $props();

	type Verdict = Autopsy['verdict'];
	type Icon = Component<IconComponentProps, Record<string, never>, ''>;
	const VERDICTS: Record<Verdict, { label: string; klass: string; icon: Icon }> = {
		good_loss: { label: 'UNLUCKY — THESIS WRONG', klass: 'bg-warn-soft text-warn', icon: CloudRain },
		bad_loss: { label: 'FIXABLE — EXECUTION', klass: 'bg-short-soft text-short', icon: Wrench },
		good_win: { label: 'EARNED — PROCESS RIGHT', klass: 'bg-long-soft text-long', icon: Trophy },
		lucky_win: { label: 'LUCKY — REVIEW THE PROCESS', klass: 'bg-warn-soft text-warn', icon: Clover }
	};
	const verdict = $derived(VERDICTS[autopsy.verdict]);

	const t = $derived(autopsy.trade);

	const fmtTs = (ts: number | undefined): string => (ts == null ? '—' : fmtDateTime(ts));
	const fmtNum = (v: number | undefined, dp = 2): string => (v == null ? '—' : fmtPx(v, dp));

	/** Loosely-typed bridge records → typed rows, dropping malformed entries. */
	const asStr = (v: unknown): string | null => (typeof v === 'string' ? v : null);
	const asNum = (v: unknown): number | null =>
		typeof v === 'number' && Number.isFinite(v) ? v : null;

	const drivers = $derived.by(() => {
		const rows: { name: string; attribution: number }[] = [];
		for (const d of autopsy.drivers) {
			const name = asStr(d['name']);
			const attribution = asNum(d['attribution']);
			if (name !== null && attribution !== null) rows.push({ name, attribution });
		}
		rows.sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution));
		const maxAbs = rows.length ? Math.abs(rows[0].attribution) || 1 : 1;
		return rows.map((r) => ({ ...r, frac: Math.abs(r.attribution) / maxAbs }));
	});

	const lessons = $derived.by(() =>
		autopsy.lessons
			.map((l) => ({ kind: asStr(l['kind']) ?? 'lesson', weight: asNum(l['weight']) }))
			.filter((l) => l.kind !== 'lesson' || l.weight !== null)
	);

	const counterfactuals = $derived(
		[...autopsy.counterfactuals].sort((a, b) => b.delta - a.delta)
	);
</script>

<div class="flex flex-col gap-3">
	<!-- verdict banner -->
	<div class={`flex items-center gap-2.5 rounded-lg px-3 py-2.5 ${verdict.klass}`} role="status">
		<verdict.icon size={22} weight="duotone" aria-hidden="true" />
		<div class="min-w-0">
			<p class="num text-[13px] font-semibold tracking-wider">{verdict.label}</p>
			<p class="text-[10px] tracking-wide uppercase opacity-70">verdict: {autopsy.verdict}</p>
		</div>
	</div>

	<!-- trade facts -->
	<dl class="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px] sm:grid-cols-3">
		<div>
			<dt class="text-[10px] tracking-wide text-ink-faint uppercase">Entry</dt>
			<dd class="num text-ink">{fmtTs(t.entry_ts)} @ {fmtNum(t.entry_px)}</dd>
		</div>
		<div>
			<dt class="text-[10px] tracking-wide text-ink-faint uppercase">Exit</dt>
			<dd class="num text-ink">{fmtTs(t.exit_ts)} @ {fmtNum(t.exit_px)}</dd>
		</div>
		<div>
			<dt class="text-[10px] tracking-wide text-ink-faint uppercase">Qty</dt>
			<dd class="num text-ink">{fmtNum(t.qty, 0)}</dd>
		</div>
		<div>
			<dt class="text-[10px] tracking-wide text-ink-faint uppercase">Stop</dt>
			<dd class="num text-ink">{fmtNum(t.stop_px)}</dd>
		</div>
		<div>
			<dt class="text-[10px] tracking-wide text-ink-faint uppercase">Target</dt>
			<dd class="num text-ink">{fmtNum(t.target_px)}</dd>
		</div>
		<div>
			<dt class="text-[10px] tracking-wide text-ink-faint uppercase">PnL</dt>
			<dd class={`num ${t.pnl == null ? 'text-ink' : pnlClass(t.pnl)}`}>
				{t.pnl == null ? '—' : fmtPnl(t.pnl)}
			</dd>
		</div>
		<div>
			<dt class="text-[10px] tracking-wide text-ink-faint uppercase">Reason</dt>
			<dd class="num text-ink">{t.exit_reason ?? '—'}</dd>
		</div>
	</dl>

	<!-- narrative -->
	<p class="text-xs leading-relaxed text-ink-dim">{autopsy.narrative}</p>

	<!-- counterfactuals -->
	{#if counterfactuals.length > 0}
		<div>
			<h3 class="mb-1 text-[10px] font-semibold tracking-wide text-ink-faint uppercase">
				Counterfactuals
			</h3>
			<table class="w-full border-collapse text-[11px]" aria-label="Counterfactuals">
				<thead>
					<tr class="text-left text-[10px] text-ink-faint">
						<th class="px-2 py-1 font-medium tracking-wide uppercase">Variation</th>
						<th class="px-2 py-1 text-right font-medium tracking-wide uppercase">PnL</th>
						<th class="px-2 py-1 text-right font-medium tracking-wide uppercase">Δ vs actual</th>
					</tr>
				</thead>
				<tbody>
					{#each counterfactuals as cf, i (cf.description)}
						<tr
							class={`border-t border-hairline ${
								i === 0 ? 'rounded bg-accent-soft ring-1 ring-accent ring-inset' : ''
							}`}
						>
							<td class="px-2 py-1 text-ink-dim">
								{cf.description}
								{#if i === 0}<span class="ml-1 text-[9px] tracking-wide text-accent uppercase"
										>best</span
									>{/if}
							</td>
							<td class={`num px-2 py-1 text-right ${pnlClass(cf.pnl)}`}>{fmtPnl(cf.pnl)}</td>
							<td class={`num px-2 py-1 text-right ${pnlClass(cf.delta)}`}>{fmtPnl(cf.delta)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}

	<!-- drivers -->
	{#if drivers.length > 0}
		<div>
			<h3 class="mb-1 text-[10px] font-semibold tracking-wide text-ink-faint uppercase">
				Drivers
			</h3>
			<ul class="flex flex-col gap-1">
				{#each drivers as d (d.name)}
					<li class="flex items-center gap-2 text-[11px]">
						<span class="num w-40 truncate text-ink-dim" title={d.name}>{d.name}</span>
						<div class="h-1.5 flex-1 overflow-hidden rounded bg-raised" aria-hidden="true">
							<div
								class={`h-full rounded ${d.attribution >= 0 ? 'bg-long' : 'bg-short'}`}
								style:width={`${(d.frac * 100).toFixed(1)}%`}
							></div>
						</div>
						<span class={`num w-12 text-right ${pnlClass(d.attribution)}`}
							>{(d.attribution * 100).toFixed(0)}%</span
						>
					</li>
				{/each}
			</ul>
		</div>
	{/if}

	<!-- lessons -->
	{#if lessons.length > 0}
		<div>
			<h3 class="mb-1 text-[10px] font-semibold tracking-wide text-ink-faint uppercase">
				Lessons
			</h3>
			<ul class="flex flex-wrap gap-1.5">
				{#each lessons as l, i (i)}
					<li class="num rounded-full bg-raised px-2 py-0.5 text-[10px] text-ink-dim">
						{l.kind}{#if l.weight !== null}
							· w {l.weight.toFixed(2)}{/if}
					</li>
				{/each}
			</ul>
		</div>
	{/if}
</div>
