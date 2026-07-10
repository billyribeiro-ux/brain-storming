<script lang="ts">
	/**
	 * World-model imagination fan: every rollout as a faint accent path,
	 * the p10–p90 band, and the p50 median. The y-axis is *latent
	 * divergence* — a proxy for how far imagined futures drift from the
	 * anchor state — and is labelled as such on the chart itself (honesty
	 * requirement: this is NOT a price forecast). Hover or use arrow keys
	 * to read p10/p50/p90 at any step ahead.
	 */
	import type { ImaginationFan } from '$lib/api/schemas';

	let { fan }: { fan: ImaginationFan } = $props();

	const W = 640;
	const H = 250;
	const PAD = { top: 12, right: 14, bottom: 34, left: 48 } as const;
	const plotW = W - PAD.left - PAD.right;
	const plotH = H - PAD.top - PAD.bottom;

	let hoverIdx = $state<number | null>(null);

	const len = $derived(Math.max(fan.quantiles.p50.length, 2));

	const domain = $derived.by(() => {
		let min = Infinity;
		let max = -Infinity;
		const scan = (arr: number[]): void => {
			for (const v of arr) {
				if (v < min) min = v;
				if (v > max) max = v;
			}
		};
		for (const p of fan.paths) scan(p);
		scan(fan.quantiles.p10);
		scan(fan.quantiles.p50);
		scan(fan.quantiles.p90);
		if (!Number.isFinite(min) || !Number.isFinite(max)) {
			min = 0;
			max = 1;
		}
		if (max - min === 0) max = min + 1;
		return { min, max };
	});

	// Plain functions over reactive state: calls from the template re-track
	// `len`/`domain` automatically, no $derived wrapper needed.
	const x = (i: number): number => PAD.left + (i / (len - 1)) * plotW;
	const y = (v: number): number =>
		PAD.top + (1 - (v - domain.min) / (domain.max - domain.min)) * plotH;

	const linePath = (arr: number[]): string =>
		arr.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join('');

	const bandPath = $derived.by(() => {
		const { p10, p90 } = fan.quantiles;
		if (p10.length === 0 || p90.length === 0) return '';
		const top = p90.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)},${y(v).toFixed(2)}`);
		const bottom = [...p10]
			.reverse()
			.map((v, i) => `L${x(p10.length - 1 - i).toFixed(2)},${y(v).toFixed(2)}`);
		return `${top.join('')}${bottom.join('')}Z`;
	});

	const xTicks = $derived.by(() => {
		const ticks: number[] = [];
		const step = len - 1 <= 4 ? 1 : Math.ceil((len - 1) / 4);
		for (let i = 0; i < len; i += step) ticks.push(i);
		if (ticks[ticks.length - 1] !== len - 1) ticks.push(len - 1);
		return ticks;
	});

	function idxFromClientX(clientX: number, svg: SVGSVGElement): number {
		const rect = svg.getBoundingClientRect();
		const px = ((clientX - rect.left) / rect.width) * W;
		const i = Math.round(((px - PAD.left) / plotW) * (len - 1));
		return Math.min(len - 1, Math.max(0, i));
	}

	function onKeydown(e: KeyboardEvent): void {
		if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
			e.preventDefault();
			const delta = e.key === 'ArrowRight' ? 1 : -1;
			hoverIdx = Math.min(len - 1, Math.max(0, (hoverIdx ?? 0) + delta));
		} else if (e.key === 'Escape') {
			hoverIdx = null;
		}
	}

	const at = (arr: number[], i: number): number | null => (i < arr.length ? arr[i] : null);
	const fmtVal = (v: number | null): string => (v === null ? '—' : v.toFixed(3));
</script>

<div class="flex flex-col gap-1.5">
	<!-- The chart doubles as its own value cursor: exposing it as a slider
	     (arrow keys move the inspected bar) is the closest ARIA idiom. -->
	<svg
		viewBox={`0 0 ${W} ${H}`}
		class="block w-full rounded focus-visible:outline focus-visible:outline-1 focus-visible:outline-accent"
		role="slider"
		aria-roledescription="fan chart"
		aria-label={`Imagination fan for ${fan.ticker}: ${fan.paths.length} rollouts over ${fan.horizon} bars of latent divergence — a proxy, not a price forecast. Use arrow keys to inspect values.`}
		aria-valuemin={0}
		aria-valuemax={len - 1}
		aria-valuenow={hoverIdx ?? 0}
		aria-valuetext={hoverIdx === null
			? 'no bar inspected'
			: `bar ${hoverIdx}: p10 ${fmtVal(at(fan.quantiles.p10, hoverIdx))}, p50 ${fmtVal(at(fan.quantiles.p50, hoverIdx))}, p90 ${fmtVal(at(fan.quantiles.p90, hoverIdx))}`}
		tabindex="0"
		onpointermove={(e) => (hoverIdx = idxFromClientX(e.clientX, e.currentTarget))}
		onpointerleave={() => (hoverIdx = null)}
		onkeydown={onKeydown}
	>
		<!-- p10–p90 band -->
		<path d={bandPath} fill="var(--color-accent-soft)" stroke="none" />

		<!-- individual rollouts -->
		{#each fan.paths as p, i (i)}
			<path d={linePath(p)} fill="none" stroke="var(--color-accent)" stroke-opacity="0.12" stroke-width="1" />
		{/each}

		<!-- median -->
		<path d={linePath(fan.quantiles.p50)} fill="none" stroke="var(--color-accent)" stroke-width="2" />

		<!-- hover rule -->
		{#if hoverIdx !== null}
			<line
				x1={x(hoverIdx)}
				x2={x(hoverIdx)}
				y1={PAD.top}
				y2={PAD.top + plotH}
				stroke="var(--color-hairline-strong)"
				stroke-dasharray="3 3"
			/>
			{#if at(fan.quantiles.p50, hoverIdx) !== null}
				<circle
					cx={x(hoverIdx)}
					cy={y(fan.quantiles.p50[hoverIdx])}
					r="3"
					fill="var(--color-accent)"
				/>
			{/if}
		{/if}

		<!-- axes -->
		<line
			x1={PAD.left}
			x2={PAD.left}
			y1={PAD.top}
			y2={PAD.top + plotH}
			stroke="var(--color-hairline)"
		/>
		<line
			x1={PAD.left}
			x2={W - PAD.right}
			y1={PAD.top + plotH}
			y2={PAD.top + plotH}
			stroke="var(--color-hairline)"
		/>
		{#each xTicks as i (i)}
			<text x={x(i)} y={H - 20} text-anchor="middle" class="num" font-size="9" fill="var(--color-ink-faint)"
				>+{i}</text
			>
		{/each}
		<text x={(PAD.left + W - PAD.right) / 2} y={H - 6} text-anchor="middle" font-size="10" fill="var(--color-ink-dim)"
			>bars ahead</text
		>
		<!-- honesty requirement: the y-axis is a latent proxy, never a price -->
		<text
			transform={`rotate(-90 12 ${PAD.top + plotH / 2})`}
			x="12"
			y={PAD.top + plotH / 2}
			text-anchor="middle"
			font-size="10"
			fill="var(--color-warn)">latent divergence (proxy — NOT a price forecast)</text
		>
		<text x={W - PAD.right} y={PAD.top + 2} text-anchor="end" class="num" font-size="9" fill="var(--color-ink-faint)"
			>{domain.max.toFixed(2)}</text
		>
		<text
			x={W - PAD.right}
			y={PAD.top + plotH - 3}
			text-anchor="end"
			class="num"
			font-size="9"
			fill="var(--color-ink-faint)">{domain.min.toFixed(2)}</text
		>
	</svg>

	<!-- readout -->
	<p class="num min-h-4 text-[11px] text-ink-dim" aria-live="polite">
		{#if hoverIdx !== null}
			bar +{hoverIdx} · p10 <span class="text-ink">{fmtVal(at(fan.quantiles.p10, hoverIdx))}</span>
			· p50 <span class="text-accent">{fmtVal(at(fan.quantiles.p50, hoverIdx))}</span>
			· p90 <span class="text-ink">{fmtVal(at(fan.quantiles.p90, hoverIdx))}</span>
		{:else}
			{fan.paths.length} rollouts · horizon {fan.horizon} bars — hover or focus + arrow keys to inspect
		{/if}
	</p>
</div>
