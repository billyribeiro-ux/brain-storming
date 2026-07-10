<script lang="ts">
	/**
	 * Dependency-free SVG equity curve: accent line + gradient area fill,
	 * drawdown band (red-soft between running max and equity), min/max/final
	 * labels and four time ticks. If the bridge emits degenerate timestamps
	 * (all equal) the x-axis falls back to bar index so the shape still reads.
	 */
	import type { EquityPoint } from '$lib/api/schemas';
	import { fmtDate, fmtPx } from '$lib/utils/format';

	let { points, height = 200 }: { points: EquityPoint[]; height?: number } = $props();

	const uid = $props.id();
	const W = 800;
	const PAD = { top: 16, right: 10, bottom: 20, left: 10 } as const;

	type Model = {
		linePath: string;
		areaPath: string;
		ddPath: string;
		min: { x: number; y: number; v: number };
		max: { x: number; y: number; v: number };
		final: { x: number; y: number; v: number };
		ticks: { x: number; label: string }[];
	};

	const model = $derived.by((): Model | null => {
		const n = points.length;
		if (n < 2) return null;

		const H = height;
		const plotW = W - PAD.left - PAD.right;
		const plotH = H - PAD.top - PAD.bottom;

		let eqMin = Infinity;
		let eqMax = -Infinity;
		for (const p of points) {
			if (p.equity < eqMin) eqMin = p.equity;
			if (p.equity > eqMax) eqMax = p.equity;
		}
		const span = eqMax - eqMin || 1;

		const t0 = points[0].t;
		const tN = points[n - 1].t;
		const timeValid = tN > t0;

		const x = (i: number): number =>
			PAD.left + (timeValid ? ((points[i].t - t0) / (tN - t0)) * plotW : (i / (n - 1)) * plotW);
		const y = (v: number): number => PAD.top + (1 - (v - eqMin) / span) * plotH;

		let line = '';
		let dd = '';
		let ddBack = '';
		let runMax = -Infinity;
		let minI = 0;
		let maxI = 0;
		for (let i = 0; i < n; i++) {
			const v = points[i].equity;
			if (v < points[minI].equity) minI = i;
			if (v > points[maxI].equity) maxI = i;
			if (v > runMax) runMax = v;
			const xi = x(i);
			line += `${i === 0 ? 'M' : 'L'}${xi.toFixed(2)},${y(v).toFixed(2)}`;
			dd += `${i === 0 ? 'M' : 'L'}${xi.toFixed(2)},${y(runMax).toFixed(2)}`;
			ddBack = `L${xi.toFixed(2)},${y(v).toFixed(2)}` + ddBack;
		}
		const baseline = (PAD.top + plotH).toFixed(2);
		const areaPath = `${line}L${x(n - 1).toFixed(2)},${baseline}L${x(0).toFixed(2)},${baseline}Z`;
		const ddPath = `${dd}${ddBack}Z`;

		const ticks: Model['ticks'] = [];
		for (let k = 0; k < 4; k++) {
			const i = Math.round((k / 3) * (n - 1));
			ticks.push({ x: x(i), label: timeValid ? fmtDate(points[i].t) : `#${i}` });
		}

		return {
			linePath: line,
			areaPath,
			ddPath,
			min: { x: x(minI), y: y(points[minI].equity), v: points[minI].equity },
			max: { x: x(maxI), y: y(points[maxI].equity), v: points[maxI].equity },
			final: { x: x(n - 1), y: y(points[n - 1].equity), v: points[n - 1].equity },
			ticks
		};
	});

	/** Keep labels inside the viewBox horizontally. */
	const clampX = (x: number, margin = 60): number => Math.min(Math.max(x, margin), W - margin);
</script>

{#if !model}
	<div class="flex min-h-24 items-center justify-center text-xs text-ink-faint" style:height={`${height}px`}>
		No equity data.
	</div>
{:else}
	<svg
		viewBox={`0 0 ${W} ${height}`}
		class="block w-full"
		style:height={`${height}px`}
		role="img"
		aria-label={`Equity curve, min ${fmtPx(model.min.v)}, max ${fmtPx(model.max.v)}, final ${fmtPx(model.final.v)}`}
		preserveAspectRatio="none"
	>
		<defs>
			<linearGradient id={`eq-area-${uid}`} x1="0" y1="0" x2="0" y2="1">
				<stop offset="0" stop-color="var(--color-accent)" stop-opacity="0.22" />
				<stop offset="1" stop-color="var(--color-accent)" stop-opacity="0" />
			</linearGradient>
		</defs>

		<!-- drawdown band: red-soft between running max and equity -->
		<path d={model.ddPath} fill="var(--color-short-soft)" stroke="none" />
		<!-- area under equity -->
		<path d={model.areaPath} fill={`url(#eq-area-${uid})`} stroke="none" />
		<!-- equity line -->
		<path
			d={model.linePath}
			fill="none"
			stroke="var(--color-accent)"
			stroke-width="1.5"
			vector-effect="non-scaling-stroke"
		/>

		<!-- min / max / final labels -->
		<circle cx={model.max.x} cy={model.max.y} r="2.5" fill="var(--color-long)" />
		<text
			x={clampX(model.max.x)}
			y={Math.max(model.max.y - 5, 10)}
			text-anchor="middle"
			class="num"
			font-size="10"
			fill="var(--color-long)">{fmtPx(model.max.v)}</text
		>
		<circle cx={model.min.x} cy={model.min.y} r="2.5" fill="var(--color-short)" />
		<text
			x={clampX(model.min.x)}
			y={Math.min(model.min.y + 12, height - PAD.bottom - 2)}
			text-anchor="middle"
			class="num"
			font-size="10"
			fill="var(--color-short)">{fmtPx(model.min.v)}</text
		>
		<text
			x={W - PAD.right}
			y={Math.max(model.final.y - 6, 10)}
			text-anchor="end"
			class="num"
			font-size="10"
			fill="var(--color-ink)">{fmtPx(model.final.v)}</text
		>

		<!-- x-axis ticks -->
		{#each model.ticks as tick, i (i)}
			<text
				x={tick.x}
				y={height - 6}
				text-anchor={i === 0 ? 'start' : i === model.ticks.length - 1 ? 'end' : 'middle'}
				class="num"
				font-size="9"
				fill="var(--color-ink-faint)">{tick.label}</text
			>
		{/each}
	</svg>
{/if}
