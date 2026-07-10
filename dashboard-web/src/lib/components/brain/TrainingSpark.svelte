<script lang="ts">
	/**
	 * Inline SVG sparkline for a training loss curve: accent polyline over a
	 * faint area fill, min/max markers, and a numeric end label. Degrades
	 * gracefully when the series has fewer than two points.
	 */
	let {
		series,
		width = 120,
		height = 28
	}: {
		series: { step: number; total: number }[];
		width?: number;
		height?: number;
	} = $props();

	const PAD = 3;

	const fmtVal = (v: number): string =>
		Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toFixed(4);

	type Pt = { x: number; y: number; total: number };

	const pts = $derived.by((): Pt[] => {
		if (series.length === 0) return [];
		const xs = series.map((p) => p.step);
		const ys = series.map((p) => p.total);
		const x0 = Math.min(...xs);
		const x1 = Math.max(...xs);
		const y0 = Math.min(...ys);
		const y1 = Math.max(...ys);
		const sx = (v: number): number =>
			x1 === x0 ? width / 2 : PAD + ((v - x0) / (x1 - x0)) * (width - 2 * PAD);
		const sy = (v: number): number =>
			y1 === y0 ? height / 2 : height - PAD - ((v - y0) / (y1 - y0)) * (height - 2 * PAD);
		return series.map((p) => ({ x: sx(p.step), y: sy(p.total), total: p.total }));
	});

	const line = $derived(pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '));
	const area = $derived.by(() => {
		if (pts.length < 2) return '';
		const first = pts[0];
		const last = pts[pts.length - 1];
		const floor = height - PAD;
		return `M ${first.x.toFixed(1)} ${floor} L ${line.split(' ').join(' L ')} L ${last.x.toFixed(1)} ${floor} Z`;
	});

	const minPt = $derived.by((): Pt | null =>
		pts.length < 2 ? null : pts.reduce((m, p) => (p.total < m.total ? p : m))
	);
	const maxPt = $derived.by((): Pt | null =>
		pts.length < 2 ? null : pts.reduce((m, p) => (p.total > m.total ? p : m))
	);
	const last = $derived(series.length ? series[series.length - 1] : null);
</script>

<span class="inline-flex items-center gap-1.5">
	{#if pts.length >= 2}
		<svg
			{width}
			{height}
			viewBox={`0 0 ${width} ${height}`}
			role="img"
			aria-label={`Training loss sparkline, ${series.length} points, last ${last ? fmtVal(last.total) : '—'}`}
		>
			<path d={area} fill="var(--color-accent-soft)" stroke="none" />
			<polyline
				points={line}
				fill="none"
				stroke="var(--color-accent)"
				stroke-width="1.5"
				stroke-linejoin="round"
				stroke-linecap="round"
			/>
			{#if minPt}<circle cx={minPt.x} cy={minPt.y} r="2" fill="var(--color-long)" />{/if}
			{#if maxPt}<circle cx={maxPt.x} cy={maxPt.y} r="2" fill="var(--color-short)" />{/if}
		</svg>
	{:else if pts.length === 1}
		<svg {width} {height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Single training point">
			<circle cx={width / 2} cy={height / 2} r="2.5" fill="var(--color-accent)" />
		</svg>
	{/if}
	<span class="num text-[11px] text-ink-dim">{last ? fmtVal(last.total) : '—'}</span>
</span>
