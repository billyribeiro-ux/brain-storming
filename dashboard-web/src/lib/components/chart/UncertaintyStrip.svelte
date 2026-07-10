<script lang="ts">
	/**
	 * "BRAIN ATTENTION" heat strip — a 28px canvas under the price chart,
	 * one column per session bar.
	 *
	 * Honesty note: this visualizes the model's per-bar UNCERTAINTY — a blue
	 * wash scaled by epistemic uncertainty ("I haven't seen enough like
	 * this") with a red-hot overlay scaled by the anomaly score ("this looks
	 * wrong"). It answers "where is the brain unsure or surprised", NOT
	 * transformer attention weights — no attention matrices are involved.
	 *
	 * Timestamps are naive-ET epoch seconds, formatted with UTC getters.
	 */
	import type { UncertaintyPoint } from '$lib/api/schemas';

	let { uncertainty, width }: { uncertainty: UncertaintyPoint[]; width?: number } = $props();

	const STRIP_H = 28;

	let canvas = $state<HTMLCanvasElement>();
	let measuredWidth = $state(0);
	const stripWidth = $derived(width ?? measuredWidth);

	const pad2 = (n: number): string => String(n).padStart(2, '0');
	const hhmm = (ts: number): string => {
		const d = new Date(ts * 1000);
		return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`;
	};

	const IDLE_TITLE = 'Model uncertainty per bar — hover for values';
	let title = $state(IDLE_TITLE);

	// Canvas compositing needs numeric rgb; read the token, fall back to its
	// documented value so the strip never drifts off-theme.
	const tokenRgb = (name: string, fallback: [number, number, number]): [number, number, number] => {
		if (typeof window === 'undefined') return fallback;
		const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
		const m = /^#([0-9a-f]{6})$/i.exec(v);
		if (!m) return fallback;
		const n = parseInt(m[1], 16);
		return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
	};

	// Session-relative normalization so the strip always shows contrast.
	const norm = $derived.by(() => {
		let eMin = Infinity;
		let eMax = -Infinity;
		let aMax = 0;
		for (const p of uncertainty) {
			if (p.epistemic < eMin) eMin = p.epistemic;
			if (p.epistemic > eMax) eMax = p.epistemic;
			if (p.anomaly > aMax) aMax = p.anomaly;
		}
		return { eMin, eMax, aMax };
	});

	$effect(() => {
		const el = canvas;
		const w = stripWidth;
		const points = uncertainty;
		const { eMin, eMax, aMax } = norm;
		if (!el || w <= 0) return;

		const dpr = window.devicePixelRatio || 1;
		el.width = Math.round(w * dpr);
		el.height = Math.round(STRIP_H * dpr);
		const ctx = el.getContext('2d');
		if (!ctx) return;
		ctx.scale(dpr, dpr);
		ctx.clearRect(0, 0, w, STRIP_H);

		if (points.length === 0) return;

		const [br, bg, bb] = tokenRgb('--color-accent', [79, 140, 255]); // epistemic = blue
		const [rr, rg, rb] = tokenRgb('--color-short', [255, 93, 108]); // anomaly = red-hot
		const eRange = eMax - eMin;
		const colW = w / points.length;

		for (let i = 0; i < points.length; i++) {
			const p = points[i];
			const x = i * colW;
			// Epistemic underlay: blue alpha, session-normalized.
			const eNorm = eRange > 0 ? (p.epistemic - eMin) / eRange : 0.5;
			ctx.fillStyle = `rgba(${br}, ${bg}, ${bb}, ${(0.06 + 0.5 * eNorm).toFixed(3)})`;
			ctx.fillRect(x, 0, colW + 0.5, STRIP_H);
			// Anomaly overlay: red alpha on top, only where anomaly fires.
			if (aMax > 0 && p.anomaly > 0) {
				const aNorm = p.anomaly / aMax;
				ctx.fillStyle = `rgba(${rr}, ${rg}, ${rb}, ${(0.9 * aNorm).toFixed(3)})`;
				ctx.fillRect(x, 0, colW + 0.5, STRIP_H);
			}
		}
	});

	const onMove = (e: MouseEvent): void => {
		if (uncertainty.length === 0 || stripWidth <= 0) return;
		const i = Math.min(
			uncertainty.length - 1,
			Math.max(0, Math.floor((e.offsetX / stripWidth) * uncertainty.length))
		);
		const p = uncertainty[i];
		title = `${hhmm(p.t)} ET — epistemic ${p.epistemic.toFixed(3)} · anomaly ${p.anomaly.toFixed(3)} · aleatoric ${p.aleatoric.toFixed(3)}`;
	};
	const onLeave = (): void => {
		title = IDLE_TITLE;
	};
</script>

<div class="flex w-full items-center gap-2">
	<span class="w-16 shrink-0 text-[10px] leading-none font-medium tracking-widest text-ink-faint">
		BRAIN ATTENTION
	</span>
	<div
		class="relative min-w-0 flex-1"
		style={`height:${STRIP_H}px`}
		bind:clientWidth={measuredWidth}
		role="img"
		aria-label="Per-bar model uncertainty heat strip: blue is epistemic uncertainty, red is anomaly score"
	>
		<canvas
			bind:this={canvas}
			class="absolute inset-0 h-full w-full rounded-sm border border-hairline"
			{title}
			onmousemove={onMove}
			onmouseleave={onLeave}
		></canvas>
	</div>
</div>
