<script lang="ts">
	/**
	 * Force-directed view of the fitted causal graph. A tiny bespoke
	 * simulation (pairwise repulsion + edge springs + cluster/center pull)
	 * runs ~150 iterations exactly once per snapshot (a $derived over the
	 * snapshot prop), then the layout is rendered statically — no continuous
	 * animation. Nodes are
	 * hue-grouped by ticker prefix (MKT.* gets the amber/warn family), edges
	 * are curved quadratics with arrowheads: green positive / red negative,
	 * width ∝ |weight|·confidence, opacity ∝ bootstrap confidence.
	 * Wheel zooms the viewBox, drag pans, hover/focus highlights a node's
	 * in/out edges, Enter or click selects.
	 */
	import { on } from 'svelte/events';
	import type { CausalEdge, CausalSnapshot } from '$lib/api/schemas';

	let {
		snapshot,
		focusTicker = '',
		onSelect
	}: {
		snapshot: CausalSnapshot;
		focusTicker?: string;
		onSelect?: (node: string) => void;
	} = $props();

	/* ---------------- constants & helpers ---------------- */

	const W = 1000;
	const H = 640;
	const EDGE_CAP = 150;
	const NODE_R = 7;

	type Pt = { x: number; y: number };

	const prefixOf = (n: string): string => {
		const i = n.indexOf('.');
		return i < 0 ? n : n.slice(0, i);
	};
	const suffixOf = (n: string): string => {
		const i = n.indexOf('.');
		return i < 0 ? '' : n.slice(i + 1).replace(/_/g, ' ');
	};
	const strength = (e: CausalEdge): number => Math.abs(e.weight) * e.confidence;
	const clampNum = (v: number, lo: number, hi: number): number =>
		hi < lo ? (lo + hi) / 2 : Math.min(hi, Math.max(lo, v));

	/* ---------------- ticker hue families ---------------- */

	// Spread hues avoiding the amber band, which is reserved for MKT.*
	// (matching the warn token family).
	const HUES = [212, 158, 268, 322, 190, 246, 132, 292, 174, 226, 350, 98];

	const colorByPrefix = $derived.by((): Record<string, string> => {
		const rec: Record<string, string> = {};
		let i = 0;
		for (const n of snapshot.nodes) {
			const p = prefixOf(n);
			if (p in rec) continue;
			rec[p] = p === 'MKT' ? 'hsl(38 92% 60%)' : `hsl(${HUES[i++ % HUES.length]} 68% 62%)`;
		}
		return rec;
	});
	const colorOf = (node: string): string => colorByPrefix[prefixOf(node)] ?? 'hsl(212 68% 62%)';

	/* ---------------- force layout (precomputed, static render) -------- */

	function simulate(nodes: string[], edges: CausalEdge[]): Record<string, Pt> {
		const n = nodes.length;
		if (n === 0) return {};
		const idx = new Map(nodes.map((node, i) => [node, i]));
		const xs = new Float64Array(n);
		const ys = new Float64Array(n);
		const ax = new Float64Array(n); // per-node cluster anchor
		const ay = new Float64Array(n);

		// Cluster anchors: one per ticker prefix, on an ellipse. Deterministic
		// golden-angle jitter seeds members around their anchor (no Math.random
		// so the layout is stable across renders).
		const prefixes = [...new Set(nodes.map(prefixOf))];
		const anchor: Record<string, Pt> = {};
		prefixes.forEach((p, i) => {
			const a = (i / prefixes.length) * 2 * Math.PI - Math.PI / 2;
			anchor[p] = { x: W / 2 + Math.cos(a) * W * 0.33, y: H / 2 + Math.sin(a) * H * 0.33 };
		});
		nodes.forEach((node, i) => {
			const a = anchor[prefixOf(node)] ?? { x: W / 2, y: H / 2 };
			const t = i * 2.399963;
			ax[i] = a.x;
			ay[i] = a.y;
			xs[i] = a.x + Math.cos(t) * 26;
			ys[i] = a.y + Math.sin(t) * 26;
		});

		// Springs: one per linked pair (dedup, self-loops excluded).
		const springs: { a: number; b: number }[] = [];
		const seen: Record<string, boolean> = {};
		for (const e of edges) {
			if (e.src === e.dst) continue;
			const a = idx.get(e.src);
			const b = idx.get(e.dst);
			if (a === undefined || b === undefined) continue;
			const key = a < b ? `${a}|${b}` : `${b}|${a}`;
			if (seen[key]) continue;
			seen[key] = true;
			springs.push({ a, b });
		}

		const ITER = 150;
		const fx = new Float64Array(n);
		const fy = new Float64Array(n);
		for (let it = 0; it < ITER; it++) {
			const cool = 1 - it / ITER;
			fx.fill(0);
			fy.fill(0);
			// pairwise repulsion
			for (let i = 0; i < n; i++) {
				for (let j = i + 1; j < n; j++) {
					let dx = xs[i] - xs[j];
					let dy = ys[i] - ys[j];
					let d2 = dx * dx + dy * dy;
					if (d2 < 1) {
						dx = (((i * 7919 + j) % 13) - 6) || 1;
						dy = ((j * 104729 + i) % 11) - 5;
						d2 = dx * dx + dy * dy;
					}
					const d = Math.sqrt(d2);
					const f = 2600 / d2;
					fx[i] += (dx / d) * f;
					fy[i] += (dy / d) * f;
					fx[j] -= (dx / d) * f;
					fy[j] -= (dy / d) * f;
				}
			}
			// springs toward linked nodes
			for (const s of springs) {
				const dx = xs[s.b] - xs[s.a];
				const dy = ys[s.b] - ys[s.a];
				const d = Math.hypot(dx, dy) || 1;
				const f = 0.03 * (d - 120);
				fx[s.a] += (dx / d) * f;
				fy[s.a] += (dy / d) * f;
				fx[s.b] -= (dx / d) * f;
				fy[s.b] -= (dy / d) * f;
			}
			// centering + gentle pull toward the cluster anchor
			for (let i = 0; i < n; i++) {
				fx[i] += (W / 2 - xs[i]) * 0.004 + (ax[i] - xs[i]) * 0.012;
				fy[i] += (H / 2 - ys[i]) * 0.004 + (ay[i] - ys[i]) * 0.012;
			}
			// integrate with a cooling displacement cap
			const lim = 14 * cool + 0.5;
			for (let i = 0; i < n; i++) {
				const m = Math.hypot(fx[i], fy[i]);
				const k = m > lim ? lim / m : 1;
				xs[i] = clampNum(xs[i] + fx[i] * k, 46, W - 46);
				ys[i] = clampNum(ys[i] + fy[i] * k, 36, H - 36);
			}
		}
		const out: Record<string, Pt> = {};
		nodes.forEach((node, i) => {
			out[node] = { x: xs[i], y: ys[i] };
		});
		return out;
	}

	// Recomputed exactly once per snapshot; the render itself is static.
	const positions = $derived.by(() => simulate(snapshot.nodes, snapshot.edges));

	/* ---------------- drawn geometry ---------------- */

	type DrawnEdge = {
		e: CausalEdge;
		id: string;
		path: string;
		width: number;
		opacity: number;
		positive: boolean;
	};

	const topEdges = $derived(
		[...snapshot.edges].sort((a, b) => strength(b) - strength(a)).slice(0, EDGE_CAP)
	);

	const drawn = $derived.by((): DrawnEdge[] => {
		const pos = positions;
		const max = topEdges.reduce((m, e) => Math.max(m, strength(e)), 0) || 1;
		const trim = (from: Pt, to: Pt, r: number): Pt => {
			const vx = to.x - from.x;
			const vy = to.y - from.y;
			const l = Math.hypot(vx, vy) || 1;
			return { x: from.x + (vx / l) * r, y: from.y + (vy / l) * r };
		};
		const out: DrawnEdge[] = [];
		topEdges.forEach((e, i) => {
			const p1 = pos[e.src];
			const p2 = pos[e.dst];
			if (!p1 || !p2) return;
			let path: string;
			if (e.src === e.dst) {
				// self-loop (lagged autocorrelation)
				path = `M ${(p1.x - NODE_R).toFixed(1)} ${(p1.y - 4).toFixed(1)} C ${(p1.x - 36).toFixed(1)} ${(p1.y - 30).toFixed(1)} ${(p1.x + 30).toFixed(1)} ${(p1.y - 42).toFixed(1)} ${(p1.x + 6).toFixed(1)} ${(p1.y - NODE_R - 4).toFixed(1)}`;
			} else {
				const dx = p2.x - p1.x;
				const dy = p2.y - p1.y;
				const d = Math.hypot(dx, dy) || 1;
				// alternate curvature side by lag so parallel edges don't overlap
				const off = Math.min(0.18 * d, 26) * (1 + 0.35 * e.lag) * (e.lag % 2 === 0 ? 1 : -1);
				const mid: Pt = { x: (p1.x + p2.x) / 2 - (dy / d) * off, y: (p1.y + p2.y) / 2 + (dx / d) * off };
				const start = trim(p1, mid, NODE_R + 1);
				const end = trim(p2, mid, NODE_R + 5);
				path = `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} Q ${mid.x.toFixed(1)} ${mid.y.toFixed(1)} ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
			}
			out.push({
				e,
				id: `${e.src}→${e.dst}@${e.lag}#${i}`,
				path,
				width: 0.5 + 3 * (strength(e) / max),
				opacity: 0.15 + 0.85 * e.confidence,
				positive: e.weight >= 0
			});
		});
		return out;
	});

	type NodeView = { name: string; x: number; y: number; color: string; label: string };

	const nodeViews = $derived.by((): NodeView[] => {
		const out: NodeView[] = [];
		for (const name of snapshot.nodes) {
			const p = positions[name];
			if (!p) continue;
			out.push({ name, x: p.x, y: p.y, color: colorOf(name), label: suffixOf(name) || name });
		}
		return out;
	});

	/* ---------------- hover / focus / selection ---------------- */

	let hovered = $state<string | null>(null);

	const neighbors = $derived.by((): Set<string> => {
		const h = hovered;
		if (!h) return new Set<string>();
		return new Set(
			drawn.flatMap((d) => {
				const linked: string[] = [];
				if (d.e.src === h) linked.push(d.e.dst);
				if (d.e.dst === h) linked.push(d.e.src);
				return linked;
			})
		);
	});

	const edgeOpacity = (d: DrawnEdge): number => {
		if (!hovered) return d.opacity;
		return d.e.src === hovered || d.e.dst === hovered ? Math.max(d.opacity, 0.9) : 0.05;
	};
	const nodeOpacity = (name: string): number =>
		!hovered || name === hovered || neighbors.has(name) ? 1 : 0.25;
	const isFocused = (name: string): boolean =>
		focusTicker !== '' && prefixOf(name) === focusTicker;

	const tipDrivers = $derived.by((): CausalEdge[] => {
		if (!hovered) return [];
		const h = hovered;
		return snapshot.edges
			.filter((e) => e.dst === h)
			.sort((a, b) => strength(b) - strength(a))
			.slice(0, 3);
	});

	function select(name: string): void {
		onSelect?.(name);
	}

	/* ---------------- zoom / pan (viewBox) ---------------- */

	// Plain (non-reactive) element ref: only read inside event handlers.
	let svgEl: SVGSVGElement | null = null;
	let cw = $state(0);
	let ch = $state(0);
	let vb = $state({ x: 0, y: 0, w: W, h: H });
	let panning = $state(false);
	let panStart: { cx: number; cy: number; vx: number; vy: number } | null = null;

	function clampVb(x: number, y: number, w: number, h: number): { x: number; y: number; w: number; h: number } {
		const mx = W * 0.3;
		const my = H * 0.3;
		return { x: clampNum(x, -mx, W + mx - w), y: clampNum(y, -my, H + my - h), w, h };
	}

	function viewScale(): number {
		if (!svgEl) return 1;
		const rect = svgEl.getBoundingClientRect();
		return Math.min(rect.width / vb.w, rect.height / vb.h) || 1;
	}

	function clientToWorld(clientX: number, clientY: number): Pt {
		if (!svgEl) return { x: W / 2, y: H / 2 };
		const rect = svgEl.getBoundingClientRect();
		const scale = viewScale();
		const ox = (rect.width - vb.w * scale) / 2;
		const oy = (rect.height - vb.h * scale) / 2;
		return {
			x: (clientX - rect.left - ox) / scale + vb.x,
			y: (clientY - rect.top - oy) / scale + vb.y
		};
	}

	function handleWheel(ev: Event): void {
		if (!(ev instanceof WheelEvent)) return;
		ev.preventDefault();
		const factor = ev.deltaY > 0 ? 1.15 : 1 / 1.15;
		const w = clampNum(vb.w * factor, W / 10, W * 1.6);
		const h = w * (H / W);
		const pt = clientToWorld(ev.clientX, ev.clientY);
		const kx = (pt.x - vb.x) / vb.w;
		const ky = (pt.y - vb.y) / vb.h;
		vb = clampVb(pt.x - kx * w, pt.y - ky * h, w, h);
	}

	// Svelte 5 registers `onwheel` passively; zoom needs preventDefault, so
	// attach a non-passive listener via an attachment (also captures the ref).
	function wheelZoom(el: SVGSVGElement): () => void {
		svgEl = el;
		const off = on(el, 'wheel', handleWheel, { passive: false });
		return () => {
			off();
			svgEl = null;
		};
	}

	function onPointerDown(ev: PointerEvent): void {
		if (ev.button !== 0 || !svgEl) return;
		panStart = { cx: ev.clientX, cy: ev.clientY, vx: vb.x, vy: vb.y };
		panning = true;
		svgEl.setPointerCapture(ev.pointerId);
	}
	function onPointerMove(ev: PointerEvent): void {
		if (!panStart) return;
		const scale = viewScale();
		vb = clampVb(
			panStart.vx - (ev.clientX - panStart.cx) / scale,
			panStart.vy - (ev.clientY - panStart.cy) / scale,
			vb.w,
			vb.h
		);
	}
	function onPointerUp(): void {
		panStart = null;
		panning = false;
	}
	function resetView(): void {
		vb = { x: 0, y: 0, w: W, h: H };
	}

	/* ---------------- tooltip & a11y summary ---------------- */

	const tipPos = $derived.by((): Pt | null => {
		if (!hovered || !cw || !ch) return null;
		const p = positions[hovered];
		if (!p) return null;
		const scale = Math.min(cw / vb.w, ch / vb.h) || 1;
		const ox = (cw - vb.w * scale) / 2;
		const oy = (ch - vb.h * scale) / 2;
		return {
			x: clampNum((p.x - vb.x) * scale + ox, 90, Math.max(90, cw - 90)),
			y: clampNum((p.y - vb.y) * scale + oy, 40, Math.max(40, ch - 10))
		};
	});

	const summary = $derived(
		topEdges
			.slice(0, 5)
			.map(
				(e) =>
					`${e.src} to ${e.dst} at lag ${e.lag}, weight ${e.weight.toFixed(2)}, confidence ${e.confidence.toFixed(2)}`
			)
			.join('; ')
	);
</script>

<div class="relative h-full min-h-96 w-full" bind:clientWidth={cw} bind:clientHeight={ch}>
	<p class="sr-only">
		Causal graph of {snapshot.nodes.length} nodes and {snapshot.edges.length} edges. Strongest
		edges: {summary}.
	</p>

	<svg
		{@attach wheelZoom}
		viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
		preserveAspectRatio="xMidYMid meet"
		class={`h-full w-full touch-none select-none ${panning ? 'cursor-grabbing' : 'cursor-grab'}`}
		role="img"
		aria-label={`Interactive causal graph, ${snapshot.nodes.length} nodes. Tab to a node and press Enter to select it.`}
		onpointerdown={onPointerDown}
		onpointermove={onPointerMove}
		onpointerup={onPointerUp}
		onpointercancel={onPointerUp}
		ondblclick={resetView}
	>
		<defs>
			<marker
				id="arrow-pos"
				viewBox="0 0 8 8"
				refX="7"
				refY="4"
				markerWidth="5"
				markerHeight="5"
				orient="auto-start-reverse"
			>
				<path d="M0,0.5 L8,4 L0,7.5 Z" fill="var(--color-long)" />
			</marker>
			<marker
				id="arrow-neg"
				viewBox="0 0 8 8"
				refX="7"
				refY="4"
				markerWidth="5"
				markerHeight="5"
				orient="auto-start-reverse"
			>
				<path d="M0,0.5 L8,4 L0,7.5 Z" fill="var(--color-short)" />
			</marker>
		</defs>

		<!-- edges -->
		<g fill="none">
			{#each drawn as d (d.id)}
				<path
					class="edge"
					d={d.path}
					stroke={d.positive ? 'var(--color-long)' : 'var(--color-short)'}
					stroke-width={d.width}
					opacity={edgeOpacity(d)}
					marker-end={d.positive ? 'url(#arrow-pos)' : 'url(#arrow-neg)'}
				/>
			{/each}
		</g>

		<!-- nodes -->
		{#each nodeViews as nv (nv.name)}
			<g class="node" opacity={nodeOpacity(nv.name)}>
				{#if isFocused(nv.name)}
					<circle
						cx={nv.x}
						cy={nv.y}
						r={NODE_R + 3.5}
						fill="none"
						stroke="var(--color-accent)"
						stroke-width="1.5"
						opacity="0.9"
					/>
				{/if}
				<circle
					cx={nv.x}
					cy={nv.y}
					r={NODE_R}
					fill={nv.color}
					fill-opacity="0.9"
					stroke="rgba(4, 6, 12, 0.85)"
					stroke-width="1.5"
					role="button"
					tabindex="0"
					aria-label={`${nv.name} — press Enter to select`}
					class="cursor-pointer focus:outline-none"
					onpointerdown={(ev) => ev.stopPropagation()}
					onpointerenter={() => (hovered = nv.name)}
					onpointerleave={() => (hovered = null)}
					onfocus={() => (hovered = nv.name)}
					onblur={() => (hovered = null)}
					onclick={() => select(nv.name)}
					onkeydown={(ev) => {
						if (ev.key === 'Enter' || ev.key === ' ') {
							ev.preventDefault();
							select(nv.name);
						}
					}}
				/>
				{#if hovered === nv.name}
					<circle cx={nv.x} cy={nv.y} r={NODE_R + 2} fill="none" stroke={nv.color} stroke-width="1" />
				{/if}
				<text
					x={nv.x}
					y={nv.y + NODE_R + 10}
					text-anchor="middle"
					font-size="8.5"
					class="pointer-events-none font-mono"
					fill={prefixOf(nv.name) === nv.label ? 'var(--color-ink-dim)' : 'var(--color-ink-faint)'}
				>
					{prefixOf(nv.name)} {nv.label}
				</text>
			</g>
		{/each}
	</svg>

	<!-- tooltip -->
	{#if hovered && tipPos}
		<div
			class="glass pointer-events-none absolute z-20 w-44 -translate-x-1/2 -translate-y-full px-2 py-1.5"
			style={`left:${tipPos.x.toFixed(0)}px; top:${(tipPos.y - 12).toFixed(0)}px`}
			role="tooltip"
		>
			<p class="truncate font-mono text-[11px] font-semibold text-ink">{hovered}</p>
			{#if tipDrivers.length > 0}
				<p class="mt-1 text-[9px] tracking-widest text-ink-faint uppercase">top in-drivers</p>
				<ul class="mt-0.5 flex flex-col gap-0.5">
					{#each tipDrivers as e (`${e.src}@${e.lag}`)}
						<li class="flex items-baseline justify-between gap-2 text-[10px]">
							<span class="truncate font-mono text-ink-dim">{e.src} <span class="text-ink-faint">L{e.lag}</span></span>
							<span class={`num shrink-0 ${e.weight >= 0 ? 'text-long' : 'text-short'}`}>
								{e.weight >= 0 ? '+' : '−'}{Math.abs(e.weight).toFixed(2)}
							</span>
						</li>
					{/each}
				</ul>
			{:else}
				<p class="mt-1 text-[10px] text-ink-faint">no inbound edges</p>
			{/if}
		</div>
	{/if}

	<p class="pointer-events-none absolute right-2 bottom-1 text-[9px] text-ink-faint">
		scroll to zoom · drag to pan · double-click to reset
	</p>
</div>

<style>
	.edge,
	.node {
		transition:
			opacity 140ms ease;
	}
	@media (prefers-reduced-motion: reduce) {
		.edge,
		.node {
			transition: none;
		}
	}
</style>
