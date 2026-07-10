<script lang="ts">
	/**
	 * Main price chart — lightweight-charts v5.
	 *
	 * Candles + volume histogram, signal/trade markers, focused-signal
	 * entry/stop/target price lines, and live-bar streaming via
	 * `series.update()`. All chart work happens inside `$effect` (client
	 * only), so the component is SSR-safe; colors are read from the design
	 * tokens in app.css at mount so the canvas stays on-theme.
	 *
	 * Timestamps: bar.t is naive-ET wall-clock epoch seconds (the epoch IS
	 * the ET wall time), so every label is formatted with UTC getters —
	 * never the browser's timezone.
	 */
	import {
		CandlestickSeries,
		HistogramSeries,
		LineStyle,
		createChart,
		createSeriesMarkers,
		type CandlestickData,
		type HistogramData,
		type IChartApi,
		type IPriceLine,
		type ISeriesApi,
		type ISeriesMarkersPluginApi,
		type MouseEventParams,
		type SeriesMarker,
		type Time,
		type UTCTimestamp
	} from 'lightweight-charts';
	import type { Bar, Signal, Trade, UncertaintyPoint } from '$lib/api/schemas';
	import { fmtPx } from '$lib/utils/format';

	let {
		bars,
		signals = [],
		trades = [],
		uncertainty = [],
		focusedSignal = null,
		liveBar = null,
		height = 0
	}: {
		bars: Bar[];
		signals?: Signal[];
		trades?: Trade[];
		uncertainty?: UncertaintyPoint[];
		focusedSignal?: Signal | null;
		liveBar?: Bar | null;
		/** Fixed pixel height; 0/omitted = fill the parent element. */
		height?: number;
	} = $props();

	const MAX_MARKERS = 300;
	/** Live bars further than this from the loaded session are ignored. */
	const SESSION_WINDOW_S = 86_400;

	// ---- token helpers (canvas needs concrete colors; read them from the
	// ---- design system so the chart never hardcodes an off-theme value) --
	const cssVar = (name: string, fallback: string): string => {
		const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
		return v || fallback;
	};
	const withAlpha = (hex: string, alpha: number): string => {
		const m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
		if (!m) return hex;
		const n = parseInt(m[1], 16);
		return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
	};

	interface Palette {
		up: string;
		down: string;
		upSoft: string;
		downSoft: string;
		accent: string;
		inkDim: string;
		hairline: string;
	}

	interface Handles {
		chart: IChartApi;
		candles: ISeriesApi<'Candlestick'>;
		volume: ISeriesApi<'Histogram'>;
		markers: ISeriesMarkersPluginApi<Time>;
		pal: Palette;
	}

	const pad2 = (n: number): string => String(n).padStart(2, '0');
	/** naive-ET epoch seconds -> "HH:mm" via UTC getters (ET wall time). */
	const hhmm = (ts: number): string => {
		const d = new Date(ts * 1000);
		return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`;
	};
	const timeLabel = (t: Time): string => (typeof t === 'number' ? hhmm(t) : String(t));

	let container = $state<HTMLDivElement>();
	let handles = $state.raw<Handles | null>(null);
	/** Time of the newest bar in the series — ordering guard for update(). */
	let lastBarTime = 0;

	interface HoverInfo {
		t: number;
		o: number;
		h: number;
		l: number;
		c: number;
		epistemic: number | null;
		anomaly: number | null;
	}
	let hover = $state<HoverInfo | null>(null);

	const uncByT = $derived(new Map(uncertainty.map((p) => [p.t, p])));

	// ---- chart lifecycle -------------------------------------------------
	$effect(() => {
		const el = container;
		if (!el) return;

		const pal: Palette = {
			up: cssVar('--color-long', '#2fd08c'),
			down: cssVar('--color-short', '#ff5d6c'),
			upSoft: withAlpha(cssVar('--color-long', '#2fd08c'), 0.35),
			downSoft: withAlpha(cssVar('--color-short', '#ff5d6c'), 0.35),
			accent: cssVar('--color-accent', '#4f8cff'),
			inkDim: cssVar('--color-ink-dim', '#8b93ad'),
			hairline: cssVar('--color-hairline', 'rgba(148, 163, 216, 0.10)')
		};

		const chart = createChart(el, {
			width: el.clientWidth,
			height: el.clientHeight,
			layout: {
				background: { color: 'transparent' },
				textColor: pal.inkDim,
				fontFamily: cssVar('--font-mono', 'ui-monospace, monospace'),
				fontSize: 10,
				attributionLogo: false
			},
			grid: {
				vertLines: { color: pal.hairline },
				horzLines: { color: pal.hairline }
			},
			rightPriceScale: { borderColor: pal.hairline },
			crosshair: {
				vertLine: { color: pal.inkDim, labelBackgroundColor: pal.accent },
				horzLine: { color: pal.inkDim, labelBackgroundColor: pal.accent }
			},
			timeScale: {
				borderColor: pal.hairline,
				timeVisible: true,
				secondsVisible: false,
				// bar.t is naive-ET epoch seconds: format via UTC getters.
				tickMarkFormatter: (time: Time) => timeLabel(time)
			},
			localization: { timeFormatter: (time: Time) => timeLabel(time) }
		});

		const candles = chart.addSeries(CandlestickSeries, {
			upColor: pal.up,
			downColor: pal.down,
			wickUpColor: pal.up,
			wickDownColor: pal.down,
			borderVisible: false,
			priceLineVisible: true
		});
		candles.priceScale().applyOptions({ scaleMargins: { top: 0.08, bottom: 0.26 } });

		// Volume lives on its own overlay scale pinned to the bottom band.
		const volume = chart.addSeries(HistogramSeries, {
			priceScaleId: 'volume',
			priceFormat: { type: 'volume' },
			lastValueVisible: false,
			priceLineVisible: false
		});
		chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

		const markers = createSeriesMarkers<Time>(candles, []);

		const onCrosshair = (param: MouseEventParams<Time>): void => {
			const cd = param.seriesData.get(candles) as CandlestickData<Time> | undefined;
			if (param.time === undefined || cd === undefined || !('close' in cd)) {
				hover = null;
				return;
			}
			const t = typeof param.time === 'number' ? param.time : 0;
			const u = uncByT.get(t);
			hover = {
				t,
				o: cd.open,
				h: cd.high,
				l: cd.low,
				c: cd.close,
				epistemic: u?.epistemic ?? null,
				anomaly: u?.anomaly ?? null
			};
		};
		chart.subscribeCrosshairMove(onCrosshair);

		const ro = new ResizeObserver(() => {
			chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
		});
		ro.observe(el);

		handles = { chart, candles, volume, markers, pal };
		return () => {
			handles = null;
			ro.disconnect();
			chart.unsubscribeCrosshairMove(onCrosshair);
			markers.detach();
			chart.remove();
		};
	});

	// ---- data ------------------------------------------------------------
	$effect(() => {
		const h = handles;
		if (!h) return;
		const candleData: CandlestickData<UTCTimestamp>[] = bars.map((b) => ({
			time: b.t as UTCTimestamp,
			open: b.o,
			high: b.h,
			low: b.l,
			close: b.c
		}));
		const volumeData: HistogramData<UTCTimestamp>[] = bars.map((b) => ({
			time: b.t as UTCTimestamp,
			value: b.v,
			color: b.c >= b.o ? h.pal.upSoft : h.pal.downSoft
		}));
		h.candles.setData(candleData);
		h.volume.setData(volumeData);
		lastBarTime = bars.length > 0 ? bars[bars.length - 1].t : 0;
		h.chart.timeScale().fitContent();
	});

	// ---- live bar (replay streaming) --------------------------------------
	$effect(() => {
		const h = handles;
		const lb = liveBar;
		if (!h || !lb || lastBarTime === 0) return;
		// Only append/replace forward within the loaded session: series.update()
		// rejects out-of-order times, and a bar from another day would smear a
		// stray candle onto this session.
		if (lb.t < lastBarTime || lb.t - lastBarTime > SESSION_WINDOW_S) return;
		h.candles.update({ time: lb.t as UTCTimestamp, open: lb.o, high: lb.h, low: lb.l, close: lb.c });
		h.volume.update({
			time: lb.t as UTCTimestamp,
			value: lb.v,
			color: lb.c >= lb.o ? h.pal.upSoft : h.pal.downSoft
		});
		lastBarTime = lb.t;
	});

	// ---- signal / trade markers -------------------------------------------
	$effect(() => {
		const h = handles;
		if (!h) return;
		const all: SeriesMarker<Time>[] = [];
		for (const s of signals) {
			all.push({
				time: s.ts as UTCTimestamp,
				position: s.side === 'long' ? 'belowBar' : 'aboveBar',
				shape: s.side === 'long' ? 'arrowUp' : 'arrowDown',
				color: s.side === 'long' ? h.pal.up : h.pal.down,
				text: `${Math.round(s.conviction * 100)}%`,
				size: 1,
				id: s.signal_id
			});
		}
		for (const t of trades) {
			all.push({
				time: t.exit_ts as UTCTimestamp,
				position: 'inBar',
				shape: 'circle',
				color: t.pnl >= 0 ? h.pal.up : h.pal.down,
				size: 0.5,
				id: t.trade_id
			});
		}
		all.sort((a, b) => (a.time as number) - (b.time as number));
		// Cap at the newest MAX_MARKERS for render performance.
		h.markers.setMarkers(all.length > MAX_MARKERS ? all.slice(all.length - MAX_MARKERS) : all);
	});

	// ---- focused-signal price lines ----------------------------------------
	$effect(() => {
		const h = handles;
		const s = focusedSignal;
		if (!h || !s) return;
		const lines: IPriceLine[] = [
			h.candles.createPriceLine({
				price: s.entry_px,
				color: h.pal.accent,
				lineWidth: 1,
				lineStyle: LineStyle.Solid,
				title: 'ENTRY',
				axisLabelVisible: true
			}),
			h.candles.createPriceLine({
				price: s.stop_px,
				color: h.pal.down,
				lineWidth: 1,
				lineStyle: LineStyle.Dashed,
				title: 'STOP',
				axisLabelVisible: true
			}),
			h.candles.createPriceLine({
				price: s.target_px,
				color: h.pal.up,
				lineWidth: 1,
				lineStyle: LineStyle.Dashed,
				title: 'TARGET',
				axisLabelVisible: true
			})
		];
		return () => {
			// Chart may already be disposed on unmount; removal is best-effort.
			try {
				for (const line of lines) h.candles.removePriceLine(line);
			} catch {
				/* chart already removed */
			}
		};
	});
</script>

<div
	class="relative w-full"
	style={height > 0 ? `height:${height}px` : 'height:100%'}
	data-testid="price-chart"
>
	<div bind:this={container} class="absolute inset-0" aria-label="Price chart" role="img"></div>
	{#if hover}
		<div
			class="num pointer-events-none absolute top-1.5 left-2 z-10 rounded border border-hairline bg-surface/80 px-2 py-1 text-[10px] text-ink-dim"
		>
			<span class="text-ink">{hhmm(hover.t)}</span>
			<span class="ml-2">O {fmtPx(hover.o)}</span>
			<span class="ml-1.5">H {fmtPx(hover.h)}</span>
			<span class="ml-1.5">L {fmtPx(hover.l)}</span>
			<span class="ml-1.5 {hover.c >= hover.o ? 'text-long' : 'text-short'}">C {fmtPx(hover.c)}</span>
			{#if hover.epistemic !== null}
				<span class="ml-2 text-accent">epi {hover.epistemic.toFixed(2)}</span>
			{/if}
			{#if hover.anomaly !== null && hover.anomaly > 0}
				<span class="ml-1.5 text-short">anom {hover.anomaly.toFixed(2)}</span>
			{/if}
		</div>
	{/if}
</div>
