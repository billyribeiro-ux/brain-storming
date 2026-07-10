<script lang="ts">
	/**
	 * Conviction pill. Accepts the engine's 0..1 conviction fraction and
	 * renders it 0-100 with the shared semantic band (>=65 conviction glows
	 * long-green, 40-65 amber, <40 fades out).
	 */
	import { convictionBand } from '$lib/utils/format';

	let { conviction }: { conviction: number } = $props();

	const pct = $derived(Math.round(Math.min(1, Math.max(0, conviction)) * 100));
	const band = $derived(convictionBand(conviction));
</script>

<span
	role="img"
	aria-label={`conviction ${pct} of 100`}
	class={`num inline-flex min-w-9 items-center justify-center rounded px-1.5 py-0.5 text-[11px] leading-none ring-1 ring-inset ${
		band === 'high'
			? 'bg-long-soft text-long ring-long/40'
			: band === 'medium'
				? 'bg-warn-soft text-warn ring-warn/40'
				: 'text-ink-faint ring-hairline'
	}`}
	style={band === 'high' ? 'box-shadow: 0 0 10px 1px var(--color-long-soft);' : undefined}
>
	{pct}
</span>
