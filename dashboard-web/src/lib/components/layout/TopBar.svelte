<script lang="ts">
	/**
	 * 48px app chrome: brand, primary nav, and the live-telemetry cluster
	 * (connection chip, replay badge, ET wall clock).
	 */
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { Brain, Broadcast, Flask, Lightning, Rewind, Skull, SquaresFour } from 'phosphor-svelte';
	import { live } from '$lib/api/ws.svelte';

	const NAV = [
		{ href: '/', label: 'Deck', icon: SquaresFour },
		{ href: '/autopsies', label: 'Autopsies', icon: Skull },
		{ href: '/playground', label: 'Playground', icon: Flask },
		{ href: '/brain', label: 'Brain', icon: Brain }
	] as const;

	const isActive = (href: string): boolean =>
		href === '/' ? page.url.pathname === '/' : page.url.pathname.startsWith(href);

	// ET wall clock — Intl carries the DST rules, so it renders identically in
	// every browser zone. Ticks once a second while mounted.
	const etFmt = new Intl.DateTimeFormat('en-US', {
		timeZone: 'America/New_York',
		hourCycle: 'h23',
		hour: '2-digit',
		minute: '2-digit',
		second: '2-digit'
	});
	let now = $state(Date.now());
	onMount(() => {
		const id = setInterval(() => (now = Date.now()), 1000);
		return () => clearInterval(id);
	});
	const clock = $derived(etFmt.format(now));

	const conn = $derived(
		live.conn === 'live'
			? { label: 'LIVE', dot: 'bg-long motion-safe:animate-pulse', text: 'text-long' }
			: live.conn === 'connecting'
				? { label: 'SYNC', dot: 'bg-warn', text: 'text-warn' }
				: { label: 'OFFLINE', dot: 'bg-short', text: 'text-short' }
	);
</script>

<header class="flex h-12 shrink-0 items-center gap-4 border-b border-hairline bg-abyss px-4">
	<a
		href="/"
		class="flex shrink-0 items-center gap-2 rounded focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none"
		aria-label="Aether home"
	>
		<Lightning size={16} weight="fill" class="text-accent" aria-hidden="true" />
		<span class="text-sm font-semibold tracking-[0.25em] text-ink">AETHER</span>
		<span class="hidden text-[10px] tracking-[0.18em] text-ink-faint lg:inline"
			>MARKET INTELLIGENCE BRAIN</span
		>
	</a>

	<nav class="flex min-w-0 flex-1 items-center justify-center gap-1" aria-label="Primary">
		{#each NAV as item (item.href)}
			{@const active = isActive(item.href)}
			<a
				href={item.href}
				aria-current={active ? 'page' : undefined}
				class={`relative flex items-center gap-1.5 rounded px-3 py-1.5 text-xs transition-colors focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none ${
					active ? 'text-ink' : 'text-ink-dim hover:text-ink'
				}`}
			>
				<item.icon size={14} aria-hidden="true" class={active ? 'text-accent' : ''} />
				{item.label}
				{#if active}
					<span class="absolute inset-x-3 bottom-0 h-px bg-accent" aria-hidden="true"></span>
				{/if}
			</a>
		{/each}
	</nav>

	<div class="flex shrink-0 items-center gap-3">
		{#if live.replay.running}
			<span
				class="num flex items-center gap-1.5 rounded-full bg-accent-soft px-2.5 py-1 text-[10px] tracking-wide text-accent"
				role="status"
				aria-label={`Replay running: ${live.replay.ticker ?? ''} ${live.replay.date ?? ''} at ${live.replay.speed} times speed`}
			>
				<Rewind size={12} weight="fill" aria-hidden="true" />
				REPLAY {live.replay.ticker ?? '—'} · {live.replay.date ?? '—'} · {live.replay.speed}×
			</span>
		{/if}

		<span
			class={`flex items-center gap-1.5 rounded-full border border-hairline px-2.5 py-1 text-[10px] font-semibold tracking-wider ${conn.text}`}
			role="status"
			aria-label={`Connection ${conn.label.toLowerCase()}`}
		>
			<Broadcast size={12} aria-hidden="true" />
			<span class={`h-1.5 w-1.5 rounded-full ${conn.dot}`} aria-hidden="true"></span>
			{conn.label}
		</span>

		<span class="num text-xs text-ink-dim" aria-label={`Eastern time ${clock}`}>
			{clock}
			<span class="ml-1 text-[10px] text-ink-faint">ET</span>
		</span>
	</div>
</header>
