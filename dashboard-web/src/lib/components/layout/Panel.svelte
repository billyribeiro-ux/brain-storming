<script lang="ts">
	/**
	 * The one panel primitive every widget lives in: glass surface, title
	 * row with optional actions snippet, loading/error/empty states built in
	 * so feature components never hand-roll them.
	 */
	import type { Snippet } from 'svelte';
	import { CircleNotch, WarningCircle } from 'phosphor-svelte';

	let {
		title,
		subtitle = '',
		loading = false,
		error = null,
		empty = null,
		actions,
		children,
		class: klass = ''
	}: {
		title: string;
		subtitle?: string;
		loading?: boolean;
		error?: string | null;
		empty?: string | null;
		actions?: Snippet;
		children: Snippet;
		class?: string;
	} = $props();
</script>

<section class={`glass glass-hover rise-in flex min-h-0 flex-col ${klass}`} aria-label={title}>
	<header class="flex items-center justify-between gap-2 border-b border-hairline px-3 py-2">
		<div class="min-w-0">
			<h2 class="truncate text-[13px] font-semibold tracking-wide text-ink">{title}</h2>
			{#if subtitle}<p class="truncate text-[11px] text-ink-faint">{subtitle}</p>{/if}
		</div>
		{#if actions}<div class="flex shrink-0 items-center gap-1">{@render actions()}</div>{/if}
	</header>
	<div class="scroll-thin min-h-0 flex-1 overflow-auto p-3">
		{#if loading}
			<div class="flex h-full min-h-24 items-center justify-center text-ink-faint" role="status">
				<CircleNotch size={20} class="animate-spin" aria-hidden="true" />
				<span class="ml-2 text-xs">Loading…</span>
			</div>
		{:else if error}
			<div class="flex h-full min-h-24 flex-col items-center justify-center gap-1 text-short" role="alert">
				<WarningCircle size={20} aria-hidden="true" />
				<p class="max-w-full truncate px-2 text-center text-xs">{error}</p>
			</div>
		{:else if empty}
			<div class="flex h-full min-h-24 items-center justify-center text-xs text-ink-faint">{empty}</div>
		{:else}
			{@render children()}
		{/if}
	</div>
</section>
