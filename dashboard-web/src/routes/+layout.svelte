<script lang="ts">
	/**
	 * App shell: top bar + ticker sidebar + routed main area.
	 * Owns the QueryClient and the WebSocket lifecycle.
	 */
	import '../app.css';
	import '@fontsource-variable/inter';
	import '@fontsource/jetbrains-mono';
	import favicon from '$lib/assets/favicon.svg';
	import { QueryClient, QueryClientProvider } from '@tanstack/svelte-query';
	import { onMount } from 'svelte';
	import { live } from '$lib/api/ws.svelte';
	import TopBar from '$lib/components/layout/TopBar.svelte';
	import Sidebar from '$lib/components/layout/Sidebar.svelte';

	let { children } = $props();

	const queryClient = new QueryClient({
		defaultOptions: {
			queries: { staleTime: 30_000, retry: 2, refetchOnWindowFocus: false }
		}
	});

	onMount(() => {
		live.connect();
		return () => live.disconnect();
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<QueryClientProvider client={queryClient}>
	<div class="flex h-dvh flex-col overflow-hidden">
		<TopBar />
		<div class="flex min-h-0 flex-1">
			<Sidebar />
			<main class="scroll-thin min-w-0 flex-1 overflow-y-auto p-3">
				{@render children()}
			</main>
		</div>
	</div>
</QueryClientProvider>
