<script lang="ts">
	/**
	 * Brain page: system vitals up top, then the fitted causal graph with its
	 * edge table. Clicking a graph node filters the edge table and, when the
	 * node belongs to a tradable ticker, refocuses the cockpit on it.
	 */
	import { createQuery } from '@tanstack/svelte-query';
	import { api } from '$lib/api/client';
	import { app } from '$lib/stores/app.svelte';
	import Panel from '$lib/components/layout/Panel.svelte';
	import BrainStatusGrid from '$lib/components/brain/BrainStatusGrid.svelte';
	import CausalGraph from '$lib/components/causal/CausalGraph.svelte';
	import EdgeTable from '$lib/components/causal/EdgeTable.svelte';

	const statusQ = createQuery(() => ({
		queryKey: ['status'],
		queryFn: () => api.status(),
		staleTime: 60_000,
		refetchInterval: 60_000
	}));

	const causalQ = createQuery(() => ({
		queryKey: ['causal'],
		queryFn: () => api.causal()
	}));

	let selectedNode = $state('');

	function onSelect(node: string): void {
		selectedNode = node;
		const dot = node.indexOf('.');
		const prefix = dot < 0 ? node : node.slice(0, dot);
		if (prefix && prefix !== 'MKT') app.ticker = prefix;
	}

	const causalSubtitle = $derived(
		causalQ.data
			? `fitted ${causalQ.data.fitted_start} → ${causalQ.data.fitted_end} · ${causalQ.data.nodes.length} nodes · ${causalQ.data.edges.length} edges`
			: 'lagged structural weights with bootstrap confidence'
	);
</script>

<svelte:head>
	<title>Brain — Aether</title>
</svelte:head>

<div class="flex flex-col gap-3">
	<header class="flex flex-wrap items-baseline justify-between gap-2">
		<h1 class="text-base font-semibold tracking-wide text-ink">Brain Status</h1>
		{#if statusQ.data}
			<span class="num text-[11px] text-ink-faint">generated {statusQ.data.generated_at}</span>
		{/if}
	</header>

	<BrainStatusGrid />

	<Panel
		title="Causal Brain"
		subtitle={causalSubtitle}
		loading={causalQ.isPending}
		error={causalQ.error ? causalQ.error.message : null}
		empty={!causalQ.isPending && !causalQ.error && !causalQ.data
			? 'No causal snapshot yet — run the causal fit.'
			: null}
		class="min-h-[32rem]"
	>
		{#if causalQ.data}
			<div class="grid h-full grid-cols-1 gap-3 xl:grid-cols-12">
				<div class="min-h-96 min-w-0 xl:col-span-8">
					<CausalGraph snapshot={causalQ.data} focusTicker={app.ticker} {onSelect} />
				</div>
				<div class="max-h-[32rem] min-h-64 min-w-0 xl:col-span-4">
					<EdgeTable snapshot={causalQ.data} filter={selectedNode} />
				</div>
			</div>
		{/if}
	</Panel>
</div>
