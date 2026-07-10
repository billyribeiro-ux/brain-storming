<script lang="ts">
	/**
	 * Compact sortable listing of the fitted causal edges. Default order is
	 * |weight|·confidence descending (the same "strength" the graph uses),
	 * filterable by node name, capped at 200 rows.
	 */
	import type { CausalEdge, CausalSnapshot } from '$lib/api/schemas';

	let { snapshot, filter = '' }: { snapshot: CausalSnapshot; filter?: string } = $props();

	const CAP = 200;

	type SortKey = 'src' | 'dst' | 'lag' | 'weight' | 'confidence' | 'strength';
	type Column = { key: SortKey; label: string; numeric: boolean };

	const COLUMNS: Column[] = [
		{ key: 'src', label: 'src', numeric: false },
		{ key: 'dst', label: 'dst', numeric: false },
		{ key: 'lag', label: 'lag', numeric: true },
		{ key: 'weight', label: 'weight', numeric: true },
		{ key: 'confidence', label: 'conf', numeric: true }
	];

	// Writable derived: the filter prop (e.g. a node clicked in the graph)
	// re-seeds the input, and the user can still edit it freely afterwards.
	let query = $derived(filter);
	let sortKey = $state<SortKey>('strength');
	let sortDir = $state<1 | -1>(-1);

	const strength = (e: CausalEdge): number => Math.abs(e.weight) * e.confidence;

	function setSort(key: SortKey, numeric: boolean): void {
		if (sortKey === key) {
			sortDir = sortDir === 1 ? -1 : 1;
		} else {
			sortKey = key;
			sortDir = numeric ? -1 : 1;
		}
	}

	const filtered = $derived.by((): CausalEdge[] => {
		const q = query.trim().toLowerCase();
		if (!q) return snapshot.edges;
		return snapshot.edges.filter(
			(e) => e.src.toLowerCase().includes(q) || e.dst.toLowerCase().includes(q)
		);
	});

	const rows = $derived.by((): CausalEdge[] => {
		const sorted = [...filtered].sort((a, b) => {
			let c: number;
			switch (sortKey) {
				case 'src':
					c = a.src.localeCompare(b.src);
					break;
				case 'dst':
					c = a.dst.localeCompare(b.dst);
					break;
				case 'lag':
					c = a.lag - b.lag;
					break;
				case 'weight':
					c = a.weight - b.weight;
					break;
				case 'confidence':
					c = a.confidence - b.confidence;
					break;
				default:
					c = strength(a) - strength(b);
			}
			return c * sortDir || strength(b) - strength(a);
		});
		return sorted.slice(0, CAP);
	});

	const ariaSort = (key: SortKey): 'ascending' | 'descending' | undefined =>
		sortKey === key ? (sortDir === 1 ? 'ascending' : 'descending') : undefined;
</script>

<div class="flex h-full min-h-0 flex-col gap-2">
	<input
		type="text"
		bind:value={query}
		placeholder="filter nodes…"
		aria-label="Filter edges by node name"
		class="w-full rounded-md border border-hairline bg-void/60 px-2 py-1 font-mono text-[11px] text-ink placeholder:text-ink-faint focus:border-accent focus:outline-none"
	/>
	<div class="scroll-thin min-h-0 flex-1 overflow-auto rounded-md border border-hairline">
		<table class="w-full border-collapse text-[11px]">
			<thead class="sticky top-0 z-10 bg-surface">
				<tr>
					{#each COLUMNS as col (col.key)}
						<th
							class={`border-b border-hairline px-2 py-1.5 font-medium text-ink-faint ${col.numeric ? 'text-right' : 'text-left'}`}
							aria-sort={ariaSort(col.key)}
						>
							<button
								type="button"
								class="cursor-pointer tracking-wide uppercase hover:text-ink focus:text-ink focus:outline-none"
								onclick={() => setSort(col.key, col.numeric)}
							>
								{col.label}{sortKey === col.key ? (sortDir === 1 ? ' ▲' : ' ▼') : ''}
							</button>
						</th>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#each rows as e, i (`${e.src}→${e.dst}@${e.lag}:${i}`)}
					<tr class="border-b border-hairline/50 hover:bg-raised/40">
						<td class="max-w-32 truncate px-2 py-1 font-mono text-ink-dim" title={e.src}>{e.src}</td>
						<td class="max-w-32 truncate px-2 py-1 font-mono text-ink-dim" title={e.dst}>{e.dst}</td>
						<td class="num px-2 py-1 text-right text-ink-faint">{e.lag}</td>
						<td class={`num px-2 py-1 text-right ${e.weight >= 0 ? 'text-long' : 'text-short'}`}>
							{e.weight >= 0 ? '+' : '−'}{Math.abs(e.weight).toFixed(3)}
						</td>
						<td class="px-2 py-1">
							<span
								class="ml-auto block h-1 w-10 overflow-hidden rounded-full bg-raised"
								title={e.confidence.toFixed(2)}
							>
								<span
									class="block h-full rounded-full bg-accent"
									style={`width:${(e.confidence * 100).toFixed(0)}%`}
								></span>
							</span>
						</td>
					</tr>
				{:else}
					<tr>
						<td colspan="5" class="px-2 py-4 text-center text-ink-faint">no edges match the filter</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
	<p class="num text-right text-[10px] text-ink-faint">
		{rows.length < filtered.length
			? `showing ${rows.length} of ${filtered.length} edges`
			: `${rows.length} edges`}
	</p>
</div>
