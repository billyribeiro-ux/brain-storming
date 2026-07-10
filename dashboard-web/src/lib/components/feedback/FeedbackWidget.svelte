<script lang="ts">
	/**
	 * Human-in-the-loop feedback: thumbs + optional comment posted to the
	 * lessons buffer. Confirmation is optimistic; a failed POST surfaces
	 * inline and the form stays editable for a retry.
	 */
	import { PaperPlaneTilt, ThumbsDown, ThumbsUp } from 'phosphor-svelte';
	import { api } from '$lib/api/client';

	let { context, ticker, signalId }: { context: string; ticker?: string; signalId?: string } =
		$props();

	let rating = $state<1 | -1 | null>(null);
	let comment = $state('');
	let status = $state<'idle' | 'sent' | 'error'>('idle');
	let errorMsg = $state('');

	function submit(event: SubmitEvent): void {
		event.preventDefault();
		if (rating === null) return;
		const body = { context, ticker, signal_id: signalId, rating, comment: comment.trim() };
		status = 'sent'; // optimistic — the buffer almost never rejects
		errorMsg = '';
		api.feedback(body).catch((e: unknown) => {
			status = 'error';
			errorMsg = e instanceof Error ? e.message : 'feedback failed';
		});
	}

	function pick(value: 1 | -1): void {
		rating = rating === value ? null : value;
		if (status !== 'idle') status = 'idle';
	}
</script>

<form class="flex flex-col gap-1.5" onsubmit={submit} aria-label="Feedback">
	<div class="flex items-center gap-1.5">
		<button
			type="button"
			onclick={() => pick(1)}
			aria-pressed={rating === 1}
			aria-label="Helpful"
			disabled={status === 'sent'}
			class={`rounded p-1.5 transition-colors focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none disabled:opacity-50 ${
				rating === 1 ? 'bg-accent-soft text-accent' : 'text-ink-faint hover:bg-raised hover:text-ink'
			}`}
		>
			<ThumbsUp size={14} weight={rating === 1 ? 'fill' : 'regular'} aria-hidden="true" />
		</button>
		<button
			type="button"
			onclick={() => pick(-1)}
			aria-pressed={rating === -1}
			aria-label="Not helpful"
			disabled={status === 'sent'}
			class={`rounded p-1.5 transition-colors focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none disabled:opacity-50 ${
				rating === -1
					? 'bg-accent-soft text-accent'
					: 'text-ink-faint hover:bg-raised hover:text-ink'
			}`}
		>
			<ThumbsDown size={14} weight={rating === -1 ? 'fill' : 'regular'} aria-hidden="true" />
		</button>
		<input
			type="text"
			bind:value={comment}
			placeholder="Teach the brain…"
			aria-label="Feedback comment"
			disabled={status === 'sent'}
			class="min-w-0 flex-1 rounded border border-hairline bg-raised px-2 py-1 text-[11px] text-ink placeholder:text-ink-faint focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none disabled:opacity-50"
		/>
		<button
			type="submit"
			disabled={rating === null || status === 'sent'}
			aria-label="Submit feedback"
			class="flex items-center gap-1 rounded bg-accent-soft px-2 py-1.5 text-[11px] text-accent transition-colors hover:bg-accent/25 focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-40"
		>
			<PaperPlaneTilt size={12} aria-hidden="true" />
			Send
		</button>
	</div>

	{#if status === 'sent'}
		<p class="text-[10px] text-long" role="status">Recorded — feeds the lessons buffer.</p>
	{:else if status === 'error'}
		<p class="text-[10px] text-short" role="alert">Could not record feedback — {errorMsg}</p>
	{/if}
</form>
