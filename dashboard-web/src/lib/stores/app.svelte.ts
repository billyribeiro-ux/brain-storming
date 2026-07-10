/**
 * Global UI state — Svelte 5 runes class, one instance per app.
 * Selection state only; server data lives in TanStack Query caches and the
 * WebSocket feed. Keep this store small and boring.
 */

export type Timeframe = '1min' | '5min';

class AppState {
	ticker = $state<string>('AAPL');
	date = $state<string>(''); // ISO session date; '' = newest available
	timeframe = $state<Timeframe>('1min');
	backtest = $state<string>(''); // '' = newest backtest
	// Overlay toggles for the main chart.
	showSignals = $state(true);
	showTrades = $state(true);
	showLevels = $state(true); // stop/target price lines of the focused signal
	showAttention = $state(true); // uncertainty/attention strip
	focusedSignalId = $state<string | null>(null);
	sidebarCollapsed = $state(false);
}

export const app = new AppState();
