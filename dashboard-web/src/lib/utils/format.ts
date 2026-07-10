/**
 * Formatting utilities — every number the cockpit renders goes through one
 * of these so precision and style stay uniform.
 *
 * Timestamps: the Aether lake stores naive-ET wall-clock epoch seconds.
 * They are therefore formatted with the UTC formatter (the epoch already
 * IS the wall time) — never converted through the browser's zone.
 */
import { format } from 'date-fns';

/** naive-ET epoch seconds -> Date whose UTC fields hold the ET wall time. */
export const etDate = (ts: number): Date => new Date(ts * 1000);

export const fmtTime = (ts: number): string => format(etDate(ts), 'HH:mm:ss');
export const fmtTimeShort = (ts: number): string => format(etDate(ts), 'HH:mm');
export const fmtDate = (ts: number): string => format(etDate(ts), 'yyyy-MM-dd');
export const fmtDateTime = (ts: number): string => format(etDate(ts), 'MMM d HH:mm');

export const fmtPx = (v: number | null | undefined, dp = 2): string =>
	v == null || Number.isNaN(v) ? '—' : v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });

export const fmtPnl = (v: number | null | undefined): string =>
	v == null ? '—' : `${v >= 0 ? '+' : '−'}$${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export const fmtPct = (v: number | null | undefined, dp = 1): string =>
	v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(dp)}%`;

/** Conviction 0..1 -> 0..100 display with a semantic band. */
export const convictionBand = (c: number): 'high' | 'medium' | 'low' =>
	c >= 0.65 ? 'high' : c >= 0.4 ? 'medium' : 'low';

export const pnlClass = (v: number): string => (v >= 0 ? 'text-long' : 'text-short');
export const sideClass = (side: 'long' | 'short'): string =>
	side === 'long' ? 'text-long' : 'text-short';
