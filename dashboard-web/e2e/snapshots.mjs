/**
 * Capture a full snapshot set of the Aether cockpit against the live bridge.
 * Drives interactive states (expanded signal, autopsy drill-down, imagination
 * fan) so the shots show the cockpit doing real work, not just at rest.
 */
import { chromium } from 'playwright';

const BASE = process.env.BASE ?? 'http://localhost:4173';
const OUT = process.env.OUT ?? '/tmp/shots';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const page = await browser.newPage({
	viewport: { width: 1680, height: 1000 },
	deviceScaleFactor: 2 // retina-crisp screenshots
});

const shot = (name) => page.screenshot({ path: `${OUT}/${name}.png` });
const goto = (path) => page.goto(BASE + path, { waitUntil: 'networkidle', timeout: 60000 });

// 1. Deck (default AAPL) ---------------------------------------------------- //
await goto('/');
await page.waitForSelector('canvas', { timeout: 30000 });
await page.waitForTimeout(3500);
await shot('deck');

// 2. Deck with a signal expanded ------------------------------------------- //
try {
	const row = page.locator('button', { hasText: 'LONG' }).first();
	await row.click({ timeout: 5000 });
	await page.waitForTimeout(1500);
	await shot('deck-signal-expanded');
} catch {
	/* signal rows may vary; skip gracefully */
}

// 3. Brain (causal graph) --------------------------------------------------- //
await goto('/brain');
await page.waitForSelector('svg', { timeout: 30000 });
await page.waitForTimeout(4000);
await shot('brain');

// 4. Autopsies + drill into a trade with an autopsy ------------------------ //
await goto('/autopsies');
await page.waitForTimeout(4000);
await shot('autopsies');
try {
	// Click a trade row bearing the autopsy (skull) flag.
	const autopsyRow = page.locator('tr, [role="row"], button').filter({ hasText: 'stop' }).first();
	await autopsyRow.click({ timeout: 5000 });
	await page.waitForTimeout(2500);
	await shot('autopsy-detail');
} catch {
	/* skip */
}

// 5. Playground + run an imagination --------------------------------------- //
await goto('/playground');
await page.waitForTimeout(3000);
await shot('playground');
try {
	const imagineBtn = page.getByRole('button', { name: /imagine/i }).first();
	await imagineBtn.click({ timeout: 5000 });
	await page.waitForTimeout(4000);
	await shot('playground-imagination');
} catch {
	/* skip */
}

await browser.close();
console.log('snapshots captured in', OUT);
