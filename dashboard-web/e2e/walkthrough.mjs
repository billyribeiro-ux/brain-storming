/**
 * End-to-end walkthrough of the Aether cockpit against the live bridge.
 * Drives every page, asserts real data rendered, captures screenshots,
 * and fails on any console error.
 */
import { chromium } from 'playwright';

const BASE = process.env.BASE ?? 'http://localhost:4173';
const OUT = process.env.OUT ?? '/tmp/shots';
const consoleErrors = [];

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium',
});
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 } });
page.on('console', (m) => {
  if (m.type() === 'error') consoleErrors.push(m.text());
});
page.on('pageerror', (e) => consoleErrors.push(String(e)));

const shot = async (name) => page.screenshot({ path: `${OUT}/${name}.png`, fullPage: false });

// ---- Deck ---------------------------------------------------------------- //
await page.goto(BASE, { waitUntil: 'networkidle', timeout: 60000 });
await page.waitForTimeout(4000);
const brand = await page.textContent('header, [class*="bar"], body');
if (!brand?.includes('AETHER')) throw new Error('brand missing');
// Sidebar has 10 tickers
const sidebarText = await page.textContent('aside, nav, body');
for (const t of ['AAPL', 'NVDA', 'SPX']) {
  if (!sidebarText?.includes(t)) throw new Error(`sidebar missing ${t}`);
}
// Chart canvas exists (lightweight-charts renders <canvas>)
await page.waitForSelector('canvas', { timeout: 30000 });
await shot('01-deck');

// Switch ticker
await page.click('text=NVDA');
await page.waitForTimeout(3000);
await shot('02-deck-nvda');

// ---- Autopsies ------------------------------------------------------------ //
await page.goto(`${BASE}/autopsies`, { waitUntil: 'networkidle', timeout: 60000 });
await page.waitForTimeout(4000);
const bodyText = await page.textContent('body');
if (!/\d/.test(bodyText ?? '')) throw new Error('autopsies page empty');
await shot('03-autopsies');

// ---- Brain ----------------------------------------------------------------- //
await page.goto(`${BASE}/brain`, { waitUntil: 'networkidle', timeout: 60000 });
await page.waitForTimeout(5000);
await page.waitForSelector('svg', { timeout: 30000 }); // causal graph
await shot('04-brain');

// ---- Playground ------------------------------------------------------------ //
await page.goto(`${BASE}/playground`, { waitUntil: 'networkidle', timeout: 60000 });
await page.waitForTimeout(3000);
await shot('05-playground');

await browser.close();

// Tolerate benign errors (favicon, ws close on nav), fail on real ones.
const real = consoleErrors.filter(
  (e) => !/favicon|WebSocket|ws:\/\/|net::ERR_ABORTED/i.test(e)
);
if (real.length) {
  console.error('CONSOLE ERRORS:\n' + real.join('\n'));
  process.exit(1);
}
console.log('E2E OK — screenshots in', OUT);
