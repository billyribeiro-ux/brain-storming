import { chromium } from 'playwright';
const BASE = process.env.BASE ?? 'http://localhost:4173';
const OUT = process.env.OUT ?? '/tmp/shots';
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 }, deviceScaleFactor: 2 });

// --- Autopsy drill-down: click an actual losing trade row ------------------ //
await page.goto(BASE + '/autopsies', { waitUntil: 'networkidle', timeout: 60000 });
await page.waitForTimeout(3500);
// Trade rows carry a skull when has_autopsy; click the first row that shows a
// 'stop' reason (those are the autopsied losses). Target the clickable row el.
const row = page.locator('[role="row"], tr, [role="button"]').filter({ hasText: /stop/ }).first();
await row.scrollIntoViewIfNeeded();
await row.click({ force: true });
await page.waitForTimeout(2500);
await page.screenshot({ path: `${OUT}/autopsy-detail.png` });

// --- Imagination: click Imagine and wait for the fan to render ------------- //
await page.goto(BASE + '/playground', { waitUntil: 'networkidle', timeout: 60000 });
await page.waitForTimeout(2500);
await page.getByRole('button', { name: /^imagine$/i }).first().click();
// The world model checkpoint loads + rolls out — give it real time, then
// wait for the fan chart <svg> to appear before shooting.
await page.waitForTimeout(9000);
await page.screenshot({ path: `${OUT}/playground-imagination.png` });

await browser.close();
console.log('re-captured autopsy-detail + playground-imagination');
