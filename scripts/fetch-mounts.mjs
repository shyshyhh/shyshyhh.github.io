// Refreshes src/data/mounts.json from SimpleArmory's rendered page.
// Runs in CI before each build (see .github/workflows/deploy.yml); the site
// falls back to the committed JSON if this fails, so failures are non-fatal.
import { chromium } from 'playwright';
import { writeFileSync } from 'node:fs';

const URL = 'https://simplearmory.com/#/us/illidan/xxwinter/collectable/mounts';
const OUT = new globalThis.URL('../src/data/mounts.json', import.meta.url);

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 60_000 });
  // Header renders like "1249 / 1290 (97%)"
  await page.waitForFunction(
    () => /\d+\s*\/\s*\d+\s*\(\d+%\)/.test(document.body.innerText),
    { timeout: 30_000 }
  );
  const text = await page.evaluate(() => document.body.innerText);
  const m = text.match(/(\d+)\s*\/\s*(\d+)\s*\((\d+)%\)/);
  if (!m) throw new Error('mount header not found');
  const data = {
    collected: Number(m[1]),
    total: Number(m[2]),
    pct: Number(m[3]),
    fetchedAt: new Date().toISOString().slice(0, 10),
  };
  writeFileSync(OUT, JSON.stringify(data, null, 2) + '\n');
  console.log('mounts.json updated:', data);
} finally {
  await browser.close();
}
