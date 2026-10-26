// Synthetic acceptance through the rendered browser, HTTP API and durable reader files.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.NDE_BROWSER_CHANNEL ? { channel: process.env.NDE_BROWSER_CHANNEL } : {}),
  });
  try {
    const page = await browser.newPage();
    const base = process.argv[2];
    await page.goto(`${base}/pairs/1`);
    assert.equal((await page.content()).includes('SECRET-'), false);
    for (const side of ['a', 'b']) {
      for (const dimension of ['coherence', 'originality']) {
        assert.equal(await page.locator(`input[name="score-${side}-${dimension}"][value="1"]`).getAttribute('required'), '');
      }
    }
    assert.equal((await page.request.post(`${base}/pairs/1`, { form: { preferred: 'A', strength: '2' } })).status(), 422);
    assert.equal((await page.request.post(`${base}/close`)).status(), 409);
    assert.equal((await page.request.get(`${base}/sealed/key.json`)).status(), 404);
    assert.equal((await page.request.get(`${base}/sealed/g1.json`)).status(), 404);
    assert.equal((await page.request.get(`${base}/api/progress`)).status(), 200);
    for (const side of ['a', 'b']) {
      await page.locator(`input[name="score-${side}-coherence"][value="4"]`).check();
      await page.locator(`input[name="score-${side}-originality"][value="3"]`).check();
    }
    await page.locator('input[name="preferred"][value="A"]').check();
    await page.locator('input[name="strength"][value="2"]').check();
    await page.getByRole('button', { name: 'Save and continue' }).click();
    const saved = await (await page.request.get(`${base}/api/pairs/1`)).json();
    assert.deepEqual(saved.judgement.a.scores, { coherence: 4, originality: 3 });
    assert.equal(JSON.stringify(saved).includes('sealed'), false);
    assert.equal(JSON.stringify(saved).includes('condition'), false);
    assert.equal(JSON.stringify(saved).includes('SECRET-'), false);
    await page.getByRole('button', { name: 'Close the batch' }).click();
    await page.reload();
    const progress = await (await page.request.get(`${base}/api/progress`)).json();
    assert.equal(progress.closed, true);
    assert.equal(progress.judged, 1);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
