// Run with a local Refdesk server and Playwright installed.
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

const base = process.env.REFDESK_TEST_URL || "http://127.0.0.1:3000";
const repo = "https://github.com/XabiBlaz/tensor-decomposition-compression";

async function run() {
  const browser = await chromium.launch({
    ...(process.env.REFDESK_CHROMIUM ? { executablePath: process.env.REFDESK_CHROMIUM } : {}),
    args: ["--no-sandbox"],
  });
  try {
    for (const [name, width, height] of [["desktop", 1440, 900], ["phone", 390, 844]]) {
      const page = await browser.newPage({ viewport: { width, height }, isMobile: name === "phone" });
      const errors = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.goto(base);
      await page.keyboard.press("Tab");
      assert.equal(await page.evaluate(() => document.activeElement.textContent), "Skip to question");
      await page.keyboard.press("Tab");
      if (process.env.REFDESK_SCREENSHOTS) await page.screenshot({ path: `/tmp/refdesk-${name}-welcome.png` });
      const url = page.getByLabel("Public GitHub URL");
      await url.fill("https://example.org/owner/repo");
      await page.getByRole("button", { name: "Index repository" }).click();
      await page.getByText("Enter a public GitHub URL", { exact: false }).waitFor();
      await url.fill(repo);
      await page.getByRole("button", { name: "Index repository" }).click();
      await page.locator("#repository-link").waitFor();
      await page.getByRole("button", { name: "Refresh sources" }).click();
      await page.getByText("Sources refreshed.", { exact: false }).waitFor();
      const compose = page.getByRole("textbox", { name: "Your question" });
      await compose.fill("How do I start the Docker Compose UI?");
      await compose.press("Enter");
      const first = page.locator("article").first();
      await first.getByText("localhost:7860", { exact: false }).waitFor();
      const citation = first.locator(".citations a").first();
      assert.match(await citation.getAttribute("href"), /github\.com\/.*\/blob\/[a-f0-9]{40}\/.*#L\d+-L\d+/);
      await compose.fill("What about CUDA?");
      await page.getByRole("button", { name: "Ask", exact: false }).last().click();
      await page.locator("article").nth(1).locator("pre").filter({ hasText: "compose.gpu.yaml" }).first().waitFor();
      if (process.env.REFDESK_SCREENSHOTS) await page.screenshot({ path: `/tmp/refdesk-${name}-chat.png` });
      await compose.fill("Which MATLAB add-on license enables the Android TensorRT exporter?");
      await compose.press("Enter");
      await page.locator("article").nth(2).getByText("Insufficient evidence", { exact: false }).waitFor();
      await page.locator("article").last().getByRole("button", { name: "Flag as unhelpful" }).click();
      await page.getByRole("button", { name: "Flagged as unhelpful" }).waitFor();
      await page.getByRole("button", { name: "Draft issue" }).click();
      assert.match(await page.getByRole("textbox", { name: "Issue draft" }).inputValue(), /MATLAB/);
      await page.getByRole("button", { name: "Close" }).click();
      await page.reload();
      assert.equal(await page.locator("article").count(), 3);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), width);
      assert.deepEqual(errors, []);
      await page.close();
      console.log(`${name} ${width}×${height}: loading, repository, refresh, follow-up, citations, abstention, feedback, issue draft and history passed`);
    }
  } finally {
    await browser.close();
  }
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
