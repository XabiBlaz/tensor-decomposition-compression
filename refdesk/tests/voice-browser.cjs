// Use a synthetic microphone fixture with the real browser, STT, chat, and TTS.
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { writeFileSync } = require("node:fs");
const { chromium } = require("playwright");

const base = process.env.REFDESK_TEST_URL || "http://127.0.0.1:3000";
const expected = "how do i search hidden files";
const keys = [
  "transcription_ms", "retrieval_ms", "generation_ms", "synthesis_first_chunk_ms",
  "end_to_audio_onset_ms", "finish_to_audio_onset_ms", "submit_to_audio_onset_ms",
];

const round = (value) => Math.round(value * 100) / 100;
function wordErrors(reference, actual) {
  const words = reference.split(" ");
  let row = words.map((_, i) => i + 1);
  row.unshift(0);
  actual.split(" ").forEach((word, i) => {
    const next = [i + 1];
    words.forEach((expectedWord, j) => {
      next.push(Math.min(next[j] + 1, row[j + 1] + 1, row[j] + Number(expectedWord !== word)));
    });
    row = next;
  });
  return row.at(-1);
}

function percentile(values, quantile) {
  return round(values.toSorted((a, b) => a - b)[Math.ceil(values.length * quantile) - 1]);
}

async function run() {
  const browser = await chromium.launch({
    ...(process.env.REFDESK_CHROMIUM ? { executablePath: process.env.REFDESK_CHROMIUM } : {}),
    args: ["--no-sandbox", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${process.env.REFDESK_AUDIO_FIXTURE}%noloop`],
  });
  try {
    const page = await browser.newPage({ permissions: ["microphone"] });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.addInitScript(() => {
      window.voiceTimings = [];
      document.addEventListener("refdesk-voice-timing", (event) => window.voiceTimings.push(event.detail));
    });
    await page.goto(base);
    await page.getByLabel("Public GitHub URL").fill("https://github.com/BurntSushi/ripgrep");
    await page.getByRole("button", { name: "Index repository" }).click();
    const repo = page.locator("#repository-link");
    await repo.waitFor();
    const start = page.getByRole("button", { name: "Start voice", exact: true });
    const finish = page.getByRole("button", { name: "Finish recording", exact: true });
    const confirm = page.getByRole("button", { name: "Ask aloud", exact: true });
    const compose = page.getByRole("textbox", { name: "Your question" });
    const stop = page.getByRole("button", { name: "Stop voice", exact: true });
    const status = page.locator("#voice-status");
    const turns = page.locator("article");

    async function record() {
      await start.click();
      await finish.waitFor();
      await new Promise((resolve) => setTimeout(resolve, 3430));
      await finish.click();
    }

    async function submitTranscript() {
      await confirm.waitFor();
      if (await compose.inputValue() !== expected) await compose.fill(expected);
      await confirm.click();
    }

    for (let i = 0; i < 5; i++) {
      await record();
      await submitTranscript();
      await page.waitForFunction((count) => window.voiceTimings.length === count, i + 1);
      const title = await turns.last().locator("h2").textContent();
      assert.equal(title, expected);
      await turns.last().locator(".citations").getByText("GUIDE.md", { exact: false }).waitFor();
      await stop.click();
      assert.equal(await status.textContent(), "Voice stopped. You can continue in text.");
    }

    // Stop during active capture before transcription can create a turn.
    await start.click();
    await finish.waitFor();
    await stop.click();
    assert.equal(await turns.count(), 5);

    const failTts = (route) => route.abort("connectionreset");
    await page.route("**/api/speech", failTts);
    await record();
    await submitTranscript();
    await status.getByText("Voice connection lost.", { exact: false }).waitFor();
    assert.equal(await turns.count(), 6); // The completed text answer remains visible.
    await page.unroute("**/api/speech", failTts);

    const failStt = (route) => route.abort("connectionreset");
    await page.route("**/api/transcribe", failStt);
    await record();
    await status.getByText("Voice connection lost.", { exact: false }).waitFor();
    assert.equal(await turns.count(), 6);
    await page.unroute("**/api/transcribe", failStt);
    await compose.fill("Can it also disable ignore rules?");
    await compose.press("Enter");
    const followUp = turns.filter({ has: page.getByRole("heading", { name: "Can it also disable ignore rules?" }) });
    await followUp.getByText("--no-ignore", { exact: false }).first().waitFor();
    assert.equal(await turns.count(), 7);
    assert.deepEqual(errors, []);

    const samples = await page.evaluate(() => window.voiceTimings);
    for (const sample of samples) {
      assert.ok(sample.transcript.length);
      for (const key of keys) assert.ok(sample[key] >= 0, key);
    }
    const report = {
      synthetic_microphone: true,
      fixture: expected,
      matched_transcripts: samples.filter((sample) => sample.transcript === expected).length,
      repetitions: samples.length,
      word_error_rate: round(samples.reduce((total, sample) => total + wordErrors(expected, sample.transcript), 0)
        / (samples.length * expected.split(" ").length)),
      scenarios_passed: ["cancel while recording", "stop playback", "transcription connection failure",
        "speech connection failure", "typed follow-up after spoken turn"],
      latency_ms: Object.fromEntries(keys.map((key) => [key, {
        p50: percentile(samples.map((sample) => sample[key]), .50),
        p95: percentile(samples.map((sample) => sample[key]), .95),
      }])),
      samples: samples.map(({ id, ...metrics }) => metrics),
    };
    const output = JSON.stringify(report, null, 2) + "\n";
    if (process.env.REFDESK_VOICE_REPORT) writeFileSync(process.env.REFDESK_VOICE_REPORT, output);
    console.log(output);
  } finally {
    await browser.close();
  }
}

if (!process.env.REFDESK_AUDIO_FIXTURE) {
  const fixture = "/tmp/refdesk-voice-fixture.wav";
  execFileSync("ffmpeg", ["-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
    "flite=text='How do I search hidden files?':voice=slt", "-af", "adelay=600:all=1,apad=pad_dur=1",
    "-ar", "16000", "-ac", "1", fixture]);
  process.env.REFDESK_AUDIO_FIXTURE = fixture;
}
run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
