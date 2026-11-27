# Text evaluation

These are synthetic prompts with source checks written for two real GitHub
repositories. The suite checks specific paths and answer facts in retrieved
passages and citations. No real users participated. Both repositories used the
same ingestion and retrieval code. Ripgrep needed only its URL and evaluation
cases, with no loader configuration or repository-specific integration.

| Suite and repository | Task completion | Hit@6 / MRR@6 | Citation support | Correct abstentions | Warm turn p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development · tensor-decomposition-compression | 7/7 | 1.00 / 1.00 | 4/4 | 3/3 | 0.099 / 1.384 ms |
| Development · ripgrep | 7/7 | 1.00 / 1.00 | 5/5 | 2/2 | 0.207 / 0.282 ms |
| Holdout · tensor-decomposition-compression | 1/3 | 0.50 / 0.25 | No citations emitted | 1/1 | 0.087 / 0.404 ms |
| Holdout · ripgrep | 1/3 | 0.50 / 0.25 | No citations emitted | 1/1 | 0.195 / 0.386 ms |

The development suite informed ranking and grounding changes. Its 14 cases
include direct questions, follow-ups, historical evidence, unknown versions,
and requests without evidence. The separate holdout contains six new cases.
The app correctly abstained on both unanswerable holdout cases and also
abstained on all four answerable ones. The ranker retrieved relevant evidence
for two of those four. This identifies a practical limit for paraphrased
questions and leaves semantic retrieval or query rewriting worth evaluating
against a separate benchmark.

Hit@6 means one curated relevant passage appeared in the first six results.
MRR@6 averages the reciprocal of its rank across evidence-backed cases. Citation
support checks whether cited excerpts contain the expected facts at acceptable
paths. Completion also requires the expected answer status and the relevant
facts in the answer. These checks use a narrow, deterministic rubric. They do
not establish support for every claim a generative provider could make.

Latency uses five repetitions per case after indexing. Values exclude network,
HTTP and rendering. Initial development-suite ingestion took 972 ms and
1,138 ms respectively. Local extractive mode incurred $0 in model charges.
GitHub REST was unavailable in this Cloud run, so release and issue ingestion
were exercised with fixtures. The live runs used commit-pinned files,
including ripgrep’s changelog. The optional model adapter was not evaluated
against a live provider.

See [development cases](cases.json), [development results](results.json),
[holdout cases](holdout.json), and [holdout results](holdout-results.json) for
revisions, per-case evidence and stage timing. Reproduce the holdout with:

```sh
python -m refdesk.evaluate --suite refdesk/evaluation/holdout.json
```

## Voice preview

Headless Chromium captured one synthetic fixture, “How do I search hidden
files?”, through its microphone interface five times. The test used the local
PocketSphinx and Flite adapters and ripgrep’s live commit-pinned file index.
It corrected the transcript before submitting each turn. The recognizer
matched **0 of 5** transcripts exactly, with **83% word error rate** on this
fixture. The corrected turns all returned the expected cited guide. No human
speakers, physical speakers, or acoustic environments were evaluated.

| Stage | p50 | p95 |
| --- | ---: | ---: |
| Transcription | 985.82 ms | 1,117.76 ms |
| Retrieval | 0.68 ms | 0.92 ms |
| Answer generation | 0.17 ms | 0.24 ms |
| Synthesis to first PCM chunk | 125.18 ms | 156.93 ms |
| Estimated end of speech to audio onset | 2,662.60 ms | 2,688.10 ms |
| Finish recording to audio onset | 1,792.60 ms | 1,848.50 ms |
| Ask aloud to audio onset | 444.00 ms | 467.80 ms |

Speech end is the last microphone PCM block above the amplitude threshold.
Audio onset is the first qualifying signal observed in the browser’s Web
Audio graph plus its output-latency estimate. These are instrumented estimates
and include scripted transcript review where applicable. There were no
metered speech or model charges.

The browser checks verified stopping recording and playback, independent
connection failures during transcription and synthesis, preservation of
the text answer after a synthesis failure, and a typed follow-up that used the
spoken turn’s history. See [raw voice timings and transcripts](voice-results.json).
With voice enabled and Playwright available to Node, run:

```sh
node refdesk/tests/voice-browser.cjs
```
