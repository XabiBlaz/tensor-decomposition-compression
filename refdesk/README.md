# Refdesk

Refdesk is a local web chat for asking technical questions about a public GitHub
repository. It shows the source and revision behind each answer, uses the last
question to resolve simple follow-ups, and asks for context when the evidence is
thin. You can refresh sources, revisit questions after a reload, flag an
unhelpful answer, and copy an editable GitHub issue draft.

## Run text chat

Use Python 3.9+ and Git. From the repository root, run:

```sh
python -m refdesk
```

Open <http://localhost:3000>. For a different port, use
`python -m refdesk --port 3010`. The server listens on loopback. For a remote
development machine, forward the chosen port over SSH and use localhost in
your browser. The compression toolkit and its dependencies are optional for
this app.

Enter `https://github.com/XabiBlaz/tensor-decomposition-compression`, then try:

- “How do I start the Docker Compose UI?” followed by “What about CUDA?”
- “Which checksum do model bundles use?”
- “Does the 0.0.3 Docker Compose UI still use port 7860?”

Or index `https://github.com/BurntSushi/ripgrep` and ask about hidden files,
ignore rules, or `RIPGREP_CONFIG_PATH`.

The default provider quotes matching passages and costs no model tokens.
For synthesis, set `REFDESK_MODEL` to a model that produces JSON
through a Chat Completions API:

| Variable | Purpose |
| --- | --- |
| `REFDESK_MODEL` | Optional model name |
| `REFDESK_MODEL_URL` | Full API endpoint, defaults to the OpenAI Chat Completions endpoint |
| `REFDESK_MODEL_KEY` | Optional server-side API key |
| `REFDESK_INPUT_USD_PER_M`, `REFDESK_OUTPUT_USD_PER_M` | Input and output rates per million tokens for cost reporting |

Model-assisted mode sends the question, recent questions, and retrieved
excerpts to the configured provider. The extractive mode has no model network
call. Stdout logs include source URLs, indexed revision, request errors,
retrieval and generation timing, token usage, and estimated cost when rates
and usage are available. Feedback logs the answer trace ID.

## Run optional local voice

Use a Chromium browser with microphone permission, FFmpeg built with the
`flite` filter, and the optional PocketSphinx English recognizer. Install it
in the Python environment that runs Refdesk.

```sh
python -m pip install 'pocketsphinx==5.0.4'
REFDESK_VOICE=1 python -m refdesk
```

Index a repository. Choose **Start voice**, speak, and choose **Finish
recording**. Review or edit the transcript, then choose **Ask aloud**.
Refdesk sends the transcript to the normal chat turn, shows the cited answer
in the same history, and plays the first 1,400 characters as local streamed
audio. The text retains the full answer and citation links. Recording auto-stops
after 15 seconds. The browser sends mono PCM to the local server for
in-memory recognition. Both speech adapters run without speech credentials.

**Stop voice** cancels recording, pending voice requests, or playback. If a
turn already finished, its text stays in the conversation. If transcription
fails, record again or type a question. If chat fails after transcription, the
confirmed question stays in the composer for retry. If synthesis fails, read
the answer in chat. Text chat stays usable when voice is off or dependencies
are unavailable.

Voice is a preview. PocketSphinx can misrecognize commands, identifiers, and
ordinary speech. The browser measures transcription, retrieval, generation,
synthesis startup, and estimated audio onset. Accuracy and timing were tested
with a synthetic microphone fixture on Chromium. See the
[voice evaluation](evaluation/README.md#voice-preview).

## Sources and limitations

The index reads UTF-8 Markdown, MDX, reStructuredText, AsciiDoc, plain text,
Org files, common extensionless support files, and common text/code file types
under example-like directories. It reads the default branch by commit and
links files to their original line ranges. The GitHub REST API adds up to
eight stable releases and selected bodies from the first twelve recently
updated closed issues, skipping pull requests. Issues are mutable and the
displayed date is the observed update date. When GitHub REST is unreachable
or rate-limited, the interface reports partial coverage.

The 24 MiB compressed archive limit, 192 KiB per-file limit, 100-file budget,
and 3,500-file-passage budget keep ingestion bounded. Repositories with sparse
docs, unsupported formats, large archives, or examples outside recognized
locations may yield little evidence. Refdesk currently serves English text.
Retrieval normalizes common English technical phrases and checks exact flags,
paths, environment variables, and version strings. The default extractive mode
still requires coverage of the normalized query before quoting a source. The
Click holdout found that relevant passages can rank well and still lead to
abstention.
Historical labels use changelog/archive conventions and GitHub activity types.
The app can surface competing excerpts on version-related questions, while
general contradiction detection and semantic citation verification remain
limited. Refreshes affect future answers. The browser keeps up to 40 turns
for the active repository in local storage, including the revision of each
answer. Switching repositories or starting a new chat clears that history.

## Evaluation and verification

```sh
python -m unittest discover -s refdesk/tests -v
python -m refdesk.evaluate --output /tmp/refdesk-evaluation.json
# With Refdesk running and Playwright available to Node:
node refdesk/tests/browser.cjs
```

Use `REFDESK_CHROMIUM=/path/to/chromium` to select a local Chromium and
`REFDESK_TEST_URL=http://localhost:3010` to select a running instance.
The benchmark checks that each live default branch matches the revision in
the case file. If a branch has advanced, review the expected sources before
updating the pin. See [the measured run](evaluation/README.md).

## System design and rationale

`github.py` validates public URLs, resolves default HEAD, downloads the
immutable archive without extracting it, and gathers both optional REST
sources. `sources.py` chooses files and chunks them with repository-relative
paths and original line ranges. Setup instructions and the code blocks they
introduce can stay in the same passage. `retrieval.py` normalizes English
phrasing and ranks by document frequency, coverage, section headings, exact
commands and version strings. Recent
questions can boost candidate passages for referential follow-ups. `chat.py`
owns abstention, historical labeling and citation validation.
`providers.py` supplies either a local extractive answer or optional synthesis.
The HTTP server serves static HTML, CSS, and JavaScript and keeps the last six
repository indexes in memory. A failed refresh leaves the previous index
available. Each turn uses a single snapshot throughout retrieval and answering.

| Choice | Alternatives and trade-off |
| --- | --- |
| Commit ZIP and optional GitHub REST calls | Reading each file through REST adds requests and rate limits. Shallow clones offer history and diff-based refresh but store more data and introduce a working tree to manage. |
| Per-repository lexical index | Sending every file to a model raises cost and context pressure. Embeddings can help paraphrases, but require evaluating another model and still need lexical matching for flags and paths. |
| Browser history and stateless turn requests | A SQLite conversation store would support cross-device sessions but needs authenticated ownership and retention controls. Browser persistence covers local reloads. |
| One local Python server | A web framework is useful when authentication, routing, and deployment complexity grow. Queueing ingestion or splitting services would add operational work before concurrent load has been measured. |
| Text-first transport | Voice uses the same turn request, with input transcription and output playback adapters. Shared turns preserve evidence, revision, abstention, and history regardless of input mode. |

Consider persistent storage once restarts or cache churn become material, and
hybrid retrieval if a held-out test set shows lexical recall is the bottleneck.
Measure concurrent ingestion before adding scheduling infrastructure.

### Private-repository boundary

A future authenticated loader should accept a server-verified principal, an
authorized GitHub installation, and a repository ID, and return the same
snapshot shape. Use short-lived, repository-scoped, read-only GitHub App
credentials for contents and issues. Authorize each index, refresh and turn,
partition caches by authorization scope and repository ID, and purge access
on revocation. Keep tokens server-side and out of traces. Private excerpts
require an approved provider policy and a history isolation model before
turns are enabled. This release accepts anonymous public URLs.
