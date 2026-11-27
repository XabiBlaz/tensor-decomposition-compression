import { connectVoice } from "./voice.js";

const $ = (id) => document.getElementById(id);
const question = $("question");
const urlInput = $("repository-url");
const empty = () => ({ repository: null, turns: [] });
let state = empty();
let pending = false;
let voiceActive = false;

try {
  const saved = JSON.parse(localStorage.getItem("refdesk-conversation"));
  if (saved?.repository && Array.isArray(saved.turns)) state = saved;
} catch {
  // Storage can be unavailable in a private or managed browser.
}

function save() {
  try {
    localStorage.setItem("refdesk-conversation", JSON.stringify(state));
  } catch {
    // Chat remains usable in memory.
  }
}

function element(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
}

function message(id, text, error = false) {
  const status = $(id);
  status.textContent = text;
  status.classList.toggle("error", error);
}

async function post(path, body, signal) {
  const response = await fetch(`/api/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: signal || AbortSignal.timeout(95000),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request failed.");
  return data;
}

function controls() {
  for (const id of ["ask", "refresh", "new-chat", "draft"]) {
    $(id).disabled = pending || voiceActive;
  }
  urlInput.form.querySelector("button").disabled = pending || voiceActive;
  $("new-chat").hidden = state.turns.length === 0;
  $("draft").hidden = state.turns.length === 0;
}

function sources(result) {
  const list = element("ul", undefined, "citations");
  list.setAttribute("aria-label", "Sources");
  for (const citation of result.citations) {
    const item = element("li");
    const link = element("a", `[${citation.id}] ${citation.path}`);
    link.href = citation.url;
    link.target = "_blank";
    link.rel = "noopener";
    const revision = citation.kind === "file" ? citation.revision.slice(0, 8) : citation.heading;
    link.append(element("small", citation.kind === "file" ? `${revision} / ${citation.heading}` : revision));
    item.append(link);
    list.append(item);
  }
  return list;
}

function answerText(text, citations) {
  const output = element("div", undefined, "answer-text");
  let buffer = [];
  let fence = null;
  function flush() {
    if (!buffer.length) return;
    const text = buffer.join("\n");
    if (fence) {
      const pre = element("pre");
      pre.append(element("code", text));
      output.append(pre);
    } else {
      const paragraph = element("p");
      for (const part of text.split(/(`[^`\n]+`|\[\d+\])/)) {
        const citation = citations.find((source) => part === `[${source.id}]`);
        if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
          paragraph.append(element("code", part.slice(1, -1)));
        } else if (citation) {
          const link = element("a", part);
          link.href = citation.url;
          link.target = "_blank";
          link.rel = "noopener";
          link.setAttribute("aria-label", `Source ${citation.id}: ${citation.path}`);
          paragraph.append(link);
        } else {
          paragraph.append(part);
        }
      }
      output.append(paragraph);
    }
    buffer = [];
  }
  for (const line of text.split("\n")) {
    const marker = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
    if (!fence && marker) {
      flush();
      fence = marker[1];
    } else if (fence && marker && marker[1].startsWith(fence) && !marker[2].trim()) {
      flush();
      fence = null;
    } else if (!fence && /^#{1,6}\s/.test(line)) {
      flush();
      output.append(element("h3", line.replace(/^#{1,6}\s+/, "")));
    } else if (!fence && !line.trim()) {
      flush();
    } else {
      buffer.push(line);
    }
  }
  flush();
  return output;
}

function renderTurn(turn) {
  const result = turn.result;
  const row = element("article", undefined, "turn");
  row.id = `turn-${result.id}`;
  const title = element("h2", turn.question);
  title.tabIndex = -1;
  row.append(title);
  const labels = {
    supported: result.provider === "extractive" ? "Source excerpt" : "Source-backed answer",
    historical: "Version check",
    insufficient: "Insufficient evidence",
  };
  const meta = element("div", `${labels[result.status]} / ${result.revision.slice(0, 8)}`, "answer-meta");
  meta.classList.toggle("uncertain", result.status !== "supported");
  row.append(meta, answerText(result.text, result.citations));
  if (result.citations.length) row.append(sources(result));
  const feedback = element("button", turn.flagged ? "Flagged as unhelpful" : "Flag as unhelpful", "feedback");
  feedback.type = "button";
  feedback.disabled = Boolean(turn.flagged);
  feedback.addEventListener("click", async () => {
    feedback.disabled = true;
    try {
      await post("flag", { id: result.id });
      turn.flagged = true;
      feedback.textContent = "Flagged as unhelpful";
      message("chat-status", "Feedback recorded.");
      save();
    } catch {
      feedback.disabled = false;
      message("chat-status", "Couldn’t record feedback. Try again.", true);
    }
  });
  row.append(feedback);
  $("thread").append(row);
  const item = element("li");
  const link = element("a", turn.question);
  link.href = `#${row.id}`;
  link.title = turn.question;
  link.addEventListener("click", () => title.focus());
  item.append(link);
  $("history").append(item);
  return row;
}

function render() {
  const repo = state.repository;
  $("repository-detail").hidden = !repo;
  if (repo) {
    $("repository-link").href = `https://github.com/${repo.repo}`;
    $("repository-link").textContent = repo.repo;
    $("active-repository").textContent = repo.repo;
    $("revision").textContent = `${repo.branch} / ${repo.revision.slice(0, 8)}`;
    $("revision").title = repo.revision;
    $("coverage").textContent = `${repo.files} files indexed. ${repo.warnings.join(" ")}`;
    $("mode").textContent = repo.provider === "extractive" ? "Text chat / source excerpts" : "Text chat / model-assisted";
  }
  $("thread").replaceChildren();
  $("history").replaceChildren();
  for (const turn of state.turns) renderTurn(turn);
  threadVisibility();
  controls();
}

function threadVisibility() {
  const emptyThread = state.turns.length === 0;
  $("welcome").hidden = !emptyThread;
  $("thread-title").hidden = emptyThread;
  $("history-empty").hidden = !emptyThread;
}

async function loadRepository(raw, refresh) {
  if (pending || voiceActive) return;
  pending = true;
  controls();
  message("repository-status", refresh ? "Refreshing GitHub sources…" : "Reading GitHub sources…");
  try {
    const repository = await post("repository", { url: raw, refresh });
    if (state.repository?.repo !== repository.repo) state.turns = [];
    state.repository = repository;
    urlInput.value = `https://github.com/${repository.repo}`;
    save();
    render();
    message("repository-status", "");
    message("chat-status", refresh ? "Sources refreshed. Earlier answers keep their original revision." : "Repository ready.");
    question.focus();
  } catch (error) {
    message("repository-status", error.message || "Couldn’t index this repository. Try again.", true);
  } finally {
    pending = false;
    controls();
  }
}

async function ask(text, signal) {
  text = text.trim();
  if (!text || pending) return null;
  if (!state.repository) {
    message("repository-status", "Index a public GitHub repository before asking.", true);
    urlInput.focus();
    return null;
  }
  pending = true;
  controls();
  message("chat-status", "Searching indexed sources…");
  const performing = $("chat-status");
  performing.scrollIntoView({ block: "nearest" });
  try {
    const result = await post("chat", {
      url: `https://github.com/${state.repository.repo}`,
      question: text,
      previous: state.turns.slice(-4).map((turn) => turn.question),
    }, signal);
    if (signal?.aborted) return null;
    const turn = { question: text, result };
    state.turns.push(turn);
    if (state.turns.length > 40) {
      state.turns.shift();
      $("thread").firstElementChild.remove();
      $("history").firstElementChild.remove();
    }
    if (question.value.trim() === text) question.value = "";
    threadVisibility();
    save();
    message("chat-status", result.status === "insufficient" ? "The answer requests more context." : "Answer ready.");
    const row = renderTurn(turn);
    if (result.revision !== state.repository.revision) {
      message("chat-status", `Answer used revision ${result.revision.slice(0, 8)}. Refresh sources to update the repository details.`);
    }
    requestAnimationFrame(() => row.scrollIntoView({ block: "start" }));
    return result;
  } catch (error) {
    if (!signal?.aborted) message("chat-status", error.message || "Connection lost. You can retry this question.", true);
    else message("chat-status", "Voice request interrupted.");
    return null;
  } finally {
    pending = false;
    controls();
  }
}

$("repository-form").addEventListener("submit", (event) => {
  event.preventDefault();
  loadRepository(urlInput.value, false);
});
$("refresh").addEventListener("click", () => loadRepository(`https://github.com/${state.repository.repo}`, true));
$("new-chat").addEventListener("click", () => {
  state.turns = [];
  save();
  render();
  message("chat-status", "");
  question.focus();
});
$("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!voiceActive) ask(question.value);
});
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});
for (const button of document.querySelectorAll("[data-example]")) {
  button.addEventListener("click", () => {
    question.value = button.textContent;
    question.focus();
  });
}
$("draft").addEventListener("click", () => {
  const last = state.turns.at(-1);
  const links = last.result.citations.map((citation) => `- ${citation.url}`);
  $("issue-body").value = [
    "## Question or problem", last.question, "", "## Version",
    `Indexed revision: ${last.result.revision}`, "", "## Steps to reproduce", "",
    "## Expected and actual behavior", "", "## Relevant sources", ...links,
  ].join("\n");
  message("copy-status", "");
  $("issue-dialog").showModal();
});
$("copy-issue").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("issue-body").value);
    message("copy-status", "Draft copied.");
  } catch {
    $("issue-body").select();
    message("copy-status", "Draft selected. Use your keyboard to copy it.");
  }
});
if (state.repository) urlInput.value = `https://github.com/${state.repository.repo}`;
render();

connectVoice({
  button: $("voice-toggle"),
  cancel: $("voice-cancel"),
  status: $("voice-status"),
  question,
  ask,
  ready: () => Boolean(state.repository && !pending),
  onActive: (active) => {
    voiceActive = active;
    controls();
  },
});
