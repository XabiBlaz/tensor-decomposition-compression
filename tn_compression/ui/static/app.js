"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { presets: [], jobs: [], config: {}, selected: null, job: null, polling: false, pending: false, uploading: false, targetEdited: false, valid: true };
  const activeStatuses = new Set(["queued", "running"]);
  const taskNames = { classification: "Image classification", segmentation: "Image segmentation", detection: "Object detection", causal_lm: "Language modeling" };
  const metricNames = { loss: "Validation loss", nll: "Token NLL", accuracy: "Accuracy", top1: "Top-1 accuracy", mean_iou: "Mean IoU", mean_dice: "Mean Dice", ap: "Average precision" };
  const rateMetrics = new Set(["accuracy", "top1", "mean_iou", "mean_dice", "ap"]);
  const resourceNames = { parameters: "Parameters", tensor_bytes: "Tensor size (bytes)", bundle_file_bytes: "Serialized bundle (bytes)", latency_mean_ms: "Mean latency (ms)", latency_p50_ms: "P50 latency (ms)", latency_p95_ms: "P95 latency (ms)", rss_load_peak_bytes: "Peak load RSS (bytes)", rss_inference_peak_bytes: "Peak inference RSS (bytes)", cuda_allocated_peak_bytes: "Peak CUDA allocation (bytes)", cuda_reserved_peak_bytes: "Peak CUDA reservation (bytes)", output_tokens_per_second: "Output tokens / second" };
  const allowedMetrics = { classification: ["loss", "accuracy", "top1"], segmentation: ["loss", "mean_iou", "mean_dice"], detection: ["ap"], causal_lm: ["nll"] };
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const show = (id, visible) => { $(id).hidden = !visible; };
  const humanize = (value) => String(value || "").replaceAll("_", " ");
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const format = (value) => finite(value) ? value.toLocaleString(undefined, Number.isInteger(value) ? { maximumFractionDigits: 0 } : { maximumSignificantDigits: 6 }) : "—";
  const mib = (value) => finite(value) ? `${(value / 1048576).toLocaleString(undefined, { maximumFractionDigits: 2 })} MiB` : "Not measured";
  const formatMetric = (key, value) => rateMetrics.has(key) && finite(value) ? `${format(100 * value)}%` : format(value);
  const formatChange = (key, value) => !finite(value) ? "—" : rateMetrics.has(key) ? `${value > 0 ? "+" : ""}${format(100 * value)} points` : `${value > 0 ? "+" : ""}${format(value)}`;
  const isDemo = (config) => config?.data?.kind === "synthetic" || config?.data?.kind === "fake" || /synthetic|smoke|demo/.test(String(config?.model?.name || ""));
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }
  function error(message) { $("global-error").textContent = message || ""; show("global-error", Boolean(message)); }
  async function api(path, options = {}) {
    const response = await fetch(path, { ...options, headers: { "Content-Type": "application/json", ...options.headers }, signal: AbortSignal.timeout(15000) });
    const text = await response.text();
    let data;
    try { data = text ? JSON.parse(text) : {}; } catch { throw new Error(`The server returned an unreadable response (${response.status}). Check the server log and retry.`); }
    if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : typeof data.detail === "string" ? data.detail : `Request failed (${response.status}). Check the configuration and server log.`);
    return data;
  }
  function setConnection(connected) { $("connection").textContent = connected ? "Connected to local workspace" : "Workspace unavailable · retrying"; $("connection-dot").classList.toggle("online", connected); }
  function validateConfig(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Configuration must be a JSON object.");
    if (!value.model || typeof value.model !== "object" || Array.isArray(value.model)) throw new Error("Configuration needs a model object.");
    if (!value.task || typeof value.task !== "string") throw new Error("Configuration needs a task name.");
    if (value.analysis && (typeof value.analysis !== "object" || Array.isArray(value.analysis))) throw new Error("analysis must be a JSON object.");
    return value;
  }
  function configError(message) { state.valid = !message; $("config-error").textContent = message || ""; show("config-error", Boolean(message)); $("config").setAttribute("aria-invalid", String(Boolean(message))); updateSubmit(); }
  const mode = () => document.querySelector('input[name="mode"]:checked').value;
  const isLanguage = () => state.config?.task === "causal_lm";
  const isLanguageStudy = () => isLanguage() && Boolean(state.config?.language_study);
  function updateSubmit() {
    const config = state.config, kind = config.data?.kind, language = isLanguage();
    const missing = !state.valid ? "Fix the Advanced configuration JSON before starting." :
      !state.presets.length ? "Loading model choices…" :
      !config.model?.name ? "Enter the model architecture or repository." :
      !kind ? "Choose evaluation data." :
      !language && (kind !== "synthetic" || $("preset").value === "trained-synthetic") && !config.model.state_dict_path && !$("checkpoint").value.trim() ? "Upload trained weights, or enter an existing model bundle." :
      !language && ["image_folder", "oxford_pet"].includes(kind) && !$("data-root").value.trim() ? "Enter the dataset folder under /data." :
      language && (!$("tokenizer-name").value.trim() || !/^[0-9a-f]{40}$/.test($("model-revision").value) || !/^[0-9a-f]{40}$/.test($("tokenizer-revision").value)) ? "Enter a tokenizer and full 40-character model and tokenizer versions." :
      mode() === "compress" && !isLanguageStudy() && !["method-tensor", "method-quantization", "method-pruning"].some((id) => $(id).checked) ? "Choose at least one compression method." : "";
    $("submit").disabled = state.pending || state.uploading || !state.valid || Boolean(missing);
    $("submit").firstChild.textContent = state.pending ? "Starting run… " : state.uploading ? "Uploading weights… " : mode() === "compress" && isLanguageStudy() ? "Run fixed study " : mode() === "compress" ? "Compress & compare " : "Inspect layers ";
    $("readiness").textContent = missing || (isDemo(config) ? "Ready for a synthetic workflow demo. Its quality numbers do not describe a trained model." : language ? $("allow-download").checked ? "Ready to start. Missing pinned Hugging Face files may be downloaded." : "Ready to start if the pinned Hugging Face files are cached. Otherwise, enable downloads." : "Ready to run. Quality will be checked on your selected data.");
    $("readiness").classList.toggle("warning", Boolean(missing) || isDemo(config));
    $("run-summary").textContent = `${mode() === "compress" ? "Compress and compare" : "Inspect layers in"} ${config.model?.name || "your model"} · ${taskNames[config.task] || "Choose a task"} · ${$("device").value.toUpperCase()}`;
  }
  function metricHelp() { const metric = $("metric").value, lower = ["loss", "nll"].includes(metric); $("quality-label").textContent = lower ? "Maximum allowed increase" : "Maximum allowed decrease"; $("metric-help").textContent = lower ? "The compressed model's loss may rise by at most this amount." : "For a metric on a 0–1 scale, 0.01 means one percentage point."; }
  function syncFields() {
    const config = state.config, analysis = config.analysis || {};
    $("task").value = taskNames[config.task] || config.task;
    $("task-label").textContent = $("task").value;
    $("model").value = config.model?.name || "";
    show("vision-fields", !isLanguage()); show("language-fields", isLanguage());
    $("model-label").textContent = isLanguage() ? "Hugging Face model repository" : "Model architecture";
    $("model-help").textContent = isLanguage() ? "For example, Qwen/Qwen2.5-0.5B. Pin its version below." : config.model?.source === "smp" ? "Enter the matching SMP architecture, such as Unet." : "Enter the TorchVision architecture matching your weights, such as resnet18.";
    $("num-classes").value = config.num_classes ?? config.model?.kwargs?.num_classes ?? config.model?.kwargs?.classes ?? "";
    $("data-kind").value = config.data?.kind || "synthetic";
    $("data-kind").querySelector('option[value="image_folder"]').disabled = config.task !== "classification";
    $("data-root").value = config.data?.root || "";
    show("data-root-field", ["image_folder", "oxford_pet"].includes(config.data?.kind));
    $("upload-status").textContent = config.model?.state_dict_path ? `Uploaded weights selected: ${config.model.state_dict_path}` : "Use a state_dict or a checkpoint containing one. For a full pickled model, first run the trusted conversion script in your original Python environment.";
    $("model-revision").value = config.model?.revision || "";
    $("tokenizer-name").value = config.tokenizer?.name || config.model?.name || "";
    $("tokenizer-revision").value = config.tokenizer?.revision || "";
    const methods = config.recipe?.methods || [];
    $("method-tensor").checked = methods.includes("tensor_decomposition");
    $("method-quantization").checked = methods.includes("quantization");
    $("method-pruning").checked = methods.includes("pruning");
    $("svd-energy").value = config.recipe?.svd_energy ?? 0.95;
    $("pruning-retention").value = config.recipe?.pruning_retention ?? 0.8;
    updateModeFields();
    $("target").value = analysis.target_size_mb ?? "";
    $("size-details").open = $("target").value !== "";
    $("quality").value = analysis.max_quality_loss ?? 0.05;
    const options = [...(allowedMetrics[config.task] || ["loss"])];
    if (analysis.quality_metric && !options.includes(analysis.quality_metric)) options.push(analysis.quality_metric);
    $("metric").replaceChildren(...options.map((metric) => { const option = element("option", metricNames[metric] || metric); option.value = metric; return option; }));
    $("metric").value = analysis.quality_metric || options[0];
    show("demo-warning", isDemo(config));
    metricHelp();
  }
  function updateModeFields() {
    show("recipe-fields", mode() === "compress" && !isLanguageStudy());
    show("language-study-fields", mode() === "compress" && isLanguageStudy());
    show("goal-fields", mode() === "compress");
    show("analyze-explainer", mode() === "analyze");
    show("download-option", isLanguage());
    $("options-title").textContent = mode() === "compress" && isLanguageStudy() ? "Run the bounded comparison" : mode() === "compress" ? "Choose compression methods" : "Inspect layer importance";
    $("quality").required = mode() === "compress";
    const pruningSupported = isLanguage() && /(?:llama|qwen)/i.test(state.config.model?.name || "");
    $("method-pruning").disabled = !pruningSupported;
    $("pruning-help").textContent = pruningSupported ? "Supported Llama/Qwen gated MLPs only; the run checks actual model capability." : "Available only for supported Llama/Qwen gated MLPs.";
    show("svd-field", $("method-tensor").checked);
    show("pruning-field", $("method-pruning").checked && pruningSupported);
    $("recovery").querySelector('option[value="finetune"]').disabled = isLanguage();
    $("recovery").querySelector('option[value="lora"]').disabled = !isLanguage();
    if (isLanguage() && $("recovery").value === "finetune" || !isLanguage() && $("recovery").value === "lora") $("recovery").value = "none";
  }
  function writeConfig() { $("config").value = JSON.stringify(state.config, null, 2); configError(""); }
  function readConfig() {
    try { const value = validateConfig(JSON.parse($("config").value)); if (value.analysis?.target_size_mb !== state.config.analysis?.target_size_mb) state.targetEdited = true; state.config = value; configError(""); syncFields(); updateSubmit(); return true; }
    catch (exception) { configError(`Invalid configuration: ${exception.message}`); return false; }
  }
  function guidedChange() {
    if (!state.valid) { $("advanced").open = true; $("config").focus(); return; }
    state.config.model.name = $("model").value.trim();
    if (isLanguage()) {
      state.config.model.revision = $("model-revision").value.trim();
      state.config.tokenizer = { ...state.config.tokenizer, name: $("tokenizer-name").value.trim(), revision: $("tokenizer-revision").value.trim() };
    } else {
      const classes = Number($("num-classes").value);
      if (Number.isInteger(classes) && classes > 0) {
        state.config.num_classes = classes;
        state.config.model.kwargs = { ...state.config.model.kwargs };
        if (state.config.model.source === "smp") state.config.model.kwargs.classes = classes;
        else state.config.model.kwargs.num_classes = classes;
      }
      const kind = $("data-kind").value, previousKind = state.config.data?.kind;
      if (kind && kind !== previousKind) {
        state.config.data = kind === "image_folder" ? { kind, root: $("data-root").value.trim(), size: 224, batch_size: 8 } : kind === "oxford_pet" ? { kind, root: $("data-root").value.trim(), size: 256, batch_size: 4 } : { kind: "synthetic", count: 8, size: 32, batch_size: 2 };
        if (kind === "image_folder") state.config.benchmark = { ...state.config.benchmark, input_shape: [1, 3, 224, 224] };
        if (previousKind === "synthetic" && kind !== "synthetic" && $("checkpoint").value) {
          $("checkpoint").value = "";
          $("preset-description").textContent = "Real data selected. Upload matching trained weights or enter a trained bundle path.";
        }
      } else if (["image_folder", "oxford_pet"].includes(kind)) state.config.data.root = $("data-root").value.trim();
    }
    state.config.analysis = { ...state.config.analysis, quality_metric: $("metric").value };
    if ($("target").value !== "") state.config.analysis.target_size_mb = Number($("target").value); else delete state.config.analysis.target_size_mb;
    if ($("quality").value !== "") state.config.analysis.max_quality_loss = Number($("quality").value); else delete state.config.analysis.max_quality_loss;
    if (mode() === "compress" && !isLanguageStudy()) {
      const methods = [["method-tensor", "tensor_decomposition"], ["method-quantization", "quantization"], ["method-pruning", "pruning"]].filter(([id]) => $(id).checked).map(([, name]) => name);
      const recipe = { ...state.config.recipe, methods, svd_energy: Number($("svd-energy").value), pruning_retention: Number($("pruning-retention").value), quantization_bits: 8 };
      if (!("include" in recipe) && Array.isArray(state.config.analysis.include) && state.config.analysis.include.length) recipe.include = [...state.config.analysis.include];
      state.config.recipe = recipe;
    }
    writeConfig(); metricHelp(); show("demo-warning", isDemo(state.config));
    updateModeFields(); show("data-root-field", ["image_folder", "oxford_pet"].includes($("data-kind").value)); updateSubmit();
  }
  function selectPreset() {
    const preset = state.presets.find((item) => item.id === $("preset").value);
    if (!preset) return;
    state.config = clone(preset.config); state.targetEdited = false; $("preset-description").textContent = preset.description || "";
    $("allow-download").checked = isLanguage();
    $("checkpoint").value = preset.checkpoint || state.config.model?.checkpoint || "";
    $("checkpoint-details").open = Boolean($("checkpoint").value);
    syncFields();
    if (mode() === "compress" && !isLanguageStudy() && !state.config.recipe?.methods?.length) {
      $("method-tensor").checked = true;
      if (isLanguage()) $("target").value = "";
    }
    writeConfig(); guidedChange();
  }
  function uploadWeights(file) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("POST", "/api/uploads");
      request.setRequestHeader("Content-Type", "application/octet-stream");
      request.setRequestHeader("X-Filename", `weights${file.name.slice(file.name.lastIndexOf("."))}`);
      request.upload.onprogress = (event) => {
        if (event.lengthComputable) $("upload-progress").value = Math.round(100 * event.loaded / event.total);
        $("upload-status").textContent = `Uploading ${file.name}: ${event.lengthComputable ? `${Math.round(100 * event.loaded / event.total)}%` : mib(event.loaded)}`;
      };
      request.onload = () => {
        let result;
        try { result = JSON.parse(request.responseText); } catch { reject(new Error(`The upload returned an unreadable response (${request.status}).`)); return; }
        if (request.status < 200 || request.status >= 300) reject(new Error(result.error || `Upload failed (${request.status}).`));
        else if (typeof result.path !== "string") reject(new Error("The upload response did not include a path."));
        else resolve(result.path);
      };
      request.onerror = () => reject(new Error("The connection failed during upload. Retry the file."));
      request.onabort = () => reject(new Error("The upload was cancelled."));
      request.send(file);
    });
  }
  function showSetup() { state.selected = null; state.job = null; show("setup", true); show("run-detail", false); error(""); renderHistory(); }
  function renderHistory() {
    if (!state.jobs.length) { $("history").replaceChildren(element("p", "Your first run starts here. Completed runs will appear in this list.", "muted")); return; }
    $("history").replaceChildren(...state.jobs.map((job) => {
      const button = element("button", undefined, `history-item${state.selected === job.id ? " active" : ""}`); button.type = "button";
      button.setAttribute("aria-current", state.selected === job.id ? "true" : "false");
      const title = job.config?.model?.name || job.model_name || job.name || `Run ${String(job.id).slice(0, 8)}`;
      const date = new Date(job.created_at);
      button.append(element("strong", title), element("small", Number.isNaN(date.valueOf()) ? job.id : date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })), element("span", humanize(job.status), `history-state ${safeStatus(job.status)}`));
      button.addEventListener("click", () => selectJob(job.id)); return button;
    }));
  }
  function safeStatus(value) { return ["queued", "running", "completed", "infeasible", "failed", "interrupted"].includes(value) ? value : "unknown"; }
  async function refreshJobs() { const result = await api("/api/jobs"); state.jobs = result.jobs || []; renderHistory(); setConnection(true); }
  async function selectJob(id) {
    state.selected = id; show("setup", false); show("run-detail", true); renderHistory();
    $("run-title").textContent = "Loading run…"; show("results", false); show("artifacts-panel", false); show("run-error", false); show("run-demo-warning", false); $("logs").textContent = "Loading worker output…"; $("reuse").disabled = true;
    try { const job = await api(`/api/jobs/${encodeURIComponent(id)}`); if (state.selected !== id) return; state.job = job; renderJob(job); error(""); }
    catch (exception) { error(exception.message); }
  }
  function resultCard(label, value, note) { const card = element("div", undefined, "result-card"); card.append(element("span", label), element("strong", value), element("p", note)); return card; }
  function metricRows(before, after) {
    const names = [...new Set([...Object.keys(before || {}), ...Object.keys(after || {})])].filter((key) => finite(before?.[key]) || finite(after?.[key]));
    $("metrics-body").replaceChildren(...names.map((key) => {
      const row = element("tr"); const delta = finite(before?.[key]) && finite(after?.[key]) ? after[key] - before[key] : null;
      row.append(element("td", metricNames[key] || resourceNames[key] || humanize(key)), element("td", formatMetric(key, before?.[key])), element("td", formatMetric(key, after?.[key])), element("td", formatChange(key, delta))); return row;
    }));
    if (!names.length) emptyRow("metrics-body", 4, "No evaluation metrics available yet.");
  }
  function emptyRow(id, columns, message) { const row = element("tr"), cell = element("td", message, "empty-row"); cell.colSpan = columns; row.append(cell); $(id).append(row); }
  function renderRelevance(relevance) {
    show("relevance-panel", Boolean(relevance));
    if (!relevance) return;
    const layers = [...(relevance.layers || [])].sort((a, b) => (b.relative_relevance || 0) - (a.relative_relevance || 0));
    $("relevance-count").textContent = `${layers.length} layers · ${relevance.batches ?? 0} batches`;
    $("relevance-note").textContent = `${relevance.metric_label || "Mean |weight × loss gradient| per parameter"} on ${format(relevance.units)} ${relevance.unit_label || "labeled units"}. Higher bars show greater local loss sensitivity.`;
    $("relevance-bars").replaceChildren(...layers.map((layer) => {
      const row = element("div", undefined, "relevance-row");
      const head = element("div", undefined, "relevance-row-head");
      head.append(element("span", layer.layer_path || "Unknown layer"), element("strong", `${format(100 * Math.max(0, Math.min(1, layer.relative_relevance || 0)))}%`));
      const track = element("div", undefined, "relevance-track"), bar = element("div", undefined, "relevance-bar");
      bar.style.width = `${100 * Math.max(0, Math.min(1, layer.relative_relevance || 0))}%`;
      track.append(bar);
      row.append(head, track, element("small", `Saliency ${format(layer.taylor_saliency)} · gradient RMS ${format(layer.gradient_rms)}`));
      return row;
    }));
  }
  function renderDirect(direct) {
    show("direct-panel", Boolean(direct));
    if (!direct) return;
    $("direct-summary").textContent = `${mib(direct.original_tensor_bytes)} original → ${mib(direct.compressed_tensor_bytes)} after compression. The final quality and runtime measurements appear below when verification completes.`;
    $("direct-stages").replaceChildren(...(direct.applied_methods || []).map((method) => element("span", humanize(method), "method-tag")));
  }
  function renderStudy(study) {
    show("study-panel", Boolean(study));
    if (!study) return;
    $("study-selection").textContent = study.selected_trial ? `Selected: ${study.selected_trial}` : "Selection pending";
    const baseline = study.baseline_validation;
    const rows = [];
    if (baseline) rows.push({ label: "Uncompressed baseline", status: "baseline", nll: baseline.nll, perplexity: baseline.perplexity });
    rows.push(...(study.trials || []));
    $("study-body").replaceChildren(...rows.map((trial) => {
      const row = element("tr");
      row.append(element("td", trial.label || trial.id), element("td", humanize(trial.status)),
        element("td", format(trial.nll)), element("td", format(trial.perplexity)),
        element("td", mib(trial.compressed_tensor_bytes)));
      return row;
    }));
    if (!rows.length) emptyRow("study-body", 5, "Trial results are not available yet.");
    $("study-note").textContent = study.selection_rule || "Candidates are selected on validation NLL before held-out test evaluation.";
  }
  function renderRecovered(recovered, comparison, quality) {
    show("recovered-panel", Boolean(recovered || comparison || quality));
    if (!recovered && !comparison && !quality) return;
    const metrics = recovered?.metrics || recovered?.evaluation?.metrics || recovered || {};
    const count = recovered?.example_ids?.length ?? recovered?.example_count;
    const verdict = quality ? `Quality ${quality.passed ? "passed" : "failed"}: ${metricNames[quality.metric] || quality.metric} ${format(quality.recovered)} after recovery vs. ${format(quality.original)} original (allowed loss ${format(quality.maximum_loss)}).` : comparison ? `Recovered artifact verification: ${humanize(comparison.status)}.` : "";
    $("recovered-note").textContent = `${humanize(recovered?.role || "Held-out")} evaluation${finite(count) ? ` · ${format(count)} examples` : ""}. ${verdict} ${quality?.limitation || ""}`;
    $("recovered-body").replaceChildren(...Object.entries(metrics).filter(([key, value]) => key in metricNames && finite(value)).map(([key, value]) => {
      const row = element("tr"); row.append(element("td", metricNames[key] || humanize(key)), element("td", formatMetric(key, value))); return row;
    }));
    if (!$("recovered-body").children.length) emptyRow("recovered-body", 2, "No recovered metrics available.");
  }
  function renderResults(job) {
    const analysis = job.results?.analysis, comparison = job.results?.comparison, relevance = job.results?.relevance, study = job.results?.study;
    const direct = Array.isArray(job.results?.direct?.applied_methods) ? job.results.direct : null;
    show("results", Boolean(analysis || comparison || relevance || direct || study)); if (!analysis && !comparison && !relevance && !direct && !study) return;
    const verdict = $("result-verdict");
    show("result-verdict", Boolean(comparison || relevance));
    if (comparison) {
      const before = comparison.resources?.bundle_file_bytes?.original, after = comparison.resources?.bundle_file_bytes?.compressed;
      const reduction = finite(before) && before > 0 && finite(after) ? ` Bundle file: ${format(Math.round(1000 * (before - after) / before) / 10)}% smaller.` : "";
      const synthetic = isDemo(job.config) || comparison.synthetic_baseline === true;
      verdict.textContent = `${comparison.quality?.passed === true ? "Compression met your quality limit." : comparison.quality?.passed === false ? "Compression failed your quality limit." : "Quality has not been verified."}${reduction}${synthetic ? " This is a synthetic workflow result, not real-world evidence." : ""}`;
      verdict.className = `notice verdict ${comparison.quality?.passed === false ? "error" : synthetic ? "warning" : "success"}`;
    } else if (relevance) {
      verdict.textContent = "Layer importance is ready below. This analysis did not compress the model.";
      verdict.className = "notice verdict";
    }
    renderRelevance(relevance); renderDirect(direct); renderStudy(study); renderRecovered(job.results?.recovered, job.results?.recovered_comparison, job.results?.recovery_quality);
    show("result-cards", Boolean(analysis || comparison || direct));
    show("metrics-panel", Boolean(analysis || comparison));
    const summary = analysis?.summary || {}, original = summary.original_tensor_bytes, compressed = summary.estimated_final_tensor_bytes;
    const savings = finite(original) && original > 0 && finite(compressed) ? 100 * (original - compressed) / original : null;
    $("result-cards").replaceChildren(resultCard("Original tensor size", mib(original), "Model tensors before compression"), resultCard("Estimated compressed tensors", mib(compressed), finite(savings) ? `${format(savings)}% smaller · not serialized file size` : "Available after validated analysis"), resultCard("Target size", summary.target_reached === true ? "Reached" : summary.target_reached === false ? "Not reached" : "Not assessed", `${summary.accepted ?? 0} accepted candidates · ${summary.rejected ?? 0} rejected`));
    if (direct && !comparison) $("result-cards").replaceChildren(
      resultCard("Original tensor size", mib(direct.original_tensor_bytes), "Measured before the selected methods"),
      resultCard("Compressed tensor size", mib(direct.compressed_tensor_bytes), `${mib(direct.tensor_bytes_saved)} saved`),
      resultCard("Methods applied", (direct.applied_methods || []).length, (direct.applied_methods || []).map(humanize).join(" · ") || "Applying selected methods")
    );
    if (comparison) {
      const resources = comparison.resources || {}, quality = comparison.quality || {}, measurements = { ...(comparison.evaluation?.metrics || {}), ...resources };
      const before = {}, after = {};
      for (const [key, row] of Object.entries(measurements)) { before[key] = row.original; after[key] = row.compressed; }
      metricRows(before, after);
      $("metrics-source").textContent = `${humanize(comparison.evaluation?.role || "Final")} evaluation · ${comparison.evaluation?.example_count ?? 0} examples`;
      $("metrics-note").textContent = `Final artifact measurements. ${comparison.size_target_scope || "Tensor size and serialized file size are distinct measurements."} Results apply only to the recorded data and benchmark workload.`;
      const measuredBytes = resources.bundle_file_bytes?.compressed ?? resources.tensor_bytes?.compressed;
      const originalBytes = resources.bundle_file_bytes?.original ?? resources.tensor_bytes?.original;
      const latency = resources.latency_mean_ms;
      const latencyNote = finite(latency?.original) ? `Original: ${format(latency.original)} ms · recorded workload` : "Runtime needs separate measurement";
      $("result-cards").replaceChildren(
        resultCard(resources.bundle_file_bytes ? "Serialized compressed bundle" : "Measured compressed tensors", mib(measuredBytes), `Original: ${mib(originalBytes)}`),
        resultCard("Quality constraint", quality.passed === true ? "Passed" : quality.passed === false ? "Failed" : "Not verified", `${metricNames[quality.metric] || quality.metric || "Metric"} change: ${rateMetrics.has(quality.metric) ? `${format(100 * quality.loss)} points` : format(quality.loss)} · limit: ${rateMetrics.has(quality.metric) ? `${format(100 * quality.maximum_loss)} points` : format(quality.maximum_loss)}`),
        resultCard("Mean inference latency", finite(latency?.compressed) ? `${format(latency.compressed)} ms` : "Not measured", latencyNote)
      );
    } else if (analysis) {
      metricRows(analysis?.baseline_metrics, analysis?.cumulative_metrics);
      $("metrics-source").textContent = "Analysis validation";
      $("metrics-note").textContent = "These metrics describe cumulative candidate validation, not a final artifact check. Run compression for held-out and reload evidence. Size savings do not establish speed improvements.";
    }
    const limitations = [...(analysis?.limitations || []), ...(relevance?.limitations || []), ...(comparison?.reasons || []), ...(direct?.caveat ? [direct.caveat] : [])]; $("limitations").replaceChildren(...limitations.map((item) => element("p", item))); show("limitations", limitations.length > 0);
  }
  function renderJob(job) {
    $("reuse").disabled = !job.config;
    $("run-title").textContent = job.config?.model?.name || job.model_name || "Compression run";
    $("run-id").textContent = `RUN ${job.id}`;
    $("run-subtitle").textContent = [taskNames[job.config?.task] || job.config?.task, job.device, job.mode === "compress" ? "Compression comparison" : "Layer importance"].filter(Boolean).join(" · ");
    $("run-status").textContent = humanize(job.status); $("run-status").className = `status ${safeStatus(job.status)}`;
    $("run-stage").textContent = humanize(job.stage || job.status || "Waiting");
    $("activity-indicator").classList.toggle("running", activeStatuses.has(job.status));
    $("run-stage-help").textContent = activeStatuses.has(job.status) ? "Running locally. You can leave this page and return to the results." : job.status === "infeasible" ? "No plan met the requested constraints. Review the evidence and adjust your goals." : job.status === "completed" && job.mode === "analyze" ? "Layer relevance is ready. These scores are diagnostic; compressed quality has not been verified." : job.status === "completed" ? "Run complete. Review quality and resource measurements before using the model." : "Review the execution log for details.";
    show("run-demo-warning", isDemo(job.config) || job.results?.comparison?.synthetic_baseline === true);
    $("run-error").textContent = job.error || ""; show("run-error", Boolean(job.error));
    const logs = $("logs"), nearBottom = logs.scrollTop + logs.clientHeight >= logs.scrollHeight - 40; logs.textContent = job.logs || "Waiting for worker output…"; if (nearBottom) logs.scrollTop = logs.scrollHeight;
    $("artifacts").replaceChildren();
    for (const artifact of job.artifacts || []) {
      try { const url = new URL(artifact.url, window.location.origin); if (url.origin !== window.location.origin || !["http:", "https:"].includes(url.protocol)) continue; const link = element("a", `${artifact.name} ↓`, "artifact-link"); link.href = url.href; link.download = ""; $("artifacts").append(link); } catch { /* Invalid artifact links are not rendered. */ }
    }
    show("artifacts-panel", $("artifacts").children.length > 0); renderResults(job);
  }
  async function poll() {
    if (state.polling || document.hidden) return;
    state.polling = true;
    try {
      await refreshJobs();
      const id = state.selected;
      if (id) { const job = await api(`/api/jobs/${encodeURIComponent(id)}`); if (state.selected === id) { state.job = job; renderJob(job); } }
    } catch { setConnection(false); } finally { state.polling = false; }
  }
  $("preset").addEventListener("change", selectPreset);
  $("model").addEventListener("change", () => {
    const previous = state.config.model?.name || "", next = $("model").value.trim();
    if (next !== previous && isLanguage()) {
      if ($("tokenizer-name").value.trim() === previous) $("tokenizer-name").value = next;
      $("model-revision").value = ""; $("tokenizer-revision").value = "";
      error("Model name changed. Enter the model and tokenizer commit revisions for the new repository.");
    }
    if (next !== previous && !isLanguage() && state.config.model?.state_dict_path) {
      delete state.config.model.state_dict_path;
      $("upload-status").textContent = "Architecture changed. Upload weights that match this architecture.";
    }
    guidedChange();
  });
  $("target").addEventListener("change", () => { state.targetEdited = true; guidedChange(); });
  ["checkpoint", "device", "allow-download"].forEach((id) => $(id).addEventListener("change", updateSubmit));
  ["quality", "metric", "num-classes", "data-kind", "data-root", "model-revision", "tokenizer-name", "tokenizer-revision", "svd-energy", "pruning-retention"].forEach((id) => $(id).addEventListener("change", guidedChange));
  ["method-tensor", "method-quantization", "method-pruning"].forEach((id) => $(id).addEventListener("change", guidedChange));
  $("config").addEventListener("input", readConfig);
  document.querySelectorAll('input[name="mode"]').forEach((input) => input.addEventListener("change", () => {
    if (mode() === "compress" && !isLanguageStudy() && !state.config.recipe?.methods?.length) {
      $("method-tensor").checked = true;
      if (isLanguage() && $("preset").value === "language" && !state.targetEdited) $("target").value = "";
    }
    if (mode() === "analyze" && !$("target").value && !state.targetEdited) $("target").value = state.presets.find((preset) => preset.id === $("preset").value)?.config?.analysis?.target_size_mb ?? "";
    updateModeFields(); guidedChange(); updateSubmit();
  }));
  $("weights-file").addEventListener("change", async (event) => {
    const file = event.target.files[0]; if (!file) return;
    if (!/\.(pt|pth)$/i.test(file.name) || file.size < 1 || file.size > 2 * 1024 ** 3) { error("Choose a nonempty .pt or .pth weights file up to 2 GiB."); event.target.value = ""; return; }
    state.uploading = true; updateSubmit(); error(""); $("upload-progress").value = 0; show("upload-progress", true);
    try {
      const path = await uploadWeights(file);
      state.config.model.state_dict_path = path; state.config.model.weights = null;
      $("checkpoint").value = "";
      writeConfig();
      $("upload-status").textContent = `${file.name} uploaded. The run will load its state_dict into the selected architecture.`;
    } catch (exception) { error(exception.message); $("upload-status").textContent = "Upload failed. Select the file again to retry."; }
    finally { state.uploading = false; updateSubmit(); show("upload-progress", false); event.target.value = ""; }
  });
  $("new-run").addEventListener("click", showSetup);
  $("refresh").addEventListener("click", async () => { try { await refreshJobs(); error(""); } catch (exception) { error(exception.message); setConnection(false); } });
  $("reuse").addEventListener("click", () => {
    if (!state.job?.config) return;
    const job = state.job; state.config = clone(job.config); state.targetEdited = true; $("preset").value = job.preset || $("preset").value; $("checkpoint").value = job.checkpoint || ""; $("device").value = job.device || "cpu"; $("allow-download").checked = Boolean(job.allow_download); $("recovery").value = job.recovery || "none"; document.querySelector(`input[name="mode"][value="${job.mode || "analyze"}"]`).checked = true;
    $("preset-description").textContent = "Configuration copied from an existing run. Review paths and goals before starting.";
    $("checkpoint-details").open = Boolean($("checkpoint").value);
    syncFields(); writeConfig(); updateSubmit(); showSetup();
  });
  $("import-config").addEventListener("change", async (event) => {
    const file = event.target.files[0]; if (!file) return;
    try { if (file.size > 200 * 1024) throw new Error("Configuration files must be smaller than 200 KiB."); const value = validateConfig(JSON.parse(await file.text())); state.config = value; state.targetEdited = true; $("checkpoint").value = ""; $("checkpoint-details").open = false; document.querySelector(`input[name="mode"][value="${value.recipe ? "compress" : "analyze"}"]`).checked = true; syncFields(); writeConfig(); updateSubmit(); $("preset-description").textContent = `Imported configuration: ${file.name}`; error(""); } catch (exception) { error(`Could not import configuration: ${exception.message}`); } finally { event.target.value = ""; }
  });
  $("download-config").addEventListener("click", () => { if (!readConfig()) return; const url = URL.createObjectURL(new Blob([JSON.stringify(state.config, null, 2) + "\n"], { type: "application/json" })); const link = element("a"); link.href = url; link.download = "compression-config.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
  $("run-form").addEventListener("submit", async (event) => {
    event.preventDefault(); if (!state.valid) { $("advanced").open = true; $("config").focus(); return; } guidedChange();
    if (isLanguage()) {
      const remote = (value) => /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(value);
      const commit = (value) => /^[0-9a-f]{40}$/.test(value);
      if ((remote(state.config.model.name) && !commit(state.config.model.revision)) || (remote(state.config.tokenizer?.name) && !commit(state.config.tokenizer?.revision))) { error("Pin both model and tokenizer to full 40-character commit revisions."); return; }
    }
    if (mode() === "compress" && !isLanguageStudy() && !state.config.recipe?.methods?.length) { error("Select at least one compression method."); return; }
    if (state.config.recipe?.methods?.includes("pruning") && (mode() === "compress") && $("method-pruning").disabled) { error("Physical pruning requires a supported Llama/Qwen gated MLP model."); return; }
    state.pending = true; updateSubmit(); error("");
    try { const job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ preset: $("preset").value, config: state.config, checkpoint: $("checkpoint").value.trim() || null, device: $("device").value, mode: mode(), allow_download: $("allow-download").checked, recovery: mode() === "compress" ? $("recovery").value : "none" }) }); await selectJob(job.id); await refreshJobs(); }
    catch (exception) { error(exception.message); } finally { state.pending = false; updateSubmit(); }
  });
  async function initialize() {
    $("history-drawer").open = window.matchMedia("(min-width: 981px)").matches;
    const outcomes = await Promise.allSettled([api("/api/presets"), api("/api/jobs"), api("/api/health")]);
    if (outcomes[0].status === "fulfilled") { state.presets = outcomes[0].value.presets || []; $("preset").replaceChildren(...state.presets.map((preset) => { const option = element("option", preset.name); option.value = preset.id; return option; })); $("preset").disabled = !state.presets.length; if (state.presets.length) selectPreset(); else error("No presets are available. Check the server configuration."); } else error(`Could not load presets: ${outcomes[0].reason.message} Reload the page to retry.`);
    if (outcomes[1].status === "fulfilled") { state.jobs = outcomes[1].value.jobs || []; renderHistory(); } else $("history").replaceChildren(element("p", "Run history unavailable. Use refresh to retry.", "muted"));
    setConnection(outcomes[2].status === "fulfilled"); updateSubmit(); setInterval(poll, 3000);
  }
  document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
  initialize();
})();
