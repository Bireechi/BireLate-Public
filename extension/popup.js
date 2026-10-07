const DEFAULTS = {
  enabledEverywhere: false,
  enabledHosts: [],
  autoTranslate: false,
  minSize: 150,
  corner: "top-left",
  serverUrl: "http://127.0.0.1:8765",
  targetLanguage: "en-US",
  // Must match background.js's DEFAULTS.ocr — that comment is the one home
  // for which engine is the default and why. The latch logic is
  // script-latch.js. A pointer rather than a paraphrase, so it cannot go
  // stale while the value beside it is updated.
  ocr: "hunyuan-ocr-1.5",
  inpainting: "lama",
  keepArt: false,
  /* The per-site card's state. Page-host keyed picks, image-host keyed latch
   * evidence, and the pointer between them -- see background.js's DEFAULTS for
   * why the two keyings differ. The popup reads all five and writes only the
   * two pick maps (plus clearing all of them on Re-detect). The old global
   * `joinSlices` checkbox is folded into `profileByHost`. */
  profileByHost: {},
  languageOverrideByHost: {},
  formatByHost: {},
  languageByHost: {},
  imageHostForPage: {},
  // Must match background.js and the server, or the first translate is a 409.
  provider: "local",
  llm: "",
  token: "",
  cacheMaxEntries: CACHE_DEFAULT_ENTRIES,
  cacheMaxBytes: CACHE_DEFAULT_BYTES,
  /* Display only, and only ever what the last reset replied with. The story id
   * is no longer a stored setting on either side: the background script DERIVES
   * it from the page's URL (`story.js::storyIdFor`), so there is nothing here
   * for the popup to keep in step with, and a value written here would be a
   * second source of truth that could never win.
   *
   * This popup does not call `settingsFingerprint` -- it reads the cache through
   * `cacheSummary`/`cacheList`, which key on nothing -- so unlike `ocr` and
   * `provider` there is no keying consequence to getting it wrong here. */
  storyId: "",
};

const $ = (id) => document.getElementById(id);
let settings = { ...DEFAULTS };
let host = "";

const save = (patch) => {
  Object.assign(settings, patch);
  return browser.storage.local.set(patch);
};

/* ------------------------------------------------------------------ render */

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

/* Read straight out of IndexedDB rather than from `settings`: the store is the
 * cache now, nothing about it goes through storage.local, and a translate in
 * another tab adds to it while this popup is open. Rows are built node by node
 * and filled with textContent, as everything in this panel always has been. */
async function renderCache() {
  const list = $("cacheEntries");
  // Not "cacheTotals": an element id becomes a window property, and cache.js
  // already has a cacheTotals of its own.
  const totals = $("cacheStats");

  const [entries, summary] = await Promise.all([
    cacheList().catch(() => null),
    cacheSummary().catch(() => null),
  ]);

  if (!entries || !summary) {
    totals.textContent = "The cache could not be read.";
    list.textContent = "";
    return;
  }

  // The hit/miss split is the only way to tell a cache that is working from one
  // that is quietly missing every time.
  totals.textContent =
    `${plural(summary.count, "page")} · ${size(summary.bytes)} · ` +
    `${plural(summary.hits, "hit")}, ${plural(summary.misses, "translation")}`;

  list.textContent = "";

  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No translations yet.";
    list.append(empty);
    return;
  }

  for (const entry of entries) {
    const row = document.createElement("div");
    row.className = "item";

    const when = document.createElement("b");
    when.textContent = new Date(entry.at).toLocaleTimeString();

    const meta = document.createElement("span");
    const served = entry.hits ? ` · ×${entry.hits} cached` : "";
    meta.textContent =
      `${(entry.ms / 1000).toFixed(1)}s · ` +
      `${Math.round(entry.bytes / 1024)} KB${served}`;

    row.append(when, meta);
    list.append(row);
  }
}

/* The two names below must track cli.rs's DEFAULT_LOCAL_MODEL. They are only a
 * placeholder and a hint -- DEFAULTS sends `llm: ""` and lets the server choose,
 * so a stale name here misinforms the reader rather than pinning the wrong
 * model. */
function renderLlmHint() {
  const ollama = settings.provider === "ollama";
  $("llmHint").textContent = ollama
    ? "An Ollama tag, e.g. qwen3:8b. Reuses a model already in VRAM."
    : "A Koharu registry id, e.g. gemma4-26b-a4b-it. Loads a second copy into VRAM.";
  // Blank is only valid under local. Ollama refuses to start without a tag,
  // because the two backends name models in different namespaces.
  $("llm").placeholder = ollama ? "qwen3:8b  (required)" : "blank = gemma4-26b-a4b-it";
}

/* The Inpainter select picks which eraser runs; Keep the artwork decides whether
 * one runs at all. Greyed rather than hidden so the chosen model is still
 * readable, and because leaving it live would offer a control with no effect --
 * and worse, changing it would still reach the server, where a different
 * PipelineConfig forces a full pipeline reload for a setting that cannot alter
 * the page. */
function renderKeepArt() {
  $("inpainting").disabled = settings.keepArt;
}

function renderCorners() {
  for (const b of document.querySelectorAll(".corner")) {
    b.classList.toggle("on", b.dataset.corner === settings.corner);
  }
}

async function checkServer() {
  const status = $("status");
  const dot = $("dot");
  const reply = await browser.runtime.sendMessage({ type: "ping" }).catch(() => null);

  if (reply && reply.ok) {
    status.textContent = `Connected to ${settings.serverUrl}`;
    status.className = "status good";
    dot.className = "dot up";
  } else {
    status.textContent = `No BireLate server at ${settings.serverUrl}`;
    status.className = "status bad";
    dot.className = "dot down";
  }
}

/* -------------------------------------------------------------- server state
 *
 * The server has no window of its own, so this panel is its only face. The
 * countdown runs off a deadline the background script cached, which means one
 * /status request per popup open rather than one per second.
 */

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;
const gib = (bytes) => `${(bytes / GIB).toFixed(1)} GB`;
const size = (bytes) =>
  bytes >= GIB ? gib(bytes) : `${(bytes / MIB).toFixed(1)} MB`;

let deadlineAt = null;
// The /status `ocr_substitute` last seen, so the OCR dropdown can re-decide the note.
let ocrSubstitute = null;
let ticker = 0;
let poller = 0;
/* One delayed re-check, armed only when the countdown crosses zero. See tick():
 * this panel's clock reaches the deadline at or just before the server's, so the
 * confirming request can arrive while the weights are still being dropped. A
 * timeout and never an interval, so the worst case is one stale readout rather
 * than a poll that never stops. */
let confirmTimer = 0;
let acting = false;

function mmss(total) {
  const m = Math.floor(total / 60);
  const s = Math.floor(total % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function stopTicker() {
  if (!ticker) return;
  clearInterval(ticker);
  ticker = 0;
}

function startTicker() {
  // Never leave two running: apply() is called again on every refresh.
  stopTicker();
  if (deadlineAt !== null) ticker = setInterval(tick, 1000);
}

/* Polling, for the one state that cannot resolve itself.
 *
 * `POST /warmup` answers 202 the instant it has queued the load, so the /status
 * that follows it necessarily lands mid-load and reports "not loaded, busy".
 * The countdown is the only self-refreshing thing in this panel and a warming
 * server has no deadline to count down, so without this the readout would sit
 * on that first snapshot until the popup was closed and reopened. */
const POLL_MS = 2500;
const POLL_LIMIT = 120; // ~5 minutes. A cold four-model load is slow, not endless.

function stopPolling() {
  if (!poller) return;
  clearInterval(poller);
  poller = 0;
}

function pollWhileLoading() {
  stopPolling();
  let left = POLL_LIMIT;
  poller = setInterval(async () => {
    if (--left <= 0) return stopPolling();

    const reply = await browser.runtime
      .sendMessage({ type: "status" })
      .catch(() => null);

    // A server that stopped answering will not start again by being asked
    // faster, and refreshStatus() has already blanked the panel for that case.
    if (!reply || !reply.ok || !reply.status) return stopPolling();

    apply(reply);
    // `busy` is not the condition: another tab's page can hold the gate long
    // after the weights this was waiting for are resident.
    if (reply.status.models_loaded) stopPolling();
  }, POLL_MS);
}

function stopConfirm() {
  if (!confirmTimer) return;
  clearTimeout(confirmTimer);
  confirmTimer = 0;
}

function stopTimers() {
  stopTicker();
  stopPolling();
  // A popup is destroyed on close, and this one fires after a delay by design:
  // without clearing it, refreshStatus() runs against a torn-down document.
  stopConfirm();
}

// Without this the intervals outlive the panel and keep firing at a document
// that is being torn down.
addEventListener("pagehide", stopTimers);
addEventListener("unload", stopTimers);

function tick() {
  if (deadlineAt === null) {
    stopTicker();
    return;
  }

  const left = Math.round((deadlineAt - Date.now()) / 1000);
  if (left > 0) {
    $("countdown").textContent = `Unloads in ${mmss(left)}`;
    return;
  }

  /* The deadline passed. One request confirms what the server actually did,
   * which is still not polling.
   *
   * Delayed, because asking immediately races the thing being confirmed: this
   * countdown is derived from `unload_in_secs`, which the server truncates to
   * whole seconds, and tick() rounds -- so the panel reaches zero up to a
   * second and a half before the server has finished dropping ~20 GB. A reply
   * arriving inside that window still says "loaded", and apply() correctly
   * refuses the already-expired deadline that comes with it, which leaves the
   * ticker stopped and the panel reading "Idle unload armed" for as long as it
   * stays open -- describing VRAM that was freed a moment later. */
  $("countdown").textContent = "Unloading…";
  deadlineAt = null;
  stopTicker();
  stopConfirm();
  confirmTimer = setTimeout(() => {
    confirmTimer = 0;
    refreshStatus();
  }, 1500);
}

function shortfall(s) {
  const free = s.vram && s.vram.available_bytes;
  const need = s.needed_bytes;
  if (typeof free !== "number" || typeof need !== "number") {
    return "Not enough free VRAM to translate right now.";
  }
  return `Not enough VRAM: ${gib(free)} free, ${gib(need)} needed.`;
}

function renderVram(s) {
  const text = $("vramText");
  const fill = $("vramFill");
  const meter = $("vramMeter");
  const device = $("vramDevice");
  const vram = s.vram;

  const measured =
    vram &&
    typeof vram.budget_bytes === "number" &&
    typeof vram.available_bytes === "number" &&
    vram.budget_bytes > 0;

  if (!measured) {
    // No sample has landed yet, or the server is running on the CPU. Missing
    // telemetry is not the same as no memory free, so draw nothing rather than
    // an empty bar that reads as exhaustion.
    text.textContent = "not reported";
    fill.style.width = "0%";
    meter.className = "meter";
    device.textContent = "";
    return;
  }

  const used = Math.max(0, vram.budget_bytes - vram.available_bytes);
  const share = Math.max(0, Math.min(1, used / vram.budget_bytes));
  text.textContent = `${gib(used)} of ${gib(vram.budget_bytes)}`;
  fill.style.width = `${(share * 100).toFixed(1)}%`;
  meter.className = share > 0.9 ? "meter hot" : "meter";

  /* These are Windows' per-process figures: the budget it hands this server,
   * and this server's own usage. Ollama's share and the browser's do not appear
   * in either, so calling it "the card" would be a lie. The device name comes
   * from the server, hence a text node and never innerHTML. */
  const caveat = "this server's budget, not the whole card";
  device.textContent = vram.device ? `${vram.device} · ${caveat}` : caveat;
}

function renderOcrNote(sub) {
  // The note is about one engine only: hide it unless the dropdown is on it.
  ocrSubstitute = sub || null;
  const show = !!ocrSubstitute && $("ocr").value === ocrSubstitute.requested;
  if (show) $("ocrServedBy").textContent = String(ocrSubstitute.served_by || "another engine");
  $("ocrNote").hidden = !show;
}

function renderStatus(s) {
  const model = $("modelState");
  if (s.busy) {
    /* Not "translating". The server computes `busy` from its GPU permit, and a
     * warmup holds that permit for the whole load -- so the word that fits both
     * is the vaguer one. Claiming a translation during a warmup was simply
     * false, and it read next to "Nothing in VRAM". */
    model.textContent = "busy";
    model.className = "v good";
  } else if (s.models_loaded) {
    model.textContent = "loaded";
    model.className = "v good";
  } else {
    model.textContent = "not loaded";
    model.className = "v";
  }

  renderVram(s);
  // Older servers send no field; null and absent both mean no substitute.
  renderOcrNote(s.ocr_substitute);

  if (deadlineAt !== null) {
    tick();
  } else if (!s.models_loaded && s.busy) {
    // Held the gate but has nothing resident yet: something is loading.
    $("countdown").textContent = "Loading models…";
  } else if (!s.models_loaded) {
    $("countdown").textContent = "Nothing in VRAM";
  } else if (s.idle_unload_secs === 0) {
    $("countdown").textContent = "Stays loaded until you free it";
  } else {
    $("countdown").textContent = "Idle unload armed";
  }

  const note = $("gpuNote");
  if (s.sufficient === false) {
    note.textContent = shortfall(s);
    note.className = "note bad";
  } else if (!acting) {
    note.textContent = "";
    note.className = "note";
  }
}

function apply(payload) {
  if (!payload || !payload.status) return;

  /* A deadline that has already passed gets no countdown and no follow-up
   * request. Accepting one would spin: tick() would see it expired, refresh,
   * and be handed the same expired deadline back. Only a live countdown
   * crossing zero is allowed to ask again. */
  const at = typeof payload.deadlineAt === "number" ? payload.deadlineAt : null;
  deadlineAt = at !== null && at > Date.now() ? at : null;

  renderStatus(payload.status);
  startTicker();
}

async function refreshStatus() {
  const reply = await browser.runtime
    .sendMessage({ type: "status" })
    .catch((err) => ({ ok: false, error: String(err.message || err) }));

  if (reply && reply.ok) {
    apply(reply);
    return;
  }

  // A server that cannot answer /status may still translate perfectly well, so
  // this is a blank readout rather than an alarm. The dot already covers reach.
  deadlineAt = null;
  stopTicker();
  $("modelState").textContent = "unknown";
  $("modelState").className = "v";
  $("countdown").textContent = "—";
  renderVram({});
  renderOcrNote(null);

  const note = $("gpuNote");
  note.textContent = String((reply && reply.error) || "").slice(0, 120);
  note.className = "note";
}

function setActions(enabled) {
  $("warmBtn").disabled = !enabled;
  $("unloadBtn").disabled = !enabled;
}

async function act(type, pending) {
  if (acting) return null;
  acting = true;
  setActions(false);
  // Whatever was being waited for, this button supersedes it.
  stopPolling();

  const note = $("gpuNote");
  note.textContent = pending;
  note.className = "note";

  const reply = await browser.runtime
    .sendMessage({ type })
    .catch((err) => ({ ok: false, error: String(err.message || err) }));

  acting = false;
  setActions(true);

  if (reply && reply.ok) {
    note.textContent = "";
    note.className = "note";
    apply(reply);
    return reply;
  }

  note.textContent = String((reply && reply.error) || "request failed").slice(0, 120);
  note.className = "note bad";
  return reply;
}

/* -------------------------------------------------------------- diagnostics
 *
 * "No images are detected" can be two failures in one: nothing is found, and
 * there is no way to see why. This panel answers the second: it asks the page
 * itself, in every frame, what the content script can see -- and its most
 * useful answer is the one where nothing answers at all.
 *
 * The frames reply by broadcast rather than as a reply to the send, because
 * tabs.sendMessage reaches every frame but surfaces only the first response,
 * and "how many frames did we reach" is precisely what needs measuring.
 */

// Long enough for a page full of frames to answer, short enough not to feel
// broken. Each frame's report is a handful of rect reads, not a network call.
const DIAG_WAIT = 700;

let diagSeq = 0;
let diagToken = 0;
let diagTabId = null;
let diagFrames = [];
// Kept so a frame that answers after the window closes redraws the panel rather
// than being dropped: a page with many frames is exactly the slow case.
let diagContext = null;

const clipText = (value, max) => {
  const text = String(value == null ? "" : value);
  return text.length > max ? `${text.slice(0, max)}…` : text;
};

function diagKV(parent, key, value) {
  const row = document.createElement("div");
  row.className = "kv";

  const k = document.createElement("span");
  k.className = "k";
  k.textContent = key;

  const v = document.createElement("span");
  v.className = "v";
  v.textContent = value;

  row.append(k, v);
  parent.append(row);
  return row;
}

function diagCard(title) {
  const card = document.createElement("div");
  card.className = "card block";

  const head = document.createElement("div");
  head.className = "diag-h";
  head.textContent = title;

  card.append(head);
  $("diagOut").append(card);
  return card;
}

function diagNote(parent, text) {
  const p = document.createElement("p");
  p.className = "hint";
  p.textContent = text;
  parent.append(p);
}

function renderFrame(frame) {
  const card = diagCard(frame.top ? "Top frame" : `Frame ${frame.frameId}`);
  diagKV(card, "URL", clipText(frame.url, 100));

  if (frame.broken) {
    diagKV(card, "Report failed", clipText(frame.broken, 100));
    return;
  }

  const counts = frame.counts || {};
  diagKV(card, "Frame host", frame.host || "(none)");
  diagKV(card, "Enabled here", frame.enabled ? "yes" : "no");
  diagKV(card, "Why", clipText(frame.reason, 90));
  diagKV(card, "Watching the page", frame.attached ? "yes" : "no");
  diagKV(
    card,
    "Overlay in the DOM",
    frame.barConnected
      ? "yes"
      : frame.barBuilt
        ? "built, but the page removed it"
        : "not built yet (built on first hover)"
  );
  diagKV(card, "Auto-translate", frame.autoTranslate ? "on" : "off");
  diagKV(card, "Minimum size", `${frame.minSize}px`);

  diagKV(card, "Images in this frame", String(counts.total || 0));
  diagKV(card, "Show a button", String(counts.eligible || 0));
  // The split that matters on a lazy-loading reader: a button over a 1x1 GIF
  // is not a page this extension can translate yet.
  diagKV(card, "Ready to translate", String(counts.ready || 0));
  diagKV(card, "Placeholder / too small to send", String(counts.waiting || 0));
  diagKV(card, "Responsive (srcset / picture)", String(counts.responsive || 0));
  diagKV(card, "Rejected: too small", String(counts.small || 0));
  diagKV(card, "Rejected: not loaded yet", String(counts.pending || 0));
  diagKV(
    card,
    "Rejected: detached / opted out",
    `${counts.detached || 0} / ${counts.skipped || 0}`
  );

  diagKV(card, "Nested frames", String(frame.frames || 0));
  diagKV(card, "Big <canvas> (unsupported)", String(frame.canvases || 0));
  diagKV(card, "CSS background panels (unsupported)", String(frame.backgrounds || 0));

  if (!frame.samples || !frame.samples.length) {
    diagNote(card, "This frame contains no <img> elements at all.");
    return;
  }

  for (const sample of frame.samples) {
    const row = document.createElement("div");
    row.className = "diag-img";

    const head = document.createElement("b");
    head.textContent = `${sample.size} — ${sample.verdict}`;

    const dims = document.createElement("span");
    dims.textContent = `natural ${sample.natural} · rendered ${sample.box}`;

    const src = document.createElement("span");
    src.textContent = sample.src || "(no src attribute)";

    row.append(head, dims, src);

    if (sample.lazy) {
      const lazy = document.createElement("span");
      lazy.className = "warn";
      lazy.textContent = `lazy: ${sample.lazy}`;
      row.append(lazy);
    }

    if (sample.responsive) {
      const responsive = document.createElement("span");
      responsive.textContent = `responsive: ${sample.responsive}`;
      row.append(responsive);
    }

    if (sample.overlay) {
      const overlay = document.createElement("span");
      overlay.className = "warn";
      overlay.textContent = sample.overlay;
      row.append(overlay);
    }

    card.append(row);
  }
}

function renderDiag() {
  const { tab, reachable, sendError } = diagContext;
  const out = $("diagOut");
  out.textContent = "";

  /* Derived here rather than read off the popup's `host`, because the two
   * hostnames disagreeing is one of the failures this panel exists to catch:
   * the popup keys the enable list on the tab URL and the content script used
   * to key it on its own `location.hostname`. */
  let tabHost = "";
  try {
    tabHost = new URL(tab.url).hostname;
  } catch {
    tabHost = "";
  }
  const listed = Array.isArray(settings.enabledHosts) ? settings.enabledHosts : [];

  const mine = diagCard("What the popup sees");
  diagKV(mine, "Tab URL", clipText(tab && tab.url, 100));
  diagKV(mine, "Tab host", tabHost || "(none)");
  diagKV(mine, "Enabled everywhere", settings.enabledEverywhere ? "yes" : "no");
  diagKV(mine, "Tab host in the list", listed.includes(tabHost) ? "yes" : "no");
  diagKV(mine, "Frames that answered", String(diagFrames.length));

  /* The honest failure, and the one that used to be indistinguishable from
   * "found nothing": no script is running in that tab at all. */
  if (!diagFrames.length) {
    const none = diagCard("No content script in this tab");
    diagNote(
      none,
      reachable
        ? "The tab accepted the request but no frame reported back within " +
            `${DIAG_WAIT} ms. Reload the page and try again.`
        : "Firefox is not running this extension on that page. That is normal " +
            "for about:, view-source:, the built-in PDF viewer and " +
            "addons.mozilla.org, and it also happens to any tab that was " +
            "already open before the extension was loaded. Reload the tab and " +
            "try again."
    );
    if (sendError) diagKV(none, "Firefox said", clipText(sendError, 100));
    return;
  }

  // Frame 0 is the top document; the rest in the order Firefox numbered them.
  diagFrames.sort((a, b) => (a.frameId || 0) - (b.frameId || 0));
  for (const frame of diagFrames) renderFrame(frame);
}

async function runDiagnose() {
  const button = $("diagnose");
  const out = $("diagOut");
  button.disabled = true;
  out.textContent = "";
  diagNote(out, "Asking the page…");

  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  diagTabId = tab && typeof tab.id === "number" ? tab.id : null;
  diagToken = ++diagSeq;
  diagFrames = [];
  diagContext = { tab, reachable: false, sendError: "", done: false };

  if (diagTabId === null) {
    button.disabled = false;
    diagContext.sendError = "no active tab";
    diagContext.done = true;
    renderDiag();
    return;
  }

  try {
    await browser.tabs.sendMessage(diagTabId, { type: "diagnose", token: diagToken });
    diagContext.reachable = true;
  } catch (err) {
    // The rejection is the diagnosis: nothing in that tab is listening.
    diagContext.sendError = String((err && err.message) || err);
  }

  await new Promise((done) => setTimeout(done, DIAG_WAIT));
  button.disabled = false;
  diagContext.done = true;
  renderDiag();
}

$("diagnose").addEventListener("click", () => {
  runDiagnose().catch((err) => {
    $("diagOut").textContent = "";
    diagNote($("diagOut"), String((err && err.message) || err).slice(0, 160));
    $("diagnose").disabled = false;
  });
});

/* ------------------------------------------------------------------- wiring */

async function init() {
  settings = { ...DEFAULTS, ...(await browser.storage.local.get(null)) };
  renderStory();

  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  try {
    host = new URL(tab.url).hostname;
  } catch {
    host = "";
  }

  $("enableHere").checked =
    settings.enabledEverywhere || settings.enabledHosts.includes(host);
  $("enableHere").disabled = settings.enabledEverywhere;
  $("autoTranslate").checked = settings.autoTranslate;
  $("enabledEverywhere").checked = settings.enabledEverywhere;
  $("minSize").value = settings.minSize;
  $("minSizeOut").textContent = `${settings.minSize}px`;
  $("serverUrl").value = settings.serverUrl;
  $("token").value = settings.token;
  $("llm").value = settings.llm;
  $("cacheMaxEntries").value = settings.cacheMaxEntries;
  $("cacheMaxMb").value = Math.round(settings.cacheMaxBytes / MIB);

  for (const id of ["targetLanguage", "ocr", "inpainting", "provider"]) {
    $(id).value = settings[id];
  }
  renderOcrNote(ocrSubstitute);
  $("keepArt").checked = settings.keepArt;
  renderSite();
  renderKeepArt();

  renderCorners();
  renderLlmHint();
  checkServer();
  refreshStatus();
  /* The cache panel is deliberately absent from that list: it is not the one on
   * screen, and reading it means opening the database. Its tab builds it. */
}

$("enableHere").addEventListener("change", async (e) => {
  const hosts = new Set(settings.enabledHosts);
  e.target.checked ? hosts.add(host) : hosts.delete(host);
  await save({ enabledHosts: [...hosts] });
});

$("enabledEverywhere").addEventListener("change", async (e) => {
  await save({ enabledEverywhere: e.target.checked });
  $("enableHere").checked = e.target.checked || settings.enabledHosts.includes(host);
  $("enableHere").disabled = e.target.checked;
});

$("autoTranslate").addEventListener("change", (e) =>
  save({ autoTranslate: e.target.checked })
);

$("minSize").addEventListener("input", (e) => {
  $("minSizeOut").textContent = `${e.target.value}px`;
});
$("minSize").addEventListener("change", (e) =>
  save({ minSize: Number(e.target.value) })
);

for (const id of ["targetLanguage", "inpainting"]) {
  $(id).addEventListener("change", (e) => save({ [id]: e.target.value }));
}

/* Picking an engine by hand turns the detector off, everywhere, permanently.
 *
 * An explicit choice must never be second-guessed by a heuristic -- and the
 * per-host latches go with it, because leaving them would make the dropdown lie:
 * it would show the chosen engine while a latched host quietly used another.
 * Turning it back on is deliberately not a control here; clearing the setting
 * is, and a reader who wants the detector back can re-enable it from Settings
 * once there is a checkbox for it. */
$("ocr").addEventListener("change", (e) => {
  renderOcrNote(ocrSubstitute);
  return save({ ocr: e.target.value, ocrAuto: false, ocrByHost: {}, ocrProbe: {}, ocrDissent: {} });
});

/* The site card: what the latch decided about THIS site, and the reader's
 * correction of it. Picks are page-host keyed and written per host -- unlike
 * the OCR dropdown above, correcting one site never touches another, and it
 * does not turn `ocrAuto` off: the heuristic stays on for every other host. */
function renderSite() {
  const pickP = (settings.profileByHost || {})[host] || "";
  const pickL = (settings.languageOverrideByHost || {})[host] || "";
  /* The latch is keyed on the IMAGE host; the pointer map is how this popup,
   * which only knows the PAGE host, finds it. Before the first translate on a
   * CDN-served site the pointer is absent and the page host is the best key
   * there is -- right on same-host sites, and merely "still detecting" on the
   * rest until one page has been translated. */
  const imgHost = (settings.imageHostForPage || {})[host] || host;
  const format = (settings.formatByHost || {})[imgHost] || "";
  const profile = pickP || PROFILE_FROM_FORMAT[format] || "";
  const language = pickL || (settings.languageByHost || {})[imgHost] || "";
  const profileWord = { manga: "Manga", webtoon: "Webtoon" }[profile];
  const langWord = { ja: "Japanese", zh: "Chinese", ko: "Korean" }[language];
  $("profileHere").value = pickP;
  $("languageHere").value = pickL;
  $("siteState").textContent = !host
    ? "—"
    : profileWord || langWord
      ? `${profileWord || "layout undecided"} · ${langWord || "language undecided"}` +
        (pickP || pickL ? " (your pick)" : " (detected)")
      : "still detecting";
}

const saveHostMap = (key, value) => {
  const map = { ...(settings[key] || {}) };
  if (value) map[host] = value;
  else delete map[host];
  return save({ [key]: map });
};

$("profileHere").addEventListener("change", async (e) => {
  await saveHostMap("profileByHost", e.target.value);
  renderSite();
});

$("languageHere").addEventListener("change", async (e) => {
  await saveHostMap("languageOverrideByHost", e.target.value);
  renderSite();
});

/* Start this site's detection over: the picks go, and so does the latch's
 * verdict for the image host that serves it. Per host, which is the whole
 * point -- the only reset that existed before this cleared EVERY host at once
 * (the OCR dropdown's escape hatch, which still behaves that way). */
$("redetectBtn").addEventListener("click", async () => {
  const imgHost = (settings.imageHostForPage || {})[host] || host;
  const drop = (key, mapHost) => {
    const map = { ...(settings[key] || {}) };
    delete map[mapHost];
    return { [key]: map };
  };
  await save({
    ...drop("profileByHost", host),
    ...drop("languageOverrideByHost", host),
    ...drop("ocrByHost", imgHost),
    ...drop("languageByHost", imgHost),
    ...drop("formatByHost", imgHost),
    ...drop("ocrProbe", imgHost),
    ...drop("ocrDissent", imgHost),
  });
  renderSite();
  $("siteNote").textContent = "Cleared. The next pages you translate here decide again.";
});

$("keepArt").addEventListener("change", async (e) => {
  await save({ keepArt: e.target.checked });
  renderKeepArt();
});

$("provider").addEventListener("change", async (e) => {
  await save({ provider: e.target.value });
  renderLlmHint();
});

$("llm").addEventListener("change", (e) => save({ llm: e.target.value.trim() }));

// /status is the one readout the token gates, so a corrected token has to
// re-ask immediately or the panel keeps showing the 401.
$("token").addEventListener("change", async (e) => {
  await save({ token: e.target.value.trim() });
  refreshStatus();
});

$("serverUrl").addEventListener("change", async (e) => {
  await save({ serverUrl: e.target.value.trim() || DEFAULTS.serverUrl });
  checkServer();
  refreshStatus();
});

$("warmBtn").addEventListener("click", async () => {
  const reply = await act("warmup", "Loading models…");
  // The 202 means "queued", not "done", so the snapshot that came back with it
  // is a picture of the load starting. Follow it until the weights are up.
  if (reply && reply.ok && reply.status && !reply.status.models_loaded) {
    pollWhileLoading();
  }
});
$("unloadBtn").addEventListener("click", () => act("unload", "Freeing VRAM…"));

/* Two clicks, because it is not undoable: stopping the server also deletes every
 * cached page. `act` cannot be reused -- it applies a status snapshot to the UI,
 * and there is no server left to describe. */
let stopArmed = false;
let stopTimer = null;
$("stopBtn").addEventListener("click", async () => {
  const button = $("stopBtn");
  const note = $("gpuNote");
  if (!stopArmed) {
    stopArmed = true;
    button.textContent = "Sure? Deletes pages";
    note.textContent = "Stops the server and clears every cached page.";
    note.className = "note";
    clearTimeout(stopTimer);
    stopTimer = setTimeout(() => {
      stopArmed = false;
      button.textContent = "Stop & clear";
      note.textContent = "";
    }, 5000);
    return;
  }
  clearTimeout(stopTimer);
  stopArmed = false;
  button.textContent = "Stop & clear";
  button.disabled = true;
  note.textContent = "Stopping…";
  stopPolling();

  const reply = await browser.runtime
    .sendMessage({ type: "stop-server" })
    .catch((err) => ({ ok: false, error: String(err.message || err) }));

  button.disabled = false;
  if (reply && reply.ok) {
    note.textContent = "Server stopped, cached pages deleted.";
    note.className = "note";
  } else {
    note.textContent = (reply && reply.error) || "could not stop the server";
    note.className = "note bad";
  }
  renderCache().catch(() => {});
});

$("startBtn").addEventListener("click", async () => {
  const note = $("gpuNote");
  note.textContent = "Starting…";
  note.className = "note";
  const reply = await browser.runtime
    .sendMessage({ type: "start-server" })
    .catch((err) => ({ ok: false, error: String(err.message || err) }));
  if (reply && reply.ok) {
    note.textContent = reply.already
      ? "A BireLate server is already running or starting - see its window."
      : "Server starting. The first start downloads about 3.7 GB; progress shows in the server window.";
    note.className = "note";
  } else {
    /* The host is registered by Setup.bat, once, so "not installed" is the ordinary
     * first-run state and deserves the instruction rather than a raw error. */
    note.textContent =
      (reply && reply.error) ||
      "no launcher registered - run Setup.bat once, then restart Firefox";
    note.className = "note bad";
  }
});

/* Always "on". The background script derives an id from the page's URL on every
 * request, so an empty `storyId` here means "this popup has not been told one"
 * and never "off". Reporting that as off would be a lie the very next translate
 * corrects. */
function renderStory() {
  $("storyState").textContent = "on";
}

async function story(type, done) {
  const note = $("storyNote");
  note.textContent = "";
  note.className = "note";
  const reply = await browser.runtime
    .sendMessage({ type })
    .catch((err) => ({ ok: false, error: String(err.message || err) }));
  if (reply && reply.ok) {
    settings.storyId = reply.storyId || "";
    renderStory();
    note.textContent = done;
  } else {
    note.textContent = (reply && reply.error) || "could not change the story";
    note.className = "note bad";
  }
}

/* "this series" and not "the story", because the reset now moves exactly one
 * series -- the one in the tab this popup is open over -- and leaves whatever
 * else the reader has open untouched. */
$("resetStoryBtn").addEventListener("click", () =>
  story("story-reset", "Reset. The next page of this series starts fresh.")
);

/* ---------------------------------------------------- the glossary editor
 *
 * The popup never computes the series key or touches the terms' storage: both
 * live behind "glossary-get"/"glossary-set" messages, because the key is
 * derived from a page URL only the background reliably sees (the same
 * reasoning as story-reset), and the validation must be the one the wire is
 * built from. This side is a textarea and two buttons. */
function renderGlossary(reply) {
  $("glossaryState").textContent = reply.count
    ? `${reply.count} term${reply.count === 1 ? "" : "s"}`
    : "none";
}

async function loadGlossary() {
  const reply = await browser.runtime
    .sendMessage({ type: "glossary-get" })
    .catch(() => null);
  if (!reply || !reply.ok) return;
  $("glossaryText").value = reply.text;
  renderGlossary(reply);
}

async function applyGlossary(text, done) {
  const note = $("glossaryNote");
  note.textContent = "";
  note.className = "note";
  const reply = await browser.runtime
    .sendMessage({ type: "glossary-set", text })
    .catch((err) => ({ ok: false, error: String(err.message || err) }));
  if (reply && reply.ok) {
    renderGlossary(reply);
    note.textContent = done;
  } else if (reply && Array.isArray(reply.errors)) {
    note.textContent = reply.errors.join("; ");
    note.className = "note bad";
  } else {
    note.textContent = (reply && reply.error) || "could not save the glossary";
    note.className = "note bad";
  }
}

$("glossaryApplyBtn").addEventListener("click", () =>
  applyGlossary(
    $("glossaryText").value,
    "Saved. Applies from the next translated page; gone when the browser closes."
  )
);

$("glossaryClearBtn").addEventListener("click", () => {
  $("glossaryText").value = "";
  return applyGlossary("", "Cleared. The next page translates without pinned terms.");
});

/* The drift suggestions. The ledger lives behind the same message
 * boundary the terms do; this side renders choices and edits the TEXTAREA,
 * never the store -- picking fills a line, Apply commits, so a reader can
 * settle several terms in one fingerprint change rather than one re-render
 * per pick. Rows are built node by node, as everywhere in this file. */
function suggestLine(text, source) {
  const lines = text.split("\n").filter((line) => line.trim());
  const kept = lines.filter((line) => {
    const eq = line.indexOf("=");
    return eq < 0 || line.slice(0, eq).trim() !== source;
  });
  return kept;
}

function renderSuggestions(items) {
  const list = $("glossarySuggest");
  list.replaceChildren();
  $("glossarySuggestHint").hidden = !items.length;
  for (const item of items.slice(0, 8)) {
    const row = document.createElement("div");
    row.className = "item";
    const label = document.createElement("div");
    label.className = "suggest-source";
    label.textContent = item.source;
    row.append(label);
    for (const one of item.renderings) {
      const pick = document.createElement("button");
      pick.type = "button";
      pick.className = "ghost suggest-pick";
      pick.textContent = one.count > 1 ? `${one.text} ×${one.count}` : one.text;
      pick.addEventListener("click", () => {
        const area = $("glossaryText");
        const kept = suggestLine(area.value, item.source);
        kept.push(`${item.source} = ${one.text}`);
        area.value = kept.join("\n");
        $("glossaryNote").textContent = "Filled in above - Apply commits.";
        $("glossaryNote").className = "note";
      });
      row.append(pick);
    }
    list.append(row);
  }
}

async function loadSuggestions() {
  const reply = await browser.runtime
    .sendMessage({ type: "glossary-suggest" })
    .catch(() => null);
  if (!reply || !reply.ok) return;
  renderSuggestions(Array.isArray(reply.items) ? reply.items : []);
}

void loadGlossary();
void loadSuggestions();

/* The background script only broadcasts what the popup cannot have caused
 * itself -- a translate finishing in another tab while this is open. The
 * snapshot travels with the message, so answering it with another /status
 * request would only loop. */
browser.runtime.onMessage.addListener((message, sender) => {
  if (!message || typeof message.type !== "string") return;

  if (message.type === "status-changed") {
    apply(message);
    return;
  }

  /* A frame answering the Diagnose button. Checked three ways -- the token from
   * this popup's own request, the tab it was asked about, and a sender that
   * really is one of our content scripts -- so a stale run cannot bleed into
   * the next one. */
  if (message.type === "diagnostic-report") {
    if (!diagToken || message.token !== diagToken) return;
    if (!sender || !sender.tab || sender.tab.id !== diagTabId) return;
    diagFrames.push({
      frameId: typeof sender.frameId === "number" ? sender.frameId : 0,
      ...message.report,
    });
    // A frame slower than the collection window still gets drawn.
    if (diagContext && diagContext.done) renderDiag();
  }
});

$("corners").addEventListener("click", async (e) => {
  const button = e.target.closest(".corner");
  if (!button) return;
  await save({ corner: button.dataset.corner });
  renderCorners();
});

/* Both limits are read back rather than trusted: the field accepts anything a
 * spinner can reach, including blank, and a zero would evict every page the
 * moment it was stored. */
$("cacheMaxEntries").addEventListener("change", (e) => {
  const entered = Math.floor(Number(e.target.value));
  const value = entered > 0 ? entered : DEFAULTS.cacheMaxEntries;
  e.target.value = value;
  save({ cacheMaxEntries: value });
});

$("cacheMaxMb").addEventListener("change", (e) => {
  const entered = Math.floor(Number(e.target.value));
  const value = entered > 0 ? entered : Math.round(DEFAULTS.cacheMaxBytes / MIB);
  e.target.value = value;
  save({ cacheMaxBytes: value * MIB });
});

$("clearCache").addEventListener("click", async () => {
  await cacheClear().catch(() => {});
  renderCache();
});

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const t of document.querySelectorAll(".tab")) t.classList.remove("on");
    tab.classList.add("on");
    for (const p of document.querySelectorAll(".panel")) p.classList.add("hidden");
    $(`panel-${tab.dataset.panel}`).classList.remove("hidden");
    // Rebuilt on every visit: a translate in another tab changes the store
    // while this popup is open, and nothing tells the popup about it.
    if (tab.dataset.panel === "cache") renderCache();
  });
}

init();
