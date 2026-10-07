/* BireLate - in-page overlay.
 *
 * Layout: a logo button that translates the hovered image, and a small
 * circular "A" toggle beside it for auto mode, which keeps translating as the
 * gallery advances.
 *
 * This runs in every frame. Manga readers very commonly paint their pages
 * inside an iframe, and a top-frame-only script finds an empty document there.
 * The overlay needs no coordinate arithmetic to cope with that: `.khr-bar` is
 * position:fixed and every measurement below is viewport-relative, and a
 * frame's viewport is its own box on the screen.
 */

/* Only the keys this script actually uses. It used to read the whole store,
 * which pulled the server token into every page's sandbox -- and with
 * all_frames that is now every advertising iframe's sandbox too. */
const DEFAULTS = {
  enabledEverywhere: false,
  enabledHosts: [],
  minSize: 150,
  corner: "top-left",
  autoTranslate: false,
  // The reader's explicit layout pick per PAGE host ("manga" | "webtoon").
  // Replaces the old global `joinSlices` checkbox: a site declared manga skips
  // seam scheduling here, cheaply, and the background's config-resolved gate
  // (which also sees the latch's detected format) is the authority for the
  // rest. Only this script can see that two <img> elements are stacked flush,
  // so the decision to try starts here even though every byte of the work
  // happens in the background.
  profileByHost: {},
};

const WATCHED = Object.keys(DEFAULTS);

let settings = { ...DEFAULTS };
let enabledHere = false;
let attached = false;

// Original sources, so a translated image can be toggled back.
const originals = new WeakMap();
/* The source set a translated image had before this script silenced it. See
 * setSrc: on a responsive image, writing `src` alone changes nothing at all. */
const stashed = new WeakMap();
// Images already handled, so auto mode does not loop on its own output.
const handled = new WeakSet();
const inFlight = new WeakSet();
/* Images auto mode has already failed on. `handled` cannot double as this: the
 * restore path deletes from it, and a failed image has no original to restore.
 * Without a record of the failure, one unreachable server turns every scrap of
 * page churn -- lazy loading, infinite scroll -- into another doomed run over
 * the same image, re-fetching and re-hashing its bytes each time. An explicit
 * click still retries, and so does the page swapping in a genuinely new
 * picture. */
let failed = new WeakSet();
/* Every attribute this script assigned, per element, kept until the observer has
 * seen it. Writing img.src -- or clearing a srcset, which is now part of the
 * same operation -- produces an attribute record indistinguishable from the page
 * swapping in a new picture, and that record used to wipe `handled` and
 * `originals` for the image we had just translated -- which in auto mode meant
 * re-uploading our own output forever, and in manual mode meant the second
 * click never found an original to restore. */
const selfWrites = new WeakMap();
/* Images whose bytes have not arrived yet. A lazy-loaded page is not "too
 * small", it is "not here yet", and rejecting it once was permanent because
 * nothing ever asked again. */
const watching = new WeakSet();

/* ------------------------------------------------------------- webtoon slices
 *
 * A webtoon is one tall strip and the host delivers it as N images, so a speech
 * bubble is free to run off the bottom of one and continue at the top of the
 * next. seam.js holds every geometric decision; this script's whole job is the
 * one question only it can answer -- are these two <img> elements contiguous
 * pieces of one picture -- and then handing the background script two URLs.
 */

/* Per translated image: the URL its translation is filed under, seam.js's
 * summary of the boxes touching its top and bottom edges, and which of those
 * edges have already been repaired. All three come back from the background on
 * a fresh translation and on a cache hit alike, so a strip read a second time
 * still joins without a single request reaching the server. */
const sliced = new WeakMap();

/* Boundaries already paid for, keyed by the two URLs, as key -> attempts. A join
 * costs a third full pipeline run, so a boundary that has been ANSWERED is never
 * asked again -- including "there was nothing to join", which is the common one
 * and would otherwise be re-asked on every mutation the page makes.
 *
 * A Map rather than a Set, and that is a fix rather than a flourish. This was a
 * Set added to in `performSeam`'s `finally`, which fires on every outcome --
 * a 507, an unreachable server, a failed refetch, `skipped: "uncached"` -- and
 * nothing anywhere ever deleted from it. So one transient failure disabled that
 * boundary for the life of the page view. The realistic trigger is the ordinary
 * one: the reader has not started the server yet, every boundary fails once,
 * and the seam then stays dead on that strip after the server comes up.
 *
 * An answer is recorded as SETTLED and never retried; a failure counts, and the
 * boundary is abandoned only after SEAM_MAX_ATTEMPTS of them. That keeps the
 * retry storm this guard exists to prevent bounded at three GPU pages per
 * boundary in the worst case, while letting a strip recover from a server that
 * was merely not running yet. */
const seamTried = new Map();
const seamRunning = new Set();

/* Boundaries that DEFERRED, keyed exactly as `seamTried` is, holding the two
 * images so the boundary can be asked again.
 *
 * WHY THIS EXISTS, and it is a measured gap rather than a new idea.
 * `seamJoinFor` returns `{wait: true}` when a name runs past what is translated,
 * and `runSeam` then returns without settling the boundary and without counting
 * a failure -- correctly, because the answer really is not known yet. But
 * `scheduleSeam` only ever asks the two boundaries TOUCHING the slice that just
 * landed, so a boundary that deferred is never revisited: the slice that would
 * have completed its run arrives, asks its own two boundaries, and the deferred
 * one is not among them.
 *
 * MEASURED on a webtoon test chapter in real Firefox, fresh profile, both
 * passes identical, over four consecutive slices k..k+3:
 *
 *     k    joined 2x  1200 x 1127     <- seamRunPlan(head=k, length=2)
 *     k+1  wait {above:false, below:true}
 *     k+2  wait {above:false, below:true}
 *     k+3  no-plan
 *
 * `seamRunPlan(head=k, length=4)` is constructible on that same data -- a
 * 1200 x 2943 plan -- and was never built. The column is
 * `苍炎之王·冥霜之王·裂风之王`; the two-slice join read `苍炎之王` and lettered
 * BLUE FLAME KING, and the other two names stayed on the page in Chinese because
 * `cross_slice_fragment` refuses to read a slice-height column unread until a
 * join makes it whole.
 *
 * SO THE DEFERRAL IS RIGHT AND THE MISSING HALF IS THE RE-ASK. `seamEdges`'
 * amendment 3 says so in its own words -- a hint-spanned end joins one slice
 * short because "joining one slice short on refused evidence costs a re-ask".
 * There was no re-ask. This is it.
 *
 * A Map rather than a Set because asking a boundary needs the two elements, and
 * a WeakMap will not do: this is iterated. It is bounded by the boundaries of
 * one page view that deferred, entries leave on the first non-deferring answer,
 * and the images are already retained by `allImages()` for the life of the
 * document -- so this holds no element the page was otherwise free to drop. */
const seamDeferred = new Map();

/** A boundary whose answer is known. Never re-asked. */
const SEAM_SETTLED = Number.POSITIVE_INFINITY;

/** Failed attempts before a boundary is abandoned for this page view. */
const SEAM_MAX_ATTEMPTS = 3;

/* Joins run one at a time, and this is a correctness guard rather than
 * politeness about the GPU.
 *
 * A MIDDLE slice of a strip belongs to two boundaries, and finishing it fires
 * both -- the join above it and the join below it -- in the same turn. Each one
 * reads that slice's stored translation, paints its own rectangle into it and
 * stores the result, so two of them in flight together both start from the
 * unjoined bytes and the second write throws the first join away. Serialising
 * makes the second read the first's output, which is what it should build on.
 * The server takes them one at a time regardless, so this costs nothing. */
let seamQueue = Promise.resolve();

/* How far apart two stacked images may sit and still count as one strip.
 *
 * Layout pixels, and fractional: a CSS-scaled strip can leave a hairline. Two is
 * generous enough for that and far too tight for a gallery with any gutter at
 * all -- which is the point, because a vertical-scroll MANGA reader stacks
 * equal-width pages exactly like this and must not be mistaken for a strip. The
 * bubble-side guard is seam.js's edge band; this is only the geometry. */
const SLICE_TOLERANCE_PX = 2;

/* A strip slice is a whole image, and both of them travel to the background as
 * byte arrays. This is the ceiling on that, per slice: past it the message costs
 * more than the join is worth, and a picture this large is not a webtoon slice. */
const SEAM_MAX_SOURCE_BYTES = 12 * 1024 * 1024;

let bar = null;
let tip = null;
let btn = null;
let autoBtn = null;
/* The re-roll button. Only shown over an image that HAS a translation --
 * `showBarFor` toggles it off everywhere else, because "retry" on an
 * untranslated image is just "translate". */
let retryBtn = null;
/* The box editor's button, same visibility rule plus one more: the reply must
 * have carried region boxes, which every translation from a current server
 * does. */
let editBtn = null;
/* A layer of per-image "working on this one" badges.
 *
 * The bar's own spinner follows the POINTER, so in auto mode -- which is exactly
 * when images the pointer never touches are being translated -- it says nothing
 * about which picture is busy. One badge is pinned over each in-flight image
 * instead. `position: fixed` and repositioned from onScroll, so it tracks a
 * scrolling reader without watching every image. */
let marks = null;
const marked = new Map();
/* The pin dialog: one shared node, built lazily like the bar, holding
 * whichever term an alt-click landed on. Interactive, unlike the marks. */
let pin = null;
let pinTerm = null;
let pinInput = null;
let pinChoices = null;
let pinNote = null;
let pinTimer = 0;
let target = null;      // image the bar currently belongs to
let hideTimer = 0;
let errorTimer = 0;
let lastError = { message: "", at: 0 };

let pointerX = -1;
let pointerY = -1;
let resolveFrame = 0;
let lastScan = { x: -1, y: -1, at: 0, hit: null };

/* ---------------------------------------------------------------- utilities */

const LOGO = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="2" y="3" width="20" height="15" rx="3" fill="#2f7cf6"/>
  <path d="M8 21l3-3H8z" fill="#2f7cf6"/>
  <text x="7.4" y="14.2" font-family="system-ui,sans-serif" font-size="9" font-weight="700" fill="#fff" text-anchor="middle">あ</text>
  <text x="16.4" y="14.2" font-family="system-ui,sans-serif" font-size="9" font-weight="700" fill="#fff" text-anchor="middle">A</text>
</svg>`;

const SPINNER = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="9" stroke="#c9ced6" stroke-width="3"/>
  <path d="M21 12a9 9 0 0 0-9-9" stroke="#2f7cf6" stroke-width="3" stroke-linecap="round"/>
</svg>`;

/* Parsed rather than assigned to innerHTML. Both strings above are our own
 * literals, so innerHTML would be safe -- but web-ext flags every use of it and
 * cannot prove otherwise, and a warning that is always present is one nobody
 * reads when it finally matters. importNode because DOMParser hands back nodes
 * belonging to its own document. */
function icon(markup) {
  const parsed = new DOMParser().parseFromString(markup, "image/svg+xml");
  return document.importNode(parsed.documentElement, true);
}

/* `document.images` is an HTMLDocument property and is simply absent in an XML
 * or SVG document, which a frame is allowed to be. */
const allImages = () => document.images || document.getElementsByTagName("img");

const minSize = () => Number(settings.minSize) || DEFAULTS.minSize;

/* Two sizes, not one. `naturalWidth` is 0 until the image decodes, and manga
 * hosts lazy-load with a 1x1 placeholder sitting in `src`, so an
 * intrinsic-size-only test rejects the very pages this extension exists for.
 * The rendered box is what the reader laid out and it is correct long before
 * the bytes land. */
function measure(img, rect) {
  const r = rect || img.getBoundingClientRect();
  return {
    w: Math.max(img.naturalWidth || 0, Math.round(r.width)),
    h: Math.max(img.naturalHeight || 0, Math.round(r.height)),
  };
}

function bigEnough(img, rect) {
  const { w, h } = measure(img, rect);
  const min = minSize();
  return w >= min && h >= min;
}

/* Codes rather than prose, because the diagnostic reports which clause rejected
 * an image and a count keyed on a sentence is a count waiting to drift. */
function rejection(img, rect) {
  if (!(img instanceof HTMLImageElement)) return "notimg";
  if (!img.isConnected) return "detached";
  if (img.dataset.khrSkip) return "skip";
  if (bigEnough(img, rect)) return "";
  return img.complete && img.naturalWidth ? "small" : "pending";
}

function eligible(img, rect) {
  return rejection(img, rect) === "";
}

/* "Worth a button" and "worth uploading" are two questions, and answering both
 * with the rendered box is what turns a lazy-loading reader into a GPU heater.
 * A lazy-loading reader theme fills an 800x1200 slot with a 1x1 GIF sitting in
 * `src` while the real URL waits in `data-src`: the box test passes, so the old
 * code uploaded 43 bytes, spent a run, and then overwrote the very `src` the
 * site's own loader keys off -- leaving the panel permanently blank. This
 * second test asks what the browser is actually holding, and only the intrinsic
 * size can answer that. */
function uploadBlock(img) {
  if (!(img instanceof HTMLImageElement)) return "loading";
  if (!img.complete || !img.naturalWidth || !img.naturalHeight) return "loading";
  const min = minSize();
  return img.naturalWidth < min || img.naturalHeight < min ? "placeholder" : "";
}

/* Stated rather than inferred. A 1x1 GIF in an 800x1200 slot is a lazy
 * placeholder, but the same clause also catches a genuinely small picture when
 * the minimum has been raised -- and calling that a placeholder would be a lie
 * the user cannot check. The numbers are true in both cases. */
function blockText(img, blocked) {
  if (blocked === "loading") return "Still loading — try again in a moment.";
  return (
    `The browser only has a ${img.naturalWidth}×${img.naturalHeight} version of ` +
    `this image, under the ${minSize()}px minimum. If the page loads its images ` +
    `lazily, scroll this one into view first.`
  );
}

/* --------------------------------------------------- writing back to the page
 *
 * Setting `src` is not on its own enough to change what the reader sees. When an
 * <img> carries a `srcset` with width descriptors, or sits inside a <picture>
 * with a matching <source>, HTML's source-set selection never considers the
 * `src` attribute at all: writing it re-runs selection, the browser picks the
 * same candidate it already had, and the translation lands nowhere with no error
 * anywhere. Every responsive-image reader is that shape -- WordPress reader
 * themes, and anything built on next/image.
 *
 * So the sources that outrank `src` are stashed and cleared first, and put back
 * verbatim when the user toggles the original back. `sizes` is deliberately left
 * alone: with no srcset to size, it selects nothing.
 */

/* Records an attribute this script is about to write, so the observer can tell
 * it from the page's own. `null` means "we removed it", which is exactly what
 * getAttribute returns for an attribute that is not there. */
function noteWrite(el, name, value) {
  let mine = selfWrites.get(el);
  if (!mine) {
    mine = new Map();
    selfWrites.set(el, mine);
  }
  mine.set(name, value);
}

/* True exactly once per write we made, for the record that write produced. */
function wasOurWrite(el, name) {
  const mine = selfWrites.get(el);
  if (!mine || !mine.has(name)) return false;
  if (el.getAttribute(name) !== mine.get(name)) return false;
  mine.delete(name);
  if (!mine.size) selfWrites.delete(el);
  return true;
}

/* Guarded so a write that changes nothing is never announced: removeAttribute on
 * an absent attribute produces no mutation record at all, and the unclaimed note
 * it left behind would sit in the ledger waiting to swallow one of the page's
 * own writes. */
function writeAttr(el, name, value) {
  if (value === null) {
    if (!el.hasAttribute(name)) return;
  } else if (el.getAttribute(name) === value) {
    return;
  }
  noteWrite(el, name, value);
  if (value === null) el.removeAttribute(name);
  else el.setAttribute(name, value);
}

/* Only the <source> children of a wrapping <picture> outrank an img's own
 * srcset, and only a <picture> can have them. */
function pictureSources(img) {
  const parent = img.parentElement;
  if (!parent || parent.tagName !== "PICTURE") return [];
  return Array.from(parent.getElementsByTagName("source"));
}

function silenceSources(img) {
  if (!stashed.has(img)) {
    stashed.set(img, {
      srcset: img.getAttribute("srcset"),
      sources: pictureSources(img).map((el) => [el, el.getAttribute("srcset")]),
    });
  }
  // The <source>s first: they beat the img's own srcset, which beats src.
  for (const [el] of stashed.get(img).sources) writeAttr(el, "srcset", null);
  writeAttr(img, "srcset", null);
}

function restoreSources(img) {
  const stash = stashed.get(img);
  if (!stash) return;
  stashed.delete(img);
  for (const [el, value] of stash.sources) {
    if (el.isConnected) writeAttr(el, "srcset", value);
  }
  writeAttr(img, "srcset", stash.srcset);
}

/* The only place this script is allowed to set a src. `mode` says which picture
 * is going up, because that is what decides the fate of the sources outranking
 * it. Recording the value first is what lets the observer tell our write apart
 * from the page's; a dataset flag would work too, but the page can read and
 * clear that one. */
function setSrc(img, value, mode = "translated") {
  if (mode === "original") restoreSources(img);
  else silenceSources(img);
  noteWrite(img, "src", value);
  img.src = value;
}

/* -------------------------------------------------------------- overlay UI */

function buildBar() {
  if (!bar) {
    bar = document.createElement("div");
    bar.className = "khr-bar";

    btn = document.createElement("button");
    btn.className = "khr-btn";
    btn.type = "button";
    btn.replaceChildren(icon(LOGO));
    btn.addEventListener("click", onTranslateClick);
    btn.addEventListener("mouseenter", showTip);
    btn.addEventListener("mouseleave", hideTip);

    /* Between the logo and the A: a re-roll of THIS image's translation.
     * Distinct from the logo button on purpose -- that one's second click is
     * the restore toggle, and a retry must reach the server from exactly the
     * state the toggle intercepts. */
    retryBtn = document.createElement("button");
    retryBtn.className = "khr-retry";
    retryBtn.type = "button";
    retryBtn.textContent = "↻";
    retryBtn.title = "Retry translation — ask for a different draw";
    retryBtn.addEventListener("click", onRetryClick);

    /* The box editor: add a box the detector missed, delete a
     * false one, retranslate with the correction. */
    editBtn = document.createElement("button");
    editBtn.className = "khr-editbox";
    editBtn.type = "button";
    editBtn.textContent = "▦";
    editBtn.title = "Edit detection boxes — add missed text, delete false boxes";
    editBtn.addEventListener("click", onEditBoxesClick);

    autoBtn = document.createElement("button");
    autoBtn.className = "khr-auto";
    autoBtn.type = "button";
    autoBtn.textContent = "A";
    autoBtn.title = "Auto-translate images as they appear";
    autoBtn.addEventListener("click", onAutoClick);

    bar.append(btn, retryBtn, editBtn, autoBtn);

    // Keep the bar alive while the pointer is on it, not just on the image.
    bar.addEventListener("mouseenter", () => clearTimeout(hideTimer));
    bar.addEventListener("mouseleave", scheduleHide);

    tip = document.createElement("div");
    tip.className = "khr-tip";

    marks = document.createElement("div");
    marks.className = "khr-marks";
  }

  /* Re-mounted, not just built once. An SPA route change that rewrites
   * document.body takes our nodes out with it, and the old `if (bar) return`
   * then reported success forever while showBarFor added a class to a node in
   * no document at all -- silent, permanent, and indistinguishable from
   * "nothing was detected". */
  const root = document.body || document.documentElement;
  if (root && (!bar.isConnected || !tip.isConnected || !marks.isConnected)) {
    root.append(bar, tip, marks);
  }

  syncAuto();
  return Boolean(bar.isConnected && tip.isConnected);
}

/* Pins each badge over its image. Cheap because it only ever runs while
 * something is in flight, and a page with nothing translating pays one
 * `Map.size` check per scroll event. */
function positionMarks() {
  if (!marked.size) return;
  for (const [img, entry] of marked) {
    if (!img.isConnected) {
      entry.el.remove();
      marked.delete(img);
      continue;
    }
    const r = img.getBoundingClientRect();
    // Off-screen or collapsed: hide rather than park a badge at 0,0.
    if (r.width < 1 || r.height < 1) {
      entry.el.style.display = "none";
      continue;
    }
    entry.el.style.display = "";
    entry.el.style.left = `${Math.round(r.left + r.width / 2)}px`;
    entry.el.style.top = `${Math.round(r.top + r.height / 2)}px`;
  }
}

/* One shared driver for every badge's elapsed readout, not one per badge --
 * the same rule hideTimer and resolveFrame already follow. Chained setTimeout
 * rather than setInterval so an emptied map stops the chain by itself, and a
 * page with nothing in flight pays nothing. The text writes land on children
 * of `marks`, which `isOurs()` covers, so the MutationObserver never sees its
 * own ticking as page churn. */
let markTicker = 0;

function tickMarks() {
  markTicker = 0;
  if (!marked.size) return;
  for (const entry of marked.values()) {
    const seconds = Math.floor((performance.now() - entry.startedAt) / 1000);
    entry.readout.textContent =
      entry.kind === "seam" ? `join ${seconds}s` : `${seconds}s`;
  }
  markTicker = setTimeout(tickMarks, 500);
}

/* Raised the moment an image is actually going to be worked on, and lowered in
 * translate's `finally` so a failure or an early return cannot strand one.
 * `kind` says WHY the image is busy -- "seam" pins the join badge on every
 * slice of the run being composed, which is what tells the reader a boundary
 * is still being worked after both slices already lettered. */
function markBusy(img, kind) {
  if (marked.has(img) || !buildBar()) return;
  const mark = document.createElement("div");
  mark.className = kind === "seam" ? "khr-mark khr-seam" : "khr-mark";
  mark.append(icon(SPINNER));
  const readout = document.createElement("span");
  readout.className = "khr-elapsed";
  readout.textContent = kind === "seam" ? "join 0s" : "0s";
  mark.append(readout);
  marks.append(mark);
  marked.set(img, { el: mark, readout, startedAt: performance.now(), kind });
  positionMarks();
  if (!markTicker) markTicker = setTimeout(tickMarks, 500);
}

function markDone(img) {
  const entry = marked.get(img);
  if (!entry) return;
  entry.el.remove();
  marked.delete(img);
}

/* --- The pin dialog ------------------------------------------------------
 * Alt-click a lettered term and choose what it should say, entirely in the
 * reader's own language: the dialog offers every rendering the session's
 * ledger has seen for that source term, plus free text. Pinning goes through
 * the background's glossary-pin message -- the same parse/validate/store
 * path as the popup's Apply -- and takes effect as pages re-render on their
 * next view, because the pin changes the cache fingerprint. */

function buildPin() {
  if (!pin) {
    pin = document.createElement("div");
    pin.className = "khr-pin";
    const title = document.createElement("div");
    title.className = "khr-pin-title";
    title.textContent = "Pin this term";
    pinChoices = document.createElement("div");
    pinChoices.className = "khr-pin-choices";
    pinInput = document.createElement("input");
    pinInput.className = "khr-pin-input";
    pinInput.type = "text";
    pinInput.spellcheck = false;
    const acts = document.createElement("div");
    acts.className = "khr-pin-acts";
    const pinBtn = document.createElement("button");
    pinBtn.type = "button";
    pinBtn.className = "khr-pin-btn";
    pinBtn.textContent = "Pin";
    pinBtn.addEventListener("click", submitPin);
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "khr-pin-btn khr-pin-cancel";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", closePin);
    acts.append(pinBtn, cancel);
    pinNote = document.createElement("div");
    pinNote.className = "khr-pin-note";
    pin.append(title, pinChoices, pinInput, acts, pinNote);
  }
  const root = document.body || document.documentElement;
  if (root && !pin.isConnected) root.append(pin);
  return Boolean(pin.isConnected);
}

/* One rendering the reader may pick. A button and not a link: picking fills
 * the input rather than pinning outright, so the choice can still be edited
 * before it commits. */
function pinChoice(text, count) {
  const choice = document.createElement("button");
  choice.type = "button";
  choice.className = "khr-pin-btn khr-pin-choice";
  choice.textContent = count > 1 ? `${text} ×${count}` : text;
  choice.addEventListener("click", () => {
    pinInput.value = text;
    pinInput.focus();
  });
  return choice;
}

function openPin(term, x, y) {
  if (!buildPin()) return;
  pinTerm = term;
  pinNote.textContent = "";
  pinNote.classList.remove("khr-bad");
  pinInput.value = term.translated;
  pinChoices.replaceChildren();
  const sourceLine = document.createElement("div");
  sourceLine.className = "khr-pin-source";
  sourceLine.textContent = term.source;
  pinChoices.append(sourceLine);
  pin.classList.add("khr-visible");
  const pad = 10;
  const width = 260;
  pin.style.left = `${Math.round(Math.min(Math.max(pad, x), innerWidth - width - pad))}px`;
  pin.style.top = `${Math.round(Math.min(Math.max(pad, y + 12), innerHeight - 180))}px`;
  document.addEventListener("keydown", onPinKey, true);
  document.addEventListener("mousedown", onPinAway, true);
  /* The ledger's other renderings arrive after the dialog opens -- the open
   * must not wait on a message round trip, and an empty choice row is the
   * honest state for a term seen once. */
  void browser.runtime
    .sendMessage({ type: "glossary-pin-options", source: term.source })
    .then((reply) => {
      if (!reply || !reply.ok || pinTerm !== term) return;
      for (const one of reply.renderings || []) {
        if (one.text !== term.translated) pinChoices.append(pinChoice(one.text, one.count));
      }
      if (reply.pinned && reply.pinned !== term.translated) {
        pinInput.value = reply.pinned;
      }
    })
    .catch(() => {});
  log("pin", "open", term.source, term.translated);
}

function closePin() {
  pinTerm = null;
  if (pinTimer) {
    clearTimeout(pinTimer);
    pinTimer = 0;
  }
  if (pin) pin.classList.remove("khr-visible");
  document.removeEventListener("keydown", onPinKey, true);
  document.removeEventListener("mousedown", onPinAway, true);
}

function onPinKey(event) {
  if (event.key === "Escape") closePin();
  if (event.key === "Enter" && event.target === pinInput) submitPin();
}

function onPinAway(event) {
  if (!isOurs(event.target)) closePin();
}

async function submitPin() {
  if (!pinTerm) return;
  const translation = pinInput.value.trim();
  if (!translation) return;
  const reply = await browser.runtime
    .sendMessage({ type: "glossary-pin", source: pinTerm.source, translation })
    .catch((err) => ({ ok: false, error: String((err && err.message) || err) }));
  if (reply && reply.ok) {
    log("pin", "pinned", pinTerm.source, translation);
    pinNote.textContent = "Pinned - pages re-render as you revisit them.";
    pinTimer = setTimeout(closePin, 1800);
  } else {
    const errors = reply && Array.isArray(reply.errors) ? reply.errors.join("; ") : "";
    pinNote.textContent = errors || (reply && reply.error) || "pin failed";
    pinNote.classList.add("khr-bad");
  }
}

/* The alt-click that opens it. Capturing, so a host's own click handling
 * cannot swallow the gesture; a click that hits no term box falls through to
 * the page untouched. */
function onPinClick(event) {
  if (!event.altKey || !event.isTrusted) return;
  const img = imageAt(event.clientX, event.clientY);
  if (!img) return;
  const entry = sliced.get(img);
  if (!entry || !entry.terms) return;
  const r = img.getBoundingClientRect();
  if (r.width < 1 || r.height < 1) return;
  const scaleX = (img.naturalWidth || r.width) / r.width;
  const scaleY = (img.naturalHeight || r.height) / r.height;
  const hit = glossaryPinHit(
    entry.terms,
    (event.clientX - r.left) * scaleX,
    (event.clientY - r.top) * scaleY
  );
  if (!hit) {
    log("pin", "no-term-at-point", entry.url);
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  openPin(hit, event.clientX, event.clientY);
}

function positionBar() {
  if (!bar || !target || !bar.isConnected) return;
  const r = target.getBoundingClientRect();
  const pad = 8;
  const w = bar.offsetWidth || 60;
  const h = bar.offsetHeight || 34;

  // Read defensively: a corner that is not a string throws here, and a throw in
  // the positioner is a bar that never appears with nothing to show for it.
  const corner =
    typeof settings.corner === "string" ? settings.corner : DEFAULTS.corner;
  const left = corner.endsWith("left") ? r.left + pad : r.right - w - pad;
  const top = corner.startsWith("top") ? r.top + pad : r.bottom - h - pad;

  // Clamp into the viewport so the control never sits off-screen on
  // partially-scrolled images. In a subframe "the viewport" is the frame's own
  // box, which is exactly where a position:fixed child of that frame paints.
  bar.style.left = `${Math.max(2, Math.min(left, innerWidth - w - 2))}px`;
  bar.style.top = `${Math.max(2, Math.min(top, innerHeight - h - 2))}px`;

  if (tip.classList.contains("khr-visible")) positionTip();
}

function positionTip() {
  const r = bar.getBoundingClientRect();
  tip.style.left = `${r.left}px`;
  tip.style.top = `${r.bottom + 6}px`;
}

function showTip() {
  if (!target) return;
  tip.textContent = originals.has(target) ? "Show original" : "Translate image";
  tip.classList.remove("khr-bad");
  tip.classList.add("khr-visible");
  positionTip();
}

function hideTip() {
  if (tip) tip.classList.remove("khr-visible");
}

function showBarFor(img) {
  if (!buildBar()) return;
  target = img;
  /* Recomputed on every show rather than kept in sync from translate(): the
   * bar re-shows on each hover, so a just-finished translation earns its
   * retry button the next time the pointer arrives. A stale click either way
   * is harmless -- retryTranslate falls back to an ordinary translate, and
   * the editor explains itself when there is nothing to edit yet. */
  if (retryBtn) retryBtn.classList.toggle("khr-usable", originals.has(img));
  if (editBtn) {
    editBtn.classList.toggle("khr-usable", originals.has(img) && regionBoxes.has(img));
  }
  bar.classList.add("khr-visible");
  positionBar();
}

function scheduleHide() {
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => {
    if (bar) bar.classList.remove("khr-visible");
    hideTip();
    target = null;
  }, 180);
}

function syncAuto() {
  if (autoBtn) autoBtn.classList.toggle("khr-on", settings.autoTranslate);
}

function setBusy(on) {
  if (!btn) return;
  btn.classList.toggle("khr-busy", on);
  btn.replaceChildren(icon(on ? SPINNER : LOGO));
}

/* Errors have to be legible without a hover: auto mode translates images the
 * pointer never touches, so a failure that only reached the console -- or that
 * bailed out on a missing button -- was a silent failure to the user. Build the
 * bar if it does not exist yet and anchor the message to the image itself. */
function flashError(message, anchor, hold = 2600) {
  const now = Date.now();
  // Auto mode can fail on a whole gallery at once. Say it once.
  if (message === lastError.message && now - lastError.at < 3000) return;
  lastError = { message, at: now };

  if (!buildBar()) {
    // Nothing to render into. Better a console line than a thrown error that
    // buries whatever went wrong in the first place.
    console.warn("[koharu]", message);
    return;
  }
  if (btn) btn.classList.add("khr-error");

  tip.textContent = message;
  tip.classList.add("khr-visible", "khr-bad");

  if (target && bar.classList.contains("khr-visible")) positionTip();
  else positionTipNear(anchor);

  clearTimeout(errorTimer);
  errorTimer = setTimeout(() => {
    if (btn) btn.classList.remove("khr-error");
    tip.classList.remove("khr-bad");
    hideTip();
  }, hold);
}

function positionTipNear(anchor) {
  const box =
    anchor && anchor.isConnected
      ? anchor.getBoundingClientRect()
      : { left: 12, top: 12 };
  const w = tip.offsetWidth || 0;
  const h = tip.offsetHeight || 22;
  tip.style.left = `${Math.max(4, Math.min(box.left + 8, innerWidth - w - 4))}px`;
  tip.style.top = `${Math.max(4, Math.min(box.top + 8, innerHeight - h - 4))}px`;
}

/* ------------------------------------------------------------- translation */

/* Fetch bytes from the page context, not the background script: manga hosts
 * commonly gate images on Referer or cookies, and a background fetch carries
 * neither. It is also the only side that can read a `blob:` URL the page
 * minted, which is how some readers hand their pages over.
 * Canvas is only a fallback, since cross-origin images taint it.
 *
 * The URL is passed in rather than re-read here, because the background script
 * files the translation under it -- the bytes and the URL they are filed under
 * have to be the same one even if the page swaps the src mid-flight. */
async function imageBytes(img, src) {
  try {
    const res = await fetch(src, {
      credentials: "include",
      referrer: location.href,
    });
    if (res.ok) return await res.blob();
  } catch {
    /* fall through to canvas */
  }

  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext("2d").drawImage(img, 0, 0);
  return await new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("canvas tainted"))), "image/png")
  );
}

/* The server's miss report, said out loud. A partially translated page is the
 * one failure that looks like success: the bubbles it did translate render
 * perfectly, and the ones it did not are still Japanese, which is exactly what
 * an OCR miss looks like. The server counts them and names the cause; this is
 * the only place the reader can be told. Not an error -- the page is readable,
 * and the run is not worth repeating -- so it goes out on the same bar with a
 * longer hold. */
function noteMisses(reply, img) {
  const text = missNotes(reply);
  if (text) flashError(text, img, 5000);
}

/* The sentence itself, split out from the bar it goes on.
 *
 * Not tidiness -- it is the only way this is testable at all. `noteMisses` ends
 * in `flashError`, which is DOM, so a test could only ever assert that the
 * function was *called*; wrapping the guard and the wording in something that
 * RETURNS its text lets `tests/drop-report.test.js` lift it into a sandbox and
 * read what the reader would actually see. The guard shipped once as
 * `if (false && reply.dropped)` with the suite still green, which is the same
 * shape of defect as a server field nothing reads.
 *
 * Returns null when there is nothing to say, so a clean page raises no bar. */
function missNotes(reply) {
  const overflowed =
    reply && Array.isArray(reply.placementOverflow) && reply.placementOverflow.length;
  if (!reply
      || (!reply.missed && !reply.dropped && !reply.cut && !reply.stillCut
          && !reply.keepArtIgnored && !overflowed)) {
    return null;
  }
  const parts = [];
  const plural = (n) => (n === 1 ? "" : "s");
  if (reply.missed) parts.push(`${reply.missed} regions left untranslated`);
  /* A still-cut segment is a defect on the page BY ITSELF -- a bubble
   * ends mid-sentence right now -- so it sits in the guard beside `dropped`,
   * not as a rider on `missed`. `cutFound` alone (cut and REPAIRED) is
   * deliberately silent here: the page in front of the reader is fine, and a
   * warning on it would cry wolf; the count still rides the wire and the cache
   * for the Diagnose panel. */
  if (reply.stillCut) {
    parts.push(`${reply.stillCut} speech segment${plural(reply.stillCut)} cut off `
               + "mid-sentence, and the retry could not repair them");
  }
  /* IN THE GUARD, beside `missed`, and not in the `missed || cut` branch below
   * where the two id counters live. A repeated id is only a CAUSE -- it explains
   * a miss and says nothing on its own. A drop is a defect on the page by
   * itself: the region was read and nothing was painted back for it.
   *
   * It is also the only one of these the page cannot show for itself. An
   * untranslated region is visibly Japanese and a truncated bubble ends
   * mid-word; a drop leaves no evidence at all. Measured on 59 regions over 22
   * pages, every one of which the server called clean on all four of its other
   * counters.
   *
   * **Deliberately NOT "erased and left blank", which is what this said first
   * and which is false half the time.** Erasure is the inpainting stage's doing,
   * and "Keep the artwork" runs the pipeline WITHOUT it -- so on that plan
   * nothing is erased and a dropped region leaves the original Japanese sitting
   * there untouched, which is the opposite picture. Telling that reader to go
   * looking for a blank patch sends them hunting for something that is not on
   * the page. What is true under both plans is the narrow claim: it was read,
   * and nothing was lettered. The reader can see which of the two they are
   * looking at; they cannot see that the server had text and dropped it. */
  if (reply.dropped) {
    parts.push(`${reply.dropped} regions came back empty, so nothing was lettered for them`);
  }
  /* The placement box is a hard fit box all the way through the size
   * search, but the search never goes below the 9px floor -- so a box too
   * small for its line SPILLS rather than shrinking into illegibility, and
   * the honest move is to say so and ask for a bigger box. The placement
   * itself stays stored, so reopening the editor shows it for enlarging. */
  if (overflowed) {
    const n = reply.placementOverflow.length;
    parts.push(
      `${n} placement box${n === 1 ? " is" : "es are"} too small for the text ` +
      "at readable size — draw a bigger one in the box editor"
    );
  }
  if (reply.cut) parts.push("the model's reply hit its token cap");
  /* The CAUSE of a miss, and only ever as a rider on one -- which is why these
   * two are inside the `missed || cut` branch and are not in the guard above.
   *
   * A mis-addressed id is not by itself a defect on the page. Under
   * `--provider local` the grammar pins the number of reply entries, so a
   * repeated id always steals a slot from another segment and `missed` is set
   * alongside it -- that is the case measured on a real volume, three pages of
   * 205, every one losing its last segment. But an unconstrained provider
   * (`openai_compatible` sends `PromptOnly`, so `--provider ollama`) can answer
   * MORE entries than it was asked for, cover every id, and still have named one
   * twice. `translation.rs` says of exactly that page: nothing is wrong with it.
   * Announcing it on its own would put a red bar and "worth re-running" on a page
   * that is completely correct, which is worse than saying nothing.
   *
   * Deliberately NOT phrased as "N of the untranslated regions": a repeated id
   * drops one reply ENTRY, and the segment that goes unanswered is usually but
   * not necessarily one-for-one with it. The count of mis-addressed entries is
   * what was measured; arithmetic over it is not. */
  if (reply.duplicateIds) {
    parts.push(`the reply repeated ${reply.duplicateIds} id${plural(reply.duplicateIds)}`);
  }
  if (reply.outOfRangeIds) {
    parts.push(`the reply named ${reply.outOfRangeIds} id${plural(reply.outOfRangeIds)} `
               + "the page does not have");
  }
  /* Said out loud because the page looks entirely normal otherwise: a correct
   * translation over erased artwork is what the reader asked this option to
   * avoid, and nothing else on screen distinguishes it. This one is never
   * replayed from the cache because a page that ignored the switch is not
   * stored -- so hearing it means the server that just answered is too old. */
  if (reply.keepArtIgnored) {
    parts.push("the server erased the artwork: it predates Keep the artwork");
  }
  return parts.join(" -- ");
}

/* Everything the join needs to know about a page that has just been shown. The
 * same three fields arrive from a live translation and from a cache hit. */
function noteSlice(img, url, reply) {
  sliced.set(img, {
    url,
    edges: (reply && reply.edges) || null,
    seamed: (reply && Array.isArray(reply.seamed) && reply.seamed) || [],
    /* The pinnable term boxes, in the translated image's own pixel
     * space, so an alt-click resolves locally with no round trip. */
    terms: (reply && Array.isArray(reply.terms) && reply.terms) || null,
  });
}

/* The image directly above or below this one in the same strip.
 *
 * Layout geometry rather than DOM order, and that is the stronger signal: a
 * reader is free to wrap each slice in its own div, lazy-load them out of order
 * or reuse one element, but two pieces of one picture are drawn edge to edge at
 * the same x and the same width or the strip does not look like a strip. Both
 * are also required to have been translated already, which is what makes this
 * symmetric -- whichever of the two finishes second finds the other.
 *
 * Intrinsic width is checked as well as laid-out width because the boxes seam.js
 * matches are in the source image's own pixels: two slices scaled to the same
 * CSS width from different intrinsic widths are not one strip. */
/* How many flush, equal-width images in a row before a document counts as a
 * vertical strip. Three rather than two because two stacked pages with no gutter
 * is an ordinary thing for a paged reader to do at a chapter boundary, and one
 * accidental pair must not decide a whole host's OCR engine. */
const SHAPE_MIN_RUN = 3;

/* `"strip"` or `"paged"` — the page's own shape, for the engine latch.
 *
 * # This is NOT `sliceNeighbour`, and it cannot be
 *
 * `sliceNeighbour` requires `sliced.has(other)` — the neighbour must ALREADY
 * have been translated — because it exists to rejoin a bubble after the fact.
 * That precondition makes it useless as an up-front classifier: on the first
 * image of a chapter nothing has been translated yet, so it would answer "no
 * neighbour" on a webtoon. The geometry below is the same four tests
 * (identical intrinsic width, aligned left edge, equal laid-out width, gap
 * within `SLICE_TOLERANCE_PX`) with that precondition removed.
 *
 * # It reports the LONGEST run, not "are all images flush"
 *
 * One advertisement in the middle of a strip would otherwise flip a webtoon to
 * `paged`, and `paged` is the answer that earns `manga-ocr`. The longest run
 * survives an interruption; requiring every image would not.
 *
 * # ITS FALSE POSITIVE IS KNOWN, DOCUMENTED, AND DELIBERATELY THE CHEAP ONE
 *
 * A vertical-scroll MANGA reader with no gutters stacks equal-width pages
 * exactly like a strip — `content.js`'s own `SLICE_TOLERANCE_PX` comment and
 * `seam.js`'s header both say so, and the seam only escapes it by following the
 * geometry with a second, independent bubble-edge guard. There is no second
 * guard here. So this WILL call some gutterless paged readers a strip, and the
 * consequence is that they get PaddleOCR-VL: ~2 s a page slower, reversible, and
 * measured at zero accuracy cost. The opposite error — manga-ocr on a webtoon —
 * costs silent fabrication over artwork. The classifier is built to fail toward
 * the cheap side rather than to be right more often.
 */
function pageShape() {
  const boxes = [];
  for (const img of allImages()) {
    if (!img.naturalWidth) continue;
    const rect = img.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) continue;
    boxes.push({
      natural: img.naturalWidth,
      left: rect.left,
      width: rect.width,
      top: rect.top,
      bottom: rect.bottom,
    });
  }
  if (boxes.length < SHAPE_MIN_RUN) return "paged";
  boxes.sort((a, b) => a.top - b.top);
  let best = 1;
  let run = 1;
  for (let i = 1; i < boxes.length; i += 1) {
    const a = boxes[i - 1];
    const b = boxes[i];
    const flush =
      a.natural === b.natural &&
      Math.abs(b.left - a.left) <= SLICE_TOLERANCE_PX &&
      Math.abs(b.width - a.width) <= SLICE_TOLERANCE_PX &&
      Math.abs(b.top - a.bottom) <= SLICE_TOLERANCE_PX;
    run = flush ? run + 1 : 1;
    if (run > best) best = run;
  }
  return best >= SHAPE_MIN_RUN ? "strip" : "paged";
}

/* `accept` widens this from "the translated slice next to me" to "the PICTURE
 * next to me, translated or not", which is a different question with a different
 * use. The default is unchanged and is the one the join itself asks. The wider
 * form answers `sliceChain`'s: a name running off the end of what we hold must
 * WAIT if there is a picture there and be joined now if there is not, and a
 * lookup that only ever sees translated slices cannot tell those apart -- it
 * reports "nothing there" for a slice that is merely next in the queue. */
function sliceNeighbour(img, below, accept) {
  const rect = img.getBoundingClientRect();
  if (!(rect.width > 0 && rect.height > 0)) return null;
  const want = accept || ((other) => sliced.has(other));

  for (const other of allImages()) {
    if (other === img || !want(other)) continue;
    if (!other.naturalWidth || other.naturalWidth !== img.naturalWidth) continue;
    const box = other.getBoundingClientRect();
    if (!(box.width > 0 && box.height > 0)) continue;
    if (Math.abs(box.left - rect.left) > SLICE_TOLERANCE_PX) continue;
    if (Math.abs(box.width - rect.width) > SLICE_TOLERANCE_PX) continue;
    const gap = below ? box.top - rect.bottom : rect.top - box.bottom;
    if (Math.abs(gap) > SLICE_TOLERANCE_PX) continue;
    return other;
  }
  return null;
}

/* The source bytes again, for a picture whose element now holds the
 * translation.
 *
 * `imageBytes` is deliberately not reused: its canvas fallback draws the <img>,
 * which at this point is our own output, so a failed fetch would quietly send
 * the server the page it produced instead of the page it was given. A refetch
 * that does not work simply means no join, and the browser's own HTTP cache
 * makes the ordinary case free. */
async function sliceSource(img) {
  const url = originals.get(img);
  if (!url) return null;
  const res = await fetch(url, { credentials: "include", referrer: location.href });
  if (!res.ok) return null;
  const blob = await res.blob();
  if (!blob.size || blob.size > SEAM_MAX_SOURCE_BYTES) return null;
  const buffer = await blob.arrayBuffer();
  // A Uint8Array for the same reason as translate()'s upload: the plain-array
  // boxing this replaced was the extension's most expensive marshal.
  return { url, mime: blob.type || "image/png", bytes: new Uint8Array(buffer) };
}

/* Put a rejoined page on screen. Not through translate(), which would read this
 * as the second click that restores the original: `originals` and `handled` are
 * both still correct and must stay untouched, so this is a plain second write of
 * a src we already own. One write, one ledger note, one mutation record -- which
 * is what the observer's filter can account for. */
function applySeam(img, dataUrl, url) {
  if (!img.isConnected || inFlight.has(img)) return false;
  /* The picture may have moved on: a reader that swaps one persistent <img> from
   * page to page does exactly that, and painting a join onto the wrong page
   * would be indistinguishable from a rendering bug.
   *
   * `handled` is the load-bearing half and `originals` alone was not enough.
   * Both are written from the same captured `src`, so comparing `originals`
   * against the URL the join was planned for can only catch a restore click or a
   * re-translation -- never the page swapping the picture underneath us, which
   * is the case this guard is for. The observer's attributes branch deletes
   * `handled` on exactly that event, which is why `sweep` uses it as the same
   * signal. */
  if (!handled.has(img) || originals.get(img) !== url) return false;
  setSrc(img, dataUrl);
  return true;
}

/* The already-translated slices around this boundary, in strip order, plus
 * whether the strip continues past each end with a picture we cannot use yet.
 *
 * CONTIGUOUS BY CONSTRUCTION, which is why this truncates and never filters. A
 * slice that is translated but has NO edge geometry is a real case -- it is the
 * `no-edges` defect logged above -- and dropping it from the middle of the list
 * would make two slices that are not neighbours adjacent in `edgesList`. Every
 * consumer downstream treats consecutive entries as adjacent: `seamRunPlan`
 * would plan across the gap and `seamImage` would compose a seam with a whole
 * slice missing from its middle, then paint that back. So the walk stops at the
 * first unusable neighbour rather than stepping over it.
 *
 * `pending` is what separates "wait" from "this is all there will ever be". A
 * name running off the end of the chain means the next slice matters; whether it
 * EXISTS is a question only the document can answer, and `seamJoinFor` cannot
 * ask it. At the top or bottom of the page nothing more is coming, and a join
 * deferred there is a join lost.
 *
 * Bounded by SEAM_MAX_RUN_SLICES in both directions rather than walked to the
 * end of the strip: this runs on every image the page shows, `sliceNeighbour` is
 * a linear scan over every image on the document, and a long chapter would turn
 * one paint into a quadratic sweep. The cap is the most slices a join may cover
 * anyway, so nothing reachable is lost. */
function sliceUsable(img) {
  const state = sliced.get(img);
  return Boolean(state && state.edges);
}

function sliceChain(upper, lower) {
  const chain = [upper, lower];
  const seen = new Set(chain);
  const grow = (below) => {
    for (let step = 0; step < SEAM_MAX_RUN_SLICES; step += 1) {
      const end = below ? chain[chain.length - 1] : chain[0];
      const next = sliceNeighbour(end, below);
      if (!next || seen.has(next) || !sliceUsable(next)) {
        // Is there a picture there at all? An untranslated one is worth waiting
        // for; the end of the document is not.
        return Boolean(sliceNeighbour(end, below, () => true));
      }
      seen.add(next);
      if (below) chain.push(next);
      else chain.unshift(next);
    }
    /* Stopped by the cap rather than by the strip. Reported as pending because
     * it is: there IS more strip, and `seamJoinFor`'s cap will bound the run
     * either way. */
    return true;
  };
  const above = grow(false);
  const below = grow(true);
  return { chain, pending: { above, below } };
}

async function runSeam(upper, lower) {
  const above = sliced.get(upper);
  const below = sliced.get(lower);
  /* Missing EDGES is reported and missing SLICE state is not, deliberately.
   * A neighbour with no entry in `sliced` is the ordinary case -- auto mode
   * translates in order, so the slice below has usually not been read yet.
   * A neighbour that IS in `sliced` with null edges is different: it means a
   * translated page came back without the geometry the join needs, which is a
   * defect and used to be indistinguishable from "no bubble crossed". */
  /* The ONE return on this path that is deliberately silent, and it is not a
   * verdict: an image at the head or foot of a strip has no translated
   * neighbour, so there is no boundary to report on. Every image would emit this
   * once, which is noise rather than evidence. Everything below here says why it
   * returned -- see the note further down. */
  if (!above || !below) return;
  if (!above.edges || !below.edges) {
    log("seam", "no-edges", above.edges ? "lower" : "upper", above.url);
    return;
  }

  /* Widen the pair into the RUN it belongs to, and only then decide.
   *
   * A skill name drawn down the side of a panel crosses slices that hold
   * neither its beginning nor its end, and no pairwise join can make it whole.
   * The run is anchored at its HEAD -- the last slice at or above
   * this one whose text is not cut by both of its own edges -- because a run
   * reached from its middle and a run reached from its end have to produce the
   * same join, or the same name is translated twice at two different lengths. */
  /* The BOUNDARY's own key, which is not the run's. A boundary that turns out to
   * cross nothing has no run to be keyed by, and it is the cheapest thing on
   * this path to skip -- so it is recorded and checked under the two URLs the
   * caller actually handed us, exactly as the pairwise seam always did. */
  /* Built by `seamBoundaryKey` rather than spelled here, so the retry walk and
   * this cannot drift: a retry that composed the string differently would look
   * up nothing, re-ask nothing, and report nothing. */
  const boundary = seamBoundaryKey(upper, lower);
  /* We are answering this boundary now, so it is no longer owed a re-ask. The
   * two branches that defer put it back, and only those two -- which is what
   * keeps the map from growing into a list of every boundary ever asked. */
  seamDeferred.delete(boundary);
  // TERMINAL: this boundary's retry budget is spent. Same argument as the run's
  // `abandoned` below -- "gave up after three failures" and "nothing to join"
  // were the same output, and they are opposite findings.
  if ((seamTried.get(boundary) || 0) >= SEAM_MAX_ATTEMPTS) {
    log("seam", "abandoned", seamTried.get(boundary), above.url);
    return;
  }

  const { chain, pending } = sliceChain(upper, lower);
  const at = chain.indexOf(upper);
  /* ANOMALOUS rather than routine: the caller handed us two images the chain does
   * not agree are adjacent. Logged because a structural disagreement that returns
   * in silence is indistinguishable from a clean boundary, and it should never
   * happen. */
  if (at < 0 || chain[at + 1] !== lower) {
    log("seam", "not-adjacent", above.url);
    return;
  }
  const edgesList = chain.map((img) => sliced.get(img).edges);

  /* The plan comes back WITH the run rather than being computed from it,
   * because choosing the run needs to know which candidates plan: a veto at one
   * cut vetoes its whole run, and the answer is the longest one that survives. */
  const found = seamJoinFor(edgesList, at, SEAM_MAX_RUN_SLICES, pending);
  /* NOT SETTLED, and this is the whole of the partial-run story. The name runs
   * off the end of what we hold and the slice that finishes it exists but is not
   * translated yet -- auto mode is usually one image behind. Recording this
   * would let the arrival order decide whether a four-slice name is ever joined. */
  /* TRANSIENT, and silent unless something re-asks it. The run extends past
   * what is translated, so the decision is deferred -- correctly, per the
   * comment above. But the boundary is neither SETTLED nor counted, so if the
   * slice that would finish the run never re-triggers this path, the join
   * simply never happens and nothing says so -- which is why `seamDeferred`
   * records it below. `pending` rides along because "waiting for one slice"
   * and "waiting for three" are the same message otherwise, and which one it
   * is decides whether a lookahead would fix it. */
  if (found.wait) {
    log("seam", "wait", pending, above.url);
    /* Recorded so `scheduleSeam` can come back to it. Without this the deferral
     * is the answer -- see `seamDeferred`'s own note, and the measurement that
     * found it. */
    seamDeferred.set(boundary, [upper, lower]);
    return;
  }
  if (!found.plan) {
    // No bubble crosses this boundary, which is the answer for almost every
    // boundary on almost every strip. SETTLED rather than counted: this one is
    // computed from geometry already in hand, cannot fail, and never reached the
    // network, so re-asking it could only ever produce the same answer.
    //
    // Logged so a run can tell "every boundary was clean" from "runSeam was
    // never reached at all". Those two produced identical output until now.
    log("seam", "no-plan", above.url);
    seamTried.set(boundary, SEAM_SETTLED);
    return;
  }
  const { head, length, plan } = found;

  const run = chain.slice(head, head + length);
  const states = run.map((img) => sliced.get(img));
  /* EVERY RETURN FROM HERE ON SAYS WHY. Measured on a warm strip where every
   * slice came from cache: the ONE boundary in the window that had a plan
   * emitted no line at all, while its six neighbours all logged `no-plan`. A
   * boundary that answers nothing is indistinguishable from one that had
   * nothing to answer -- the same trap the `no-plan` line above was added to
   * close.
   *
   * The verdicts below are deliberately split into TERMINAL and TRANSIENT,
   * because they want opposite fixes: a terminal one is the seam working, and a
   * transient one is a boundary that may never be asked again. */
  if (
    states.every((state, index) =>
      seamRunEdges(index, length).every((edge) => state.seamed.includes(edge))
    )
  ) {
    // TERMINAL, and the good outcome: this run's edges are already repaired.
    log("seam", "already-joined", `${length}x`, states[0].url);
    return;
  }
  /* TRANSIENT, and this is the one that lost the measured boundary. A downward
   * pass reaches the upper slice's boundary while the lower slice is still being
   * translated, so the run is skipped -- and nothing re-asks it, because the
   * boundary is neither SETTLED nor counted as a failure. It simply never
   * happens. Logging it does not fix that; it makes it visible, which is the
   * prerequisite. */
  if (run.some((img) => inFlight.has(img))) {
    log("seam", "in-flight", `${length}x`, states[0].url);
    /* The SAME shape as `wait` and the same repair, and this comment's own text
     * above already named it as a gap: "nothing re-asks it, because the boundary
     * is neither SETTLED nor counted as a failure". Recorded here so it is.
     *
     * Stated honestly: `wait` is the branch MEASURED firing on a real strip
     * (twice). This one is covered because it is the identical
     * mechanism reached one step later, not because it was observed. */
    seamDeferred.set(boundary, [upper, lower]);
    return;
  }

  /* The RUN's key, so that the same name reached from two of its own boundaries
   * is asked once. Equal to `boundary` for an ordinary two-slice join, which is
   * why that case behaves exactly as it did. */
  const key = states.map((state) => state.url).join("\n");
  // TRANSIENT: another turn owns this run right now and will report its own
  // outcome. Distinct from `in-flight`, which is about the IMAGES rather than
  // the join, and the two arrive for different reasons.
  if (seamRunning.has(key)) {
    log("seam", "running", `${length}x`, states[0].url);
    return;
  }
  // TERMINAL: the retry budget is spent. Never silent, because "abandoned after
  // three failures" and "nothing to join" were the same output until now, and
  // they are opposite findings.
  if ((seamTried.get(key) || 0) >= SEAM_MAX_ATTEMPTS) {
    log("seam", "abandoned", seamTried.get(key), `${length}x`, states[0].url);
    return;
  }

  seamRunning.add(key);
  seamQueue = seamQueue.then(() => performSeam(run, plan, states, key));
  return seamQueue;
}

async function performSeam(run, plan, states, key) {
  for (const img of run) markBusy(img, "seam");
  try {
    const sources = await Promise.all(run.map((img) => sliceSource(img)));
    /* TRANSIENT: a slice's bytes could not be re-fetched. Silent until now, and
     * it is the outcome most easily mistaken for success -- `performSeam` was
     * reached, the boundary was counted as attempted, and nothing said the join
     * never left the page. */
    if (sources.some((source) => !source)) {
      log("seam", "no-source", `${run.length}x`, states[0].url);
      return;
    }
    /* The per-slice cap is not a run cap, and generalising from two slices to
     * SEAM_MAX_RUN_SLICES multiplied the worst case by three with nothing to
     * bound it. These bytes cross `sendMessage` as a structured-cloned
     * Uint8Array (a plain number array was the most expensive marshal in the
     * extension); the cap stays, because it bounds the composite the SERVER is
     * asked to build, not just the copy. */
    const bytes = sources.reduce((sum, source) => sum + source.bytes.length, 0);
    if (bytes > 2 * SEAM_MAX_SOURCE_BYTES) {
      log("seam", "too-big", run.length, bytes, states[0].url);
      seamTried.set(key, SEAM_SETTLED);
      return;
    }

    const reply = await browser.runtime.sendMessage({ type: "seam", plan, slices: sources });
    if (!reply || !reply.ok) throw new Error((reply && reply.error) || "join failed");

    /* SAY WHAT HAPPENED. If a successful join and a declined one were both
     * silent, a test harness would see the same output for "it worked", "no
     * bubble crossed" and "the whole feature is dead" -- and the feature has
     * been dead before with nothing to show it. Logged
     * before the paint so a throw inside `applySeam` still leaves the reason on
     * record. The slice count rides along because a run of four and a run of two
     * are the same message otherwise, and which one fired is the finding. */
    log("seam", reply.skipped || "joined", `${run.length}x`, plan.width, plan.height, states[0].url);
    const painted = Array.isArray(reply.painted) ? reply.painted : [];
    for (let index = 0; index < run.length; index += 1) {
      const state = states[index];
      if (!painted[index]) continue;
      if (!applySeam(run[index], painted[index], state.url)) continue;
      for (const edge of seamRunEdges(index, run.length)) {
        if (!state.seamed.includes(edge)) state.seamed.push(edge);
      }
    }
    /* Reached only on an `ok` reply -- but an `ok` reply is not always the
     * boundary ANSWERED. A reply the server produced is final, including one
     * that painted nothing; the background's own pre-flight declines are not,
     * and three of its four reasons are transient by their own comments --
     * "glossary-skew" even promises "the boundary can try again on a later
     * view", which the unconditional settle here was breaking. A transient
     * decline counts against the attempt budget exactly like a throw, so
     * "uncached" during the race with a neighbour's cache write no longer
     * kills the boundary for the life of the page view. `seamSkipFinal` owns
     * the split, in seam.js, where node can test it. */
    if (seamSkipFinal(reply.skipped)) {
      seamTried.set(key, SEAM_SETTLED);
    } else {
      const attempts = seamTried.get(key) || 0;
      if (attempts < SEAM_MAX_ATTEMPTS) seamTried.set(key, attempts + 1);
    }
  } catch (err) {
    /* Logged, not raised at the reader. Every slice already carries a translation
     * and the page reads: what failed is an improvement to one bubble, not the
     * page, and a toast here would fire on the ordinary case where a seam holds
     * nothing translatable and the server answers 502. */
    console.warn("[koharu] seam:", String((err && err.message) || err));
    /* Counted, not settled. This used to live in the `finally` below, where it
     * could not tell a 507 from an answer. */
    const attempts = seamTried.get(key) || 0;
    if (attempts < SEAM_MAX_ATTEMPTS) seamTried.set(key, attempts + 1);
  } finally {
    seamRunning.delete(key);
    for (const img of run) markDone(img);
  }
}

/* Called after every page this script puts on screen, from both the live path
 * and the cache path, because either one can be the second half to arrive. */
function scheduleSeam(img) {
  /* An explicit "this site is paged manga" pick skips the planning outright.
   * Page-host keyed, and this script IS the page, so its own hostname is the
   * key. The auto-detected case (a host the latch decided was paged) falls
   * through to the background's gate, which resolves the full profile -- this
   * check is the cheap exact half, not the whole predicate. */
  if ((settings.profileByHost || {})[location.hostname] === "manga") return;
  const below = sliceNeighbour(img, true);
  if (below) runSeam(img, below);
  const above = sliceNeighbour(img, false);
  if (above) runSeam(above, img);
  retryDeferredNear(img);
}

/* Ask again at every boundary near `img` that DEFERRED, because this slice may
 * be the one that completes its run.
 *
 * The two calls above ask only the boundaries this image TOUCHES, and that is
 * the whole of the defect: a name spanning four slices defers at its middle
 * boundaries, and the slice that finishes it arrives, asks its own two
 * boundaries, and never returns to them. Measured -- see `seamDeferred`.
 *
 * WHY THE WINDOW IS SEAM_MAX_RUN_SLICES, and why that is a reason rather than a
 * round number. A run is at most that many slices long, so a boundary further
 * away than that cannot be in any run this slice could complete. Walking further
 * would ask boundaries whose answer this arrival cannot have changed.
 *
 * BOTH DIRECTIONS, because a deferral is not always downward. `sliceChain`
 * reports `pending.above` as well, so a run can be waiting on a slice that has
 * not been read ABOVE it -- a reader scrolling up, or a lazy loader filling in
 * behind. The measured case was `{above:false, below:true}`; the symmetric one
 * costs one more walk and no request.
 *
 * IT CANNOT LOOP. `runSeam` deletes the entry on entry and only the two
 * deferring branches put it back, so a boundary that still defers is re-asked
 * only on the NEXT arrival, never within this one. A boundary that stops
 * deferring either settles, joins, or starts counting failures against
 * `SEAM_MAX_ATTEMPTS` -- every one of which is terminal or bounded. And the walk
 * itself does no work when nothing deferred, which is almost every strip:
 * `seamDeferred` is empty and the loop exits on its first test. */
function retryDeferredNear(img) {
  if (!seamDeferred.size) return;
  const window = [img];
  for (const below of [false, true]) {
    let end = img;
    for (let step = 0; step < SEAM_MAX_RUN_SLICES; step += 1) {
      const next = sliceNeighbour(end, below);
      if (!next || window.includes(next)) break;
      if (below) window.push(next);
      else window.unshift(next);
      end = next;
    }
  }
  for (let i = 0; i + 1 < window.length; i += 1) {
    const pair = seamDeferred.get(seamBoundaryKey(window[i], window[i + 1]));
    if (pair) runSeam(pair[0], pair[1]);
  }
}

/* The boundary key, in ONE place. It was spelled inline in `runSeam` and the
 * retry above has to produce the identical string or it looks up nothing and
 * silently repairs nothing -- a failure with no symptom, which is the shape this
 * file keeps paying for. Returns null when either slice has no recorded url,
 * which is not a boundary anyone can key. */
function seamBoundaryKey(upper, lower) {
  const a = sliced.get(upper);
  const b = sliced.get(lower);
  if (!a || !b) return null;
  return `${a.url}\n${b.url}`;
}

async function translate(img, opts = {}) {
  if (inFlight.has(img)) return;

  /* A retry is the one caller that must NOT be read as a second click: it
   * arrives precisely when `originals` has the image, asking for a fresh
   * draw of it rather than the original back. */
  const retry = Boolean(opts.retry) && originals.has(img);

  // Second click on an already-translated image restores the original -- both
  // the src and the source set that was silenced to let our src win.
  if (!retry && originals.has(img)) {
    setSrc(img, originals.get(img), "original");
    originals.delete(img);
    handled.delete(img);
    return;
  }

  inFlight.add(img);
  // One bar serves every image on the page, so an image that never raised the
  // spinner must not lower it for one that did.
  let spinning = false;
  /* Set only when this image really did end up on screen with edge boxes, and
   * consumed in the `finally` below. The seam CANNOT be scheduled from where the
   * page is shown, and that is not a style choice: `runSeam` refuses a pair
   * either of whose images is still in flight, and at that point this one is --
   * `inFlight.delete` is in the finally. Scheduling from inside the try made
   * every single join bail on that guard, silently, because a refused pair is
   * indistinguishable from the ordinary "no bubble crosses this boundary". */
  let joinable = null;
  try {
    /* On a retry the <img> is showing OUR lettering under a data: url, so the
     * src the server gets is the ORIGINAL the first translation was filed
     * under -- the host's own bytes rule, and the only url whose cache entry
     * the re-roll should overwrite. */
    const src = retry ? originals.get(img) : img.currentSrc || img.src;
    /* What the reader is LOOKING at as this run leaves -- for the moved-on
     * guard below. On an ordinary translate it is `src`; on a retry it is the
     * previous draw's data: url, which `src` deliberately is not. */
    const shownAtStart = img.currentSrc || img.src;

    /* A page the user is rereading should not refetch its images, let alone
     * re-upload them. The background script answers this from IndexedDB and
     * touches neither the network nor the Koharu server. A retry skips the
     * ask entirely: the stored entry is the very thing it exists to replace. */
    const cached = retry
      ? null
      : await browser.runtime
          .sendMessage({ type: "cache-lookup", url: src })
          .catch(() => null);

    if (cached && cached.ok && cached.hit) {
      // The same three lines, in the same order, as a fresh translation below.
      // setSrc and not img.src: the observer has to see this as our own write.
      originals.set(img, src);
      handled.add(img);
      setSrc(img, cached.dataUrl);
      noteMisses(cached, img);
      noteSlice(img, src, cached);
      noteBoxes(img, cached);
      joinable = img;
      return;
    }

    // Only now is this going to take long enough to be worth a spinner. A
    // cached gallery would otherwise flash one on every image it did not use.
    setBusy(true);
    spinning = true;
    // And on the picture itself, which is the only indication auto mode gives:
    // the bar follows the pointer, and in auto mode the pointer is elsewhere.
    markBusy(img, "translate");

    /* A retry fetches NOTHING. The first live click proved why: on a real
     * host the original url is routinely dead by retry time (signed urls
     * expire, page-minted blob: urls are revoked), and the canvas is showing
     * OUR lettering -- so the only honest byte stream is the one the
     * background stored with the entry at first-translate time, which is
     * also the exact stream whose hash lands the overwrite on the same key. */
    let bytes = null;
    let mime = "";
    if (!retry) {
      const blob = await imageBytes(img, src);
      const buffer = await blob.arrayBuffer();
      /* A Uint8Array, structured-cloned across the messaging boundary. Not
       * `Array.from(new Uint8Array(buffer))`, which boxes every byte into its
       * own JS Number -- measured at 38-194ms of main-thread time and ~8x
       * transient heap per page; Firefox's runtime messaging structured-clones
       * typed arrays, and background.js's `new Uint8Array(bytes)` accepts
       * either shape. */
      bytes = new Uint8Array(buffer);
      mime = blob.type || "image/png";
    }

    const reply = await browser.runtime.sendMessage({
      type: "translate",
      /* Tells the background to skip its own cache tier, pull the entry's
       * stored SOURCE bytes, and re-seed the translator's sampler -- absent,
       * an identical request is answered byte-identically (fixed seed), and
       * a retry that changed nothing would be indistinguishable from a
       * working one. */
      retry,
      /* The box editor's changes, riding the retry transport.
       * Null on everything else, including the plain retry -- the background
       * re-sends the entry's own stored edits there, so a re-roll cannot
       * resurrect a deleted box. */
      edits: (retry && opts.edits) || null,
      bytes,
      mime,
      url: src,
      /* The server reports region boxes in the uploaded image's own pixel space
       * and never reports the size of that space, so the one side that knows it
       * has to say so. `uploadBlock` has already required both to be non-zero. */
      width: img.naturalWidth,
      height: img.naturalHeight,
      /* The page's shape, for the engine latch. Sent from here because the
       * background script has no DOM and cannot derive it -- the same reason
       * `width`/`height` are sent. Accumulated per host, never acted on per
       * page: flipping the engine mid-chapter would force a `Pipeline::reload`
       * each time. */
      shape: pageShape(),
    });

    if (!reply || !reply.ok) {
      const failure = new Error(reply?.error || "translation failed");
      // The background script distinguishes a VRAM refusal from an ordinary
      // server error, and the two deserve different reactions.
      failure.kind = reply?.kind;
      throw failure;
    }

    /* The picture may have moved on while the run was out -- a reader that
     * swaps one persistent <img> from page to page does exactly that, and a
     * warm page is still 2.8s, a cold one 13s. Writing now would paint the
     * previous page's translation over the current one, and `originals` would
     * then hold the NEW page's URL as the old translation's original, so
     * "Show original" would swap in an untranslated page 6 and look correct.
     * uploadBlock has already required `complete && naturalWidth`, so this can
     * only differ if the picture genuinely changed. */
    const now = img.currentSrc || img.src;
    if (!img.isConnected || now !== shownAtStart) {
      // Said rather than dropped silently: a viewport resize re-picking a
      // srcset candidate lands here too, and a run that vanished without a
      // word is indistinguishable from one that never started.
      flashError("The page changed while this image was translating.", img);
      return;
    }

    // The captured `src`, never a re-read: see above.
    originals.set(img, src);
    handled.add(img);
    setSrc(img, reply.dataUrl);
    noteMisses(reply, img);
    noteSlice(img, src, reply);
    noteBoxes(img, reply);
    joinable = img;
  } catch (err) {
    // Nothing here may throw a second time and bury the first failure.
    const message = String((err && err.message) || err || "translation failed");
    const starved = Boolean(err) && err.kind === "insufficient_memory";
    console.warn("[koharu]", err);

    /* A VRAM refusal is not this image's problem -- every following one hits
     * the same wall. Auto mode stops rather than hammering a server that has
     * already said no. */
    if (starved && settings.autoTranslate) {
      settings.autoTranslate = false;
      syncAuto();
      browser.storage.local.set({ autoTranslate: false });
    }

    /* Remember it, so auto mode does not come back to this same image on the
     * next mutation. A 507 has already switched auto mode off above; every
     * other failure -- a server that is down, a 401 before the token is
     * pasted, a 409 provider mismatch -- would otherwise be retried forever. */
    failed.add(img);

    // 120 characters, because that is where the server puts the meaning.
    flashError(message.slice(0, 120), img, starved ? 6000 : 2600);
  } finally {
    inFlight.delete(img);
    if (spinning) setBusy(false);
    // Unconditional and last: a badge stranded by a throw would spin forever on
    // a picture nothing is working on, which is worse than never showing one.
    markDone(img);
    // After the delete above, never before it. See `joinable`.
    if (joinable) scheduleSeam(joinable);
  }
}

/* The one path a person takes to translate a single image, shared by the hover
 * button and the context menu. The bar appears over anything the layout has made
 * page-sized, which deliberately includes images whose bytes are still a
 * placeholder -- so this is where that is caught and said out loud instead of
 * being uploaded. */
function requestTranslate(img) {
  /* A person asking is always worth another go: this is the one path that
   * retries an image auto mode has written off, and the only way back after the
   * server has been started or the token pasted. */
  failed.delete(img);

  // Restoring needs no bytes at all, and has to keep working on an image whose
  // src the page has since replaced.
  if (originals.has(img)) return translate(img);

  const blocked = uploadBlock(img);
  if (!blocked) return translate(img);

  watchLoad(img);
  flashError(blockText(img, blocked), img);
  return Promise.resolve();
}

/* The person's "that translation is wrong -- go again": a fresh DRAW of the
 * same page, not a restore. Distinct from `requestTranslate` on purpose: the
 * second click's toggle is load-bearing there, and a retry must reach the
 * server from exactly the state the toggle intercepts. On an image with no
 * translation yet there is nothing to re-roll, so it degrades to the ordinary
 * path -- which also covers a stale button. */
function retryTranslate(img) {
  failed.delete(img);
  if (!originals.has(img)) return requestTranslate(img);
  return translate(img, { retry: true });
}

function onRetryClick(event) {
  if (!event.isTrusted) return;
  event.preventDefault();
  event.stopPropagation();
  if (target) retryTranslate(target);
}

/* ------------------------------------------------------------- box editor
 *
 * The Add + Delete scope: draw a rectangle over text the detector
 * missed, or click a detected box to delete it, then Apply -- which rides the
 * retry transport with the merged edits. All arithmetic lives in boxedit.js;
 * this block is the DOM glue: one fixed-position overlay sized to the image,
 * box divs inside it, a drag handler for new rectangles, and a toolbar.
 */

/* img -> {boxes, edits} from the last reply that carried them. The editor's
 * whole substrate: without an entry here there is nothing to show and the
 * button stays hidden. */
const regionBoxes = new WeakMap();

function noteBoxes(img, reply) {
  if (Array.isArray(reply.boxes) && reply.boxes.length) {
    regionBoxes.set(img, { boxes: reply.boxes, edits: reply.edits || null });
  }
}

let boxEditor = null; // the overlay; non-null exactly while editing
let boxEditorImg = null;
let boxEditorDrawn = []; // source rects drawn this session (null = removed again)
let boxEditorDeleted = []; // source rects marked for deletion this session
let boxEditorResized = []; // {from, to} source-rect pairs, in the order dragged
let boxEditorPlaced = []; // flat wire placements set this session (null = removed)
let boxEditorPlaceDivs = []; // the div behind each boxEditorPlaced index
let boxEditorUnplaced = []; // target rects of STORED placements removed this session
let boxEditorDrag = null; // {startX, startY, el} in overlay coordinates
let boxEditorResize = null; // {edge, el, startRect, commit} while a handle drags
let boxEditorMove = null; // {el, startX, startY, startRect, commit, moved} -- an amber box mid-drag
let boxEditorMode = "box"; // "box" | "place" -- what a click and a drag mean
let boxEditorSelected = null; // {rect(), el} -- the box a placement aims at

const BOXEDIT_HANDLE_EDGES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

/* Eight drag handles on a box div. Each handle owns its mousedown
 * (stopPropagation, so the parent box's click never fires a delete off a
 * resize) and hands the commit callback the final VIEW rect; the callback
 * answers false to refuse, and the div snaps back. */
function attachBoxHandles(el, commit) {
  for (const edge of BOXEDIT_HANDLE_EDGES) {
    const handle = document.createElement("div");
    handle.className = `khr-box-handle khr-h-${edge}`;
    handle.addEventListener("mousedown", (event) => {
      if (!event.isTrusted || event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      boxEditorResize = {
        edge,
        el,
        startRect: {
          left: parseFloat(el.style.left) || 0,
          top: parseFloat(el.style.top) || 0,
          width: parseFloat(el.style.width) || 0,
          height: parseFloat(el.style.height) || 0,
        },
        commit,
      };
    });
    // The click that follows the mouseup must die here, or it bubbles into
    // the box's own listener and toggles a delete the reader never asked for.
    handle.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    el.append(handle);
  }
}

/* The one mapping every commit needs: the final view rect, in source pixels. */
function boxEditorSourceOf(viewRect) {
  return boxeditToSource(
    viewRect,
    boxEditorView(boxEditorImg),
    boxEditorImg.naturalWidth,
    boxEditorImg.naturalHeight
  );
}

/* A flat wire placement: target's four keys beside the place_* half. */
function boxEditorFlatPlacement(target, place) {
  return {
    x: target.x, y: target.y, width: target.width, height: target.height,
    place_x: place.x, place_y: place.y,
    place_width: place.width, place_height: place.height,
  };
}

/* In placement mode a click on a box SELECTS it as the target the next drawn
 * rectangle places. Clicking the selected box again deselects. */
function boxEditorSelect(el, rectOf) {
  if (boxEditorSelected && boxEditorSelected.el === el) {
    el.classList.remove("khr-box-sel");
    boxEditorSelected = null;
    return;
  }
  if (boxEditorSelected) boxEditorSelected.el.classList.remove("khr-box-sel");
  el.classList.add("khr-box-sel");
  boxEditorSelected = { el, rect: rectOf };
}

/* One placement rectangle on the overlay: amber, removable by click,
 * resizable by handles (the place_* half only -- the target stays put).
 * `stored` entries live in the entry's saved edits; removing one goes through
 * `unplaced`, and resizing one becomes a session `placed` replacement. An
 * orphaned stored placement (its target matches no displayed box after a
 * re-detection) renders muted rather than being silently dropped -- never
 * take the reader's rectangle off the page without their word. */
function boxEditorAddPlaceDiv(entry, { stored = false, orphan = false } = {}) {
  const img = boxEditorImg;
  const view = boxEditorView(img);
  const el = document.createElement("div");
  el.className = "khr-box khr-box-place" + (orphan ? " khr-box-place-orphan" : "");
  positionBoxDiv(
    el,
    boxeditToView(
      { x: entry.place_x, y: entry.place_y, width: entry.place_width, height: entry.place_height },
      view,
      img.naturalWidth,
      img.naturalHeight
    )
  );
  const target = boxeditPlaceTarget(entry);
  // The div's own target, so boxEditorDropAmberFor can find it later.
  el.dataset.khrTx = String(target.x);
  el.dataset.khrTy = String(target.y);
  el.dataset.khrTw = String(target.width);
  el.dataset.khrTh = String(target.height);
  let index = -1;
  if (!stored) {
    index = boxEditorPlaced.push(entry) - 1;
    boxEditorPlaceDivs[index] = el;
  }
  const commitPlace = (place) => {
    if (place.width < BOXEDIT_MIN_PLACEMENT || place.height < BOXEDIT_MIN_PLACEMENT) {
      flashError(
        `A placement box needs at least ${BOXEDIT_MIN_PLACEMENT}px each way.`,
        img
      );
      return false;
    }
    const next = boxEditorFlatPlacement(target, place);
    if (stored) {
      // A moved or resized stored placement becomes a session replacement.
      boxEditorAddPlaceDivEntryOnly(next);
    } else {
      boxEditorPlaced[index] = next;
    }
    return true;
  };
  el.addEventListener("click", (event) => {
    if (!event.isTrusted) return;
    event.preventDefault();
    event.stopPropagation();
    /* A drag that just moved this box fires a click on release; consuming
     * the marker here is what keeps a move from also deleting the box. */
    if (el.dataset.khrMoved) {
      delete el.dataset.khrMoved;
      return;
    }
    if (stored) boxEditorUnplaced.push(target);
    else boxEditorPlaced[index] = null;
    el.remove();
  });
  /* Drag the body to MOVE the English: size preserved, clamped to the page,
   * committed like a resize. The handles own their own mousedown, so
   * event.target distinguishes them. */
  el.addEventListener("mousedown", (event) => {
    if (!event.isTrusted || event.button !== 0) return;
    if (event.target !== el) return; // a handle owns its own drag
    event.preventDefault();
    event.stopPropagation();
    const bounds = boxEditor.getBoundingClientRect();
    boxEditorMove = {
      el,
      commit: commitPlace,
      startX: event.clientX - bounds.left,
      startY: event.clientY - bounds.top,
      startRect: {
        left: parseFloat(el.style.left) || 0,
        top: parseFloat(el.style.top) || 0,
        width: parseFloat(el.style.width) || 0,
        height: parseFloat(el.style.height) || 0,
      },
      moved: false,
    };
  });
  attachBoxHandles(el, (viewRect) => commitPlace(boxEditorSourceOf(viewRect)));
  boxEditor.append(el);
  return el;
}

/* Every amber div (session or stored) showing a placement for `target` goes,
 * along with any session entry -- the on-screen half of the merge's
 * replace-by-target rule. A STORED entry needs no unplace here: the session
 * replacement shadows it at Apply. */
function boxEditorDropAmberFor(target) {
  for (const el of boxEditor.querySelectorAll(".khr-box-place")) {
    const held = {
      x: +el.dataset.khrTx, y: +el.dataset.khrTy,
      width: +el.dataset.khrTw, height: +el.dataset.khrTh,
    };
    if (boxeditSameRect(held, target)) el.remove();
  }
  for (let i = 0; i < boxEditorPlaced.length; i++) {
    if (
      boxEditorPlaced[i] &&
      boxeditSameRect(boxeditPlaceTarget(boxEditorPlaced[i]), target)
    ) {
      boxEditorPlaced[i] = null;
    }
  }
}

/* The direct gesture: in placement mode, dragging the TRANSLATED box itself
 * moves its English -- an amber ghost tears off under the cursor and becomes
 * the region's placement where it lands. The ghost spawns only past the click
 * threshold, so a plain click still selects; mousedown-on-region +
 * mouseup-on-ghost dispatches no click, so a real drag never toggles
 * anything. */
function attachPlaceDrag(el, rectOf) {
  el.addEventListener("mousedown", (event) => {
    if (!event.isTrusted || event.button !== 0) return;
    if (boxEditorMode !== "place") return;
    if (event.target !== el) return; // a handle owns its own drag
    event.preventDefault();
    event.stopPropagation();
    const img = boxEditorImg;
    const bounds = boxEditor.getBoundingClientRect();
    const target = rectOf();
    const startRect = boxeditToView(
      target,
      boxEditorView(img),
      img.naturalWidth,
      img.naturalHeight
    );
    boxEditorMove = {
      el: null, // spawned at the click threshold, so a plain click still selects
      ghost: true,
      spawn: () => {
        const ghost = document.createElement("div");
        ghost.className = "khr-box khr-box-place khr-box-ghost";
        positionBoxDiv(ghost, startRect);
        boxEditor.append(ghost);
        return ghost;
      },
      commit: (place) => {
        if (place.width < BOXEDIT_MIN_PLACEMENT || place.height < BOXEDIT_MIN_PLACEMENT) {
          flashError(
            `A placement box needs at least ${BOXEDIT_MIN_PLACEMENT}px each way.`,
            img
          );
          return false;
        }
        boxEditorDropAmberFor(target);
        boxEditorAddPlaceDiv(boxEditorFlatPlacement(target, place));
        return true;
      },
      startX: event.clientX - bounds.left,
      startY: event.clientY - bounds.top,
      startRect,
      moved: false,
    };
  });
}

/* The state half of the above, for a stored placement that was just resized:
 * the visible div is already on screen, only the session entry is missing. */
function boxEditorAddPlaceDivEntryOnly(entry) {
  const target = boxeditPlaceTarget(entry);
  for (let i = 0; i < boxEditorPlaced.length; i++) {
    if (boxEditorPlaced[i] && boxeditSameRect(boxeditPlaceTarget(boxEditorPlaced[i]), target)) {
      boxEditorPlaced[i] = entry;
      return;
    }
  }
  boxEditorPlaced.push(entry);
  boxEditorPlaceDivs[boxEditorPlaced.length - 1] = null;
}

function boxEditorView(img) {
  const rect = img.getBoundingClientRect();
  return { left: 0, top: 0, width: rect.width, height: rect.height };
}

function openBoxEditor(img) {
  const state = regionBoxes.get(img);
  if (!originals.has(img) || !state) {
    flashError("Translate the image first, then edit its boxes.", img);
    return;
  }
  closeBoxEditor();
  boxEditorImg = img;
  boxEditorDrawn = [];
  boxEditorDeleted = [];
  boxEditorResized = [];
  boxEditorPlaced = [];
  boxEditorPlaceDivs = [];
  boxEditorUnplaced = [];
  boxEditorMode = "box";
  boxEditorSelected = null;

  boxEditor = document.createElement("div");
  boxEditor.className = "khr-boxedit";
  boxEditor.addEventListener("mousedown", onBoxEditorDown);
  boxEditor.addEventListener("mousemove", onBoxEditorMove);
  boxEditor.addEventListener("mouseup", onBoxEditorUp);

  const view = boxEditorView(img);
  for (const box of state.boxes) {
    const el = document.createElement("div");
    el.className = "khr-box";
    /* The box's CURRENT rect, mutated by every resize -- identity is this
     * object, not rect equality: sameRect's 2px tolerance cannot recognise a
     * box that is mid-resize over its own old position. */
    const current = { x: box.x, y: box.y, width: box.width, height: box.height };
    positionBoxDiv(el, boxeditToView(current, view, img.naturalWidth, img.naturalHeight));
    el.addEventListener("click", (event) => {
      if (!event.isTrusted) return;
      event.preventDefault();
      event.stopPropagation();
      if (boxEditorMode === "place") {
        boxEditorSelect(el, () => ({ ...current }));
        return;
      }
      const index = boxEditorDeleted.findIndex((rect) => boxeditSameRect(rect, current));
      if (index === -1) boxEditorDeleted.push({ ...current });
      else boxEditorDeleted.splice(index, 1);
      el.classList.toggle("khr-box-del", index === -1);
    });
    attachBoxHandles(el, (viewRect) => {
      const to = boxEditorSourceOf(viewRect);
      boxEditorResized.push({ from: { ...current }, to: { ...to } });
      Object.assign(current, to);
      return true;
    });
    attachPlaceDrag(el, () => ({ ...current }));
    boxEditor.append(el);
  }

  /* The entry's stored placements, back on screen -- without this the editor
   * looks empty while the server still applies them. Orphans (a target no
   * displayed box's centre falls inside, after a re-detection) render muted
   * rather than vanishing. */
  for (const entry of (state.edits && state.edits.place) || []) {
    const target = boxeditPlaceTarget(entry);
    const orphan = !state.boxes.some((box) => boxeditCenterInside(target, box));
    boxEditorAddPlaceDiv(entry, { stored: true, orphan });
  }

  const toolbar = document.createElement("div");
  toolbar.className = "khr-boxedit-bar";
  const hint = document.createElement("span");
  hint.className = "khr-boxedit-hint";
  hint.textContent =
    "drag: add · click: delete · handles: resize · placement ON: drag a box to move its English · ";
  const mode = document.createElement("button");
  mode.className = "khr-boxedit-btn khr-boxedit-mode";
  mode.type = "button";
  mode.textContent = "placement: off";
  mode.addEventListener("click", (event) => {
    if (!event.isTrusted) return;
    boxEditorMode = boxEditorMode === "box" ? "place" : "box";
    const placing = boxEditorMode === "place";
    mode.textContent = placing ? "placement: ON" : "placement: off";
    mode.classList.toggle("khr-boxedit-mode-on", placing);
    boxEditor.classList.toggle("khr-mode-place", placing);
    if (!placing && boxEditorSelected) {
      boxEditorSelected.el.classList.remove("khr-box-sel");
      boxEditorSelected = null;
    }
  });
  const apply = document.createElement("button");
  apply.className = "khr-boxedit-btn";
  apply.type = "button";
  apply.textContent = "Apply";
  apply.addEventListener("click", onBoxEditorApply);
  const cancel = document.createElement("button");
  cancel.className = "khr-boxedit-btn khr-boxedit-cancel";
  cancel.type = "button";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", (event) => {
    if (!event.isTrusted) return;
    closeBoxEditor();
  });
  toolbar.append(hint, mode, apply, cancel);
  boxEditor.append(toolbar);

  document.body.append(boxEditor);
  positionBoxEditor();
  window.addEventListener("scroll", positionBoxEditor, true);
  window.addEventListener("resize", positionBoxEditor);
  window.addEventListener("keydown", onBoxEditorKey, true);
}

function closeBoxEditor() {
  if (!boxEditor) return;
  window.removeEventListener("scroll", positionBoxEditor, true);
  window.removeEventListener("resize", positionBoxEditor);
  window.removeEventListener("keydown", onBoxEditorKey, true);
  boxEditor.remove();
  boxEditor = null;
  boxEditorImg = null;
  boxEditorDrag = null;
  boxEditorResize = null;
  boxEditorMove = null;
  boxEditorSelected = null;
  boxEditorMode = "box";
}

function positionBoxEditor() {
  if (!boxEditor || !boxEditorImg) return;
  if (!boxEditorImg.isConnected) {
    closeBoxEditor();
    return;
  }
  const rect = boxEditorImg.getBoundingClientRect();
  boxEditor.style.left = `${rect.left}px`;
  boxEditor.style.top = `${rect.top}px`;
  boxEditor.style.width = `${rect.width}px`;
  boxEditor.style.height = `${rect.height}px`;
}

function positionBoxDiv(el, view) {
  el.style.left = `${view.left}px`;
  el.style.top = `${view.top}px`;
  el.style.width = `${view.width}px`;
  el.style.height = `${view.height}px`;
}

function onBoxEditorKey(event) {
  if (event.key === "Escape") {
    event.preventDefault();
    event.stopPropagation();
    closeBoxEditor();
  }
}

/* Drawing. Coordinates are relative to the overlay, which is sized to the
 * image, so the view for the mapping is simply (0, 0, width, height). In
 * placement mode the same drag draws the SELECTED box's placement instead. */
function onBoxEditorDown(event) {
  if (!event.isTrusted || event.button !== 0) return;
  if (event.target !== boxEditor) return; // a box or the toolbar owns this click
  event.preventDefault();
  if (boxEditorMode === "place" && !boxEditorSelected) {
    flashError("Select a box first — the placement says where ITS English goes.", boxEditorImg);
    return;
  }
  const bounds = boxEditor.getBoundingClientRect();
  const el = document.createElement("div");
  el.className =
    boxEditorMode === "place" ? "khr-box khr-box-place" : "khr-box khr-box-new";
  boxEditor.append(el);
  boxEditorDrag = {
    startX: event.clientX - bounds.left,
    startY: event.clientY - bounds.top,
    el,
    placing: boxEditorMode === "place",
  };
}

function onBoxEditorMove(event) {
  /* Resize BEFORE draw: a handle drag must never fall through to the draw
   * path, whose 8px click threshold would silently discard a small nudge. */
  if (boxEditorResize) {
    const bounds = boxEditor.getBoundingClientRect();
    positionBoxDiv(
      boxEditorResize.el,
      boxeditResizeRect(
        boxEditorResize.startRect,
        boxEditorResize.edge,
        event.clientX - bounds.left,
        event.clientY - bounds.top,
        { width: bounds.width, height: bounds.height }
      )
    );
    return;
  }
  if (boxEditorMove) {
    const bounds = boxEditor.getBoundingClientRect();
    const dx = event.clientX - bounds.left - boxEditorMove.startX;
    const dy = event.clientY - bounds.top - boxEditorMove.startY;
    /* Under ~3px it is still a click; past it, it is a move and the click
     * that follows the release must not delete the box. A region drag's
     * ghost spawns HERE, at the threshold, so a plain click never leaves
     * an amber box behind and still selects as before. */
    if (Math.abs(dx) + Math.abs(dy) > 3) {
      boxEditorMove.moved = true;
      if (!boxEditorMove.el && boxEditorMove.spawn) {
        boxEditorMove.el = boxEditorMove.spawn();
      }
      if (boxEditorMove.el) boxEditorMove.el.dataset.khrMoved = "1";
    }
    if (boxEditorMove.el) {
      positionBoxDiv(
        boxEditorMove.el,
        boxeditMoveRect(boxEditorMove.startRect, dx, dy,
                        { width: bounds.width, height: bounds.height })
      );
    }
    return;
  }
  if (!boxEditorDrag) return;
  const bounds = boxEditor.getBoundingClientRect();
  positionBoxDiv(boxEditorDrag.el, dragRect(event, bounds));
}

function onBoxEditorUp(event) {
  if (boxEditorResize) {
    const resize = boxEditorResize;
    boxEditorResize = null;
    const bounds = boxEditor.getBoundingClientRect();
    const viewRect = boxeditResizeRect(
      resize.startRect,
      resize.edge,
      event.clientX - bounds.left,
      event.clientY - bounds.top,
      { width: bounds.width, height: bounds.height }
    );
    if (resize.commit(viewRect)) positionBoxDiv(resize.el, viewRect);
    else positionBoxDiv(resize.el, resize.startRect);
    return;
  }
  if (boxEditorMove) {
    const move = boxEditorMove;
    boxEditorMove = null;
    if (!move.moved || !move.el) return; // a plain press-release is the click's
    const bounds = boxEditor.getBoundingClientRect();
    const viewRect = boxeditMoveRect(
      move.startRect,
      event.clientX - bounds.left - move.startX,
      event.clientY - bounds.top - move.startY,
      { width: bounds.width, height: bounds.height }
    );
    const committed = move.commit(boxEditorSourceOf(viewRect));
    if (move.ghost) move.el.remove(); // the commit made the real amber div
    else positionBoxDiv(move.el, committed ? viewRect : move.startRect);
    return;
  }
  if (!boxEditorDrag) return;
  const bounds = boxEditor.getBoundingClientRect();
  const rect = dragRect(event, bounds);
  const drag = boxEditorDrag;
  boxEditorDrag = null;
  /* Below ~8 display px it was a click, not a box -- drop it silently. */
  if (rect.width < 8 || rect.height < 8) {
    drag.el.remove();
    return;
  }
  const source = boxEditorSourceOf(rect);

  if (drag.placing) {
    drag.el.remove(); // the real div comes from boxEditorAddPlaceDiv
    if (source.width < BOXEDIT_MIN_PLACEMENT || source.height < BOXEDIT_MIN_PLACEMENT) {
      flashError(
        `A placement box needs at least ${BOXEDIT_MIN_PLACEMENT}px each way in source pixels.`,
        boxEditorImg
      );
      return;
    }
    const target = boxEditorSelected.rect();
    /* One placement per target on screen: a redraw replaces, like the merge. */
    boxEditorDropAmberFor(target);
    boxEditorAddPlaceDiv(boxEditorFlatPlacement(target, source));
    return;
  }

  positionBoxDiv(drag.el, rect);
  const index = boxEditorDrawn.push(source) - 1;
  // A drawn box is removable by clicking it, like any other -- by INDEX, not
  // by rect match: a resize has moved the rect out from under sameRect.
  drag.el.addEventListener("click", (clickEvent) => {
    if (!clickEvent.isTrusted) return;
    clickEvent.preventDefault();
    clickEvent.stopPropagation();
    if (boxEditorMode === "place") {
      boxEditorSelect(drag.el, () => ({ ...boxEditorDrawn[index] }));
      return;
    }
    boxEditorDrawn[index] = null;
    drag.el.remove();
  });
  attachBoxHandles(drag.el, (viewRect) => {
    boxEditorDrawn[index] = boxEditorSourceOf(viewRect);
    return true;
  });
  attachPlaceDrag(drag.el, () => ({ ...boxEditorDrawn[index] }));
}

function dragRect(event, bounds) {
  const x = Math.max(0, Math.min(event.clientX - bounds.left, bounds.width));
  const y = Math.max(0, Math.min(event.clientY - bounds.top, bounds.height));
  return {
    left: Math.min(boxEditorDrag.startX, x),
    top: Math.min(boxEditorDrag.startY, y),
    width: Math.abs(x - boxEditorDrag.startX),
    height: Math.abs(y - boxEditorDrag.startY),
  };
}

function onBoxEditorApply(event) {
  if (!event.isTrusted) return;
  const img = boxEditorImg;
  const state = regionBoxes.get(img);
  if (!img || !state) {
    closeBoxEditor();
    return;
  }
  const drawn = boxEditorDrawn.filter(Boolean);
  const placed = boxEditorPlaced.filter(Boolean);
  if (
    !drawn.length &&
    !boxEditorDeleted.length &&
    !boxEditorResized.length &&
    !placed.length &&
    !boxEditorUnplaced.length
  ) {
    closeBoxEditor();
    return;
  }
  const merged = boxeditMerge(state.edits, {
    drawn,
    deleted: boxEditorDeleted,
    resized: boxEditorResized,
    placed,
    unplaced: boxEditorUnplaced,
  });
  if (merged.error) {
    flashError(merged.error, img);
    return;
  }
  closeBoxEditor();
  failed.delete(img);
  translate(img, { retry: true, edits: merged });
}

function onEditBoxesClick(event) {
  if (!event.isTrusted) return;
  event.preventDefault();
  event.stopPropagation();
  if (target) openBoxEditor(target);
}

/* These controls live in the page's own DOM, so the page can reach them: a
 * `mouseover` it dispatches picks the target, and `.click()` on the button would
 * then spend the user's GPU -- a cold model load and an unbounded run of
 * translations, all carrying our token because we are the ones sending them.
 * `isTrusted` is the one property a page cannot forge. */
function onTranslateClick(event) {
  if (!event.isTrusted) return;
  event.preventDefault();
  event.stopPropagation();
  if (target) requestTranslate(target);
}

function onAutoClick(event) {
  if (!event.isTrusted) return;
  event.preventDefault();
  event.stopPropagation();
  settings.autoTranslate = !settings.autoTranslate;
  syncAuto();
  browser.storage.local.set({ autoTranslate: settings.autoTranslate });
  if (settings.autoTranslate) {
    /* Switching auto mode on is the user saying "try again", and it is the way
     * back once the server is finally up: a whole page written off while it was
     * down would otherwise stay written off until a reload. A WeakSet cannot be
     * emptied, so replace it. */
    failed = new WeakSet();
    sweep();
  }
}

/* --------------------------------------------------------------- auto mode */

/* A page that has not arrived yet gets asked again rather than dropped. One
 * listener per image, not one per sweep: a gallery that sweeps repeatedly while
 * an image is still loading used to stack a closure per pass.
 *
 * `complete` is not the test for whether there is anything to wait for. A lazy
 * placeholder is complete -- it is a real 1x1 GIF that really did load -- so
 * waiting only on `!complete` waits on everything except the pages this matters
 * for. The site's own loader swapping the real URL in fires `load` again, and
 * that is the wake-up. */
function watchLoad(img) {
  if (watching.has(img)) return;
  if (img.complete && !uploadBlock(img)) return;
  watching.add(img);

  img.addEventListener(
    "load",
    () => {
      watching.delete(img);
      if (!enabledHere) return;
      // The pointer may have been sitting on it the whole time it was blank.
      scheduleResolve();

      if (!eligible(img) || handled.has(img) || inFlight.has(img)) return;
      if (failed.has(img)) return;
      // A two-stage loader swaps one placeholder for another. Wait again; a
      // `load` event only fires on a real load, so this cannot spin.
      if (uploadBlock(img)) {
        watchLoad(img);
        return;
      }
      /* Through the queue, not straight to `translate`. This fires whenever a
       * lazy image finishes loading, which on a manga reader is a burst of them
       * at once -- the exact fan-out the queue exists to stop. `enqueue`
       * re-checks the viewport when the image reaches the front, so an image
       * that loaded far below the fold waits its turn or is dropped. */
      if (settings.autoTranslate) enqueue(img);
    },
    { once: true }
  );

  img.addEventListener("error", () => watching.delete(img), { once: true });
}

/* Auto-mode tracing, for test harnesses that read the browser console.
 *
 * `console.debug` rather than `console.log`: Firefox hides Debug level unless
 * the console is asked for it, so this is silent in ordinary reading and a full
 * trace of the queue the moment somebody turns it on. That is the whole reason
 * it can be unconditional -- there is no setting to plumb, and no risk of a
 * release build being noisy, because the reader never opens the console.
 *
 * The queue is the one part of this file whose behaviour is a *sequence* rather
 * than a state, so a stack trace after the fact says nothing useful about it.
 * The order these lines appear in IS the property under test. */
function log(...parts) {
  console.debug("[birelate]", ...parts);
}

/* How far outside the viewport still counts as "the reader is looking at it".
 * Small on purpose: the lookahead below is what prefetches, not this. Its job is
 * only to stop an image flapping in and out of the set while the reader nudges
 * the scroll by a few pixels. */
const VIEWPORT_MARGIN_PX = 100;

/* Images past the furthest the reader has reached to translate anyway, so the
 * next pages are ready before they are looked at. */
const AUTO_LOOKAHEAD = 2;

/* The furthest image the reader has actually reached, as an index into document
 * order. Everything up to here plus the lookahead is translated, IN ORDER, and
 * nothing is dropped for having been scrolled past.
 *
 * **That is a context requirement, not a convenience.** The story window carries
 * source/target pairs from one page into the next page's prompt, which is what
 * holds names, honorifics and register steady across a chapter. It is built by
 * appending each page's pairs as that page finishes, so a page skipped because
 * the reader scrolled by it quickly is a hole in the window every later page
 * reads through. Translating pages 1, 2, 5, 6 gives page 6 a context that never
 * saw 3 and 4 -- which is exactly how a name drifts.
 *
 * So a fast scroll enqueues everything it passed rather than the pages it
 * happened to stop on. -1 means the reader has not reached anything yet. */
let autoReach = -1;

/* Auto mode runs ONE translation at a time, in reading order.
 *
 * It used to fire an un-awaited `translate(img)` per eligible image straight out
 * of the sweep, so a chapter with every image already in the DOM started all of
 * them at once. Two things were wrong with that. The reader waited on the whole
 * chapter to reach the page they were actually looking at, since the server's
 * one-permit gate serialises the work anyway and the queue order was whatever
 * the sweep happened to emit. And the story window -- the thing that keeps names
 * and register consistent across a chapter -- is snapshotted per request BEFORE
 * that gate, so N concurrent pages all build their prompt from the same stale
 * context and none of them sees the page before it. Serialising is what makes
 * the window mean anything.
 *
 * A promise chain rather than a worker pool, because the concurrency that is
 * wanted here is exactly one. */
let autoChain = Promise.resolve();
const queued = new WeakSet();

function inViewport(img, margin) {
  const rect = img.getBoundingClientRect();
  const height = window.innerHeight || document.documentElement.clientHeight;
  const width = window.innerWidth || document.documentElement.clientWidth;
  // Zero-area boxes are `display:none` or not laid out, and are nobody's view.
  if (rect.width <= 0 || rect.height <= 0) return false;
  return (
    rect.bottom > -margin &&
    rect.top < height + margin &&
    rect.right > -margin &&
    rect.left < width + margin
  );
}

/** Adds `img` to the back of the single auto queue, at most once. */
function enqueue(img) {
  if (queued.has(img)) return;
  queued.add(img);
  log("queue", img.currentSrc || img.src);
  autoChain = autoChain
    .then(async () => {
      queued.delete(img);
      /* Re-checked at the front of the queue rather than only at the back of
       * it. Between enqueuing and running, the reader may have scrolled far
       * past, the image may have been translated by hand, or the site may have
       * swapped it out. Every one of those makes the run pointless, and the
       * whole point of this queue is that a pointless run delays a real one. */
      if (!enabledHere || !settings.autoTranslate) return;
      if (handled.has(img) || inFlight.has(img) || failed.has(img)) return;
      if (!img.isConnected || !eligible(img) || uploadBlock(img)) return;
      /* Deliberately NOT re-checked against the viewport. An earlier version
       * dropped an image that had drifted more than a couple of screens away by
       * the time its turn came, which bounded the queue but tore holes in the
       * story window -- see `autoReach`. The queue is bounded by the reach plus
       * the lookahead instead, which bounds it by where the reader has been
       * rather than by where they are now. */
      log("translate", img.currentSrc || img.src);
      await translate(img);
    })
    // `translate` reports its own failures and records them in `failed`; this is
    // only here so one rejection cannot break the chain for every later image.
    .catch(() => {});
}

/* Auto mode has to cope with galleries that swap the src of one persistent
 * <img> rather than inserting a new node, so watch both. */
function sweep() {
  if (!enabledHere || !settings.autoTranslate) return;

  /* Document order throughout: `document.images` is in tree order, which for a
   * manga reader is reading order, and it is the order the story window has to
   * be fed in to be worth carrying. */
  const images = Array.from(allImages());

  for (let index = 0; index < images.length; index += 1) {
    const img = images[index];
    if (!eligible(img)) {
      // Not necessarily "too small" -- possibly just blank. Ask again on load.
      if (img.isConnected && !img.complete) watchLoad(img);
      continue;
    }
    // The reach only ever grows. Scrolling back up must not un-queue the pages
    // below, which are already in the window the pages after them will read.
    if (inViewport(img, VIEWPORT_MARGIN_PX) && index > autoReach) {
      autoReach = index;
      log("reach", index, img.currentSrc || img.src);
    }
  }

  /* Nothing reached is nothing to do. A background tab, a reader scrolled into
   * a comment section, a page whose images have not been laid out yet -- all of
   * them used to start the entire chapter. */
  if (autoReach < 0) return;

  const limit = Math.min(images.length, autoReach + 1 + AUTO_LOOKAHEAD);
  for (let index = 0; index < limit; index += 1) {
    const img = images[index];
    if (handled.has(img) || inFlight.has(img) || failed.has(img)) continue;
    if (!eligible(img)) continue;

    /* Eligible is only "worth a button". Uploading needs bytes that are
     * actually the picture: `complete` alone is true for an image that failed
     * to load and for a lazy placeholder that has not been replaced yet, and
     * both cost a GPU run and destroy the image they were standing in for. */
    if (uploadBlock(img)) watchLoad(img);
    else enqueue(img);
  }
}

/* Scrolling is what changes the answer above, and nothing else was asking.
 * rAF-throttled: a scroll fires far faster than layout is worth re-reading, and
 * `sweep` walks every image on the page.
 *
 * `passive` because this never calls `preventDefault`, and a non-passive scroll
 * listener on a reader is a jank complaint waiting to happen. */
let scrollPending = false;
function onViewportChanged() {
  if (scrollPending) return;
  scrollPending = true;
  requestAnimationFrame(() => {
    scrollPending = false;
    sweep();
  });
}
addEventListener("scroll", onViewportChanged, { passive: true });
addEventListener("resize", onViewportChanged, { passive: true });

const observer = new MutationObserver((records) => {
  if (!enabledHere) return;
  let touched = false;

  for (const record of records) {
    if (record.type === "attributes") {
      const el = record.target;
      /* Our own write is not a new picture. Treating it as one dropped the
       * history of the image we had just translated, and in auto mode the
       * sweep below then fed our output straight back to the server. Asked for
       * <source srcset> as well as <img src>, because silencing a source set is
       * now part of putting a translation on screen. */
      if (wasOurWrite(el, record.attributeName)) continue;

      if (el instanceof HTMLImageElement) {
        // A different picture, so drop its history -- and with it the source
        // set held for a restore that would now put the wrong image back.
        handled.delete(el);
        originals.delete(el);
        stashed.delete(el);
        // Its edge boxes belonged to the picture that has just left, and a
        // neighbour matching against them would join two different strips.
        sliced.delete(el);
        // A different picture deserves its own attempt, whatever happened to
        // the one that was here before.
        failed.delete(el);
      }
      /* A <picture>'s <source srcset> changing is a new picture too, even
       * though the record's target is the <source> and not the <img> that will
       * end up displaying it. That record used to be dropped on the floor. */
      touched = true;
    } else if (record.addedNodes.length) {
      /* Our own overlay sits in the observed subtree, and both halves of a
       * failure write to it: flashError replaces the tip's text node and the
       * finally clause's setBusy replaces the button's icon. Counted as page
       * churn, those close a loop with no exit -- sweep, translate, failure,
       * setBusy, sweep -- because the catch path leaves the image in neither
       * `handled` nor `inFlight`. Only a 507 escaped it, by switching auto mode
       * off; a 401 from an unpasted token did not. The attributes branch above
       * has carried `wasOurWrite` since the same mistake bit through img.src.
       * This is that filter's missing half. */
      if (isOurs(record.target)) continue;
      // On the bar's first mount the target is the body and every added node is
      // ours, so the whole record has to be examined, not just its target.
      for (const node of record.addedNodes) {
        if (!isOurs(node)) {
          touched = true;
          break;
        }
      }
    }
  }

  if (touched && settings.autoTranslate) queueMicrotask(sweep);
  if (touched && target && !target.isConnected) scheduleHide();
});

/* ------------------------------------------------------- finding the image */

/* Every node this script puts in the page, so the observer can tell our own
 * churn from the reader's. `marks` is here for the same reason `tip` is, and
 * leaving it out would be the auto-mode loop all over again: a badge appears and
 * disappears around every single translation, and an unfiltered childList record
 * for it counts as page churn and re-drives `sweep`. */
const isOurs = (el) =>
  Boolean(
    (bar && (el === bar || el === tip || bar.contains(el))) ||
      (marks && (el === marks || marks.contains(el))) ||
      (pin && (el === pin || pin.contains(el)))
  );

const within = (r, x, y) =>
  r.width > 0 && r.height > 0 && x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;

/* A reader built as a web component keeps its <img> inside a shadow root, where
 * the document-level hit test stops. One level deep is enough for the shapes
 * that exist, and the whole thing is guarded because `elementsFromPoint` on a
 * ShadowRoot is not universally present. */
function pierce(el, x, y) {
  const root = el && el.shadowRoot;
  if (!root || typeof root.elementsFromPoint !== "function") return null;
  try {
    for (const inner of root.elementsFromPoint(x, y)) {
      if (eligible(inner)) return inner;
    }
  } catch {
    /* a closed or detached root: nothing to see */
  }
  return null;
}

/* The last resort, and the only one that finds an image carrying
 * `pointer-events: none` -- such an element is skipped by hit testing entirely,
 * so it appears in no stack at all. Smallest containing box wins, because a
 * page-sized wrapper <img> should not beat the panel the pointer is actually
 * on.
 *
 * This is O(images) rect reads, so the answer is remembered rather than the
 * scan being skipped. Skipping would return "nothing here" for a pointer that
 * had simply stopped moving, and the bar would hide itself off an image it was
 * still sitting on. */
const SCAN_MS = 100;

function imageUnder(x, y) {
  const now = performance.now();
  if (lastScan.x === x && lastScan.y === y && now - lastScan.at < SCAN_MS) {
    return lastScan.hit && lastScan.hit.isConnected ? lastScan.hit : null;
  }

  let best = null;
  let bestArea = Infinity;

  for (const img of allImages()) {
    const rect = img.getBoundingClientRect();
    if (!within(rect, x, y)) continue;
    if (!eligible(img, rect)) {
      if (!img.complete) watchLoad(img);
      continue;
    }
    const area = rect.width * rect.height;
    if (area < bestArea) {
      best = img;
      bestArea = area;
    }
  }

  lastScan = { x, y, at: now, hit: best };
  return best;
}

/* Resolve by position, never by `event.target`. Manga hosts routinely lay a
 * transparent element over the picture to deter saving, which makes the target
 * of every mouseover that shield rather than the image. `elementsFromPoint`
 * returns the whole hit-test stack -- occluded boxes included, topmost first --
 * so the image is still in there. */
function imageAt(x, y) {
  let stack = [];
  try {
    stack = document.elementsFromPoint(x, y) || [];
  } catch {
    stack = [];
  }

  for (const el of stack) {
    if (isOurs(el)) continue;
    if (eligible(el)) return el;
    if (el instanceof HTMLImageElement && !el.complete) watchLoad(el);
    const inner = pierce(el, x, y);
    if (inner) return inner;
  }

  return imageUnder(x, y);
}

/* ------------------------------------------------------------------ wiring */

function scheduleResolve() {
  if (!attached || resolveFrame) return;
  resolveFrame = requestAnimationFrame(resolveHover);
}

function resolveHover() {
  resolveFrame = 0;
  if (!enabledHere || pointerX < 0) return;

  const x = pointerX;
  const y = pointerY;

  // The pointer is on our own control: it belongs to whatever raised it.
  if (bar && bar.isConnected && bar.classList.contains("khr-visible")) {
    if (within(bar.getBoundingClientRect(), x, y)) {
      clearTimeout(hideTimer);
      return;
    }
  }

  // Still inside the image the bar already belongs to. Nothing to recompute,
  // and no scan to pay for -- which is what keeps the geometric pass off the
  // hot path while the user reads.
  if (target && target.isConnected && within(target.getBoundingClientRect(), x, y)) {
    clearTimeout(hideTimer);
    return;
  }

  const img = imageAt(x, y);
  if (img) {
    clearTimeout(hideTimer);
    showBarFor(img);
    return;
  }

  if (target) scheduleHide();
}

/* `mousemove` as well as `mouseover`, because an image with
 * `pointer-events: none` never fires `mouseover` at all -- the events belong to
 * whatever is underneath it. Coalesced onto one animation frame. */
function onPointer(event) {
  if (!enabledHere) return;
  pointerX = event.clientX;
  pointerY = event.clientY;
  scheduleResolve();
}

function onOut(event) {
  if (!enabledHere) return;
  /* Only the pointer leaving the document entirely. Everything else is decided
   * by position now: `event.target` is the shield and not the picture, so the
   * old `event.target === target` test could never hold again. */
  if (!event.relatedTarget) {
    pointerX = -1;
    scheduleHide();
  }
}

function onScroll() {
  positionBar();
  positionMarks();
  // A scrolling reader moves a different page under a stationary pointer.
  scheduleResolve();
}

/* Both primitives are idempotent -- addEventListener dedupes on
 * (type, callback, capture) and observe() replaces an existing registration's
 * options rather than adding a second one -- so this may be called freely. */
function attach() {
  if (attached) return;
  attached = true;

  document.addEventListener("mousemove", onPointer, true);
  document.addEventListener("mouseover", onPointer, true);
  document.addEventListener("mouseout", onOut, true);
  // The pin gesture. Capturing for the same reason onPointer is: a
  // host's own click handling must not be able to swallow it.
  document.addEventListener("click", onPinClick, true);
  // Captured on the document, not the window: a reader that scrolls an inner
  // container never fires a scroll event the window can see.
  document.addEventListener("scroll", onScroll, { passive: true, capture: true });
  addEventListener("resize", onScroll, { passive: true });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src", "srcset"],
  });
}

function detach() {
  if (!attached) return;
  attached = false;

  document.removeEventListener("mousemove", onPointer, true);
  document.removeEventListener("mouseover", onPointer, true);
  document.removeEventListener("mouseout", onOut, true);
  document.removeEventListener("click", onPinClick, true);
  closePin();
  document.removeEventListener("scroll", onScroll, true);
  removeEventListener("resize", onScroll);
  observer.disconnect();

  if (resolveFrame) {
    cancelAnimationFrame(resolveFrame);
    resolveFrame = 0;
  }
  clearTimeout(hideTimer);
  if (markTicker) {
    clearTimeout(markTicker);
    markTicker = 0;
  }
  if (bar) bar.classList.remove("khr-visible");
  hideTip();
  target = null;
  pointerX = -1;
  // Drops the strong reference the memo holds on a page image.
  lastScan = { x: -1, y: -1, at: 0, hit: null };

  /* Deliberately keeps `handled`, `originals` and `inFlight`. Clearing
   * `originals` strands a translated image with no route back to its original;
   * clearing `handled` makes the next enable re-upload our own output for every
   * image on the page, which is the exact loop `selfWrites` exists to prevent. */
}

/* An about:blank or srcdoc frame has no host of its own -- it inherits its
 * parent's origin, and `location.hostname` there is the empty string. Reading an
 * ancestor's location succeeds only when the origin really is shared, which
 * makes the attempt itself the check. */
function computeHost() {
  if (location.hostname) return location.hostname;
  for (const frame of [window.parent, window.top]) {
    try {
      if (frame && frame !== window && frame.location.hostname) {
        return frame.location.hostname;
      }
    } catch {
      /* a cross-origin ancestor: not ours to read, and not ours to inherit */
    }
  }
  return "";
}

// Memoised: a document does not change origin, and this is read on every
// storage change and once per image in the diagnostic.
let cachedHost = null;
const frameHost = () =>
  cachedHost === null ? (cachedHost = computeHost()) : cachedHost;

// The tab's own host, as the background script sees it. Only ever filled in for
// a subframe, and only for the diagnostic to quote.
let tabHost = "";

/* A `file://` page has no hostname at all and the popup stores that empty string
 * in the list, so an empty host has to be allowed to match. It must not match
 * for an about:blank frame that failed to inherit one, though, or a single
 * enabled local file would switch this extension on inside every opaque frame on
 * the web. */
function hostListed(hosts) {
  const host = frameHost();
  if (host) return hosts.includes(host);
  return location.protocol === "file:" && hosts.includes("");
}

/* Whether the user switched this page on.
 *
 * A subframe's own hostname is not necessarily what they ticked: the popup keys
 * the list on the *tab's* host, and a reader whose pages come from a sibling
 * subdomain would otherwise never match. Only the background script can see a
 * frame's tab URL, so ask it -- but only when the answer can still be yes,
 * because with all_frames this runs in every advertising iframe on the page. */
async function isEnabledHere() {
  if (settings.enabledEverywhere) return true;

  const hosts = Array.isArray(settings.enabledHosts) ? settings.enabledHosts : [];
  if (!hosts.length) return false;
  if (hostListed(hosts)) return true;
  if (window.top === window) return false;

  const reply = await browser.runtime
    .sendMessage({ type: "page-enabled" })
    .catch(() => null);
  if (reply && typeof reply.host === "string") tabHost = reply.host;
  return Boolean(reply && reply.enabled);
}

/* The reported half of the same question, so the diagnostic and the detector
 * can never disagree about why a page is off. */
function enabledReason() {
  const host = frameHost() || "(no host — about:blank or srcdoc)";
  if (settings.enabledEverywhere) return "enabled on every page and every frame";

  const hosts = Array.isArray(settings.enabledHosts) ? settings.enabledHosts : [];
  if (!hosts.length) return "no host is enabled yet";
  if (hostListed(hosts)) return `${host} is enabled`;
  if (window.top === window) return `${host} is not in the enabled list`;
  if (enabledHere) return `part of ${tabHost || "the tab's site"}, which is enabled`;
  return tabHost
    ? `this frame is ${host}, a different site from the tab's ${tabHost}`
    : `${host} is not enabled, and neither is the site this frame belongs to`;
}

/* A later answer always wins. Two storage changes in flight would otherwise be
 * free to land in either order and leave the page in the older one's state. */
let enabledSeq = 0;

async function refreshEnabled() {
  const seq = ++enabledSeq;
  const next = await isEnabledHere();
  if (seq !== enabledSeq || next === enabledHere) return;

  enabledHere = next;
  if (next) attach();
  else detach();
}

/* Nodes left behind by an earlier instance of this script in this same document.
 *
 * Reloading a temporary add-on re-injects the content script without unloading
 * the old one's DOM, and `bar`/`tip` live in that dead instance's closure, so
 * nothing here can recognise them as ours. They stay on screen with
 * `khr-visible` still set, and clicking one starts a spinner that never ends:
 * its listeners still fire, but its `browser.runtime` connection was severed by
 * the reload, so the message goes nowhere.
 *
 * In development `web-ext run` reloads on every source save, and repeated
 * reloads leave a stack of corpses across the top of the page that looks
 * exactly like a detection bug. An extension update does the same thing to a
 * real user, once, silently.
 *
 * Safe because it runs before ours are built: at this point every match is by
 * definition somebody else's. */
function sweepOrphanedOverlays() {
  for (const node of document.querySelectorAll(".khr-bar, .khr-tip")) {
    if (node !== bar && node !== tip) node.remove();
  }
}

async function init() {
  sweepOrphanedOverlays();

  const stored = await browser.storage.local.get(WATCHED);
  settings = { ...DEFAULTS, ...stored };

  await refreshEnabled();
  if (enabledHere && settings.autoTranslate) sweep();
}

browser.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;

  let gated = false;
  let autoTurnedOn = false;
  let credentialsChanged = false;
  for (const [key, change] of Object.entries(changes)) {
    /* The token is the one key this script must react to WITHOUT reading, and
     * the two halves of that sentence are both load-bearing.
     *
     * It is deliberately absent from DEFAULTS so its value never enters a page
     * sandbox -- that exclusion is a security property and stays. But the
     * `hasOwnProperty` skip below therefore also swallowed the transition, and
     * pasting the token is the single most likely thing a reader does after
     * every image on the page has already failed. The documented first-run flow
     * walks straight into it: turn auto mode on, watch everything fail 401,
     * paste the token, and nothing retries because `failed` still holds every
     * image and only an autoTranslate off -> on transition cleared it.
     *
     * Noting that it changed is not reading it. Nothing here touches
     * `change.newValue`, and `settings` never gains the key. */
    if (key === "token" || key === "serverUrl") {
      credentialsChanged = true;
      continue;
    }
    // Anything else in the store is not this script's business, and copying it
    // in was how the server token ended up here.
    if (!Object.prototype.hasOwnProperty.call(DEFAULTS, key)) continue;
    /* A removed key carries no `newValue` at all. Assigning that blindly put
     * `undefined` into minSize, where every comparison against it is false and
     * detection dies silently and extension-wide. */
    const next = "newValue" in change ? change.newValue : DEFAULTS[key];
    /* Read BEFORE `settings` is updated, and keyed on autoTranslate itself:
     * only a real off -> on transition is the user asking to try again. Every
     * other watched key reaches the sweep below as well, and clearing `failed`
     * there would re-drive a doomed run over every written-off image on an
     * unrelated minSize or corner edit. */
    if (key === "autoTranslate" && next && !settings.autoTranslate) {
      autoTurnedOn = true;
    }
    settings[key] = next;
    if (key === "enabledEverywhere" || key === "enabledHosts") gated = true;
  }

  /* The popup's checkbox is auto mode's other switch, and turning it on is the
   * same "try again" the in-page button means. Without this the set outlived
   * the only remedy for it: a 507 switches auto mode off by itself, so getting
   * going again REQUIRES switching it back on, and doing that from the popup
   * left every image on the page permanently written off. onAutoClick assigns
   * settings.autoTranslate before it writes storage, so its own echo arrives
   * with the transition already false and does not clear the set twice. */
  if (autoTurnedOn) failed = new WeakSet();

  /* A new token or a new server address is the other "try again", and until now
   * it was the only remedy the set did not answer to. Both seam and image state
   * are cleared, because both were written off against a server that has just
   * been replaced: `seamTried` counts failures per boundary, and a whole strip
   * can burn its attempts against a 401 before the reader has pasted anything. */
  if (credentialsChanged) {
    failed = new WeakSet();
    seamTried.clear();
  }

  syncAuto();

  /* Without this, ticking "enable here" wrote the host list and nothing
   * recomputed anything, so the tab stayed dead until it was reloaded.
   * Unticking it was equally inert in the other direction. */
  if (gated) {
    refreshEnabled().then(() => {
      if (settings.autoTranslate) sweep();
    });
  } else if (settings.autoTranslate) {
    sweep();
  }
});

/* The context menu names exactly one image, and `info.srcUrl` is how the
 * background script says which. `currentSrc` is checked first because that is
 * what the browser actually painted, and therefore what it put in the menu. */
function imageBySrc(url) {
  if (!url) return null;
  for (const img of allImages()) {
    if (img.currentSrc === url || img.src === url) return img;
  }
  return null;
}

function translateOne(message) {
  /* Deliberately outside `enabledHere` and `autoTranslate`. Right-clicking one
   * picture is a complete instruction on its own: it should not need the host
   * switched on first, it should not switch auto mode on behind the user's
   * back, and it must never turn into a page-wide job. */
  const img = imageBySrc(message.srcUrl);
  if (!img) {
    /* With all_frames every frame hears an untargeted broadcast, and the ones
     * without the image would each raise this toast. Only complain when the
     * message was aimed at this frame, or when this frame is the whole page. */
    if (message.targeted || window.top === window) {
      flashError("Could not find that image on the page.", null);
    }
    return;
  }

  showBarFor(img);
  requestTranslate(img).finally(() => {
    // Not scheduleHide(): that hides the tooltip too, and an error toast the
    // user has not read yet lives there. Nothing to do if the pointer has since
    // moved on -- the hover handlers own the bar again.
    if (target !== img) return;
    if (bar) bar.classList.remove("khr-visible");
    target = null;
  });
}

/* The context menu's retry, `translateOne`'s shape exactly -- including the
 * deliberate absence of the enabled/auto gates: right-clicking one picture is
 * a complete instruction on its own. On a translated image `info.srcUrl` is
 * the shown data: url, which `imageBySrc` matches against `img.src`, so the
 * lookup finds the element and `retryTranslate` recovers the original from
 * `originals` -- and on an untranslated one it degrades to a plain translate. */
function retryOne(message) {
  const img = imageBySrc(message.srcUrl);
  if (!img) {
    if (message.targeted || window.top === window) {
      flashError("Could not find that image on the page.", null);
    }
    return;
  }

  showBarFor(img);
  retryTranslate(img).finally(() => {
    if (target !== img) return;
    if (bar) bar.classList.remove("khr-visible");
    target = null;
  });
}

/* ------------------------------------------------------------- diagnostics */

/* "No images were found" can be two failures in one: nothing is detected,
 * and there is no way to see why. This answers the second. Everything below only
 * reads, and it reports the same predicates the detector uses -- a count that
 * came from a second implementation would be a count that can lie. */

const SAMPLE = 6;
// getComputedStyle is not cheap and some pages have tens of thousands of nodes.
const BG_SCAN_LIMIT = 3000;

const VERDICT = {
  "": "ready to translate",
  notimg: "not an <img>",
  detached: "not in the document",
  skip: "opted out (data-khr-skip)",
  small: "too small",
  pending: "not loaded yet",
};

/* An image can pass every filter the button uses and still have nothing worth
 * uploading behind it. Naming that separately is the difference between "we see
 * your pages" and "we see the placeholders standing in for your pages". */
const BLOCKED_VERDICT = {
  loading: "button shows, but the bytes have not arrived",
  placeholder: "button shows, but the loaded image is under the minimum",
};

const clip = (value, max) => {
  const text = String(value == null ? "" : value);
  return text.length > max ? `${text.slice(0, max)}…` : text;
};

/* The lazy-loading attributes reader themes commonly use, in the order one
 * widely used theme checks them. Reported and never followed: fetching an
 * attribute the browser has not painted would translate a picture the user is
 * not looking at, and setting src behind the site's own lazy loader is a fight
 * we would lose. Seeing the name here is what tells the user their reader is
 * lazy-loading at all. */
const LAZY_ATTRS = [
  "data-src",
  "data-lazy-src",
  "data-original",
  "data-cfsrc",
  "data-manga-src",
];

function lazyLabel(img) {
  for (const name of LAZY_ATTRS) {
    const value = img.getAttribute(name);
    if (value) return `${name}=${clip(value, 60)}`;
  }
  return "";
}

function describe(el) {
  const id = el.id ? `#${el.id}` : "";
  const cls =
    typeof el.className === "string" && el.className.trim()
      ? `.${el.className.trim().split(/\s+/)[0]}`
      : "";
  return `${el.tagName.toLowerCase()}${id}${cls}`;
}

/* "Something is painted on top of this image" is the single most useful thing
 * the old build could not say, because it is exactly the shape that defeated
 * `event.target`. Hit testing only works inside the viewport, and on a long
 * reader most pages are scrolled past it -- so say which of the two answers
 * this is rather than reporting an unchecked image as clean. */
function overlayLabel(img, rect) {
  const x = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;
  if (x < 0 || y < 0 || x > innerWidth || y > innerHeight) {
    return "scrolled out of view — overlay not checked";
  }

  let stack = [];
  try {
    stack = document.elementsFromPoint(x, y) || [];
  } catch {
    return "";
  }

  for (const el of stack) {
    if (isOurs(el)) continue;
    if (el === img) return "";
    return `covered by ${clip(describe(el), 40)}`;
  }
  return "";
}

function countBackgrounds(min) {
  let found = 0;
  let scanned = 0;
  for (const el of document.getElementsByTagName("*")) {
    if (++scanned > BG_SCAN_LIMIT) break;
    // Cheap box test first, so getComputedStyle only runs on panel-sized nodes.
    if (el.clientWidth < min || el.clientHeight < min) continue;
    const bg = getComputedStyle(el).backgroundImage;
    if (bg && bg !== "none" && bg.includes("url(")) found += 1;
  }
  return found;
}

function collect() {
  const min = minSize();
  const counts = {
    total: 0,
    eligible: 0,
    ready: 0,
    waiting: 0,
    responsive: 0,
    small: 0,
    pending: 0,
    detached: 0,
    skipped: 0,
  };
  const rows = [];

  for (const img of allImages()) {
    counts.total += 1;
    const rect = img.getBoundingClientRect();
    const { w, h } = measure(img, rect);
    const why = rejection(img, rect);
    // Only asked of an image that got that far, and it is the question the
    // upload gate asks: eligible says "a button appears", not "there are bytes".
    const blocked = why === "" ? uploadBlock(img) : "";

    if (why === "") {
      counts.eligible += 1;
      if (blocked) counts.waiting += 1;
      else counts.ready += 1;
      if (img.srcset || pictureSources(img).length) counts.responsive += 1;
    } else if (why === "small") counts.small += 1;
    else if (why === "pending") counts.pending += 1;
    else if (why === "detached") counts.detached += 1;
    else counts.skipped += 1;

    rows.push({ img, rect, w, h, why, blocked, area: w * h });
  }

  rows.sort((a, b) => b.area - a.area);

  const samples = rows.slice(0, SAMPLE).map((row) => ({
    size: `${row.w}×${row.h}`,
    natural: `${row.img.naturalWidth || 0}×${row.img.naturalHeight || 0}`,
    box: `${Math.round(row.rect.width)}×${Math.round(row.rect.height)}`,
    verdict: row.blocked
      ? BLOCKED_VERDICT[row.blocked]
      : VERDICT[row.why] || row.why,
    src: clip(row.img.currentSrc || row.img.getAttribute("src") || "", 90),
    lazy: lazyLabel(row.img),
    // A source set outranks `src`, so it has to be silenced for a translation
    // to appear at all. Worth seeing when a translation seems to do nothing.
    responsive: row.img.srcset
      ? "srcset"
      : pictureSources(row.img).length
        ? "<picture>"
        : "",
    overlay: row.why === "" ? overlayLabel(row.img, row.rect) : "",
  }));

  let canvases = 0;
  for (const canvas of document.getElementsByTagName("canvas")) {
    if (canvas.width >= min && canvas.height >= min) canvases += 1;
  }

  return {
    url: clip(location.href, 100),
    host: frameHost(),
    tabHost,
    top: window.top === window,
    readyState: document.readyState,
    enabled: enabledHere,
    reason: enabledReason(),
    attached,
    barBuilt: Boolean(bar),
    barConnected: Boolean(bar && bar.isConnected),
    autoTranslate: Boolean(settings.autoTranslate),
    minSize: min,
    counts,
    samples,
    frames:
      document.getElementsByTagName("iframe").length +
      document.getElementsByTagName("frame").length,
    canvases,
    backgrounds: countBackgrounds(min),
  };
}

/* Answered by broadcast rather than as a reply. `tabs.sendMessage` reaches
 * every frame but surfaces only one response, and "how many frames did we
 * reach" is half the answer the user needs. The popup collects these. */
function reportDiagnostics(token) {
  let report;
  try {
    report = collect();
  } catch (err) {
    report = { url: clip(location.href, 100), broken: String((err && err.message) || err) };
  }
  browser.runtime
    .sendMessage({ type: "diagnostic-report", token, report })
    .catch(() => {});
}

browser.runtime.onMessage.addListener((message) => {
  if (!message || typeof message.type !== "string") return;

  // None of these replies. `translate-one` and `retry-one` must not make the
  // caller wait on a GPU run, and `diagnose` answers on its own channel for
  // the reason above.
  if (message.type === "translate-one") translateOne(message);
  else if (message.type === "retry-one") retryOne(message);
  else if (message.type === "diagnose") reportDiagnostics(message.token);
});

init();
