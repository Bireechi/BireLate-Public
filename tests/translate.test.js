/* Executable pinning for background.js's translate() -- the wire heart.
 *
 * Run:  node --test tests/translate.test.js
 *
 * (Not `node --test tests/` -- see the header of seam.test.js for why a bare
 * directory argument reports a green suite failing for an unrelated reason.)
 *
 * WHY THIS FILE EXISTS. Before this file, translate() had ZERO executable
 * coverage: drop-report.test.js says outright it "cannot be executed here
 * at all", and its two static regexes plus five whole-file line-scans grazed
 * the text without running a single one of its ~20 decision points. A
 * mutation check measured the cost precisely: deleting the `!` in
 * `if (!keepArtIgnored)` -- filing an erased-artwork page under a
 * keep-the-art cache key for 24 hours -- left all 166 tests green. This file
 * exists so that exact mutation, and its neighbours, go red.
 *
 * HOW IT RUNS THE UNRUNNABLE. translate() reaches storage, fetch and
 * IndexedDB, so it cannot be required. It is lifted out of the SHIPPED source
 * text (the same trick report() uses one function up) and evaluated with
 * every collaborator injected as a named parameter -- each a recorder, so
 * tests assert on both the reply shape and WHO was called. The lift is the
 * function under test verbatim; only its environment is synthetic.
 *
 * ANCHORS THIS FILE BINDS (a refactor that moves either must update this
 * file in the same commit and re-prove a red):
 *   START  "async function translate("
 *   END    "/* -------------------------------------------------------------
 *           the seam" (the section banner that follows translate())
 * Both asserted below, so a miss fails loudly rather than lifting garbage.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const BACKGROUND_JS = fs.readFileSync(
  path.join(ROOT, "extension", "background.js"),
  "utf8"
);

const START_ANCHOR = "async function translate(";
const END_ANCHOR = "/* ------------------------------------------------------------- the seam";

function shippedTranslate(stubs) {
  const start = BACKGROUND_JS.indexOf(START_ANCHOR);
  assert.ok(start > 0, "could not find translate() in background.js");
  const end = BACKGROUND_JS.indexOf(END_ANCHOR);
  assert.ok(end > start, "could not find the seam banner that ends translate()");
  const source = BACKGROUND_JS.slice(start, end);
  const names = Object.keys(stubs);
  const factory = new Function(...names, `${source}; return translate;`);
  return factory(...names.map((name) => stubs[name]));
}

/* One harness per test: every collaborator is a recorder pushing
 * [name, args] into `calls`, with a default implementation that keeps the
 * happy path alive. Override per test. */
/* The shipped translatePins, lifted pure (the missNotes trick) and injected
 * as its own stub -- so the wiring test below exercises the real filter, not
 * a copy that could drift. */
function shippedTranslatePins() {
  const start = BACKGROUND_JS.indexOf("function translatePins(regions)");
  assert.ok(start > 0, "could not find translatePins() in background.js");
  const end = BACKGROUND_JS.indexOf("\nfunction ", start + 10);
  assert.ok(end > start, "could not find the function that follows translatePins()");
  const source = BACKGROUND_JS.slice(start, end);
  return new Function(`${source}; return translatePins;`)();
}

function harness(overrides = {}) {
  const calls = [];
  const track = (name, impl) => (...args) => {
    calls.push([name, args]);
    return impl ? impl(...args) : undefined;
  };
  const called = (name) => calls.filter(([who]) => who === name);

  const cfg = Object.assign(
    { keepArt: false, touchJoin: false, imageHostForPage: {} },
    overrides.cfg || {}
  );
  const payload = Object.prototype.hasOwnProperty.call(overrides, "payload")
    ? overrides.payload
    : { image: "data:image/png;base64,AA" };
  const reported = Object.assign(
    {
      regions: [{ x: 1 }],
      edgeHints: [],
      missed: null,
      dropped: null,
      cut: false,
      duplicateIds: 0,
      outOfRangeIds: 0,
      artKept: true,
    },
    overrides.reported || {}
  );

  const stubs = {
    hostOf: track("hostOf", (u) => {
      try {
        return new URL(u).host;
      } catch {
        return "";
      }
    }),
    config: track("config", async () => cfg),
    settingsFingerprint: track("settingsFingerprint", () => "fp"),
    sha256Hex: track("sha256Hex", async () => "hash"),
    cacheKey: track("cacheKey", (h, f) => `${h}:${f}`),
    cacheGetByKey: track("cacheGetByKey", async () => overrides.entry || null),
    /* The retry's byte source: the tier-1 entry whose `source` blob holds the
     * exact bytes the first translation hashed. Default: an entry WITH a
     * source, so retry tests exercise the happy path; override with null for
     * the pre-feature-entry refusal. */
    cacheGetByUrl: track("cacheGetByUrl", async () =>
      Object.prototype.hasOwnProperty.call(overrides, "priorEntry")
        ? overrides.priorEntry
        : {
            source: {
              size: 3,
              type: "image/jpeg",
              arrayBuffer: async () => new Uint8Array([9, 9, 9]).buffer,
            },
          }),
    cacheDataUrl: track("cacheDataUrl", async () => "data:image/png;base64,HIT"),
    cacheWarn: (tag) => () => undefined,
    cacheTouch: track("cacheTouch", async () => {}),
    cacheBump: track("cacheBump", async () => {}),
    cachePut: track("cachePut", async () => {}),
    cacheEnforceCaps: track("cacheEnforceCaps", async () => {}),
    cacheLimits: track("cacheLimits", () => ({})),
    fetchStatus: track("fetchStatus", async () => ({
      status: { sufficient: true, models_loaded: true },
    })),
    shortfall: track("shortfall", () => "3 GiB short"),
    warmup: track("warmup", async () => {}),
    storyId: track("storyId", async () => "story-1"),
    installGlossary: track("installGlossary", async () => {}),
    glossaryDrifted: track("glossaryDrifted", async () => false),
    base: track("base", () => "http://127.0.0.1:8765"),
    translateForm: track("translateForm", () => ({})),
    authHeaders: track("authHeaders", () => ({})),
    fetch: track("fetch", async () => ({
      ok: true,
      status: 200,
      json: async () => payload,
    })),
    detail: track("detail", async () => "the detail"),
    pngFromDataUrl: track("pngFromDataUrl", () => ({ size: 3 })),
    report: track("report", () => reported),
    noteScript: track("noteScript", async () => {}),
    /* The term ledger's two riders on translate(): the ledger harvest beside
     * noteScript, and the reply's pinnable-terms builder. The harvest is a
     * recorder like noteScript; translateTerms defaults to "nothing
     * term-shaped", which is most pages. */
    glossaryLedgerHarvest: track("glossaryLedgerHarvest", async () => {}),
    translateTerms: track("translateTerms", () => null),
    translatePins: track("translatePins", shippedTranslatePins()),
    browser: {
      storage: { local: { set: track("storage.set", async () => {}) } },
    },
    edgeWhiteProfile: track("edgeWhiteProfile", async () => ({ white: true })),
    seamEdges: track("seamEdges", () => ({ top: [], bottom: [] })),
    noteRunFinished: track("noteRunFinished", () => {}),
    /* Shadows the real WebCrypto inside the lift, so the retry's seed is a
     * constant the test can assert exactly instead of a range check. */
    crypto: {
      getRandomValues: track("getRandomValues", (array) => {
        array[0] = 0x12345678;
        return array;
      }),
    },
    ...(overrides.stubs || {}),
  };

  return { translate: shippedTranslate(stubs), calls, called, cfg };
}

const REQUEST = {
  bytes: [1, 2, 3],
  mime: "image/png",
  url: "https://img.example/p/001.png",
  width: 800,
  height: 1200,
  shape: "paged",
};
const PAGE_URL = "https://reader.example/series/9/ch1";

/* Tier-2 hit: the cached reply and the live reply must be ONE shape, with
 * every absent stored field defaulted -- `missed: null` not undefined,
 * `seamed: []` when the stored value is not an array, `cut` boolean-ised.
 * The server is never contacted. */
test("a tier-2 hit answers in the live reply's shape and never fetches", async () => {
  const h = harness({
    entry: {
      blob: {},
      dropped: "2/7",
      cut: 1,
      outOfRangeIds: 2,
      seamed: "not-an-array",
    },
  });
  const reply = await h.translate(REQUEST, PAGE_URL);

  assert.deepEqual(reply, {
    ok: true,
    dataUrl: "data:image/png;base64,HIT",
    cached: true,
    missed: null,
    dropped: "2/7",
    cut: true,
    duplicateIds: 0,
    outOfRangeIds: 2,
    /* The cut-repair pair, read back off the entry with the id counters' own
     * absent-falls-to-zero rule -- this entry predates the fields. */
    cutFound: 0,
    stillCut: 0,
    /* The placement overflow, same rule again -- an entry from before the field reads as
     * "all inside", never as unreported. */
    placementOverflow: [],
    edges: null,
    seamed: [],
    /* An entry from before the term ledger has no terms field and must read as nothing to
     * pin, exactly as `missed` and `dropped` read for old entries. */
    terms: null,
    // Same absent-field rule for the box editor's pair.
    boxes: null,
    edits: null,
  });
  assert.equal(h.called("fetch").length, 0, "a hit must not reach the server");
  assert.equal(h.called("cacheTouch").length, 1, "the hit renews its own entry");
  assert.deepEqual(h.called("cacheBump")[0][1], ["hits"]);
});

/* THE CACHE-POISON GATE. keepArt asked for, art not kept: the page is shown
 * but never filed, or every later view tier-1-hits an erased image behind a
 * keep-the-art key for 24 hours. Proven able to fail: deleting the `!` in
 * `if (!keepArtIgnored)` turns the first assertion red. */
test("a keep-art page whose art was erased is shown but never cached", async () => {
  const h = harness({ cfg: { keepArt: true }, reported: { artKept: false } });
  const reply = await h.translate(REQUEST, PAGE_URL);

  assert.equal(h.called("cachePut").length, 0, "an erased page must not be filed");
  assert.equal(h.called("cacheEnforceCaps").length, 0);
  assert.equal(reply.keepArtIgnored, true, "the reader is told the art went");
  assert.deepEqual(h.called("cacheBump")[0][1], ["misses"]);
});

test("a keep-art page whose art survived is cached under its key", async () => {
  const h = harness({ cfg: { keepArt: true }, reported: { artKept: true } });
  const reply = await h.translate(REQUEST, PAGE_URL);

  assert.equal(h.called("cachePut").length, 1);
  const put = h.called("cachePut")[0][1][0];
  assert.equal(put.srcHash, "hash");
  assert.equal(put.fingerprint, "fp");
  assert.equal(reply.keepArtIgnored, false);
});

/* The three refusal shapes are DIFFERENT errors: a starved pre-flight and a
 * server 507 both carry kind === "insufficient_memory" (content.js disables
 * auto mode on that kind); an ordinary failure carries no kind at all. */
test("a starved pre-flight refuses before the upload", async () => {
  const h = harness({
    stubs: {
      fetchStatus: async () => ({ status: { sufficient: false } }),
    },
  });
  await assert.rejects(
    () => h.translate(REQUEST, PAGE_URL),
    (err) => err.kind === "insufficient_memory"
  );
  assert.equal(h.called("fetch").length, 0, "no upload after a refusal");
});

test("a server 507 and an ordinary failure are different kinds", async () => {
  const starved = harness({
    stubs: { fetch: async () => ({ ok: false, status: 507 }) },
  });
  await assert.rejects(
    () => starved.translate(REQUEST, PAGE_URL),
    (err) => err.kind === "insufficient_memory" && err.message === "the detail"
  );

  const ordinary = harness({
    stubs: { fetch: async () => ({ ok: false, status: 500 }) },
  });
  await assert.rejects(
    () => ordinary.translate(REQUEST, PAGE_URL),
    (err) => err.kind === undefined && /^server 500: /.test(err.message)
  );
});

test("a reply without an image is an error, not a blank page", async () => {
  const h = harness({ payload: { image: undefined } });
  await assert.rejects(() => h.translate(REQUEST, PAGE_URL), /without an image/);
});

/* The seam summary's two gates are independent: the edge summary needs only
 * real dimensions; the white-continuation probe additionally needs the
 * reader's touch-join switch. Both arms reach the cache entry. */
test("zero dimensions mean no edge summary anywhere", async () => {
  const h = harness();
  const reply = await h.translate({ ...REQUEST, width: 0, height: 0 }, PAGE_URL);

  assert.equal(h.called("seamEdges").length, 0);
  assert.equal(reply.edges, null);
  assert.equal(h.called("cachePut")[0][1][0].edges, null);
});

test("the white probe runs only for touch-join, and its verdict rides into seamEdges", async () => {
  const off = harness();
  await off.translate(REQUEST, PAGE_URL);
  assert.equal(off.called("edgeWhiteProfile").length, 0);
  assert.equal(off.called("seamEdges")[0][1][4], null, "white is null without the switch");

  const on = harness({ cfg: { touchJoin: true } });
  await on.translate(REQUEST, PAGE_URL);
  assert.equal(on.called("edgeWhiteProfile").length, 1);
  assert.deepEqual(
    on.called("seamEdges")[0][1][4],
    { white: true },
    "the probe's verdict is the fifth argument"
  );
});

/* The popup can only see the PAGE host; the latch is keyed on the IMAGE host.
 * The pointer between them is written once, on change, and not awaited. */
test("the image-host pointer is written for the popup", async () => {
  const h = harness();
  await h.translate(REQUEST, PAGE_URL);
  const writes = h.called("storage.set");
  assert.equal(writes.length, 1);
  assert.deepEqual(writes[0][1][0], {
    imageHostForPage: { "reader.example": "img.example" },
  });
});

/* ------------------------------------------------------------- the glossary
 *
 * The install is LOAD-BEARING and ordered: after the story id is resolved
 * (the id is what the server keys the store on) and before the upload (the
 * upload is what consumes the store). A throw from it must abort the
 * translate -- rendering past a failed install would file a glossary-less
 * render behind a glossary cache key for 24 hours. */
test("the glossary is installed after the story and before the upload", async () => {
  const h = harness({ cfg: { glossaryTerms: [{ source: "a", translation: "b" }] } });
  await h.translate(REQUEST, PAGE_URL);
  const order = h.calls.map(([who]) => who);
  const story = order.indexOf("storyId");
  const install = order.indexOf("installGlossary");
  const upload = order.indexOf("fetch");
  assert.ok(story >= 0 && install > story, "installGlossary runs, after storyId");
  assert.ok(upload > install, "the upload waits for the install");
  assert.equal(h.called("installGlossary")[0][1][1], "story-1", "installed under the request's own story id");
});

test("a glossary install failure aborts the translate before the upload", async () => {
  const h = harness({
    cfg: { glossaryTerms: [{ source: "a", translation: "b" }] },
    stubs: {
      installGlossary: async () => {
        throw new Error("glossary install failed: server 400");
      },
    },
  });
  await assert.rejects(() => h.translate(REQUEST, PAGE_URL), /glossary install failed/);
  assert.equal(h.called("fetch").length, 0, "nothing was uploaded");
  assert.equal(h.called("cachePut").length, 0, "nothing was cached");
});

/* The response-side guard: the wire's `glossary_terms` is the server saying
 * which arm actually ran. A mismatch is shown and never remembered --
 * keepArtIgnored's rule, one field over -- because it is the one desync the
 * install path cannot see (terms cleared here while the server still holds
 * some, or a restart between the install and this reply). */
test("a glossary skew is shown but never cached", async () => {
  const h = harness({
    cfg: { glossaryTerms: [{ source: "a", translation: "b" }] },
    payload: { image: "data:image/png;base64,AA" }, // no glossary_terms: 0 ran
  });
  const reply = await h.translate(REQUEST, PAGE_URL);
  assert.equal(reply.ok, true, "the page is still shown");
  assert.equal(h.called("cachePut").length, 0, "the wrong-arm render is not remembered");
});

test("a matching glossary count caches normally", async () => {
  const h = harness({
    cfg: { glossaryTerms: [{ source: "a", translation: "b" }] },
    payload: { image: "data:image/png;base64,AA", glossary_terms: 1 },
  });
  await h.translate(REQUEST, PAGE_URL);
  assert.equal(h.called("cachePut").length, 1);
});

test("a glossary-less reader is not skewed by a glossary-less server", async () => {
  // cfg carries no glossaryTerms at all (an older stored config, or simply no
  // terms): absent must compare equal to the absent wire field, or every
  // ordinary page would silently stop being cached.
  const h = harness();
  await h.translate(REQUEST, PAGE_URL);
  assert.equal(h.called("cachePut").length, 1);
});

/* The count is not the identity: an edit that keeps the term count (fixing
 * one translation) is invisible to `glossary_terms`, so the reply-time
 * re-read of the session terms is what closes the same-count window.
 * Residual (accepted): an edit-and-revert
 * completing entirely inside one request's flight is still invisible --
 * closing that needs the server to echo a term digest, a wire change. */
test("a same-count term edit mid-flight is shown but never cached", async () => {
  const h = harness({
    cfg: { glossaryTerms: [{ source: "a", translation: "b" }] },
    payload: { image: "data:image/png;base64,AA", glossary_terms: 1 },
    stubs: { glossaryDrifted: async () => true },
  });
  const reply = await h.translate(REQUEST, PAGE_URL);
  assert.equal(reply.ok, true, "the page is still shown");
  assert.equal(h.called("cachePut").length, 0, "the maybe-wrong-arm render is not remembered");
});

/* ------------------------------------------------------------- the retry
 *
 * The server's sampler seed is a fixed constant, so an identical request is a
 * byte-identical answer -- and the cache would short-circuit
 * even that. A retry therefore has three duties, each pinned here: never
 * answer from the cached RESULT, ride a fresh u32 seed out on the form, and
 * still overwrite the stored entry on the way back so the re-roll is what the
 * next visit serves. Its BYTES come from the entry's stored `source` -- the
 * first live click proved the original url is routinely dead by retry time
 * (expiring signed urls, revoked blob: urls), and the stored stream is the
 * exact one that produced srcHash, so the overwrite provably lands on the
 * SAME key. Born red against the shapes they replaced. The
 * harness pins crypto.getRandomValues to 0x12345678 so the seed assert is
 * exact rather than a range check. */

test("a retry answers from the server, on the STORED source bytes", async () => {
  const h = harness({
    // A tier-2 hit for any ordinary request -- the retry must not serve it.
    entry: { blob: {}, seamed: [] },
  });
  const reply = await h.translate({ ...REQUEST, retry: true, bytes: null }, PAGE_URL);

  assert.equal(reply.ok, true);
  assert.ok(!reply.cached, "a retry must never answer from the cache");
  assert.equal(h.called("cacheGetByKey").length, 0, "the cached RESULT is not even read");
  assert.equal(h.called("fetch").length, 1, "the retry must reach the server");
  assert.equal(h.called("cacheGetByUrl").length, 1, "the bytes come from the stored entry");
  const hashed = h.called("sha256Hex")[0][1][0];
  assert.deepEqual(
    Array.from(hashed),
    [9, 9, 9],
    "the stored source is what gets hashed, so the overwrite hits the same key"
  );
});

test("a retry with no stored source is refused with the fresh-translation message", async () => {
  const h = harness({ priorEntry: null });
  await assert.rejects(
    () => h.translate({ ...REQUEST, retry: true, bytes: null }, PAGE_URL),
    (err) => /fresh|translate it again/i.test(String(err.message))
  );
  assert.equal(h.called("fetch").length, 0, "nothing to upload, nothing to send");
});

test("a retry hands translateForm a seed; an ordinary translate hands none", async () => {
  const h = harness();
  await h.translate({ ...REQUEST, retry: true, bytes: null }, PAGE_URL);
  const seed = h.called("translateForm")[0][1][4];
  assert.equal(seed, 0x12345678, "the retry's seed is the fresh draw's whole mechanism");

  const plain = harness();
  await plain.translate(REQUEST, PAGE_URL);
  const plainSeed = plain.called("translateForm")[0][1][4];
  assert.ok(
    plainSeed === null || plainSeed === undefined,
    "an ordinary translate must keep the deterministic draw"
  );
});

test("a retry's reply overwrites the cache entry under the same key", async () => {
  const h = harness();
  await h.translate({ ...REQUEST, retry: true, bytes: null }, PAGE_URL);
  assert.equal(h.called("cachePut").length, 1, "the re-roll is what the next visit serves");
  const put = h.called("cachePut")[0][1][0];
  assert.equal(put.srcHash, "hash", "same hash, same fingerprint: the SAME key, overwritten");
  assert.equal(put.fingerprint, "fp");
});

test("a live translate stores its own source bytes, so a later retry has them", async () => {
  const h = harness();
  await h.translate(REQUEST, PAGE_URL);
  const put = h.called("cachePut")[0][1][0];
  assert.ok(put.source, "the entry must carry the bytes the hash was taken from");
  assert.equal(put.source.type, "image/png", "the upload blob itself, type included");
});

/* ------------------------------------------------------- the box editor
 *
 * The box editor rides the retry transport. Three couplings pinned, born
 * red: an edit apply hands its edits to translateForm and SUPPRESSES
 * the seed (deterministic corrections; the retry button stays the only
 * re-roll); a plain retry re-sends the entry's STORED edits so a re-roll
 * never resurrects a deleted box; and the reply's region boxes are stored
 * with the entry so a cache hit can still open the editor. */

const EDITS = {
  add: [{ x: 10, y: 20, width: 100, height: 40 }],
  remove: [{ x: 500, y: 500, width: 60, height: 60 }],
};

test("an edit apply threads its edits to translateForm and suppresses the seed", async () => {
  const h = harness();
  await h.translate({ ...REQUEST, retry: true, bytes: null, edits: EDITS }, PAGE_URL);
  const args = h.called("translateForm")[0][1];
  assert.deepEqual(args[5], EDITS, "the edits must reach the form builder");
  assert.ok(
    args[4] === null || args[4] === undefined,
    "an edit apply is deterministic -- the retry button stays the only re-roll"
  );
  const put = h.called("cachePut")[0][1][0];
  assert.deepEqual(put.edits, EDITS, "the entry remembers the cumulative edits");
});

test("a plain retry re-sends the STORED edits and keeps its seed", async () => {
  const h = harness({
    priorEntry: {
      source: {
        size: 3,
        type: "image/png",
        arrayBuffer: async () => new Uint8Array([9, 9, 9]).buffer,
      },
      edits: EDITS,
    },
  });
  await h.translate({ ...REQUEST, retry: true, bytes: null }, PAGE_URL);
  const args = h.called("translateForm")[0][1];
  assert.deepEqual(args[5], EDITS, "a re-roll must not resurrect a deleted box");
  assert.equal(args[4], 0x12345678, "a plain retry still re-rolls");
});

/* The placement list rides the same edits object. The trap this
 * fences: background.js keys "suppress the seed" on `retry && !edits`, so
 * anything that ever normalized a placement-only edits object to falsy would
 * silently turn a placement apply into a re-roll. */

const PLACE_ONLY = {
  add: [],
  remove: [],
  place: [{
    x: 410, y: 120, width: 180, height: 240,
    place_x: 430, place_y: 300, place_width: 300, place_height: 160,
  }],
};

test("an edit apply carrying only a placement still suppresses the seed", async () => {
  const h = harness();
  await h.translate({ ...REQUEST, retry: true, bytes: null, edits: PLACE_ONLY }, PAGE_URL);
  const args = h.called("translateForm")[0][1];
  assert.deepEqual(args[5], PLACE_ONLY, "the placement must reach the form builder");
  assert.ok(
    args[4] === null || args[4] === undefined,
    "a placement apply is deterministic -- it must not become a re-roll"
  );
});

test("a plain retry re-sends the stored placement too", async () => {
  const withPlace = { ...EDITS, place: PLACE_ONLY.place };
  const h = harness({
    priorEntry: {
      source: {
        size: 3,
        type: "image/png",
        arrayBuffer: async () => new Uint8Array([9, 9, 9]).buffer,
      },
      edits: withPlace,
    },
  });
  await h.translate({ ...REQUEST, retry: true, bytes: null }, PAGE_URL);
  const args = h.called("translateForm")[0][1];
  assert.deepEqual(
    args[5],
    withPlace,
    "a re-roll must no more drop a placement than resurrect a deleted box"
  );
});

/* The entry stores each region's (source -> translated) pair so a
 * later EDIT APPLY can pin the untouched bubbles' wording. */

test("a live translate stores the reply's pins for a later apply", async () => {
  const h = harness({
    reported: {
      regions: [
        { x: 5, y: 6, width: 70, height: 80, source: "カサカサ…", translated: "AH..." },
        { x: 90, y: 6, width: 70, height: 80, source: "ごめん", translated: "SORRY.", refused: "watermark" },
        { x: 5, y: 200, width: 70, height: 80, source: "", translated: "GHOST" },
      ],
    },
  });
  await h.translate(REQUEST, PAGE_URL);
  const put = h.called("cachePut")[0][1][0];
  assert.deepEqual(
    put.pins,
    [{ s: "カサカサ…", t: "AH..." }],
    "lettered regions pin; refused and sourceless ones never do"
  );
});

test("an edit apply sends the stored pins and a plain retry sends none", async () => {
  const stored = [{ s: "カサカサ…", t: "AH..." }];
  const entry = {
    source: {
      size: 3,
      type: "image/png",
      arrayBuffer: async () => new Uint8Array([9, 9, 9]).buffer,
    },
    edits: EDITS,
    pins: stored,
  };
  const apply = harness({ priorEntry: entry });
  await apply.translate({ ...REQUEST, retry: true, bytes: null, edits: EDITS }, PAGE_URL);
  assert.deepEqual(
    apply.called("translateForm")[0][1][6],
    stored,
    "the apply pins every bubble the edit did not touch"
  );

  const reroll = harness({ priorEntry: entry });
  await reroll.translate({ ...REQUEST, retry: true, bytes: null }, PAGE_URL);
  const sent = reroll.called("translateForm")[0][1][6];
  assert.ok(
    sent === null || sent === undefined,
    "a plain retry pins NOTHING -- it exists to re-roll"
  );
});

test("a live translate stores the reply's boxes for the editor", async () => {
  const h = harness({
    reported: {
      regions: [{ x: 5, y: 6, width: 70, height: 80, source: "猫", translated: "cat" }],
    },
  });
  const reply = await h.translate(REQUEST, PAGE_URL);
  const put = h.called("cachePut")[0][1][0];
  assert.deepEqual(put.boxes, [{ x: 5, y: 6, width: 70, height: 80 }]);
  assert.deepEqual(reply.boxes, [{ x: 5, y: 6, width: 70, height: 80 }]);
});
