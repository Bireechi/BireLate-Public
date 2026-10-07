/* Executable pinning for content.js's translate(img) -- the six-piece state
 * machine whose ordering is enforced by nothing but comments.
 *
 * Run:  node --test tests/content-translate.test.js
 *
 * (Not `node --test tests/` -- see the header of seam.test.js.)
 *
 * WHY THIS FILE EXISTS. translate(img) coordinates inFlight / originals /
 * handled / failed / spinning / joinable across try/catch/finally, and it
 * already shipped one silent defect from exactly this coupling: scheduling
 * the seam from inside the try made every join bail on runSeam's own
 * in-flight guard, invisibly (the comment at the `joinable` declaration
 * narrates it). Before this file, no test could reach any of it. This file
 * lifts the SHIPPED function and asserts on the ORDER of the recorded calls,
 * which is the property a refactor moves first.
 *
 * ANCHORS THIS FILE BINDS (update in the same commit as any move, and
 * re-prove a red):
 *   START  "async function translate(img)"
 *   END    "function requestTranslate("
 * seam.test.js's own content.js lift ends at seamBoundaryKey, immediately
 * before this one begins -- code inserted between the two moves BOTH.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const CONTENT_JS = fs.readFileSync(
  path.join(ROOT, "extension", "content.js"),
  "utf8"
);

/* The signature carries `opts` for the retry, so the anchor includes it. */
const START_ANCHOR = "async function translate(img, opts";
const END_ANCHOR = "function requestTranslate(";

function shippedTranslate(stubs) {
  const start = CONTENT_JS.indexOf(START_ANCHOR);
  assert.ok(start > 0, "could not find translate(img) in content.js");
  const end = CONTENT_JS.indexOf(END_ANCHOR);
  assert.ok(end > start, "could not find requestTranslate() after translate(img)");
  const source = CONTENT_JS.slice(start, end);
  const names = Object.keys(stubs);
  const factory = new Function(...names, `${source}; return translate;`);
  return factory(...names.map((name) => stubs[name]));
}

function harness(overrides = {}) {
  const calls = [];
  const track = (name, impl) => (...args) => {
    calls.push([name, ...args.slice(0, 3)]);
    return impl ? impl(...args) : undefined;
  };
  const called = (name) => calls.filter(([who]) => who === name);
  const orderOf = (name) => calls.findIndex(([who]) => who === name);

  const backing = { inFlight: new WeakSet(), handled: new WeakSet(), failed: new WeakSet() };
  const trackedSet = (name) => ({
    has: (x) => backing[name].has(x),
    add: (x) => {
      calls.push([`${name}.add`]);
      backing[name].add(x);
    },
    delete: (x) => {
      calls.push([`${name}.delete`]);
      return backing[name].delete(x);
    },
  });
  const originalsBacking = new WeakMap();
  const originals = {
    has: (x) => originalsBacking.has(x),
    get: (x) => originalsBacking.get(x),
    set: (x, v) => {
      calls.push(["originals.set", v]);
      originalsBacking.set(x, v);
    },
    delete: (x) => {
      calls.push(["originals.delete"]);
      return originalsBacking.delete(x);
    },
  };

  const settings = Object.assign({ autoTranslate: true }, overrides.settings || {});
  const replies = Object.assign(
    {
      "cache-lookup": { ok: true, hit: false },
      translate: { ok: true, dataUrl: "data:image/png;base64,AA" },
    },
    overrides.replies || {}
  );

  const stubs = {
    inFlight: trackedSet("inFlight"),
    handled: trackedSet("handled"),
    failed: trackedSet("failed"),
    originals,
    settings,
    setSrc: track("setSrc"),
    setBusy: track("setBusy"),
    markBusy: track("markBusy"),
    markDone: track("markDone"),
    flashError: track("flashError"),
    noteMisses: track("noteMisses"),
    noteSlice: track("noteSlice"),
    noteBoxes: track("noteBoxes"),
    scheduleSeam: track("scheduleSeam"),
    syncAuto: track("syncAuto"),
    pageShape: track("pageShape", () => "strip"),
    imageBytes: track("imageBytes", async () => ({
      type: "image/png",
      arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
    })),
    browser: {
      runtime: {
        sendMessage: track("sendMessage", async (message) => {
          const reply = replies[message.type];
          if (overrides.onMessage) overrides.onMessage(message);
          return typeof reply === "function" ? reply(message) : reply;
        }),
      },
      storage: { local: { set: track("storage.set", async () => {}) } },
    },
    ...(overrides.stubs || {}),
  };

  const img = Object.assign(
    {
      currentSrc: "https://img.example/p/001.png",
      src: "https://img.example/p/001.png",
      naturalWidth: 800,
      naturalHeight: 1200,
      isConnected: true,
    },
    overrides.img || {}
  );

  return {
    translate: shippedTranslate(stubs),
    calls,
    called,
    orderOf,
    img,
    settings,
    backing,
    originalsBacking,
  };
}

test("an image already in flight is left entirely alone", async () => {
  const h = harness();
  h.backing.inFlight.add(h.img);
  await h.translate(h.img);
  assert.equal(h.called("sendMessage").length, 0);
  assert.equal(h.called("setBusy").length, 0);
});

test("a second click restores the original and never re-enters the machine", async () => {
  const h = harness();
  h.originalsBacking.set(h.img, "https://img.example/original.png");
  await h.translate(h.img);

  assert.deepEqual(h.called("setSrc")[0].slice(1), [
    h.img,
    "https://img.example/original.png",
    "original",
  ]);
  assert.equal(h.called("sendMessage").length, 0);
  assert.equal(h.called("inFlight.add").length, 0, "the restore returns before the add");
  assert.equal(h.called("originals.delete").length, 1);
});

/* THE SHIPPED-BUG ORDER PIN. The seam may only be scheduled after
 * inFlight.delete -- runSeam refuses a pair either of whose images is in
 * flight, and scheduling from inside the try made every join bail silently.
 * Proven able to fail: moving `if (joinable) scheduleSeam(joinable)` above
 * `inFlight.delete(img)` in the finally turns both order assertions red. */
test("a cache hit schedules the seam only after leaving flight, and raises no spinner", async () => {
  const h = harness({
    replies: {
      "cache-lookup": {
        ok: true,
        hit: true,
        dataUrl: "data:image/png;base64,HIT",
        seamed: [],
      },
    },
  });
  await h.translate(h.img);

  assert.equal(h.called("setBusy").length, 0, "a hit must not touch the one shared bar");
  assert.ok(h.orderOf("scheduleSeam") > h.orderOf("inFlight.delete"));
  assert.equal(h.called("noteSlice").length, 1);
});

test("a live run raises the spinner, paints, and leaves flight before the seam", async () => {
  const h = harness();
  await h.translate(h.img);

  /* The real order, as shipped: the cache lookup happens BEFORE the spinner
   * (a cached gallery must not flash a bar it will not use), and the seam is
   * scheduled strictly after inFlight.delete. */
  const sendIndex = (type) =>
    h.calls.findIndex(([who, arg]) => who === "sendMessage" && arg && arg.type === type);
  const milestones = [
    h.orderOf("inFlight.add"),
    sendIndex("cache-lookup"),
    h.orderOf("setBusy"),
    sendIndex("translate"),
    h.orderOf("originals.set"),
    h.orderOf("inFlight.delete"),
    h.orderOf("scheduleSeam"),
  ];
  for (let i = 0; i < milestones.length; i += 1) {
    assert.ok(milestones[i] >= 0, `milestone ${i} never happened: ${JSON.stringify(h.calls)}`);
    if (i > 0) {
      assert.ok(
        milestones[i] > milestones[i - 1],
        `milestone ${i} out of order: ${JSON.stringify(h.calls)}`
      );
    }
  }
  const busy = h.called("setBusy");
  assert.deepEqual(
    busy.map((c) => c[1]),
    [true, false],
    "the spinner an image raised is the one it lowers"
  );
  assert.equal(h.called("markDone").length, 1);
});

test("a page that moved on mid-run is told, not painted", async () => {
  const h = harness({
    onMessage: (message) => {
      if (message.type === "translate") {
        h.img.currentSrc = "https://img.example/p/002.png";
        h.img.src = "https://img.example/p/002.png";
      }
    },
  });
  await h.translate(h.img);

  assert.equal(h.called("flashError").length, 1);
  assert.equal(h.called("originals.set").length, 0, "the stale reply must not be painted");
  assert.equal(h.called("scheduleSeam").length, 0, "an unpainted image is not joinable");
  assert.equal(h.called("markDone").length, 1, "the badge never spins forever");
  assert.deepEqual(h.called("setBusy").map((c) => c[1]), [true, false]);
});

/* The starved branch is THREE effects -- the in-memory flip, the button
 * sync, and the persisted write -- plus the longer toast. An ordinary
 * failure is none of them. */
test("a starved failure turns auto mode off everywhere, once", async () => {
  const h = harness({
    replies: { translate: { ok: false, error: "3 GiB short", kind: "insufficient_memory" } },
  });
  await h.translate(h.img);

  assert.equal(h.settings.autoTranslate, false);
  assert.equal(h.called("syncAuto").length, 1);
  assert.deepEqual(h.called("storage.set")[0][1], { autoTranslate: false });
  assert.equal(h.called("failed.add").length, 1);
  assert.equal(h.called("flashError")[0][3], 6000, "a starved toast holds longer");
  assert.equal(h.called("scheduleSeam").length, 0);
});

test("an ordinary failure marks the image and touches nothing global", async () => {
  const h = harness({
    replies: { translate: { ok: false, error: "server 500: boom" } },
  });
  await h.translate(h.img);

  assert.equal(h.settings.autoTranslate, true);
  assert.equal(h.called("storage.set").length, 0);
  assert.equal(h.called("syncAuto").length, 0);
  assert.equal(h.called("failed.add").length, 1);
  assert.equal(h.called("flashError")[0][3], 2600, "an ordinary toast holds the short beat");
  assert.equal(h.called("inFlight.delete").length, 1);
});

/* ------------------------------------------------------------- the retry
 *
 * A retry is a fresh DRAW of an already-translated image, which is exactly the
 * state the second-click toggle above intercepts -- so the retry rides an
 * `opts` flag through the same translate() rather than a second machine. Its
 * load-bearing differences, each pinned here: the cache is not consulted (the
 * stored entry is what the retry exists to replace), and NO bytes are fetched
 * or sent at all -- the first live click on a real host proved the original
 * url is routinely dead by retry time (expiring signed urls, revoked blob:
 * urls), and the canvas is showing OUR lettering, so the background's stored
 * source bytes are the only honest stream. Born red against the shape they
 * replaced. */

const RETRY_ORIGINAL_URL = "https://img.example/p/001.png";
const RETRY_SHOWN_URL = "data:image/png;base64,SHOWN";

test("a retry on a translated image reaches the server instead of restoring", async () => {
  const h = harness({ img: { currentSrc: RETRY_SHOWN_URL, src: RETRY_SHOWN_URL } });
  h.originalsBacking.set(h.img, RETRY_ORIGINAL_URL);
  await h.translate(h.img, { retry: true });

  const sent = h.calls.filter(([who]) => who === "sendMessage");
  assert.ok(
    !sent.some(([, m]) => m && m.type === "cache-lookup"),
    "a retry must not consult the cache it exists to replace"
  );
  const wire = sent.find(([, m]) => m && m.type === "translate");
  assert.ok(wire, "the retry must reach the background");
  assert.equal(wire[1].retry, true, "the background must be told this is a re-roll");
  assert.equal(
    wire[1].url,
    RETRY_ORIGINAL_URL,
    "the retry files under the ORIGINAL url, never the shown data: url"
  );
  assert.ok(!wire[1].bytes, "a retry carries no bytes -- the background holds the source");
  assert.ok(h.originalsBacking.has(h.img), "the original survives for the next restore");
  /* The page-changed guard must compare against what was SHOWN when the retry
   * left (the data: url, unchanged), not against the original url -- the old
   * comparison reads every successful retry as "the page changed" and drops
   * the paint. The paint below is therefore also the guard's pin. */
  assert.ok(
    h.called("setSrc").some((c) => c[2] === "data:image/png;base64,AA"),
    "the fresh draw must be painted back"
  );
});

test("a retry never fetches: no imageBytes call, live or canvas", async () => {
  const h = harness({ img: { currentSrc: RETRY_SHOWN_URL, src: RETRY_SHOWN_URL } });
  h.originalsBacking.set(h.img, RETRY_ORIGINAL_URL);
  await h.translate(h.img, { retry: true });

  /* The first live click proved the refetch design wrong: the host's url had
   * already expired, and the canvas would have re-encoded the extension's OWN lettering
   * back through the pipeline. The stored source in the background is the
   * only stream that is both alive and honest. */
  assert.equal(h.called("imageBytes").length, 0, "nothing on this page is worth fetching");
});

test("an edit apply rides the retry transport with its edits on the wire", async () => {
  const h = harness({ img: { currentSrc: RETRY_SHOWN_URL, src: RETRY_SHOWN_URL } });
  h.originalsBacking.set(h.img, RETRY_ORIGINAL_URL);
  const edits = { add: [{ x: 1, y: 2, width: 3, height: 4 }], remove: [] };
  await h.translate(h.img, { retry: true, edits });

  const wire = h.calls
    .filter(([who]) => who === "sendMessage")
    .find(([, m]) => m && m.type === "translate");
  assert.ok(wire);
  assert.equal(wire[1].retry, true);
  assert.deepEqual(wire[1].edits, edits, "the editor's changes must reach the background");
});

test("an ordinary translate sends no retry flag on the wire", async () => {
  const h = harness();
  await h.translate(h.img);
  const wire = h.calls.filter(([who]) => who === "sendMessage").find(([, m]) => m.type === "translate");
  assert.ok(wire);
  assert.ok(!wire[1].retry, "only the retry button may vary the draw");
});
