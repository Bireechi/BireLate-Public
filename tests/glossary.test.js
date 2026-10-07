/* Unit tests for the per-series glossary: the module, the installer, the key.
 *
 * Run:  node --test tests/glossary.test.js
 *
 * (Not `node --test tests/` -- see the header of seam.test.js for why a bare
 * directory argument reports a green suite failing for an unrelated reason.)
 *
 * WHY THIS FILE EXISTS. The glossary rides three couplings that can
 * each desync silently: the editor's validation against the server's
 * (server/src/glossary.rs is the authority -- 64 terms, 256 BYTES a field),
 * the cache fingerprint against the terms actually installed, and the install
 * POST against the /translate that consumes it. The first two live in
 * glossary.js and cache.js and are tested directly; the ordering lives in
 * translate() and is pinned executably in translate.test.js; the two
 * call-sites-exist checks at the bottom are static, and say so.
 *
 * ANCHORS THIS FILE BINDS (a refactor that moves one must update this file
 * in the same commit and re-prove a red):
 *   "async function installGlossary("  ..  "async function glossaryGet"
 *   (background.js; both asserted before lifting)
 *   "async function glossaryLedgerFor("  ..  "async function glossarySuggest"
 *   (background.js, the term-ledger block; both asserted before lifting)
 * cache.js's settingsFingerprint is lifted with cache-key.test.js's own
 * anchors ("function settingsFingerprint" .. "const cacheKey =").
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const {
  GLOSSARY_MAX_TERMS,
  GLOSSARY_MAX_TERM_BYTES,
  glossaryParse,
  glossaryFormat,
  glossaryFingerprint,
  glossaryStorageKey,
  GLOSSARY_LEDGER_MAX_TERMS,
  GLOSSARY_LEDGER_MAX_RENDERINGS,
  glossaryLedgerStorageKey,
  glossaryLedgerTermKey,
  glossaryLedgerAdd,
  glossaryLedgerRenderings,
  glossaryLedgerSuggestions,
  glossaryPinHit,
} = require(path.join(ROOT, "extension", "glossary.js"));

const BACKGROUND_JS = fs.readFileSync(
  path.join(ROOT, "extension", "background.js"),
  "utf8"
);
const CACHE_JS = fs.readFileSync(path.join(ROOT, "extension", "cache.js"), "utf8");

/* ------------------------------------------------------------------ parsing */

test("the editor format parses, with comments and blanks passed over", () => {
  const { terms, errors } = glossaryParse(
    "# names\n흑월회 = the Black Moon Society\n\n풍랑=storm waves\n  담왕  =  Prince Dam  "
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(terms, [
    { source: "흑월회", translation: "the Black Moon Society" },
    { source: "풍랑", translation: "storm waves" },
    { source: "담왕", translation: "Prince Dam" },
  ]);
});

test("the translation may carry an equals sign; the source may not", () => {
  const { terms, errors } = glossaryParse("x = a = b");
  assert.deepEqual(errors, []);
  assert.deepEqual(terms, [{ source: "x", translation: "a = b" }]);
});

test("a line without an equals sign is an error naming its line", () => {
  const { terms, errors } = glossaryParse("흑월회 the Black Moon Society");
  assert.deepEqual(terms, []);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /line 1/);
});

test("empty halves are refused, each naming the empty side", () => {
  const { errors } = glossaryParse("= x\ny =");
  assert.equal(errors.length, 2);
  assert.match(errors[0], /empty source/);
  assert.match(errors[1], /empty translation/);
});

test("the byte cap is the server's cap, measured in BYTES not characters", () => {
  // 86 Korean syllables is 258 UTF-8 bytes -- inside the JS length limit a
  // character count would wave through, over the server's 256-byte refusal.
  const over = "가".repeat(86);
  assert.ok(over.length < GLOSSARY_MAX_TERM_BYTES, "the trap this test exists for");
  const { errors } = glossaryParse(`${over} = x`);
  assert.equal(errors.length, 1);
  assert.match(errors[0], new RegExp(String(GLOSSARY_MAX_TERM_BYTES)));
});

test("duplicate sources are refused here even though the server accepts them", () => {
  const { terms, errors } = glossaryParse("a = x\na = y");
  assert.deepEqual(terms, [{ source: "a", translation: "x" }]);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /duplicate/);
});

test("the term-count cap is the server's", () => {
  const lines = Array.from({ length: GLOSSARY_MAX_TERMS + 1 }, (_, i) => `s${i} = t${i}`);
  const { errors } = glossaryParse(lines.join("\n"));
  assert.equal(errors.length, 1);
  assert.match(errors[0], new RegExp(String(GLOSSARY_MAX_TERMS)));
});

test("format round-trips what parse produced", () => {
  const text = "흑월회 = the Black Moon Society\n풍랑 = storm waves";
  const { terms } = glossaryParse(text);
  assert.equal(glossaryFormat(terms), text);
});

/* -------------------------------------------------------------- fingerprint */

test("no terms is the empty string, because absence must not change old keys", () => {
  assert.equal(glossaryFingerprint([]), "");
  assert.equal(glossaryFingerprint(undefined), "");
});

test("the fingerprint is stable, order-sensitive, and content-sensitive", () => {
  const a = [{ source: "s", translation: "t" }, { source: "u", translation: "v" }];
  const b = [{ source: "u", translation: "v" }, { source: "s", translation: "t" }];
  assert.equal(glossaryFingerprint(a), glossaryFingerprint(a));
  assert.notEqual(glossaryFingerprint(a), glossaryFingerprint(b), "order is prompt order");
  const c = [{ source: "s", translation: "T" }, { source: "u", translation: "v" }];
  assert.notEqual(glossaryFingerprint(a), glossaryFingerprint(c));
});

test("field boundaries cannot alias -- the collision the separator exists for", () => {
  // Under a printable separator these two fed the hash one identical stream.
  const a = [{ source: "a", translation: "b c" }];
  const b = [{ source: "a b", translation: "c" }];
  assert.notEqual(glossaryFingerprint(a), glossaryFingerprint(b));
});

test("the storage key is namespaced", () => {
  assert.equal(glossaryStorageKey("host|series"), "glossary:host|series");
});

/* --------------------------------------------- the fingerprint in the KEY
 *
 * Lifted with cache-key.test.js's own anchors so both suites break together
 * if settingsFingerprint moves. The load-bearing half is the FIRST assertion:
 * a glossary-less config must key byte-identically to a config from before
 * the field existed, or shipping this feature would invalidate every cached
 * page for every reader with no glossary at all. */

function shippedFingerprint() {
  const src = CACHE_JS.slice(
    CACHE_JS.indexOf("function settingsFingerprint"),
    CACHE_JS.indexOf("const cacheKey =")
  );
  assert.ok(src.length > 0, "could not lift settingsFingerprint out of cache.js");
  return new Function(`${src}; return settingsFingerprint;`)();
}

const BASE_CFG = {
  targetLanguage: "en-US",
  ocr: "hunyuan-ocr-1.5",
  inpainting: "lama",
  provider: "local",
  llm: "",
  sourceLanguage: "ko",
  keepArt: false,
  storyId: "S1",
};

test("a glossary-less reader keys exactly as before the field existed", () => {
  const settingsFingerprint = shippedFingerprint();
  const before = settingsFingerprint({ ...BASE_CFG });
  const absent = settingsFingerprint({ ...BASE_CFG, glossaryFingerprint: "" });
  const missing = settingsFingerprint({ ...BASE_CFG, glossaryFingerprint: undefined });
  assert.equal(before, absent);
  assert.equal(before, missing);
});

test("a glossary changes the key, and a different glossary changes it again", () => {
  const settingsFingerprint = shippedFingerprint();
  const none = settingsFingerprint({ ...BASE_CFG });
  const one = settingsFingerprint({ ...BASE_CFG, glossaryFingerprint: "g1-abcd1234" });
  const two = settingsFingerprint({ ...BASE_CFG, glossaryFingerprint: "g2-ffff0000" });
  assert.notEqual(none, one);
  assert.notEqual(one, two);
});

/* ------------------------------------------------------------ the installer
 *
 * Lifted from the shipped source like translate() is in translate.test.js.
 * The stubs record; the assertions read both the reply and who was called. */

function shippedInstall(stubs) {
  const start = BACKGROUND_JS.indexOf("async function installGlossary(");
  assert.ok(start > 0, "could not find installGlossary() in background.js");
  const end = BACKGROUND_JS.indexOf("async function glossaryGet");
  assert.ok(end > start, "could not find glossaryGet, which ends the lift");
  const source = BACKGROUND_JS.slice(start, end);
  const names = Object.keys(stubs);
  return new Function(...names, `${source}; return installGlossary;`)(
    ...names.map((name) => stubs[name])
  );
}

function installHarness(fetchImpl, extra) {
  const posts = [];
  const stubs = {
    base: () => "http://127.0.0.1:8765",
    /* Reads the LIVE cfg.token, because the 401 self-heal's whole point is
     * that a resynced token reaches the retry through the same cfg object. */
    authHeaders: (cfg) => ({ "X-Koharu-Token": (cfg && cfg.token) || "t" }),
    detail: async () => "the detail",
    fetch: async (url, init) => {
      posts.push({ url, init });
      return fetchImpl
        ? fetchImpl(url, init)
        : { ok: true, status: 200, json: async () => ({ terms: 1 }) };
    },
    FormData: class {
      constructor() {
        this.fields = [];
      }
      append(name, value) {
        this.fields.push([name, value]);
      }
    },
    /* The 401 self-heal's collaborators; inert on the happy path. Injected
     * as a parameter so tokenResync's `tokenFromHost = null` binds it. */
    tokenFromHost: null,
    NATIVE_HOST: "dev.birelate.host",
    browser: {
      runtime: { sendNativeMessage: async () => null },
      storage: { local: { set: async () => {} } },
    },
    ...(extra || {}),
  };
  return { install: shippedInstall(stubs), posts };
}

const ONE_TERM = [{ source: "흑월회", translation: "the Black Moon Society" }];

test("no story installs nothing", async () => {
  const a = installHarness();
  await a.install({ glossaryTerms: ONE_TERM }, "");
  assert.equal(a.posts.length, 0);
});

/* Empty terms POST a CLEAR rather than skipping. This is the heal for a
 * stranded-server state: a browser restart wipes
 * storage.session while a standing server keeps the terms under the same
 * (URL-derived, restart-stable) story id -- skip the POST and every reply
 * carries glossary_terms > 0 against a glossary-less key, so the skew guard
 * refuses to cache the series for the life of the server process. The server
 * treats an empty install as remove-and-return-0 (glossary.rs::set), so the
 * count check passes unchanged and no empty entry joins the 8-slot store. */
test("empty terms POST the clear, and the zero count round-trips", async () => {
  const h = installHarness(async () => ({ ok: true, status: 200, json: async () => ({ terms: 0 }) }));
  await h.install({ glossaryTerms: [] }, "story-1");
  await h.install({}, "story-1");
  assert.equal(h.posts.length, 2);
  assert.deepEqual(h.posts[0].init.body.fields, [
    ["story", "story-1"],
    ["terms", "[]"],
  ]);
});

test("the install POSTs the story and the terms as /glossary multipart fields", async () => {
  const h = installHarness();
  await h.install({ glossaryTerms: ONE_TERM }, "story-1");
  assert.equal(h.posts.length, 1);
  assert.equal(h.posts[0].url, "http://127.0.0.1:8765/glossary");
  assert.equal(h.posts[0].init.method, "POST");
  assert.deepEqual(h.posts[0].init.headers, { "X-Koharu-Token": "t" });
  assert.deepEqual(h.posts[0].init.body.fields, [
    ["story", "story-1"],
    ["terms", JSON.stringify(ONE_TERM)],
  ]);
});

test("a server error is a throw, not a shrug", async () => {
  const h = installHarness(async () => ({ ok: false, status: 400, json: async () => ({}) }));
  await assert.rejects(
    () => h.install({ glossaryTerms: ONE_TERM }, "story-1"),
    /glossary install failed: server 400/
  );
});

test("a stored count that is not the sent count is the same throw wearing success", async () => {
  const h = installHarness(async () => ({ ok: true, status: 200, json: async () => ({ terms: 0 }) }));
  await assert.rejects(
    () => h.install({ glossaryTerms: ONE_TERM }, "story-1"),
    /stored 0 of 1/
  );
});

/* ------------------------------------------------- static call-site pins
 *
 * STATIC, and labelled as such: translate()'s ordering is pinned executably
 * in translate.test.js, but seamJoin() has no executable harness yet, so its
 * two glossary touchpoints are held by text. A refactor that renames either
 * call must update this count in the same commit. */

test("both /translate call sites install the glossary first (static)", () => {
  const calls = BACKGROUND_JS.split("await installGlossary(cfg, story);").length - 1;
  assert.equal(calls, 2, "translate() and seamJoin() each install before their POST");
});

test("the seam counts its run before refusing a skewed join (static)", () => {
  // The server DID run a full translate; the idle-unload accounting must see
  // it whether or not the join is painted -- translate()'s skew path keeps
  // its noteRunFinished for the same reason.
  const seamStart = BACKGROUND_JS.indexOf("async function seamJoin");
  assert.ok(seamStart > 0, "could not find seamJoin");
  const noted = BACKGROUND_JS.indexOf("noteRunFinished();", seamStart);
  const skew = BACKGROUND_JS.indexOf('skipped: "glossary-skew"', seamStart);
  assert.ok(noted > 0 && skew > 0, "seamJoin lost a landmark");
  assert.ok(noted < skew, "the skew return skips noteRunFinished()");
});

test("the seam refuses to paint a skewed join (static)", () => {
  assert.ok(
    BACKGROUND_JS.includes('skipped: "glossary-skew"'),
    "seamJoin's skew skip is gone -- a wrong-arm join would paint into two cached entries"
  );
});

test("the glossary fingerprint is minted in config() and nowhere else (static)", () => {
  const mints = BACKGROUND_JS.split("cfg.glossaryFingerprint =").length - 1;
  assert.equal(mints, 1, "one minting site, inside config(), or lookup and write desync");
  const minted = BACKGROUND_JS.indexOf("cfg.glossaryFingerprint =");
  const configStart = BACKGROUND_JS.indexOf("async function config(");
  const configEnd = BACKGROUND_JS.indexOf("const base = (cfg)");
  assert.ok(
    minted > configStart && minted < configEnd,
    "the mint has left config(); the cache key will desync"
  );
});

/* ------------------------------------------------------------ the term ledger
 *
 * The authoring problem the ledger answers: a reader cannot say what a term
 * SHOULD be without the official translation, but they can always pick among
 * the renderings they have already read. These tests pin the pure logic the
 * popup suggestions and the alt-click pin both stand on, then lift the
 * shipped background block so the wiring is what is tested, not a copy. */

test("a term key sheds edge punctuation and keeps interior structure", () => {
  assert.equal(glossaryLedgerTermKey("裂风之王！"), "裂风之王");
  assert.equal(glossaryLedgerTermKey("  흑월검대 "), "흑월검대");
  assert.equal(glossaryLedgerTermKey("灵光识·苍炎之王"), "灵光识·苍炎之王");
});

test("a sentence is not a term", () => {
  assert.equal(glossaryLedgerTermKey("문을 닫지 마라。"), "");
  assert.equal(glossaryLedgerTermKey("this is a whole sentence of words"), "");
  assert.equal(glossaryLedgerTermKey("a".repeat(25)), "");
  assert.equal(glossaryLedgerTermKey(""), "");
});

test("the ledger counts renderings per term", () => {
  const ledger = {};
  glossaryLedgerAdd(ledger, "흑월검대", "the Moon Blades", 1);
  glossaryLedgerAdd(ledger, "흑월검대!", "the Moon Blades", 2);
  glossaryLedgerAdd(ledger, "흑월검대", "the moon squad", 3);
  assert.deepEqual(glossaryLedgerRenderings(ledger, "흑월검대"), [
    { text: "the Moon Blades", count: 2 },
    { text: "the moon squad", count: 1 },
  ]);
});

test("the renderings cap evicts the least seen, never the newest observation", () => {
  const ledger = {};
  for (let i = 0; i < GLOSSARY_LEDGER_MAX_RENDERINGS; i += 1) {
    for (let n = 0; n <= i; n += 1) glossaryLedgerAdd(ledger, "term", `way-${i}`, n);
  }
  glossaryLedgerAdd(ledger, "term", "the newest", 99);
  const texts = glossaryLedgerRenderings(ledger, "term").map((one) => one.text);
  assert.equal(texts.length, GLOSSARY_LEDGER_MAX_RENDERINGS);
  assert.ok(texts.includes("the newest"), "the just-observed rendering was evicted");
  assert.ok(!texts.includes("way-0"), "the least-seen rendering survived the cap");
});

test("the term cap evicts the oldest-touched series entry", () => {
  const ledger = {};
  for (let i = 0; i < GLOSSARY_LEDGER_MAX_TERMS; i += 1) {
    glossaryLedgerAdd(ledger, `term-${i}`, "x", i + 10);
  }
  glossaryLedgerAdd(ledger, "latecomer", "y", 5000);
  assert.equal(Object.keys(ledger).length, GLOSSARY_LEDGER_MAX_TERMS);
  assert.ok(!ledger["term-0"], "the oldest term survived the cap");
  assert.ok(ledger["latecomer"]);
});

test("suggestions are drifted terms only, minus what is already pinned", () => {
  const ledger = {};
  glossaryLedgerAdd(ledger, "steady", "one way", 1);
  glossaryLedgerAdd(ledger, "steady", "one way", 2);
  glossaryLedgerAdd(ledger, "drifted", "way a", 3);
  glossaryLedgerAdd(ledger, "drifted", "way b", 4);
  glossaryLedgerAdd(ledger, "pinned-already", "way c", 5);
  glossaryLedgerAdd(ledger, "pinned-already", "way d", 6);
  const items = glossaryLedgerSuggestions(ledger, [
    { source: "pinned-already", translation: "way c" },
  ]);
  assert.deepEqual(items.map((one) => one.source), ["drifted"]);
  assert.deepEqual(items[0].renderings.map((one) => one.text).sort(), ["way a", "way b"]);
});

test("the pin hit-test takes the smallest containing box and forgives the pad", () => {
  const terms = [
    { x: 0, y: 0, width: 400, height: 400, source: "big", translated: "BIG" },
    { x: 100, y: 100, width: 50, height: 200, source: "col", translated: "COL" },
  ];
  assert.equal(glossaryPinHit(terms, 120, 200).source, "col");
  assert.equal(glossaryPinHit(terms, 300, 300).source, "big");
  assert.equal(glossaryPinHit(terms, 96, 200, 8).source, "col");
  assert.equal(glossaryPinHit(terms, 500, 500), null);
});

/* Lifted out of the SHIPPED background.js -- the ledger block between its two
 * anchors -- so the wiring is what is tested. Stubs record storage traffic. */
function shippedLedgerBlock() {
  const start = BACKGROUND_JS.indexOf("async function glossaryLedgerFor(");
  assert.ok(start > 0, "could not find glossaryLedgerFor() in background.js");
  const stop = BACKGROUND_JS.indexOf("async function glossarySuggest");
  assert.ok(stop > start, "could not find glossarySuggest() after it");
  const source = BACKGROUND_JS.slice(start, BACKGROUND_JS.lastIndexOf("\n}", stop) + 2);

  const session = new Map();
  const writes = [];
  const browser = {
    storage: {
      session: {
        get: async (key) => (session.has(key) ? { [key]: session.get(key) } : {}),
        set: async (record) => {
          for (const key of Object.keys(record)) {
            session.set(key, record[key]);
            writes.push(key);
          }
        },
      },
    },
  };
  const glossarySetCalls = [];
  const build = new Function(
    "browser",
    "glossaryLedgerStorageKey",
    "glossaryLedgerTermKey",
    "glossaryLedgerAdd",
    "glossaryLedgerRenderings",
    "glossaryStorageKey",
    "glossaryFormat",
    "storyPageKey",
    "glossarySet",
    `${source}; return { glossaryLedgerHarvest, translateTerms, glossaryPin, glossaryPinOptions };`
  );
  const lifted = build(
    browser,
    glossaryLedgerStorageKey,
    glossaryLedgerTermKey,
    glossaryLedgerAdd,
    glossaryLedgerRenderings,
    glossaryStorageKey,
    glossaryFormat,
    () => ({ key: "host|series" }),
    async (cfg, text) => {
      glossarySetCalls.push(text);
      return { ok: true, count: glossaryParse(text).terms.length };
    }
  );
  return { ...lifted, session, writes, glossarySetCalls };
}

test("the harvest counts term-shaped regions and skips refusals and sentences", async () => {
  const { glossaryLedgerHarvest, session } = shippedLedgerBlock();
  await glossaryLedgerHarvest("host|series", [
    { source: "흑월검대", translated: "the Moon Blades" },
    { source: "흑월검대!", translated: "the moon squad" },
    { source: "긴 문장은 용어가 아니다。", translated: "a whole sentence" },
    { source: "거절", translated: "refused one", refused: "sub-floor" },
  ]);
  const ledger = session.get("ledger:host|series");
  assert.ok(ledger, "nothing was written");
  assert.deepEqual(Object.keys(ledger), ["흑월검대"]);
  assert.deepEqual(Object.keys(ledger["흑월검대"].r).sort(), [
    "the Moon Blades",
    "the moon squad",
  ]);
});

test("a harvest with nothing term-shaped writes nothing", async () => {
  const { glossaryLedgerHarvest, writes } = shippedLedgerBlock();
  await glossaryLedgerHarvest("host|series", [
    { source: "这是一整句话。", translated: "a sentence" },
  ]);
  assert.deepEqual(writes, []);
});

test("the reply's terms carry only what an alt-click may pin", () => {
  const { translateTerms } = shippedLedgerBlock();
  const terms = translateTerms([
    { x: 1, y: 2, width: 3, height: 4, source: "裂风之王！", translated: "GALE KING" },
    { x: 5, y: 6, width: 7, height: 8, source: "一整句话在这里。", translated: "a sentence" },
    { x: 9, y: 10, width: 11, height: 12, source: "拒绝", translated: "REFUSED", refused: "x" },
  ]);
  assert.equal(terms.length, 1);
  assert.equal(terms[0].source, "裂风之王！");
  assert.equal(translateTerms([]), null);
});

test("a pin replaces its own term and rides the popup's validation path", async () => {
  const { glossaryPin, session, glossarySetCalls } = shippedLedgerBlock();
  session.set(glossaryStorageKey("host|series"), [
    { source: "裂风之王", translation: "GALE DEMON" },
    { source: "흑월회", translation: "the Black Moon Society" },
  ]);
  const reply = await glossaryPin({}, "https://host/series/1", "裂风之王！", "Gale King");
  assert.equal(reply.ok, true);
  assert.equal(glossarySetCalls.length, 1);
  const { terms } = glossaryParse(glossarySetCalls[0]);
  assert.deepEqual(terms, [
    { source: "흑월회", translation: "the Black Moon Society" },
    { source: "裂风之王", translation: "Gale King" },
  ]);
});

test("a pin with nothing to say is refused before it touches the store", async () => {
  const { glossaryPin, glossarySetCalls } = shippedLedgerBlock();
  const reply = await glossaryPin({}, "https://host/series/1", "裂风之王", "   ");
  assert.equal(reply.ok, false);
  assert.deepEqual(glossarySetCalls, []);
});

/* --------------------------------------------------------- the 401 self-heal
 *
 * A stored token was never re-asked while non-empty, so a stale one stranded
 * the reader on a manual popup re-sync. The
 * install is the session's front door (it runs before BOTH authenticated
 * POSTs), so the heal lives there: one resync, one retry, never a loop. */

test("a stale token heals itself: 401, resync, one retry with the fresh token", async () => {
  const stored = [];
  let asked = 0;
  const h = installHarness(
    (() => {
      let calls = 0;
      return async () => {
        calls += 1;
        return calls === 1
          ? { ok: false, status: 401, json: async () => ({}) }
          : { ok: true, status: 200, json: async () => ({ terms: 1 }) };
      };
    })(),
    {
      browser: {
        runtime: {
          sendNativeMessage: async () => {
            asked += 1;
            return { ok: true, token: "fresh-token" };
          },
        },
        storage: { local: { set: async (record) => stored.push(record) } },
      },
    }
  );
  const cfg = { glossaryTerms: ONE_TERM, token: "stale-token" };
  await h.install(cfg, "story-1");
  assert.equal(h.posts.length, 2, "one retry, exactly");
  assert.equal(asked, 1, "the host is asked once");
  assert.equal(cfg.token, "fresh-token", "the live cfg carries the resync");
  assert.equal(
    h.posts[1].init.headers["X-Koharu-Token"],
    "fresh-token",
    "the retry authenticates with the fresh token"
  );
  assert.deepEqual(stored, [{ token: "fresh-token" }]);
});

test("a host answering with the same wrong token does not loop", async () => {
  const h = installHarness(
    async () => ({ ok: false, status: 401, json: async () => ({}) }),
    {
      browser: {
        runtime: { sendNativeMessage: async () => ({ ok: true, token: "stale-token" }) },
        storage: { local: { set: async () => {} } },
      },
    }
  );
  await assert.rejects(
    () => h.install({ glossaryTerms: ONE_TERM, token: "stale-token" }, "story-1"),
    /401/
  );
  assert.equal(h.posts.length, 1, "no retry without a different token");
});
