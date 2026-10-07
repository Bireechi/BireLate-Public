/* Unit tests for the cache key's host dependency.
 *
 * Run:  node --test tests/cache-key.test.js
 *
 * (Not `node --test tests/` -- see the header of seam.test.js for why a bare
 * directory argument reports a green suite failing for an unrelated reason.)
 *
 * WHY THIS FILE EXISTS. `settingsFingerprint` reads two fields that only exist
 * once the per-host OCR latch has been resolved: `cfg.ocr`, which `config(host)`
 * overwrites from `ocrByHost`, and `cfg.sourceLanguage`, which `config(host)`
 * is the ONLY writer of anywhere in the extension. A reader that keys without a
 * host therefore computes a different fingerprint from the writer, for ever,
 * on every host the latch has fired on -- and because `cacheGetByUrl` queries
 * the `urls` index with `IDBKeyRange.only`, there is no near-miss and no
 * fallback. It is a total miss with no symptom.
 *
 * That is not hypothetical. `cacheLookup` and `seamJoin` both shipped calling
 * `config()` with no argument. Tier 1 was dead on every latched host, and the
 * webtoon seam was worse than dead: its `skipped: "uncached"` arm returns
 * BEFORE the POST, `performSeam` reads that `ok` reply as the boundary answered
 * and writes SEAM_SETTLED, so every boundary settled as answered-nothing after
 * paying two image re-fetches and two multi-MB marshals per join.
 *
 * Both halves below guard that. The first pins the behaviour of the shipped
 * function; the second is a static check over background.js, because the defect
 * was never in the fingerprint -- it was in who called it.
 *
 * cache.js is a classic browser script with no export guard, so the two pure
 * functions are lifted out of its source text and evaluated in a sandbox rather
 * than required. That deliberately tests the SHIPPED source and not a copy.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const CACHE_JS = fs.readFileSync(path.join(ROOT, "extension", "cache.js"), "utf8");
const BACKGROUND_JS = fs.readFileSync(
  path.join(ROOT, "extension", "background.js"),
  "utf8"
);

function shippedCacheFns() {
  const fp = CACHE_JS.slice(
    CACHE_JS.indexOf("function settingsFingerprint"),
    CACHE_JS.indexOf("const cacheKey =")
  );
  const uk = CACHE_JS.slice(
    CACHE_JS.indexOf("function cacheUrlKey"),
    CACHE_JS.indexOf("/* Hashed here")
  );
  assert.ok(fp.length > 0 && uk.length > 0, "could not lift the functions out of cache.js");
  const max = /CACHE_MAX_URL\s*=\s*(\d+)/.exec(CACHE_JS);
  return new Function(
    "CACHE_MAX_URL",
    `${fp}${uk}; return { settingsFingerprint, cacheUrlKey };`
  )(max ? Number(max[1]) : 2048);
}

const BASE = {
  targetLanguage: "en-US",
  ocr: "manga-ocr",
  inpainting: "lama",
  provider: "local",
  llm: "",
  keepArt: false,
  segmentContext: false,
  storyId: "S1",
};

test("the segment-context toggle keys differently", () => {
  const { settingsFingerprint } = shippedCacheFns();
  /* It changes the PROMPT, so it changes the words and therefore the pixels --
   * measured on one page and one binary, an ability name rendered two
   * different ways with it on and with it off. A fingerprint blind to it would
   * serve the pre-flip render for 24 hours and the toggle would look broken. */
  assert.notStrictEqual(
    settingsFingerprint({ ...BASE, segmentContext: true }),
    settingsFingerprint(BASE),
    "flipping the toggle must not reuse the other arm's cached translation",
  );
});

test("a store written before the segment-context toggle keys as off", () => {
  const { settingsFingerprint } = shippedCacheFns();
  /* `undefined` from an older profile must land on the same key as an explicit
   * false, or every reader upgrading loses their whole cache on first launch --
   * the coercion `Boolean(cfg.segmentContext)` is what buys that, and this is
   * what stops someone removing it as noise. */
  const older = { ...BASE };
  delete older.segmentContext;
  assert.strictEqual(
    settingsFingerprint(older),
    settingsFingerprint({ ...BASE, segmentContext: false }),
    "an absent setting must key identically to an explicit off",
  );
});

test("an unresolved host keys differently from a latched one", () => {
  const { settingsFingerprint } = shippedCacheFns();

  // `config()` with no host never assigns sourceLanguage at all.
  const unresolved = settingsFingerprint({ ...BASE });
  // `config(host)` after the latch fires on a Japanese host: same engine, but a
  // declared language. The engine matching is what makes this easy to miss.
  const japanese = settingsFingerprint({ ...BASE, sourceLanguage: "ja" });
  // ...and on a Chinese one, where the engine moves too.
  const chinese = settingsFingerprint({
    ...BASE,
    ocr: "paddleocr-vl-1.6",
    sourceLanguage: "zh",
  });

  assert.notEqual(unresolved, japanese);
  assert.notEqual(unresolved, chinese);
  assert.notEqual(japanese, chinese);
});

test("a pre-latch entry keys identically to an unresolved lookup", () => {
  const { settingsFingerprint, cacheUrlKey } = shippedCacheFns();
  // This is the sharp edge, and it is why the bug served STALE renders rather
  // than merely missing: pages translated before the latch fired key exactly as
  // a host-less lookup does, so those entries kept being served for the whole
  // 24 h window while every later page missed.
  const prelatch = settingsFingerprint({ ...BASE, sourceLanguage: "" });
  const unresolved = settingsFingerprint({ ...BASE });
  assert.equal(prelatch, unresolved);

  const url = "https://example.test/001.jpg";
  assert.equal(cacheUrlKey(url, prelatch), cacheUrlKey(url, unresolved));
  assert.notEqual(
    cacheUrlKey(url, unresolved),
    cacheUrlKey(url, settingsFingerprint({ ...BASE, sourceLanguage: "ja" }))
  );
});

test("every settingsFingerprint caller in background.js resolves a host first", () => {
  const lines = BACKGROUND_JS.split(/\r?\n/);
  const offenders = [];

  lines.forEach((line, index) => {
    if (!/settingsFingerprint\s*\(/.test(line)) return;
    // Comments referring to the function are not call sites.
    const code = line.replace(/^\s*(\/\/|\*|\/\*).*$/, "");
    if (!/settingsFingerprint\s*\(/.test(code)) return;

    // Walk back to the nearest `config(...)` that produced the object. Twelve
    // lines is generous: in every real call site it is the line above.
    let resolved = null;
    for (let back = index; back >= Math.max(0, index - 12); back -= 1) {
      const m = /\bconfig\s*\(([^)]*)\)/.exec(lines[back]);
      if (m) {
        resolved = m[1].trim();
        break;
      }
    }
    if (resolved === null || resolved === "") {
      offenders.push(`${index + 1}: ${line.trim()}`);
    }
  });

  assert.deepEqual(
    offenders,
    [],
    "these call settingsFingerprint on a config() resolved without a host, so " +
      "they key under the unlatched ocr/sourceLanguage and can never match what " +
      "translate() stored:\n  " + offenders.join("\n  ")
  );
});

/* ------------------------------------------- which edges a join actually fixed
 *
 * `cacheSeamedWith` is the multi-edge branch, and it is the only thing standing
 * between a run and an unbounded re-join on every revisit: a slice a run passes
 * THROUGH was cut at its top and at its bottom, and one join repaired both. It
 * is invisible to every other test -- `cacheMarkSeamed` itself is a transaction
 * over IndexedDB -- so it is lifted out of the shipped source in the same way
 * the fingerprint is above, and not copied here.
 */

function shippedSeamedWith() {
  const src = CACHE_JS.slice(
    CACHE_JS.indexOf("function cacheSeamedWith"),
    CACHE_JS.indexOf("async function cacheMarkSeamed")
  );
  assert.ok(src.length > 0, "could not lift cacheSeamedWith out of cache.js");
  return new Function(`${src}; return cacheSeamedWith;`)();
}

test("one edge or several, and a run's middle slice needs several", () => {
  const seamedWith = shippedSeamedWith();
  // The pairwise case, unchanged: a string.
  assert.deepEqual(seamedWith([], "bottom"), ["bottom"]);
  assert.deepEqual(seamedWith(["bottom"], "top"), ["bottom", "top"]);
  // The run case: both edges of one slice, in one write.
  assert.deepEqual(seamedWith([], ["top", "bottom"]), ["top", "bottom"]);
});

test("marking is idempotent, so a re-join cannot grow the list", () => {
  const seamedWith = shippedSeamedWith();
  assert.deepEqual(seamedWith(["top"], "top"), ["top"]);
  assert.deepEqual(seamedWith(["top", "bottom"], ["top", "bottom"]), ["top", "bottom"]);
  assert.deepEqual(seamedWith(["bottom"], ["top", "bottom"]), ["bottom", "top"]);
});

test("a record with no seamed list at all is not corrupted by a mark", () => {
  const seamedWith = shippedSeamedWith();
  // Entries written before the field existed carry undefined, and an entry that
  // survived a bad write could carry anything.
  assert.deepEqual(seamedWith(undefined, "top"), ["top"]);
  assert.deepEqual(seamedWith(null, ["top", "bottom"]), ["top", "bottom"]);
  assert.deepEqual(seamedWith("nonsense", "top"), ["top"]);
});

test("the caller is never handed back the array it passed in", () => {
  const seamedWith = shippedSeamedWith();
  // The entry's own list is read, edited and put in one transaction; mutating it
  // in place would make the write depend on whether the read was a live object.
  const before = ["bottom"];
  const after = seamedWith(before, "top");
  assert.notEqual(after, before);
  assert.deepEqual(before, ["bottom"]);
});
