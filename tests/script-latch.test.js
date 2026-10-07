/* Unit tests for extension/script-latch.js -- the per-host language latch.
 *
 * Run:  node --test tests/script-latch.test.js
 *
 * NOT `node --test tests/`: on Node 24 a bare directory argument is
 * resolved as a MODULE and reports `Cannot find module '<repo>\tests'` while
 * never loading this file. See tests/seam.test.js's header.
 *
 * `latchDecision` is the COMPOSED PREDICATE the caller calls. `noteScript` in
 * background.js accumulates the counts and owns storage; this owns the whole
 * decision. Testing "does the threshold open" and "does it map to a language"
 * separately would leave both green with the join wrong -- which is precisely
 * how the Korean defect below survived, and how the FIRST attempt at fixing it
 * produced a worse bug than the one it fixed.
 */

const test = require("node:test");
const assert = require("node:assert");

const {
  latchDecision,
  SCRIPT_MIN_CHARS,
  SCRIPT_KANA_SHARE,
  BILINGUAL_OCR,
} = require("../extension/script-latch.js");

/* The real evidence, from a full-pipeline run over a Korean manhwa chapter and
 * reproduced on a second run: 59 pages of Korean dialogue carry 658 hangul,
 * 8 kana and 8 han. Both runs give the same three numbers, which is why they
 * are hard-coded here rather than sampled. */
const MANHWA_KO = { kana: 8, han: 8, hangul: 658 };

test("the measured Korean corpus latches, and latches as Korean", () => {
  const decided = latchDecision(MANHWA_KO);
  assert.equal(decided.ready, true, "658 hangul is not thin evidence");
  assert.equal(decided.language, "ko");
  assert.equal(decided.engine, BILINGUAL_OCR);
});

test("the OLD gate could never have opened on it", () => {
  // The regression this file exists for: the shipped gate was
  // `kana + han >= SCRIPT_MIN_CHARS`, and on real Korean that sum is 16.
  assert.ok(
    MANHWA_KO.kana + MANHWA_KO.han < SCRIPT_MIN_CHARS,
    "if this ever passes, the fixture stopped being the defect"
  );
  // ...so `languageByHost` was never written, `source_language` stayed "", the
  // server resolved SourceScript::Unknown, and labels.rs's script rule -- the
  // only gate that refuses a fabricated kana read on a non-Japanese page --
  // could not fire on any manhwa page.
});

test("hangul dominance beats the kana share, which the threshold fix alone got WRONG", () => {
  /* The first fix counted hangul toward the threshold and stopped there. The
   * gate opened and the answer came back manga-ocr / ja, because
   * share = 8/16 = 0.5 >= 0.35 and the 658 hangul were not in that denominator.
   * That is strictly worse than never latching: a Japanese-only engine on Korean
   * material, PLUS a `ja` declaration that makes the script rule structurally
   * unable to fire for a second, independent reason. */
  const share = MANHWA_KO.kana / (MANHWA_KO.kana + MANHWA_KO.han);
  assert.ok(share >= SCRIPT_KANA_SHARE, "the share really does look Japanese");
  assert.equal(latchDecision(MANHWA_KO).language, "ko", "and it must lose anyway");
});

test("a page of nothing but hangul does not divide by zero", () => {
  const decided = latchDecision({ kana: 0, han: 0, hangul: 100 });
  assert.equal(decided.ready, true);
  assert.equal(decided.language, "ko");
  assert.equal(decided.engine, BILINGUAL_OCR);
  assert.ok(Number.isFinite(decided.share), `share was ${decided.share}`);
});

test("Japanese and Chinese name their language exactly as they did before", () => {
  // Corpus-shaped: manga-ja is kana-dominant, manhua-zh measured 5 kana / 1114 han.
  // The ENGINE now also depends on the format, so this pins the LANGUAGE half --
  // which the format must never touch -- and the engine is pinned by the matrix
  // test above.
  const ja = latchDecision({ kana: 200, han: 100, hangul: 0 }, "paged");
  assert.equal(ja.engine, "manga-ocr", "Japanese AND paged still earns it");
  assert.equal(ja.language, "ja");

  const zh = latchDecision({ kana: 5, han: 1114, hangul: 0 }, "paged");
  assert.equal(zh.engine, BILINGUAL_OCR);
  assert.equal(zh.language, "zh");
});

test("a stray Korean sign on a Japanese page does not flip it", () => {
  /* The conservative half of the dominance test: `hangul > kana + han`, not
   * `hangul > han`. A Japanese page carrying a few hangul must stay Japanese, or
   * the fix trades a Korean bug for a Japanese one. */
  const decided = latchDecision({ kana: 100, han: 50, hangul: 10 }, "paged");
  assert.equal(decided.language, "ja");
  assert.equal(decided.engine, "manga-ocr");
});

test("thin evidence still decides nothing, in either script", () => {
  assert.equal(latchDecision({ kana: 5, han: 5, hangul: 0 }).ready, false);
  assert.equal(
    latchDecision({ kana: 0, han: 0, hangul: SCRIPT_MIN_CHARS - 1 }).ready,
    false,
    "the hangul path must respect the same floor, not bypass it"
  );
  assert.equal(latchDecision({}).ready, false, "an empty probe is not evidence");
  assert.equal(
    latchDecision({ kana: 0, han: 0 }).ready,
    false,
    "a probe written before `hangul` existed must not throw"
  );
});

test("the language mapping has ONE definition, and this is the case that proved it did not", () => {
  /* `noteScript` used to recompute the language itself with a different test --
   * `hangul > han` rather than this function's `hangul > kana + han` -- and the
   * value IT stored was the one that reached the server. The two disagree here,
   * so this fixture is the difference between the two definitions:
   *
   *   latchDecision      hangul 22 > kana+han 23 ?  no  -> share path -> "zh"
   *   the old duplicate  hangul 22 > han 20      ?  yes            -> "ko"
   *
   * Han outnumbers Hangul in the scripted total, so `zh` is the defensible
   * reading and `ko` was an artifact of comparing against `han` alone. If this
   * ever returns "ko" again, a second definition has grown back. */
  const decided = latchDecision({ kana: 3, han: 20, hangul: 22 });
  assert.equal(decided.ready, true);
  assert.equal(decided.language, "zh");
  assert.equal(decided.engine, BILINGUAL_OCR);
});

test("engine and language are returned together, so a caller cannot take one without the other", () => {
  /* The coupling that makes this dangerous: the mapping is
   * `engine === "manga-ocr" ? "ja" : ...`, so ANY caller that overrides the
   * engine and then re-derives the language silently re-declares a Japanese page
   * as `zh` -- and labels.rs then refuses its kana as wrong-script, 559 of 583
   * regions on the measured Japanese webtoon. The contract is
   * that both fields come from one call. */
  const paged = latchDecision({ kana: 2772, han: 741, hangul: 0 }, "paged");
  assert.equal(paged.engine, "manga-ocr");
  assert.equal(paged.language, "ja");

  /* The same evidence with only the FORMAT changed. The engine moves and the
   * language must not: that is the contract, and it is what makes it safe for a
   * caller to read both from one call. */
  const strip = latchDecision({ kana: 2772, han: 741, hangul: 0 }, "strip");
  assert.notEqual(strip.engine, paged.engine, "the format must move the engine");
  assert.equal(strip.language, paged.language, "...and must not move the language");
});

/* The measured evidence for each corpus, so the matrix below is driven by real
 * counters rather than invented ones. Sources: Japanese paged manga, a 90-page
 * Japanese webtoon, and the Chinese manhua and Korean manhwa chapters of a
 * full-pipeline baseline run. */
const MANGA_JA = { kana: 200, han: 100, hangul: 0 };
const WEBTOON_JA = { kana: 2772, han: 741, hangul: 0 };
const MANHUA_ZH = { kana: 5, han: 1114, hangul: 0 };

test("manga-ocr is EARNED: only Japanese AND paged gets it", () => {
  assert.equal(latchDecision(MANGA_JA, "paged").engine, "manga-ocr", "the one cell that keeps it");
  assert.equal(latchDecision(MANGA_JA, "strip").engine, BILINGUAL_OCR, "Japanese in a strip does not");
  assert.equal(latchDecision(WEBTOON_JA, "strip").engine, BILINGUAL_OCR, "the measured Japanese webtoon");
  assert.equal(latchDecision(MANHUA_ZH, "paged").engine, BILINGUAL_OCR, "Chinese never does, paged or not");
  assert.equal(latchDecision(MANHWA_KO, "paged").engine, BILINGUAL_OCR, "nor Korean");
});

test("an unknown format falls to the bilingual engine, not to manga-ocr", () => {
  /* The fallback direction IS the design. The format classifier's documented
   * false positive is a gutterless paged reader reading as a strip, and every
   * unclassifiable case must land on the engine whose error is ~2s a page rather
   * than the one whose error is silent fabrication. */
  for (const format of ["", undefined, null, "unknown", "STRIP", "Paged"]) {
    assert.equal(
      latchDecision(MANGA_JA, format).engine, BILINGUAL_OCR,
      `format ${JSON.stringify(format)} must not earn manga-ocr`
    );
  }
});

test("THE WRONG-SCRIPT TRAP: a Japanese webtoon still declares ja while using paddle", () => {
  /* The language must NOT be re-derived from the engine. If it were, forcing
   * paddle here would take the `han > 0` branch and declare `zh` -- and
   * labels.rs's script rule would then refuse this corpus's kana as wrong-script,
   * 559 of 583 regions. This assertion is the whole reason
   * language is computed before the engine and never from it. */
  const decided = latchDecision(WEBTOON_JA, "strip");
  assert.equal(decided.engine, BILINGUAL_OCR, "the engine moved");
  assert.equal(decided.language, "ja", "...and the language must NOT have moved with it");
});

test("the format does not disturb the language on any corpus", () => {
  for (const format of ["paged", "strip", ""]) {
    assert.equal(latchDecision(MANGA_JA, format).language, "ja");
    assert.equal(latchDecision(WEBTOON_JA, format).language, "ja");
    assert.equal(latchDecision(MANHUA_ZH, format).language, "zh");
    assert.equal(latchDecision(MANHWA_KO, format).language, "ko");
  }
});

test("the threshold is met exactly at the boundary, not one past it", () => {
  assert.equal(latchDecision({ kana: 0, han: 0, hangul: SCRIPT_MIN_CHARS }).ready, true);
  assert.equal(latchDecision({ kana: SCRIPT_MIN_CHARS, han: 0, hangul: 0 }).ready, true);
});

/* -------------------------------------------------------------- resolveSite */
/* `resolveSite` is the OTHER composed predicate this file guards: the
 * precedence between the reader's page-host picks and the latch's image-host
 * evidence, joined to the engine rule. `config()` in background.js calls it
 * once per request; testing "picks win" and "the engine rule" separately would
 * leave both green with the join wrong -- the same trap `latchDecision`'s
 * header describes, on a new function. */

const { resolveSite, engineFor, PROFILE_FROM_FORMAT } = require("../extension/script-latch.js");

const NO_PICKS = { profile: "", language: "" };
const NO_LATCH = { engine: "", language: "", format: "" };

test("no picks and no latch resolve to everything undeclared", () => {
  const site = resolveSite(NO_PICKS, NO_LATCH, true);
  assert.equal(site.language, "");
  assert.equal(site.profile, "");
  assert.equal(site.ocr, null, "nothing may touch the configured engine");
});

test("the latch passes through untouched when the reader picked nothing", () => {
  const site = resolveSite(
    NO_PICKS,
    { engine: BILINGUAL_OCR, language: "ko", format: "strip" },
    true
  );
  assert.equal(site.language, "ko");
  assert.equal(site.profile, "webtoon", "strip is the webtoon profile");
  assert.equal(site.ocr, BILINGUAL_OCR);
});

test("a Korean pick on a manga-ocr host forces the bilingual engine", () => {
  /* The one failure this precedence must not have: manga-ocr cannot represent
   * hangul and does not fail, it fabricates. A reader declaring Korean must
   * therefore move the engine in the same resolution, not on the next latch. */
  const site = resolveSite(
    { profile: "", language: "ko" },
    { engine: "manga-ocr", language: "ja", format: "paged" },
    true
  );
  assert.equal(site.language, "ko", "the pick beats the latch");
  assert.equal(site.ocr, BILINGUAL_OCR, "and the engine moves with it");
});

test("declaring manga on a strip-latched Japanese host unlocks manga-ocr", () => {
  /* The shape classifier's documented cheap error: a gutterless vertical
   * Japanese reader latches as a strip, so the host never earns manga-ocr.
   * The pick is how the reader corrects exactly that, per site. */
  const site = resolveSite(
    { profile: "manga", language: "" },
    { engine: BILINGUAL_OCR, language: "ja", format: "strip" },
    true
  );
  assert.equal(site.profile, "manga");
  assert.equal(site.language, "ja", "the latched language still counts");
  assert.equal(site.ocr, "manga-ocr");
});

test("declaring webtoon on a manga-ocr host breaks the paged half", () => {
  const site = resolveSite(
    { profile: "webtoon", language: "" },
    { engine: "manga-ocr", language: "ja", format: "paged" },
    true
  );
  assert.equal(site.ocr, BILINGUAL_OCR, "manga-ocr requires BOTH halves");
});

test("picks work before any latch exists", () => {
  const site = resolveSite({ profile: "manga", language: "ja" }, NO_LATCH, true);
  assert.equal(site.ocr, "manga-ocr", "a reader who declared both halves need not wait 40 chars");
});

test("ocrAuto off: picks still declare, but the engine is left alone", () => {
  /* An explicit engine choice must never be second-guessed -- by the latch OR
   * by a layout pick. The declaration still rides the request, because the
   * script gate and the seam read it regardless of who picked the engine. */
  const site = resolveSite(
    { profile: "manga", language: "ja" },
    { engine: BILINGUAL_OCR, language: "ko", format: "strip" },
    false
  );
  assert.equal(site.ocr, null, "the hand-picked engine stands");
  assert.equal(site.language, "ja");
  assert.equal(site.profile, "manga");
});

test("ocrAuto off with no picks declares nothing from the latch", () => {
  // The shipped behavior before this function existed: the whole latch block
  // was inside `if (cfg.ocrAuto)`, so turning the heuristic off silenced the
  // declaration too. Preserved, not accidental.
  const site = resolveSite(NO_PICKS, { engine: BILINGUAL_OCR, language: "ko", format: "strip" }, false);
  assert.equal(site.language, "");
  assert.equal(site.profile, "");
  assert.equal(site.ocr, null);
});

test("the format-to-profile mapping speaks only the two words the wire accepts", () => {
  assert.equal(PROFILE_FROM_FORMAT.strip, "webtoon");
  assert.equal(PROFILE_FROM_FORMAT.paged, "manga");
  assert.equal(PROFILE_FROM_FORMAT[""], undefined, "no format is no profile, never a guess");
});

test("engineFor and latchDecision agree on every language-format pair", () => {
  /* The rule exists twice on purpose (latchDecision decides from evidence,
   * engineFor from resolved declarations); this pins them together so a change
   * to one that misses the other fails here instead of shipping. */
  const evidence = { ja: MANGA_JA, zh: MANHUA_ZH, ko: MANHWA_KO };
  for (const [language, seen] of Object.entries(evidence)) {
    for (const format of ["paged", "strip", ""]) {
      const profile = PROFILE_FROM_FORMAT[format] || "";
      assert.equal(
        engineFor(language, profile),
        latchDecision(seen, format).engine,
        `${language} + ${format || "unknown"} must resolve the same engine both ways`
      );
    }
  }
});

/* ------------------------------------------------------------- the re-latch
 *
 * The latch used to decide once and stop looking (`if (latched[host]) return;`),
 * so a wrong early verdict poisoned every later series on the host forever --
 * observed live: a Japanese+paged latch put manga-ocr under a Chinese chapter,
 * and its Japanese styling was mistaken for a seam defect. Now a latched
 * host keeps a DISSENT probe: evidence still accumulates, and when it reaches
 * the same readiness floor the original latch needed, its verdict is compared
 * -- agreement CLEARS the probe (the hysteresis: a mixed stream keeps
 * resetting and never flips), contradiction REWRITES the latch and pays one
 * reload. Lifted from the SHIPPED background.js, because
 * the settle-vs-watch split is the caller's join, not the pure function's. */

const fs = require("node:fs");
const path = require("node:path");
const BACKGROUND_JS = fs.readFileSync(
  path.join(__dirname, "..", "extension", "background.js"),
  "utf8"
);

function shippedNoteScript(state) {
  const start = BACKGROUND_JS.indexOf("async function noteScript(");
  assert.ok(start > 0, "could not find noteScript() in background.js");
  const stop = BACKGROUND_JS.indexOf("/* `host` is the IMAGE's host");
  assert.ok(stop > start, "could not find the host comment that ends the lift");
  const source = BACKGROUND_JS.slice(start, BACKGROUND_JS.lastIndexOf("\n}", stop) + 2);

  const writes = [];
  const browser = {
    storage: {
      local: {
        get: async () => JSON.parse(JSON.stringify(state)),
        set: async (record) => {
          Object.assign(state, JSON.parse(JSON.stringify(record)));
          writes.push(Object.keys(record).sort().join(","));
        },
      },
    },
  };
  const build = new Function(
    "DEFAULTS",
    "browser",
    "latchDecision",
    "KANA",
    "HAN",
    "HANGUL",
    `${source}; return noteScript;`
  );
  const noteScript = build(
    { ocrAuto: true },
    browser,
    latchDecision,
    /[぀-ゟ゠-ヿ]/gu,
    /[一-鿿]/gu,
    /[가-힣ᄀ-ᇿ]/gu,
    );
  return { noteScript, state, writes };
}

const POISONED = () => ({
  ocrAuto: true,
  ocrByHost: { "img.host": "manga-ocr" },
  languageByHost: { "img.host": "ja" },
  formatByHost: { "img.host": "paged" },
});

const hanRegions = (chars) => [{ source: "魔".repeat(chars) }];
const kanaRegions = (chars) => [{ source: "あ".repeat(chars) }];

test("a poisoned ja+paged latch re-decides to zh once dissent reaches the floor", async () => {
  const { noteScript, state } = shippedNoteScript(POISONED());
  await noteScript("img.host", hanRegions(SCRIPT_MIN_CHARS - 10), "strip");
  assert.equal(state.ocrByHost["img.host"], "manga-ocr", "below the floor nothing moves");
  await noteScript("img.host", hanRegions(20), "strip");
  assert.equal(state.ocrByHost["img.host"], BILINGUAL_OCR, "the engine re-latched");
  assert.equal(state.languageByHost["img.host"], "zh", "the language re-latched");
  assert.equal(state.formatByHost["img.host"], "strip", "the format re-latched");
  assert.ok(!(state.ocrDissent || {})["img.host"], "the dissent probe cleared on the flip");
});

test("agreement at readiness clears the dissent probe and moves nothing", async () => {
  const { noteScript, state } = shippedNoteScript(POISONED());
  await noteScript("img.host", kanaRegions(SCRIPT_MIN_CHARS + 5), "paged");
  assert.equal(state.ocrByHost["img.host"], "manga-ocr", "an agreeing window flips nothing");
  assert.equal(state.languageByHost["img.host"], "ja");
  assert.ok(!(state.ocrDissent || {})["img.host"], "agreement resets the probe");
});

test("a mixed stream keeps resetting and never flips", async () => {
  const { noteScript, state } = shippedNoteScript(POISONED());
  for (let round = 0; round < 4; round += 1) {
    await noteScript("img.host", hanRegions(20), "strip");
    await noteScript("img.host", kanaRegions(SCRIPT_MIN_CHARS), "paged");
  }
  assert.equal(state.ocrByHost["img.host"], "manga-ocr", "the mixed host never flips");
  assert.equal(state.languageByHost["img.host"], "ja");
});

test("below the floor the dissent probe persists and the latch stands", async () => {
  const { noteScript, state } = shippedNoteScript(POISONED());
  await noteScript("img.host", hanRegions(10), "strip");
  assert.equal(state.ocrByHost["img.host"], "manga-ocr");
  assert.ok((state.ocrDissent || {})["img.host"], "the partial dissent evidence survives");
});

test("an unlatched host still takes the ordinary first-latch path", async () => {
  const { noteScript, state } = shippedNoteScript({ ocrAuto: true });
  await noteScript("fresh.host", hanRegions(SCRIPT_MIN_CHARS + 5), "strip");
  assert.equal(state.ocrByHost["fresh.host"], BILINGUAL_OCR);
  assert.equal(state.languageByHost["fresh.host"], "zh");
  assert.equal(state.formatByHost["fresh.host"], "strip");
});
