/* The script latch's decision, as a pure function of accumulated evidence.
 *
 * Loaded as a background script before `background.js`, which owns the
 * accumulation (`noteScript`), the storage and the host key. This file owns only
 * the arithmetic, so a test can drive it with no `browser.*` and no profile --
 * the same split, and the same `typeof module` foot, that `seam.js` uses.
 *
 * # Why it is a separate function at all
 *
 * It is the composed predicate the caller calls. A test of the threshold and a
 * test of the language mapping, written separately, would BOTH stay green with
 * the two wired together wrongly -- which is exactly how the defect below
 * survived: the threshold and the language decision each looked right on their
 * own, and the bug was in the join between them.
 */

/* CJK characters to see before deciding anything. A title page carrying one word
 * must never pick the engine for a chapter, and the failure mode this guards is
 * measured rather than imagined: on the Chinese chapter, manga-ocr hallucinated
 * kana into 2 of the 8 regions it actually read, so a sparse page can score
 * 1.000 on its own while the accumulated window scores 0.066. */
const SCRIPT_MIN_CHARS = 40;

/* At or above this share of kana, the material is Japanese and `manga-ocr`
 * stays. Below it, switch.
 *
 * **The threshold is a safety margin, not an accuracy optimum, because the two
 * errors are not symmetric.** Guessing PaddleOCR-VL wrongly costs ~2s a page,
 * is loud (the reader watches pages slow down), reversible, and costs zero
 * measured accuracy -- 0 of 66 kanji disagreements against manga-ocr on real
 * Japanese pages. Guessing manga-ocr wrongly costs silent fabrication. So this
 * is set to require positive evidence of kana to KEEP manga-ocr, and never to
 * require evidence of Chinese to leave it. Every close call must fall to
 * PaddleOCR-VL. Measured margins either side: Japanese 0.716, Chinese 0.020,
 * and a *wrong* manga-ocr read of Chinese still only 0.066. */
const SCRIPT_KANA_SHARE = 0.35;

/** The engine that reads both scripts, for when the evidence says not Japanese.
 * `hunyuan-ocr-1.5`, which took this role over from `paddleocr-vl-1.6`; paddle
 * remains selectable as the reserve. */
const BILINGUAL_OCR = "hunyuan-ocr-1.5";

/* Decide, from accumulated character counts, whether there is enough evidence and
 * what it says.
 *
 * `seen` is `{ kana, han, hangul }`; every field is optional because a probe
 * written by an older version has no `hangul`.
 *
 * # HANGUL COUNTS TOWARD THE THRESHOLD. THAT IS THE BUG FIX.
 *
 * This gated on `kana + han` alone, and Korean dialogue is Hangul -- it carries
 * almost no kana and almost no han. So **the gate could never open on a Korean
 * host.** Measured over a 59-page Korean test chapter, twice: its pages carry
 * **658 hangul** against a kana+han total of **16**, needing 40. The evidence
 * that decides `ko` -- `hangul > han`, below -- was never counted toward the
 * threshold gating the line that reads it, so `languageByHost` was never
 * written, `source_language` stayed `""`, the server resolved
 * `SourceScript::Unknown`, and `labels.rs`'s script rule never fired on manhwa
 * on any page. That rule is the only thing that refuses a fabricated kana read
 * on a non-Japanese page: on a Chinese test page it is exactly what separates
 * an arm that letters fabricated sound effects across falling petals from one
 * that letters nothing. **49 sound-effect regions on the Korean test chapter
 * were exposed by it.**
 *
 * # THE KANA SHARE STILL EXCLUDES HANGUL, DELIBERATELY
 *
 * `share` answers "kana or han?", which is the ENGINE question, and Hangul is
 * not an answer to it -- PaddleOCR-VL reads Hangul and Han alike. Folding hangul
 * into that denominator would drag the share down on any page carrying stray
 * Hangul and could flip a genuinely Japanese page to the bilingual engine. So
 * the threshold counts three scripts and the share counts two: they are
 * different questions, and this is the line where that stops being implicit.
 *
 * The `scripted > 0` guard is load-bearing rather than defensive. Without it a
 * pure-Hangul page divides by zero, `share` is `NaN`, `NaN >= 0.35` is `false`,
 * and PaddleOCR-VL is chosen -- the right engine for entirely the wrong reason,
 * which is the kind of accident that survives until the day it inverts.
 *
 * # HANGUL DOMINANCE IS TESTED BEFORE THE SHARE, AND FIXING ONLY THE THRESHOLD
 * # PRODUCED A CONFIDENTLY WRONG ANSWER
 *
 * The first version of this fix counted hangul toward the threshold and left the
 * rest alone. On the real manhwa evidence -- `{kana: 8, han: 8, hangul: 658}` --
 * the gate then opened and the decision came back **`manga-ocr` / `ja`**, because
 * `share = 8 / 16 = 0.5` is above `SCRIPT_KANA_SHARE` and the 658 Hangul
 * characters were not in that denominator. A latch that never fired had become a
 * latch that fired and said Japanese, which is strictly worse: it would put a
 * Japanese-only engine on Korean material *and* declare `ja`, so the script rule
 * would then be structurally unable to fire for a second, independent reason.
 *
 * So the dominant script decides first. `hangul > scripted` means Hangul
 * outnumbers kana and han **combined** -- a deliberately conservative test that
 * cannot fire on Japanese (which carries no Hangul) and cannot fire on a
 * Japanese page carrying a stray Korean sign. Only when Hangul does not dominate
 * does the kana-vs-han share get asked, unchanged, so the ja/zh decision is
 * exactly the one that shipped.
 */
function latchDecision(seen, format) {
  const kana = (seen && seen.kana) || 0;
  const han = (seen && seen.han) || 0;
  const hangul = (seen && seen.hangul) || 0;
  const scripted = kana + han;
  const total = scripted + hangul;
  if (total < SCRIPT_MIN_CHARS) {
    return { ready: false, engine: null, language: null, format: null, total, share: null };
  }
  const share = scripted > 0 ? kana / scripted : 0;

  /* LANGUAGE FIRST, FROM THE SCRIPT ALONE. It used to be read off the engine --
   * `engine === "manga-ocr" ? "ja" : ...` -- and that coupling is a trap with a
   * measured price: anything that moves the engine away from manga-ocr then
   * silently re-declares a Japanese page as `zh`, and `labels.rs`'s script rule
   * refuses its kana as wrong-script. On a measured Japanese webtoon that is
   * **559 of 583 regions**. Since the engine now depends on the format, keeping
   * that order would have made every Japanese webtoon declare Chinese. The two
   * are decided separately and in this order, permanently. */
  const language =
    hangul > scripted ? "ko" : share >= SCRIPT_KANA_SHARE ? "ja" : han > 0 ? "zh" : "";

  /* ENGINE SECOND, AND `manga-ocr` IS EARNED RATHER THAN ASSUMED.
   *
   * It is chosen only on POSITIVE evidence of both halves: the script says
   * Japanese and the page shape says discrete pages. Everything else -- Chinese,
   * Korean, undeclared, a Japanese *webtoon*, or a format nobody could classify
   * -- falls to the bilingual engine.
   *
   * The asymmetry that decides the fallback direction is the one this file's
   * `SCRIPT_KANA_SHARE` comment already sets out: guessing PaddleOCR-VL wrongly
   * costs ~2 s a page, is loud, reversible, and measured at zero accuracy cost
   * (0 of 66 kanji disagreements on real Japanese pages, and a blind panel on a
   * Japanese webtoon found no separation on dialogue, 15-9, p = 0.31). Guessing
   * manga-ocr wrongly costs SILENT FABRICATION: on 10 boxes two blind seats
   * called pure artwork, manga-ocr lettered 8 and paddle 4, because manga-ocr
   * answers unreadable ink with fluent kana that nothing refuses on a Japanese
   * page while paddle answers with a symbol the punctuation gate already
   * catches.
   *
   * So an unknown format resolves to the bilingual engine, and that is the
   * feature rather than a gap: the format classifier's documented false positive
   * is a gutterless vertical-scroll manga reader reading as a strip
   * (`content.js`, `seam.js` both say so in writing), and that error lands on
   * the cheap side by construction. */
  const paged = format === "paged";
  const engine = language === "ja" && paged ? "manga-ocr" : BILINGUAL_OCR;
  return { ready: true, engine, language, format: format || "", total, share };
}

/* The latch's format verdict, as the profile axis the rest of the extension
 * speaks. FORMAT ONLY, never language: a Japanese webtoon test chapter has
 * exactly the Korean test chapter's 690px width, so the one thing a page's
 * shape may never imply is its script. An unknown format is an
 * undeclared profile, not a guessed one. */
const PROFILE_FROM_FORMAT = { strip: "webtoon", paged: "manga" };

/* The engine rule, alone, so `resolveSite` below and `latchDecision` above
 * cannot drift: `manga-ocr` is earned by positive evidence of BOTH halves
 * (Japanese, and discrete pages), everything else reads with the bilingual
 * engine. Same asymmetry as `SCRIPT_KANA_SHARE` documents -- the bilingual
 * engine wrongly costs seconds, `manga-ocr` wrongly costs silent fabrication. */
function engineFor(language, profile) {
  return language === "ja" && profile === "manga" ? "manga-ocr" : BILINGUAL_OCR;
}

/* What one request should carry for a site, as a pure function of the reader's
 * explicit picks and the latch's evidence.
 *
 * This is the composed predicate `config()` calls, extracted for the same
 * reason `latchDecision` was: the precedence (a pick beats the latch) and the
 * engine re-derivation are a join a test of either half would miss.
 *
 * - `picks` are PAGE-host keyed (the host the reader sees and the popup can
 *   name). They are deliberately not image-host keyed: one CDN serves many
 *   sites, and an explicit pick for one site must never apply to another.
 * - `latch` is IMAGE-host keyed, unchanged (`noteScript`'s evidence).
 * - Any explicit pick re-derives the engine from the RESOLVED pair via
 *   `engineFor`, so a reader declaring "Japanese + Manga" on a gutterless
 *   vertical reader (the shape classifier's documented cheap error) unlocks
 *   `manga-ocr`, and a reader declaring "Korean" on a host latched to
 *   `manga-ocr` gets the bilingual engine -- `manga-ocr` cannot represent
 *   hangul and does not fail, it fabricates.
 * - `ocr: null` means "leave the configured engine alone": no picks and no
 *   latch, or the reader turned the heuristic off without picking here.
 */
function resolveSite(picks, latch, ocrAuto) {
  const p = picks || {};
  const l = latch || {};
  const language = p.language || (ocrAuto ? l.language || "" : "");
  const profile = p.profile || (ocrAuto ? PROFILE_FROM_FORMAT[l.format] || "" : "");
  let ocr = null;
  if (ocrAuto) {
    if (p.language || p.profile) ocr = engineFor(language, profile);
    else if (l.engine) ocr = l.engine;
  }
  return { language, profile, ocr };
}

/* Same foot as `seam.js`: this stays a classic script the extension loads with a
 * plain entry in `manifest.json`, and the guard is what lets a test require it
 * without the extension having to become a module. */
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    latchDecision,
    engineFor,
    resolveSite,
    PROFILE_FROM_FORMAT,
    SCRIPT_MIN_CHARS,
    SCRIPT_KANA_SHARE,
    BILINGUAL_OCR,
  };
}
