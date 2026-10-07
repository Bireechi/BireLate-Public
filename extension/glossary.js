/* The per-series glossary: parsing, validation and the cache fingerprint.
 *
 * A classic script, loaded into the BACKGROUND page (between story.js and
 * background.js) and into the CONTENT script (before seam.js)
 * for the ledger's pure geometry -- `glossaryPinHit` and the term-key filter
 * run at click time in the page. Storage still belongs to the background
 * alone: the popup and the pin dialog edit glossaries through messages,
 * because the story id the store is keyed by is derived from a page URL the
 * background already has (`story.js::storyIdFor`), and story.js is
 * background-only by design. Deliberately free of DOM, fetch and storage,
 * exactly like seam.js and story.js, so `node --test tests/glossary.test.js`
 * loads it directly through the `typeof module` guard and the shipped source is
 * what the suite tests.
 *
 * WHY THIS EXISTS. The server keeps per-series glossaries -- `POST /glossary`,
 * whole-glossary replace, keyed by the same story id every /translate sends --
 * and without a client that installs one, a reading session through the
 * extension always ran glossary-less. This module is the extension half: the
 * reader's terms for the series they are on, held in `browser.storage.session`
 * (WIPED when the browser closes, by design, mirroring the server store dying
 * with the server process), installed by the background before every
 * /translate.
 *
 * THE LIMITS MIRROR THE SERVER'S, and the server is the authority
 * (`server/src/glossary.rs`: MAX_TERMS 64, MAX_TERM_LEN 256 BYTES -- Rust
 * `len()` is bytes, so a Korean source spends three bytes a syllable). The
 * client validates the same bounds so a refusal happens in the editor with a
 * line number, not as a 400 after the reader closed the popup. Duplicate
 * sources are refused HERE and accepted by the server: two entries for one
 * source is a reader mistake with a silent winner, and refusing it client-side
 * cannot desync the wire because only parsed terms are ever sent.
 */

"use strict";

const GLOSSARY_MAX_TERMS = 64;
const GLOSSARY_MAX_TERM_BYTES = 256;

/* The editor format: one term per line, `source = translation`, split on the
 * FIRST `=` so the translation may contain one and the source may not (sources
 * are Korean/Japanese/Chinese; an `=` in one is a typo, not a term). Blank
 * lines and `#` comments pass through parsing and are lost on format() -- the
 * stored value is the TERMS, not the text, because the terms are what the
 * fingerprint and the wire are built from. */
function glossaryParse(text) {
  const terms = [];
  const errors = [];
  const seen = new Set();
  const bytes =
    typeof TextEncoder === "undefined"
      ? (value) => Buffer.byteLength(value, "utf8")
      : (value) => new TextEncoder().encode(value).length;
  const lines = typeof text === "string" ? text.split(/\r?\n/) : [];
  lines.forEach((line, index) => {
    const at = index + 1;
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) return;
    const eq = trimmed.indexOf("=");
    if (eq < 0) {
      errors.push(`line ${at}: no "=" -- write "source = translation"`);
      return;
    }
    const source = trimmed.slice(0, eq).trim();
    const translation = trimmed.slice(eq + 1).trim();
    if (!source || !translation) {
      errors.push(`line ${at}: empty ${source ? "translation" : "source"}`);
      return;
    }
    if (bytes(source) > GLOSSARY_MAX_TERM_BYTES || bytes(translation) > GLOSSARY_MAX_TERM_BYTES) {
      errors.push(`line ${at}: over ${GLOSSARY_MAX_TERM_BYTES} bytes (the server's cap)`);
      return;
    }
    if (seen.has(source)) {
      errors.push(`line ${at}: duplicate source "${source}"`);
      return;
    }
    seen.add(source);
    terms.push({ source, translation });
  });
  if (terms.length > GLOSSARY_MAX_TERMS) {
    errors.push(`${terms.length} terms; the server stores at most ${GLOSSARY_MAX_TERMS}`);
  }
  return { terms, errors };
}

function glossaryFormat(terms) {
  if (!Array.isArray(terms)) return "";
  return terms.map((term) => `${term.source} = ${term.translation}`).join("\n");
}

/* The cache-key component. Deterministic over the term list, ORDER INCLUDED --
 * the server feeds terms to the prompt in installation order, so a reordered
 * list is a different prompt and must be a different key.
 *
 * Empty string for no glossary, ON PURPOSE AND LOAD-BEARING:
 * `settingsFingerprint` appends this component only when it is non-empty, so a
 * glossary-less reader keys byte-identically to every entry written before
 * this feature existed -- the same "a store written before this existed must
 * key the same" rule every coercion in that function already follows.
 *
 * FNV-1a over UTF-16 code units, synchronous where crypto.subtle is not; a
 * collision's whole blast radius is one stale cached render inside the 24 h
 * window, which is what the term count beside it is for. */
function glossaryFingerprint(terms) {
  if (!Array.isArray(terms) || terms.length === 0) return "";
  let hash = 0x811c9dc5;
  const feed = (value) => {
    for (let i = 0; i < value.length; i += 1) {
      hash ^= value.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  };
  /* Control characters as field/record separators: nothing a reader can
   * type into the editor survives into a term carrying one, so two different
   * term lists cannot feed the hash one stream ("a" = "b c" against
   * "a b" = "c" was a REAL collision under a printable separator). */
  for (const term of terms) {
    feed(term.source);
    feed("\u0000");
    feed(term.translation);
    feed("\u0001");
  }
  return `g${terms.length}-${hash.toString(16).padStart(8, "0")}`;
}

/* The storage.session key for a series. Prefixed so the glossary namespace can
 * never collide with anything else that lands in session storage later. */
const glossaryStorageKey = (storyId) => `glossary:${storyId}`;

/* ---------------------------------------------------------------------------
 * The term ledger: what this session has SEEN, so a reader can pin
 * terms by choosing among renderings in their own language.
 *
 * The authoring problem it answers: a reader usually has no English
 * translation to compare against, so correcting terms from the source
 * language is difficult. A reader cannot say what a term
 * SHOULD be, but they can always say which of the renderings they have
 * already read they prefer -- so the ledger records, per series, every
 * term-shaped source string and every English rendering it has produced, and
 * the popup/pin surfaces offer those as choices. The source language never
 * reaches the reader's eyes as something to type.
 *
 * Held in storage.session beside the glossary store, same lifetime, same
 * wipe-at-browser-close rule. The ledger is observation, not configuration:
 * losing it costs suggestions, never terms.
 * ------------------------------------------------------------------------- */

const GLOSSARY_LEDGER_MAX_TERMS = 128;
const GLOSSARY_LEDGER_MAX_RENDERINGS = 6;
const GLOSSARY_LEDGER_MAX_SOURCE_CHARS = 24;

const glossaryLedgerStorageKey = (seriesKey) => `ledger:${seriesKey}`;

/* Edge punctuation a term sheds: quotes, brackets, exclamation and the CJK
 * middle-dot family survive INSIDE a term (`灵光识·苍炎之王`) but not at its
 * edges (`裂风之王！` is the term `裂风之王`). Full stops are deliberately NOT
 * in this class: a shouted name ends in `！`, a sentence ends in `。`, and
 * stripping the latter here would launder an unspaced Chinese sentence into a
 * "term" -- the sentence test below must still see it. */
const GLOSSARY_LEDGER_EDGE = /^[\s!?~、,:;·！？〜「」『』()（）\[\]【】"'“”-]+|[\s!?~、,:;·！？〜「」『』()（）\[\]【】"'“”-]+$/g;
/* Sentence punctuation ANYWHERE inside disqualifies: a glossary pins terms,
 * and pinning a sentence would ask the model to reproduce it verbatim. */
const GLOSSARY_LEDGER_SENTENCE = /[。．.?!？！…‥]/;

/* A region source string reduced to a ledger term key, or "" for a source
 * that is not term-shaped: too long, sentence-punctuated, or multi-phrase.
 * The 24-char cap is generous for the class this exists for -- skill names,
 * proper nouns, ranks -- while refusing dialogue. Space-separated Korean is
 * allowed two tokens (`흑월 검`-style spacing), not more. */
function glossaryLedgerTermKey(source) {
  if (typeof source !== "string") return "";
  const stripped = source.replace(GLOSSARY_LEDGER_EDGE, "").replace(/\s+/g, " ");
  if (!stripped || stripped.length > GLOSSARY_LEDGER_MAX_SOURCE_CHARS) return "";
  if (GLOSSARY_LEDGER_SENTENCE.test(stripped)) return "";
  if (stripped.split(" ").length > 2) return "";
  return stripped;
}

/* One observation into the ledger. Mutates and returns `ledger`, a plain
 * object `{ [term]: { r: { [rendering]: count }, at } }` -- plain because it
 * round-trips storage.session as JSON. `now` is passed in, never read from a
 * clock here, so node can test eviction deterministically. Caps: renderings
 * per term evict the lowest count (never the one just observed); terms per
 * series evict the oldest-touched. */
function glossaryLedgerAdd(ledger, source, translated, now) {
  const term = glossaryLedgerTermKey(source);
  if (!term) return ledger;
  const rendering =
    typeof translated === "string" ? translated.trim().replace(/\s+/g, " ") : "";
  if (!rendering || rendering.length > GLOSSARY_LEDGER_MAX_SOURCE_CHARS * 4) return ledger;
  const entry = ledger[term] || (ledger[term] = { r: {}, at: 0 });
  entry.at = now;
  entry.r[rendering] = (entry.r[rendering] || 0) + 1;
  const renderings = Object.keys(entry.r);
  if (renderings.length > GLOSSARY_LEDGER_MAX_RENDERINGS) {
    let worst = null;
    for (const text of renderings) {
      if (text === rendering) continue;
      if (!worst || entry.r[text] < entry.r[worst]) worst = text;
    }
    if (worst) delete entry.r[worst];
  }
  const terms = Object.keys(ledger);
  if (terms.length > GLOSSARY_LEDGER_MAX_TERMS) {
    let oldest = null;
    for (const key of terms) {
      if (key === term) continue;
      if (!oldest || ledger[key].at < ledger[oldest].at) oldest = key;
    }
    if (oldest) delete ledger[oldest];
  }
  return ledger;
}

/* The renderings the ledger holds for one source term, most-seen first --
 * what the pin dialog offers as choices. */
function glossaryLedgerRenderings(ledger, source) {
  const term = glossaryLedgerTermKey(source);
  const entry = term && ledger && ledger[term];
  if (!entry || !entry.r) return [];
  return Object.keys(entry.r)
    .map((text) => ({ text, count: entry.r[text] }))
    .sort((a, b) => b.count - a.count || a.text.localeCompare(b.text));
}

/* The popup's suggestion list: terms that DRIFTED -- two or more distinct
 * renderings -- and are not already pinned, busiest first. Drift is the
 * signal because a term rendered one way every time has nothing to ask the
 * reader about. */
function glossaryLedgerSuggestions(ledger, glossaryTerms) {
  const pinned = new Set(
    (Array.isArray(glossaryTerms) ? glossaryTerms : []).map((term) =>
      glossaryLedgerTermKey(term.source) || term.source
    )
  );
  const items = [];
  for (const term of Object.keys(ledger || {})) {
    if (pinned.has(term)) continue;
    const renderings = glossaryLedgerRenderings(ledger, term);
    if (renderings.length < 2) continue;
    items.push({
      source: term,
      renderings,
      total: renderings.reduce((sum, one) => sum + one.count, 0),
    });
  }
  return items.sort((a, b) => b.total - a.total || a.source.localeCompare(b.source));
}

/* Hit-test a click against the reply's term boxes, in the image's natural
 * pixel space. Smallest containing box wins, because a display column can sit
 * inside a larger refused region and the click means the visible term. `pad`
 * forgives the fingertip. */
function glossaryPinHit(terms, x, y, pad) {
  const reach = typeof pad === "number" ? pad : 8;
  let hit = null;
  for (const term of Array.isArray(terms) ? terms : []) {
    if (
      x >= term.x - reach &&
      x <= term.x + term.width + reach &&
      y >= term.y - reach &&
      y <= term.y + term.height + reach
    ) {
      if (!hit || term.width * term.height < hit.width * hit.height) hit = term;
    }
  }
  return hit;
}

if (typeof module !== "undefined") {
  module.exports = {
    GLOSSARY_MAX_TERMS,
    GLOSSARY_MAX_TERM_BYTES,
    glossaryParse,
    glossaryFormat,
    glossaryFingerprint,
    glossaryStorageKey,
    GLOSSARY_LEDGER_MAX_TERMS,
    GLOSSARY_LEDGER_MAX_RENDERINGS,
    GLOSSARY_LEDGER_MAX_SOURCE_CHARS,
    glossaryLedgerStorageKey,
    glossaryLedgerTermKey,
    glossaryLedgerAdd,
    glossaryLedgerRenderings,
    glossaryLedgerSuggestions,
    glossaryPinHit,
  };
}
