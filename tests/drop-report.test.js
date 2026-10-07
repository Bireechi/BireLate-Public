/* Unit tests for the erased-and-unlettered report, browser side.
 *
 * Run:  node --test tests/drop-report.test.js
 *
 * (Not `node --test tests/` -- see the header of seam.test.js for why a bare
 * directory argument reports a green suite failing for an unrelated reason.)
 *
 * WHY THIS FILE EXISTS. The server now reports `dropped`: regions it read,
 * erased, and lettered with NOTHING. Measured over stored test runs, 59 regions
 * on 22 pages, and on every one of those pages `untranslated` was empty, `truncated`
 * was false and both id counters were zero -- so this is the only channel that
 * can say anything at all about them.
 *
 * That makes the browser half of it the whole point. `x-birelate-untranslated`
 * once shipped with the server emitting it and the extension discarding it:
 * the server did all its work and the browser threw the answer away. This file
 * is a guard against the same shape: a field on the wire that nothing reads.
 *
 * `report` and `missNotes` are lifted out of the shipped source text and
 * evaluated in a sandbox rather than required, exactly as cache-key.test.js
 * lifts the two pure cache functions -- both files are classic browser scripts
 * with no export guard, and this deliberately tests the SHIPPED source and not a
 * copy. `missNotes` exists as a separate function from `noteMisses` for exactly
 * this reason: `noteMisses` ends in `flashError`, which is DOM, so the wording
 * could not otherwise be read at all. It shipped once as
 * `if (false && reply.dropped)` -- an empty warning bar, suite still green.
 *
 * The cache test is a static check over the shipped files, because the failure
 * it guards is never in the arithmetic; it is in a value that stops being passed
 * along one hop before the reader.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const read = (name) => fs.readFileSync(path.join(ROOT, "extension", name), "utf8");
const BACKGROUND_JS = read("background.js");
const CACHE_JS = read("cache.js");
const CONTENT_JS = read("content.js");

/* The SHIPPED seam module, required rather than lifted -- it has a proper export
 * guard, unlike the three classic scripts above. It is here so the edge-hint test
 * at the foot of this file can drive the two halves as ONE path: what `report()`
 * returns, handed to the function `translate()` hands it to. */
const seam = require("../extension/seam.js");

function shippedReport() {
  const source = BACKGROUND_JS.slice(
    BACKGROUND_JS.indexOf("function report(payload)"),
    BACKGROUND_JS.indexOf("async function translate(")
  );
  assert.ok(source.length > 0, "could not lift report() out of background.js");
  return new Function(`${source}; return report;`)();
}

/* The other half of the same trick, on the other file. `missNotes` is pure --
 * it takes the reply and returns the sentence -- so it evaluates in a sandbox
 * with no DOM at all. */
function shippedMissNotes() {
  const start = CONTENT_JS.indexOf("function missNotes(reply)");
  assert.ok(start > 0, "could not find missNotes() in content.js");
  const source = CONTENT_JS.slice(start, CONTENT_JS.indexOf("function noteSlice("));
  return new Function(`${source}; return missNotes;`)();
}

/* A measured page, verbatim off a stored run: seven regions, four of them
 * erased and painted with nothing, and every other counter clean. */
const DROPPED_PAGE = {
  regions: [{}, {}, {}, {}, {}, {}, {}],
  untranslated: [],
  dropped: [3, 4, 5, 6],
  truncated: false,
  duplicate_ids: 0,
  out_of_range_ids: 0,
  stages_selected: ["detection", "ocr", "translation", "inpainting"],
};

test("a page with erased and unlettered boxes says so", () => {
  const report = shippedReport();
  const out = report(DROPPED_PAGE);
  assert.equal(out.dropped, "4/7");
  // And it is genuinely the only thing that fired. If `missed` or `cut` could
  // carry this page there would be no reason for the field to exist.
  assert.equal(out.missed, null);
  assert.equal(out.cut, false);
  assert.equal(out.duplicateIds, 0);
  assert.equal(out.outOfRangeIds, 0);
});

test("an ordinary page stays silent", () => {
  const report = shippedReport();
  // Every region translated. This is the population the rule must not touch:
  // 4,630 of the 4,689 current-code Japanese regions measured look like this.
  const clean = report({ ...DROPPED_PAGE, dropped: [] });
  assert.equal(clean.dropped, null);

  /* A page that is untranslated rather than erased must ALSO stay silent here,
   * and that is the sharp edge. The pipeline writes an unanswered id back
   * holding its SOURCE text, so a segment the model skipped renders as Japanese
   * -- visible, and already reported by `missed`. Announcing it as an erased
   * blank would send the reader looking for a hole that is not there. */
  const untranslated = report({
    ...DROPPED_PAGE,
    untranslated: [0, 1],
    dropped: [],
  });
  assert.equal(untranslated.dropped, null);
  assert.equal(untranslated.missed, "2/7");
});

test("a server too old to report drops is not read as a page full of them", () => {
  const report = shippedReport();
  const { dropped, ...old } = DROPPED_PAGE;
  assert.equal(report(old).dropped, null);
  // A server answering with the wrong type must fall the same way rather than
  // throwing inside the reply path: the extension and the server are updated
  // independently, so version skew between them is a normal state.
  assert.equal(report({ ...DROPPED_PAGE, dropped: 4 }).dropped, null);
  assert.equal(report({ ...DROPPED_PAGE, dropped: null }).dropped, null);
});

/* A region the server marked `occluded_by` -- a refused watermark or
 * an unread sub-floor box within 16 px, so glyphs may be hidden under it and
 * missing from the read. The wire field is the whole remedy (a glyph hidden
 * under a site banner was once lost inside an otherwise fluent read, with
 * nothing on the wire saying the read was suspect), so this is the guard
 * against the shape x-birelate-untranslated shipped in: emitted by the server,
 * discarded here. */
test("a suspect read reaches the report, and old or clean pages stay silent", () => {
  const report = shippedReport();
  const marked = report({
    ...DROPPED_PAGE,
    dropped: [],
    regions: [{ occluded_by: "unread-box" }, {}, { occluded_by: "watermark" }],
  });
  assert.equal(marked.occluded, "2/3");

  // A clean page and a server too old to send the field count the same zero:
  // the only act downstream is "distrust these reads", and there is nothing to
  // distrust on either.
  assert.equal(report({ ...DROPPED_PAGE, dropped: [] }).occluded, null);
  assert.equal(
    report({ ...DROPPED_PAGE, dropped: [], regions: [null, {}] }).occluded,
    null
  );
});

test("the drop report rides the cache entry, both writing it and reading it", () => {
  /* Not decoration. A drop leaves an erased hole with NO Japanese under it, so
   * a warning that appeared on a page's first view and vanished on its second
   * would leave the reader with a blank patch and no explanation for it at all
   * -- strictly worse than the `missed` case this rule was first written for,
   * where the page at least still shows the source text. */
  assert.match(
    CACHE_JS,
    /async function cachePut\(\{[^}]*\bdropped\b/s,
    "cachePut does not accept `dropped`, so it can never be stored"
  );
  assert.match(
    CACHE_JS,
    /^\s*dropped: dropped \|\| null,$/m,
    "cachePut accepts `dropped` but does not write it onto the entry"
  );

  // Both hit paths: tier 1 is the URL lookup in `cacheLookup`, tier 2 is the
  // hash lookup inside `translate`. A field read back on only one of them
  // reappears and disappears depending on how the reader arrived at the page.
  const reads = BACKGROUND_JS.match(/^\s*dropped: entry\.dropped \|\| null,$/gm) || [];
  assert.equal(reads.length, 2, "both cache-hit replies must read `dropped` back");

  // And the live path has to store what it just reported, or only the second
  // view of a page would ever mention it.
  assert.match(
    BACKGROUND_JS,
    /await cachePut\(\{[^}]*\bdropped,/s,
    "translate() does not pass `dropped` to cachePut"
  );
});

test("the reader is told, and told on a page whose only fault is a drop", () => {
  const missNotes = shippedMissNotes();

  /* A drop-ONLY reply, which is the measured shape: on all 59 instances every
   * other counter read clean. This is the case the guard has to let through --
   * `duplicateIds` and `outOfRangeIds` deliberately do NOT, because a repeated
   * id is only a CAUSE and announcing it alone would put a red bar on a page
   * that is completely correct. A drop is a defect by itself. */
  const text = missNotes({
    missed: null,
    dropped: "4/7",
    cut: false,
    duplicateIds: 0,
    outOfRangeIds: 0,
    keepArtIgnored: false,
  });
  assert.ok(text, "the reader is told nothing about a page that lost four regions");
  // The COUNT, not just some words: "some regions were dropped" cannot be acted
  // on, and 4 of 7 versus 1 of 40 are different pages.
  assert.match(text, /4\/7/);

  /* The wording has to be true under BOTH plans and this is the one that broke
   * it. "Keep the artwork" runs no inpainting stage, so nothing is erased and a
   * dropped region leaves the original Japanese sitting there untouched --
   * telling that reader to look for a blank patch sends them hunting for
   * something that is not on the page. */
  assert.doesNotMatch(text, /erased/);
  assert.doesNotMatch(text, /blank/);
  assert.match(text, /nothing was lettered/);

  // A clean page still raises no bar at all.
  assert.equal(
    missNotes({ missed: null, dropped: null, cut: false, keepArtIgnored: false }),
    null
  );
  assert.equal(missNotes(null), null);

  /* And the drop line is its own, not a rider on the miss line: a page carrying
   * both says both, in that order. */
  const both = missNotes({ missed: "2/7", dropped: "4/7", cut: false });
  assert.match(both, /2\/7 regions left untranslated/);
  assert.match(both, /4\/7/);
});

/* ===================================================================== *
 * EDGE HINTS ON THE WIRE
 *
 * `edge_hints` is a field the server emits and the browser must read. Before
 * these tests, `grep edge_hints tests/` found TWO COMMENTS and no code:
 * background.js parses the wire name in `report()` and `translate()` hands the
 * result to `seamEdges`, and NEITHER line was asserted anywhere. Renaming the
 * field on both sides left the JS suites at 117 passed / 0 failed -- the
 * feature inert, the board green. That is the exact shape this file's own
 * header says it exists to guard against.
 *
 * The seam tests do not cover it and it is worth being exact about why: they
 * drive `seamEdges` and `seamJoinFor` DIRECTLY, with hints handed in as
 * arguments. That proves the geometry and proves nothing about the wire.
 * ===================================================================== */

/* A 908 px slice, so `seamEdgeBand` is max(8, round(908 * 0.006)) = 8. */
const SLICE = { width: 1200, height: 908 };

/* Slice `113` of a measured webtoon chapter, the box this whole channel exists
 * for: the top of a vertical display column, scored 0.2363 against a 0.25 text
 * floor and therefore refused as a region. y + height = 904.5, which is inside
 * the bottom band and outside the top one, so a working wire files it under
 * `bottom` and nothing else. */
const COLUMN_HINT_113 = { x: 52.1, y: 712.9, width: 191.7, height: 191.6, edge: "bottom" };

/* The overflow sentence lives in missNotes' GUARD as well as its
 * body -- the same `if (false && ...)` trap the file header records: a guard
 * that never fires ships a silent page with the suite green. Born red
 * against the guard without `overflowed`. */
test("a spilling placement box is said out loud, and a clean page stays quiet", () => {
  const missNotes = shippedMissNotes();
  const text = missNotes({
    missed: null, dropped: null, cut: false, stillCut: 0,
    keepArtIgnored: false, placementOverflow: [2],
  });
  assert.ok(text, "an overflow on an otherwise clean page must still raise the bar");
  assert.match(text, /placement box/i);
  assert.match(text, /draw a bigger one/i);
  assert.equal(
    missNotes({
      missed: null, dropped: null, cut: false, stillCut: 0,
      keepArtIgnored: false, placementOverflow: [],
    }),
    null,
    "an empty overflow list is a checked, clean page"
  );
});

test("a hint on the wire reaches the edge summary, through report() and into seamEdges", () => {
  const report = shippedReport();
  const summary = report({ ...DROPPED_PAGE, regions: [], edge_hints: [COLUMN_HINT_113] });
  assert.equal(summary.edgeHints.length, 1, "report() dropped the hint on the floor");

  /* THE COMPOSED PATH, which is the whole point of the test. Asserting that
   * `report()` surfaces the field, and separately that `seamEdges` files a hint,
   * is two halves with the `||` between them untested -- and this repo has already
   * shipped a fix whose halves were both green while the join between them was
   * gone. So the value that comes OUT of `report()` is the value handed in here,
   * exactly as `translate()` does it in background.js. */
  const edges = seam.seamEdges([], SLICE.width, SLICE.height, summary.edgeHints);
  assert.equal(edges.bottom.length, 1, "the hint never reached the edge summary");
  assert.equal(edges.top.length, 0, "a bottom-band hint must not be filed at the top");
  assert.equal(edges.bottom[0].y, COLUMN_HINT_113.y);
  assert.equal(edges.bottom[0].height, COLUMN_HINT_113.height);

  /* And the control that makes the assertion above about the WIRE rather than
   * about `seamEdges`: with no regions and no hints this page has no evidence at
   * either edge at all. Without this line a `seamEdges` that invented a box would
   * pass the assertions above. */
  const withoutTheWire = seam.seamEdges([], SLICE.width, SLICE.height, []);
  assert.equal(withoutTheWire.bottom.length, 0);
});

test("a server too old to send edge_hints summarises exactly as it did before hints existed", () => {
  const report = shippedReport();
  /* Absent and empty fall the same way here, unlike `stages_selected` -- the only
   * act downstream is "consider this box when looking for a cut bubble", so a
   * server that never looked and a server that found none produce the same join,
   * which is precisely the join the seam made before hints existed.
   *
   * The wrong-type arms are not decoration: the extension and the server are
   * updated independently, so version skew between them is a normal state. */
  const { edge_hints: _absent, ...old } = { ...DROPPED_PAGE, edge_hints: [] };
  assert.deepEqual(report(old).edgeHints, []);
  assert.deepEqual(report({ ...DROPPED_PAGE, edge_hints: null }).edgeHints, []);
  assert.deepEqual(report({ ...DROPPED_PAGE, edge_hints: 4 }).edgeHints, []);
  assert.deepEqual(report({ ...DROPPED_PAGE, edge_hints: "two" }).edgeHints, []);
});

test("translate() still hands report()'s hints to seamEdges", () => {
  /* A static check, for the same reason the cache test above is one: this hop
   * cannot be executed here at all -- `translate()` is async, takes bytes, and
   * reaches storage, fetch and IndexedDB before it gets anywhere near the seam.
   * What can be checked is that the two ends still name the same value, which is
   * the only way it has ever broken.
   *
   * Deleting the 4th argument to `seamEdges` in translate() is a one-character edit that
   * turns hints off completely, passes every other test in every suite, and
   * cannot be seen in any rendered pixel until a webtoon column is cut. */
  assert.match(
    BACKGROUND_JS,
    /const \{[^}]*\bedgeHints\b[^}]*\} =\s*\n?\s*report\(payload\);/,
    "translate() no longer takes `edgeHints` out of report()"
  );
  /* The 5th argument is the white-continuation profile, pinned the same way and
   * for the same reason: dropping either argument is a one-character edit that
   * turns its channel off completely with every other suite green. */
  assert.match(
    BACKGROUND_JS,
    /seamEdges\(\s*regions,\s*width,\s*height,\s*edgeHints,\s*white\s*\)/,
    "translate() no longer passes the hints and the white profile to seamEdges -- a channel is inert"
  );
});

/* The cut-repair verdict. Until this pair, `cut_found`/`still_cut`
 * lived in the server's tracing only, so "the retry fired and repaired it" and
 * "the retry never fired" were indistinguishable to a reader -- the exact shape
 * x-birelate-untranslated shipped in, an emitted report the browser discarded. */
test("a repaired page is distinguishable from a silently cut one", () => {
  const report = shippedReport();
  const repaired = report({ ...DROPPED_PAGE, dropped: [], cut_found: 3, still_cut: 0 });
  assert.equal(repaired.cutFound, 3);
  assert.equal(repaired.stillCut, 0);

  const silent = report({ ...DROPPED_PAGE, dropped: [], cut_found: 3, still_cut: 2 });
  assert.equal(silent.stillCut, 2);

  // A server too old to send the pair falls to zero -- "no cause to name",
  // exactly as the id counters beside it do. Wrong types fall the same way.
  const { cut_found: _a, still_cut: _b, ...old } = { ...DROPPED_PAGE, cut_found: 0, still_cut: 0 };
  assert.equal(report(old).cutFound, 0);
  assert.equal(report(old).stillCut, 0);
  assert.equal(report({ ...DROPPED_PAGE, cut_found: null, still_cut: "2" }).stillCut, 0);
});

test("missNotes names a shipped cut and stays silent on a repaired one", () => {
  const missNotes = shippedMissNotes();
  /* A still-cut segment is a defect ON THE PAGE by itself -- a bubble ends
   * mid-word right now -- so it belongs in the guard beside `dropped`, not as
   * a rider on `missed`. */
  const note = missNotes({ stillCut: 2 });
  assert.ok(note && /2 .*cut off mid-sentence/.test(note), `got: ${note}`);
  assert.ok(/could not repair/.test(note), `got: ${note}`);

  // A page that was cut and REPAIRED carries no reader-visible defect: the
  // verdict rides the wire and the cache for diagnosis, and the warning bar
  // stays quiet. A note here would cry wolf on a page that is fine.
  assert.equal(missNotes({ cutFound: 3, stillCut: 0 }), null);
});

test("the cut verdict rides the cache entry, both writing it and reading it", () => {
  // Same shape as the drop test above, for the same reason: a warning that
  // vanishes on the second view of the same page is barely a warning.
  assert.match(
    CACHE_JS,
    /async function cachePut\(\{[^}]*\bcutFound\b[^}]*\bstillCut\b/s,
    "cachePut does not accept the cut verdict, so it can never be stored"
  );
  assert.match(
    CACHE_JS,
    /^\s*cutFound: Number\(cutFound\) \|\| 0,$/m,
    "cachePut accepts `cutFound` but does not write it onto the entry"
  );
  assert.match(
    CACHE_JS,
    /^\s*stillCut: Number\(stillCut\) \|\| 0,$/m,
    "cachePut accepts `stillCut` but does not write it onto the entry"
  );

  // Both cache-hit reply paths must read the pair back.
  const cutReads = BACKGROUND_JS.match(/^\s*cutFound: entry\.cutFound \|\| 0,$/gm) || [];
  const stillReads = BACKGROUND_JS.match(/^\s*stillCut: entry\.stillCut \|\| 0,$/gm) || [];
  assert.equal(cutReads.length, 2, "both cache-hit replies must read `cutFound` back");
  assert.equal(stillReads.length, 2, "both cache-hit replies must read `stillCut` back");

  // And the live path stores what it just reported and answers with it.
  assert.match(
    BACKGROUND_JS,
    /const \{[^}]*\bcutFound\b[^}]*\bstillCut\b[^}]*\} =\s*\n?\s*report\(payload\);/s,
    "translate() no longer takes the cut verdict out of report()"
  );
  assert.match(
    BACKGROUND_JS,
    /await cachePut\(\{[^}]*\bcutFound\b[^}]*\bstillCut\b/s,
    "the live path does not store the cut verdict"
  );
});
