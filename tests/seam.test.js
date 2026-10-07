/* Unit tests for extension/seam.js -- the webtoon seam's geometry.
 *
 * Run:  node --test tests/seam.test.js
 *
 * NOT `node --test tests/`. On Node 24 a bare directory argument is resolved
 * as a MODULE -- it reports
 * `Cannot find module '<repo>\tests'`, one failing test named `tests`, and
 * exit 1, while never loading this file at all. That is a green suite failing
 * for a reason that has nothing to do with the code, which is the worst kind.
 * `node --test "tests/*.test.js"` works too, and quoting matters: PowerShell
 * does not glob, so node has to be handed the pattern.
 *
 * There is no package.json in this repo ON PURPOSE: `npx --yes web-ext ...` is
 * what lints and packages the extension, and a package.json at the root changes
 * what npx resolves. So this uses node's own built-in runner and nothing else --
 * no dependencies, no lockfile, no install step.
 *
 * seam.js is a classic browser script; it exports through a `typeof module`
 * guard at the bottom precisely so this file can require it without the
 * extension having to become a module.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  SEAM_MARGIN_PX,
  SEAM_MIN_ASPECT,
  SEAM_MAX_ASPECT,
  SEAM_SFX_LABEL,
  seamEffectiveBox,
  seamEdgeBand,
  seamEdges,
  seamIsSfx,
  seamPairVeto,
  seamPairScore,
  seamCandidates,
  seamPlan,
  seamRunPlan,
  seamRunEdges,
  seamSpansSlice,
  seamRunLength,
  seamJoinFor,
  SEAM_MAX_RUN_SLICES,
  seamCompositeRects,
  seamSkipFinal,
} = require("../extension/seam.js");

/* A plan is a RUN of N slices with N-1 cuts. These name the two-slice case's
 * three landmarks, so every assertion below reads as it did when the plan had a
 * `top`, a `bottom` and one `boundary` -- and so the numbers in them are the
 * same numbers, which is the point: the pairwise join is OBSERVED working and
 * the run generalisation must not move it by a pixel. */
const head = (plan) => plan.slices[0];
const tail = (plan) => plan.slices[plan.slices.length - 1];
const cut = (plan) => plan.boundaries[0];

/* A webtoons.com CANVAS slice, which is the one delivered size with a primary
 * source behind it: the host states it slices uploads to a maximum of 800x1280. */
const W = 800;
const H = 1280;

/* `label` is the detector's own class, exactly as the server spells it on each
 * region of a `format=json` reply (`server/src/regions.rs`: `text` or
 * `onomatopoeia`). Omitted entirely when not passed, because an ABSENT label is
 * a case with its own behaviour and an empty string would not exercise it. */
const region = (x, y, width, height, fit, label) => ({
  x,
  y,
  width,
  height,
  source: "",
  translated: "",
  ...(fit
    ? { fit_x: fit[0], fit_y: fit[1], fit_width: fit[2], fit_height: fit[3] }
    : {}),
  ...(label ? { label } : {}),
});

// A balloon cut by the bottom edge of the upper slice, and its other half.
const cutTop = (label) =>
  seamEdges([region(300, 1100, 200, 150, [280, 1080, 240, 200], label)], W, H);
const cutBottom = (label) =>
  seamEdges([region(300, 10, 200, 90, [280, 0, 240, 120], label)], W, H);

test("seamEffectiveBox prefers the fit box, because the balloon is what the edge cuts", () => {
  assert.deepEqual(seamEffectiveBox(region(10, 20, 30, 40, [1, 2, 3, 4])), {
    x: 1,
    y: 2,
    width: 3,
    height: 4,
  });
  assert.deepEqual(seamEffectiveBox(region(10, 20, 30, 40)), {
    x: 10,
    y: 20,
    width: 30,
    height: 40,
  });
});

test("seamEffectiveBox refuses anything it cannot use", () => {
  assert.equal(seamEffectiveBox(null), null);
  assert.equal(seamEffectiveBox(region(0, 0, 0, 10)), null);
  assert.equal(seamEffectiveBox({ x: 1, y: 2, width: "3", height: 4 }), null);
  // A partial fit set falls back whole rather than mixing the two spaces.
  const partial = { x: 5, y: 6, width: 7, height: 8, fit_x: 1, fit_y: 2 };
  assert.deepEqual(seamEffectiveBox(partial), { x: 5, y: 6, width: 7, height: 8 });
});

test("the edge band is tight, so an ordinary bottom margin cannot reach it", () => {
  /* 16, and it is a CEILING rather than a preference. Swept over both corpora
   * against a vision census of which boundaries are really cut: manga holds at
   * 0 joins through band 16 and produces 2 at band 20, which is the false seam
   * this constant's own comment predicts. */
  assert.equal(seamEdgeBand(1280), 16);
  assert.equal(seamEdgeBand(4000), 24); // the fraction still governs a tall slice
  // The whole point of the tight band: a manga page stacked flush in a
  // vertical-scroll reader has its lowest bubble inside a margin, not on the cut.
  const margined = seamEdges([region(300, 1000, 200, 150)], W, H);
  assert.equal(margined.bottom.length, 0);
});

test("the band admits the near miss it was widened for, and nothing further", () => {
  /* One measured boundary of twelve is the near miss the band reaches: the
   * lower slice's box begins 14.17 px below its own top edge, six
   * pixels outside the old 8 px band. Pinned at the boundary on BOTH sides so a
   * later widening has to change this test and say why.
   *
   * Asserted through `seamEdges`, which is what the caller calls, rather than on
   * `seamEdgeBand` alone -- the band is only interesting through the admission
   * it decides. */
  const admitted = seamEdges([region(94, 14.17, 148, 457)], W, H);
  assert.equal(admitted.top.length, 1, "14.17 px is inside a 16 px band");

  const refused = seamEdges([region(94, 20, 148, 457)], W, H);
  assert.equal(refused.top.length, 0, "20 px is outside it, and band 20 is where "
    + "the manga corpus starts producing false joins");
});

test("seamEdges reports a box against whichever edge it touches, via the fit box", () => {
  const edges = cutTop();
  assert.equal(edges.bottom.length, 1);
  assert.equal(edges.top.length, 0);
  assert.deepEqual(edges.bottom[0], { x: 280, y: 1080, width: 240, height: 200 });

  const other = cutBottom();
  assert.equal(other.top.length, 1);
  assert.equal(other.bottom.length, 0);
});

test("a box can touch both edges at once, and is reported twice", () => {
  const edges = seamEdges([region(10, 0, 100, H)], W, H);
  assert.equal(edges.top.length, 1);
  assert.equal(edges.bottom.length, 1);
});

test("seamPairScore rejects two balloons that merely share an edge", () => {
  const top = { x: 0, y: 0, width: 100, height: 50, edgeDistance: 0 };
  // Three times as wide: not the same balloon.
  assert.equal(seamPairScore(top, { x: 0, y: 0, width: 300, height: 50, edgeDistance: 0 }, 8), null);
  // Same width, opposite side of the page: no overlap and the centres disagree.
  assert.equal(
    seamPairScore(top, { x: 600, y: 0, width: 100, height: 50, edgeDistance: 0 }, 8),
    null
  );
});

test("a balloon clipped by the cut is paired despite the width ratio, and only then", () => {
  /* The real numbers from a measured boundary `011|012`. The balloon
   * is 528px wide on slice 011; the cut leaves 31.5px of glyph on 012, which the
   * detector boxes at 145.3 wide. Width ratio 3.63 against a 2.2 ceiling, so the
   * veto refused it -- and the remnant `跑上去？` was then read as `」ト土？`,
   * refused as not-Chinese, never lettered AND never erased. That residue is what
   * the reader sees. */
  const balloon = { x: 308, y: 639, width: 528, height: 328, edgeDistance: 0 };
  const remnant = {
    x: 424.21875,
    y: 0.5908203125,
    width: 145.3125,
    height: 31.5498046875,
    edgeDistance: 0,
  };
  assert.equal(seamPairVeto(balloon, remnant), null);
  assert.ok(seamPairScore(balloon, remnant, 8) > 0);

  /* AND ONLY THEN. Containment alone is not enough -- the same width disparity
   * between two boxes the cut did NOT clip must still be refused, which is the
   * fixture above at `tests/seam.test.js`'s "merely share an edge". Restated here
   * against `seamPairVeto` directly so the reason is visible beside the case it
   * is distinguished from: equal heights mean nothing was clipped vertically. */
  const flushLeftNarrow = { x: 0, y: 0, width: 100, height: 50, edgeDistance: 0 };
  const flushLeftWide = { x: 0, y: 0, width: 300, height: 50, edgeDistance: 0 };
  assert.equal(seamPairVeto(flushLeftNarrow, flushLeftWide), "width-ratio");

  /* A remnant that is NOT inside the balloon's column stays refused too, so the
   * suspension needs both halves and not just the clipping. */
  const elsewhere = { ...remnant, x: 40 };
  assert.equal(seamPairVeto(balloon, elsewhere), "width-ratio");
});

test("the cut-clip slack admits 015|016, whose upper balloon was never detected", () => {
  /* The real numbers from a measured boundary `015|016`. One black balloon
   * straddles the cut; the upper slice's only detection is the bare text line
   * (no containing bubble found, so `fit` falls back to the padded text box)
   * while the lower's `fit` is the balloon. heightRatio 0.253807 against
   * inverseWidthRatio 0.222173 -- the bare comparison misses by 14.24% and the
   * reader got a spurious fragment on 015 and a subjectless sentence on 016.
   * The 1.2 slack admits it: 0.253807 < 0.222173 * 1.2. */
  const top = { x: 304.5703125, y: 1543.75, width: 102.421875, height: 50, label: "text" };
  const bottom = { x: 128, y: 0, width: 461, height: 197, label: "text" };
  assert.equal(seamPairVeto(top, bottom), null);

  // And end to end: the pair plans the same 690x1380 crop shape the chapter's
  // eleven firing boundaries produce, holding the whole balloon.
  const a = seamEdges([region(top.x, top.y, top.width, top.height)], 690, 1600);
  const b = seamEdges([region(bottom.x, bottom.y, bottom.width, bottom.height)], 690, 1600);
  const plan = seamPlan(a, b);
  assert.ok(plan, "the boundary must plan");
  assert.equal(plan.width, 690);
  assert.equal(plan.height, 1380);
});

test("the slack does not reach the nearest unwanted cases, and a widening must say why", () => {
  /* The window measured over 581 width-ratio vetoes across eleven corpora:
   * wanted 1.1424 (015|016 above); nearest unwanted 1.2579 (116|117, ratio
   * below) and 1.3055 (138|139 on another chapter, a VERIFIED false join -- a
   * dialogue balloon against an unrelated blue drawn SFX).
   * The slack ships at 1.2, between them. Anyone raising it re-admits these. */
  const t116 = { x: 76.8, y: 3.5, width: 192.8, height: 904.5, label: "text" };
  const b117 = { x: 167.6, y: 0, width: 35.2, height: 207.5, label: "text" };
  assert.equal(seamPairVeto(t116, b117), "width-ratio");

  /* 138|139 is refused twice over: its ratio 1.3055 is outside the slack, AND
   * the sfx guard refuses the escape by category -- a drawn sound effect is
   * never a clipped remnant of a dialogue balloon. The guard is what keeps
   * this false join out even if the constant is ever widened. */
  const t138 = { x: 685, y: 517, width: 594, height: 482, label: "text" };
  const b139 = { x: 955, y: 0.48828125, width: 230, height: 243.65234375, label: "onomatopoeia" };
  assert.equal(seamPairVeto(t138, b139), "width-ratio");
});

/* The dropped reject, pinned as arithmetic so nobody restores it.
 *
 * comic-translate refuses a pair when the overlap is poor AND the centres
 * disagree. `overlap >= 1 - centre` holds for every pair of boxes, so the second
 * test is implied by the first and the conjunction is just the overlap test. A
 * brute-force sweep is a stronger statement than the algebra alone: if any
 * (width, offset) at all could satisfy one and not the other, this finds it. */
test("no pair can have a poor overlap and agreeing centres, so the centre reject is dead", () => {
  let reachable = 0;
  for (let width = 20; width <= 400; width += 4) {
    for (let other = 20; other <= 400; other += 4) {
      for (let offset = -500; offset <= 500; offset += 2) {
        const min = Math.min(width, other);
        const left = Math.max(0, offset);
        const right = Math.min(width, offset + other);
        const overlap = Math.max(0, right - left) / min;
        const center = Math.abs(width / 2 - (offset + other / 2)) / min;
        if (overlap < 0.35 && center <= 0.35) reachable += 1;
      }
    }
  }
  assert.equal(reachable, 0);
});

test("a tolerable difference in detected width is still paired", () => {
  const top = { x: 100, y: 0, width: 200, height: 50, edgeDistance: 0 };
  // The lower half detected 20% narrower and shifted: the same balloon, read
  // twice by a segmentation head whose mask cells are several pixels wide.
  const narrower = { x: 118, y: 0, width: 160, height: 50, edgeDistance: 0 };
  assert.ok(seamPairScore(top, narrower, 8) !== null);
});

test("seamPairScore prefers the better-agreeing pair", () => {
  const top = { x: 100, y: 0, width: 200, height: 50, edgeDistance: 0 };
  const exact = { x: 100, y: 0, width: 200, height: 50, edgeDistance: 0 };
  const loose = { x: 140, y: 0, width: 240, height: 50, edgeDistance: 0 };
  assert.ok(seamPairScore(top, exact, 8) > seamPairScore(top, loose, 8));
});

test("seamPlan needs text on BOTH sides of the cut", () => {
  assert.equal(seamPlan(cutTop(), seamEdges([], W, H)), null);
  assert.equal(seamPlan(seamEdges([], W, H), cutBottom()), null);
});

test("seamPlan refuses slices that are not the same strip", () => {
  const narrower = seamEdges([region(300, 10, 200, 90, [280, 0, 240, 120])], 600, H);
  assert.equal(seamPlan(cutTop(), narrower), null);
});

test("seamPlan cuts a crop that holds the whole bubble with its margin", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  assert.ok(plan);
  assert.equal(plan.width, W);
  assert.equal(plan.height, head(plan).height + tail(plan).height);
  assert.equal(cut(plan), head(plan).height);
  assert.equal(head(plan).pageHeight, H);
  assert.equal(tail(plan).pageHeight, H);

  // The upper half starts at least a margin above the balloon's top edge...
  assert.ok(head(plan).y <= 1080 - SEAM_MARGIN_PX);
  // ...and the lower half runs at least a margin past the balloon's bottom.
  assert.ok(tail(plan).height >= 120 + SEAM_MARGIN_PX);
  assert.equal(plan.pairs.length, 1);
});

test("seamPlan grows the crop to the slices' own shape, not to a constant", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  // Portrait slices: one slice tall.
  assert.equal(plan.height, H);
  // Which is the whole point: a band this wide and this short would have its
  // vertical axis upscaled several times harder than its horizontal one.
  assert.ok(plan.height / plan.width > SEAM_MIN_ASPECT);
});

/* The real material this was corrected against: a stripline chapter delivers
 * 1280x1000 LANDSCAPE slices, where the old fixed 1.35 asked for a 1280x1728
 * crop -- taller than either slice, and nearly two slices of translation per
 * join, to reach a shape that chapter is not translated at anywhere else. */
test("a landscape strip gets a landscape seam, not a manga page", () => {
  const wide = 1280;
  const tall = 1000;
  const a = seamEdges([region(400, 900, 300, 100, [380, 880, 340, 120])], wide, tall);
  const b = seamEdges([region(400, 10, 300, 80, [380, 0, 340, 110])], wide, tall);
  const plan = seamPlan(a, b);
  assert.equal(plan.height, tall);
  assert.ok(plan.height < Math.round(wide * 1.35));
});

test("a short tail slice still gets a usable crop rather than a band", () => {
  // 1280x213 is a real tail slice size. Without the floor
  // the crop would be a couple of hundred pixels tall against 1280 wide, which
  // is the aspect that upscales glyphs several times harder on one axis.
  const wide = 1280;
  const a = seamEdges([region(400, 900, 300, 100, [380, 880, 340, 120])], wide, 1000);
  const b = seamEdges([region(400, 10, 300, 80, [380, 0, 340, 110])], wide, 213);
  const plan = seamPlan(a, b);
  assert.ok(plan.height >= Math.round(wide * SEAM_MIN_ASPECT));
  assert.ok(plan.height <= 1000 + 213);
});

/* The width test is a VETO, and real material is why. Chapter 381206 changes
 * width mid-chapter -- slices 60-67 are 1000 wide inside an otherwise 1280-wide
 * run -- so the two boundaries either side of that run pair images of different
 * widths. Stacking them needs a scale or a pad: a scale breaks the 1:1 mapping
 * every paint-back offset depends on, and a pad invents an edge mid-image for
 * the detector to read as a panel border. Equal width is also the signal that
 * separates a strip from a stack of pages at all. */
test("a width change between slices vetoes the join outright", () => {
  const a = seamEdges([region(400, 900, 300, 100, [380, 880, 340, 120])], 1280, 1000);
  const b = seamEdges([region(300, 10, 300, 80, [280, 0, 340, 110])], 1000, 1000);
  assert.equal(seamPlan(a, b), null);
  assert.equal(seamPlan(b, a), null);
});

test("padding takes what it can from the side that has room", () => {
  // A cut only 60px into a SHORT lower slice: nearly all the growth must come
  // from above, and the crop must still reach the target height.
  const short = seamEdges([region(300, 5, 200, 40, [280, 0, 240, 60])], W, 100);
  const plan = seamPlan(cutTop(), short);
  assert.ok(plan);
  assert.equal(tail(plan).height, 100);
  // The taller slice sets the target; the short one contributes what it has.
  assert.equal(plan.height, H);
  assert.equal(head(plan).height, H - 100);
});

test("padding never shrinks a crop below the bubble it exists to hold", () => {
  // A balloon 1240px tall on the upper slice, so the crop already exceeds the
  // slice height before any padding: the growth term goes negative and must not
  // be applied backwards.
  const tall = seamEdges([region(300, 40, 200, 1240, [280, 40, 240, 1240])], W, H);
  const plan = seamPlan(tall, cutBottom());
  assert.ok(plan);
  assert.ok(head(plan).height >= 1240 + SEAM_MARGIN_PX - 40);
  assert.ok(plan.height > H);
  assert.ok(plan.height <= Math.round(W * SEAM_MAX_ASPECT));
});

test("seamPlan refuses a crop that is really a whole panel", () => {
  // Both slices covered edge to edge: nothing here is a bubble, and the result
  // would be a 1:2.5 strip that RF-DETR reads at the wrong aspect anyway.
  const all = seamEdges([region(0, 0, W, H)], W, H);
  assert.equal(seamPlan(all, all), null);
});

test("seamPlan matches two side-by-side bubbles 1:1 and holds both", () => {
  const tops = seamEdges(
    [region(60, 1150, 200, 130, [50, 1150, 220, 130]), region(520, 1150, 200, 130, [510, 1150, 220, 130])],
    W,
    H
  );
  const bottoms = seamEdges(
    [region(60, 0, 200, 90, [50, 0, 220, 90]), region(520, 0, 200, 90, [510, 0, 220, 90])],
    W,
    H
  );
  const plan = seamPlan(tops, bottoms);
  assert.equal(plan.pairs.length, 2);
  // Each was paired with the one above it, not across the page.
  for (const pair of plan.pairs) {
    assert.ok(Math.abs(pair.top.x - pair.bottom.x) < 5);
  }
});

test("one balloon cannot claim two partners", () => {
  const tops = seamEdges([region(300, 1150, 200, 130, [290, 1150, 220, 130])], W, H);
  const bottoms = seamEdges(
    [region(300, 0, 200, 90, [290, 0, 220, 90]), region(320, 0, 180, 90, [310, 0, 200, 90])],
    W,
    H
  );
  assert.equal(seamPlan(tops, bottoms).pairs.length, 1);
});

/* ------------------------------------------------- the sound-effect veto
 *
 * The geometry cannot tell a cut balloon from a drawn effect clipped by a page
 * trim, because there is nothing about the SHAPE of the pairing that differs.
 * Only the detector's class does, and a cut balloon is dialogue on both sides of
 * the cut because it is one balloon.
 *
 * Measured, and this is why the rule is worth its bytes:
 *  - manga, 213 pages: 1 join of 212 boundaries, and it was an
 *    impact effect against the next chapter's clipped title logo. The veto takes
 *    it to 0 of 212 and reclassifies one already-refused boundary.
 *  - webtoon, chapter 381206, 219 slices: 26 joins become 16. All ten it removes
 *    are effect-against-effect, and every one of them was junk -- mean recovered
 *    0.10, two painting no rectangle at all, one turning two stray glyphs into
 *    an invented "thank you" painted back over a correct page. The sixteen that survive
 *    include every rejoined sentence the feature exists for. 117.3s of GPU time
 *    on that chapter alone, 5.0% of its pipeline total, bought single glyphs.
 */

test("two sound effects at one cut are never a cut balloon", () => {
  // The identical geometry that joins above, with both boxes labelled.
  assert.ok(seamPlan(cutTop(), cutBottom()));
  assert.equal(seamPlan(cutTop(SEAM_SFX_LABEL), cutBottom(SEAM_SFX_LABEL)), null);
});

/* THE ONE THAT MUST NOT REGRESS. Every cache entry written before the label was
 * summarised holds bare four-number boxes, and the extension reads those back
 * for 24 hours. If an absent label counted as an effect the veto would refuse
 * every join against every entry already on disk -- the whole feature silently
 * off, with nothing in the UI to say so and no error anywhere to find it by.
 * Unknown is unknown. */
test("an ABSENT label is unknown, not an effect, so an old cache entry still joins", () => {
  const stored = cutTop();
  const other = cutBottom();
  // Exactly the shape a pre-veto entry carries: four numbers and no class.
  assert.deepEqual(Object.keys(stored.bottom[0]).sort(), ["height", "width", "x", "y"]);
  assert.ok(seamPlan(stored, other));
  assert.equal(seamIsSfx(stored.bottom[0]), false);
  assert.equal(seamPairVeto({ x: 0, y: 0, width: 100, height: 50 }, { x: 0, y: 0, width: 100, height: 50 }), null);
});

/* The narrowest rule that covers the observed failure, and deliberately no
 * wider. A mismatched pair is not something the measurement has an opinion
 * about: nothing in either chapter shows an effect paired with dialogue being
 * either a real join or a false one, so it is left exactly as it was. */
test("one effect against one line of dialogue is left alone", () => {
  assert.ok(seamPlan(cutTop(SEAM_SFX_LABEL), cutBottom()));
  assert.ok(seamPlan(cutTop(), cutBottom(SEAM_SFX_LABEL)));
  assert.ok(seamPlan(cutTop(SEAM_SFX_LABEL), cutBottom("text")));
  // And two ordinary text regions -- the overwhelmingly common labelled case --
  // are untouched by any of this.
  assert.ok(seamPlan(cutTop("text"), cutBottom("text")));
});

test("the label reaches the summary that rides the cache entry", () => {
  const edges = seamEdges([region(300, 1100, 200, 150, [280, 1080, 240, 200], "onomatopoeia")], W, H);
  assert.equal(edges.bottom[0].label, "onomatopoeia");
  // And nothing else does: the summary is still four numbers and a class name,
  // because anything stored beside the blob is invisible to the LRU byte cap.
  assert.deepEqual(Object.keys(edges.bottom[0]).sort(), ["height", "label", "width", "x", "y"]);
});

test("seamEffectiveBox carries a label through and invents none", () => {
  assert.equal(seamEffectiveBox(region(10, 20, 30, 40, null, "text")).label, "text");
  assert.equal("label" in seamEffectiveBox(region(10, 20, 30, 40)), false);
  // Junk from the network is not a class. An empty string must not be stored:
  // it would read as a real label that merely is not `onomatopoeia`.
  assert.equal("label" in seamEffectiveBox({ x: 1, y: 2, width: 3, height: 4, label: "" }), false);
  assert.equal("label" in seamEffectiveBox({ x: 1, y: 2, width: 3, height: 4, label: 7 }), false);
});

/* The veto has to be COUNTABLE, or its only evidence is a join that stopped
 * happening. Tallying this string per boundary across a whole chapter is how
 * the ten webtoon joins above were identified as effect-against-effect rather
 * than inferred to be. */
test("seamCandidates names the veto, and only when it refused everything", () => {
  const both = seamCandidates(cutTop(SEAM_SFX_LABEL), cutBottom(SEAM_SFX_LABEL));
  assert.equal(both.refused, "sfx");
  assert.equal(both.candidates[0].refused, "sfx");
  assert.equal(both.candidates[0].score, null);

  // A boundary refused by geometry keeps saying so; the two reasons are not
  // interchangeable and a measurement must not have to guess which fired.
  const apart = seamCandidates(
    seamEdges([region(0, 1150, 100, 130, [0, 1150, 100, 130], SEAM_SFX_LABEL)], W, H),
    seamEdges([region(700, 0, 100, 90, [700, 0, 100, 90])], W, H)
  );
  assert.equal(apart.refused, "no-pair");
  assert.equal(apart.candidates[0].refused, "overlap");
  assert.equal(seamPairVeto({ x: 0, y: 0, width: 100, height: 50 }, { x: 0, y: 0, width: 300, height: 50 }), "width-ratio");
});

/* The real material, to the pixel. Both boxes are read straight off the two
 * pages' stored replies -- the fit boxes of the two `onomatopoeia` regions
 * that produced the manga volume's single false join. Every
 * geometric test passes them: width ratio 1.46 against a 2.2 ceiling, overlap
 * 0.75 against a 0.35 floor. Stripping the labels fires the join again, which is
 * what makes this a test of the veto rather than of the arithmetic. */
test("the measured false join between two manga pages is refused, and nothing else refuses it", () => {
  const PAGE_W = 844;
  const PAGE_H = 1200;
  const impact = (label) =>
    seamEdges(
      [region(276.9375, 946.875, 270.34375, 253.125, [276.9375, 946.875, 270.34375, 253.125], label)],
      PAGE_W,
      PAGE_H
    );
  const logo = (label) =>
    seamEdges(
      [
        region(
          332.2,
          -32.9,
          418.7,
          242.7,
          [344.5234375, 0, 393.9765625, 176.953125],
          label
        ),
      ],
      PAGE_W,
      PAGE_H
    );

  // As it shipped: a full seam translation, and a rectangle painted back.
  const before = seamPlan(impact(), logo());
  assert.ok(before);
  assert.equal(before.pairs.length, 1);

  // Labelled, which is what the server reports.
  assert.equal(seamPlan(impact("onomatopoeia"), logo("onomatopoeia")), null);
  assert.equal(seamCandidates(impact("onomatopoeia"), logo("onomatopoeia")).refused, "sfx");
});

/* ------------------------------------------------------- painting it back */

const spanning = (plan) => [
  region(
    280,
    cut(plan) - 60,
    240,
    160,
    [280, cut(plan) - 60, 240, 160]
  ),
];

test("a composite rectangle maps back into both slices without an offset", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  const rects = seamCompositeRects(plan, spanning(plan));
  assert.equal(rects.length, 1);
  const parts = rects[0].parts;
  assert.equal(parts.length, 2);

  const top = parts.find((part) => part.slice === 0);
  const bottom = parts.find((part) => part.slice === 1);

  // The top part reads from the seam above the cut and writes into the upper
  // slice at exactly the offset the crop was taken from.
  assert.equal(top.target.y, head(plan).y + top.source.y);
  assert.equal(top.source.y + top.source.height, cut(plan));
  assert.equal(top.target.y + top.target.height, head(plan).pageHeight);

  // The bottom part starts at the cut and writes into the lower slice from the
  // top of the crop, which is row 0 of that slice.
  assert.equal(bottom.source.y, cut(plan));
  assert.equal(bottom.target.y, 0);
  assert.equal(bottom.source.width, bottom.target.width);
  assert.equal(bottom.source.height, bottom.target.height);
});

test("every composited rectangle stays inside the seam and inside both slices", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  for (const rect of seamCompositeRects(plan, spanning(plan))) {
    assert.ok(rect.rect.x >= 0 && rect.rect.y >= 0);
    assert.ok(rect.rect.x + rect.rect.width <= plan.width);
    assert.ok(rect.rect.y + rect.rect.height <= plan.height);
    for (const part of rect.parts) {
      const page = plan.slices[part.slice];
      assert.ok(part.target.y >= 0);
      assert.ok(part.target.y + part.target.height <= page.pageHeight);
      assert.ok(part.target.x + part.target.width <= plan.width);
    }
  }
});

test("the seam's own spanning region widens the rectangle when it is bigger", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  /* The narrow arm used to be `[]` -- a composite that read nothing -- and the
   * part gate rightly makes that paint nothing at all now. The comparison this
   * test exists for is unchanged: a spanning region bigger than the pair's own
   * union is what widens the rectangle. */
  const narrow = seamCompositeRects(plan, spanning(plan));
  const wide = seamCompositeRects(plan, [
    region(100, cut(plan) - 80, 600, 200, [100, cut(plan) - 80, 600, 200]),
  ]);
  assert.ok(wide[0].rect.width > narrow[0].rect.width);
  assert.ok(wide[0].rect.width <= plan.width);
});

test("a rectangle that half-covers a neighbouring region swallows it whole", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  // A caption sitting just left of the rejoined bubble and overlapping it, which
  // the seam lettered at its own page-coherent size. Cutting through it would
  // put two different renderings of one caption either side of a straight edge.
  const neighbour = region(60, cut(plan) - 40, 300, 80, [60, cut(plan) - 40, 300, 80]);
  const rects = seamCompositeRects(plan, [neighbour]);
  const rect = rects[0].rect;
  assert.ok(rect.x <= 60);
  assert.ok(rect.x + rect.width >= 360);
});

test("a region the rectangle does not touch is left alone", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  const far = region(0, 0, 120, 60, [0, 0, 120, 60]);
  // The spanning region carries the rectangle; `far` must not be dragged into it.
  const rects = seamCompositeRects(plan, [...spanning(plan), far]);
  assert.ok(rects[0].rect.y > 60);
});

test("no plan means nothing to paint", () => {
  assert.deepEqual(seamCompositeRects(null, []), []);
  assert.deepEqual(seamCompositeRects({ pairs: [] }, []), []);
});

/* ------------------------------------------------ the part gate on delivery
 *
 * A part whose window holds no lettered region delivers nothing: every pixel it
 * would paint is the composite's own SOURCE, so the only thing painting it can
 * do to a canvas is restore raw ink over lettering some other pass delivered.
 * Measured on boundary `043` of a webtoon chapter: the 043|044
 * pair's rectangle covered all 972 rows of slice 043 while every lettered region
 * on that composite sat in the 044 band, so painting the 043 part restored the
 * raw 三重防御 column over the "TRIPLE DEFENSE!" lettering the 042|043
 * join had just delivered -- 80,687 px un-lettered.
 *
 * RAW boxes, not padded: on that same composite the 044-band balloon sits
 * 1.75 px below the cut, so a padded test would re-admit the very part this
 * gate exists to drop. A genuinely cut bubble needs no padding -- its composite
 * box CROSSES the cut and admits both parts on its own. */

test("a part whose window holds no lettered region is not emitted", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  // The composite read one region, entirely below the cut -- the shape of
  // `那时的你…` on seam-043: y starts 2 px under the boundary.
  const below = region(300, cut(plan) + 2, 200, 150, [300, cut(plan) + 2, 200, 150]);
  const rects = seamCompositeRects(plan, [below]);
  assert.equal(rects.length, 1);
  assert.deepEqual(
    rects[0].parts.map((part) => part.slice),
    [1],
    "only the band that holds the lettering is painted back"
  );
});

test("a refused region does not admit a part", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  const below = region(300, cut(plan) + 2, 200, 150, [300, cut(plan) + 2, 200, 150]);
  const refusedAbove = {
    ...region(300, cut(plan) - 80, 200, 60, [300, cut(plan) - 80, 200, 60]),
    refused: "a site watermark, not dialogue",
  };
  const rects = seamCompositeRects(plan, [below, refusedAbove]);
  assert.equal(rects.length, 1);
  assert.deepEqual(
    rects[0].parts.map((part) => part.slice),
    [1],
    "a region the server refused was never lettered, so it admits nothing"
  );
});

test("a spanning region admits both parts, exactly as before the gate", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  const rects = seamCompositeRects(plan, spanning(plan));
  assert.deepEqual(
    rects[0].parts.map((part) => part.slice),
    [0, 1]
  );
});

/* --------------------------------------------------- the part CLIP on delivery
 *
 * Admission is not delivery: the gate above decides WHICH bands paint, and an
 * admitted band used to paint its whole slab. At boundary `075` of a measured
 * chapter, the composite read one region (112,34,122x451), the
 * plan's pair box spanned all 919 rows of slice 076, and the admitted slab
 * restored raw `五灯齐明。` over the base render's delivered "...FIVE LAMPS
 * ALIGHT!" column -- ~68,874 px un-lettered, the same shape on 24-26 of
 * the chapter's 30 stored composites. The part is now clipped to the padded
 * union of the boxes that admitted it.
 *
 * These assert on `parts[i].source` / `parts[i].target` -- the rectangles
 * `seamPaint` actually consumes -- because the three gate tests above assert
 * only `parts.map(p => p.slice)` and pass with a clip entirely unwired (the
 * lesson: name the composed predicate and assert on what the caller calls). */

// The 075|076 shape: the pair's lower box covers nearly the whole lower slice,
// the composite letters only a short region crossing the cut.
const cutBottomTall = () =>
  seamEdges([region(300, 10, 200, 90, [280, 0, 240, 1200])], W, H);

test("an admitted part is clipped to the padded union of the boxes that admitted it", () => {
  const plan = seamPlan(cutTop(), cutBottomTall());
  const read = region(300, cut(plan) - 50, 200, 120, [300, cut(plan) - 50, 200, 120]);
  const rects = seamCompositeRects(plan, [read]);
  assert.equal(rects.length, 1);
  const parts = rects[0].parts;
  assert.deepEqual(parts.map((part) => part.slice), [0, 1]);

  const top = parts.find((part) => part.slice === 0);
  const bottom = parts.find((part) => part.slice === 1);

  // The lower slab used to run to the rect's bottom -- the pair box drags that
  // ~1,100 rows down -- and now ends one margin below the lettered read.
  assert.equal(bottom.source.y, cut(plan));
  assert.equal(
    bottom.source.height,
    Math.ceil(read.y + read.height + SEAM_MARGIN_PX) - cut(plan)
  );
  assert.ok(
    bottom.source.height < rects[0].rect.y + rects[0].rect.height - cut(plan),
    "the clip must deliver less than the slab the pair box admitted"
  );

  // The upper part still ends exactly at the cut, and starts no higher than one
  // margin above the read -- the clip floor.
  assert.equal(top.source.y + top.source.height, cut(plan));
  assert.ok(top.source.y >= Math.floor(read.y - SEAM_MARGIN_PX));

  // Delivery still covers the lettered box itself, with its margin, on both
  // axes -- the clip removes source rows, never lettering.
  for (const part of [top, bottom]) {
    assert.ok(part.source.x <= read.x - SEAM_MARGIN_PX + 1);
    assert.ok(part.source.x + part.source.width >= read.x + read.width + SEAM_MARGIN_PX - 1);
    assert.equal(part.target.x, part.source.x);
    assert.equal(part.target.width, part.source.width);
    assert.equal(part.target.height, part.source.height);
  }
  // And the band mapping is unchanged: the lower part still writes from the
  // top of its slice.
  assert.equal(bottom.target.y, tail(plan).y);
});

test("a box covering its whole admitted slab keeps the slab byte-for-byte", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  // The composite region IS the pair's lower box: its padded union covers the
  // whole lower slab, so the clip must reproduce the pre-clip part exactly.
  const read = region(280, cut(plan), 240, 120, [280, cut(plan), 240, 120]);
  const rects = seamCompositeRects(plan, [read]);
  assert.equal(rects.length, 1);
  const bottom = rects[0].parts.find((part) => part.slice === 1);
  assert.ok(bottom, "the lower band is admitted");
  assert.deepEqual(bottom.source, {
    x: rects[0].rect.x,
    y: cut(plan),
    width: rects[0].rect.width,
    height: rects[0].rect.y + rects[0].rect.height - cut(plan),
  });
});

/* ------------------------------------------ the neighbour sweep, after review
 *
 * The sweep used to be a single forward pass over a rectangle it grew as it
 * went. That is the worst of both: a later box is tested against the ALREADY
 * grown rectangle, so it cascades anyway, while an earlier box the growth passed
 * over is never revisited -- so which regions ended up cut in half depended on
 * the order the server happened to list its regions in. Both halves are pinned
 * here.
 */

test("the swallow sweep reaches a fixpoint, whatever order the regions arrive in", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  // A chain: each box overlaps the previous one, reaching leftwards away from
  // the rejoined bubble. Order-independence is the property under test.
  const chain = [
    region(430, cut(plan) - 20, 120, 60, [430, cut(plan) - 20, 120, 60]),
    region(330, cut(plan) - 20, 120, 60, [330, cut(plan) - 20, 120, 60]),
    region(230, cut(plan) - 20, 120, 60, [230, cut(plan) - 20, 120, 60]),
  ];
  const forward = seamCompositeRects(plan, chain)[0].rect;
  const backward = seamCompositeRects(plan, [...chain].reverse())[0].rect;
  assert.deepEqual(forward, backward);
  // And every one of them is covered, not merely the ones the pass happened to
  // reach after the rectangle had grown.
  for (const box of chain) {
    const b = seamEffectiveBox(box);
    assert.ok(forward.x <= b.x && forward.x + forward.width >= b.x + b.width);
  }
});

test("a swallowed region keeps its margin, so the edge never lands on its box", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  const neighbour = region(60, cut(plan) - 40, 300, 80, [60, cut(plan) - 40, 300, 80]);
  const rect = seamCompositeRects(plan, [neighbour])[0].rect;
  // x is clamped to the seam at 0, so test the right edge, which is not.
  assert.ok(rect.x + rect.width >= 360 + SEAM_MARGIN_PX);
});

test("a rectangle that would repaint most of the seam abandons the join instead", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  // One enormous region across the middle: swallowing it would repaint the page
  // rather than the bubble, dragging every layer in it to the seam's size base.
  const huge = region(0, 0, plan.width, plan.height, [0, 0, plan.width, plan.height]);
  assert.deepEqual(seamCompositeRects(plan, [huge]), []);
});

/* ---- run detection: a name drawn down more than two slices -----------------
 *
 * The pairwise seam cannot repair a region that is cut above AND below, because
 * no pair of slices contains it. Measured on one webtoon chapter: `115` and
 * `159` are both spanned end to end, and the runs `115-118` and `143-146` are
 * four slices long. `115`'s fragment was lettered `NO. 1: KUMOMORI!` before a
 * server-side gate refused it.
 */

/* A slice whose only region runs its whole height -- the middle of a run. */
const spanned = () => seamEdges([region(10, 0, 100, H)], W, H);
/* A slice whose region stops inside it -- a run ends here. */
const contained = () => seamEdges([region(10, 300, 100, 200)], W, H);
/* A slice cut only at its bottom, which is the ordinary pairwise case. */
const cutAtBottom = () => seamEdges([region(10, H - 100, 100, 100)], W, H);

test("a region touching both cuts is spanned, and one touching neither is not", () => {
  assert.equal(seamSpansSlice(spanned()), true);
  assert.equal(seamSpansSlice(contained()), false);
  assert.equal(seamSpansSlice(cutAtBottom()), false);
});

test("spanning is identity against seamEdges' own bands, not a second rule", () => {
  // The same box object has to be in both lists; a page with two DIFFERENT
  // regions, one at each cut, is not spanned and must not be treated as a run.
  const two = seamEdges([region(10, 0, 100, 90), region(500, H - 90, 100, 90)], W, H);
  assert.equal(two.top.length, 1);
  assert.equal(two.bottom.length, 1);
  assert.equal(seamSpansSlice(two), false);
});

test("a degenerate or absent edge list is never spanned", () => {
  assert.equal(seamSpansSlice(null), false);
  assert.equal(seamSpansSlice(undefined), false);
  assert.equal(seamSpansSlice({}), false);
  assert.equal(seamSpansSlice(seamEdges([], W, H)), false);
});

test("an ordinary boundary is a run of two, which is what the seam always did", () => {
  assert.equal(seamRunLength([cutAtBottom(), contained()], 0), 2);
});

// SYNTHETIC, and the name used to claim otherwise. `115-118` is not a run in
// that chapter -- only `115` and `159` are crossed end to end, so the real runs are
// three slices. The four-slice GEOMETRY is still worth a test, because
// SEAM_MAX_RUN_SLICES is 6 and a longer strip elsewhere will reach it.
test("a four-slice run is found whole, with three spanned slices inside it", () => {
  // Three spanned slices between the ends; the last one ends the run.
  const list = [cutAtBottom(), spanned(), spanned(), contained()];
  assert.equal(seamRunLength(list, 0), 4);
});

test("a run stops at the first slice whose text ends inside it", () => {
  const list = [cutAtBottom(), spanned(), contained(), spanned()];
  assert.equal(seamRunLength(list, 0), 3);
});

test("a run that would leave the end of the list is refused rather than truncated", () => {
  // Nothing after the last spanned slice, so the name has no end to join to.
  assert.equal(seamRunLength([cutAtBottom(), spanned()], 0), 0);
  assert.equal(seamRunLength([contained()], 0), 0);
  assert.equal(seamRunLength([], 0), 0);
  assert.equal(seamRunLength([cutAtBottom(), contained()], -1), 0);
});

test("the cap bounds a pathological page rather than any real one", () => {
  // Every slice spanned: without the cap this would swallow the whole chapter.
  const list = Array.from({ length: 40 }, spanned);
  assert.equal(seamRunLength(list, 0), SEAM_MAX_RUN_SLICES);
  // And an explicit limit is honoured, so the caller can be stricter.
  assert.equal(seamRunLength(list, 0, 3), 3);
  // A nonsense limit falls back to the constant rather than to zero or one.
  assert.equal(seamRunLength(list, 0, 1), SEAM_MAX_RUN_SLICES);
  assert.equal(seamRunLength(list, 0, NaN), SEAM_MAX_RUN_SLICES);
});

/* ------------------------------------- deciding WHICH run a boundary belongs to
 *
 * The content script's own half of the wiring, extracted here because it has no
 * other harness -- `runSeam` needs a document, a WeakMap of translated images
 * and a live layout. Everything in it that is a DECISION rather than a DOM walk
 * is this function, and this repo has already shipped a fix whose halves were
 * both tested while the predicate between them was never called.
 */

/* `{head, length}` alone, so a run is legible beside the numbers it produces. */
const runOf = (found) =>
  found.wait ? "wait" : found.plan ? { head: found.head, length: found.length } : "none";
/* Neither end of the strip continues: nothing more will ever arrive. */
const ENDED = { above: false, below: false };
/* Both ends continue with a picture that is merely untranslated. */
const MORE = { above: true, below: true };

/* A slice crossed end to end by a column WIDE enough to pair with `cutTop`'s
 * balloon. `spanned()` is a narrow strip at x=10 and pairs with nothing over
 * there -- fine for testing detection, useless for testing a join. */
const spannedWide = () => seamEdges([region(280, 0, 240, H)], W, H);

test("an ordinary boundary plans itself, exactly as the pairwise seam did", () => {
  const list = [cutTop(), cutBottom()];
  const found = seamJoinFor(list, 0, undefined, ENDED);
  assert.deepEqual(runOf(found), { head: 0, length: 2 });
  assert.deepEqual(found.plan, seamPlan(list[0], list[1]));
});

/* The three arrivals of one four-slice name. Auto mode translates in order, so
 * `115-118` becomes joinable at `115|116`, then `116|117`, then `117|118` -- and
 * only the last of those can plan the whole thing. */
test("a run is planned once, from its head, whichever boundary got there last", () => {
  const list = fourSliceRun();
  // 115|116 and 116|117: the name has no end in view and the slice that would
  // finish it is merely untranslated. WAIT -- not an answer, and not a refusal.
  assert.equal(runOf(seamJoinFor(list.slice(0, 2), 0, undefined, MORE)), "wait");
  assert.equal(runOf(seamJoinFor(list.slice(0, 3), 1, undefined, MORE)), "wait");
  // 117|118 completes it, and walks back to 115 rather than joining 117 to 118.
  assert.deepEqual(runOf(seamJoinFor(list, 2, undefined, ENDED)), { head: 0, length: 4 });
  // Entered from any boundary inside it, the answer is the same join -- which is
  // what lets the caller's "already asked" key match instead of translating one
  // name twice at two different lengths.
  assert.deepEqual(runOf(seamJoinFor(list, 1, undefined, ENDED)), { head: 0, length: 4 });
  assert.deepEqual(runOf(seamJoinFor(list, 0, undefined, ENDED)), { head: 0, length: 4 });
});

/* WAITING AND DISCARDING ARE NOT THE SAME ANSWER, and collapsing them lost a
 * join. The first image of a chapter can carry a region crossed end to end;
 * there is no slice above it and there never will be, so deferring is not a
 * postponement, it is a refusal that the caller records as final. */
test("a name running off the top waits only while a picture is still coming", () => {
  const list = [spanned(), cutAtTop()];
  assert.equal(runOf(seamJoinFor(list, 0, undefined, MORE)), "wait");
  // Top of the document: join what we hold, because nothing else is coming.
  assert.deepEqual(runOf(seamJoinFor(list, 0, undefined, ENDED)), { head: 0, length: 2 });
  assert.ok(seamJoinFor(list, 0, undefined, ENDED).plan);
});

test("a name running off the bottom waits only while a picture is still coming", () => {
  const list = [cutAtBottom(), spanned()];
  assert.equal(runOf(seamJoinFor(list, 0, undefined, MORE)), "wait");
  assert.deepEqual(runOf(seamJoinFor(list, 0, undefined, ENDED)), { head: 0, length: 2 });
});

test("a boundary outside the strip is nothing to join, and never a wait", () => {
  for (const call of [
    () => seamJoinFor([cutAtBottom(), contained()], 1, undefined, ENDED),
    () => seamJoinFor([cutAtBottom(), contained()], -1, undefined, ENDED),
    () => seamJoinFor([], 0, undefined, ENDED),
    () => seamJoinFor(null, 0, undefined, ENDED),
  ]) {
    assert.equal(runOf(call()), "none");
  }
});

/* THE REGRESSION AT THE HEAD, in its real numbers.
 *
 * Slices `114/115/116` of the measured chapter: `115` is crossed end to end,
 * so a run anchors at `114` -- and `114` has NO box at its own bottom edge, so
 * the cut `114|115` is refused "one-sided" and the three-slice run plans null.
 * The pairwise seam had always joined `115|116`. Widening the search must never
 * take a join away. */
const realHeadWithNothingAtItsEdge = () =>
  seamEdges([region(300, 300, 200, 200)], 1200, 907);
const realSpanned = () => seamEdges([region(0, 7, 433, 901)], 1200, 908);
const realTail = () => seamEdges([region(0, 4, 407, 780)], 1200, 908);

test("a refused cut at the HEAD falls back rather than losing the join below it", () => {
  const list = [realHeadWithNothingAtItsEdge(), realSpanned(), realTail()];
  // The premise: 115 is spanned, 114 cannot be joined to it, and the longest run
  // therefore plans null.
  assert.equal(seamSpansSlice(list[1]), true);
  assert.equal(seamCandidates(list[0], list[1]).refused, "one-sided");
  assert.equal(seamRunPlan(list, 0, 3), null);
  // The pairwise join that must survive, and does.
  assert.ok(seamPlan(list[1], list[2]));
  assert.deepEqual(runOf(seamJoinFor(list, 1, undefined, ENDED)), { head: 1, length: 2 });
  // Entered at 114|115 there is genuinely nothing to join, and it says so rather
  // than borrowing the join below it.
  assert.equal(runOf(seamJoinFor(list, 0, undefined, ENDED)), "none");
});

/* THE SAME DEFECT AT THE OTHER END, which the first fix did not cover: it gave
 * the head back one slice at a time and had no equivalent at the tail, so a
 * refusal at the LAST cut still nulled the whole run -- and `runSeam` records a
 * null as settled, permanently discarding a boundary the pairwise seam joined.
 * Searching longest-first covers both ends and the middle with one rule. */
test("a refused cut at the TAIL falls back rather than losing the join above it", () => {
  const badTail = () => seamEdges([region(280, 300, 240, 200)], W, H);
  const list = [cutTop(), spannedWide(), badTail()];
  assert.equal(seamCandidates(list[1], list[2]).refused, "one-sided");
  assert.equal(seamRunPlan(list, 0, 3), null);
  // The join the pairwise seam made, and which must not be lost.
  assert.ok(seamPlan(list[0], list[1]));
  assert.deepEqual(runOf(seamJoinFor(list, 0, undefined, ENDED)), { head: 0, length: 2 });
  // And the boundary that genuinely cannot be joined still says so.
  assert.equal(runOf(seamJoinFor(list, 1, undefined, ENDED)), "none");
});

test("a joinable run is not given back, so two entries agree on one join", () => {
  const list = [cutAtBottom(), spanned(), cutAtTop()];
  assert.deepEqual(runOf(seamJoinFor(list, 0, undefined, ENDED)), { head: 0, length: 3 });
  assert.deepEqual(runOf(seamJoinFor(list, 1, undefined, ENDED)), { head: 0, length: 3 });
});

test("a run and its neighbour are separate joins, not one long one", () => {
  // Two two-slice boundaries in a row: each plans itself and neither swallows
  // the other, because no slice between them is crossed end to end.
  const list = [cutTop(), cutBottom(), cutTop(), cutBottom()];
  assert.deepEqual(runOf(seamJoinFor(list, 0, undefined, ENDED)), { head: 0, length: 2 });
  assert.deepEqual(runOf(seamJoinFor(list, 2, undefined, ENDED)), { head: 2, length: 2 });
  // And the boundary between the two, where nothing crosses, is not a join.
  assert.equal(runOf(seamJoinFor(list, 1, undefined, ENDED)), "none");
});

test("the run cap bounds the search, and bounds it around the boundary asked about", () => {
  // Every slice crossed end to end: without the cap one join would swallow the
  // chapter. The run must still contain the boundary it was called for.
  const list = Array.from({ length: 20 }, spanned);
  for (const index of [0, 9, 18]) {
    const found = seamJoinFor(list, index, 3, ENDED);
    if (found.wait || !found.plan) continue;
    assert.ok(found.length <= 3, `run of ${found.length} exceeded the cap of 3`);
    assert.ok(found.head <= index, "the run starts at or above the boundary");
    assert.ok(found.head + found.length - 1 >= index + 1, "and ends at or below it");
  }
});

/* ------------------------------------------------- planning a run, not a pair
 *
 * Everything above this line tests run DETECTION, which shipped tested and
 * unwired. These test the join itself: the crop a run produces, the bound that
 * would have forbidden every run there is, and the paint-back across four
 * slices. `115-118` is the four-slice case they are built against.
 */

/* A slice cut only at its TOP -- where a run ends. `contained()` is not this:
 * its region touches neither band, so `seamCandidates` refuses it "one-sided"
 * and it can end a run's DETECTION without being able to end its JOIN. */
const cutAtTop = () => seamEdges([region(10, 0, 100, 100)], W, H);
/* `115-118`: cut at the bottom of the head, two slices crossed end to end, and
 * the name finally stopping inside the fourth. */
const fourSliceRun = () => [cutAtBottom(), spanned(), spanned(), cutAtTop()];

/* NOT `deepEqual(seamRunPlan([a,b],0,2), seamPlan(a,b))`, which is what this
 * used to be and could not fail: `seamPlan`'s whole body IS that call, so the
 * two sides are one expression written twice. It looked like the guard against
 * the two-slice regression and was worth nothing.
 *
 * The numbers below are the pairwise plan's, taken from the fixture the rest of
 * this file uses: a balloon whose fit box runs 1080..1280 on the upper slice and
 * 0..120 on the lower. They are what the OBSERVED-working join produces, and
 * they are asserted against the arithmetic rather than against itself. */
test("a run of two is the pairwise crop, in the numbers the pairwise seam cut", () => {
  const plan = seamRunPlan([cutTop(), cutBottom()], 0, 2);
  assert.ok(plan);
  assert.equal(plan.slices.length, 2);
  assert.equal(plan.boundaries.length, 1);
  assert.equal(plan.width, W);

  // Grown to one slice's own shape, taken from both sides where both have room.
  assert.equal(plan.height, H);
  assert.equal(head(plan).y + head(plan).height, H, "the head's band runs to its foot");
  assert.equal(tail(plan).y, 0, "the tail's band starts at its head");
  assert.equal(cut(plan), head(plan).height);
  assert.equal(head(plan).offset, 0);
  assert.equal(tail(plan).offset, head(plan).height);

  // Both halves of the balloon, with the margin, are inside the crop.
  assert.ok(head(plan).y <= 1080 - SEAM_MARGIN_PX);
  assert.ok(tail(plan).height >= 120 + SEAM_MARGIN_PX);
  assert.equal(plan.pairs.length, 1);
  assert.equal(plan.pairs[0].cut, 0);
});

// Synthetic too -- see the note above `seamRunLength`'s four-slice test.
test("a four-slice run plans four bands and three cuts", () => {
  const list = fourSliceRun();
  assert.equal(seamRunLength(list, 0), 4);
  const plan = seamRunPlan(list, 0, 4);
  assert.ok(plan);
  assert.equal(plan.slices.length, 4);
  assert.equal(plan.boundaries.length, 3);

  // The bands tile the seam with no gap and no overlap, which is what makes a
  // paint-back offset arithmetic rather than a search.
  let offset = 0;
  for (const band of plan.slices) {
    assert.equal(band.offset, offset);
    offset += band.height;
  }
  assert.equal(offset, plan.height);
  assert.deepEqual(
    plan.boundaries,
    plan.slices.slice(1).map((band) => band.offset)
  );
});

test("a slice the run passes THROUGH is taken whole, because no shorter band holds the text", () => {
  const plan = seamRunPlan(fourSliceRun(), 0, 4);
  for (const index of [1, 2]) {
    assert.equal(plan.slices[index].y, 0);
    assert.equal(plan.slices[index].height, H);
    assert.equal(plan.slices[index].height, plan.slices[index].pageHeight);
  }
  // Only the ends are cropped.
  assert.ok(plan.slices[0].height < H);
  assert.ok(plan.slices[3].height < H);
});

/* THE BOUND THAT WOULD HAVE FORBIDDEN EVERY RUN THERE IS.
 *
 * `SEAM_MAX_ASPECT` is 2.0 and refuses a crop taller than twice the width, as a
 * "this is a whole panel, not a bubble" test. Three 1280px slices of an 800px
 * strip is 4.8x the width -- so a FLAT bound does not merely fail to help a run,
 * it rejects the shortest run that exists. Scaled per cut it is exactly the old
 * constant at one cut, which the two-slice tests above still pin. */
test("the aspect bound scales with the cuts, or no run could ever be planned", () => {
  const plan = seamRunPlan(fourSliceRun(), 0, 4);
  assert.ok(plan);
  assert.ok(plan.height > Math.round(W * SEAM_MAX_ASPECT), "a flat bound would refuse this");
  assert.ok(plan.height <= Math.round(W * SEAM_MAX_ASPECT * 3));
});

test("every cut in a run has to be a join in its own right", () => {
  // A width change at the LAST cut, which no pairwise test of the first two
  // slices would ever see.
  const list = fourSliceRun();
  list[3] = seamEdges([region(10, 0, 100, 100)], W - 200, H);
  assert.equal(seamRunPlan(list, 0, 4), null);

  // And at the first, where the head has nothing at its own bottom edge.
  const headless = fourSliceRun();
  headless[0] = contained();
  assert.equal(seamRunPlan(headless, 0, 4), null);
});

test("a run is refused rather than truncated when it does not fit the list", () => {
  const list = fourSliceRun();
  assert.equal(seamRunPlan(list, 0, 5), null);
  assert.equal(seamRunPlan(list, 2, 4), null);
  assert.equal(seamRunPlan(list, -1, 2), null);
  assert.equal(seamRunPlan(list, 0, 1), null);
  assert.equal(seamRunPlan(list, 0, SEAM_MAX_RUN_SLICES + 1), null);
  assert.equal(seamRunPlan(null, 0, 2), null);
});

test("one rectangle paints back into all four slices, each at its own offset", () => {
  const plan = seamRunPlan(fourSliceRun(), 0, 4);
  /* The seam's own read of the name, spanning all three cuts -- the region the
   * paint-back exists to deliver. This used to pass `[]`, and the part gate
   * rightly makes a composite that read nothing paint nothing now; the offset
   * arithmetic this test pins needs a lettered region to ride on. */
  const first = plan.boundaries[0];
  const last = plan.boundaries[plan.boundaries.length - 1];
  const name = region(10, first - 40, 100, last - first + 80, [10, first - 40, 100, last - first + 80]);
  const rects = seamCompositeRects(plan, [name]);
  assert.equal(rects.length, 1, "the three pairs of one column merge into one rectangle");

  const parts = rects[0].parts;
  assert.deepEqual(
    parts.map((part) => part.slice),
    [0, 1, 2, 3],
    "a name crossing three cuts touches every slice of the run"
  );

  for (const part of parts) {
    const band = plan.slices[part.slice];
    // Read from inside the seam...
    assert.ok(part.source.y >= band.offset);
    assert.ok(part.source.y + part.source.height <= band.offset + band.height);
    // ...written inside the slice it came from, at the row it was cut from.
    assert.equal(part.target.y, band.y + (part.source.y - band.offset));
    assert.ok(part.target.y >= 0);
    assert.ok(part.target.y + part.target.height <= band.pageHeight);
    assert.equal(part.source.height, part.target.height);
    assert.equal(part.source.width, part.target.width);
  }

  // The two slices the run passes through are repainted end to end -- they hold
  // nothing but the middle of the name.
  for (const index of [1, 2]) {
    const part = parts.find((one) => one.slice === index);
    assert.equal(part.target.y, 0);
    assert.equal(part.target.height, H);
  }
});

/* THE ASSERTION HERE IS THE EXACT EDGE, not "it got wider", and that is the
 * whole test. A region straddling a cut can reach the rectangle by two routes:
 * the `spanning` filter, which unions the RAW box, and the neighbour sweep,
 * which unions the PADDED box. If the boundaries test were still the pairwise
 * single-`plan.boundary` one, this region would miss `spanning` and be swept up
 * instead -- producing a rectangle a whole margin WIDER. So "wider" passes
 * either way, and only the precise edge separates them. */
test("the seam's own spanning region is found across ANY cut, not just the first", () => {
  const plan = seamRunPlan(fourSliceRun(), 0, 4);
  // A region straddling the LAST cut only.
  const late = plan.boundaries[2];
  const box = [100, late - 80, 300, 200];
  assert.ok(box[1] < late && box[1] + box[3] > late, "the fixture must straddle the cut");
  const rect = seamCompositeRects(plan, [region(...box, box)])[0].rect;

  // Reached through `spanning`: the raw box, then ONE margin.
  assert.equal(rect.x + rect.width, box[0] + box[2] + SEAM_MARGIN_PX);
  // Not through the sweep, which would have padded it twice.
  assert.notEqual(rect.x + rect.width, box[0] + box[2] + 2 * SEAM_MARGIN_PX);
});

/* A run does NOT get a bigger licence to repaint, and the number is worth
 * saying out loud because it is easy to assume otherwise: the cap is a fraction
 * of the SEAM's area, and a four-slice seam is four times the area, so it looks
 * more permissive. It is not, because the rectangle grows with it. A column of
 * type down the side of a run is cheap -- tall but narrow -- while a wide band
 * that reaches across the artwork is refused exactly as it is on a pair. */
test("a run that would repaint most of its own composite is abandoned too", () => {
  const plan = seamRunPlan(fourSliceRun(), 0, 4);
  const late = plan.boundaries[2];
  assert.deepEqual(
    seamCompositeRects(plan, [region(100, late - 80, 600, 200, [100, late - 80, 600, 200])]),
    []
  );
});

/* --------------------------------------------- which edge a join actually fixed
 *
 * Read by two callers that must agree: the content script's in-memory `seamed`
 * list, which decides whether to ask again this page view, and the cache entry's
 * stored one, which decides it after a reload. */

test("the head is repaired at its bottom and the tail at its top, never the band it gave", () => {
  assert.deepEqual(seamRunEdges(0, 2), ["bottom"]);
  assert.deepEqual(seamRunEdges(1, 2), ["top"]);
});

test("a slice the run passes through is repaired at BOTH edges", () => {
  assert.deepEqual(seamRunEdges(0, 4), ["bottom"]);
  assert.deepEqual(seamRunEdges(1, 4), ["top", "bottom"]);
  assert.deepEqual(seamRunEdges(2, 4), ["top", "bottom"]);
  assert.deepEqual(seamRunEdges(3, 4), ["top"]);
});

/* --------------------------------------------------------------- EDGE HINTS
 *
 * A box the server found at a page edge and REFUSED as a region for scoring
 * under its own text floor, reported on its own channel (`edge_hints` on the
 * `format=json` reply) because the seam is the only thing that can use it.
 *
 * It exists because a column can be real and still score under the floor.
 * Slice `113` of a measured chapter holds the top of a five-slice display
 * column: the best box scores **0.2363** against a 0.25 floor and matches the
 * measured ink to ~2 px, so today it is never reported at all and the join
 * starts at `114` -- losing a
 * whole glyph of the name, whose opening character runs composite y 1622..1810
 * against a `113|114` boundary at 1816.
 *
 * These tests assert on `seamJoinFor`, which is what `runSeam` in content.js
 * actually calls. Asserting on `seamEdges` alone would test one half of the
 * wiring and none of the join it exists to produce -- this repo has already
 * shipped a fix whose two halves were both tested while the predicate between
 * them was never called, 128 green and zero behaviour.
 */

/* The wire shape, exactly: four numbers and the edge the server measured it
 * against. No label, no `fit_*` -- there is no balloon behind a box the detector
 * refused, and it carries no class because it is never lettered. */
const hintBox = (x, y, width, height, edge) => ({ x, y, width, height, edge });

/* `[y, height]` per band, which is the plan's whole geometry in the form the
 * paint-back uses. */
const bandsOf = (plan) => plan.slices.map((band) => [band.y, band.height]);

/* Slices `113-117`, in the numbers measured on a real GPU run.
 *
 * The column's ink runs from `113` local y714 to `117` local y207 -- 3124 px
 * across five slices. `113` is the HINT (0.2363, below the floor, bottom band
 * only); `114` and `116` are the boxes the ink-walk repair grows to their own
 * slice edge; `115` already spans; `117` is correct as detected. */
const COLUMN_HINT_113 = hintBox(52.1, 712.9, 191.7, 191.6, "bottom");

/* Slice `114` as the pipeline can really produce it: a SUB-FLOOR box, grown by the
 * ink walk and flagged `spans`, never an admitted region.
 *
 * IT USED TO BE A REGION HERE AND THAT WAS THE FIXTURE'S ONE LIE. A later check
 * refuted a four-slice figure measured against it, because `settle_detections` drops
 * sub-floor boxes before the repair loop and `114` has no admitted box at any score.
 * The numbers survived the correction unchanged and are now measured rather than
 * modelled -- on a GPU run, the raw box is
 * `(52.1, 3.5) 198.6 x 577.5` at **0.2295**, and the walk grows it to the foot of a
 * 907 px slice, i.e. height 903.5. What was wrong was the KIND, not the extent.
 *
 * Keeping it a hint is also what makes the tests below exercise the real path: the
 * run only forms if `spans` actually admits the box to both edge lists. */
const COLUMN_HINT_114 = { ...hintBox(52.1, 3.5, 198.6, 903.5, "top"), spans: true };

const columnRun = (hints113) => [
  seamEdges([], 1200, 908, hints113),
  seamEdges([], 1200, 907, [COLUMN_HINT_114]),
  seamEdges([region(64.5, 7.1, 196.8, 900.9)], 1200, 908),
  seamEdges([region(76.8, 3.5, 192.7, 904.5)], 1200, 908),
  seamEdges([region(167.6, 0, 35.1, 207.5)], 1200, 908),
];

/* THE SAME FIVE SLICES AS THEY ARE TODAY, which is the regression guard.
 *
 * `113` and `114` report no column box at all -- every candidate scores under
 * the floor -- and `116`'s box stops 124 px above its own ink, touching the top
 * band only. Called with THREE arguments, exactly as every caller written before
 * hints existed calls it, and as a cache entry written before this change reads
 * back. What it plans is the join that ships now: `115|116`, 1200x1716. */
const shippedRun = () => [
  seamEdges([], 1200, 908),
  seamEdges([], 1200, 907),
  seamEdges([region(64.5, 7.1, 196.8, 900.9)], 1200, 908),
  seamEdges([region(76.8, 3.5, 192.7, 780.4)], 1200, 908),
  seamEdges([region(167.6, 0, 35.1, 207.5)], 1200, 908),
];

test("a hint is filed by the same band test the regions are, and by nothing else", () => {
  // The real hint, against the real page height: bottom band only.
  const at = seamEdges([], 1200, 908, [COLUMN_HINT_113]);
  assert.equal(at.bottom.length, 1);
  assert.equal(at.top.length, 0);
  /* `hint: true` rides on the box and nothing else does. It is what
   * `seamRunPlan` sorts on so a hint cannot take a partner a region needed, and
   * what `seamRegionSpansSlice` reads so a refused box cannot defer a join
   * forever. Asserted in the deep-equal rather than checked loosely, because the
   * two rules that depend on it fail SILENTLY if the marker ever stops being
   * set: the sort would simply treat every box as a region again. */
  assert.deepEqual(at.bottom[0], {
    x: 52.1,
    y: 712.9,
    width: 191.7,
    height: 191.6,
    hint: true,
  });

  /* The `edge` field is NOT what routes it. A hint the server labelled "top"
   * while its geometry sits at the foot of the page is filed where the geometry
   * says, because that is the box the rest of this file does arithmetic with. */
  const lying = seamEdges([], 1200, 908, [{ ...COLUMN_HINT_113, edge: "top" }]);
  assert.equal(lying.bottom.length, 1);
  assert.equal(lying.top.length, 0);

  // And nothing new appears on the summary for anything else to consume.
  assert.deepEqual(Object.keys(at).sort(), ["bottom", "height", "top", "width"]);
});

test("a hint away from either edge is not evidence of anything, and is dropped", () => {
  const middle = seamEdges([], 1200, 908, [hintBox(52.1, 300, 191.7, 191.6, "bottom")]);
  assert.equal(middle.top.length, 0);
  assert.equal(middle.bottom.length, 0);

  // Nor is one the network mangled: the same finite-number checks the regions get.
  const junk = seamEdges([], 1200, 908, [
    { x: 1, y: "900", width: 10, height: 10, edge: "bottom" },
    { x: 1, y: 900, width: 0, height: 10, edge: "bottom" },
    { x: 1, y: 900, width: 10, height: NaN, edge: "bottom" },
    null,
  ]);
  assert.equal(junk.bottom.length, 0);
});

/* A HINT MUST NOT BE ABLE TO DECLARE A SLICE CROSSED END TO END.
 *
 * `seamSpansSlice` tests IDENTITY against the two lists, so one object in both
 * is what says "this slice is spanned" -- and that decides how many slices a run
 * covers, which is the largest lever in this file. The server only reports a
 * hint touching exactly one band; honouring that here is what keeps a sub-floor
 * fragment away from the lever. */
test("a hint touching both bands is dropped rather than spanning the slice", () => {
  const both = seamEdges([], 1200, 908, [hintBox(10, 0, 100, 908, "top")]);
  assert.equal(both.top.length, 0);
  assert.equal(both.bottom.length, 0);
  assert.equal(seamSpansSlice(both), false);
  // A REGION of the same geometry still spans, so this is the hint rule and not
  // a change to the band test.
  assert.equal(seamSpansSlice(seamEdges([region(10, 0, 100, 908)], 1200, 908)), true);
});

/* FIX (a): the server can ATTEST to a crossing, and only then.
 *
 * The rule above is unchanged -- a hint that merely happens to touch both bands is
 * still dropped. What `spans` adds is a narrower door: the server walked THIS box's
 * own ink from one band to the other, which is a judgement only it can make because
 * the walk reads pixels and this file never sees one. */
test("an ink-verified spanning hint enters BOTH lists as one object", () => {
  const spanning = { ...hintBox(10, 0, 100, 908, "top"), spans: true };
  const edges = seamEdges([], 1200, 908, [spanning]);

  assert.equal(edges.top.length, 1);
  assert.equal(edges.bottom.length, 1);
  // ONE object, not two equal ones. `seamSpansSlice` is a strict `===`, so a copy
  // in each list would read as two unrelated boxes and the run would stop here.
  assert.ok(edges.top[0] === edges.bottom[0], "both lists must hold the SAME object");
  assert.equal(seamSpansSlice(edges), true);
  assert.equal(edges.top[0].hint, true, "it is still a hint, and still marked one");

  // The control that makes this about `spans` and nothing else: identical geometry,
  // flag absent, still dropped.
  assert.equal(seamSpansSlice(seamEdges([], 1200, 908, [hintBox(10, 0, 100, 908, "top")])), false);
});

test("the spans flag cannot override the geometry", () => {
  /* A server that says `spans` about a box which plainly does not reach both bands
   * is contradicting itself, and the geometry is what the rest of this file does
   * arithmetic with. Filed by the band test like any other one-sided hint -- never
   * pushed into both lists on the strength of the flag alone. */
  const lying = { ...hintBox(10, 0, 100, 200, "top"), spans: true };
  const edges = seamEdges([], 1200, 908, [lying]);
  assert.equal(edges.top.length, 1);
  assert.equal(edges.bottom.length, 0);
  assert.equal(seamSpansSlice(edges), false);
});

/* AMENDMENT 1 -- A HINT MUST NEVER TAKE A PARTNER A REGION NEEDED.
 *
 * A grown spanning hint is flush with both page edges, so its `edgeDistance` is 0 at
 * both cuts while a real balloon sitting inside the band is charged for the gap. On
 * score alone the hint wins the greedy 1:1 match nearly always and the join is then
 * planned on the hint's geometry instead of the balloon's.
 *
 * Measured need: slice `115` of the measured chapter returns the column as a REGION *and* a
 * sub-floor hint of the same ink, so two objects really do compete at one cut. */
test("a hint never takes a partner a region needed", () => {
  // Both sit in the bottom band and both overlap the box below. The hint is flush
  // with the page edge (edgeDistance 0); the region is 8 px inside it.
  const upper = seamEdges([region(100, 800, 100, 100)], 1200, 908, [
    hintBox(100, 808, 100, 100, "bottom"),
  ]);
  const lower = seamEdges([region(100, 0, 100, 100)], 1200, 908);
  assert.equal(upper.bottom.length, 2, "both must be candidates, or this proves nothing");

  const plan = seamRunPlan([upper, lower], 0, 2);
  assert.ok(plan, "the boundary still plans");
  assert.equal(plan.pairs.length, 1, "one partner below, so exactly one pairing");
  assert.ok(
    !plan.pairs[0].top.hint,
    "the REGION must take the pairing; the hint scored higher and must still lose"
  );
  assert.equal(plan.pairs[0].top.y, 800);
});

/* AMENDMENT 3 -- A REFUSED BOX MUST NOT POSTPONE A JOIN FOREVER.
 *
 * `seamJoinFor` defers when the name still reaches an end of what we hold and another
 * picture could arrive. That is right for a region. For a hint it is a silent lost
 * join: if the next picture never becomes usable, the deferral is never revisited and
 * a join the pairwise seam always made simply stops happening. */
test("a hint-spanned end does not defer the join, but a region-spanned end does", () => {
  const MORE_BELOW = { above: false, below: true };

  // Last slice crossed by a HINT: joinable now.
  const byHint = [
    seamEdges([region(100, 800, 100, 108)], 1200, 908),
    seamEdges([], 1200, 908, [{ ...hintBox(100, 0, 100, 908, "top"), spans: true }]),
  ];
  const hinted = seamJoinFor(byHint, 0, undefined, MORE_BELOW);
  assert.equal(hinted.wait, false, "a refused box must not hold the join open");
  assert.ok(hinted.plan, "and the join it was blocking is planned");

  // Same shape with a REGION doing the spanning: still deferred, unchanged.
  const byRegion = [
    seamEdges([region(100, 800, 100, 108)], 1200, 908),
    seamEdges([region(100, 0, 100, 908)], 1200, 908),
  ];
  assert.equal(seamJoinFor(byRegion, 0, undefined, MORE_BELOW).wait, true);
});

/* THE TARGET, end to end, through the predicate `runSeam` calls.
 *
 * With `113` present only as a hint and `114`/`116` grown by the ink walk, the
 * join runs `113-116`: composite y 1596..4539 against a first glyph at 1622, so
 * the name starts 26 px inside the crop instead of a glyph outside it. */
test("the sub-floor box at 113 is what makes the column a four-slice join", () => {
  const list = columnRun([COLUMN_HINT_113]);
  const found = seamJoinFor(list, 0, undefined, ENDED);
  assert.deepEqual(runOf(found), { head: 0, length: 4 });
  assert.equal(found.plan.width, 1200);
  assert.equal(found.plan.height, 2943);
  assert.deepEqual(bandsOf(found.plan), [
    [688, 220],
    [0, 907],
    [0, 908],
    [0, 908],
  ]);
  // The head's band starts 24 px above the hint, which is SEAM_MARGIN_PX: the
  // crop is anchored on the hint and on nothing else.
  assert.equal(found.plan.slices[0].y, Math.floor(712.9 - SEAM_MARGIN_PX));

  /* `116|117` is refused, and that is correct rather than a shortfall. `116` is
   * 192.7 px wide against `117`'s 35.1 -- ratio 5.5 over a 2.2 ceiling -- and
   * all twelve glyphs and both interpuncts are inside `113-116`; only a trailing
   * em-dash is left on `117`. Relaxing the width veto to chase it would reopen
   * the measured false join this file's `SEAM_SFX_LABEL` comment records. */
  assert.equal(seamCandidates(list[3], list[4]).refused, "no-pair");

  // Entered from any boundary inside the run, the same join -- so the caller's
  // "already asked" key matches and the column is translated once.
  for (const index of [1, 2]) {
    assert.deepEqual(runOf(seamJoinFor(list, index, undefined, ENDED)), { head: 0, length: 4 });
  }
});

/* A LONGER RUN MUST NEVER COST A SHORTER JOIN, and this is that rule for a
 * server or a cache entry that has no hints at all.
 *
 * A stored entry written before this change carries `edges` with no hint in it,
 * and an older server sends no `edge_hints` field -- both arrive here as
 * `seamEdges(regions, width, height)` with three arguments. What they plan has
 * to be pixel-for-pixel what ships today, or the compatibility story is that the
 * feature quietly deletes a join for 24 hours. */
test("the same five slices with no hints plan exactly the join that ships today", () => {
  const list = shippedRun();
  const found = seamJoinFor(list, 2, undefined, ENDED);
  assert.deepEqual(runOf(found), { head: 2, length: 2 });
  assert.equal(found.plan.width, 1200);
  assert.equal(found.plan.height, 1716, "115 whole plus 116 rows 0..808");
  assert.deepEqual(bandsOf(found.plan), [
    [0, 908],
    [0, 808],
  ]);

  // And the boundary the hint would have opened is still nothing to join, rather
  // than a crash or a wait that never resolves.
  assert.equal(runOf(seamJoinFor(list, 0, undefined, ENDED)), "none");
});

/* The same fixture with the hint taken away, which separates the two mechanisms
 * (the edge hint and the ink walk) and says exactly what each one buys.
 *
 * The ink walk alone still reaches `114-116` -- a real join, 2723 px, and one
 * the pairwise seam never made. What the hint adds is the HEAD, and only the
 * head: `113|114` has nothing at `113`'s bottom edge without it, so no run
 * containing that boundary can plan and entering there is "none". The column is
 * still joined, one slice shorter, from the boundaries inside it. */
test("without the hint the same grown boxes plan a shorter run, and never a longer one", () => {
  const bare = columnRun(undefined);
  // The boundary the hint exists for: unjoinable, and settled rather than a wait.
  assert.equal(runOf(seamJoinFor(bare, 0, undefined, ENDED)), "none");
  assert.equal(seamCandidates(bare[0], bare[1]).refused, "one-sided");

  const found = seamJoinFor(bare, 1, undefined, ENDED);
  assert.deepEqual(runOf(found), { head: 1, length: 3 });
  assert.equal(found.plan.height, 2723, "220 px of head short of the four-slice join");

  // null and a non-array are the same answer as omitting it entirely, which is
  // the shape a cache entry from before this change and an older server both
  // arrive in.
  for (const hints of [null, "", 0, { x: 1 }]) {
    const other = seamJoinFor(columnRun(hints), 1, undefined, ENDED);
    assert.deepEqual(runOf(other), runOf(found));
    assert.deepEqual(bandsOf(other.plan), bandsOf(found.plan));
  }
});

/* A hint that is not at an edge is not evidence, so it cannot move a join. The
 * assertion is against the WHOLE plan of the same fixture with no hints at all
 * -- entered at a boundary that genuinely plans, so this compares two real plans
 * rather than two nulls, which is the way this test first passed for nothing. */
test("a hint away from the edge changes no join at all", () => {
  const bare = seamJoinFor(columnRun(undefined), 1, undefined, ENDED);
  const decoy = seamJoinFor(
    columnRun([hintBox(52.1, 300, 191.7, 191.6, "bottom")]),
    1,
    undefined,
    ENDED
  );
  assert.ok(bare.plan, "the fixture must plan, or this compares two nulls");
  assert.deepEqual(runOf(decoy), runOf(bare));
  assert.deepEqual(decoy.plan, bare.plan);
});

/* An ordinary page carries hints too -- the server reports every sub-floor box
 * at an edge, on every page, and 179 slices of one chapter produced 9. A page with
 * nothing to join must still have nothing to join. */
test("a hint at an edge with no partner below it is still not a join", () => {
  const list = [
    seamEdges([region(300, 300, 200, 200)], W, H, [hintBox(300, H - 60, 200, 60, "bottom")]),
    seamEdges([region(300, 400, 200, 200)], W, H),
  ];
  assert.equal(list[0].bottom.length, 1, "the hint is there to be refused");
  assert.equal(runOf(seamJoinFor(list, 0, undefined, ENDED)), "none");
});

/* ------------------------------------------------------------------------- *
 * THE RE-ASK: a boundary that DEFERRED is asked again when a later slice
 * arrives.
 *
 * The browser half rather than the geometry, and it is here rather than in a new
 * file because it is the same defect `seamJoinFor`'s `wait` return is about.
 *
 * WHAT IT GUARDS. `seamJoinFor` returns `{wait: true}` when a name runs past
 * what is translated. `runSeam` then returns without settling the boundary --
 * correctly -- but `scheduleSeam` only ever asked the two boundaries TOUCHING
 * the slice that just landed, so the deferred one was never revisited. Measured
 * in a real Firefox over slices `111`-`119` of one chapter, both passes identical:
 * `114` and `115` logged `wait`, `seamRunPlan(head=113, length=4)` was
 * constructible on that same data, and the four-slice join never happened. The
 * reader got BLUE FLAME KING and two more names left in Chinese.
 *
 * IT DRIVES `scheduleSeam`, WHICH IS THE FUNCTION THE PRODUCT CALLS. An earlier
 * draft of this block drove `retryDeferredNear` directly and passed with the
 * call site deleted -- the same shape as a fix that once shipped with 128 green
 * tests and zero behaviour, found here by disarming the fix and watching
 * nothing go red.
 * ------------------------------------------------------------------------- */

const fs = require("node:fs");
const path = require("node:path");
const CONTENT_JS = fs.readFileSync(
  path.join(__dirname, "..", "extension", "content.js"),
  "utf8"
);

/* Lifted out of the SHIPPED source, so this tests what runs rather than a copy.
 * All three functions come across together deliberately: the drift they are
 * written to prevent is between them. */
function shippedScheduleSeam(neighbours, deferredPairs, settings) {
  const start = CONTENT_JS.indexOf("function scheduleSeam(img)");
  assert.ok(start > 0, "could not find scheduleSeam() in content.js");
  const keyAt = CONTENT_JS.indexOf("function seamBoundaryKey(upper, lower)");
  assert.ok(keyAt > start, "could not find seamBoundaryKey() after it");
  const source = CONTENT_JS.slice(start, CONTENT_JS.indexOf("\n}", keyAt) + 2);

  const asked = [];
  const sliced = new Map();
  for (const name of Object.keys(neighbours)) sliced.set(name, { url: name });
  const seamDeferred = new Map();
  for (const [a, b] of deferredPairs) seamDeferred.set(`${a}\n${b}`, [a, b]);

  const build = new Function(
    "settings",
    "location",
    "seamDeferred",
    "sliced",
    "sliceNeighbour",
    "SEAM_MAX_RUN_SLICES",
    "runSeam",
    `${source}; return scheduleSeam;`
  );
  const scheduleSeam = build(
    // The fold of `joinSlices`: the gate is now the page host's
    // profile, so the harness owns a hostname and callers may override the map.
    settings || { profileByHost: {} },
    { hostname: "test.example" },
    seamDeferred,
    sliced,
    (img, below) => neighbours[img][below ? 1 : 0] || null,
    6,
    (a, b) => asked.push(`${a}\n${b}`)
  );
  /* The two boundaries an arrival TOUCHES are asked by the shipped code and are
   * not what this block is about; dropping them keeps each assertion about the
   * re-ask alone. */
  const retries = (img) => {
    asked.length = 0;
    scheduleSeam(img);
    const touching = new Set([
      `${neighbours[img][0]}\n${img}`,
      `${img}\n${neighbours[img][1]}`,
    ]);
    return asked.filter((k) => !touching.has(k));
  };
  return { scheduleSeam, retries, asked };
}

/* Nine slices in a row -- exactly the fixture the browser run used. */
const STRIP = {};
for (let i = 111; i <= 119; i += 1) {
  STRIP[String(i)] = [i > 111 ? String(i - 1) : null, i < 119 ? String(i + 1) : null];
}

test("a slice that lands re-asks the boundary that deferred above it", () => {
  /* `115`-`116` deferred waiting for a slice below. `117` is what arrives.
   * Nothing connected those two, which is why the four-slice run was never
   * built on a real strip. */
  const { retries } = shippedScheduleSeam(STRIP, [["115", "116"]]);
  assert.deepEqual(retries("117"), ["115\n116"]);
});

test("a deferred boundary further away than a run can reach is left alone", () => {
  /* The window is SEAM_MAX_RUN_SLICES for a reason rather than for tidiness: a
   * run cannot be longer than that, so an arrival cannot have changed the answer
   * at a boundary further off. At a cap of 6, `111`-`112` is out of reach from
   * `119`. */
  const { retries } = shippedScheduleSeam(STRIP, [["111", "112"]]);
  assert.deepEqual(retries("119"), [], "an unreachable boundary must not be re-asked");
});

test("a deferral BELOW the arrival is re-asked too", () => {
  /* `sliceChain` reports `pending.above` as well as `pending.below`, so a run
   * can be waiting on a slice ABOVE it -- a reader scrolling up, or a lazy
   * loader filling in behind. The measured case was downward; this is the
   * symmetric one, and it costs one more walk and no request. */
  const { retries } = shippedScheduleSeam(STRIP, [["117", "118"]]);
  assert.deepEqual(retries("115"), ["117\n118"]);
});

test("nothing deferred means nothing is walked", () => {
  const { retries } = shippedScheduleSeam(STRIP, []);
  assert.deepEqual(retries("115"), [], "the common strip must cost no extra work");
});

test("the seam stays off when the site is declared manga", () => {
  /* The fold of the `joinSlices` checkbox: `scheduleSeam` returns on the page
   * host's explicit manga pick before anything else, and the retry must sit
   * behind that gate too -- a disabled seam that still walks and still posts
   * is a pick that does not mean what it says. Another host's pick must NOT
   * gate this one. */
  const on = shippedScheduleSeam(STRIP, [["115", "116"]]);
  on.scheduleSeam("117");
  assert.ok(on.asked.length > 0, "sanity: the undeclared build does ask");

  const other = shippedScheduleSeam(STRIP, [["115", "116"]], {
    profileByHost: { "other.example": "manga" },
  });
  other.scheduleSeam("117");
  assert.ok(other.asked.length > 0, "another host's manga pick must not gate this site");

  const manga = shippedScheduleSeam(STRIP, [["115", "116"]], {
    profileByHost: { "test.example": "manga" },
  });
  manga.scheduleSeam("117");
  assert.equal(manga.asked.length, 0, "a manga site must not walk or post");
});

test("the retry composes the SAME boundary key runSeam records", () => {
  /* The failure this prevents has NO SYMPTOM: a retry that spelled the key
   * differently would look up nothing, re-ask nothing and log nothing -- the
   * seam would go on failing exactly as it does today, with the fix in the tree
   * and the suite green.
   *
   * So the guard is that both sides go through ONE function. Structural, and
   * asserted on the shipped text because that is what the property is about. */
  const uses = CONTENT_JS.match(/seamBoundaryKey\(/g) || [];
  assert.ok(uses.length >= 3, "expected a definition and both call sites");
  assert.deepEqual(
    CONTENT_JS.match(/`\$\{above\.url\}\n\$\{below\.url\}`/g) || [],
    [],
    "the boundary key is spelled inline again -- it must go through seamBoundaryKey()"
  );
});

/* A refused region is not swept over, so it cannot drag the paint rectangle
 * across the composite to reach a box that carries no text.
 *
 * Measured on boundary `101|102` of one chapter: with the server's OCR area ceiling
 * fixed the join was read and lettered (`regions=2 layers=1 chars=11`), and the
 * paint-back then absorbed the OTHER region on that composite -- a site
 * watermark refused as "a site watermark, not dialogue" -- growing the rectangle
 * until `SEAM_MAX_RECT_FRACTION` abandoned the join one step from the screen.
 *
 * This does NOT on its own put that boundary under the cap; the balloon alone is
 * 0.728 of its crop. It removes one of the two reasons the rectangle was too
 * big, and it is the reason that was never intentional. */
test("a region the server refused is not swallowed by the paint rectangle", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  const bubble = region(300, cut(plan) - 100, 200, 200, [300, cut(plan) - 100, 200, 200]);
  /* OVERLAPPING the bubble's padded rectangle, or the sweep would never look at
   * it and this test would pass without exercising anything. The control at the
   * foot is what caught exactly that: a first draft placed this at the top of the
   * page, where an ADMITTED copy is not swallowed either. */
  const far = [480, cut(plan) - 60, 300, 80];
  const refused = {
    ...region(far[0], far[1], far[2], far[3], far),
    refused: "a site watermark, not dialogue",
  };

  const alone = seamCompositeRects(plan, [bubble]);
  const withRefused = seamCompositeRects(plan, [bubble, refused]);
  assert.ok(alone.length, "the bubble alone must produce a rectangle");
  assert.deepEqual(
    withRefused,
    alone,
    "a refused region must change nothing about where the join paints"
  );

  // The control that makes this about `refused` and nothing else: the SAME box
  // without the marker is swept in, and moves the rectangle.
  const { refused: _dropped, ...admitted } = refused;
  const withAdmitted = seamCompositeRects(plan, [bubble, admitted]);
  assert.notDeepEqual(
    withAdmitted,
    alone,
    "an ADMITTED region at the same place must still be swallowed, or this test "
      + "is passing for the wrong reason"
  );
});

/* An oversized rectangle that is almost entirely the joined region is painted;
 * one that is mostly artwork dragged along is still refused.
 *
 * Boundary `154|155` of the same chapter: the server read and lettered the joined sound
 * effect (`regions=1 layers=1 chars=16 refused=null`, box 1100x1394 on a
 * 1200x1384 composite) and `seamCompositeRects` then discarded it, because the
 * rectangle is essentially the whole composite. It is the whole composite because
 * the EFFECT is -- 0.92 of it -- so there was no collateral to prevent. The
 * reader got a per-slice "Rumble..." over un-erased Chinese instead.
 *
 * The two guards above keep their own cases and are the control for this one:
 * they are the low-payload end of the same measurement. */
test("an oversized rectangle is painted when it is mostly the join itself", () => {
  const plan = seamPlan(cutTop(), cutBottom());
  const composite = plan.width * plan.height;

  // Nearly the whole composite, as a real joined effect is.
  const w = Math.round(plan.width * 0.95);
  const h = Math.round(plan.height * 0.95);
  const x = Math.round((plan.width - w) / 2);
  const y = Math.round((plan.height - h) / 2);
  const effect = region(x, y, w, h, [x, y, w, h]);
  // 0.6 mirrors `SEAM_MAX_RECT_FRACTION`, which seam.js does not export. Spelled
  // rather than imported, with this line as the reason: if the constant moves,
  // this assertion is what fails and says the fixture stopped being oversized.
  assert.ok(
    w * h > composite * 0.6,
    "the fixture must be over the fraction cap, or this proves nothing"
  );
  assert.ok(
    seamCompositeRects(plan, [effect]).length,
    "a rectangle that IS the joined region has no collateral to prevent"
  );

  /* THE LOW-PAYLOAD CONTROL IS NOT WRITTEN HERE, and that is deliberate: the two
   * guards above it in this file already are it, and they still pass.
   * `a rectangle that would repaint most of the seam abandons the join instead`
   * is a region covering the whole composite -- excluded from the payload, so it
   * scores 0 and is refused. `a run that would repaint most of its own composite
   * is abandoned too` is 120,000 px inside an ~1.82M px rectangle, a payload of
   * 0.066, refused. A third fixture here would only restate them.
   *
   * A first draft did write one -- two 60x60 specks at opposite corners -- and it
   * FAILED, because specks that far from the pair's own rectangle are never swept
   * into it, so the rectangle never grew and never reached the cap at all. The
   * assertion was wrong rather than the code, which is exactly what a control is
   * for. */
});

/* -------------------------------------------------------------------------
 * The run decision from RAW slice replies, and the serializer pin.
 *
 * `seamSpansSlice` is reference identity: seamEdges puts ONE box object into
 * both of a slice's edge lists, and that shared reference is the whole "this
 * slice is crossed end to end" signal. Measured: in-process true,
 * structuredClone true, JSON FALSE -- so any caller that assembles an
 * edgesList across a JSON round trip caps every run at two slices, silently,
 * and must build its edges in-process from the raw replies instead. These
 * tests pin all three arms.
 * ------------------------------------------------------------------------- */

const RUN_W = W;
const RUN_H = H;

/* A four-slice drawn column, one region per slice, same x/width throughout:
 * head touches only its bottom band, middles run edge to edge, tail touches
 * only its top band -- the measured 113-116 column's shape at test scale. */
const rawRunSlices = () => [
  { regions: [region(300, 1100, 200, 180)], width: RUN_W, height: RUN_H, hints: [] },
  { regions: [region(300, 0, 200, RUN_H)], width: RUN_W, height: RUN_H, hints: [] },
  { regions: [region(300, 0, 200, RUN_H)], width: RUN_W, height: RUN_H, hints: [] },
  { regions: [region(300, 0, 200, 200)], width: RUN_W, height: RUN_H, hints: [] },
];

test("a four-slice chain plans one run when its edges never cross a serializer", () => {
  const edgesList = rawRunSlices().map((s) =>
    seamEdges(s.regions, s.width, s.height, s.hints)
  );
  for (const index of [0, 1, 2]) {
    const found = seamJoinFor(edgesList, index, SEAM_MAX_RUN_SLICES, {
      above: false,
      below: false,
    });
    assert.equal(found.wait, false, `boundary ${index} must not wait on a full chapter`);
    assert.ok(found.plan, `boundary ${index} must plan`);
    assert.equal(found.head, 0, `boundary ${index} must anchor at the head`);
    assert.equal(found.length, 4, `boundary ${index} must span the whole chain`);
    assert.equal(found.plan.slices.length, 4);
    assert.equal(found.plan.boundaries.length, 3);
  }
});

test("structuredClone keeps the run and a JSON round trip destroys it", () => {
  const edgesList = rawRunSlices().map((s) =>
    seamEdges(s.regions, s.width, s.height, s.hints)
  );
  const ask = (list) =>
    seamJoinFor(list, 1, SEAM_MAX_RUN_SLICES, { above: false, below: false });

  // structuredClone serializes the object GRAPH, so the shared box reference
  // survives -- this is the serializer the product's message channel relies on.
  assert.equal(ask(structuredClone(edgesList)).length, 4);

  /* JSON rebuilds the graph as a TREE: two structurally-equal boxes where one
   * shared object stood, `includes` misses, and the middles read un-spanned.
   * On this fixture the collapse is total -- not even the pair plans, because
   * the pair candidates are the same full-height boxes whose spanning signal
   * just vanished. The claim pinned here is the mechanism, not the exact
   * wreckage: the four-run is UNDETECTABLE through a JSON trip. If this arm
   * ever fails, identity stopped being the signal -- find out why before
   * "fixing" it. */
  const parsed = ask(JSON.parse(JSON.stringify(edgesList)));
  assert.notEqual(parsed.length, 4, "a JSON-tripped edgesList must lose the run");
  assert.ok(
    !parsed.plan || parsed.plan.slices.length <= 2,
    "nothing longer than a pair may survive the trip"
  );
});

/* --- THE TOUCHES TRIGGER -------------------------------------------------
 *
 * A balloon whose SHAPE is cut by the slice boundary while its TEXT is wholly
 * on one side. Driven end to end through `seamCandidates`/`seamJoinFor` -- the
 * predicates the product calls -- never through the helper alone, so the
 * wiring between the probe, the synthetic box and the ordinary join machinery
 * is what these assert (the composed-predicate rule).
 *
 * The fire fixture is a measured 058|059 boundary, numbers verbatim off the
 * stored wire: 059's single dialogue region, box
 * (178.125, 32.8, 281.25x157.8), fit (101, 0, 439, 348) -- clipped flush at
 * the canvas top edge -- role `dialogue`. The balloon's empty white sliver is
 * on 058's foot; the census measured the ten A-class lobes at median 122 px,
 * which is the depth the profile below carries. */

const touchRegion = (role, label, fit) => ({
  x: 178.125,
  y: 32.8,
  width: 281.25,
  height: 157.8,
  source: "",
  translated: "",
  ...(fit || { fit_x: 101, fit_y: 0, fit_width: 439, fit_height: 348 }),
  ...(role ? { role } : {}),
  ...(label ? { label } : {}),
});

/* A white profile whose buckets under [x0, x1) read `deep` and elsewhere 0. */
const whiteUnder = (x0, x1, deep) => {
  const step = 8;
  const buckets = Math.ceil(W / step);
  const profile = new Array(buckets).fill(0);
  for (let k = Math.floor(x0 / step); k * step < x1 && k < buckets; k += 1) profile[k] = deep;
  return { step, profile };
};

const touchUpper = (white) =>
  seamEdges([], W, H, null, white ? { top: new Array(Math.ceil(W / 8)).fill(0), bottom: white.profile, step: white.step } : undefined);

test("the TOUCHES probe joins 058|059: a flush dialogue fit plus a deep white lobe", () => {
  const lower = seamEdges([touchRegion("dialogue")], W, H);
  const upper = touchUpper(whiteUnder(101, 540, 122));
  const result = seamCandidates(upper, lower);
  assert.equal(result.refused, null, "the boundary must not refuse");
  assert.equal(result.tops.length, 1, "the empty side gains exactly one synthetic box");
  assert.equal(result.tops[0].touch, true, "and it is marked as probe evidence");
  assert.equal(result.tops[0].height, 122, "as tall as the measured lobe");
  assert.equal(result.tops[0].x, 101, "under the balloon's own columns");
  assert.equal(result.tops[0].width, 439);
  const candidate = result.candidates.find((one) => one.score !== null);
  assert.ok(candidate, "the ordinary veto chain passes the pair");

  // And through the caller the product actually calls: a two-slice run plans.
  const answer = seamJoinFor([upper, lower], 0, SEAM_MAX_RUN_SLICES);
  assert.ok(answer.plan, "seamJoinFor plans the join");
  assert.equal(answer.plan.slices.length, 2, "covering both slices");
});

test("the TOUCHES probe stays silent without pixels: no profile, no trigger", () => {
  const lower = seamEdges([touchRegion("dialogue")], W, H);
  // No `white` at all -- the flag-off arm, and every cache entry written
  // before the profile existed. Byte-identical to the old refusal.
  const upper = seamEdges([], W, H);
  const result = seamCandidates(upper, lower);
  assert.equal(result.refused, "one-sided");
  assert.deepEqual(result.counts, [0, 1]);
  assert.equal(seamJoinFor([upper, lower], 0, SEAM_MAX_RUN_SLICES).plan, null);
});

test("a shallow lobe is the harmless class and must not fire", () => {
  // 112|113's shape: the JSON looks identical, only the pixels differ -- the
  // continuation is under the 60 px gate everywhere.
  const lower = seamEdges([touchRegion("dialogue")], W, H);
  const upper = touchUpper(whiteUnder(101, 540, 20));
  assert.equal(seamCandidates(upper, lower).refused, "one-sided");
});

test("caption columns and effects are the C class and never fire", () => {
  const upper = touchUpper(whiteUnder(101, 540, 122));
  const caption = seamEdges([touchRegion("free-text")], W, H);
  assert.equal(seamCandidates(upper, caption).refused, "one-sided");
  const effect = seamEdges([touchRegion("dialogue", SEAM_SFX_LABEL)], W, H);
  assert.equal(seamCandidates(upper, effect).refused, "one-sided");
  const unroled = seamEdges([touchRegion()], W, H);
  assert.equal(seamCandidates(upper, unroled).refused, "one-sided", "an absent role stays absent");
});

test("the flush gate admits the measured near-miss and refuses a set-back box", () => {
  const upper = touchUpper(whiteUnder(101, 540, 122));
  // A measured region ends 6.2 px short of flush and is half of an A site: within 8.
  const nearMiss = seamEdges(
    [touchRegion("dialogue", null, { fit_x: 101, fit_y: 6.2, fit_width: 439, fit_height: 341.8 })],
    W,
    H
  );
  assert.equal(seamCandidates(upper, nearMiss).refused, null);
  // A box 12 px off the cut is inside the edge band (16) but not flush.
  const setBack = seamEdges(
    [touchRegion("dialogue", null, { fit_x: 101, fit_y: 12, fit_width: 439, fit_height: 336 })],
    W,
    H
  );
  assert.equal(seamCandidates(upper, setBack).refused, "one-sided");
});

test("the fraction gate needs most of the columns, not a sliver", () => {
  const lower = seamEdges([touchRegion("dialogue")], W, H);
  // Deep white under less than half the balloon's columns: a panel border or
  // background patch, not a lobe.
  const sliver = touchUpper(whiteUnder(101, 260, 122));
  assert.equal(seamCandidates(sliver, lower).refused, "one-sided");
  // Under nearly all of them: fires.
  const lobe = touchUpper(whiteUnder(101, 530, 122));
  assert.equal(seamCandidates(lobe, lower).refused, null);
});

test("a sub-floor hint never buys a TOUCHES join", () => {
  // A hint is a refused detection, not a read region; the trigger is scoped to
  // dialogue the pipeline actually lettered. Role is absent on hints anyway,
  // but the hint mark is asserted on its own so the scope survives a future
  // role-carrying hint.
  const upper = touchUpper(whiteUnder(101, 540, 122));
  const hinted = seamEdges([], W, H, [
    { x: 101, y: 0, width: 439, height: 120, edge: "top", role: "dialogue" },
  ]);
  assert.equal(hinted.top[0].hint, true, "the fixture really is a hint");
  assert.equal(seamCandidates(upper, hinted).refused, "one-sided");
});

/* --------------------------------------- the TOUCHES trigger, live-sidedness
 *
 * The trigger's one-sidedness is decided on LIVE boxes -- skipping both a box the
 * pipeline REFUSED (rendered_text empty; it never lettered anything) and a
 * sub-floor HINT (a different channel entirely: "below my floor", never a
 * read). The 163|164 exhibit is the fire shape verbatim: 163's dialogue fit
 * flush at the cut, 164's top band holding a refused orphan `...` AND four
 * hints, so the raw lists read two-sided and the shipped trigger could never
 * run. The REFUSAL still counts raw lists byte-identically (deciding it on
 * live counts deletes joins -- a first draft that did so lost some), and
 * the synthetic is APPENDED beside the refused boxes, never substituted. */

const refusedRegion = (x, y, width, height) => ({
  x,
  y,
  width,
  height,
  source: "",
  translated: "",
  fit_x: x,
  fit_y: y,
  fit_width: width,
  fit_height: height,
  role: "dialogue",
  label: "text",
  refused: "the script does not belong to the source language",
});

test("a refused fragment no longer keeps the boundary two-sided: 163|164 fires", () => {
  // The dialogue lives on the LOWER slice's top band, as in the fire fixture
  // above; the UPPER slice's bottom band holds 164's shape -- one refused
  // orphan plus a sub-floor hint, raw-populated but LIVE-empty -- and the
  // white lobe profile under the balloon's columns.
  const lower = seamEdges([touchRegion("dialogue")], W, H);
  const upper = seamEdges(
    [refusedRegion(560, H - 62, 60, 60)],
    W,
    H,
    [{ x: 620, y: H - 31, width: 40, height: 30, edge: "bottom" }],
    { top: new Array(Math.ceil(W / 8)).fill(0), bottom: whiteUnder(101, 540, 122).profile, step: 8 }
  );
  const result = seamCandidates(upper, lower);
  assert.equal(result.refused, null, "the trigger runs despite the refused fragment");
  const synthetic = result.tops.find((box) => box.touch);
  assert.ok(synthetic, "the synthetic counterpart is appended");
  assert.ok(
    result.tops.length >= 2,
    "APPENDED beside the refused box, never substituted for it"
  );
  const answer = seamJoinFor([upper, lower], 0, SEAM_MAX_RUN_SLICES);
  assert.ok(answer.plan, "and the boundary plans");
});

test("without pixels the refused-fragment boundary refuses exactly as shipped", () => {
  // Same shape, no white profile: the probe stays silent and the ORDINARY
  // path runs on the raw lists -- the fragment pairs, dies on the width veto,
  // and the boundary refuses "no-pair" byte-identically to before.
  const lower = seamEdges([touchRegion("dialogue")], W, H);
  const upper = seamEdges([refusedRegion(560, H - 62, 60, 60)], W, H, [
    { x: 620, y: H - 31, width: 40, height: 30, edge: "bottom" },
  ]);
  const result = seamCandidates(upper, lower);
  assert.equal(result.refused, "no-pair");
});

test("a refused box that anchors a genuine join still anchors it", () => {
  // The LOST=0 guard: a two-sided boundary whose far side is all refused
  // still pairs and still plans -- the ko lever's refused-but-real dialogue
  // boxes anchor genuine cuts, and deciding the REFUSAL on live counts would
  // delete them. Same columns as the near box, so the pair passes every veto.
  const lower = seamEdges([touchRegion("dialogue")], W, H);
  const upper = seamEdges([refusedRegion(101, H - 120, 439, 120)], W, H);
  const result = seamCandidates(upper, lower);
  assert.equal(result.refused, null, "the refused box still pairs");
  const answer = seamJoinFor([upper, lower], 0, SEAM_MAX_RUN_SLICES);
  assert.ok(answer.plan, "and still plans, with no white profile in sight");
});

/* ------------------------------------------- the TOUCHES paint, whole-extent
 *
 * The TOUCHES doubling, diagnosed on a deterministic 042|043 window: the
 * composite's detected box in the TOUCHES class is
 * the INK box, wholly on one side and smaller than either lettering, so the
 * part clip cut the paint to it -- the slice-local first and last lines
 * survived around the painted band (the reader saw the sentence twice), and
 * the empty lobe's band held no witness so its part was gated outright. The
 * pair boxes ARE the missing witnesses (the populated side's fit frame is the
 * local lettering's own solve extent, the synthetic box is the measured
 * lobe), scoped to plans carrying a `touch` pair so an ordinary join stays
 * byte-identical. Asserted through the chain the product calls:
 * `seamJoinFor` -> plan -> `seamCompositeRects`. */

test("a TOUCHES plan paints both bands and the populated side's whole fit", () => {
  const lower = seamEdges([touchRegion("dialogue")], W, H);
  const upper = touchUpper(whiteUnder(101, 540, 122));
  const answer = seamJoinFor([upper, lower], 0, SEAM_MAX_RUN_SLICES);
  assert.ok(answer.plan, "the fixture must plan (the 058|059 fire shape)");
  const seamCut = cut(answer.plan);
  // The composite's one read region: the joined balloon's INK box, wholly
  // below the cut and much smaller than the pair's fit frame -- the 042|043
  // shape. Effective box == box (fit equal), no spanning.
  const ink = region(178, seamCut + 40, 281, 158, [178, seamCut + 40, 281, 158]);
  const rects = seamCompositeRects(answer.plan, [ink]);
  assert.equal(rects.length, 1, "the join must not be abandoned");
  const parts = rects[0].parts;
  assert.deepEqual(
    parts.map((part) => part.slice).sort(),
    [0, 1],
    "the empty lobe's band paints too -- the joined text can reach the lobe"
  );
  const lowerPart = parts.find((part) => part.slice === 1);
  // The fit frame is (101, 0, 439x348) in the lower slice; the paint must
  // cover all of it (plus margin), or the local lettering's tail survives
  // beside the joined text -- the doubling.
  assert.ok(lowerPart.target.y <= 0, "the paint reaches the slice top");
  assert.ok(
    lowerPart.target.y + lowerPart.target.height >= 348,
    `the paint covers the whole fit extent, got ${lowerPart.target.y}+${lowerPart.target.height}`
  );
});

test("an ORDINARY plan's clip is untouched by the touch witnesses", () => {
  // The same under-spanning composite region shape on a plain two-sided join:
  // the clip must still stop at the padded region box, exactly as before the
  // touch witnesses existed -- widening it here would re-open the
  // raw-restore hole the clip guards.
  const plan = seamPlan(cutTop(), cutBottom());
  assert.ok(plan, "the ordinary fixture must plan");
  assert.ok(
    plan.pairs.every((pair) => !pair.top.touch && !pair.bottom.touch),
    "and it carries no touch pair"
  );
  const seamCut = cut(plan);
  const ink = region(300, seamCut + 20, 200, 60, [300, seamCut + 20, 200, 60]);
  const rects = seamCompositeRects(plan, [ink]);
  assert.equal(rects.length, 1);
  const parts = rects[0].parts;
  assert.deepEqual(
    parts.map((part) => part.slice),
    [1],
    "only the band holding the region paints, exactly as shipped"
  );
  // SEAM_MARGIN_PX of composite erase around the box, and nothing more.
  const only = parts[0];
  assert.ok(
    only.source.height <= 60 + 2 * 24 + 2,
    `the clip stays at the padded box, got height ${only.source.height}`
  );
});

/* -------------------------------------------------------------------------
 * The settle split: which `ok` replies answer a boundary forever.
 *
 * `performSeam` used to write SEAM_SETTLED on EVERY ok reply, including the
 * background's own pre-flight declines -- so one "uncached" during the race
 * with a neighbour's cache write killed the boundary for the life of the
 * page view, while glossary-skew's own comment promised "the boundary can
 * try again on a later view". The predicate lives in seam.js so node can
 * test it; the block after the unit rows lifts the SHIPPED performSeam and
 * drives it with crafted replies, because testing the predicate alone is how
 * a fix once shipped completely unwired (128 green, zero behaviour).
 * ------------------------------------------------------------------------- */

test("a server answer is final, with or without paint", () => {
  assert.equal(seamSkipFinal(undefined), true);
  assert.equal(seamSkipFinal(""), true);
});

test("the site profile is a fact about the page, not the attempt", () => {
  assert.equal(seamSkipFinal("manga-profile"), true);
});

test("the three transient declines earn their retries", () => {
  for (const reason of ["uncached", "glossary-skew", "mismatch"]) {
    assert.equal(seamSkipFinal(reason), false, reason);
  }
});

test("an unrecognised reason keeps the old behaviour: final", () => {
  assert.equal(seamSkipFinal("some-future-reason"), true);
});

/* Lifted out of the SHIPPED source, same trick as shippedScheduleSeam above:
 * the first unindented `\n}` after the anchor is the function's own closer. */
function shippedPerformSeam(reply) {
  const start = CONTENT_JS.indexOf("async function performSeam(run, plan, states, key)");
  assert.ok(start > 0, "could not find performSeam() in content.js");
  const end = CONTENT_JS.indexOf("\n}", start);
  assert.ok(end > start, "could not find performSeam()'s closer");
  const source = CONTENT_JS.slice(start, end + 2);

  const seamTried = new Map();
  const SEAM_SETTLED = Number.POSITIVE_INFINITY;
  const build = new Function(
    "markBusy",
    "markDone",
    "sliceSource",
    "log",
    "SEAM_MAX_SOURCE_BYTES",
    "browser",
    "applySeam",
    "seamRunEdges",
    "seamTried",
    "SEAM_SETTLED",
    "SEAM_MAX_ATTEMPTS",
    "seamSkipFinal",
    "seamRunning",
    `${source}; return performSeam;`
  );
  const performSeam = build(
    () => {},
    () => {},
    async () => ({ bytes: new Uint8Array(4), mime: "image/jpeg" }),
    () => {},
    1 << 20,
    { runtime: { sendMessage: async () => reply } },
    () => true,
    seamRunEdges,
    seamTried,
    SEAM_SETTLED,
    3,
    seamSkipFinal,
    new Set()
  );
  const run = [{}, {}];
  const states = [
    { url: "upper", seamed: [] },
    { url: "lower", seamed: [] },
  ];
  const drive = () => performSeam(run, { width: 800, height: 600 }, states, "upper\nlower");
  return { drive, seamTried, SEAM_SETTLED, key: "upper\nlower" };
}

test("a painted join settles its boundary for the page view", async () => {
  const { drive, seamTried, SEAM_SETTLED, key } = shippedPerformSeam({
    ok: true,
    painted: [],
  });
  await drive();
  assert.equal(seamTried.get(key), SEAM_SETTLED);
});

test("an uncached decline counts against the budget instead of settling", async () => {
  const { drive, seamTried, SEAM_SETTLED, key } = shippedPerformSeam({
    ok: true,
    joined: 0,
    skipped: "uncached",
  });
  await drive();
  assert.notEqual(seamTried.get(key), SEAM_SETTLED, "settled forever on a transient decline");
  assert.equal(seamTried.get(key), 1);
  await drive();
  assert.equal(seamTried.get(key), 2, "a second attempt keeps counting");
});

test("a manga-profile decline is about the page and settles", async () => {
  const { drive, seamTried, SEAM_SETTLED, key } = shippedPerformSeam({
    ok: true,
    joined: 0,
    skipped: "manga-profile",
  });
  await drive();
  assert.equal(seamTried.get(key), SEAM_SETTLED);
});

test("a failed reply still merely counts, as before", async () => {
  const { drive, seamTried, SEAM_SETTLED, key } = shippedPerformSeam({
    ok: false,
    error: "join failed",
  });
  await drive();
  assert.notEqual(seamTried.get(key), SEAM_SETTLED);
  assert.equal(seamTried.get(key), 1);
});
