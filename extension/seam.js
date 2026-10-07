/* BireLate - the webtoon seam.
 *
 * A webtoon is one tall strip, and the host delivers it as N separate images.
 * A speech bubble is free to straddle the cut between two of them, so each half
 * is uploaded on its own, OCR'd on its own, and translated on its own: two
 * fragments, neither of which is a sentence. The reader sees a bubble whose top
 * says one thing and whose bottom says another.
 *
 * THE OBVIOUS FIX IS ALREADY RULED OUT. Stitching the strip back together and
 * translating it tall makes detection WORSE, not better: RF-DETR squashes every
 * input to a fixed 1152x1152 square with independent per-axis scale factors and
 * no letterbox (`koharu_layout_rfdetr_seg_2xl/processor.rs:56-65`), so aspect
 * ratio is not free. A 1200x1700 manga page arrives with the two axes 0.71 apart
 * and that is what the checkpoint is used to; an 800x2400 stitch is 2.1x off
 * that, and a whole chapter strip is hopeless.
 *
 * So each slice is still translated exactly as it is today, and only the cut
 * bubble is redone: a SEAM image is cut from the tail of slice k and the head of
 * slice k+1, shaped so its aspect stays near a page's, translated as an ordinary
 * page, and the one rectangle holding the rejoined bubble is painted back over
 * both slices. The server needs no change -- a seam is just another image.
 *
 * WHAT THIS FILE IS. Every geometric decision, and nothing else: no DOM, no
 * fetch, no canvas, no browser API at all. It is loaded as a CLASSIC script into
 * both the content script and the background page, the same way cache.js is, so
 * the two sides cannot disagree about where the cut is -- and it is loaded by
 * `node --test` for tests/seam.test.js, which is the only way any of this
 * arithmetic gets checked without a browser.
 *
 * Every top-level name here starts with `seam` or `SEAM_`. A duplicate top-level
 * binding across classic scripts sharing one global is a SyntaxError that kills
 * all of them, and there are already 69 names taken between background.js and
 * cache.js.
 *
 * PRIOR ART, taken as behaviour and not as code. comic-translate (ogkalu2)
 * ships this in `pipeline/webtoon_batch/`: it pairs blocks
 * touching the bottom edge of file N with blocks touching the top edge of file
 * N+1, rejects on width ratio and horizontal disagreement, scores the survivors
 * and matches greedily 1:1. The reject rules and the score below are theirs, and
 * two things are deliberately NOT:
 *
 *  - Their edge test is an absolute 50px band on the text block. Ours is a tight
 *    band on the BUBBLE (see seamEffectiveBox), because a balloon cut by the
 *    image edge has its mask clipped at that edge and therefore genuinely
 *    touches it, while its text can stop far short. A tight band is what keeps
 *    this from firing on an ordinary vertical-scroll manga reader, where pages
 *    are also stacked flush and equal-width but bubbles sit inside a margin.
 *  - They stitch only to re-run OCR and inpainting, reusing the per-file
 *    detections. We cannot: the server's unit of work is a whole page, so the
 *    seam is re-detected. That is a cost, not a choice.
 *
 * Their own abandoned first attempt is worth knowing about too: they shipped
 * de-duplication (suppress the clipped copy) and replaced it with merging two
 * days later. Do not re-derive it.
 */

/* How close to the edge a box has to come to count as CUT BY it.
 *
 * Deliberately tight, and relative to the slice so it means the same thing on
 * an 800x1280 webtoons.com slice and on a 1500px-tall one. A balloon the image
 * edge actually cuts has its segmentation mask clipped at that edge, so its
 * bounds land within a mask cell of it -- and the mask is a 288x288 head
 * stretched back to page size, which is ~4.4px per cell vertically on a 1280px
 * slice. The floor is that quantisation plus slack; the fraction covers taller
 * slices.
 *
 * The cost of widening it is a FALSE seam: a vertical-scroll manga reader also
 * stacks equal-width pages flush, and comic-translate's 50px band would reach a
 * bubble sitting in an ordinary bottom margin. The cost of narrowing it is a
 * missed seam, which is the state we are already in.
 *
 * # 8 -> 16, and 16 is a CEILING that was measured rather than a round number
 *
 * The band was swept over a webtoon test chapter and a manga test volume,
 * against a vision census of which boundaries really are cut, with a control
 * proving the sweep's admission test reproduces `seamEdges` exactly at the
 * shipping band on every slice it scored.
 *
 *   band   webtoon, 172 boundaries                manga, 220 boundaries
 *      8   13 plans, 13 real, 0 false             0 joins
 *     12   13 plans, 13 real, 0 false             0 joins
 *     16   14 plans, 14 real, 0 false   <- here   0 joins
 *     20   14 plans, 14 real, 0 false             2 JOINS
 *     24   14 plans, 14 real, 0 false             5 joins
 *     32   19 plans, 14 real, 5 FALSE             7 joins
 *
 * **The manga cost appears at 20, exactly where the paragraph above predicts
 * it.** So 16 is the largest band that costs nothing on either side, and the
 * margin to the first false join is one step of the sweep and not more.
 *
 * What it buys is ONE boundary whose lower box sits 14.17 px below its cut --
 * six pixels outside the old band. That is the whole gain, stated plainly: this
 * is not the fix for the twelve missed boundaries, it is the fix for the one of
 * them that was a near miss. Eight of the rest are detector recall and three
 * are pairing vetoes. */
const SEAM_EDGE_FLOOR_PX = 16;
const SEAM_EDGE_FRACTION = 0.006;

/* comic-translate's reject rules and score. The two halves of one balloon are
 * the same balloon, so they agree on width and on horizontal position, and every
 * term below is that agreement. `chunk.py:282-297`.
 *
 * ONE OF THEIR THREE REJECTS IS DROPPED, because it cannot fire. Theirs refuses
 * a pair when the overlap ratio is under 0.35 AND the centre offset is over
 * 0.35, both measured against the narrower box. Write m for that narrower width
 * and d for the distance between the centres. Whichever box is wider, it reaches
 * at least m/2 either side of its own centre, so the intersection is at least
 * (min(c1,c2) + m/2) - (max(c1,c2) - m/2) = m - d, i.e.
 *
 *     overlap >= 1 - centre
 *
 * always. So overlap < 0.35 forces centre > 0.65, and the second half of their
 * conjunction is true whenever the first is: the pair of tests is exactly the
 * overlap test. Keeping the unreachable half would suggest a case it can save,
 * and there is none. The centre offset still earns its place in the SCORE, where
 * it separates two candidates that both survive. */
const SEAM_MAX_WIDTH_RATIO = 2.2;
const SEAM_MIN_OVERLAP = 0.35;

/* How far a box may poke out of the other's column and still count as CONTAINED.
 *
 * One pixel, i.e. essentially exact. It exists for float noise in box arithmetic
 * -- `fit_*` values arrive as fractions like 424.21875 -- and not as a tolerance
 * anybody should tune. Containment is a categorical claim ("this fragment sits
 * inside that balloon's column"), so widening this would turn it into a soft
 * overlap test, which `SEAM_MIN_OVERLAP` already is and which the width veto is
 * deliberately not. */
const SEAM_CONTAINMENT_SLACK_PX = 1;

/* How much taller (proportionally) a clipped remnant may run than a pure
 * height-vs-width comparison expects, and still count as clipped by the cut.
 *
 * The case that set it: one balloon straddles the cut, the upper slice's only
 * detection is a bare text line (its balloon was never detected, so `fit`
 * falls back to the text box) and the lower's is the balloon. The remnant's
 * height ratio came out 0.253807 against an inverse width ratio of 0.222173 --
 * the cut's signature missed by 14.24% relative, the pair was vetoed
 * `width-ratio`, and the reader got a spurious name on one slice and a
 * sentence missing its subject on the next.
 *
 * THE WINDOW IS ~10% WIDE AND THIS VALUE IS NOT WELL SEPARATED -- measured
 * over every width-ratio veto across eleven test chapters (581 candidates, 25
 * distinct):
 *
 *     1.1424  the wanted case (verified crossing)
 *     1.2563  nearest unwanted (unverified; already plans)
 *     1.2579  pinned REFUSED by tests/seam.test.js
 *     1.3055  a verified FALSE join (balloon vs unrelated SFX)
 *
 * 1.2 sits +5.0% above the wanted case and -4.7% below the nearest unwanted
 * one. Do not round it up: 1.3 lengthens a run the test suite pins, and 1.6
 * admits the false join. At 1.2, the sweep over all eleven chapters changed
 * exactly one boundary anywhere -- the wanted case itself. */
const SEAM_CUT_CLIP_SLACK = 1.2;

/* The detector's class for a drawn sound effect, spelled exactly as the server
 * emits it (`server/src/regions.rs`, `label` on each region of a `format=json`
 * reply -- `text` or `onomatopoeia`).
 *
 * A PAIR OF THESE IS NEVER A CUT BUBBLE, and that is a measured statement rather
 * than an aesthetic one. Driving this file over all 213 pages of a manga test
 * volume fired exactly 1 join of 212 boundaries, and it was wrong: an impact
 * effect bleeding off one page's bottom trim, paired with the chapter's title
 * logo clipped at the next page's top. Both boxes are labelled
 * `onomatopoeia`; they agree on width (ratio 1.46) and overlap by 0.75, so every
 * geometric test comic-translate has passes them. Nothing about the SHAPE of that
 * pairing distinguishes it from a real one -- only what the two things are.
 *
 * A speech balloon cut by the image edge is dialogue on both sides of the cut,
 * because it is one balloon and the detector labels it `text` on each slice. So
 * refusing a pair only when BOTH sides are effects cannot cost a real join.
 *
 * The trade, stated: a drawn effect genuinely straddling a webtoon cut is no
 * longer rejoined. That is deliberate. An effect is not a sentence -- rejoining
 * it buys a slightly better-shaped katakana render, against a measured false
 * join that repaints a correct page to "fix" a title logo. One side an effect
 * and the other dialogue is left ALONE, not refused: that is a mismatched pair
 * the geometry has no measured opinion about, and the rule is meant to be the
 * narrowest one that covers the observed failure. */
const SEAM_SFX_LABEL = "onomatopoeia";

/* The TOUCHES-class join trigger. Default ON (the flag is `touchJoin` in the
 * extension's settings); what actually arms this file is the presence of a
 * `white` profile on the neighbour's edge summary, which background.js
 * computes only when the flag is on.
 *
 * The class: a balloon whose SHAPE is cut by the slice boundary while its TEXT
 * is wholly on one side. The seam deliberately never joined it -- its precision
 * record was earned on cut text -- so the letterer centers in the clipped frame
 * it can see and the far side's white lobe stays empty. Censused in pixels
 * over all 50 strict candidates on four test chapters: 10 class A (far-side
 * empty lobe >= 60 px AND text-centre displacement >= 40 px), 18 B, 22 C;
 * joining is the direction the census settles because re-centring in an
 * extended frame pushes 8-45 px of a line off the canvas at four of the ten A
 * sites, and only a join can remove an in-lobe duplicate.
 *
 * The trigger's three tests, each carrying a censused reason:
 * - the box sits within `SEAM_TOUCH_FLUSH_PX` of the cut. Not "fit clipped
 *   flush at zero": one measured A-site box ends 6.2 px short of flush, so 8
 *   covers the measured near-miss.
 * - the NEIGHBOUR slice's edge band continues balloon white for at least
 *   `SEAM_TOUCH_MIN_CONTINUATION` px under most of the box's own columns.
 *   Nothing in one slice's JSON separates a harmless boundary from a bad one
 *   -- that was measured -- so this is a PIXEL question, answered by the
 *   `white` profile background.js measures from bytes it already holds
 *   (seam.js itself stays pixel-free by contract). 60 is the A-class
 *   calibration; the ten A lobes measure >= 300, 180, 155, 135, 125, 120,
 *   112, 95, 90, 75 (median 122).
 * - the box is DIALOGUE and not an effect. 9 of the 22 class-C sites are
 *   caption columns or SFX, and joining them re-renders rotated captions;
 *   `role` rides the wire for exactly this test.
 *
 * At the census's own calibration a >= 60 px continuation gate fires on 21 of
 * the 50 candidates: 10 real fixes, 11 harmless re-centres, +4 joins on one
 * chapter's existing 26. One boundary stays unfixable by any trigger (a site
 * banner interrupts the balloon INSIDE the slice). */
const SEAM_TOUCH_FLUSH_PX = 8;
const SEAM_TOUCH_MIN_CONTINUATION = 60;
const SEAM_TOUCH_MIN_FRACTION = 0.6;

/* How far the crop reaches past the matched bubble before the aspect padding
 * starts. Two jobs, and the larger of them is the second: the erase mask is
 * dilated by `round(max(W,H)/1024*6)` px computed from the SEAM image's own size
 * (`stages/detection.rs:1740-1743`), which is not the radius the parent slice
 * used, so the rejoined bubble's erased halo has to finish comfortably inside
 * the rectangle we paint back or the join shows as a hairline. */
const SEAM_MARGIN_PX = 24;

/* The most slices one join may cover. See `seamRunLength` for why this is a
 * guard and not a limit: the longest run measured on a 173-slice webtoon test
 * chapter is THREE (only two of its slices are crossed end to end), and the
 * server's own 8192 px side cap is eight of these slices. */
const SEAM_MAX_RUN_SLICES = 6;

/* The seam is grown towards the shape of the slices it came from.
 *
 * NOT towards a manga page, which is what this used to say. A constant was the
 * wrong shape of rule and real material is what showed it: webtoons.com CANVAS
 * slices at 800x1280 are portrait, while another host's test chapter is
 * **1280x1000** -- landscape. A fixed height/width of 1.35
 * would have asked for a 1280x1728 crop there, taller than either slice and
 * nearly two slices' worth of translation per join, to reach a shape the rest of
 * that chapter is not being translated at anyway.
 *
 * The slices themselves are the best available evidence of what this material's
 * detection copes with: every one of them is being detected successfully, at
 * their own aspect, on every page of the chapter. So the target is simply to be
 * that shape. The floor stops a short tail slice from producing a band -- the
 * failure this exists to prevent -- and the ceiling bounds the cost, because
 * every extra bubble the crop swallows is another segment in the seam's own
 * translation and the seam is a third full pipeline run queued behind the two
 * slices.
 *
 * The mechanism the floor guards against is severe and silent: RF-DETR squashes
 * every input to a fixed 1152 square with independent per-axis factors, so an
 * 800x150 band arrives with its vertical axis UPSCALED 7.7x against the
 * horizontal's 1.4x -- glyphs at 5.3x their trained aspect, and plausible-looking
 * degraded detections rather than an error. Both bounds are a ceiling and a
 * floor on PADDING only; neither can shrink a crop below the bubble it holds. */
const SEAM_MIN_ASPECT = 0.75;
const SEAM_MAX_ASPECT = 2.0;

/* The server refuses a page with a side over 8192px before it does any work
 * (`scene.rs:133-148`). Refusing here instead turns a 400 into a skipped seam. */
const SEAM_MAX_SIDE_PX = 8192;

/* Bounds on the neighbour sweep in seamCompositeRects. Four passes is far more
 * than a real page needs -- detection's own NMS keeps regions from overlapping
 * much, so a chain longer than two is already unusual -- and it is a termination
 * guard rather than a tuning knob. The fraction is the real limit: past it the
 * rectangle is repainting most of a page that is already correct. */
const SEAM_SWEEP_PASSES = 4;
const SEAM_MAX_RECT_FRACTION = 0.6;

/* How much of an OVERSIZED paint rectangle has to be the thing it exists to
 * repair, before the size stops being an objection.
 *
 * `SEAM_MAX_RECT_FRACTION` above is a proxy, and its own comment says what for:
 * a rectangle that large is "repainting most of a page that is already correct,
 * dragging every bubble in it to the seam's own size base, to fix one". The harm
 * named there is the DRAGGING -- artwork and other lettering imported from the
 * composite as a side effect -- and area alone cannot see it. A rectangle that is
 * almost entirely the joined region is not dragging anything; it IS the repair.
 *
 * **Measured on the two cases that bracket it, and they are three quarters
 * apart.**
 *
 *   `tests/seam.test.js`, the four-slice run guard: one 600x200 region, 120,000
 *   px, inside a rectangle of ~1.82M px on an 800x2808 composite -- **0.066**.
 *   The rectangle is tall because the run's own pair geometry spans whole slices,
 *   so nearly all of it is artwork being dragged to fix a band 5% of its size.
 *   Refused, as it was before.
 *
 *   A webtoon test boundary: the joined sound effect is 1100x1394 = 1,533,400 px
 *   inside a rectangle of ~1,660,800 on a 1200x1384 composite -- **0.92**. There
 *   is no collateral to speak of: the rectangle is the effect. Refusing it left
 *   the reader a per-slice effect translation over un-erased Chinese, because the
 *   lower half is 0.999 of its own slice and is refused unread as a cross-slice
 *   fragment.
 *
 * 0.5 sits in the middle of that gap and nothing measured lands near it.
 *
 * It is a SECOND condition, never a replacement: a rectangle under the fraction
 * is taken as before and never consults this. */
const SEAM_MIN_RECT_PAYLOAD = 0.5;

const seamClamp = (value, low, high) => Math.min(Math.max(value, low), high);

/* Where a region actually SITS on the page, which is not always its own box.
 *
 * `fit_*` is where the translated text was painted -- the surrounding balloon
 * when the region is inside one, the region itself otherwise -- and the balloon
 * is the thing the image edge cuts. Using the text box instead would miss every
 * bubble whose cut falls between two lines, which is most of them: the ink stops
 * where the line stops, the balloon runs on to the edge.
 *
 * All four fit fields appear or vanish together (`regions.rs:187`), so one test
 * covers them, but each is still checked for being a finite number because they
 * arrive from the network.
 *
 * The detector's `label` rides along when the region carries one, because the
 * sfx veto below needs it and this box is the only thing that survives onto the
 * cache entry. It is set only when it is a non-empty string: an ABSENT label has
 * to stay absent all the way down, since "we do not know what this is" and "this
 * is not an effect" must reach the veto as different answers -- see
 * `seamPairVeto`. */
function seamEffectiveBox(region) {
  if (!region) return null;
  const numbers = [region.fit_x, region.fit_y, region.fit_width, region.fit_height];
  const fitted = numbers.every((value) => typeof value === "number" && Number.isFinite(value));
  const x = fitted ? region.fit_x : region.x;
  const y = fitted ? region.fit_y : region.y;
  const width = fitted ? region.fit_width : region.width;
  const height = fitted ? region.fit_height : region.height;
  if (![x, y, width, height].every((value) => typeof value === "number" && Number.isFinite(value))) {
    return null;
  }
  if (width <= 0 || height <= 0) return null;
  const box = { x, y, width, height };
  if (typeof region.label === "string" && region.label) box.label = region.label;
  /* `role` earns its bytes the same way `label` does: it is what scopes the
   * TOUCHES trigger to dialogue -- 9 of the census's 22 class-C sites are
   * caption columns or SFX, and a join there re-renders rotated captions.
   * Absent stays absent, and an absent role never fires the trigger. */
  if (typeof region.role === "string" && region.role) box.role = region.role;
  /* The pipeline's refusal rides the box so LIVE-sidedness (below) can be
   * decided without it. Nothing else reads it: the box still enters
   * `tops`/`bottoms`, still pairs, still scores, still anchors a crop. */
  if (region.refused) box.refused = true;
  return box;
}

/* The boxes at an edge the pipeline actually LETTERED -- what "this side is
 * populated" means once a refused fragment or a sub-floor hint is no longer
 * allowed to answer for it. Both exclusions are load-bearing and neither
 * alone reaches the measured case: one slice's top band held a refused orphan
 * `...` (conf 0.463, `rendered_text` empty) AND four edge hints, and a hint
 * carries no `refused` at all -- it is a different channel, the server
 * saying "below my floor", never "I refused this text". Skipping only the
 * refusal leaves the side four boxes deep and the boundary still two-sided,
 * which is exactly how a sweep measured skipping refusals alone as an exact
 * no-op across four chapters. */
const seamLive = (list) =>
  Array.isArray(list) ? list.filter((box) => !box.refused && !box.hint) : [];

function seamEdgeBand(height) {
  return Math.max(SEAM_EDGE_FLOOR_PX, Math.round(height * SEAM_EDGE_FRACTION));
}

/* The whole of what one translated slice has to remember for its neighbours.
 *
 * Kept deliberately small: this rides on the 24-hour cache entry so that a
 * REVISITED page can still be seamed, and the entry's byte cap is computed from
 * `blob.size` alone (`cache.js:270`) -- anything else stored beside it is
 * invisible to the LRU and inflates real disk use past the configured ceiling
 * with nothing to notice. Two lists of four numbers and a class name is nothing;
 * the region list with its source and translated strings would not be. The label
 * is the one non-numeric field and it earns its ~14 bytes: it is what stops the
 * measured false join `SEAM_SFX_LABEL` describes, and there is no way to
 * recover it later -- the stored blob is a picture, and the regions it came
 * from are gone.
 *
 * `width`/`height` travel with it because a box is meaningless without the page
 * it was measured on, and the caller reading this back from the cache has no
 * other way to know what the image was when it was translated.
 *
 * # `hints` -- EDGE EVIDENCE THAT IS NOT A REGION
 *
 * `edge_hints` on a `format=json` reply: detections the server found BELOW its
 * own text floor and therefore refused as regions, reported on their own channel
 * because they touch a page edge. They are never OCR'd, never lettered, never
 * erased and never counted -- `mask_includes` builds the erase mask from raw
 * detections, so admitting one as a region would strip artwork on every ordinary
 * page while nothing on the page ever letters it (`detection.rs:2494`, and the
 * server refuses them in two places for exactly that reason).
 *
 * They exist because a column can be REAL and still score under the floor. On
 * one webtoon test slice a display column's best box scores 0.2363 against a
 * 0.25 floor and matches the measured ink to ~2 px; without it the join starts
 * one slice later and loses a whole glyph of the name off the top.
 *
 * WHAT A HINT IS ALLOWED TO DO IS EXACTLY WHAT A BOX IN THESE TWO LISTS DOES,
 * and nothing else. It is admitted by the SAME band test the regions above use,
 * against the same `height`, and it becomes an ordinary four-number box -- so
 * every rule downstream (the sfx veto, the width ratio, the overlap floor, the
 * score) applies to it unchanged. There is no second code path to keep in step.
 *
 * # It is routed by GEOMETRY, not by the `edge` field it arrives with
 *
 * The wire carries `edge: "top" | "bottom"`, which is the server saying which
 * band IT measured the box against. Re-deriving it here rather than trusting it
 * keeps one source of truth: the band is a function of the page height, both
 * sides compute it from the same height, and a disagreement can only mean the
 * two heights differ -- in which case the geometry we hold is the one the rest
 * of this file is about to do arithmetic with. A hint declared `top` and sitting
 * at the foot of the page would otherwise be filed as a top box with an
 * `edgeDistance` of ~900, which is not a conservative failure, it is a wrong one.
 *
 * # A hint touching both bands is dropped UNLESS THE SERVER SAYS IT SPANS
 *
 * `seamSpansSlice` tests IDENTITY against these two lists, so one object in both
 * is what declares a slice CROSSED END TO END, and that decides how many slices a
 * run covers. **A sub-floor fragment must not be able to reach that lever by
 * accident, and that rule is unchanged.**
 *
 * What changed is that the server can now attest to a crossing. `spans: true`
 * means it walked THIS box's own ink from the band it touches to the opposite one
 * -- `repaired_column`'s walk, the same one an above-floor box is grown by -- and
 * the four numbers are the grown box. Only that admits an object to both lists.
 *
 * The judgement is the server's because it is the only side that can make it: the
 * walk reads pixels and this file never sees one. A hint arriving with `spans`
 * absent or false behaves exactly as it did before `spans` existed, which is
 * what keeps every older server, and every fragment that really does end inside
 * its slice, on the old path.
 *
 * The measured case: a webtoon test slice with no admitted box at all -- its
 * best candidate scores 0.2295 against a 0.25 floor -- while its ink runs the
 * full height of the slice. Before this, no run could pass through it and the
 * column was read as two fragments. */
function seamEdges(regions, width, height, hints, white) {
  const top = [];
  const bottom = [];
  const band = seamEdgeBand(height);

  for (const region of Array.isArray(regions) ? regions : []) {
    const box = seamEffectiveBox(region);
    if (!box) continue;
    if (box.y <= band) top.push(box);
    if (box.y + box.height >= height - band) bottom.push(box);
  }

  /* AFTER the regions, deliberately. `seamRunPlan` breaks a score tie by index,
   * so appending means an exact tie between a hint and a box the detector was
   * confident about goes to the confident one. Absent, null, or a server too old
   * to send the field all arrive here as an empty list and change nothing.
   *
   * THE ONE RESIDUAL RISK, named because it is not guarded here. The match at a
   * cut is greedy and 1:1, so a hint at the same edge as a real box, scoring
   * HIGHER against the same partner, takes that partner and the real pairing is
   * skipped -- a join made on the hint's geometry instead of the balloon's, and
   * a crop anchored on the hint's `y`. Guarding it needs a marker on the box so
   * the sort can prefer a region, and a marker is a fifth field on every cached
   * box, which the byte argument above is exactly about. It is left unguarded
   * because a hint is a real detection and pairing on it is not wrong, only
   * different -- but if a chapter is ever measured losing a join it used to
   * make, this is the first place to look. */
  for (const hint of Array.isArray(hints) ? hints : []) {
    /* The same normaliser the regions use, which also does the finite-number
     * checks these need for arriving off the network. A hint carries no `fit_*`,
     * so it falls through to its own four numbers -- there is no balloon behind
     * a box the detector refused. */
    const box = seamEffectiveBox(hint);
    if (!box) continue;
    /* MARKED, and this is what amendment 1 hangs on. A grown spanning hint is
     * flush with both page edges by construction, so its `edgeDistance` is 0 at
     * both cuts against a divisor of `3 * band` -- it wins the greedy 1:1 match
     * at `seamRunPlan` nearly always, and could take a partner a real region
     * needed. `seamRunPlan` sorts regions ahead of hints to stop that, and this
     * flag is how it tells them apart. See the sort there for the property it
     * buys: the region-only matching is left byte-identical to the no-hints arm.
     *
     * It is a fifth field on a cached box and the doc above weighs those bytes
     * carefully. It earns them: without it a measured slice that returns the
     * column as a REGION and a sub-floor hint of the same ink puts two spanning
     * objects at one cut with nothing to choose between. */
    box.hint = true;
    const atTop = box.y <= band;
    const atBottom = box.y + box.height >= height - band;
    if (atTop !== atBottom) {
      (atTop ? top : bottom).push(box);
      continue;
    }
    /* Touching both. Admitted to BOTH lists as the SAME object -- which is what
     * `seamSpansSlice` reads -- only on the server's explicit attestation, and
     * only when the geometry agrees with it. `spans` on a box that does not
     * actually reach both bands is a contradiction, and the geometry is what the
     * rest of this file does arithmetic with, so the geometry wins. */
    if (hint.spans === true && atTop && atBottom) {
      top.push(box);
      bottom.push(box);
    }
  }

  const edges = { width, height, top, bottom };
  /* The white-continuation profile, attached only when the caller
   * measured one (background.js, flag `touchJoin` on). It rides the cache
   * entry like everything else here -- ~2 KB against the doc's byte argument
   * above, paid only by a reader who armed the flag -- and its ABSENCE is the
   * off arm: no profile, no trigger, byte-identical behaviour to before it
   * existed. Validated shape-wise here so a malformed value degrades to off. */
  if (
    white &&
    Array.isArray(white.top) &&
    Array.isArray(white.bottom) &&
    Number.isFinite(white.step) &&
    white.step > 0
  ) {
    edges.white = { top: white.top, bottom: white.bottom, step: white.step };
  }
  return edges;
}

/* Is this box a drawn sound effect, as far as we can tell?
 *
 * FALSE FOR AN ABSENT LABEL, and that is the whole of the backwards
 * compatibility story. A cache entry written before the label was summarised
 * carries bare four-number boxes, and so does a region the server gave no class
 * to (`label` is `skip_serializing_if = "Option::is_none"`). Reading "no label"
 * as "an effect" would turn the veto below into a blanket refusal of every join
 * against every entry already on disk -- the feature silently off for 24 hours,
 * with nothing in the UI to say so. Unknown means unknown, and unknown does not
 * veto. */
const seamIsSfx = (box) => box.label === SEAM_SFX_LABEL;

/* Why this pairing cannot be one balloon, or null if nothing forbids it.
 *
 * Split out of seamPairScore so there is exactly one statement of the rejects
 * and a measurement can still name which one fired -- `seamCandidates` reports
 * it per candidate, so a measurement can count it across a whole chapter.
 *
 * These are categorical, not soft: a width ratio of 3 does not mean "probably
 * not the same balloon", it means these are two different balloons that happen
 * to sit at the same edge. */
function seamPairVeto(top, bottom) {
  // Cheapest first, and the only one that looks at what the boxes ARE.
  if (seamIsSfx(top) && seamIsSfx(bottom)) return "sfx";

  const narrower = Math.max(1, Math.min(top.width, bottom.width));
  const widthRatio = Math.max(top.width, bottom.width) / narrower;

  const left = Math.max(top.x, bottom.x);
  const right = Math.min(top.x + top.width, bottom.x + bottom.width);
  const shared = Math.max(0, right - left);
  /* Is the narrower box entirely inside the wider one's column? Measured before
   * the width veto because it is what tells the two failure shapes apart. */
  const contained = shared >= narrower - SEAM_CONTAINMENT_SLACK_PX;

  /* A CUT REMOVES HEIGHT, NOT WIDTH, and that is what separates a clipped
   * remnant from two balloons that merely meet at an edge.
   *
   * The slice boundary is horizontal, so the fragment it leaves behind is
   * proportionally far more clipped VERTICALLY than horizontally: on one
   * measured boundary the remnant is 31.5 px tall against the balloon's 328
   * (0.096) while being 145.3 wide against 528 (0.275). Two different balloons
   * sharing an edge are not clipped at all, so their height ratio is ordinary --
   * the fixture in `tests/seam.test.js` that pins this is 50 px tall on both
   * sides (1.000) against a width ratio of 0.333, and must keep being refused.
   *
   * Comparing the two ratios needed no threshold until a real cut balloon
   * was measured missing the bare comparison by 14.24%: the upper slice's
   * balloon was never detected, so its `fit` fell back to the padded TEXT box,
   * whose height is not the remnant's ink height. `SEAM_CUT_CLIP_SLACK`
   * absorbs exactly that padding, and its comment carries the measured window
   * -- the value is NOT free to grow.
   *
   * The sfx guard is the cheap half: a drawn sound effect on either side is
   * never a clipped remnant of a dialogue balloon, and refusing the escape for
   * it is what blocks the two nearest false joins (a balloon against an
   * unrelated blue SFX; an sfx against a hint) by category rather than by a 5%
   * numeric margin. Measured a strict no-op on every test chapter at the
   * unslacked comparison. */
  const heightRatio =
    Math.min(top.height, bottom.height) / Math.max(1, Math.max(top.height, bottom.height));
  const inverseWidthRatio = narrower / Math.max(1, Math.max(top.width, bottom.width));
  const clippedByTheCut =
    !seamIsSfx(top) &&
    !seamIsSfx(bottom) &&
    heightRatio < inverseWidthRatio * SEAM_CUT_CLIP_SLACK;

  /* THE WIDTH VETO COMPARES A BALLOON AGAINST A TEXT FRAGMENT, and on a cut
   * balloon that is not a like-for-like comparison.
   *
   * `seamEffectiveBox` prefers `fit_*`, which is the BUBBLE where one was
   * detected and the TEXT box where one was not. A balloon cut near its foot
   * leaves only a sliver of glyph on the next slice -- too little for a bubble
   * to be detected at all -- so the upper side reports the balloon's full width
   * and the lower side reports the width of whatever few characters happen to
   * survive the cut. The ratio then measures how much of the line was clipped,
   * not whether the two halves belong together.
   *
   * Measured on that same boundary: the balloon is 528 px wide, the surviving
   * sliver 145.3, ratio **3.63** against a 2.2 ceiling -- vetoed -- while the
   * horizontal overlap is **1.0000**, i.e. the sliver sits entirely inside the
   * balloon's column. The line `跑上去？` is cut across the boundary; its lower
   * halves read as `」ト土？`, which the script rule refuses as not-Chinese, so it
   * is never lettered AND never erased. That residue is what a reader sees.
   *
   * Containment ALONE is too weak -- two boxes flush against the same margin are
   * contained without being one balloon, which is exactly the fixture
   * `tests/seam.test.js:122` pins. It takes containment AND the cut's own
   * signature (see `clippedByTheCut`) to suspend the width veto, and nothing else
   * -- the edge band, the sfx veto, the overlap floor and the score
   * all still apply, and the score's own `-0.35 * |widthRatio - 1|` term still
   * charges for the disparity rather than ignoring it. */
  if (widthRatio > SEAM_MAX_WIDTH_RATIO && !(contained && clippedByTheCut)) {
    return "width-ratio";
  }

  if (shared / narrower < SEAM_MIN_OVERLAP) return "overlap";

  return null;
}

/* Reject or score one candidate pairing, comic-translate's rules verbatim.
 *
 * Returns null for a reject rather than a very negative score, for the reason
 * given on seamPairVeto. */
function seamPairScore(top, bottom, band) {
  if (seamPairVeto(top, bottom)) return null;

  const widthRatio = Math.max(top.width, bottom.width) / Math.max(1, Math.min(top.width, bottom.width));

  const left = Math.max(top.x, bottom.x);
  const right = Math.min(top.x + top.width, bottom.x + bottom.width);
  const overlap = Math.max(0, right - left) / Math.max(1, Math.min(top.width, bottom.width));

  const topCenter = top.x + top.width / 2;
  const bottomCenter = bottom.x + bottom.width / 2;
  const center = Math.abs(topCenter - bottomCenter) / Math.max(1, Math.min(top.width, bottom.width));

  return (
    2.6 * overlap -
    0.75 * center -
    0.35 * Math.abs(widthRatio - 1) -
    (top.edgeDistance + bottom.edgeDistance) / Math.max(1, 3 * band)
  );
}

/* Which bubbles cross the cut between two contiguous slices, and what image to
 * send so they can be read as one.
 *
 * `a` is the upper slice and `b` the lower, both as returned by seamEdges. Null
 * means there is nothing to do, which is the overwhelmingly common answer and
 * has to stay cheap.
 *
 * Both sides are required to have an edge-touching box, and that is not a
 * conservatism to be relaxed later -- it is the definition. A balloon cut above
 * all of its text leaves the whole sentence on the lower slice and needs no
 * seam; one cut below all of its text leaves it whole on the upper slice. Text
 * on both sides of the cut is exactly the case this exists for. */
/* Every pairing this boundary could make, scored, and the reason it could not.
 *
 * Split out of seamPlan so a measurement can see the REJECTS. `null` from
 * seamPlan is the answer at almost every boundary of almost every strip, and
 * without this there is no way to tell "no bubble is near the cut" from "a
 * bubble was there and the width ratio refused it" -- which is exactly the
 * distinction an adversarial sample of non-firing boundaries has to be drawn on.
 * `refused` is a string rather than a boolean because the two rejects fail for
 * different reasons and a measurement should not have to guess which. */
/* The columns of one box, probed against the neighbour's white profile.
 *
 * Returns the MEDIAN continuation depth when at least
 * `SEAM_TOUCH_MIN_FRACTION` of the box's columns continue white for
 * `SEAM_TOUCH_MIN_CONTINUATION` px or more, else null. The median is >= the
 * threshold by construction whenever the fraction test passes (0.6 of columns
 * at or above 60 puts the 50th percentile at or above it), so the synthetic
 * box built from it is never shorter than the gate that admitted it. Bucket
 * depths are the MINIMUM within each `step` columns (background.js measures
 * them that way), so this reads conservative evidence, not averages. */
function seamTouchProbe(profile, step, x, width) {
  const from = Math.max(0, Math.floor(x / step));
  const to = Math.min(profile.length - 1, Math.ceil((x + width) / step) - 1);
  if (to < from) return null;
  const depths = [];
  for (let bucket = from; bucket <= to; bucket += 1) {
    const depth = profile[bucket];
    depths.push(Number.isFinite(depth) && depth > 0 ? depth : 0);
  }
  const qualifying = depths.filter((depth) => depth >= SEAM_TOUCH_MIN_CONTINUATION).length;
  if (qualifying / depths.length < SEAM_TOUCH_MIN_FRACTION) return null;
  const sorted = depths.slice().sort((p, q) => p - q);
  return sorted[Math.floor(sorted.length / 2)];
}

/* The whole TOUCHES decision, composed and named so a test calls exactly what
 * `seamCandidates` calls. Given a one-sided boundary, build
 * the synthetic counterpart boxes the empty side has earned -- or return an
 * empty list, which keeps the one-sided refusal byte-identical to today.
 *
 * A synthetic box copies its source's columns, sits flush at the neighbour's
 * facing edge, and is as tall as the measured white lobe. From there the
 * ORDINARY machinery runs unchanged: identical columns pass the width veto at
 * 1.0 and the overlap floor at 1.0, and the join, the crop and the paint-back
 * neither know nor care that one side's evidence was a probe. `touch: true`
 * marks it for observability, the same way `hint` marks a sub-floor box. */
function seamTouchCounterparts(a, b) {
  /* One-sidedness is decided on LIVE boxes only: a fragment the pipeline
   * refused, or a sub-floor hint, cannot keep a boundary two-sided for the
   * trigger's purposes. Priced before building over five arms, adversarially
   * verified: +1 join (the measured case), 0 lost, nothing else moves. */
  const textOnTop = seamLive(a.bottom).length > 0 && seamLive(b.top).length === 0;
  const textOnBottom = seamLive(b.top).length > 0 && seamLive(a.bottom).length === 0;
  if (textOnTop === textOnBottom) return [];
  const source = textOnTop ? a.bottom : b.top;
  const neighbour = textOnTop ? b : a;
  const white = neighbour.white;
  if (!white) return [];
  const profile = textOnTop ? white.top : white.bottom;
  if (!Array.isArray(profile) || !Number.isFinite(white.step) || white.step <= 0) return [];
  const counterparts = [];
  for (const box of source) {
    if (box.hint) continue;
    if (box.role !== "dialogue") continue;
    if (box.label === SEAM_SFX_LABEL) continue;
    const gap = textOnTop ? a.height - (box.y + box.height) : box.y;
    if (gap > SEAM_TOUCH_FLUSH_PX) continue;
    const depth = seamTouchProbe(profile, white.step, box.x, box.width);
    if (depth === null) continue;
    counterparts.push({
      x: box.x,
      width: box.width,
      y: textOnTop ? 0 : neighbour.height - depth,
      height: depth,
      touch: true,
    });
  }
  return counterparts;
}

function seamCandidates(a, b) {
  if (!a || !b) return { refused: "missing", candidates: [] };
  if (!a.width || !b.width) return { refused: "no-size", candidates: [] };
  if (a.width !== b.width) {
    return { refused: "width-mismatch", widths: [a.width, b.width], candidates: [] };
  }
  if (!a.height || !b.height) return { refused: "no-size", candidates: [] };
  if (!Array.isArray(a.bottom) || !Array.isArray(b.top)) {
    return { refused: "no-edges", candidates: [] };
  }
  /* LIVE-SIDEDNESS FOR THE TRIGGER, RAW COUNTS FOR THE REFUSAL, and the
   * separation is the load-bearing part. Two questions the
   * previous form answered with one count:
   *
   *   "may the TOUCHES trigger run here?" -- asked of the LIVE boxes, so a
   *      fragment the pipeline refused (or a sub-floor hint) can no longer
   *      answer for a side it never lettered.
   *   "is this boundary refused as one-sided?" -- still asked of the RAW
   *      lists, byte-identically to before. Refusing on live counts instead
   *      DELETES joins: a two-sided boundary whose far-side boxes are all
   *      refused still pairs and still plans today (the ko lever's
   *      refused-but-real dialogue boxes anchor genuine cuts), and a first
   *      draft that did so lost 2/2/1/2 joins on four test chapters exactly
   *      that way.
   *
   * The refused box also still RIDES: the synthetic counterpart is APPENDED
   * beside it, never substituted for it. When the target list is genuinely
   * empty the append is the previous assignment verbatim, so a truly
   * one-sided boundary behaves exactly as it did. */
  const aLive = seamLive(a.bottom).length;
  const bLive = seamLive(b.top).length;
  if (!aLive || !bLive) {
    /* Before refusing, ask whether the empty side's PIXELS continue
     * a dialogue balloon the populated side holds flush against the cut. New
     * OBJECTS, never mutation -- `a`/`b` are the caller's edge summaries and on
     * the product path they are the cached ones. An empty answer (no profile,
     * no qualifying box, flag off, old cache entry) is the old refusal verbatim. */
    const touched = seamTouchCounterparts(a, b);
    if (touched.length) {
      if (aLive) b = { ...b, top: [...b.top, ...touched] };
      else a = { ...a, bottom: [...a.bottom, ...touched] };
    } else if (!a.bottom.length || !b.top.length) {
      return {
        refused: "one-sided",
        counts: [a.bottom.length, b.top.length],
        candidates: [],
      };
    }
  }

  const band = Math.max(seamEdgeBand(a.height), seamEdgeBand(b.height));
  const tops = a.bottom.map((box) => ({
    ...box,
    edgeDistance: Math.max(0, a.height - (box.y + box.height)),
  }));
  const bottoms = b.top.map((box) => ({ ...box, edgeDistance: Math.max(0, box.y) }));

  const candidates = [];
  for (let i = 0; i < tops.length; i += 1) {
    for (let j = 0; j < bottoms.length; j += 1) {
      const veto = seamPairVeto(tops[i], bottoms[j]);
      candidates.push({
        i,
        j,
        refused: veto,
        score: veto ? null : seamPairScore(tops[i], bottoms[j], band),
      });
    }
  }
  /* `sfx` rather than `no-pair` when that is what refused every candidate, so a
   * chapter-wide run can COUNT the veto instead of inferring it from a join that
   * stopped happening. A measurement can tally this string per boundary, and a
   * rule whose only evidence is an absence is a rule nobody can re-check. */
  let refused = null;
  if (candidates.every((one) => one.score === null)) {
    refused = candidates.every((one) => one.refused === "sfx") ? "sfx" : "no-pair";
  }
  return { refused, band, tops, bottoms, candidates };
}

/* The crop for a RUN of slices, of which the two-slice pair is the ordinary
 * case and not a separate code path.
 *
 * `edgesList` is the strip in order, `index` the head of the run and `length`
 * how many slices it covers -- which is what `seamRunLength` returns. A run of
 * two is byte-for-byte the join this file has always planned; a run of three or
 * four is the case where a skill name drawn down the side
 * of a panel crosses slices that hold neither its beginning nor its end.
 *
 * # Every cut in the run is a join in its own right
 *
 * The per-boundary test is `seamCandidates`, reused rather than reimplemented,
 * so the width-mismatch veto, the sfx veto and the one-sided refusal all still
 * apply at every cut. A spanned middle is NOT exempt: its box arrives as the
 * upper slice's `bottom` and the lower slice's `top`, so the x-overlap and
 * width-ratio tests still ask whether the fragment below is the continuation of
 * the one above. That is the only thing separating one tall column from two
 * unrelated columns on opposite sides of the page, and dropping it would make a
 * long run easier to form than a short one.
 *
 * # The middles are taken WHOLE, and there is nothing to choose
 *
 * A slice a region spans end to end contributes all of itself: there is no
 * shorter band that holds the text. Only the head's tail and the tail's head
 * are cropped, which is why the padding arithmetic below is unchanged from the
 * pairwise version -- it still has exactly two places to grow into. */
function seamRunPlan(edgesList, index, length) {
  const list = Array.isArray(edgesList) ? edgesList : [];
  const count = Number.isFinite(length) ? Math.floor(length) : 0;
  if (count < 2 || count > SEAM_MAX_RUN_SLICES) return null;
  if (index < 0 || index + count > list.length) return null;

  const run = list.slice(index, index + count);
  if (run.some((edges) => !edges || !edges.width || !edges.height)) return null;

  const probes = [];
  for (let i = 0; i + 1 < count; i += 1) {
    const probe = seamCandidates(run[i], run[i + 1]);
    if (probe.refused) return null;
    probes.push(probe);
  }

  const width = run[0].width;
  const cuts = probes.length;

  /* Greedy 1:1, best first, PER CUT. Two balloons side by side at the same cut
   * is a real arrangement and both should be rejoined; one balloon claiming two
   * partners is not. Ties broken by index so the result cannot depend on sort
   * stability. Each cut has its own top and bottom index spaces, so the used
   * sets are per cut too -- a spanned column is legitimately the bottom of one
   * pair and the top of the next, and one shared set would refuse it. */
  const pairs = [];
  for (let cut = 0; cut < cuts; cut += 1) {
    const { tops, bottoms } = probes[cut];
    const candidates = probes[cut].candidates.filter((one) => one.score !== null);
    /* AMENDMENT 1 -- A HINT MUST NEVER TAKE A PARTNER A REGION NEEDED.
     *
     * The match below is greedy and 1:1, so whichever candidate is considered
     * first claims both its boxes. A grown spanning hint is flush with both page
     * edges by construction, so `edgeDistance` is 0 at both cuts against a
     * divisor of `3 * band` -- on 908 px slices that is a 0.83 advantage over a
     * balloon sitting 20 px inside the band, against an overlap term worth at
     * most 2.6. On score alone the hint wins nearly every time.
     *
     * `seamEdges` already appends hints AFTER regions, so region indices are
     * unchanged; ranking by hint-count first and leaving the rest of the
     * comparator alone therefore buys a property rather than a heuristic:
     *
     *   EVERY REGION-REGION PAIRING IS CONSIDERED, IN THE SAME ORDER, AS IT
     *   WOULD BE WITH NO HINTS PRESENT -- so the region matching at any cut is
     *   byte-identical to the no-hints arm, and hints can only consume partners
     *   the regions did not want.
     *
     * That is LOST-0 for pairs by construction rather than by chapter sweep,
     * which matters because a chapter sweep cannot see this at all: the
     * measured case that needs it is a slice that returns the column as a
     * REGION *and* a sub-floor hint of the same ink, putting two spanning
     * objects at one cut. `seamEdges`' hint doc names this risk as unguarded
     * there; this is the guard. */
    const hintRank = (one) =>
      (tops[one.i] && tops[one.i].hint ? 1 : 0) +
      (bottoms[one.j] && bottoms[one.j].hint ? 1 : 0);
    candidates.sort(
      (left, right) =>
        hintRank(left) - hintRank(right) ||
        right.score - left.score ||
        left.i - right.i ||
        left.j - right.j
    );
    const usedTop = new Set();
    const usedBottom = new Set();
    for (const candidate of candidates) {
      if (usedTop.has(candidate.i) || usedBottom.has(candidate.j)) continue;
      usedTop.add(candidate.i);
      usedBottom.add(candidate.j);
      pairs.push({
        cut,
        top: tops[candidate.i],
        bottom: bottoms[candidate.j],
        score: candidate.score,
      });
    }
  }
  if (!pairs.length) return null;

  const first = run[0];
  const last = run[count - 1];

  /* The crop has to hold every matched bubble whole, with the margin, before any
   * aspect padding is considered. Only the FIRST cut can move the head's crop
   * and only the LAST can move the tail's; a pair at a middle cut sits inside
   * slices that are already being taken whole. */
  let topY = first.height;
  let bottomY = 0;
  for (const pair of pairs) {
    if (pair.cut === 0) topY = Math.min(topY, pair.top.y);
    if (pair.cut === cuts - 1) bottomY = Math.max(bottomY, pair.bottom.y + pair.bottom.height);
  }
  topY = seamClamp(Math.floor(topY - SEAM_MARGIN_PX), 0, first.height);
  bottomY = seamClamp(Math.ceil(bottomY + SEAM_MARGIN_PX), 0, last.height);

  let fromTop = first.height - topY;
  let fromBottom = bottomY;
  if (fromTop <= 0 || fromBottom <= 0) return null;

  let middle = 0;
  for (let i = 1; i < count - 1; i += 1) middle += run[i].height;

  /* Grow towards the slices' own shape. Taken evenly from both where both have
   * room, and wholly from one where the other has none -- a bubble cut 40px into
   * the lower slice must still be able to reach a usable height by taking the
   * rest from the upper one.
   *
   * NOT SCALED BY THE RUN, and that is a correction: scaling it made the growth
   * fire on runs that are already three slices tall, adding 19-37% more pixels
   * to reach an aspect the crop had passed several slices ago. The target is one
   * slice's shape because that is the evidence -- every slice of the chapter is
   * being detected successfully at its own aspect. A run is past that target
   * before any padding is considered, so `want` goes negative and nothing grows,
   * which is the correct answer rather than a coincidence.
   *
   * The REFUSAL bound below is the one that has to scale. These were one
   * expression and they are two questions: "what shape should I grow to?" and
   * "is this a bubble or a whole panel?" -- and a run answers the second with
   * evidence the growth has no use for. */
  const tallest = run.reduce((most, edges) => Math.max(most, edges.height), 0);
  const total = run.reduce((sum, edges) => sum + edges.height, 0);
  const ceiling = Math.min(
    Math.max(
      Math.min(tallest, Math.round(width * SEAM_MAX_ASPECT)),
      Math.round(width * SEAM_MIN_ASPECT)
    ),
    total,
    SEAM_MAX_SIDE_PX
  );
  let want = ceiling - (fromTop + middle + fromBottom);
  if (want > 0) {
    const roomTop = topY;
    const roomBottom = last.height - bottomY;
    let up = Math.min(roomTop, Math.ceil(want / 2));
    let down = Math.min(roomBottom, want - up);
    up = Math.min(roomTop, want - down);
    topY -= up;
    bottomY += down;
    fromTop = first.height - topY;
    fromBottom = bottomY;
  }

  const height = fromTop + middle + fromBottom;
  /* Refused rather than sent: the server rejects a side over 8192px with a 400
   * before it does any work, and a bubble that needs more than SEAM_MAX_ASPECT
   * of the width per cut to hold it is not a bubble, it is a whole panel that
   * happened to touch both edges. */
  if (height > SEAM_MAX_SIDE_PX || width > SEAM_MAX_SIDE_PX) return null;
  if (height > Math.round(width * SEAM_MAX_ASPECT * cuts)) return null;

  /* What each slice measured when this was planned, so the side doing the
   * compositing can refuse a mismatch rather than paint at an offset. The
   * translated PNG the rectangle lands on comes out of a 24-hour cache and is
   * only the same picture as long as it is the same size.
   *
   * `offset` is where this slice's band starts inside the seam image, and it is
   * the one number every mapping out of seam coordinates goes through. */
  const slices = [];
  let offset = 0;
  for (let i = 0; i < count; i += 1) {
    const y = i === 0 ? topY : 0;
    const band = i === 0 ? fromTop : i === count - 1 ? fromBottom : run[i].height;
    slices.push({ y, height: band, pageHeight: run[i].height, offset });
    offset += band;
  }

  return {
    width,
    height,
    slices,
    /* Where each cut lands inside the seam image. `boundaries[i]` separates
     * `slices[i]` from `slices[i + 1]`, so there is always one fewer of these
     * than there are slices. */
    boundaries: slices.slice(1).map((slice) => slice.offset),
    pairs: pairs.map((pair) => ({
      cut: pair.cut,
      /* `touch` survives the rebuild because `seamCompositeRects` keys the
       * TOUCHES-class paint witnesses on it -- without it the part clip cuts
       * the paint to the composite's ink box and the reader sees the local
       * lettering twice (the TOUCHES-class doubling). The rebuild stays
       * an allowlist; a probe-synthesized box is the only source of the flag. */
      top: {
        x: pair.top.x,
        y: pair.top.y,
        width: pair.top.width,
        height: pair.top.height,
        ...(pair.top.touch ? { touch: true } : {}),
      },
      bottom: {
        x: pair.bottom.x,
        y: pair.bottom.y,
        width: pair.bottom.width,
        height: pair.bottom.height,
        ...(pair.bottom.touch ? { touch: true } : {}),
      },
      score: pair.score,
    })),
  };
}

/* The ordinary boundary: one cut, two slices. Kept as the name every caller
 * outside the run path already uses, and as a single expression rather than a
 * second implementation, so a two-slice join cannot drift from a four-slice one. */
function seamPlan(a, b) {
  return seamRunPlan([a, b], 0, 2);
}

/* Which of its OWN edges a join repaired, for the slice at `index` of a run of
 * `length`.
 *
 * The head contributed the seam's first band and was cut at its BOTTOM; the tail
 * contributed the last band and was cut at its TOP. The two are deliberately
 * opposite to the band each supplied, and conflating them marks the wrong edge
 * and lets the same boundary be joined again on every revisit.
 *
 * A slice the run passes THROUGH was cut at both, and gets both -- being spanned
 * is what put it in the run. Leaving either unmarked would have every revisit
 * re-run the whole join to repair an edge already repaired.
 *
 * IT LIVES HERE BECAUSE TWO CALLERS NEED THE SAME ANSWER: the content script's
 * in-memory `seamed` list and the cache entry's stored one. Those are read
 * together -- the in-memory list decides whether to ask, the stored one survives
 * the reload -- so a rule written twice is a rule that can disagree with itself
 * about whether a join already happened. */
function seamRunEdges(index, length) {
  if (index === 0) return ["bottom"];
  if (index === length - 1) return ["top"];
  return ["top", "bottom"];
}

const seamOverlaps = (one, two) =>
  one.x < two.x + two.width &&
  two.x < one.x + one.width &&
  one.y < two.y + two.height &&
  two.y < one.y + one.height;

function seamUnion(one, two) {
  const x = Math.min(one.x, two.x);
  const y = Math.min(one.y, two.y);
  return {
    x,
    y,
    width: Math.max(one.x + one.width, two.x + two.width) - x,
    height: Math.max(one.y + one.height, two.y + two.height) - y,
  };
}

const seamPad = (rect) => ({
  x: rect.x - SEAM_MARGIN_PX,
  y: rect.y - SEAM_MARGIN_PX,
  width: rect.width + 2 * SEAM_MARGIN_PX,
  height: rect.height + 2 * SEAM_MARGIN_PX,
});

const seamContains = (outer, inner) =>
  inner.x >= outer.x &&
  inner.y >= outer.y &&
  inner.x + inner.width <= outer.x + outer.width &&
  inner.y + inner.height <= outer.y + outer.height;

/* Which rectangles of the finished seam are worth painting back, and where each
 * one lands on the two slices.
 *
 * NOT the whole seam. The seam is a page in its own right: it is detected,
 * translated and lettered on its own, so the server's page-level size coherence
 * computes a different base from it than the parent slice did, and any bubble
 * that happens to fall inside the crop is re-lettered at a size that belongs to
 * the seam's population rather than the slice's. Painting the whole band back
 * would drag those along and make a correct page inconsistent to fix one bubble.
 * Only the rejoined bubbles are taken.
 *
 * `seamRegions` is what the server reported for the seam image. The rejoined
 * bubble is preferentially found there -- a region that genuinely SPANS the
 * boundary is the seam having done its job, and its box is better than anything
 * derived from the two fragments -- but the union of the fragments is carried as
 * the floor, so a seam whose detection came out differently still paints back a
 * rectangle that covers what it set out to fix. */
/* Does a region run the WHOLE height of this slice, so its text is cut above
 * AND below?
 *
 * `seamEdges` pushes a box into `top` when it touches the top band and into
 * `bottom` when it touches the bottom one, and it pushes the SAME OBJECT into
 * both when a box does both -- so identity, not geometry, is the test, and it
 * cannot disagree with the bands `seamEdges` already applied.
 *
 * # Why this is the whole of run detection
 *
 * The pairwise seam joins slice k to slice k+1. A skill name drawn down the
 * side of a panel can cross three or four of them, and then the middle slices
 * hold a piece with no beginning and no end. Measured on a 173-slice webtoon
 * test chapter: only two slices are crossed end to end, so its runs are THREE
 * slices each.
 *
 * BEWARE what this predicate is actually measuring. It asks whether a DETECTED
 * BOX spans the slice, not whether the ink does -- and on a measured five-slice
 * column those differ on three slices of five: the first two return no column
 * box at all, and the fourth's box stops 124 px above its own ink, so the run
 * this function reports is one slice shorter at each end than the ink.
 *
 * A spanned slice is exactly the signal that a pair is not enough, because a
 * box touching both cuts must continue past both -- no pairwise join can ever
 * make it whole.
 *
 * # It is the same predicate the PIPELINE refuses on, deliberately
 *
 * The server's `cross_slice_fragment` refuses a region at >= 0.95 of the slice
 * height so a fragment is never fabricated into `NO. 1: KUMOMORI!`. That
 * refusal is the stopgap and this is the repair: the same shape, seen from the
 * browser, where the neighbouring slices are actually available. When a run is
 * joined the whole name reaches OCR as one string and the refusal never fires,
 * because the region handed to the server is no longer slice-height. */
function seamSpansSlice(edges) {
  if (!edges || !Array.isArray(edges.top) || !Array.isArray(edges.bottom)) return false;
  return edges.top.some((box) => edges.bottom.includes(box));
}

/* Is this slice crossed by a box the server ADMITTED, rather than one it refused?
 *
 * AMENDMENT 3 -- THE DEFERRAL, AND WHY IT IS DECIDED THIS WAY.
 *
 * `seamJoinFor` returns `{wait: true}` when the name still reaches an end of the
 * chain and another picture could still arrive: joining early would letter half a
 * name and pay for the whole thing twice. That is right for a REGION.
 *
 * It is wrong for a hint, and the failure is silent. A hint is the server saying
 * *"I refused this box"*. If a hint-spanned end deferred, then a strip whose next
 * picture never becomes usable -- lazy-load stalls, the reader stops scrolling,
 * the chapter ends where we did not expect -- would wait forever, and the join
 * the pairwise seam has always made would simply never happen. Nothing in the UI
 * says so. Weighed against that, joining one slice short on refused evidence
 * costs a re-ask; **losing a join is the expensive error**, and
 * `seamJoinFor`'s own header says a deferral recorded as an answer
 * permanently deletes the join it was only meant to postpone.
 *
 * So: a region-spanned end waits, a hint-spanned end does not. The run itself
 * still crosses hint-spanned slices -- `seamRunLength` and `seamJoinFor`'s walk
 * use `seamSpansSlice`, unchanged -- this decides only whether to WAIT at an end
 * of what we currently hold.
 *
 * Two alternatives are rejected: asserting `wait: true` for a hint-spanned end
 * (the silent deferral above), and guarding it with a fifth key on the edges
 * summary, which breaks the exact-keys assertion in `seam.test.js`. The marker
 * already exists on the BOX for amendment 1, so this reads it from there and
 * adds no key to `edges` at all. */
function seamRegionSpansSlice(edges) {
  if (!edges || !Array.isArray(edges.top) || !Array.isArray(edges.bottom)) return false;
  return edges.top.some((box) => !box.hint && edges.bottom.includes(box));
}

/* How many slices this join has to cover, starting at `index`.
 *
 * Walks forward while the NEXT slice is spanned, so a run ends at the first
 * slice whose text stops inside it. Returns 2 for the ordinary boundary, which
 * is what the pairwise seam has always done, and 3+ only where a slice is
 * genuinely crossed end to end.
 *
 * `limit` is a hard stop rather than a guess. A run is cropped into ONE image
 * for the server, and `scene.rs` caps a side at
 * `MAX_SURFACE_DIMENSION.min(MAX_TEXTURE_DIMENSION)` = 8192 px. At ~920 px
 * slices that is eight, and the longest run measured on a webtoon test chapter
 * is THREE -- so the cap is not a limit anyone reaches, it is a guard against a
 * pathological page whose every slice looks spanned turning one join into a
 * whole-chapter upload.
 *
 * Deliberately does NOT ask whether consecutive boxes pair with each other.
 * `seamCandidates` already answers that for the ends of the run, and a spanned
 * middle has nothing to pair WITH -- its box is one object touching both bands,
 * so a pairing test on it compares a box against itself. */
function seamRunLength(edgesList, index, limit) {
  const list = Array.isArray(edgesList) ? edgesList : [];
  const cap = Number.isFinite(limit) && limit >= 2 ? Math.floor(limit) : SEAM_MAX_RUN_SLICES;
  if (index < 0 || index + 1 >= list.length) return 0;
  let length = 2;
  while (
    length < cap &&
    index + length - 1 < list.length &&
    seamSpansSlice(list[index + length - 1])
  ) {
    length += 1;
  }
  return index + length - 1 < list.length ? length : 0;
}

/* A y on one slice, in the seam image's own pixels. Every mapping out of a
 * slice and into the crop goes through this, and its inverse is the `target`
 * arithmetic at the foot of `seamCompositeRects`. */
const seamIntoImage = (slice, y) => y - slice.y + slice.offset;

/* Which run the boundary at `index` belongs to: `{ head, length }`, or null if
 * this is not the moment to join it.
 *
 * `edgesList` is the contiguous, already-translated slices in strip order, and
 * `index` is the upper slice of the boundary that has just become joinable.
 *
 * # A run is anchored at its HEAD, and that is what makes it deterministic
 *
 * The same four-slice name becomes joinable three times -- once per boundary
 * inside it -- and the last slice to arrive decides which. Walking back to the
 * head from wherever we entered means all three arrivals plan the SAME join, so
 * the caller's "have I already asked this?" key matches and the name is
 * translated once. Anchoring anywhere else translates one name twice at two
 * different lengths, and the second overwrites the first.
 *
 * # A LONGER RUN MUST NEVER COST A SHORTER JOIN
 *
 * This is the rule the first three versions of this function each broke in a
 * different place, so it is stated as a rule rather than as three guards: the
 * answer is the LONGEST run containing this boundary THAT ACTUALLY PLANS, and
 * the two-slice pair is the last candidate rather than a special case. A veto at
 * any one cut vetoes its whole run -- `seamRunPlan` requires every cut to be a
 * join in its own right -- so a run that fails must fall back rather than
 * refuse, or widening the search silently deletes joins the pairwise seam has
 * always made.
 *
 * Measured on a webtoon test chapter: slice k+1 is crossed end to end, so a run
 * anchors at slice k -- which has NO box at its own bottom edge, so `k|k+1` is
 * refused "one-sided" and the three-slice run plans null. Falling back finds
 * `k+1|k+2`, which is exactly what shipped before. The same shape occurs at the
 * tail and in the middle of longer runs, which is why searching beats patching
 * the head.
 *
 * # NULL MEANS "TOO EARLY" ONLY WHEN SOMETHING CAN STILL ARRIVE
 *
 * `pending.above` / `pending.below` are the caller's answer to "is there a
 * picture just past this end of the chain that simply has not been translated
 * yet?". When the name runs off an end of what we hold AND such a picture
 * exists, this returns null so the caller can ask again -- joining now would
 * letter a partial name and pay for the whole thing twice.
 *
 * When no such picture exists -- the top or bottom of the document -- there is
 * no later arrival, and deferring loses the join for good. That was a real
 * defect: the first image of a chapter carrying a spanned region got no join at
 * all, where the pairwise seam had joined it. So the distinction is not
 * decoration; it is the difference between waiting and discarding.
 *
 * # THREE ANSWERS, BECAUSE THE CALLER MUST TREAT THEM DIFFERENTLY
 *
 *   { wait: true }              ask again when the next slice arrives
 *   { wait: false, plan: null } nothing crosses here; record it and stop asking
 *   { wait: false, plan, head, length }  join these slices
 *
 * Collapsing the first two into one null is the bug this shape exists to make
 * impossible: the caller records "nothing to join" as SETTLED and never asks
 * again, so a deferral recorded as an answer permanently deletes the join it was
 * only meant to postpone. */
function seamJoinFor(edgesList, index, limit, pending) {
  const list = Array.isArray(edgesList) ? edgesList : [];
  const nothing = { wait: false, plan: null };
  if (index < 0 || index + 1 >= list.length) return nothing;
  const cap =
    Number.isFinite(limit) && limit >= 2 ? Math.floor(limit) : SEAM_MAX_RUN_SLICES;
  const more = pending || {};

  /* How far the name reaches through the slices we actually hold. */
  let first = index;
  while (first > 0 && seamSpansSlice(list[first])) first -= 1;
  let last = index + 1;
  while (last + 1 < list.length && seamSpansSlice(list[last])) last += 1;

  /* Still crossed at an end of the chain: the name continues onto a picture we do
   * not have. Wait for it only if it is there to wait for -- and only on a box
   * the server ADMITTED. See `seamRegionSpansSlice` for why a refused box must
   * not be able to postpone a join indefinitely. */
  if (seamRegionSpansSlice(list[first]) && more.above) return { wait: true, plan: null };
  if (seamRegionSpansSlice(list[last]) && more.below) return { wait: true, plan: null };

  /* The cap is a guard against a pathological page turning one join into a
   * whole-chapter upload, so it is applied around the boundary we were asked
   * about rather than from the top -- truncating from one end only would make
   * which slices survive depend on which way the walk happened to run. */
  while (last - first + 1 > cap) {
    if (last > index + 1) last -= 1;
    else if (first < index) first += 1;
    else break;
  }

  /* Longest first, earliest head to break a tie, and every candidate must hold
   * the boundary we were called for -- that last condition is what makes two
   * entries into one run agree on one join, so the caller's "already asked" key
   * matches and the name is translated once rather than twice. */
  for (let length = last - first + 1; length >= 2; length -= 1) {
    for (let head = first; head + length - 1 <= last; head += 1) {
      if (head > index || head + length - 1 < index + 1) continue;
      const plan = seamRunPlan(list, head, length);
      if (plan) return { wait: false, plan, head, length };
    }
  }
  return nothing;
}

function seamCompositeRects(plan, seamRegions) {
  if (!plan || !plan.pairs || !plan.pairs.length) return [];
  if (!Array.isArray(plan.slices) || !Array.isArray(plan.boundaries)) return [];

  /* A REFUSED region is not swept over, and that is what keeps the cap below
   * from abandoning a join that worked.
   *
   * The sweep exists so a region the rectangle only PARTLY covers gets swallowed
   * whole rather than repainted down the middle. A region the server refused was
   * never lettered, so there is no half of it to repaint and nothing about it to
   * protect -- absorbing it only drags the rectangle across the page to reach a
   * box that carries no text.
   *
   * Measured on a webtoon test boundary, and it is the second gate on one
   * defect. With the OCR area ceiling fixed the server DID read and letter the
   * joined balloon -- `regions=2 layers=1 chars=11 refused=null` on the 766x927
   * box. The reader still saw the two un-joined halves, because the sweep then
   * absorbed the OTHER region on that composite: a site watermark at
   * `[31,399,278,69]`, refused as "a site watermark, not dialogue". Padded, the
   * union spans about 1047x1011 of a 1200x908 crop -- past
   * `SEAM_MAX_RECT_FRACTION` -- so `seamCompositeRects` returned `[]` and the
   * join was thrown away one step from the screen.
   *
   * The balloon alone pads to 814x975 = 0.728 of the crop, so this does NOT by
   * itself put that case under the cap; see the cap's own comment for the rest. */
  const boxes = [];
  for (const region of Array.isArray(seamRegions) ? seamRegions : []) {
    if (region && region.refused) continue;
    const box = seamEffectiveBox(region);
    if (box) boxes.push(box);
  }
  /* THE TOUCH-CLASS WITNESSES -- the TOUCHES-class doubling, diagnosed on a
   * deterministic test window. The part gate and the
   * part clip below witness only the COMPOSITE's region boxes, which for an
   * ordinary join cover the lettering -- the joined text sits where the cut
   * ink was. In the TOUCHES class the ink is wholly on one side, so the
   * composite's detected box under-spans BOTH letterings: the clip cut the
   * paint to the padded ink box, the slice-local first and last lines
   * survived around it (the reader saw the sentence twice), the composite's
   * own first line was halved at the clip edge, and the empty lobe's band
   * held no witness at all, so its part was gated and the joined text could
   * never reach the lobe.
   *
   * The pair boxes ARE the missing witnesses, and they are already in hand:
   * for a touch pair, `pair.bottom` (or `.top`) is the populated side's FIT
   * frame -- exactly the extent the local pass lettered into -- and the
   * synthetic box is the measured lobe. Adding them, scoped to plans that
   * carry a `touch` pair, admits the lobe band and stretches the clip over
   * both letterings; the composite's balloon flat fill supplies the clean
   * pixels that erase the local remnant. An ordinary plan carries no `touch`
   * flag, takes none of this, and stays byte-identical -- which is what keeps
   * the part clip's raw-restore guard (the reason the clip exists) closed
   * where it was earned. */
  const touchWitnesses = [];
  for (const pair of plan.pairs) {
    if (!(pair.top && pair.top.touch) && !(pair.bottom && pair.bottom.touch)) continue;
    const above = plan.slices[pair.cut];
    const below = plan.slices[pair.cut + 1];
    if (!above || !below) continue;
    touchWitnesses.push(
      {
        x: pair.top.x,
        y: seamIntoImage(above, pair.top.y),
        width: pair.top.width,
        height: pair.top.height,
      },
      {
        x: pair.bottom.x,
        y: seamIntoImage(below, pair.bottom.y),
        width: pair.bottom.width,
        height: pair.bottom.height,
      }
    );
  }
  const partWitnesses = touchWitnesses.length ? boxes.concat(touchWitnesses) : boxes;
  /* ANY cut, not the cut. On a run the seam's own detection should find one
   * region crossing two or three of them, and that region is the whole point --
   * it is the first time the name exists as one box anywhere in the pipeline. */
  const spanning = boxes.filter((box) =>
    plan.boundaries.some((cut) => box.y < cut && box.y + box.height > cut)
  );

  let rects = [];
  for (const pair of plan.pairs) {
    const above = plan.slices[pair.cut];
    const below = plan.slices[pair.cut + 1];
    if (!above || !below) continue;
    // Both halves, moved into the seam's own pixel space.
    const fromTop = {
      x: pair.top.x,
      y: seamIntoImage(above, pair.top.y),
      width: pair.top.width,
      height: pair.top.height,
    };
    const fromBottom = {
      x: pair.bottom.x,
      y: seamIntoImage(below, pair.bottom.y),
      width: pair.bottom.width,
      height: pair.bottom.height,
    };
    let rect = seamUnion(fromTop, fromBottom);

    const found = spanning.find((box) => seamOverlaps(box, rect));
    if (found) rect = seamUnion(rect, found);

    rect = seamPad(rect);

    /* A rectangle that cuts a neighbouring region in half is the one way this
     * can look worse than doing nothing: that region was erased and lettered
     * twice, once on the slice and once on the seam, at two different erase
     * radii and two different fitted sizes, and the rectangle's edge would run
     * straight through the difference. So a partially covered region is
     * swallowed whole, with the margin re-applied -- a rectangle whose edge
     * lands exactly on the box it just absorbed has not solved anything.
     *
     * Run to a FIXPOINT, and this is a correction rather than a refinement. It
     * used to be a single forward pass over a rectangle it grew as it went,
     * which is the worst of both: it cascaded anyway (a later box is tested
     * against the already-grown rectangle) while still missing an earlier box
     * the growth passed over, so which regions ended up half-covered depended on
     * the order the server happened to list them in.
     *
     * And a real bound instead of a hoped-for one. If the fixpoint reaches more
     * than SEAM_MAX_RECT_FRACTION of the seam, the join is ABANDONED for that
     * pair: a rectangle that large is repainting most of a page that is already
     * correct, dragging every bubble in it to the seam's own size base, to fix
     * one. Leaving the bubble split is the lesser harm and is what happens
     * today. */
    const cap = plan.width * plan.height * SEAM_MAX_RECT_FRACTION;
    let grew = true;
    let passes = 0;
    while (grew && passes < SEAM_SWEEP_PASSES) {
      grew = false;
      passes += 1;
      for (const box of boxes) {
        if (!seamOverlaps(box, rect) || seamContains(rect, box)) continue;
        /* The BOX is padded, not the result. Padding the result instead makes
         * the sweep non-idempotent -- a box swallowed on pass one and a box
         * swallowed on pass two differ by a margin -- and the rectangle then
         * depends on the order the regions arrived in, which is the fault this
         * loop was rewritten to remove. Union over a fixed set of padded boxes
         * is order-independent by construction. */
        rect = seamUnion(rect, seamPad(box));
        grew = true;
      }
      /* THE CAP PROTECTS OTHER LETTERED CONTENT, so it does not fire when there
       * is none. Its own reason, above, is that a rectangle this large is
       * "repainting most of a page that is already correct, dragging every
       * bubble in it to the seam's own size base, to fix one" -- a statement
       * about COLLATERAL, not about area. A rectangle covering exactly the one
       * region the join went to fetch has no collateral to cause: the only thing
       * it repaints is the thing that was wrong.
       *
       * Measured on that same boundary: the joined balloon is 766x927, and padded
       * it is 0.728 of a 1200x908 crop -- over the 0.6 cap on its own, with the
       * refused watermark already excluded above. The composite carries no other
       * lettered region, so abandoning the join here discarded a read that had
       * already succeeded (`layers=1`, `chars=11`) and left the reader the two
       * un-joined halves, "HEY" and "OVER HERE!", for `快来！`.
       *
       * `boxes` holds only regions the server did NOT refuse, so "covers more
       * than one" is exactly "would drag something else". */
      const area = rect.width * rect.height;
      if (area > cap) {
        /* Over the proxy. Ask the question the proxy stands in for: how much of
         * what we are about to repaint is the repair?
         *
         * A region as large as the whole composite is the detector-error shape
         * `implausible_region` refuses on the server, and it is excluded from the
         * payload rather than being allowed to justify itself -- otherwise one
         * bad box covering everything would always score 1.0 and always paint.
         *
         * Areas are summed rather than unioned. Detection's NMS keeps regions
         * from overlapping much, and any error is in the SAFE direction: an
         * overlap counts twice, inflating the payload, so the only way it can be
         * wrong is by refusing a join it could have taken. */
        const composite = plan.width * plan.height;
        let payload = 0;
        for (const box of boxes) {
          if (!seamOverlaps(box, rect)) continue;
          const covered = box.width * box.height;
          if (covered >= composite) continue;
          payload += covered;
        }
        if (payload < area * SEAM_MIN_RECT_PAYLOAD) return [];
      }
    }

    rects.push(rect);
  }

  // Two bubbles side by side can end up with rectangles that overlap once the
  // margin and the neighbour sweep have been applied. Overlapping paints are
  // idempotent here (both come from the same seam image), but merging keeps the
  // count honest and the compositing cheaper.
  rects.sort((one, two) => one.y - two.y || one.x - two.x);
  const merged = [];
  for (const rect of rects) {
    const last = merged[merged.length - 1];
    if (last && seamOverlaps(last, rect)) merged[merged.length - 1] = seamUnion(last, rect);
    else merged.push(rect);
  }

  const out = [];
  for (const rect of merged) {
    const x = seamClamp(Math.floor(rect.x), 0, plan.width);
    const y = seamClamp(Math.floor(rect.y), 0, plan.height);
    const right = seamClamp(Math.ceil(rect.x + rect.width), 0, plan.width);
    const bottom = seamClamp(Math.ceil(rect.y + rect.height), 0, plan.height);
    if (right <= x || bottom <= y) continue;

    /* Split at every cut it crosses. A rectangle that reaches across none of
     * them still splits cleanly -- every other band simply contributes no part
     * -- and that is the case a seam whose bubble was found entirely inside one
     * slice produces. On a run of four a single rectangle can legitimately
     * produce three or four parts, one per slice it covers.
     *
     * `slice` is an INDEX into `plan.slices`, not the string "top"/"bottom" it
     * used to be. The caller matches it against the slice it is writing back,
     * and a run has no two sides to name. */
    const parts = [];
    for (let index = 0; index < plan.slices.length; index += 1) {
      const band = plan.slices[index];
      const start = Math.max(y, band.offset);
      const end = Math.min(bottom, band.offset + band.height);
      if (end <= start) continue;
      /* THE PART GATE. A part whose window holds no lettered region is not
       * emitted: every pixel it would paint is the composite's own SOURCE, so
       * the only thing painting it can do to a canvas is restore raw ink over
       * lettering some other pass delivered. Painting is cumulative -- a middle
       * slice belongs to two boundaries, and the second join re-reads the bytes
       * the first one wrote (`seamJoin`'s re-read comment owns that rule) -- so
       * a source-restoring part UNDOES its neighbour. Measured on a webtoon
       * test chapter: one boundary's rectangle covered all 972 rows of the
       * upper slice, every lettered region sat in the lower band, and the
       * upper part restored the raw 三重防御 column over the lettering the
       * previous boundary's join had just delivered (80,687 px).
       *
       * RAW boxes, deliberately unpadded: on that same composite the lower-band
       * balloon starts 1.75 px below the cut, so a padded test re-admits the
       * very part this gate exists to drop. A bubble genuinely cut by the
       * boundary needs no padding -- and does not even need its box to cross
       * the cut: the view is the band's whole slab across the rect, so any box
       * that merely TOUCHES a band admits it, and the server's per-line cells
       * each touch their own band. `boxes` already excludes refused regions,
       * so a refusal admits nothing, matching the sweep above.
       *
       * Two things a gated part CAN decline to deliver, both deliberately: a
       * refused region's ERASE (the mask is written from raw detections before
       * OCR, so a refused box arrives blanked -- declining it keeps the
       * slice's own content, which is the safe direction), and up to the erase
       * dilation (~6-23 px) of a halo reaching across the cut from a detection
       * just beyond it. Adversarially replayed over 1,061 stored composites:
       * 0 of 459 gated windows overlapped delivered lettering
       * (rotated placed hulls, not literal rects); 224 overlapped only
       * refused-region blanks. */
      const view = { x, y: start, width: right - x, height: end - start };
      const witnesses = partWitnesses.filter((box) => seamOverlaps(box, view));
      if (!witnesses.length) continue;
      /* THE PART CLIP. Admission used to deliver the band's whole slab, and one
       * overlapping box could admit rows the composite never lettered -- rows
       * that are the composite's own SOURCE, painted over lettering the base
       * render delivered. Measured on a webtoon test chapter: the composite
       * read one region (112,34,122x451), the plan's pair box made the rect span
       * all 919 rows of the lower slice, and the admitted slab restored raw
       * `五灯齐明。` over the base's delivered column -- ~68,874 px un-lettered,
       * the same shape on 24-26 of the chapter's 30 stored composites (the part
       * GATE above only ever asked which bands hold lettering, never how much
       * of an admitted band does).
       *
       * So the part is clipped to the padded union of the boxes that admitted
       * it. PADDED boxes here, deliberately, where the gate above tests RAW
       * ones: the gate's padding trap (a balloon 1.75 px past the cut
       * re-admitting a dropped part) cannot recur because the clip runs only
       * inside a band the raw test already admitted -- and the padding keeps
       * `SEAM_MARGIN_PX` of composite erase around each lettered box, exactly
       * the margin the rect itself carries, so the clip edge does not land on
       * the lettering it protects. Clipping to the region BOXES and not to the
       * lettering hulls is load-bearing the other way: where the base left ink
       * raw, the composite's erase inside its box is real work a hull-tight
       * clip would strip. A box spanning its whole band reproduces the old
       * slab byte-for-byte; the chapter's two pure-loss composites already die
       * at the gate. */
      let union = null;
      for (const box of witnesses) {
        const padded = seamPad(box);
        union = union ? seamUnion(union, padded) : padded;
      }
      const clipX = seamClamp(Math.floor(Math.max(x, union.x)), 0, plan.width);
      const clipRight = seamClamp(
        Math.ceil(Math.min(right, union.x + union.width)),
        0,
        plan.width
      );
      const clipY = Math.max(start, Math.floor(union.y));
      const clipBottom = Math.min(end, Math.ceil(union.y + union.height));
      if (clipRight <= clipX || clipBottom <= clipY) continue;
      parts.push({
        slice: index,
        source: { x: clipX, y: clipY, width: clipRight - clipX, height: clipBottom - clipY },
        target: {
          x: clipX,
          y: band.y + (clipY - band.offset),
          width: clipRight - clipX,
          height: clipBottom - clipY,
        },
      });
    }
    if (parts.length) out.push({ rect: { x, y, width: right - x, height: bottom - y }, parts });
  }
  return out;
}

/* Which background declines ANSWER a boundary, and which merely could not act
 * on this attempt. `performSeam` settles a boundary for the life of the page
 * view on any `ok` reply, and that is right for a reply the server actually
 * produced -- joined, or looked and found nothing to paint -- and wrong for
 * the background's own pre-flight declines, three of which are transient by
 * their own comments: a cache entry can land or age back in ("uncached"), the
 * glossary can stop drifting ("glossary-skew" -- whose own comment says "the
 * boundary can try again on a later view", a promise the unconditional settle
 * was breaking), and re-fetched bytes can decode to the planned picture again
 * ("mismatch"). Only the site profile ("manga-profile") is a fact about the
 * page rather than about the attempt. An unrecognised reason counts as FINAL,
 * because that is what every reason got before this predicate existed -- a new
 * transient reason must be added here to earn its retries. */
const SEAM_SKIP_TRANSIENT = new Set(["uncached", "glossary-skew", "mismatch"]);

function seamSkipFinal(reason) {
  if (!reason) return true;
  return !SEAM_SKIP_TRANSIENT.has(reason);
}

/* Node loads this file directly for `node --test tests/seam.test.js`, which is
 * the only harness any of this arithmetic has. The browser has no `module`, so
 * the guard is a no-op there -- and it is a guard rather than an `export`
 * because seam.js is a CLASSIC script sharing one global with content.js and
 * background.js, exactly as cache.js is, and a module would break that. */
if (typeof module === "object" && module !== null && module.exports) {
  module.exports = {
    SEAM_EDGE_FLOOR_PX,
    SEAM_EDGE_FRACTION,
    SEAM_MARGIN_PX,
    SEAM_MAX_RUN_SLICES,
    SEAM_MIN_ASPECT,
    SEAM_MAX_ASPECT,
    SEAM_MAX_SIDE_PX,
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
    seamRegionSpansSlice,
    seamRunLength,
    seamJoinFor,
    seamCompositeRects,
    seamSkipFinal,
  };
}
