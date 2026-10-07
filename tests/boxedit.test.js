/* Unit tests for the box editor's pure half: coordinate mapping, the edit
 * merge, and the wire fields.
 *
 * Run:  node --test tests/boxedit.test.js
 *
 * (Not `node --test tests/` -- see the header of seam.test.js for why a bare
 * directory argument reports a green suite failing for an unrelated reason.)
 *
 * WHY THIS FILE EXISTS. The editor's DOM half is glue; everything
 * that can be WRONG lives here: a display->source mapping off by the CSS
 * scale letters the correction in the wrong place, a merge that appends a
 * deleted previously-added box to the REMOVE list instead of dropping it
 * from ADD makes the server add-then-remove forever, and a wire field sent
 * when empty would make every ordinary retry carry an edits payload. All
 * born red against the absent module.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const {
  BOXEDIT_MAX_BOXES,
  BOXEDIT_MIN_PLACEMENT,
  boxeditToSource,
  boxeditToView,
  boxeditSameRect,
  boxeditCenterInside,
  boxeditResizeRect,
  boxeditMoveRect,
  boxeditMerge,
  boxeditWireFields,
} = require(path.join(ROOT, "extension", "boxedit.js"));

/* A 1000x2000 source displayed at half size, offset on the page. */
const VIEW = { left: 40, top: 100, width: 500, height: 1000 };
const NATURAL = { width: 1000, height: 2000 };

test("display and source round-trip through the CSS scale", () => {
  const drawn = { left: 140, top: 350, width: 50, height: 30 };
  const source = boxeditToSource(drawn, VIEW, NATURAL.width, NATURAL.height);
  assert.deepEqual(source, { x: 200, y: 500, width: 100, height: 60 });
  const back = boxeditToView(source, VIEW, NATURAL.width, NATURAL.height);
  assert.deepEqual(back, { left: 140, top: 350, width: 50, height: 30 });
});

test("rect identity tolerates sub-pixel wobble and nothing more", () => {
  const a = { x: 100, y: 200, width: 50, height: 30 };
  assert.ok(boxeditSameRect(a, { x: 100.9, y: 199.2, width: 50.5, height: 30.4 }));
  assert.ok(!boxeditSameRect(a, { x: 104, y: 200, width: 50, height: 30 }));
});

test("deleting a previously-ADDED box drops it from add, never joins remove", () => {
  const prior = { add: [{ x: 10, y: 10, width: 40, height: 20 }], remove: [] };
  const merged = boxeditMerge(prior, {
    drawn: [],
    deleted: [{ x: 10.4, y: 9.8, width: 40.2, height: 20 }],
  });
  assert.deepEqual(merged, { add: [], remove: [], place: [] });
});

test("deleting a network box joins remove; drawing joins add; both dedupe", () => {
  const prior = { add: [], remove: [{ x: 500, y: 500, width: 60, height: 60 }] };
  const merged = boxeditMerge(prior, {
    drawn: [
      { x: 10, y: 10, width: 40, height: 20 },
      { x: 10.2, y: 10.1, width: 40, height: 20 }, // the same box, wobbled
    ],
    deleted: [
      { x: 500.3, y: 500, width: 60, height: 59.8 }, // already removed
      { x: 700, y: 700, width: 30, height: 30 },
    ],
  });
  assert.equal(merged.add.length, 1);
  assert.equal(merged.remove.length, 2);
});

test("the caps refuse rather than truncate", () => {
  const many = Array.from({ length: BOXEDIT_MAX_BOXES + 1 }, (_, i) => ({
    x: i * 100,
    y: 10,
    width: 40,
    height: 20,
  }));
  const merged = boxeditMerge({ add: [], remove: [] }, { drawn: many, deleted: [] });
  assert.ok(merged.error, "an over-cap edit set must be refused, never truncated");
});

test("wire fields appear exactly when their list is non-empty", () => {
  assert.deepEqual(boxeditWireFields({ add: [], remove: [] }), {});
  assert.deepEqual(boxeditWireFields(null), {});
  const fields = boxeditWireFields({
    add: [{ x: 1.25, y: 2, width: 3, height: 4 }],
    remove: [],
  });
  assert.equal(fields.regions_remove, undefined);
  const parsed = JSON.parse(fields.regions_add);
  assert.deepEqual(parsed, [{ x: 1.25, y: 2, width: 3, height: 4 }]);
});

/* ---- resize + hard-boundary placement ---- */

/* A flat wire-shaped placement: the first four keys are the TARGET (the
 * region rect the editor displayed, matched server-side by centre-inside),
 * the place_* half is the hard boundary for that region's English. */
function placement(target, place) {
  return {
    x: target.x, y: target.y, width: target.width, height: target.height,
    place_x: place.x, place_y: place.y,
    place_width: place.width, place_height: place.height,
  };
}

test("a resize becomes one remove and one add", () => {
  const from = { x: 100, y: 100, width: 50, height: 40 };
  const to = { x: 90, y: 95, width: 80, height: 50 };
  const merged = boxeditMerge(null, { resized: [{ from, to }] });
  assert.deepEqual(merged, { add: [to], remove: [from], place: [] });
});

test("resizing a previously-added box leaves exactly one add", () => {
  /* `from` is the INK-CLAMPED region rect the server actually returned for a
   * drawn box -- far more than 2px inside the drawn rect, so the sameRect arm
   * misses and only centre-inside can find the prior add. Without it the old
   * add survives beside a new remove and the server re-admits the old box. */
  const prior = { add: [{ x: 100, y: 100, width: 200, height: 200 }], remove: [] };
  const from = { x: 118, y: 126, width: 161, height: 148 };
  const to = { x: 90, y: 90, width: 240, height: 240 };
  const merged = boxeditMerge(prior, { resized: [{ from, to }] });
  assert.deepEqual(merged.add, [to]);
  assert.deepEqual(merged.remove, []);
});

test("deleting an ink-clamped previously-added box no longer resurrects it", () => {
  /* The same defect on the plain delete path: the region rect
   * the server returns for a drawn box misses the ±2px match, the old drawn
   * rect stayed in add, and the server re-admitted it on the next apply. */
  const prior = { add: [{ x: 100, y: 100, width: 200, height: 200 }], remove: [] };
  const merged = boxeditMerge(prior, {
    deleted: [{ x: 118, y: 126, width: 161, height: 148 }],
  });
  assert.deepEqual(merged, { add: [], remove: [], place: [] });
});

test("a placement replaces its predecessor rather than appending", () => {
  const target = { x: 400, y: 100, width: 180, height: 240 };
  const first = placement(target, { x: 420, y: 300, width: 300, height: 160 });
  const second = placement(target, { x: 430, y: 320, width: 280, height: 150 });
  const merged = boxeditMerge(null, { placed: [first, second] });
  assert.equal(merged.place.length, 1);
  assert.deepEqual(merged.place[0], second);
});

test("deleting a region drops its placement", () => {
  const target = { x: 400, y: 100, width: 180, height: 240 };
  const prior = {
    add: [], remove: [],
    place: [placement(target, { x: 420, y: 300, width: 300, height: 160 })],
  };
  const exact = boxeditMerge(prior, { deleted: [target] });
  assert.deepEqual(exact.place, []);
  /* The centre-inside variant: the deleted rect is the target jittered 6px --
   * past the sameRect tolerance, but its centre is still inside the target. */
  const jittered = boxeditMerge(prior, {
    deleted: [{ x: 406, y: 106, width: 180, height: 240 }],
  });
  assert.deepEqual(jittered.place, []);
});

test("resizing a region re-targets its placement", () => {
  const from = { x: 400, y: 100, width: 180, height: 240 };
  const to = { x: 380, y: 90, width: 220, height: 260 };
  const box = { x: 420, y: 300, width: 300, height: 160 };
  const prior = { add: [], remove: [], place: [placement(from, box)] };
  const merged = boxeditMerge(prior, { resized: [{ from, to }] });
  assert.equal(merged.place.length, 1);
  assert.deepEqual(merged.place[0], placement(to, box));
});

test("removing a placement takes it out of the stored edits", () => {
  const target = { x: 400, y: 100, width: 180, height: 240 };
  const prior = {
    add: [], remove: [],
    place: [placement(target, { x: 420, y: 300, width: 300, height: 160 })],
  };
  const merged = boxeditMerge(prior, { unplaced: [target] });
  assert.deepEqual(merged, { add: [], remove: [], place: [] });
  /* Remove-then-redraw in one session: the unplace must not eat the redraw. */
  const redrawn = boxeditMerge(prior, {
    unplaced: [target],
    placed: [placement(target, { x: 10, y: 10, width: 100, height: 100 })],
  });
  assert.equal(redrawn.place.length, 1);
  assert.equal(redrawn.place[0].place_x, 10);
});

test("the place list refuses past the cap like the others", () => {
  const many = Array.from({ length: BOXEDIT_MAX_BOXES + 1 }, (_, i) =>
    placement(
      { x: i * 100, y: 10, width: 40, height: 30 },
      { x: i * 100, y: 60, width: 40, height: 30 }
    )
  );
  const merged = boxeditMerge(null, { placed: many });
  assert.ok(merged.error, "an over-cap place list must be refused, never truncated");
  assert.equal(merged.place, undefined);
});

test("a placement under the minimum span is refused in the editor", () => {
  const merged = boxeditMerge(null, {
    placed: [
      placement(
        { x: 400, y: 100, width: 180, height: 240 },
        { x: 420, y: 300, width: BOXEDIT_MIN_PLACEMENT - 4, height: 160 }
      ),
    ],
  });
  assert.ok(merged.error, "an under-minimum placement must be refused here, not by a 400");
  assert.ok(
    String(merged.error).includes(String(BOXEDIT_MIN_PLACEMENT)),
    "the refusal names the minimum"
  );
});

test("wire fields carry regions_place only when non-empty", () => {
  assert.deepEqual(boxeditWireFields({ add: [], remove: [], place: [] }), {});
  const one = placement(
    { x: 410, y: 120, width: 180, height: 240 },
    { x: 430, y: 300, width: 300, height: 160 }
  );
  const fields = boxeditWireFields({ add: [], remove: [], place: [one] });
  assert.equal(fields.regions_add, undefined);
  assert.equal(fields.regions_remove, undefined);
  assert.deepEqual(JSON.parse(fields.regions_place), [one]);
});

test("a handle drag past the opposite edge normalizes rather than inverting", () => {
  const rect = { left: 100, top: 100, width: 50, height: 50 };
  const bounds = { width: 500, height: 500 };
  const flipped = boxeditResizeRect(rect, "w", 200, 120, bounds);
  assert.ok(flipped.width > 0, "width stays positive through the flip");
  assert.equal(flipped.left, 150);
  assert.equal(flipped.width, 50);
  const clamped = boxeditResizeRect(rect, "nw", -30, -30, bounds);
  assert.equal(clamped.left, 0);
  assert.equal(clamped.top, 0);
  /* A corner handle moves both axes; an edge handle moves one. */
  const north = boxeditResizeRect(rect, "n", 300, 80, bounds);
  assert.equal(north.left, 100);
  assert.equal(north.width, 50);
  assert.equal(north.top, 80);
  assert.equal(north.height, 70);
});

test("a moved rect keeps its size and stops at the page edge", () => {
  const rect = { left: 100, top: 100, width: 50, height: 40 };
  const bounds = { width: 500, height: 500 };
  const moved = boxeditMoveRect(rect, 30, -20, bounds);
  assert.deepEqual(moved, { left: 130, top: 80, width: 50, height: 40 });
  /* Clamped, size preserved: a drag past the edge parks flush, never shrinks
   * and never leaves the page. */
  const parked = boxeditMoveRect(rect, 10000, 10000, bounds);
  assert.deepEqual(parked, { left: 450, top: 460, width: 50, height: 40 });
  const cornered = boxeditMoveRect(rect, -10000, -10000, bounds);
  assert.deepEqual(cornered, { left: 0, top: 0, width: 50, height: 40 });
});

test("centre-inside is inside, not intersects", () => {
  const rect = { x: 100, y: 100, width: 100, height: 100 };
  assert.ok(boxeditCenterInside(rect, { x: 140, y: 140, width: 20, height: 20 }));
  /* Overlapping, but the box's centre lies outside the rect. */
  assert.ok(!boxeditCenterInside(rect, { x: 180, y: 180, width: 100, height: 100 }));
});
