/* The box editor's pure half: coordinate mapping between the
 * displayed image and the source's own pixel space, the cumulative edit
 * merge, and the wire fields for /translate. Loaded as a classic script in
 * BOTH the content script (mapping, merge) and the background page (wire
 * fields) -- seam.js's arrangement, and for the same reason: one definition,
 * two sides, nothing to drift.
 *
 * Everything DOM-shaped stays in content.js; everything here is arithmetic
 * over plain objects so `node --test tests/boxedit.test.js` runs it directly.
 *
 * Boxes are {x, y, width, height} in SOURCE-image pixels -- the space the
 * server reports regions in and accepts edits in, so the round trip needs no
 * server-side transform. Display rects are {left, top, width, height} in the
 * page's CSS pixels.
 */

"use strict";

/* Mirrors the server's CALLER_REGION_CAP (routes.rs). The server refuses past
 * it; refusing HERE means the reader is told at the editor, not by a 400. */
const BOXEDIT_MAX_BOXES = 32;

/* Two source rects are "the same box" within a whisker: the editor sends back
 * the exact rectangles it displayed, but they cross a CSS-scale round trip,
 * so sub-pixel wobble is expected and a hard equality would split one box
 * into an add AND a remove. 2px in source space is far below any real box. */
const BOXEDIT_SAME_TOLERANCE = 2;

/* Mirrors the server's PLACEMENT_MIN_SPAN (routes.rs). Below it the renderer's
 * text inset leaves no room for even the smallest line; the server refuses
 * with a 400, and refusing HERE means the reader is told at the editor. */
const BOXEDIT_MIN_PLACEMENT = 24;

function boxeditToSource(rect, view, naturalWidth, naturalHeight) {
  const scaleX = naturalWidth / view.width;
  const scaleY = naturalHeight / view.height;
  return {
    x: (rect.left - view.left) * scaleX,
    y: (rect.top - view.top) * scaleY,
    width: rect.width * scaleX,
    height: rect.height * scaleY,
  };
}

function boxeditToView(box, view, naturalWidth, naturalHeight) {
  const scaleX = view.width / naturalWidth;
  const scaleY = view.height / naturalHeight;
  return {
    left: view.left + box.x * scaleX,
    top: view.top + box.y * scaleY,
    width: box.width * scaleX,
    height: box.height * scaleY,
  };
}

function boxeditSameRect(a, b, tolerance = BOXEDIT_SAME_TOLERANCE) {
  return (
    Math.abs(a.x - b.x) <= tolerance &&
    Math.abs(a.y - b.y) <= tolerance &&
    Math.abs(a.width - b.width) <= tolerance &&
    Math.abs(a.height - b.height) <= tolerance
  );
}

/* `box`'s centre lies within `rect` -- the server's own matching predicate
 * (remove_caller_boxes, and regions_place targeting), stated once on this
 * side of the wire. Inside, not intersects: an overlap whose centre is
 * elsewhere belongs to another region. */
function boxeditCenterInside(rect, box) {
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  return (
    cx >= rect.x && cx <= rect.x + rect.width &&
    cy >= rect.y && cy <= rect.y + rect.height
  );
}

/* Resize a view rect by one handle. `edge` is one of nw/ne/sw/se/n/s/e/w;
 * `x`/`y` is the pointer in overlay-local pixels; `bounds` the overlay's
 * {width, height}. Owns the flip case -- dragging the west handle past the
 * east edge yields a valid rect, never a negative width -- and clamps to the
 * image, because a box hanging off the page maps to source pixels that do
 * not exist. */
function boxeditResizeRect(rect, edge, x, y, bounds) {
  const px = Math.min(Math.max(x, 0), bounds.width);
  const py = Math.min(Math.max(y, 0), bounds.height);
  let left = rect.left;
  let right = rect.left + rect.width;
  let top = rect.top;
  let bottom = rect.top + rect.height;
  if (edge.includes("w")) left = px;
  if (edge.includes("e")) right = px;
  if (edge.includes("n")) top = py;
  if (edge.includes("s")) bottom = py;
  return {
    left: Math.min(left, right),
    top: Math.min(top, bottom),
    width: Math.abs(right - left),
    height: Math.abs(bottom - top),
  };
}

/* Translate a view rect by a drag delta, size preserved, clamped so the box
 * parks flush at the page edge rather than shrinking or leaving it. The
 * placement box's move: moving the English is never destructive,
 * unlike moving a detection box, which re-reads and re-erases. */
function boxeditMoveRect(rect, dx, dy, bounds) {
  return {
    left: Math.min(Math.max(rect.left + dx, 0), bounds.width - rect.width),
    top: Math.min(Math.max(rect.top + dy, 0), bounds.height - rect.height),
    width: rect.width,
    height: rect.height,
  };
}

/* The target half of a flat wire placement, as a plain rect. */
function boxeditPlaceTarget(entry) {
  return { x: entry.x, y: entry.y, width: entry.width, height: entry.height };
}

/* The cumulative merge. `prior` is the entry's stored edits
 * ({add, remove, place}), `changes` is what this editor session did
 * ({drawn, deleted, resized, placed}, all in source rects; `placed` entries
 * are flat wire-shaped 8-key objects, `resized` entries are {from, to}).
 *
 * The one subtle arm, pinned by its own test: deleting a box that matches a
 * PRIOR ADD leaves the add list -- it must never join the remove list, or the
 * server would add-then-remove that rect on every later run and the stored
 * edits would grow a pair per click. The match is exact-first, centre-inside
 * as the fallback: the server returns a drawn box as its
 * INK-CLAMPED subset, routinely far outside the 2px tolerance, and without
 * the fallback the old add survives and the server re-admits the box the
 * reader just deleted. When one large prior rect contains two regions'
 * centres, exact-match-first keeps the tie deterministic; the residual is a
 * redraw, not an id scheme.
 *
 * A resize is the delete path for `from` plus the draw path for `to`, and it
 * re-targets (never drops) any placement aimed at `from` -- one decision,
 * made here so a test can reach it. A delete DOES drop the region's
 * placement: keeping it would aim the reader's box at whatever lands there
 * later. Placements replace by target rather than append, or every nudge
 * would grow an entry with the server free to apply any of them.
 *
 * Everything dedupes under the same tolerance, and past the cap the merge
 * REFUSES rather than truncates -- a silently dropped correction is the
 * exact failure the editor exists to remove. An under-minimum placement is
 * refused here too, so the reader hears it at the editor rather than as a
 * 400. */
function boxeditMerge(prior, changes) {
  const add = (prior && Array.isArray(prior.add) ? prior.add : []).slice();
  const remove = (prior && Array.isArray(prior.remove) ? prior.remove : []).slice();
  const place = (prior && Array.isArray(prior.place) ? prior.place : []).slice();

  const deleteBox = (gone, dropPlacements) => {
    let i = add.findIndex((box) => boxeditSameRect(box, gone));
    if (i === -1) i = add.findIndex((box) => boxeditCenterInside(box, gone));
    if (i !== -1) {
      add.splice(i, 1);
    } else if (!remove.some((box) => boxeditSameRect(box, gone))) {
      remove.push(gone);
    }
    if (dropPlacements) {
      for (let j = place.length - 1; j >= 0; j--) {
        const target = boxeditPlaceTarget(place[j]);
        if (boxeditSameRect(target, gone) || boxeditCenterInside(target, gone)) {
          place.splice(j, 1);
        }
      }
    }
  };
  const drawBox = (drawn) => {
    if (!add.some((box) => boxeditSameRect(box, drawn))) add.push(drawn);
  };

  for (const { from, to } of changes.resized || []) {
    for (let j = 0; j < place.length; j++) {
      const target = boxeditPlaceTarget(place[j]);
      if (boxeditSameRect(target, from) || boxeditCenterInside(target, from)) {
        place[j] = {
          x: to.x, y: to.y, width: to.width, height: to.height,
          place_x: place[j].place_x, place_y: place[j].place_y,
          place_width: place[j].place_width, place_height: place[j].place_height,
        };
      }
    }
    deleteBox(from, false);
    drawBox(to);
  }
  for (const gone of changes.deleted || []) deleteBox(gone, true);
  for (const drawn of changes.drawn || []) drawBox(drawn);
  /* Unplace BEFORE the placed arm, so remove-then-redraw in one session nets
   * to the redraw. `unplaced` entries are the placement's TARGET rects. */
  for (const gone of changes.unplaced || []) {
    for (let j = place.length - 1; j >= 0; j--) {
      const target = boxeditPlaceTarget(place[j]);
      if (boxeditSameRect(target, gone) || boxeditCenterInside(target, gone)) {
        place.splice(j, 1);
      }
    }
  }
  for (const entry of changes.placed || []) {
    const target = boxeditPlaceTarget(entry);
    const i = place.findIndex((p) => boxeditSameRect(boxeditPlaceTarget(p), target));
    if (i !== -1) place[i] = entry;
    else place.push(entry);
  }

  for (const entry of place) {
    if (
      entry.place_width < BOXEDIT_MIN_PLACEMENT ||
      entry.place_height < BOXEDIT_MIN_PLACEMENT
    ) {
      return {
        error:
          `a placement box must be at least ${BOXEDIT_MIN_PLACEMENT}px each ` +
          "way in source pixels; draw a bigger one",
      };
    }
  }
  if (
    add.length > BOXEDIT_MAX_BOXES ||
    remove.length > BOXEDIT_MAX_BOXES ||
    place.length > BOXEDIT_MAX_BOXES
  ) {
    return {
      error:
        `this image carries too many edits (the cap is ${BOXEDIT_MAX_BOXES} ` +
        "boxes each way); translate it fresh to start over",
    };
  }
  return { add, remove, place };
}

/* The multipart fields for translateForm. A field appears exactly when its
 * list is non-empty: an ordinary retry must not carry an edits payload at
 * all, and the server treats absent as "the detector's answer stands". */
function boxeditWireFields(edits) {
  const fields = {};
  if (edits && Array.isArray(edits.add) && edits.add.length) {
    fields.regions_add = JSON.stringify(edits.add);
  }
  if (edits && Array.isArray(edits.remove) && edits.remove.length) {
    fields.regions_remove = JSON.stringify(edits.remove);
  }
  if (edits && Array.isArray(edits.place) && edits.place.length) {
    fields.regions_place = JSON.stringify(edits.place);
  }
  return fields;
}

if (typeof module !== "undefined") {
  module.exports = {
    BOXEDIT_MAX_BOXES,
    BOXEDIT_MIN_PLACEMENT,
    BOXEDIT_SAME_TOLERANCE,
    boxeditToSource,
    boxeditToView,
    boxeditSameRect,
    boxeditCenterInside,
    boxeditResizeRect,
    boxeditMoveRect,
    boxeditPlaceTarget,
    boxeditMerge,
    boxeditWireFields,
  };
}
