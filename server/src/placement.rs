//! The reader's placement overrides, applied to the finished scene.
//!
//! Runs between the scene gates and the render, on the session the handler
//! still owns -- `lettering.rs`'s seam, and its shape: gather by value from a
//! snapshot, one patch, one commit, best-effort. Nothing here touches a device
//! or a model, and **nothing here touches the Koharu tree**: render is not a
//! `Stage`, it is a direct call after `Pipeline::execute` returns, so a
//! placement never rides the pipeline `Request` at all.
//!
//! The mechanism is the compositor's own: a `Geometry` written onto a text
//! layer IS that layer's frame, which is the fit box the renderer solves the
//! English into (`RegionOut::fit_*` reads the same component back). For an
//! axis-aligned rectangle the balloon-contour branch reduces to the rectangle
//! itself, so the effective boundary is the caller's box inset by
//! `text_inset` -- a tightening, never a loosening.
//!
//! One honest limit, reported rather than hidden: the size search never goes
//! below the theme's 9px floor, so a placement box too small for the line at
//! floor size SPILLS instead of shrinking into illegibility.  `overflowed`
//! below is what makes that loud (`placement_overflow` on the wire); refusing
//! the render or lettering 4px text would each lose the information outright.
//!
//! Its opposite number is `clamp_to_page`, which asks the question
//! `overflowed` structurally cannot: not "did the ink stay in the box" but
//! "was the box on the page at all". That one runs before the pipeline rather
//! than after the render, because it has to be the single site every later
//! reader of a placement sees.

use koharu_scene::{EntityId, Geometry, Session, Snapshot};

/// Whether `box_`'s centre lies inside `target` -- the same predicate
/// `remove_caller_boxes` uses pipeline-side and the editor uses client-side,
/// stated once per side of each seam. Inside, not intersects: an overlap
/// whose centre is elsewhere belongs to another region.
fn center_inside(target: &koharu_pipeline::CallerRegion, cx: f64, cy: f64) -> bool {
    cx >= f64::from(target.x)
        && cx <= f64::from(target.x + target.width)
        && cy >= f64::from(target.y)
        && cy <= f64::from(target.y + target.height)
}

/// Writes each matched region's placement rectangle onto its text layer as
/// the layer's own frame, and returns how many landed.
///
/// Matching is centre-inside on the REGION rect (the recognized-from
/// geometry, which this pass never modifies), first placement wins. A target
/// that matches nothing writes nothing and counts nothing -- visible as
/// `asked != placed` in the log line, the same asked-vs-done discipline the
/// detection splice logs.
///
/// Best-effort by `lettering.rs`'s reasoning: a page whose scene cannot be
/// patched still renders, just in the renderer's own fit boxes. No
/// generation on the edit, so `Edit::prepare_value` stamps `Origin::User`
/// and the write may overwrite the pipeline-authored frame -- the desktop
/// editor's own mechanism.
pub fn apply_caller_placement(
    session: &mut Session,
    page: EntityId,
    placements: &[crate::routes::CallerPlacement],
) -> usize {
    if placements.is_empty() {
        return 0;
    }
    let snapshot = session.snapshot();
    // Reuse the shipped walk: a second "which entity is a region's layer"
    // definition is exactly the kind that rots.
    let regions = crate::regions::regions(&snapshot, page);

    let mut pending: Vec<(EntityId, Geometry)> = Vec::new();
    for region in &regions {
        let cx = region.x + region.width / 2.0;
        let cy = region.y + region.height / 2.0;
        let Some(entry) = placements
            .iter()
            .find(|entry| center_inside(&entry.target, cx, cy))
        else {
            continue;
        };
        pending.push((
            region.layer,
            Geometry::rectangle(
                f64::from(entry.place.x),
                f64::from(entry.place.y),
                f64::from(entry.place.width),
                f64::from(entry.place.height),
            ),
        ));
    }

    let placed = pending.len();
    tracing::info!(
        asked = placements.len(),
        placed,
        page = %page,
        "applying the reader's placement boxes"
    );
    if placed == 0 {
        return 0;
    }

    let patched = snapshot.patch(|edit| {
        for (layer, geometry) in &pending {
            edit.set(*layer, geometry)?;
        }
        Ok(())
    });
    let Ok(patch) = patched else { return 0 };
    if session.commit(patch).is_ok() { placed } else { 0 }
}

/// Gives every settled region whose centre lies in one of the reader's ADDED
/// rects its OWN rect as a reader-authored frame, and returns how
/// many were framed.
///
/// The frame's `Origin::User` is what the renderer keys on: a reader-authored
/// rectangle becomes a hard contour, which routes layout through the balanced
/// balloon search -- so a resized or drawn box re-WRAPS its English for the
/// new shape instead of merely re-sizing it, which would read as "zooming".
/// Runs before `apply_caller_placement`, which overwrites the same
/// component for regions the reader also placed -- the placement wins.
pub fn assert_caller_frames(
    session: &mut Session,
    page: EntityId,
    added: &[koharu_pipeline::CallerRegion],
) -> usize {
    if added.is_empty() {
        return 0;
    }
    let snapshot = session.snapshot();
    let regions = crate::regions::regions(&snapshot, page);

    let mut pending: Vec<(EntityId, Geometry)> = Vec::new();
    for region in &regions {
        let cx = region.x + region.width / 2.0;
        let cy = region.y + region.height / 2.0;
        if !added.iter().any(|rect| center_inside(rect, cx, cy)) {
            continue;
        }
        pending.push((
            region.layer,
            Geometry::rectangle(region.x, region.y, region.width, region.height),
        ));
    }

    let framed = pending.len();
    tracing::info!(
        asked = added.len(),
        framed,
        page = %page,
        "framing the reader's own boxes"
    );
    if framed == 0 {
        return 0;
    }
    let patched = snapshot.patch(|edit| {
        for (layer, geometry) in &pending {
            edit.set(*layer, geometry)?;
        }
        Ok(())
    });
    let Ok(patch) = patched else { return 0 };
    if session.commit(patch).is_ok() { framed } else { 0 }
}

/// One axis of the clamp: where a span of `span` starting at `start` has to
/// begin for it to lie inside `[0, page]`, and how long it may be.
///
/// Size is preserved and the box parks flush at the edge -- the editor's
/// `boxeditMoveRect` semantics, restated on this side of the seam. The one
/// arm that differs is `span >= page`: the editor leaves an oversized box at
/// a negative left (its `bounds.width - rect.width` goes negative and the
/// `min` wins), which is unreachable through the UI because the reader cannot
/// drag past the overlay. Here it shrinks onto the page instead, because a
/// server is not the UI and the visible box beats the faithful one.
///
/// The early return is also what makes the `clamp` below sound: `page - span`
/// is negative exactly when `span > page`, and `f32::clamp` panics on
/// `min > max`. Non-finite spans cannot arrive -- `parse_caller_placements`
/// refuses them at the edge -- and a page's own dimensions are validated
/// finite and positive by `Page::validate`.
fn clamp_axis(start: f32, span: f32, page: f32) -> (f32, f32) {
    if span >= page {
        return (0.0, page);
    }
    (start.clamp(0.0, page - span), span)
}

/// Parks every placement rectangle inside the page's own raster and returns
/// the indices of the ones that had to move.
///
/// **Hardening, not a live bug.** The editor already clamps a dragged box to
/// the page (`extension/boxedit.js`, `boxeditMoveRect` -- "parks flush at the
/// page edge"), so nothing a reader can do in the UI today produces an
/// off-page `place` rect. This is the server declining to trust that, and it
/// is worth the twenty lines because of how the failure LOOKS when it does
/// arrive: measured on an 849x1198 page, a place rect at `y=1159 height=318`
/// -- 279 px past the bottom -- lettered six lines of dialogue off the canvas
/// and left a single visible letter at the page edge, while the run logged
/// `asked=1 placed=1` and reported `placement_overflow: []`. Silent, total
/// destruction of the English, with every channel reading clean.
///
/// `placement_overflow` structurally cannot catch it, and that is not a gap in
/// it: `overflowed` asks whether the placed INK stayed inside the placement
/// BOX, and it did -- the box itself was off the page. Nothing anywhere asked
/// whether the box was on the canvas.
///
/// Runs at the TOP of `translate_page`, before the pipeline, which is the one
/// place with both the reader's list and the page: every later reader --
/// `apply_caller_placement`'s frame write and `overflowed`'s containment test
/// -- then sees the clamped rectangle, and there is no second clamp site to
/// disagree with this one. Clamping later, inside the placement pass, would
/// leave `overflowed` comparing the placed ink against the ORIGINAL off-page
/// box and reporting a spill that no longer exists.
///
/// Only `place` moves. `target` is the matcher -- the region rect the editor
/// displayed, tested centre-inside against the settled regions -- and moving
/// it would aim the reader's box at a different region, or at none.
///
/// A page with no `Page` component cannot happen (`prepare_page` writes one
/// before anything else sees the session) and is reported rather than
/// guessed at: nothing is clamped, and the log says why.
pub fn clamp_to_page(
    snapshot: &Snapshot,
    page: EntityId,
    placements: &mut [crate::routes::CallerPlacement],
) -> Vec<usize> {
    if placements.is_empty() {
        return Vec::new();
    }
    let Ok(raster) = snapshot.page(page).and_then(koharu_scene::PageRef::page) else {
        tracing::warn!(
            page = %page,
            asked = placements.len(),
            "no page component to clamp placements against; leaving them as sent"
        );
        return Vec::new();
    };
    let (page_width, page_height) = (raster.width as f32, raster.height as f32);

    let mut out = Vec::new();
    for (index, entry) in placements.iter_mut().enumerate() {
        let (x, width) = clamp_axis(entry.place.x, entry.place.width, page_width);
        let (y, height) = clamp_axis(entry.place.y, entry.place.height, page_height);
        if (x, y, width, height)
            == (
                entry.place.x,
                entry.place.y,
                entry.place.width,
                entry.place.height,
            )
        {
            continue;
        }
        /* WARN, not INFO: the editor clamps client-side, so a box arriving off
         * the page means a caller that is not this extension, or an extension
         * whose clamp broke. Both are worth a line naming the numbers. */
        tracing::warn!(
            index,
            page = %page,
            was = ?(entry.place.x, entry.place.y, entry.place.width, entry.place.height),
            now = ?(x, y, width, height),
            page_size = ?(page_width, page_height),
            "a placement box left the page; parking it on the canvas"
        );
        entry.place.x = x;
        entry.place.y = y;
        entry.place.width = width;
        entry.place.height = height;
        out.push(index);
    }
    out
}

/// Region indices whose placed ink did not stay inside their placement box,
/// within half a pixel -- the report that makes the 9px-floor spill loud.
///
/// Keyed on `RenderedText::region`, so it must run AFTER
/// `attribute_rendered_text`. A region with no placement is never reported
/// here whatever its extent -- this is not a general overflow counter, which
/// `layout_warnings` already is. The placed extent is rotated about its own
/// centre when `placed_angle` is non-zero (the column-turn population, the
/// one exception to never rotating the English), so a turned line is judged by
/// its true axis-aligned envelope rather than its un-rotated one.
pub fn overflowed(
    regions: &[crate::regions::RegionOut],
    text: &[crate::render::RenderedText],
    placements: &[crate::routes::CallerPlacement],
) -> Vec<usize> {
    if placements.is_empty() {
        return Vec::new();
    }
    const SLACK: f32 = 0.5;
    let mut out = Vec::new();
    for entry in text {
        let Some(index) = entry.region else { continue };
        let Some(region) = regions.get(index) else { continue };
        let cx = region.x + region.width / 2.0;
        let cy = region.y + region.height / 2.0;
        let Some(placement) = placements
            .iter()
            .find(|placement| center_inside(&placement.target, cx, cy))
        else {
            continue;
        };
        // The axis-aligned envelope of the placed extent at its angle.
        let angle = f32::to_radians(entry.placed_angle);
        let (sin, cos) = (angle.sin().abs(), angle.cos().abs());
        let half_w = cos * entry.placed_width / 2.0 + sin * entry.placed_height / 2.0;
        let half_h = sin * entry.placed_width / 2.0 + cos * entry.placed_height / 2.0;
        let center_x = entry.placed_x + entry.placed_width / 2.0;
        let center_y = entry.placed_y + entry.placed_height / 2.0;
        let place = &placement.place;
        let inside = center_x - half_w >= place.x - SLACK
            && center_y - half_h >= place.y - SLACK
            && center_x + half_w <= place.x + place.width + SLACK
            && center_y + half_h <= place.y + place.height + SLACK;
        if !inside {
            out.push(index);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regions::RegionOut;
    use crate::render::RenderedText;
    use crate::routes::CallerPlacement;

    fn region_at(x: f64, y: f64, width: f64, height: f64) -> RegionOut {
        RegionOut {
            content: EntityId::new(),
            layer: EntityId::new(),
            x,
            y,
            width,
            height,
            source: String::new(),
            translated: String::new(),
            fit_x: None,
            fit_y: None,
            fit_width: None,
            fit_height: None,
            source_language: None,
            target_language: None,
            detection_confidence: None,
            ocr_confidence: None,
            direction: None,
            writing_mode: None,
            region_kind: None,
            label: None,
            role: None,
            font_size: None,
            color: None,
            stroke_color: None,
            stroke_width: None,
            font_weight: None,
            refused: None,
            occluded_by: None,
        }
    }

    fn text_at(region: usize, x: f32, y: f32, width: f32, height: f32) -> RenderedText {
        RenderedText {
            entity: EntityId::new(),
            region: Some(region),
            font_size: 9.0,
            chars: 12,
            x,
            y,
            width,
            height,
            placed_x: x,
            placed_y: y,
            placed_width: width,
            placed_height: height,
            placed_angle: 0.0,
        }
    }

    fn placement(target: &RegionOut, place: (f32, f32, f32, f32)) -> CallerPlacement {
        CallerPlacement {
            target: koharu_pipeline::CallerRegion {
                x: target.x as f32,
                y: target.y as f32,
                width: target.width as f32,
                height: target.height as f32,
            },
            place: koharu_pipeline::CallerRegion {
                x: place.0,
                y: place.1,
                width: place.2,
                height: place.3,
            },
        }
    }

    /// The third arm matters most: a region with no placement is never
    /// reported here whatever its extent -- this must not become a general
    /// overflow counter, which `layout_warnings` already is.
    #[test]
    fn placement_overflow_names_the_regions_whose_ink_left_the_box() {
        let regions = vec![
            region_at(100.0, 100.0, 80.0, 60.0),
            region_at(400.0, 100.0, 80.0, 60.0),
            region_at(700.0, 100.0, 80.0, 60.0),
        ];
        let placements = vec![
            placement(&regions[0], (90.0, 200.0, 200.0, 100.0)),
            placement(&regions[1], (390.0, 200.0, 200.0, 100.0)),
        ];
        let text = vec![
            // Inside its box: absent from the report.
            text_at(0, 100.0, 210.0, 180.0, 80.0),
            // 40px past its box's right edge: present.
            text_at(1, 400.0, 210.0, 230.0, 80.0),
            // No placement at all, wildly overflowing everything: never present.
            text_at(2, 0.0, 0.0, 2000.0, 2000.0),
        ];
        assert_eq!(overflowed(&regions, &text, &placements), vec![1]);
        assert_eq!(
            overflowed(&regions, &text, &[]),
            Vec::<usize>::new(),
            "no placements, no report -- whatever the extents"
        );
    }

    /// A page of the measured shape, with nothing on it: the clamp reads the
    /// `Page` component and the placements, and touches no layer.
    fn page_sized(width: f64, height: f64) -> (Session, EntityId) {
        let mut session = Session::memory().expect("an in-memory session");
        let mut id = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                id = Some(edit.add_page(
                    koharu_scene::PageDraft::new("page", width, height),
                    koharu_scene::At::End,
                )?);
                Ok(())
            })
            .expect("the page is valid");
        session.commit(patch).expect("the page commits");
        (session, id.expect("the edit ran to completion"))
    }

    fn place_of(entry: &CallerPlacement) -> (f32, f32, f32, f32) {
        (
            entry.place.x,
            entry.place.y,
            entry.place.width,
            entry.place.height,
        )
    }

    fn target_of(entry: &CallerPlacement) -> (f32, f32, f32, f32) {
        (
            entry.target.x,
            entry.target.y,
            entry.target.width,
            entry.target.height,
        )
    }

    /// Measured on an 849x1198 page: a place
    /// rect at `y=1159 height=318` lettered SIX LINES below the bottom of the
    /// canvas, leaving one visible letter at the edge -- and the run reported
    /// `placed=1` with `placement_overflow: []`, because the ink stayed
    /// perfectly inside the box and the box was off the page.
    ///
    /// Composed on purpose: the page dimensions are read from the scene INSIDE
    /// the tested unit, so this cannot pass against a clamp handed the right
    /// numbers by a caller that reads the wrong ones. Size is preserved and the
    /// box parks flush at the edge -- `boxeditMoveRect`'s semantics, stated
    /// once per side of the seam like `center_inside` above.
    #[test]
    fn an_off_page_placement_is_parked_on_the_page_and_reported() {
        let (session, page) = page_sized(849.0, 1198.0);
        let mut placements = vec![
            // The measured one: 279 px past the bottom.
            placement(&region_at(0.0, 0.0, 10.0, 10.0), (100.0, 1159.0, 300.0, 318.0)),
            // Wholly on the page: untouched, and absent from the report.
            placement(&region_at(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 300.0, 200.0)),
            // Past the left edge.
            placement(&region_at(0.0, 0.0, 10.0, 10.0), (-40.0, 500.0, 300.0, 200.0)),
            // Past the right edge: 700 + 300 = 1000 against a 849-wide page.
            placement(&region_at(0.0, 0.0, 10.0, 10.0), (700.0, 500.0, 300.0, 200.0)),
        ];
        let clamped = clamp_to_page(&session.snapshot(), page, &mut placements);
        assert_eq!(clamped, vec![0, 2, 3], "the three that had to move, by index");
        assert_eq!(
            place_of(&placements[0]),
            (100.0, 880.0, 300.0, 318.0),
            "parked flush at the bottom with its size intact"
        );
        assert_eq!(
            place_of(&placements[1]),
            (100.0, 100.0, 300.0, 200.0),
            "a box already on the page is not nudged"
        );
        assert_eq!(place_of(&placements[2]), (0.0, 500.0, 300.0, 200.0));
        assert_eq!(place_of(&placements[3]), (549.0, 500.0, 300.0, 200.0));
        assert_eq!(
            target_of(&placements[0]),
            (0.0, 0.0, 10.0, 10.0),
            "the TARGET is the matcher and is never moved -- clamping it would \
             aim the reader's box at a different region"
        );
        assert!(
            clamp_to_page(&session.snapshot(), page, &mut placements).is_empty(),
            "idempotent: a second pass over clamped boxes reports nothing"
        );
    }

    /// The one arm where size cannot be preserved. The editor's own
    /// `boxeditMoveRect` leaves an oversized box at a negative left rather than
    /// shrinking it, which is unreachable through the UI -- the reader cannot
    /// draw past the overlay -- so this side of the seam picks the answer that
    /// keeps the lettering visible instead of the one that matches it.
    #[test]
    fn a_placement_larger_than_the_page_shrinks_onto_it() {
        let (session, page) = page_sized(849.0, 1198.0);
        let mut placements = vec![placement(
            &region_at(0.0, 0.0, 10.0, 10.0),
            (-30.0, -50.0, 900.0, 1400.0),
        )];
        let clamped = clamp_to_page(&session.snapshot(), page, &mut placements);
        assert_eq!(clamped, vec![0]);
        assert_eq!(place_of(&placements[0]), (0.0, 0.0, 849.0, 1198.0));
    }

    /// A turned line is judged by its true axis-aligned envelope: at 90
    /// degrees a placed extent taller than the box is wide overflows even
    /// though its un-rotated rectangle would fit.
    #[test]
    fn a_turned_line_is_judged_by_its_rotated_envelope() {
        let regions = vec![region_at(100.0, 100.0, 80.0, 60.0)];
        let placements = vec![placement(&regions[0], (100.0, 200.0, 60.0, 200.0))];
        // 40x180 fits the 60x200 box upright; at 90 degrees its envelope is
        // 180x40, which cannot fit a 60-wide box.
        let mut turned = text_at(0, 110.0, 210.0, 40.0, 180.0);
        turned.placed_angle = 90.0;
        assert_eq!(overflowed(&regions, &[turned], &placements), vec![0]);
        let upright = text_at(0, 110.0, 210.0, 40.0, 180.0);
        assert_eq!(overflowed(&regions, &[upright], &placements), Vec::<usize>::new());
    }
}
