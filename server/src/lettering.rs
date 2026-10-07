//! Comic typesetting conventions applied to the finished scene.
//!
//! Runs between `Pipeline::execute` and the render, on the session the handler
//! still owns. Nothing here touches a device or a model: it is a pure rewrite of
//! text the pipeline has already produced.
//!
//! Written as a scene patch rather than as a Koharu change on purpose. The
//! pipeline's job is to read a page and translate it; how the result is *set* is
//! a house style, and BireLate is the thing that has one. Keeping it here also
//! keeps the change out of the Koharu fork.

use koharu_scene::{EntityId, Session, TextLayout, Translation, Typography, WritingMode};

/// Traditional comic dialogue is lettered in capitals, and the fonts the genre
/// uses are drawn caps-first. Applied to **dialogue only**: a narration box, a
/// signboard, a caption and the author credit are ordinary prose and shouting
/// them is wrong. That distinction is exactly the `dev.koharu.text.dialogue`
/// role `link_dialogue_regions` assigns to text it could place inside a bubble,
/// which until now nothing read.
const DIALOGUE_ROLE: &str = "dev.koharu.text.dialogue";

/// Uppercases the translated dialogue on `page`, in place.
///
/// Best-effort by design: a page whose scene cannot be patched still renders,
/// just in mixed case. A typesetting preference is not worth failing a
/// translation the user has already paid a GPU run for.
pub fn uppercase_dialogue(session: &mut Session, page: EntityId) -> bool {
    let snapshot = session.snapshot();
    let Ok(descendants) = snapshot.descendants(page) else {
        return false;
    };

    /* Gathered BEFORE the patch is opened, and by value. The edit closure below
     * borrows the snapshot it is derived from, and walking the tree inside it
     * while also writing to it is the kind of thing that reads fine and then
     * aliases. Same shape as `regions.rs`: a text layer is the entity carrying a
     * TextLayout -- content entities and the inpainting cleanup layer do not. */
    let mut pending: Vec<(EntityId, Translation)> = Vec::new();
    for entity in descendants {
        if !matches!(entity.component::<TextLayout>(), Ok(Some(_))) {
            continue;
        }
        let Ok(layer) = snapshot.text_layer(entity.id()) else {
            continue;
        };
        let Ok(content) = layer.content() else { continue };

        let is_dialogue = content
            .role()
            .ok()
            .flatten()
            .is_some_and(|role| role.role == DIALOGUE_ROLE);
        if !is_dialogue {
            continue;
        }

        let Ok(Some(translation)) = content.translation() else {
            continue;
        };
        let upper = translation.text.value.to_uppercase();
        // Nothing to say for digits, punctuation, or a line already capitalised
        // -- and skipping them keeps the patch empty on a page that needs none,
        // which is the difference between a no-op and a pointless revision.
        if upper == translation.text.value {
            continue;
        }

        let mut next = translation.clone();
        next.text.value = upper;
        pending.push((content.id(), next));
    }

    if pending.is_empty() {
        return false;
    }

    /* No generation, so `Edit::prepare_value` stamps `Origin::User` and
     * `validate_origin_removal` returns early -- which is what permits
     * overwriting a component the detection stage authored. The same mechanism
     * the desktop editor uses when a human retypes a line. */
    let patched = snapshot.patch(|edit| {
        for (content, translation) in &pending {
            edit.set(*content, translation)?;
        }
        Ok(())
    });
    let Ok(patch) = patched else { return false };
    session.commit(patch).is_ok()
}

/// The halo width for in-bubble text, as a fraction of the layer's own size.
///
/// A **ratio**, never an absolute width, and that is forced rather than
/// preferred: the renderer (`text_renderer.rs`) scales every stroke by
/// `layout.font_size / layer.font_size` after auto-fit solves the layout, so a
/// width authored as `size * R` arrives as `rendered * R` exactly, while a fixed
/// number arrives divided by the layer's size. It also means detection's
/// `.max(0.5)` halo floor is not expressible here -- the scaling happens after it
/// and nothing re-clamps.
///
/// 0.05 rather than the 0.03 detection uses for free-standing text. The bound is
/// Arial's counters: the renderer strokes centred at `width * 2`, so the halo
/// reaches a full `width` outward from every contour, an `e` eye of about
/// 0.12 em is closed from both sides at once, and the aperture shuts at
/// `R = 0.06`. 0.05 keeps about a sixth of the eye open. The reason for spending
/// that margin is that the halo has to be a whole pixel wide to read as a rim
/// rather than a grey tint, and measured on real pages auto-fit renders in-bubble
/// English at a median of 17.6px -- where 0.03 is 0.53px and invisible.
pub(crate) const IN_BUBBLE_HALO_RATIO: f32 = 0.05;

/// White. A balloon interior is a flat light fill and `infer_text_color` snaps
/// the lettering it measured to pure black or white, so on the ordinary
/// black-on-white balloon this is the colour the glyphs have to stand out from.
/// On the rare white-on-black one it only thickens white glyphs against black,
/// which cannot make them harder to read.
pub(crate) const IN_BUBBLE_HALO_COLOR: [u8; 4] = [255, 255, 255, 255];

/// Gives in-bubble text a glyph halo, for a page rendered with the artwork kept.
///
/// Detection refuses a halo to text inside a bubble, on the stated premise that
/// it "sits on a flat inpainted fill that already separates it from the
/// artwork". Skipping inpainting is precisely what makes that premise false: no
/// cleanup layer is written, so the compositor draws the original page and then
/// the English straight onto the un-erased Japanese, in the same colour, with
/// nothing between them. Measured on the four-bubble fixture, every line was
/// illegible.
///
/// Only dialogue is touched, and only where detection left no stroke of its own,
/// so free-standing text keeps the background colour `infer_text_color` measured
/// for it. Best-effort like its neighbour: a page whose scene will not take the
/// patch still renders.
///
/// **Known limit, not a defect.** No counter-safe ratio rescues a dense page. At
/// the 25th percentile of measured rendered sizes -- 13.3px -- even the closure
/// bound itself is only 0.8px, so pages whose lettering auto-fits into the teens
/// gain little. It helps roomy pages and does not save cramped ones.
pub fn halo_bubble_text(session: &mut Session, page: EntityId) -> bool {
    let snapshot = session.snapshot();
    let Ok(descendants) = snapshot.descendants(page) else {
        return false;
    };

    // Gathered by value before the patch opens, for the same aliasing reason as
    // `uppercase_dialogue` above.
    let mut pending: Vec<(EntityId, Typography)> = Vec::new();
    for entity in descendants {
        if !matches!(entity.component::<TextLayout>(), Ok(Some(_))) {
            continue;
        }
        let layer_id = entity.id();
        let Ok(layer) = snapshot.text_layer(layer_id) else {
            continue;
        };
        let Ok(content) = layer.content() else { continue };

        let is_dialogue = content
            .role()
            .ok()
            .flatten()
            .is_some_and(|role| role.role == DIALOGUE_ROLE);
        if !is_dialogue {
            continue;
        }

        let Some(typography) = layer.typography().ok().flatten() else {
            continue;
        };
        // Detection measured a halo for this layer already -- leave its colour
        // alone. Only the in-bubble refusal is being made up for here.
        if typography.stroke_width.is_some_and(|width| width > 0.0) {
            continue;
        }
        /* No size, no ratio. `Typography.size` is legitimately absent -- only
         * regions detection labelled "text" get one -- and the renderer's
         * stroke rescale skips exactly the same layers, so there is no size to
         * author against and nothing that would arrive at a predictable width. */
        let Some(size) = typography.size else { continue };
        if !size.is_finite() || size <= 0.0 {
            continue;
        }

        let mut next = typography.clone();
        next.stroke_color = Some(IN_BUBBLE_HALO_COLOR);
        next.stroke_width = Some(size * IN_BUBBLE_HALO_RATIO);
        pending.push((layer_id, next));
    }

    if pending.is_empty() {
        return false;
    }

    /* Set on the **layer** entity, never on the content entity `uppercase_dialogue`
     * writes its `Translation` to. `schema::validate_entity` refuses a text
     * content entity that also carries presentation components, so writing
     * `Typography` there is a hard invalid-scene error rather than a silent
     * miss. Detection's own `write_typography` targets the layer for this reason. */
    let patched = snapshot.patch(|edit| {
        for (layer, typography) in &pending {
            edit.set(*layer, typography)?;
        }
        Ok(())
    });
    let Ok(patch) = patched else { return false };
    session.commit(patch).is_ok()
}

/// Ink a CJK text region lays down, as a fraction of its own reported bounds.
///
/// **Empirical, and it has to be.** Measured at 11.8% on the title-page caption:
/// 29,570 glyph pixels inside a source region the scene reports as ~250,000px.
/// Vertical Japanese is set with generous inter-column air, so the region
/// overstates the ink by roughly eight to one.
///
/// Counting the ink directly was tried and is worse. Thresholding the region
/// against its own median counts the *artwork* under the caption as ink -- on
/// this page, hatched brickwork -- which came out at 64% and pushed the cap back
/// up to 90px. Gating that count by the detection text-mask would fix it, but the
/// mask is a scene asset behind a `BlobId` and reaching it from here is more
/// plumbing than the result justifies.
/// **The shipping default, and `--source-ink-fraction` overrides it.** The
/// switch exists because this number is a single-page measurement carrying a
/// whole product complaint: a skill name can letter at a quarter of the size of
/// the glyphs it replaces, and a replay showed it is **this cap and not the
/// layout box** that binds — the exemplar's 61.67 is
/// `sqrt(0.12 * 204.49 * 1692.54 / (0.42 * 26))` to an integer character count.
/// A single-column skill name has none of the inter-column air the 0.12 was
/// measured on, so an A/B needs to be able to say so in numbers.
pub(crate) const SOURCE_INK_FRACTION: f32 = 0.12;

/// Ink one Latin character lays down, as a fraction of its em square.
///
/// Comic lettering is heavier than a text face and this is set for CCWildWords
/// rather than for Arial. It only ever appears as a ratio against
/// `SOURCE_INK_FRACTION`, so an error in either is an error in the target size
/// by its square root.
const LATIN_INK_PER_CHARACTER: f32 = 0.42;

/// Never shrink a caption below this. Matches the renderer's own
/// `minimum_font_size`, under which it reports `TextBelowReadableSize` anyway.
const MINIMUM_FREE_TEXT_SIZE: f32 = 9.0;

/// Floor for a halo width this module rescales, mirroring
/// `FREE_TEXT_STROKE_MINIMUM_WIDTH` in koharu's `stages/detection.rs`.
///
/// Mirrored rather than imported because that constant is private to the
/// pipeline crate, and mirrored rather than ignored because dropping it is the
/// one way this rescale can make a halo *worse*. It is safe in both directions
/// of drift: too low loses a fraction of a pixel on a caption already at the
/// legibility floor, too high thickens one. Against the alternative -- scaling
/// with no floor -- it is strictly better, and the band where it matters is
/// narrow: the floor only applied at authoring time for a source under
/// `0.5 / 0.03` = 16.7px, and `fit_free_text` only shrinks, never below 9.0.
///
/// It cannot push the halo over the counter-closure bound either. At the
/// smallest size this module will produce, 0.5/9.0 = 0.056, still under the
/// 0.06 that closes Arial's `e`.
const HALO_MINIMUM_WIDTH: f32 = 0.5;

/// Caps free-standing text so its translation lays down about as much ink as the
/// Japanese it replaces.
///
/// Auto-fit maximises: it grows the text until the *box* is full. For text in a
/// balloon that is right, because the balloon is a container that is meant to be
/// filled. For text sitting on artwork it is wrong, and detection makes it
/// visibly wrong -- it widens a vertical caption's layout box to
/// `max(width, height * 0.6)` to escape a 6-10px illegible setting, and auto-fit
/// then spends the whole of the extra room on glyph size.
///
/// Measured on the title page: `requested=104 fitted=73.4 box=373x768
/// used=308x768 chars=55`. The source caption's own ink is 29,570px; 55
/// characters at 73.4px lay down about 124,000 -- **4.2x** the ink it replaced.
/// That is what "too large" looks like as a number, and matching the area of the
/// bounding *box* does not catch it, because the box is mostly the air between
/// vertical columns.
///
/// So the target is ink, not extent:
///
/// ```text
/// size = sqrt(SOURCE_INK_FRACTION * w * h / (LATIN_INK_PER_CHARACTER * chars))
/// ```
///
/// which gives 36px for that caption, half of what it rendered at.
///
/// The cap only ever lowers `Typography.size`. `auto_fit` stays on, so the
/// renderer may still shrink further to fit the box -- this removes the licence
/// to grow, it does not pin a size. In-bubble dialogue is untouched: it measured
/// at 14-16px on real pages, which is already right.
pub fn fit_free_text(session: &mut Session, page: EntityId, source_ink_fraction: f32) -> bool {
    let snapshot = session.snapshot();
    let Ok(descendants) = snapshot.descendants(page) else {
        return false;
    };

    // By value before the patch opens, for the same aliasing reason as the two
    // passes above.
    let mut pending: Vec<(EntityId, Typography)> = Vec::new();
    for entity in descendants {
        if !matches!(entity.component::<TextLayout>(), Ok(Some(_))) {
            continue;
        }
        let layer_id = entity.id();
        let Ok(layer) = snapshot.text_layer(layer_id) else {
            continue;
        };
        let Ok(content) = layer.content() else { continue };

        // Dialogue is text detection could place inside a bubble. Everything
        // else -- captions, signs, credits -- sits on artwork.
        let is_dialogue = content
            .role()
            .ok()
            .flatten()
            .is_some_and(|role| role.role == DIALOGUE_ROLE);
        if is_dialogue {
            continue;
        }

        // The tight box the source was read out of, NOT `layer.frame()`, which
        // for exactly these layers is the widened cell detection wrote.
        let Some(region) = content.source_region().ok().flatten() else {
            continue;
        };
        let Some((_, _, width, height)) = region
            .geometry()
            .ok()
            .as_ref()
            .and_then(crate::regions::axis_aligned_bounds)
        else {
            continue;
        };
        let Some(translation) = content.translation().ok().flatten() else {
            continue;
        };
        // Whitespace carries no ink, so counting it would shrink a caption for
        // being wordy rather than for being heavy.
        let characters = translation
            .text
            .value
            .chars()
            .filter(|character| !character.is_whitespace())
            .count();
        let Some(typography) = layer.typography().ok().flatten() else {
            continue;
        };
        /* Exactly the layers detection widens, and no others. Detection grows
         * the layout box only for free-standing text whose SOURCE was vertical,
         * so those are the only ones auto-fit was handed extra room to spend.
         *
         * The gate is not tidiness. Without it the credit block on this same
         * page -- dense horizontal lettering in a tight 206x88 box -- was capped
         * from 20.8px to the 9px floor, because `SOURCE_INK_FRACTION` describes
         * the air between vertical Japanese columns and a horizontal credit line
         * has none of it. Capping a box nobody widened can only ever shrink text
         * that already fitted. */
        if typography.writing_mode != Some(WritingMode::Vertical) {
            continue;
        }
        let Some(size) = typography.size else { continue };
        if characters == 0 || !size.is_finite() || size <= 0.0 || width <= 0.0 || height <= 0.0 {
            continue;
        }

        let source_ink = source_ink_fraction * (width as f32) * (height as f32);
        let target = (source_ink / (LATIN_INK_PER_CHARACTER * characters as f32))
            .sqrt()
            .max(MINIMUM_FREE_TEXT_SIZE);
        if !target.is_finite() || target >= size {
            continue;
        }

        let mut next = typography.clone();
        next.size = Some(target);
        /* The halo has to come down with the size, and this line is the whole of
         * a real defect.
         *
         * A halo width is authored against the size in the component at the time
         * -- detection writes `inferred.font_size * FREE_TEXT_STROKE_RATIO` -- and
         * the renderer later divides it by `layer.font_size` to rescale it to
         * whatever auto-fit actually solved. That is a closed loop only while
         * nothing moves `size` in between. This function is exactly something
         * moving `size` in between: the clone carried `stroke_width` across
         * unchanged while `size` dropped, so the renderer divided the old width
         * by the new size and the painted ratio became `RATIO * (authored /
         * capped)`.
         *
         * On the one case the code above records -- 104px requested, 73.4px
         * fitted, 55 characters -- the target is 35.8px and the effective ratio
         * is 0.087, against the 0.06 at which the halo closes Arial's `e`. The
         * general condition is `authored / capped > 2`, which this function
         * reaches whenever a caption is badly oversized, which is when it fires.
         *
         * Neither existing test could see it, and that is worth knowing before
         * trusting either. The detection-side test asserts the bound against
         * `inferred.font_size` and cannot see a rewrite in another crate; the
         * test below asserts the ratio survives the renderer's rescale -- and it
         * does, against the size that was in the component when the width was
         * written, which is no longer the size in the component at render. Both
         * green, both measuring a quantity the shipping pipeline had stopped
         * preserving.
         *
         * Scaling preserves whatever ratio the layer already carries rather than
         * recomputing one, so this stays correct if detection ever changes
         * `FREE_TEXT_STROKE_RATIO` and needs no copy of it here. */
        next.stroke_width = typography
            .stroke_width
            .filter(|width| width.is_finite() && *width > 0.0)
            .map(|width| (width * (target / size)).max(HALO_MINIMUM_WIDTH));
        pending.push((layer_id, next));
    }

    if pending.is_empty() {
        return false;
    }

    let patched = snapshot.patch(|edit| {
        for (layer, typography) in &pending {
            edit.set(*layer, typography)?;
        }
        Ok(())
    });
    let Ok(patch) = patched else { return false };
    session.commit(patch).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_measured_caption_is_capped_to_about_half_its_rendered_size() {
        // The title-page caption: a 198x768 detected region, 55 translated
        // characters, rendered at 73.4px for 4.2x the ink it replaced.
        // 29,570 dark pixels measured inside the 198x768 detected box.
        let target = (29_570.0 / (LATIN_INK_PER_CHARACTER * 47.0)).sqrt();
        assert!(
            (33.0..=42.0).contains(&target),
            "expected about 36px, got {target}"
        );
        assert!(target < 73.4 * 0.6);
    }

    #[test]
    fn the_cap_matches_ink_rather_than_extent() {
        /* The trap this avoids: matching the bounding BOX area leaves the size
         * almost unchanged, because a vertical caption's box is mostly the air
         * between its columns. Only the ink comparison shows the 4.2x. */
        let box_area = 198.0f32 * 768.0;
        let by_extent = (box_area / (LATIN_INK_PER_CHARACTER * 47.0)).sqrt();
        let by_ink = (29_570.0 / (LATIN_INK_PER_CHARACTER * 47.0)).sqrt();
        assert!(by_extent > 70.0, "extent matching would not shrink it");
        assert!(by_ink < by_extent * 0.5);
    }

    #[test]
    fn the_halo_stays_inside_arials_counters_at_every_size() {
        /* The bound asserted rather than restated as the arithmetic that
         * produced it. The stroke is centred at twice the authored width, so it
         * reaches a full width outward from every contour, inner ones included,
         * and the fill pass does not reopen a counter the stroke closed. An `e`
         * eye of ~0.12 em is attacked from both sides, so it shuts at 0.06.
         * Both sides scale with the size, so one assertion covers all of them. */
        assert!(IN_BUBBLE_HALO_RATIO < 0.06);
        // And thick enough to read as a rim at the measured median rendered size
        // of 17.6px, which is what 0.03 was not.
        assert!(17.6 * IN_BUBBLE_HALO_RATIO > 0.8);
    }

    #[test]
    fn a_ratio_survives_the_fitted_size_scaling_that_an_absolute_width_does_not() {
        /* The renderer multiplies the authored width by `fitted / authored`. A
         * width authored as `authored * R` therefore arrives as `fitted * R` --
         * the same ratio at any fitted size. This is why the theme-wide stroke
         * had to be abandoned: one absolute width becomes a different ratio for
         * every layer on the page. */
        for (authored, fitted) in [(98.0_f32, 20.0_f32), (28.0, 28.0), (62.0, 17.9)] {
            let painted = (authored * IN_BUBBLE_HALO_RATIO) * (fitted / authored);
            assert!(
                (painted / fitted - IN_BUBBLE_HALO_RATIO).abs() < 1e-6,
                "authored {authored} fitted {fitted} painted {painted}"
            );
            assert!(painted < 0.06 * fitted);
        }

        // The counter-example, at the same sizes: one page-wide 0.72px width.
        let page_wide = 0.72_f32;
        assert!((page_wide * (20.0 / 98.0)) / 20.0 < 0.008, "invisible on big text");
        assert!((page_wide * (5.0 / 5.0)) / 5.0 > 0.06, "and over the bound on small");
    }

    /// The test the one above could not be, and the reason a real defect lived
    /// beside two green assertions.
    ///
    /// The test above holds `authored` fixed across the renderer's rescale, which
    /// is true of every layer this module does not touch. `fit_free_text` is the
    /// exception: it rewrites `size` after the width was authored against the
    /// old one, so the quantity the renderer divides by is not the quantity the
    /// width was written against. This asserts the ratio across BOTH steps -- the cap
    /// and then the rescale -- which is the only place the whole chain is
    /// visible, because the two halves live in different crates.
    #[test]
    fn a_capped_caption_keeps_its_halo_ratio_through_both_rescales() {
        const RATIO: f32 = 0.03; // detection's FREE_TEXT_STROKE_RATIO

        // (authored size, capped target, size auto-fit finally solved)
        for (authored, capped, fitted) in [
            (104.0_f32, 35.8_f32, 35.8_f32), // the case lettering.rs records
            (104.0, 35.8, 21.0),             // and again where auto-fit went lower still
            (68.0, 24.0, 24.0),
            (24.0, 12.0, 12.0),
        ] {
            let width = (authored * RATIO).max(HALO_MINIMUM_WIDTH);
            let rescaled = (width * (capped / authored)).max(HALO_MINIMUM_WIDTH);
            let painted = rescaled * (fitted / capped);

            assert!(
                painted < 0.06 * fitted,
                "authored {authored} capped {capped} fitted {fitted}: \
                 {painted}px halo on {fitted}px text closes the counter"
            );

            // Without the rescale the same page is over the bound, which is the
            // regression this guards. Stated as an assertion rather than a
            // comment so it fails if anyone removes the rescale and the test
            // above still passes -- which is exactly what happened before.
            if authored / capped > 2.0 {
                let unfixed = width * (fitted / capped);
                assert!(
                    unfixed > 0.06 * fitted,
                    "authored {authored} capped {capped}: this case no longer \
                     demonstrates the defect, so the guard is measuring nothing"
                );
            }
        }
    }

    /// The floor is the one way the rescale can make a halo worse, so pin where
    /// it takes over and check it stays under the bound when it does.
    #[test]
    fn the_rescaled_halo_never_falls_below_the_visible_floor() {
        const RATIO: f32 = 0.03;

        // Authoring floors below 0.5/0.03 = 16.7px, and the cap never goes under
        // MINIMUM_FREE_TEXT_SIZE, so this band is the whole of the floor's reach.
        for (authored, capped) in [(16.0_f32, 9.0_f32), (13.0, 9.0), (12.0, 10.0)] {
            let width = (authored * RATIO).max(HALO_MINIMUM_WIDTH);
            assert_eq!(width, HALO_MINIMUM_WIDTH, "{authored}px should have floored");

            let rescaled = (width * (capped / authored)).max(HALO_MINIMUM_WIDTH);
            assert_eq!(rescaled, HALO_MINIMUM_WIDTH, "the floor must hold");
            assert!(
                rescaled < 0.06 * capped,
                "the floor itself closes the counter at {capped}px"
            );
        }

        // And the floor cannot reach past MINIMUM_FREE_TEXT_SIZE into the bound.
        assert!(HALO_MINIMUM_WIDTH < 0.06 * MINIMUM_FREE_TEXT_SIZE);
    }

    #[test]
    fn only_dialogue_is_shouted() {
        // The role string is the whole of the decision, so pin its spelling:
        // `link_dialogue_regions` writes this exact value, and a drift here
        // would silently uppercase nothing at all rather than fail.
        assert_eq!(DIALOGUE_ROLE, "dev.koharu.text.dialogue");
    }

    #[test]
    fn uppercasing_is_idempotent_and_leaves_caseless_text_alone() {
        // The guard that keeps the patch empty. Japanese, digits and punctuation
        // are caseless, so a page of them must produce no revision at all.
        for text in ["...", "1500", "!?", "カタカナ", "SHOUT"] {
            assert_eq!(text.to_uppercase(), text, "{text} should be unchanged");
        }
        assert_ne!("Hello".to_uppercase(), "Hello");
    }
}
