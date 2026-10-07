//! One utterance, lettered twice, on top of itself.
//!
//! # The defect
//!
//! The detector can find the same text twice -- once alone and once as part of a
//! larger region that contains it -- and the pipeline letters both. The two
//! translations are then painted over each other and at least one is destroyed.
//!
//! Measured on the two collision corpora (40 Japanese manga pages, 60 manhua
//! pages), against the post-layout signal that is the only instrument able to see
//! a real collision: **10 pairs of regions actually overlap where they were
//! drawn, and 4 of the 10 are this** --
//!
//! | corpus | one region | the other |
//! |---|---|---|
//! | manga | `hah!!` | `hah!!` |
//! | manga | `paper lanterns` | `chapter 9: paper lanterns` |
//! | manga | `i can do this!!` | `i can do this!!` |
//! | manhua | the chapter title | the title + the credits block |
//!
//! A five-judge vision panel looked at all ten and independently named this
//! class: "duplicate translations of the same source rendered twice at different
//! sizes -- the pipeline lettering one utterance in two regions".
//!
//! # Why this one may default ON when the collision patches before it did not
//!
//! **In the strict-containment branch it cannot lose a word.** There the rule
//! fires only when one region's translation is a *proper* substring of the
//! other's, and it drops the contained side, so every character it removes is
//! still on the page in the region that survives. An earlier collision fix spared
//! a whole sound effect and a review panel refused it for exactly the reason that
//! branch does not have: it took legible text off the page.
//!
//! **THAT PROMISE DOES NOT COVER THE IDENTICAL-STRING BRANCH, AND THAT IS THE
//! BRANCH THAT FIRES.** Identical strings contain each other, so `loser` resolves
//! them through `(true, true)` and drops one on area. The survivor is then
//! somewhere else on the page entirely — a different utterance, drawn in a
//! different place — so the reader loses a mark from where the artist put it. Both
//! firings measured on 219 webtoon pages took this branch, not containment: on
//! one page it is one drawn `Drill` detected twice and dropping is right; on the
//! other it is two separate 嘿 marks 221 px apart, both reading `Hey`, and
//! dropping one loses a sound effect the inpainter had already erased.
//!
//! Whether the identical branch should drop at all, or shrink per
//! `koharu-renderer`'s `request.rs` — *"shrink rather than drop, because a dropped
//! layer cannot be recovered"* — is an open question. Narrowing the geometry to
//! oriented frames fixed the two observed pages and left this mechanism standing.
//!
//! The erase is untouched, and that is deliberate. A refusal here hides the
//! *layer*; the source ink was erased during the inpainting stage long before
//! (`labels.rs` sets out why no post-execute gate can un-erase), and un-erasing
//! is not wanted anyway -- the duplicated Japanese should go, and its English is
//! still painted by the survivor.
//!
//! # The two guards, and the numbers that set them
//!
//! **Nesting alone is not enough, by a factor of twenty.** 85 lettered pairs on
//! those 100 pages have translations that nest, because `...` is a substring of
//! nearly every line carrying an ellipsis and `!` of every exclamation. Of those
//! 85, five also have overlapping boxes. So the box overlap is doing real work
//! here -- not as a *collision* predictor, which is refuted and stays refuted,
//! but as corroboration that two nesting strings are the same utterance rather
//! than two panels that happen to share a word.
//!
//! **[`MIN_SUBSTANTIVE_ALNUM`] is the whole of the second guard**, and it is
//! bounded on both sides rather than chosen:
//!
//! | threshold | fires | wrong | real duplicates missed |
//! |---|---|---|---|
//! | 2 | 4 | 0 | 0 |
//! | **3** | **4** | **0** | **0** |
//! | 4 | 3 | 0 | 1 -- loses `hah!!` |
//!
//! Without it the rule makes a fifth firing, dropping a region whose entire
//! translation is `!` because `!` appears inside `kuuuuuuh!`. Three is the
//! largest value that keeps every real duplicate, which is why it is three and
//! not two.

use koharu_scene::{EntityId, Origin, Session, TextLayout, Visibility};

use crate::regions::axis_aligned_bounds;

/// How many letters or digits the shorter translation must carry before its
/// being a substring of the other means anything.
///
/// See the module table: 4 loses a real duplicate, 2 buys nothing over 3.
const MIN_SUBSTANTIVE_ALNUM: usize = 3;

/// How much of the smaller SOURCE box must lie inside the intersection before
/// the pair counts as one ink read twice (`--duplicate-shared-source`).
///
/// The question this gate actually asks -- "is this one utterance detected
/// twice?" -- is about the SOURCE ink, and the source geometry answers it
/// directly where the painted frames cannot: English set into two separate
/// containers routinely overlaps as painted while the source marks never touch.
/// Measured as intersection-over-smaller of the two recognized-from boxes, over
/// every duplicate-gate firing on the test corpora (a Japanese manga volume, the
/// collision pages, a 219-page webtoon set and a Chinese chapter), each site
/// adjudicated in pixels:
///
/// | population | IoS values |
/// |---|---|
/// | one ink read twice (drop RIGHT) | 0.510, 0.679, 0.722, 0.814, 0.839, 0.855, 0.866, 0.885, 0.887, 0.968, 1.000 x6 |
/// | separate authored marks (drop WRONG) | 0.000 x2, 0.025, 0.083, 0.120, 0.184, 0.326, 0.337 |
///
/// The hole is (0.337, 0.510) and 0.42 is its midpoint. The lower edge is a
/// pair of twin heartbeat columns (two distinct marks, each with its own
/// trailing dots -- looked at); the upper edge is a fragment re-read of the tail
/// of one wavy hand-drawn column (looked at). The floor
/// test below pins both edges, so moving this out of the hole in either
/// direction goes red.
const DUPLICATE_SOURCE_IOS_FLOOR: f64 = 0.42;

/// One lettered layer, reduced to what the rule needs.
struct Candidate {
    layer: EntityId,
    content: EntityId,
    text: String,
    /// The frame the translation was painted into, as `(x, y, w, h)`.
    ///
    /// This is the AABB of `poly`, so for rotated text it is inflated — up to 2x
    /// at 45 degrees. Kept because `loser`'s tie-break asks which frame is
    /// *bigger*, and an AABB ranks the pair the same way an oriented area does.
    fit: (f64, f64, f64, f64),
    /// The same frame as the polygon it really is, before `axis_aligned_bounds`
    /// discards the rotation. Read by `convex_overlaps`.
    poly: Vec<(f64, f64)>,
    /// The recognized-from region's AABB -- where the SOURCE ink was read, not
    /// where the English landed. `None` when the layer has no source region
    /// (a minted or synthesised layer), which `shares_source` treats as NOT
    /// corroborated: without geometry the same-ink claim cannot be checked, and
    /// a wrong drop erases an authored mark while a wrong keep only letters a
    /// redundant copy.
    source_box: Option<(f64, f64, f64, f64)>,
}

/// Do two source boxes substantially coincide -- one ink, read twice?
///
/// Intersection over the SMALLER area, so a tight re-read nested inside a large
/// over-detection (measured: a 33x176 column inside a 322x430 panel box, IoS
/// 0.679) still counts as the same ink even though the big box dwarfs it.
fn shares_source(left: &Candidate, right: &Candidate) -> bool {
    let (Some(a), Some(b)) = (left.source_box, right.source_box) else {
        return false;
    };
    let w = (a.0 + a.2).min(b.0 + b.2) - a.0.max(b.0);
    let h = (a.1 + a.3).min(b.1 + b.3) - a.1.max(b.1);
    let intersection = w.max(0.0) * h.max(0.0);
    let smaller = (a.2 * a.3).min(b.2 * b.3);
    smaller > 0.0 && intersection / smaller >= DUPLICATE_SOURCE_IOS_FLOOR
}

fn alnum(text: &str) -> usize {
    text.chars().filter(|c| c.is_alphanumeric()).count()
}

fn overlaps(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> bool {
    a.0 < b.0 + b.2 && b.0 < a.0 + a.2 && a.1 < b.1 + b.3 && b.1 < a.1 + a.3
}

/// Do two frames overlap *as drawn*, rather than as their bounding boxes?
///
/// A separating-axis test over both polygons' edge normals. `frame()` returns a
/// polygon — a **rotated** rectangle for angled text — and `axis_aligned_bounds`
/// throws that rotation away. For a rectangle at 45 degrees the AABB is 2x the
/// true area, so two effects that never touch can be reported as overlapping,
/// which is exactly how the gate came to drop a separate sound effect on a
/// webtoon page: measured 572 px of AABB overlap against **0** oriented, at 40
/// and -45 degrees. The other known case, whose layers are both at angle 0, is
/// unaffected — 194,765.6 px either way — so this separates the two exactly.
///
/// **Convex only.** A dialogue balloon's frame is a mask contour and may be
/// non-convex, and SAT then answers for the convex hull. That errs toward
/// *reporting* an overlap, i.e. toward the current behaviour, so it cannot start
/// dropping a pair the AABB rule would have spared. Bubble frames are not
/// rotation-inflated in the first place, so the AABB is already adequate there;
/// the population this rule is aimed at is rotated free-standing text.
///
/// Strict, matching `overlaps`: sharing only an edge is not overlapping.
fn convex_overlaps(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
    if a.len() < 3 || b.len() < 3 {
        return false;
    }
    for polygon in [a, b] {
        for i in 0..polygon.len() {
            let (x1, y1) = polygon[i];
            let (x2, y2) = polygon[(i + 1) % polygon.len()];
            // The edge normal. A degenerate edge yields no axis to test on.
            let (nx, ny) = (y2 - y1, x1 - x2);
            if nx.abs() < f64::EPSILON && ny.abs() < f64::EPSILON {
                continue;
            }
            let project = |points: &[(f64, f64)]| {
                points.iter().fold((f64::MAX, f64::MIN), |(lo, hi), (x, y)| {
                    let p = x * nx + y * ny;
                    (lo.min(p), hi.max(p))
                })
            };
            let (a_lo, a_hi) = project(a);
            let (b_lo, b_hi) = project(b);
            if a_hi <= b_lo || b_hi <= a_lo {
                return false;
            }
        }
    }
    true
}

/// The predicate the walk actually calls — the one a test has to drive.
///
/// Testing `overlaps` and `convex_overlaps` separately tests neither this choice
/// nor the flag that makes it, which is precisely how a fix can ship unwired with
/// every test green.
fn pair_overlaps(left: &Candidate, right: &Candidate, oriented: bool) -> bool {
    if oriented {
        convex_overlaps(&left.poly, &right.poly)
    } else {
        overlaps(left.fit, right.fit)
    }
}

/// Which candidates lose, as a mask parallel to `candidates`.
///
/// Split out of `hide_duplicate_lettering` for the same reason `loser` is: so the
/// decision can be driven from a test with real page geometry and no scene, no
/// model and no device.
fn resolve_duplicates(candidates: &[Candidate], oriented: bool, shared_source: bool) -> Vec<bool> {
    /* `dropped` is consulted as the walk goes, so three copies of one line lose
     * two rather than all three: once a side is dropped it can no longer act as
     * the survivor that condemns another. Without this a page carrying the same
     * shout three times would letter none of it. */
    let mut dropped: Vec<bool> = vec![false; candidates.len()];
    for a in 0..candidates.len() {
        for b in (a + 1)..candidates.len() {
            if dropped[a] || dropped[b] {
                continue;
            }
            let (left, right) = (&candidates[a], &candidates[b]);
            if !pair_overlaps(left, right, oriented) {
                continue;
            }
            /* Additive, never a replacement: with the arm ON every drop is a
             * drop the bare gate would also have made, so an A/B's pixel diff
             * is exactly the spared sites. See `DUPLICATE_SOURCE_IOS_FLOOR`
             * for the populations that set the floor. */
            if shared_source && !shares_source(left, right) {
                continue;
            }
            let Some(which) = loser(&left.text, &right.text, area(left.fit), area(right.fit)) else {
                continue;
            };
            dropped[if which == 0 { a } else { b }] = true;
        }
    }
    dropped
}

fn area(a: (f64, f64, f64, f64)) -> f64 {
    a.2 * a.3
}

/// Which of a nesting pair is the one to drop, or `None` if the pair does not
/// qualify.
///
/// Returns an index into the pair: `0` for the left, `1` for the right.
///
/// Split out from the walk so the decision can be tested without a scene.
///
/// **The tie-break matters, and this comment used to describe it backwards.** It
/// said two regions carrying the *identical* string "neither contains the other".
/// They contain each other — that is why the arm below is `(true, true)` and not
/// `(false, false)` — and the whole reason the arm exists is that without it both
/// sides would qualify as redundant and the utterance would leave the page.
///
/// What survives the correction is the consequence: in that branch the survivor
/// is somewhere **else** on the page, so unlike strict containment this does not
/// keep every character where the artist drew it. See the module header.
#[must_use]
fn loser(left: &str, right: &str, left_area: f64, right_area: f64) -> Option<usize> {
    let (left, right) = (left.trim().to_lowercase(), right.trim().to_lowercase());
    if left.is_empty() || right.is_empty() {
        return None;
    }
    let shorter = if left.len() <= right.len() { &left } else { &right };
    if alnum(shorter) < MIN_SUBSTANTIVE_ALNUM {
        return None;
    }
    match (left.contains(&right), right.contains(&left)) {
        // Identical strings contain each other. Keep the one drawn in the bigger
        // frame, which is the one a reader is more likely to be able to read;
        // fall back to the left so the answer never depends on float equality.
        (true, true) => Some(usize::from(left_area >= right_area)),
        // The right is contained in the left, so the right is the redundant one.
        (true, false) => Some(1),
        (false, true) => Some(0),
        (false, false) => None,
    }
}

/// Hide the redundant half of every duplicate-lettered pair on `page`.
///
/// Returns the *content* entities that were hidden, in the same shape
/// `labels::hide_implausible` returns, so `routes.rs` can merge the two lists and
/// report them through the one `refused` field.
///
/// A scene patch between `Pipeline::execute` and the render, exactly like
/// `labels.rs`, `sfx.rs` and `lettering.rs` -- no koharu patch, no pipeline
/// reload.
///
/// `oriented` selects which geometry decides that two frames touch. `false` is
/// the shipping behaviour — the axis-aligned bounding box of each frame. `true`
/// tests the frames **as drawn**; see `convex_overlaps` for why that matters and
/// for the two pages that separate the arms.
///
/// `shared_source` additionally requires the two SOURCE boxes to substantially
/// coincide (`shares_source`) before a pair may drop — the arm that separates
/// one ink read twice from two authored marks whose English happens to collide.
/// OFF is the shipping behaviour; `--duplicate-shared-source` turns it on.
pub fn hide_duplicate_lettering(
    session: &mut Session,
    page: EntityId,
    oriented: bool,
    shared_source: bool,
) -> Vec<EntityId> {
    let snapshot = session.snapshot();
    let Ok(descendants) = snapshot.descendants(page) else {
        return Vec::new();
    };

    // Gathered by value before the patch opens, for the aliasing reason set out
    // in `lettering.rs`.
    let mut candidates: Vec<Candidate> = Vec::new();
    for entity in descendants {
        if !matches!(entity.component::<TextLayout>(), Ok(Some(_))) {
            continue;
        }
        let Ok(layer) = snapshot.text_layer(entity.id()) else {
            continue;
        };
        let Ok(content) = layer.content() else { continue };
        /* A layer an earlier gate already hid is not a candidate, and skipping it
         * is not an optimisation. This walk reads `Translation`, not
         * `Visibility`, so without the check it re-refuses a region that
         * `hide_implausible` had already refused -- and because `routes.rs`
         * stamps the merged list onto the regions, the second reason overwrites
         * the first. Measured on a manhua test page: a region reading `000` was
         * correctly refused as "no letters, only punctuation or symbols" and came
         * back reported as a duplicate, changing nothing on the
         * page and destroying the true reason in `format=json`. A refusal that
         * erases its own evidence is untestable from outside, which is the rule
         * `labels.rs` states and this now honours. */
        if matches!(
            entity.component::<Visibility>(),
            Ok(Some(Visibility { visible: false, .. }))
        ) {
            continue;
        }
        let Ok(Some(translation)) = content.translation() else {
            continue;
        };
        if translation.text.value.trim().is_empty() {
            continue;
        }
        /* The frame is where the translation was PAINTED, which is the rectangle
         * the two regions would have to share for this to be one utterance. The
         * recognized-from region is the tight box the Japanese was read out of
         * and is the wrong one: a nested detection reads two different boxes of
         * source and the question is about the English. */
        let Ok(Some(frame)) = layer.frame() else {
            continue;
        };
        let Some(fit) = axis_aligned_bounds(&frame) else {
            continue;
        };
        if fit.2 <= 0.0 || fit.3 <= 0.0 {
            continue;
        }
        // Kept alongside `fit` rather than instead of it: the AABB still ranks
        // the pair for `loser`'s tie-break, while the polygon answers whether
        // they touch at all.
        let poly: Vec<(f64, f64)> = frame.points.iter().map(|p| (p.x, p.y)).collect();
        /* The recognized-from box, NOT the frame fallback `regions.rs` uses --
         * falling back to the frame here would fake source corroboration out of
         * the very quantity the arm exists to distinguish from. */
        let source_box = content
            .source_region()
            .ok()
            .flatten()
            .and_then(|region| region.geometry().ok())
            .as_ref()
            .and_then(axis_aligned_bounds);
        candidates.push(Candidate {
            layer: entity.id(),
            content: content.id(),
            text: translation.text.value.clone(),
            fit,
            poly,
            source_box,
        });
    }

    let dropped = resolve_duplicates(&candidates, oriented, shared_source);

    let pending: Vec<&Candidate> = candidates
        .iter()
        .zip(&dropped)
        .filter_map(|(candidate, dropped)| dropped.then_some(candidate))
        .collect();
    if pending.is_empty() {
        return Vec::new();
    }

    let hidden = snapshot.patch(|edit| {
        for candidate in &pending {
            edit.set(
                candidate.layer,
                &Visibility {
                    // `User`, for the reason `labels.rs` gives: this is operator
                    // policy applied to a finished scene, not a model's output.
                    origin: Origin::User,
                    visible: false,
                    opacity: 1.0,
                },
            )?;
        }
        Ok(())
    });
    let Ok(patch) = hidden else { return Vec::new() };
    if session.commit(patch).is_err() {
        return Vec::new();
    }
    pending.iter().map(|candidate| candidate.content).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_contained_side_is_the_one_dropped() {
        assert_eq!(
            loser("Paper Lanterns", "Chapter 9: Paper Lanterns", 100.0, 400.0),
            Some(0)
        );
        assert_eq!(
            loser("Chapter 9: Paper Lanterns", "Paper Lanterns", 400.0, 100.0),
            Some(1)
        );
    }

    /// Two regions carrying the identical string contain each other, so the
    /// containment test alone cannot choose and a naive implementation drops
    /// BOTH -- taking the utterance off the page, which is the one outcome this
    /// rule exists to avoid. Exactly one side must lose.
    #[test]
    fn identical_strings_lose_exactly_one_side() {
        // The bigger frame survives, so the loser is the side with the smaller
        // one -- whichever position it happens to be in.
        assert_eq!(loser("Hah!!", "Hah!!", 900.0, 100.0), Some(1));
        assert_eq!(loser("Hah!!", "Hah!!", 100.0, 900.0), Some(0));
        // Equal areas still answer, rather than depending on float equality.
        assert!(loser("Hah!!", "Hah!!", 100.0, 100.0).is_some());
    }

    /// The measured false positive, and the whole reason for the alnum floor:
    /// `!` really is a substring of `kuuuuuuh!`, and the two boxes really do
    /// overlap on one measured page.
    #[test]
    fn a_punctuation_only_substring_is_not_a_duplicate() {
        assert_eq!(loser("!", "Kuuuuuuh!", 10.0, 500.0), None);
        assert_eq!(loser("...", "Well... if you insist", 10.0, 500.0), None);
        assert_eq!(loser("!!", "Everyone!!", 10.0, 500.0), None);
    }

    /// Three alnum is the largest floor that keeps every real duplicate, and
    /// `hah!!` -- h, a, h -- is the case that sets it. A test on the bound, not a
    /// restatement of the constant.
    #[test]
    fn the_alnum_floor_is_the_largest_that_keeps_hah() {
        assert_eq!(alnum("Hah!!"), 3);
        assert!(alnum("Hah!!") >= MIN_SUBSTANTIVE_ALNUM);
        // One higher and it would be refused, which is the measured cost of 4.
        assert!(alnum("Hah!!") < MIN_SUBSTANTIVE_ALNUM + 1);
        // And the false positive stays out at the shipped value.
        assert!(alnum("!") < MIN_SUBSTANTIVE_ALNUM);
    }

    #[test]
    fn unrelated_text_is_left_alone() {
        assert_eq!(loser("Thank goodness...", "Nyanya-nyanya!", 100.0, 100.0), None);
        assert_eq!(loser("", "anything", 1.0, 1.0), None);
    }

    /// Case and surrounding whitespace are not a reason to letter something
    /// twice: the detector's two reads of one utterance routinely differ by a
    /// trailing space, and `to_uppercase` runs later in the renderer anyway.
    #[test]
    fn the_match_ignores_case_and_padding() {
        assert_eq!(loser("  paper lanterns ", "CHAPTER 9: PAPER LANTERNS", 1.0, 9.0), Some(0));
    }

    #[test]
    fn boxes_that_do_not_touch_are_not_a_pair() {
        assert!(overlaps((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 10.0, 10.0)));
        assert!(!overlaps((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 10.0, 10.0)));
        assert!(!overlaps((0.0, 0.0, 10.0, 10.0), (0.0, 20.0, 10.0, 10.0)));
    }

    /// A rectangle of `w` x `h` centred on `(cx, cy)` and turned `degrees`.
    ///
    /// Built rather than hardcoded so the two arms cannot silently be fed
    /// different geometry — the whole comparison is that they are not.
    fn rotated(cx: f64, cy: f64, w: f64, h: f64, degrees: f64) -> Vec<(f64, f64)> {
        let (s, c) = degrees.to_radians().sin_cos();
        [(-w / 2.0, -h / 2.0), (w / 2.0, -h / 2.0), (w / 2.0, h / 2.0), (-w / 2.0, h / 2.0)]
            .into_iter()
            .map(|(x, y)| (cx + x * c - y * s, cy + x * s + y * c))
            .collect()
    }

    fn candidate(text: &str, poly: Vec<(f64, f64)>) -> Candidate {
        let xs = poly.iter().map(|p| p.0);
        let ys = poly.iter().map(|p| p.1);
        let (min_x, max_x) = (xs.clone().fold(f64::MAX, f64::min), xs.fold(f64::MIN, f64::max));
        let (min_y, max_y) = (ys.clone().fold(f64::MAX, f64::min), ys.fold(f64::MIN, f64::max));
        Candidate {
            layer: EntityId::default(),
            content: EntityId::default(),
            text: text.to_owned(),
            fit: (min_x, min_y, max_x - min_x, max_y - min_y),
            poly,
            source_box: None,
        }
    }

    fn with_source(mut candidate: Candidate, source_box: (f64, f64, f64, f64)) -> Candidate {
        candidate.source_box = Some(source_box);
        candidate
    }

    /// A webtoon page: ONE drawn `Drill` detected twice, both layers upright.
    /// Dropping the redundant copy is correct, and BOTH arms must keep doing it —
    /// this is the case the gate exists for.
    #[test]
    fn an_upright_pair_over_one_mark_is_dropped_under_either_geometry() {
        let pair = [
            candidate("Drill", rotated(710.0, 284.0, 360.0, 676.0, 0.0)),
            candidate("Drill", rotated(710.0, 284.0, 712.0, 582.0, 0.0)),
        ];
        assert_eq!(resolve_duplicates(&pair, false, false).iter().filter(|d| **d).count(), 1);
        assert_eq!(resolve_duplicates(&pair, true, false).iter().filter(|d| **d).count(), 1);
    }

    /// Another webtoon page: TWO SEPARATE drawn marks that both read `Hey`, at 40 and
    /// -45 degrees. Their bounding boxes graze by 572 px while the frames as
    /// drawn never touch, so the shipping arm drops one — and its ink was already
    /// erased, so the page loses a sound effect. The oriented arm must not.
    ///
    /// This is the composed predicate the caller calls, driven end to end:
    /// asserting on `convex_overlaps` alone would pass with the flag unwired.
    #[test]
    fn two_separate_marks_reading_the_same_word_survive_only_the_oriented_arm() {
        let pair = [
            candidate("Hey", rotated(145.0, 505.0, 149.0, 162.0, 40.0)),
            candidate("Hey", rotated(281.0, 680.0, 114.0, 102.0, -45.0)),
        ];
        // The bounding boxes DO overlap, which is why the shipping arm fires.
        assert!(overlaps(pair[0].fit, pair[1].fit));
        assert_eq!(
            resolve_duplicates(&pair, false, false).iter().filter(|d| **d).count(),
            1,
            "the shipping arm drops one of two separate marks"
        );
        assert_eq!(
            resolve_duplicates(&pair, true, false).iter().filter(|d| **d).count(),
            0,
            "the oriented arm keeps both, because as drawn they never touch"
        );
    }

    /// The flag must be the only thing that differs. A rotated pair that really
    /// does overlap is still a duplicate under both arms, so `oriented` cannot be
    /// read as "never fire on anything rotated".
    #[test]
    fn a_rotated_pair_that_really_touches_is_still_a_duplicate() {
        let pair = [
            candidate("Kaboom", rotated(100.0, 100.0, 160.0, 60.0, 30.0)),
            candidate("Kaboom", rotated(120.0, 110.0, 140.0, 50.0, 30.0)),
        ];
        assert!(convex_overlaps(&pair[0].poly, &pair[1].poly));
        assert_eq!(resolve_duplicates(&pair, false, false).iter().filter(|d| **d).count(), 1);
        assert_eq!(resolve_duplicates(&pair, true, false).iter().filter(|d| **d).count(), 1);
    }

    /// Real geometry from a Japanese test page: two separate drawn laugh columns
    /// in two balloon lobes, whose SOURCE boxes are disjoint (IoS 0.000) while
    /// both translate to `Pfft...` and their painted
    /// frames overlap. The bare gate drops one authored mark; the shared-source
    /// arm keeps both. Driven through `resolve_duplicates` -- the predicate the
    /// walk calls -- with the flag as the only difference.
    #[test]
    fn two_authored_marks_with_disjoint_source_ink_survive_the_shared_source_arm() {
        let pair = [
            with_source(
                candidate("Pfft...", rotated(321.4, 205.1, 23.1, 112.5, 0.0)),
                (309.90625, 148.828125, 23.078125, 112.5),
            ),
            with_source(
                candidate("Pfft...", rotated(311.3, 258.4, 23.1, 87.9, 0.0)),
                (258.8046875, 214.453125, 23.078125, 87.890625),
            ),
        ];
        // The painted frames DO overlap -- that is why the bare gate fires.
        assert!(overlaps(pair[0].fit, pair[1].fit));
        assert_eq!(resolve_duplicates(&pair, true, false).iter().filter(|d| **d).count(), 1);
        assert_eq!(
            resolve_duplicates(&pair, true, true).iter().filter(|d| **d).count(),
            0,
            "disjoint source ink means two marks, and both must letter"
        );
    }

    /// Real geometry from a Japanese test page: ONE vertical column read twice,
    /// once tight (33x176) and once inside a panel-sized over-detection
    /// (322x430), source IoS 0.679. One ink, so the drop must survive the
    /// shared-source arm -- the gate's core case cannot be lost to the flag.
    #[test]
    fn one_ink_read_twice_still_drops_under_the_shared_source_arm() {
        let pair = [
            with_source(
                candidate("Watch me win", rotated(606.6, 192.8, 321.7, 430.4, 0.0)),
                (445.78078642000463, -22.40249784395627, 321.6884271599907, 430.35187068791254),
            ),
            with_source(
                candidate("Watch me win", rotated(761.6, 214.5, 33.0, 175.8, 0.0)),
                (745.09375, 126.5625, 32.96875, 175.78125),
            ),
        ];
        assert_eq!(resolve_duplicates(&pair, true, false).iter().filter(|d| **d).count(), 1);
        assert_eq!(
            resolve_duplicates(&pair, true, true).iter().filter(|d| **d).count(),
            1,
            "one ink read twice is exactly what the gate exists to drop"
        );
    }

    /// The floor pinned against BOTH edges of the measured hole, with the two
    /// cases that set them (see `DUPLICATE_SOURCE_IOS_FLOOR`): the twin
    /// heartbeat columns at IoS 0.337 must be spared, and the fragment re-read of
    /// one wavy column at IoS 0.510 must still drop. A
    /// floor moved below 0.337 or above 0.510 goes red here.
    #[test]
    fn the_source_floor_sits_inside_the_measured_hole() {
        let spared = [
            with_source(
                candidate("Thump... thump...", rotated(629.7, 1064.1, 30.0, 77.0, 0.0)),
                (614.6897635966004, 1025.550938470508, 30.02672280679917, 77.02312305898386),
            ),
            with_source(
                candidate("Thump... thump...", rotated(619.8, 1101.6, 33.0, 68.3, 0.0)),
                (603.3159065315346, 1067.4106552839276, 32.9931869369309, 68.30368943214489),
            ),
        ];
        assert!(overlaps(spared[0].fit, spared[1].fit));
        assert_eq!(
            resolve_duplicates(&spared, true, true).iter().filter(|d| **d).count(),
            0,
            "0.337 sits under the floor: two marks, both letter"
        );

        let dropped = [
            with_source(
                candidate("What's up?", rotated(682.5, 97.9, 45.4, 77.1, 0.0)),
                (659.763098068024, 59.29575982240408, 45.38005386395207, 77.11160535519184),
            ),
            with_source(
                candidate("What's up?", rotated(662.7, 114.3, 32.6, 60.5, 0.0)),
                (646.3520935047351, 84.03123993669502, 32.63956299052984, 60.45314512660997),
            ),
        ];
        assert_eq!(
            resolve_duplicates(&dropped, true, true).iter().filter(|d| **d).count(),
            1,
            "0.510 clears the floor: a fragment re-read of the same ink drops"
        );
    }

    /// A candidate with no source region cannot corroborate the same-ink claim,
    /// and the arm must then KEEP both: a wrong drop erases an authored mark, a
    /// wrong keep only letters a redundant copy. The bare gate is unaffected.
    #[test]
    fn missing_source_geometry_is_not_corroboration() {
        let pair = [
            candidate("Drill", rotated(710.0, 284.0, 360.0, 676.0, 0.0)),
            candidate("Drill", rotated(710.0, 284.0, 712.0, 582.0, 0.0)),
        ];
        assert_eq!(resolve_duplicates(&pair, true, false).iter().filter(|d| **d).count(), 1);
        assert_eq!(
            resolve_duplicates(&pair, true, true).iter().filter(|d| **d).count(),
            0,
            "no geometry, no drop under the shared-source arm"
        );
    }

    #[test]
    fn a_frame_sharing_only_an_edge_is_not_an_overlap_under_either_geometry() {
        let left = rotated(5.0, 5.0, 10.0, 10.0, 0.0);
        let right = rotated(15.0, 5.0, 10.0, 10.0, 0.0);
        assert!(!convex_overlaps(&left, &right));
        assert!(!overlaps((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 10.0, 10.0)));
    }

    #[test]
    fn a_degenerate_frame_is_never_an_overlap() {
        let square = rotated(5.0, 5.0, 10.0, 10.0, 0.0);
        assert!(!convex_overlaps(&square, &[]));
        assert!(!convex_overlaps(&square, &[(5.0, 5.0), (6.0, 6.0)]));
    }
}
