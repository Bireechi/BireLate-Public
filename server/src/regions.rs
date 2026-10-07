//! Per-region source and translated text, read back out of the finished scene.
//!
//! This is a pure in-memory read of the snapshot the committer produced, so it
//! costs no pipeline call and touches no device. The traversal is the same one
//! koharu's own project commands use: a text layer presents text content, which
//! was recognized from an analysis region.

use koharu_scene::{
    EntityId, Geometry, Region, RegionSpec, Snapshot, TextDirection, TextLayout, TextRegion,
    WritingMode,
};
use serde::Serialize;

use crate::labels::Refusal;

/// The detector's label for text it recognises as a sound effect.
///
/// RF-DETR emits four classes -- text, onomatopoeia, bubble, panel -- and the
/// pipeline consumes this one destructively and only destructively: `mask_for`
/// folds it into the text mask, so the inpainter erases it, while `write_region`
/// returns early for any non-`"text"` label so no text layer is ever created and
/// `ocr.rs` never visits it. The lettering is removed and nothing is painted
/// back.
const ONOMATOPOEIA: &str = "onomatopoeia";

/// A region the detector found and the pipeline never translated.
///
/// This exists because `untranslated` cannot answer the question a reader
/// actually has. That field is built from `Progress::Untranslated`, which
/// reports *a detected text region the translator skipped* -- so it is empty on
/// a page whose Japanese was never detected as text at all, and it was measured
/// empty on all 32 audit pages including ones with plainly visible untranslated
/// Japanese. It is a translator-stage report, not a completeness gate, and using
/// it as one is a silent false negative.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct SkippedOut {
    /// The detector's own class name, e.g. `onomatopoeia`.
    pub label: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RegionOut {
    /// The text *content* entity behind this region -- the same id the pipeline's
    /// translation stage submits segments under, and so the only reliable way to
    /// match a `Progress::Untranslated` report back to a region. Never
    /// serialized: it is a scene-internal uuid, meaningless to the extension,
    /// and regenerated on every run.
    #[serde(skip)]
    pub content: EntityId,

    /// The text *layer* entity, which is a different id from `content` and is the
    /// one the renderer's diagnostics name.
    ///
    /// Carried for exactly one reason: a `LayoutWarning` arrives identified by its
    /// layer, and without this the only join available was a heuristic over
    /// `font_size` -- walk the warnings in order and take the next solved layer
    /// whose size matches, with a second pass beside it as a sensitivity bound
    /// *because the pairing is not unique*. Most overflowing layers bottom out at
    /// the same 9px floor, so several candidates on a page share the checksum.
    /// Not serialized, for the same reason `content` is not: a scene-internal
    /// uuid regenerated every run.
    #[serde(skip)]
    pub layer: EntityId,

    /// Axis-aligned bounds in source-image pixel space, which is also the
    /// returned PNG's pixel space -- no transform is needed. Text geometry is a
    /// *rotated* rectangle whenever the detector inferred an angle, so this
    /// hull can be a few pixels looser than the true box.
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub source: String,
    pub translated: String,

    /// Where the translated text was actually painted: the surrounding bubble
    /// when the region sits inside one, otherwise the text region itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit_x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit_y: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit_width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit_height: Option<f64>,

    /// The language the caller DECLARED for the request, echoed back per region
    /// -- never detected. No OCR engine here reports a language; the pipeline's
    /// `stamped_language` stamps the declared tag as a full BCP-47 string
    /// (`ko-KR`, `zh-CN`) and keeps `ja-JP` as the undeclared default. A reader
    /// wanting the resolved script should take the body's `source_script`; a
    /// reader treating this as the engine's own judgement of the page is reading
    /// the declaration back. PRESENCE of the key is a separate signal ("this
    /// region was OCR'd") that audit tooling keys on -- do not make the field
    /// conditional.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_language: Option<String>,
    /// The detector's own score for this region -- a property of the BOX, not of
    /// the read. The read's own score is `ocr_confidence`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detection_confidence: Option<f32>,
    /// The kept OCR read's own confidence, `exp(mean log p)` of its decode.
    /// Present only when the engine reports one: PaddleOCR-VL and the Hunyuan
    /// sidecar do; `manga-ocr`, `baberu-ocr` and Ollama do not, and their absence
    /// is "not measured", never zero. When the stage took a second read of the
    /// crop, this is the score of the read that SHIPPED.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_confidence: Option<f32>,
    /// From the OCR analysis -- a bounding-box aspect-ratio heuristic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<&'static str>,
    /// From the layer's typography, inferred from the segmentation mask's
    /// projection profile. Better grounded than `direction`; prefer it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub writing_mode: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region_kind: Option<String>,

    /// The detector's own class for this region -- `text` or `onomatopoeia`.
    ///
    /// Reported because `region_kind` cannot answer the question any more: the
    /// pipeline promotes an effect to `TextRegion` so OCR will read it, so after
    /// promotion an effect and a line of dialogue carry the *same* kind and the
    /// label is the only thing that still tells them apart. [`skipped`] used to
    /// be the only place a label reached a caller, and it reports exactly the
    /// effects that were **not** translated -- which is now none of them.
    ///
    /// Without this, a run cannot count its own sound effects, and the sfx
    /// dictionary's gate (`sfx::pin_sound_effects`, same label) is unobservable
    /// from the outside.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,

    /// `dialogue` when detection found a bubble containing this text, `free-text`
    /// otherwise -- the `dev.koharu.text.` prefix is stripped. This is the one
    /// classification the pipeline already computes and has never reported:
    /// `link_dialogue_regions` overwrites the role only for text it could place
    /// inside a bubble, so it separates speech from signs, captions and the
    /// overlaid text a light novel page is made of.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// The glyph height detection measured off the segmentation mask's
    /// projection profile, in source pixels. Compared against the page height it
    /// is the most direct evidence of a shout that exists without asking an LLM.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f32>,

    /// The colour the lettering is configured to fill with. For free-standing
    /// text this is the drawn ink's eroded-core sample; for dialogue it is
    /// `infer_text_color`'s contrast sample. **Snapped to pure black or white
    /// only when near-neutral** -- `normalize_text_color` leaves a chromatic
    /// sample untouched. White is the usual tell for inverted narration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<[u8; 4]>,

    /// The configured halo/outline colour, `None` when detection refused one
    /// (in-bubble text before `halo_bubble_text`, or nothing inferred). Without
    /// the stroke fields the outline half of a lettering change is verifiable
    /// only in pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stroke_color: Option<[u8; 4]>,

    /// The authored halo width in px, rescaled by the renderer to the fitted
    /// size, so read it as `width / font_size` a ratio, not px.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stroke_width: Option<f32>,

    /// The configured face weight. Only the sampled-ink pass writes one today,
    /// and only for heavy drawn strokes (`HEAVY_INK_RATIO`); absent means the
    /// family default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<u16>,

    /// Why this region was refused the page, if it was -- `labels::Refusal::why`,
    /// or [`HIDDEN_LAYER_REFUSAL`] when the hidden layer itself is the only
    /// record.
    ///
    /// The region stays in this list when it is refused, holding its OCR string
    /// and its translation, because a gate that deleted its own evidence could
    /// not be checked from outside the process. Present means nothing was drawn
    /// for it; absent means it was lettered. The pipeline's lettering veto hides
    /// a layer without returning a `Refusal`, so without a fallback such a region
    /// -- a 0.44-confidence read, in the measured case -- would letter nothing
    /// and appear in NO ledger: not here, not `dropped` (its translation is
    /// non-empty), not `untranslated`, not `rendered_text`. [`regions`] therefore
    /// stamps the fallback off the layer's own `Visibility`; [`stamp_refusals`]
    /// overwrites it wherever a server gate recorded the specific reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refused: Option<&'static str>,

    /// This read is SUSPECT: a box the pipeline did not read sits within
    /// [`OCCLUSION_GAP`] of it, so glyphs may be hidden under that box and
    /// missing from `source`. Measured case: a display column whose first glyph
    /// sat under the host's banner was read without it, and nothing said the
    /// read was incomplete.
    ///
    /// Two producing classes, named honestly because conflating them is the
    /// trap this field exists to avoid:
    /// - `"watermark"` -- a sibling region in this same array was refused as
    ///   `Refusal::Watermark`; the ledger CLASSIFIED it.
    /// - `"unread-box"` -- a sub-floor edge hint nearby; nothing ever read it,
    ///   so nothing classified it, and calling it a watermark would be a
    ///   re-classification this layer must not make (a drawn badge is page
    ///   content).
    ///
    /// Observation-only, like the drop report: no pixel and no status code
    /// moves on this field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub occluded_by: Option<&'static str>,
}

/// How close a refused/unread box must sit to a read region before that
/// region's read is marked suspect, in page pixels.
///
/// Measured on the motivating case: the seam composite's read sits 12.34 px
/// below the banner hint and the single page's 10.60 -- the spread is detector
/// jitter between two inferences of the same geometry, not a projection
/// difference (the composite's top rows ARE that page's frame, offset 0). A
/// census over a 179-slice chapter is flat from 13 to 48 px, so 16 is not
/// perched on a cliff. Known costs at 16, disclosed rather than hidden: one
/// slice's status text fires against two sub-floor fragments 3.78 px below it
/// (one false suspect in 179 slices), and a line on another chapter fires on its
/// refused site plate at a true intersection -- which is a CORRECT fire.
const OCCLUSION_GAP: f64 = 16.0;

/// koharu-pipeline's own extent helper is crate-private, so this recomputes it.
/// Geometry is a polygon: a rotated rectangle for angled text, a mask contour
/// for a bubble.
#[must_use]
pub fn axis_aligned_bounds(geometry: &Geometry) -> Option<(f64, f64, f64, f64)> {
    let mut points = geometry.points.iter();
    let first = points.next()?;
    let (mut min_x, mut min_y) = (first.x, first.y);
    let (mut max_x, mut max_y) = (first.x, first.y);
    for point in points {
        min_x = min_x.min(point.x);
        min_y = min_y.min(point.y);
        max_x = max_x.max(point.x);
        max_y = max_y.max(point.y);
    }
    Some((min_x, min_y, max_x - min_x, max_y - min_y))
}

/// The fallback `refused` reason for a layer a gate hid without recording why
/// on the wire.
///
/// Exactly one producer writes `Visibility { visible: false }` without also
/// returning a `Refusal` for [`stamp_refusals`] to report: koharu-pipeline's
/// lettering veto in `stages/ocr.rs`, shared by the artwork-spotting gate, the
/// illegible-read floor and the synthesised-region withdrawal -- its
/// `tracing::info!` line names which one fired. The server's own hiders
/// (`labels::hide_implausible`, the duplicate gate) return their reasons and
/// overwrite this string, so on a finished wire this text means "the pipeline
/// vetoed the lettering". Consumers then treat the region like any refusal:
/// `dropped` excludes it, the extension's report counts it, and `seam.js`
/// stops using its box to admit repaint parts.
pub const HIDDEN_LAYER_REFUSAL: &str =
    "nothing was lettered: the pipeline hid this layer (its log line names the veto)";

/// Reads every translated region on `page`.
///
/// Ordering follows the scene's document order, which the detection stage
/// already sorted into manga reading order. That is an observation about the
/// current pipeline, not a promise made on the wire.
#[must_use]
pub fn regions(snapshot: &Snapshot, page: EntityId) -> Vec<RegionOut> {
    let Ok(descendants) = snapshot.descendants(page) else {
        return Vec::new();
    };
    // Text layers carry a TextLayout; the text *content* entities and the
    // inpainting cleanup layer do not. Filtering here rather than going through
    // the page's text group matters: that helper resolves every child of the
    // group and fails outright if any one of them is not a text layer.
    let layer_ids = descendants
        .filter(|entity| matches!(entity.component::<TextLayout>(), Ok(Some(_))))
        .map(|entity| entity.id())
        .collect::<Vec<_>>();

    let mut out = Vec::with_capacity(layer_ids.len());
    for id in layer_ids {
        let Ok(layer) = snapshot.text_layer(id) else {
            continue;
        };
        let Ok(content) = layer.content() else {
            continue;
        };

        // The recognized-from region is the tight box the source text was read
        // out of; the layer frame is where the translation was painted.
        let region = content.source_region().ok().flatten();
        let frame = layer.frame().ok().flatten();
        let geometry = match region {
            Some(region) => region.geometry().ok(),
            None => frame.clone(),
        };
        let Some((x, y, width, height)) = geometry.as_ref().and_then(axis_aligned_bounds) else {
            continue;
        };
        let fit = frame.as_ref().and_then(axis_aligned_bounds);

        let source = content.source().ok().flatten();
        let translation = content.translation().ok().flatten();
        let ocr = region.and_then(|region| region.ocr().ok().flatten());
        let detection = region.and_then(|region| region.detection().ok().flatten());
        // Fetched once and read twice, for `kind` and for `label`.
        let region_component = region.and_then(|region| region.region().ok());
        // Fetched once. `writing_mode` used to reach for this on its own and drop
        // `size` and `color` on the same line, which is why neither has ever been
        // visible even though detection has always computed them.
        let typography = layer.typography().ok().flatten();
        // A hidden layer lettered nothing, and `refused` is the field that says
        // so; `stamp_refusals` replaces this with the specific reason wherever a
        // server gate owns the hide.
        let hidden = layer
            .visibility()
            .ok()
            .flatten()
            .is_some_and(|visibility| !visibility.visible);

        out.push(RegionOut {
            content: content.id(),
            layer: id,
            x,
            y,
            width,
            height,
            fit_x: fit.map(|(value, ..)| value),
            fit_y: fit.map(|(_, value, ..)| value),
            fit_width: fit.map(|(_, _, value, _)| value),
            fit_height: fit.map(|(.., value)| value),
            source_language: source
                .as_ref()
                .and_then(|value| value.language.as_ref())
                .map(|tag| tag.as_str().to_owned()),
            target_language: translation
                .as_ref()
                .and_then(|value| value.language.as_ref())
                .map(|tag| tag.as_str().to_owned()),
            source: source.map(|value| value.text.value).unwrap_or_default(),
            translated: translation.map(|value| value.text.value).unwrap_or_default(),
            detection_confidence: detection
                .and_then(|value| value.labels.first().map(|label| label.confidence)),
            ocr_confidence: ocr.as_ref().and_then(|analysis| analysis.confidence),
            direction: ocr.as_ref().map(|analysis| match analysis.direction {
                TextDirection::Vertical => "vertical",
                TextDirection::Horizontal => "horizontal",
                TextDirection::Auto => "auto",
            }),
            writing_mode: typography
                .as_ref()
                .and_then(|value| value.writing_mode)
                .map(|mode| match mode {
                    WritingMode::Vertical => "vertical",
                    WritingMode::Horizontal => "horizontal",
                }),
            region_kind: region_component
                .as_ref()
                .map(|value| value.kind.as_str().to_owned()),
            label: region_component.as_ref().and_then(|value| value.label.clone()),
            role: content
                .role()
                .ok()
                .flatten()
                .map(|value| short_role(&value.role)),
            font_size: typography.as_ref().and_then(|value| value.size),
            color: typography.as_ref().and_then(|value| value.color),
            stroke_color: typography.as_ref().and_then(|value| value.stroke_color),
            stroke_width: typography.as_ref().and_then(|value| value.stroke_width),
            font_weight: typography.as_ref().and_then(|value| value.font_weight),
            refused: hidden.then_some(HIDDEN_LAYER_REFUSAL),
            occluded_by: None,
        });
    }
    out
}

/// The axis gap between two boxes: zero on an axis where they overlap,
/// otherwise the empty distance between their nearest edges.
fn axis_gaps(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> (f64, f64) {
    let gap = |a0: f64, a1: f64, b0: f64, b1: f64| (b0 - a1).max(a0 - b1).max(0.0);
    (
        gap(a.0, a.0 + a.2, b.0, b.0 + b.2),
        gap(a.1, a.1 + a.3, b.1, b.1 + b.3),
    )
}

/// The shape test that separates "this hint is the region's own ink,
/// re-detected below the floor" from "an unread box lands beside or across
/// the read". True when the overlap is most of EITHER box.
///
/// **An IoU > 0.5 test over-fired 9 of 14 marks**, because a fragment hint
/// inside a joined column has IoU 0.30 while its containment is 0.98: same ink,
/// different extents. Measured on the render where it over-fired, all nine extra fires were
/// containment >= 0.53 on one side and every genuine occluder was disjoint,
/// so the honest boundary is containment, not IoU. The accepted blind spot:
/// an unreadable occluder lying WHOLLY inside a read region's box is skipped
/// as same-ink -- no such instance exists in the test corpus, and when the
/// plate is readable it becomes a refused region, which the watermark arm
/// catches.
fn mostly_same_ink(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> bool {
    let ix = (a.0 + a.2).min(b.0 + b.2) - a.0.max(b.0);
    let iy = (a.1 + a.3).min(b.1 + b.3) - a.1.max(b.1);
    if ix <= 0.0 || iy <= 0.0 {
        return false;
    }
    let inter = ix * iy;
    let (area_a, area_b) = (a.2 * a.3, b.2 * b.3);
    area_a > 0.0 && area_b > 0.0 && (inter / area_a > 0.5 || inter / area_b > 0.5)
}

/// Mark every read region whose box sits within [`OCCLUSION_GAP`] of a box the
/// pipeline refused or never read -- see [`RegionOut::occluded_by`] for the two
/// classes and why they are named apart. Run AFTER [`stamp_refusals`]: the
/// watermark arm keys on `refused == Refusal::Watermark.why()`, i.e. on what
/// was REFUSED AS a watermark, and never re-runs classification. `hints` are
/// the page's sub-floor edge-hint boxes in page pixels; a hint that is the
/// region's own ink -- most of EITHER box inside the overlap, see
/// [`mostly_same_ink`], which owns why this is containment and NOT IoU --
/// is skipped, while a small intersecting occluder still fires.
/// `"watermark"` wins where both arms apply.
pub fn stamp_occlusion(regions: &mut [RegionOut], hints: &[(f64, f64, f64, f64)]) {
    let watermark_reason = crate::labels::Refusal::Watermark.why();
    let watermarks: Vec<(f64, f64, f64, f64)> = regions
        .iter()
        .filter(|region| region.refused == Some(watermark_reason))
        .map(|region| (region.x, region.y, region.width, region.height))
        .collect();
    for index in 0..regions.len() {
        if regions[index].refused.is_some() {
            continue;
        }
        let own = (
            regions[index].x,
            regions[index].y,
            regions[index].width,
            regions[index].height,
        );
        let near = |other: (f64, f64, f64, f64)| {
            let (dx, dy) = axis_gaps(own, other);
            dx <= OCCLUSION_GAP && dy <= OCCLUSION_GAP
        };
        if watermarks.iter().any(|&plate| near(plate)) {
            regions[index].occluded_by = Some("watermark");
        } else if hints
            .iter()
            .any(|&hint| !mostly_same_ink(own, hint) && near(hint))
        {
            regions[index].occluded_by = Some("unread-box");
        }
    }
}

/// Carry each refusal back onto the region it belongs to, keyed on the content
/// entity.
///
/// Lifted out of `routes.rs` so a test can drive the WHOLE chain -- scene →
/// [`regions`] → refusal walk → this → `RegionOut.refused` -- with no model and no
/// device. Asserting on the walk alone would pass with this join deleted: a fix
/// can be complete, called by nothing, and green in every test.
///
/// The alternative to keeping a refused region in the list -- deleting it -- would
/// leave a caller unable to tell a refusal from a detection that never happened.
/// It stays, holding its OCR string and its translation.
///
/// Last write wins where two refusals name the same content, so the caller's merge
/// order is the priority order.
/// Whether this region's `(source, translated)` pair may enter the story
/// window. **The composed predicate the record site calls**: the refusal half
/// and the SFX half are one decision, and a test of either alone does not test
/// the `&&` between them.
///
/// - A refused region contributes no pair regardless of the flag. The window is
///   fed verbatim into the next page's prompt, so a hallucinated pair would be
///   carried forward as established terminology (`routes.rs` says this at the
///   call site, and has since before this function existed).
/// - With `excludes_sfx`, an `onomatopoeia`-labelled region contributes no pair
///   either. Measured: the window held `嘭 -> "Boom!"` four times, the prompt
///   says to preserve terminology, and a skill name built on a repeated glyph
///   collapsed to "Boom! Boom! Boom! Boom!" in 12 of 14 window-on renders while
///   every window-off render kept the name.
///
/// The gate is the DETECTOR'S label, deliberately not a shape test: a
/// character-repetition guard was rejected because faithful translations of
/// genuinely repetitive sources are indistinguishable by shape, and the label
/// is exactly the fact [`RegionOut::label`]'s own doc calls "the only thing
/// that still tells them apart" after SFX promotion. The region itself is still
/// translated and lettered -- this decides TEACHING, not rendering.
pub fn feeds_story(region: &RegionOut, excludes_sfx: bool) -> bool {
    region.refused.is_none()
        && !(excludes_sfx && region.label.as_deref() == Some("onomatopoeia"))
}

pub fn stamp_refusals(regions: &mut [RegionOut], refused: &[(EntityId, Refusal)]) {
    if refused.is_empty() {
        return;
    }
    let reasons: std::collections::HashMap<_, _> = refused
        .iter()
        .map(|(content, refusal)| (*content, refusal.why()))
        .collect();
    // Overwrite only entries this gate owns. The blanket assignment this used
    // to be was equivalent while `regions()` always produced `refused: None`;
    // with the [`HIDDEN_LAYER_REFUSAL`] fallback it would have CLEARED the
    // pipeline-veto stamp on any page that also carried a server refusal.
    for region in regions.iter_mut() {
        if let Some(why) = reasons.get(&region.content) {
            region.refused = Some(why);
        }
    }
}

/// Detected regions the pipeline erased without ever translating them.
///
/// Enumerates by the `Region` component rather than by text layer, which is the
/// whole point: [`regions`] filters descendants down to entities carrying a
/// `TextLayout`, so a detection that never became a text layer is structurally
/// invisible to it. `write_region` creates the entity and its `Region` -- label
/// and geometry included -- before returning early, so the evidence is in the
/// scene and simply had no reader.
///
/// Bubbles and panels are regions too and are deliberately not reported: they
/// carry no lettering, so they are not missing text. Only the classes that hold
/// glyphs count, which today is exactly [`ONOMATOPOEIA`].
///
/// **The kind test is what keeps this honest now that effects are lettered.**
/// `translate_sfx` promotes an onomatopoeia detection to a `TextRegion`, so it
/// is read, translated and painted like any other lettering -- and reporting it
/// as skipped afterwards would turn a working feature into a permanent warning.
/// The label alone can no longer answer the question; only the label *and* the
/// absence of the text kind mean "erased with nothing painted back".
#[must_use]
pub fn skipped(snapshot: &Snapshot, page: EntityId) -> Vec<SkippedOut> {
    let Ok(descendants) = snapshot.descendants(page) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entity in descendants {
        let Ok(Some(region)) = entity.component::<Region>() else {
            continue;
        };
        if region.label.as_deref() != Some(ONOMATOPOEIA) || region.kind == TextRegion::kind() {
            continue;
        }
        let Ok(Some(geometry)) = entity.component::<Geometry>() else {
            continue;
        };
        let Some((x, y, width, height)) = axis_aligned_bounds(&geometry) else {
            continue;
        };
        out.push(SkippedOut {
            label: ONOMATOPOEIA.to_owned(),
            x,
            y,
            width,
            height,
        });
    }
    out
}

/// `dev.koharu.text.dialogue` -> `dialogue`.
///
/// The namespace is koharu's internal component addressing and means nothing to
/// a reader of the JSON; the last segment is the whole of the signal. An
/// unnamespaced value is passed through rather than mangled.
fn short_role(role: &str) -> String {
    role.rsplit('.').next().unwrap_or(role).to_owned()
}

/// Which entries of `regions` the pipeline reported as untranslated.
///
/// Matched on the content entity, never by comparing `source` with
/// `translated`. Text that legitimately translates to itself -- romaji, digits,
/// a sound effect the model left alone, and the lone ellipsis the translation
/// stage deliberately maps to itself -- would otherwise be reported as failures,
/// and a warning that cries wolf is worse than none.
///
/// Anything the walk did not find is dropped rather than guessed at, so this can
/// be shorter than `entities`; the caller keeps the pipeline's own count for
/// anything it wants to state as a total.
#[must_use]
pub fn untranslated_indices(regions: &[RegionOut], entities: &[EntityId]) -> Vec<usize> {
    if entities.is_empty() {
        return Vec::new();
    }
    regions
        .iter()
        .enumerate()
        .filter_map(|(index, region)| entities.contains(&region.content).then_some(index))
        .collect()
}

/// Which entries of `regions` were erased and then lettered with nothing.
///
/// **[`untranslated_indices`] structurally cannot see this class**, which is the
/// whole reason it exists. The translation stage seeds its result from the
/// *source* segments and fills in only the ids the model answered, so a segment
/// the model never answered is written back holding its Japanese verbatim: it
/// renders as untranslated Japanese and lands in `untranslated`. It can never
/// come back empty. An empty translation is therefore always the other thing --
/// an entry the model positively answered with an empty string, under a grammar
/// that pins the *number* of reply entries and nothing about their content. A
/// bigger token budget does not touch it, and `duplicate_ids` cannot see it
/// either.
///
/// Under the default plan the region was still erased. The inpainting mask is
/// written by *detection* from the raw boxes, long before OCR or translation
/// run, so a region that is erased and then lettered with nothing is a blank
/// patch with nothing painted back: 502 px at the smallest measured, 92,296 px
/// at the largest, on an 844x1200 page.
///
/// **That is a fact about `Plan::Full`, not about this function**, and anything
/// phrased for a reader has to survive the other plans. `Plan::KeepArt` runs no
/// inpainting stage at all, so nothing is erased and a dropped region leaves the
/// original artwork -- Japanese included -- exactly as it was. What is true
/// under every plan is the narrower claim this reports: the region was read, and
/// nothing was lettered for it. Calling that "erased" in a user-facing string is
/// wrong under `KeepArt`, which is why the extension does not.
///
/// # It does not poison the story window, and that was claimed once
///
/// A drop cannot reach the next page's prompt. [`crate::story::Stories::record`]
/// filters on `!translation.trim().is_empty()` before a pair is ever stored --
/// `an_untranslated_segment_is_never_carried_forward` asserts it -- so an empty
/// translation is discarded at the door. An earlier note here claimed 59
/// poisoned story pairs as deferred work; there are none, and there is nothing
/// to defer.
///
/// Measured over every stored page response: 59 instances in 33,268 regions.
/// Restricted to responses the current wire produced -- the ones carrying
/// `duplicate_ids` -- that is 59 of 4,689 Japanese
/// regions, 1.26%, spread over 22 of 426 pages, worst page 13 of 15. On all 59,
/// `untranslated` was empty, `truncated` false, `duplicate_ids` 0 and
/// `out_of_range_ids` 0: every channel already on the wire called the page
/// clean. `rendered_text` corroborates independently -- it holds one entry per
/// *non-empty* translation, and on all 59 pages its length equals the count of
/// non-empty translations, so the renderer set no layer for any of them.
///
/// # What this excludes, and why
///
/// **An empty source.** That is every deliberately blanked read and every
/// `clean_only` run, and all 1,502 empty-source regions in the corpus also
/// carry an empty translation -- so testing the source is what keeps a
/// deliberately blanked read out of this count, rather than a special case for
/// it.
///
/// **A region carrying [`RegionOut::refused`].** That is a decision this server
/// took on purpose and already reports through its own field; 341 regions in the
/// corpus are refused and 17 of them hold an empty translation, and counting
/// those would report the gate as the defect it exists to prevent.
///
/// **Nothing else -- and that is a measured answer, not an omission.** The one
/// candidate for a third exemption is a source with no letters in it, where an
/// empty English is arguably correct. The corpus holds exactly two, `？` and
/// `！？`, and both are in-bubble: their boxes were erased to flat white, which
/// is precisely the damage `labels::hide_implausible`'s free-standing-only gate
/// exists to avoid creating. The 27 sound effects in the count are here for the
/// same reason -- the pipeline letters effects, so an effect answered with an
/// empty string is an *erased* effect, not a skipped one, and [`skipped`] will
/// not report it because the region was promoted to `TextRegion`.
/// Name the region each layout warning is about, by layer identity.
///
/// # Why this is not cosmetic
///
/// A `LayoutWarning` has always carried its layer in `entity`, and `entity` has
/// always been `#[serde(skip)]`. Nothing else on the warning identifies it, so
/// every reader downstream of the JSON has had to *reconstruct* the mapping from
/// `font_size` alone -- and that checksum collides by construction, because the
/// interesting layers are exactly the ones auto-fit drove down to the same 9px
/// floor. A reader that walks the warnings in order taking the next solved
/// layer whose size matches has to carry a second pass beside it as a
/// sensitivity bound *because the pairing is not unique*.
///
/// So an overflow measurement built that way measures its own guess, and every
/// per-role split it prints inherits it. This makes the mapping exact and the
/// guess unnecessary.
///
/// Matched on the layer, never on geometry or size: two layers in one bubble can
/// share a box after detection's cell split, and a whole page can share the
/// floor size.
pub fn attribute_warnings(warnings: &mut [crate::render::LayoutWarning], regions: &[RegionOut]) {
    // Built once rather than scanned per warning: a dense page is ~40 regions and
    // a warned page can carry more warnings than regions, since one layer may
    // report `too_small` and `overflow` both.
    let index = layer_index(regions);
    for warning in warnings {
        warning.region = index.get(&warning.entity).copied();
    }
}

/// Name the region each solved text layer is, by layer identity.
///
/// Exactly [`attribute_warnings`], for exactly the same reason and with exactly
/// the same failure mode if it is skipped: `rendered_text` arrives in
/// composition order, which is not the region order, so a reader holding only
/// the JSON has to guess -- and `font_size` cannot break the tie, because a
/// dense page drives most of its layers onto the same 9px floor.
///
/// Kept as its own function rather than folded into the one above because the
/// two arrays have different lengths and different `None` meanings: a page can
/// carry more warnings than regions, and it can carry fewer solved layers than
/// regions when a region was never lettered at all.
pub fn attribute_rendered_text(text: &mut [crate::render::RenderedText], regions: &[RegionOut]) {
    let index = layer_index(regions);
    for entry in text {
        entry.region = index.get(&entry.entity).copied();
    }
}

fn layer_index(regions: &[RegionOut]) -> std::collections::HashMap<EntityId, usize> {
    regions
        .iter()
        .enumerate()
        .map(|(position, region)| (region.layer, position))
        .collect()
}

#[must_use]
pub fn dropped_indices(regions: &[RegionOut]) -> Vec<usize> {
    regions
        .iter()
        .enumerate()
        .filter_map(|(index, region)| {
            // Trimmed on both sides: a translation of a single space paints
            // nothing, and a source of a single space was never text.
            (!region.source.trim().is_empty()
                && region.translated.trim().is_empty()
                && region.refused.is_none())
            .then_some(index)
        })
        .collect()
}

/// Which entries of `regions` carry a translation that stops mid-sentence — the
/// tell of a reply segment SPLIT across two ids, which shifts every later
/// region's sentence onto its neighbour.
///
/// The translator sometimes splits ONE source across TWO reply ids; the first
/// half ends with trailing whitespace (`It's `, `By `, `the `), the extra slot
/// is consumed, and every later region on the page inherits its predecessor's
/// sentence. Adjudicated on a 213-page Japanese baseline: 6 of 213 pages
/// shifted, ~50 regions displaced, and **every
/// existing channel called those pages clean** — `truncated` false,
/// `untranslated` empty, `duplicate_ids` 0, `out_of_range_ids` 0 on all six.
/// The model returns the right NUMBER of ids with shifted CONTENT, so the
/// reconciliation above cannot see it. This marker fires on 22 regions across
/// 17 of those 213 pages and flags the page holding six of the seven
/// adjudicated misplacements.
///
/// **A flag is evidence about the PAGE, not a verdict on its every region.**
/// The counterexample is on record: one page carries two fragments and
/// recovers alignment before its end, so downstream must read this as "look at
/// this page", never as "every later region is displaced" — which is also why
/// it is a diagnostic and not a gate.
///
/// What it excludes, deliberately: a refused region (the gate already reports
/// itself), and a translation that is entirely whitespace — that is
/// [`dropped_indices`]' population, and one defect must not be accused twice.
///
/// The fingerprint core is `koharu_translator::is_cut` — the SAME two clauses
/// the local retry keys on, so the diagnostic and the repair cannot silently
/// disagree on what a fragment looks like. The `refused` arm is this layer's
/// own and stays here.
#[must_use]
pub fn split_fragment_indices(regions: &[RegionOut]) -> Vec<usize> {
    regions
        .iter()
        .enumerate()
        .filter_map(|(index, region)| {
            (koharu_translator::is_cut(&region.translated) && region.refused.is_none())
                .then_some(index)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use koharu_scene::{
        At, Authored, Origin, PageDraft, Point, RecognizedFrom, Session, SourceText, TextLayoutKind,
        Translation, Visibility,
    };

    fn geometry(points: &[(f64, f64)]) -> Geometry {
        Geometry {
            origin: Origin::User,
            points: points.iter().map(|&(x, y)| Point { x, y }).collect(),
        }
    }

    /// One page, two lettered layers, one of them hidden -- the shape of a
    /// measured case: the pipeline's illegible-read floor hid a 0.44-confidence
    /// read and the wire showed a healthy region, absent from `refused`,
    /// `dropped`, `untranslated` and
    /// `rendered_text` all at once. Asserted through [`regions`], the walk the
    /// route actually calls, on a committed scene rather than a hand-built
    /// `RegionOut` -- a hand-built one cannot fail on the `Visibility` read.
    #[test]
    fn a_hidden_layer_reaches_the_wire_as_refused_and_a_visible_one_does_not() {
        let mut session = Session::memory().expect("an in-memory session");
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 1200.0, 919.0), At::End)?;
                let mut layers = Vec::new();
                for (source, translated) in [("白鹭！", "Egret!"), ("苍炎之王！", "Blue Flame King!")] {
                    let region = edit.add_analysis_region::<TextRegion>(
                        page,
                        At::End,
                        &Geometry::rectangle(747.0, 368.0, 333.0, 438.0),
                        None,
                    )?;
                    let content = edit.add_text_content(page, At::End)?;
                    edit.set(
                        content,
                        &SourceText {
                            text: Authored::user(source.to_owned()),
                            language: None,
                        },
                    )?;
                    edit.set(
                        content,
                        &Translation {
                            text: Authored::user(translated.to_owned()),
                            language: None,
                        },
                    )?;
                    let layer = edit.add_text_layer(
                        page,
                        At::End,
                        content,
                        &TextLayout {
                            origin: Origin::User,
                            kind: TextLayoutKind::Paragraph,
                        },
                    )?;
                    edit.relate::<RecognizedFrom>(content, region)?;
                    layers.push(layer);
                }
                // The veto's exact write, `stages/ocr.rs`: the layer is hidden,
                // nothing else about the region changes.
                edit.set(
                    layers[0],
                    &Visibility {
                        origin: Origin::User,
                        visible: false,
                        opacity: 1.0,
                    },
                )?;
                ids = Some((page, layers));
                Ok(())
            })
            .expect("the fixture scene is valid");
        session.commit(patch).expect("the fixture scene commits");
        let (page, layers) = ids.expect("the edit ran to completion");

        let out = regions(&session.snapshot(), page);
        assert_eq!(out.len(), 2, "both layers reach the wire");
        let hidden = out.iter().find(|r| r.layer == layers[0]).expect("the hidden layer");
        let visible = out.iter().find(|r| r.layer == layers[1]).expect("the visible layer");
        assert_eq!(hidden.refused, Some(HIDDEN_LAYER_REFUSAL));
        assert_eq!(visible.refused, None);
        // And the region stays out of `dropped`: a refusal is a decision with
        // its own field, not a silent loss -- the doc on `dropped_indices`.
        assert!(dropped_indices(&out).is_empty());
    }

    /// [`feeds_story`]'s whole truth table, in one place, because the two
    /// halves are one decision. The ON column is the shipping behaviour and the
    /// OFF column is the byte-exact control arm; a change to either is a change
    /// to what every reader's window carries.
    #[test]
    fn feeds_story_refuses_sfx_only_when_asked_and_refused_regions_always() {
        let mut sfx = region(EntityId::new(), "嘭", "Boom!");
        sfx.label = Some("onomatopoeia".to_owned());
        let mut name = region(EntityId::new(), "三重防御", "Triple Defense");
        name.label = Some("text".to_owned());
        let unlabelled = region(EntityId::new(), "冥霜之王！", "King Frost!");
        let mut refused_sfx = region(EntityId::new(), "嘭", "Boom!");
        refused_sfx.label = Some("onomatopoeia".to_owned());
        refused_sfx.refused = Some(HIDDEN_LAYER_REFUSAL);
        let mut refused_text = region(EntityId::new(), "纸叶漫画", "Paperleaf Comics");
        refused_text.refused = Some(Refusal::Watermark.why());

        for excludes_sfx in [false, true] {
            assert!(
                !feeds_story(&refused_sfx, excludes_sfx),
                "a refused region never teaches, whatever the flag says"
            );
            assert!(!feeds_story(&refused_text, excludes_sfx));
            assert!(
                feeds_story(&name, excludes_sfx),
                "a skill name's pair is exactly what the window is FOR"
            );
            assert!(
                feeds_story(&unlabelled, excludes_sfx),
                "no label means no evidence of SFX -- the safe direction is to keep"
            );
        }
        assert!(
            feeds_story(&sfx, false),
            "the control arm: the Boom pair still teaches"
        );
        assert!(
            !feeds_story(&sfx, true),
            "the shipping arm: the Boom pair stays out of the window"
        );
    }

    /// The fallback yields to a specific reason and ONLY to its own entry.
    /// `stamp_refusals` used to blanket-assign the map lookup, which was
    /// equivalent while `regions()` always produced `refused: None` -- with the
    /// fallback it would have cleared the pipeline-veto stamp on any page that
    /// also carried a server refusal.
    #[test]
    fn stamp_refusals_overwrites_its_own_entry_and_keeps_the_fallback() {
        let vetoed = EntityId::new();
        let watermarked = EntityId::new();
        let mut regions = vec![region(vetoed, "白鹭！", "Egret!"), region(watermarked, "纸叶", "Leaf")];
        regions[0].refused = Some(HIDDEN_LAYER_REFUSAL);
        stamp_refusals(&mut regions, &[(watermarked, Refusal::Watermark)]);
        assert_eq!(regions[0].refused, Some(HIDDEN_LAYER_REFUSAL), "the fallback survives");
        assert_eq!(regions[1].refused, Some(Refusal::Watermark.why()), "the specific reason wins");
    }

    #[test]
    fn a_rotated_rectangle_yields_its_hull() {
        // Corners arrive in TL, TR, BR, BL order for angled text.
        let bounds = axis_aligned_bounds(&geometry(&[
            (10.0, 20.0),
            (50.0, 15.0),
            (55.0, 45.0),
            (15.0, 50.0),
        ]))
        .unwrap();
        assert_eq!(bounds, (10.0, 15.0, 45.0, 35.0));
    }

    #[test]
    fn an_empty_polygon_has_no_bounds() {
        assert!(axis_aligned_bounds(&geometry(&[])).is_none());
    }

    #[test]
    fn a_single_point_has_zero_extent() {
        let bounds = axis_aligned_bounds(&geometry(&[(7.0, 9.0)])).unwrap();
        assert_eq!(bounds, (7.0, 9.0, 0.0, 0.0));
    }

    #[test]
    fn absent_extras_are_omitted_from_the_json() {
        let region = RegionOut {
            content: EntityId::new(),
            layer: EntityId::new(),
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
            source: "ソース".to_owned(),
            translated: "source".to_owned(),
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
        };
        let json = serde_json::to_value(&region).unwrap();
        for key in ["x", "y", "width", "height", "source", "translated"] {
            assert!(json.get(key).is_some(), "{key} should be present");
        }
        for key in [
            "fit_x",
            "direction",
            "writing_mode",
            "detection_confidence",
            // "Not measured" must stay distinguishable from "measured as zero":
            // three of the five engines report no confidence at all.
            "ocr_confidence",
            // The lettering signals are absent on a region detection could not
            // measure, and an absent signal must not read as a real value: a
            // font_size of 0 or a role of "" would both be taken for data.
            "role",
            "font_size",
            "color",
            // A region detection gave no class to must not read as one: an empty
            // string here would be taken for a real label that is not
            // `onomatopoeia`, which is exactly the test the sfx gate makes.
            "label",
            // Absent stroke and weight mean "detection refused one" / "family
            // default", and a zero here would read as a measured value.
            "stroke_color",
            "stroke_width",
            "font_weight",
        ] {
            assert!(json.get(key).is_none(), "{key} should be omitted");
        }
        // Internal, and never on the wire.
        assert!(json.get("content").is_none(), "{json}");
    }

    /// The one classification a caller cannot reconstruct for itself. Once an
    /// effect is promoted, it and a line of dialogue share `region_kind`, and
    /// `skipped` reports only effects that were never lettered -- so if this
    /// stops being emitted, nothing on the wire can count sound effects.
    #[test]
    fn the_detectors_label_reaches_the_json() {
        let mut region = region(EntityId::new(), "ピッ", "*Tap*");
        // The kind a real promoted effect carries, read off a measured run.
        region.region_kind = Some("dev.koharu.region.text".to_owned());
        region.label = Some("onomatopoeia".to_owned());
        let json = serde_json::to_value(&region).unwrap();
        assert_eq!(json.get("label").and_then(|v| v.as_str()), Some("onomatopoeia"));
        // The kind does NOT distinguish it, which is why the label is reported.
        assert_ne!(json.get("region_kind"), json.get("label"));
    }

    fn region(content: EntityId, source: &str, translated: &str) -> RegionOut {
        RegionOut {
            content,
            // Distinct from `content` on purpose: they are different entities in
            // a real scene, and a helper that reused one id would let a join keyed
            // on the wrong field pass its test.
            layer: EntityId::new(),
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            source: source.to_owned(),
            translated: translated.to_owned(),
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

    /// The occlusion marker on measured geometry. The unread-banner arm fires
    /// on a display column (banner hint 10.60 px above the read box); the
    /// watermark arm fires on a seam composite's balloon, which truly
    /// intersects its refused plate; and the controls hold: a read 255/631 px
    /// from its page's refused watermark stays unmarked, a refused region is
    /// never itself marked, and a hint that is the region's own sub-floor
    /// near-duplicate (IoU ~0.95) is skipped while a small intersecting
    /// occluder -- the strongest case of the class -- still fires.
    #[test]
    fn occlusion_marks_suspect_reads_and_only_those() {
        let watermark_reason = crate::labels::Refusal::Watermark.why();

        // Hint arm, disjoint by 10.60 px on y.
        let mut banner_page = vec![region(EntityId::new(), "铁甲寒光", "Cold Iron Gleam")];
        banner_page[0].x = 98.4;
        banner_page[0].y = 111.684;
        banner_page[0].width = 136.8;
        banner_page[0].height = 759.996;
        stamp_occlusion(&mut banner_page, &[(33.984375, 3.3251953, 280.07812, 97.76074)]);
        assert_eq!(banner_page[0].occluded_by, Some("unread-box"));

        // Watermark arm: a seam composite's balloon truly intersects the refused plate.
        let mut seam = vec![
            region(EntityId::new(), "最新免费漫画", ""),
            region(EntityId::new(), "听雨", "Listening to the Rain"),
        ];
        seam[0].x = 31.0546875;
        seam[0].y = 399.0234375;
        seam[0].width = 278.3203125;
        seam[0].height = 69.1640625;
        seam[0].refused = Some(watermark_reason);
        seam[1].x = 240.387;
        seam[1].y = 12.025;
        seam[1].width = 766.101;
        seam[1].height = 926.513;
        stamp_occlusion(&mut seam, &[]);
        assert_eq!(seam[1].occluded_by, Some("watermark"));
        assert_eq!(
            seam[0].occluded_by, None,
            "a refused region is never itself marked"
        );

        // The sharp control -- its own page carries a refused watermark, far
        // away. A pass that marks this is marking the whole page.
        let mut far_page = vec![
            region(EntityId::new(), "裂风之王！", "King Gale!"),
            region(EntityId::new(), "最新免费漫画", ""),
        ];
        far_page[0].x = 562.844;
        far_page[0].y = -35.637;
        far_page[0].width = 430.562;
        far_page[0].height = 209.603;
        far_page[1].x = 41.6015625;
        far_page[1].y = 805.140625;
        far_page[1].width = 265.4296875;
        far_page[1].height = 42.5625;
        far_page[1].refused = Some(watermark_reason);
        stamp_occlusion(&mut far_page, &[]);
        assert_eq!(far_page[0].occluded_by, None);

        // The same-ink shape, both extents: a hint that is the region's own ink --
        // near-duplicate OR a slice-half fragment inside a joined column
        // (IoU 0.30, containment 0.98, the case that over-fired the first
        // render) -- is skipped; a banner CROSSING the column, extending well
        // past both sides, still fires.
        let mut column = vec![region(EntityId::new(), "炎之王·冥", "King Blue Flame: M")];
        column[0].x = 100.0;
        column[0].y = 10.0;
        column[0].width = 204.0;
        column[0].height = 906.0;
        stamp_occlusion(&mut column, &[(103.0, 12.0, 198.0, 903.0)]);
        assert_eq!(
            column[0].occluded_by, None,
            "a near-duplicate of the region itself is not an occluder"
        );
        stamp_occlusion(&mut column, &[(103.0, 12.0, 198.0, 450.0)]);
        assert_eq!(
            column[0].occluded_by, None,
            "a fragment of the region's own ink is not an occluder either"
        );
        stamp_occlusion(&mut column, &[(0.0, 400.0, 600.0, 80.0)]);
        assert_eq!(
            column[0].occluded_by,
            Some("unread-box"),
            "a band crossing the read and extending past it still fires"
        );
    }

    /// The read's own score is a different quantity from the box's, and the wire
    /// must carry both under different names or a caller graphs the detector and
    /// calls it OCR.
    #[test]
    fn the_ocr_confidence_reaches_the_json_beside_the_detectors() {
        let mut region = region(EntityId::new(), "冥霜之王！", "King Frost!");
        region.detection_confidence = Some(0.5078125);
        region.ocr_confidence = Some(0.42);
        let json = serde_json::to_value(&region).unwrap();
        assert_eq!(
            json.get("ocr_confidence").and_then(|v| v.as_f64()),
            Some(f64::from(0.42f32))
        );
        assert_ne!(
            json.get("ocr_confidence"),
            json.get("detection_confidence"),
            "two scores, two names -- collapsing them is how the box's score got quoted as the read's"
        );
    }

    #[test]
    fn reported_entities_become_region_indices() {
        let missed = EntityId::new();
        let regions = [
            region(EntityId::new(), "こんにちは", "Hello"),
            region(missed, "もう帰ろう！", "もう帰ろう！"),
            region(EntityId::new(), "…", "…"),
        ];
        assert_eq!(untranslated_indices(&regions, &[missed]), [1]);
    }

    /// The reason this matches on entities instead of on the text: index 2 above
    /// is a translation the pipeline made on purpose, and index 0 would round
    /// trip identically if the bubble held romaji or a number.
    #[test]
    fn text_that_translates_to_itself_is_not_a_miss() {
        let regions = [
            region(EntityId::new(), "…", "…"),
            region(EntityId::new(), "Ka-boom!!", "Ka-boom!!"),
        ];
        assert!(untranslated_indices(&regions, &[]).is_empty());
    }

    #[test]
    fn an_entity_with_no_region_is_dropped_rather_than_guessed_at() {
        let regions = [region(EntityId::new(), "こんにちは", "Hello")];
        assert!(untranslated_indices(&regions, &[EntityId::new()]).is_empty());
    }

    /// The four regions a measured page lost, in invented text of the same
    /// shape. Every counter on that page read clean -- `untranslated: []`,
    /// `truncated: false`, `duplicate_ids: 0`, `out_of_range_ids: 0` -- and
    /// four boxes were erased and painted with nothing.
    #[test]
    fn a_region_answered_with_an_empty_string_is_a_drop() {
        let regions = [
            region(
                EntityId::new(),
                "ミナにとって\n生まれて初めて聞く\n『波の音』だった",
                "was a ",
            ),
            region(
                EntityId::new(),
                "ミナの朝食は\n港町の小さなパン屋から\n毎朝六時ちょうどに\n焼きたてが届けられる",
                "",
            ),
            region(
                EntityId::new(),
                "香ばしい皮は勿論\n季節の果実を練り込んだ\n甘さ控えめのパン",
                "",
            ),
            region(EntityId::new(), "それがミナの知る『朝食』である", ""),
            region(EntityId::new(), "ｧ3 呢", ""),
        ];
        assert_eq!(dropped_indices(&regions), [1, 2, 3, 4]);
        // The counter this one exists beside sees none of it: no entity was
        // ever reported, because the model answered every id.
        assert!(untranslated_indices(&regions, &[]).is_empty());
    }

    /// The mid-sentence split marker fires on exactly the trailing
    /// fragment, and its exclusions really exclude.
    #[test]
    fn a_trailing_fragment_is_flagged_and_the_neighbour_classes_are_not() {
        let mut regions = [
            // A reply segment that stopped mid-sentence, shaped like the corpus.
            region(EntityId::new(), "ミナにとって", "was a "),
            // An ordinary complete translation is left alone.
            region(EntityId::new(), "駄目", "No way."),
            // All-whitespace is the DROP class, not this one -- one defect must
            // not be accused twice.
            region(EntityId::new(), "それが", "   "),
            // A trailing newline is a trailing fragment too: the census
            // predicate is rstrip, not a literal space.
            region(EntityId::new(), "栄養", "and the\n"),
        ];
        assert_eq!(split_fragment_indices(&regions), [0, 3]);

        // A refused region is the gate's business, however its text ends.
        regions[0].refused = Some("a site watermark, not dialogue");
        assert_eq!(split_fragment_indices(&regions), [3]);
    }

    /// The population this rule must never touch. A translation that came back
    /// in the source language is `untranslated`'s business and is *lettered* --
    /// the box holds Japanese, not a blank patch -- and the two deliberate
    /// self-translations are not misses at all.
    #[test]
    fn ordinary_and_untranslated_regions_stay_silent() {
        let regions = [
            // A measured 465x198 box has this shape (the text is invented)
            // when the page works: source in, English out.
            region(
                EntityId::new(),
                "その頑固な性格を除けば、とても親切な先生といえる。",
                "Aside from that stubborn streak, she can be called a very kind teacher.",
            ),
            // Written back verbatim by the translation stage for an id the model
            // never answered. This renders as Japanese, not as a hole.
            region(EntityId::new(), "もう帰ろう！", "もう帰ろう！"),
            // The ellipsis the translation stage maps to itself on purpose.
            region(EntityId::new(), "…", "…"),
            // Romaji round-tripping identically.
            region(EntityId::new(), "Ka-boom!!", "Ka-boom!!"),
        ];
        assert!(dropped_indices(&regions).is_empty());
    }

    /// A deliberately blanked read, and the `clean_only` plan, both arrive as an
    /// empty source. 1,502 regions in the corpus look like this and every one of
    /// them also has an empty translation, so the source test -- not a special
    /// case -- is what keeps them out.
    #[test]
    fn a_blanked_or_unread_source_is_not_a_drop() {
        let regions = [
            region(EntityId::new(), "", ""),
            region(EntityId::new(), "   ", ""),
        ];
        assert!(dropped_indices(&regions).is_empty());
    }

    /// A refusal is this server's own decision and is reported through its own
    /// field. A 611x394 box whose read was deliberately blanked is the measured
    /// case; the watermark and punctuation rules land the same way.
    #[test]
    fn a_refused_region_is_the_gates_business_and_not_this_ones() {
        let regions = [
            RegionOut {
                refused: Some("OCR returned nothing"),
                ..region(EntityId::new(), "ピーーーーーーーー", "")
            },
            RegionOut {
                refused: Some("a site watermark, not dialogue"),
                ..region(EntityId::new(), "www.paperleaf.test", "")
            },
            RegionOut {
                refused: Some("no letters, only punctuation or symbols"),
                ..region(EntityId::new(), "・・・", "")
            },
        ];
        assert!(dropped_indices(&regions).is_empty());
    }

    /// The two cases nearest to a legitimate exemption, both real, and both
    /// counted anyway. `？` and `！？` came back empty on two runs of one
    /// measured page; an empty English is arguable, but both are
    /// in-bubble and both boxes were erased to flat white, which is the damage
    /// the refusal gate refuses to create on a balloon.
    #[test]
    fn a_punctuation_only_source_is_still_an_erased_box() {
        let regions = [
            region(EntityId::new(), "？", ""),
            region(EntityId::new(), "！？", ""),
        ];
        assert_eq!(dropped_indices(&regions), [0, 1]);
    }

    /// A reply of whitespace paints exactly as much as a reply of nothing.
    #[test]
    fn a_whitespace_reply_paints_nothing_and_counts() {
        let regions = [region(EntityId::new(), "ふざけないでよ！！", " \n ")];
        assert_eq!(dropped_indices(&regions), [0]);
    }

    fn warning(kind: &'static str, entity: EntityId, font_size: f32) -> crate::render::LayoutWarning {
        crate::render::LayoutWarning {
            kind,
            entity,
            region: None,
            font_size,
            minimum_font_size: None,
            actual_width: None,
            actual_height: None,
        }
    }

    /// The case the `font_size` heuristic cannot decide, and the reason this
    /// function exists.
    ///
    /// Three regions all solved at the 9px floor -- which is the *normal* state
    /// for warned layers, since that is where auto-fit stops -- and the warning
    /// belongs to the last one. Any join keyed on size alone has three equally
    /// good candidates and no way to choose; taking the first free one in order
    /// picks region 0 here, and would misattribute every per-role split built on
    /// it. Keyed on the layer there is exactly one answer.
    #[test]
    fn a_shared_font_size_does_not_make_the_join_ambiguous() {
        let mut regions = [
            region(EntityId::new(), "あ", "a"),
            region(EntityId::new(), "い", "b"),
            region(EntityId::new(), "う", "c"),
        ];
        // Distinct layers, identical solved size.
        for region in &mut regions {
            region.layer = EntityId::new();
        }
        let mut warnings = [warning("overflow", regions[2].layer, 9.0)];
        attribute_warnings(&mut warnings, &regions);
        assert_eq!(warnings[0].region, Some(2));
    }

    /// Two warnings on ONE layer both name it, and a `too_small` followed by an
    /// `overflow` on the same layer is the one sequence the renderer really does
    /// emit -- auto-fit bottoms out, then the floored text still does not fit.
    ///
    /// The layer under test is deliberately NOT region 0. An earlier draft of
    /// this test used a single region and passed against a join hardcoded to
    /// `Some(0)` -- it asserted the shape of the answer without discriminating
    /// the answer, which is why a test must be shown able to go red.
    #[test]
    fn one_layer_carrying_two_warnings_names_itself_twice() {
        let mut regions = [
            region(EntityId::new(), "あ", "a"),
            region(EntityId::new(), "い", "b"),
        ];
        for region in &mut regions {
            region.layer = EntityId::new();
        }
        let layer = regions[1].layer;
        let mut warnings = [warning("too_small", layer, 9.0), warning("overflow", layer, 9.0)];
        attribute_warnings(&mut warnings, &regions);
        assert_eq!(
            warnings.iter().map(|w| w.region).collect::<Vec<_>>(),
            [Some(1), Some(1)]
        );
    }

    /// An unmatched warning reports `None` rather than borrowing a neighbour's
    /// index. Silence here would be worse than a gap: a reader cannot tell a
    /// wrong attribution from a right one, but it can see a missing key.
    #[test]
    fn a_warning_with_no_region_behind_it_is_left_unattributed() {
        let regions = [region(EntityId::new(), "あ", "a")];
        let mut warnings = [warning("overflow", EntityId::new(), 9.0)];
        attribute_warnings(&mut warnings, &regions);
        assert_eq!(warnings[0].region, None);
    }

    fn solved(entity: EntityId, placed: (f32, f32, f32, f32)) -> crate::render::RenderedText {
        crate::render::RenderedText {
            entity,
            region: None,
            font_size: 9.0,
            chars: 1,
            // The box is deliberately the SAME on every layer here. It is what a
            // reader holding only the JSON would have to join on, and joining on
            // it is exactly what this function exists to make unnecessary.
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            placed_x: placed.0,
            placed_y: placed.1,
            placed_width: placed.2,
            placed_height: placed.3,
            placed_angle: 0.0,
        }
    }

    /// The solved layers arrive in COMPOSITION order, which is not the region
    /// order, and this is the case that proves the join is keyed on the layer
    /// rather than on position: three regions, three layers, presented backwards.
    ///
    /// A join that simply zipped the two arrays would answer 0, 1, 2 and pass any
    /// test whose layers happened to be in order. This one goes red.
    #[test]
    fn solved_layers_out_of_region_order_still_name_themselves() {
        let mut regions = [
            region(EntityId::new(), "あ", "a"),
            region(EntityId::new(), "い", "b"),
            region(EntityId::new(), "う", "c"),
        ];
        for region in &mut regions {
            region.layer = EntityId::new();
        }
        let mut text = [
            solved(regions[2].layer, (10.0, 10.0, 5.0, 5.0)),
            solved(regions[0].layer, (20.0, 20.0, 5.0, 5.0)),
            solved(regions[1].layer, (30.0, 30.0, 5.0, 5.0)),
        ];
        attribute_rendered_text(&mut text, &regions);
        assert_eq!(
            text.iter().map(|t| t.region).collect::<Vec<_>>(),
            [Some(2), Some(0), Some(1)]
        );
        // The placed extent travels with the layer it was solved for, which is
        // the whole reason to carry the index: read the wrong way round, region 0
        // would be handed region 2's rectangle.
        assert_eq!(text[1].region, Some(0));
        assert_eq!((text[1].placed_x, text[1].placed_y), (20.0, 20.0));
    }

    /// A solved layer with no region behind it is left unattributed rather than
    /// borrowing index 0, for the same reason a warning is.
    #[test]
    fn a_solved_layer_with_no_region_behind_it_is_left_unattributed() {
        let regions = [region(EntityId::new(), "あ", "a")];
        let mut text = [solved(EntityId::new(), (1.0, 2.0, 3.0, 4.0))];
        attribute_rendered_text(&mut text, &regions);
        assert_eq!(text[0].region, None);
    }

    /// The wire contract, pinned against two pages the server actually served:
    /// `src/fixtures/wire/{001,007}.json` are recorded `format=json` response
    /// bodies minus the `image` key, with the page text (`source`,
    /// `translated`) replaced by invented text of the same shape and
    /// `rendered_text[].chars` updated to match. Between them
    /// the seven regions carry `refused` and `occluded_by` present AND
    /// absent, so every `skip_serializing_if` on
    /// those fields is pinned in both directions; `stroke_color`,
    /// `stroke_width` and `font_weight` are pinned in the absent direction
    /// only (this corpus never emits them), and the envelope around
    /// `regions[]` is out of scope -- `TranslateJson` is routes-private and
    /// its synthetic `a_json_body_carries_*` tests already cover it.
    ///
    /// Both sides go through the STRING serializer deliberately:
    /// `serde_json::to_value` widens `f32` through `f64` (0.99955285 becomes
    /// 0.999552845954895) while `to_string` uses the shortest-f32 formatter
    /// the real wire uses, so a `to_value` compare fails on every confidence.
    ///
    /// A red here means the wire format CHANGED. The fixture cannot be
    /// regenerated without a GPU run -- if the change is deliberate,
    /// hand-edit the fixture JSON in the same commit and say so.
    ///
    /// Proven able to fail three ways, each restored: deleting `refused`'s
    /// `skip_serializing_if` (three regions gain `"refused":null`), deleting
    /// `layer`'s `#[serde(skip)]` (an extra key appears), and renaming
    /// `fit_x` to `fitx`.
    #[test]
    fn the_recorded_wire_files_pin_region_serialization() {
        fn recorded(
            rect: [f64; 4],
            source: &str,
            translated: &str,
            detection_confidence: f32,
            ocr_confidence: f32,
            font_size: f32,
            color: [u8; 4],
        ) -> RegionOut {
            RegionOut {
                content: EntityId::new(),
                layer: EntityId::new(),
                x: rect[0],
                y: rect[1],
                width: rect[2],
                height: rect[3],
                source: source.to_owned(),
                translated: translated.to_owned(),
                fit_x: Some(rect[0]),
                fit_y: Some(rect[1]),
                fit_width: Some(rect[2]),
                fit_height: Some(rect[3]),
                source_language: Some("zh-CN".to_owned()),
                target_language: Some("en-US".to_owned()),
                detection_confidence: Some(detection_confidence),
                ocr_confidence: Some(ocr_confidence),
                direction: Some("horizontal"),
                writing_mode: Some("horizontal"),
                region_kind: Some("dev.koharu.region.text".to_owned()),
                label: Some("text".to_owned()),
                role: Some("free-text".to_owned()),
                font_size: Some(font_size),
                color: Some(color),
                stroke_color: None,
                stroke_width: None,
                font_weight: None,
                refused: None,
                occluded_by: None,
            }
        }

        let page_001 = [
            recorded(
                [459.375, 144.7177734375, 290.625, 51.318359375],
                "第一百二十三话",
                "Chapter 123",
                0.2578125,
                0.99955285,
                51.0,
                [241, 129, 19, 255],
            ),
            RegionOut {
                refused: Some("the same text is lettered by an overlapping region"),
                ..recorded(
                    [379.6875, 316.12109375, 407.8125, 53.37109375],
                    "雾海孤舟夜渡寒江",
                    "A Lone Boat Crosses the Cold River by Night",
                    0.25585938,
                    0.9994354,
                    52.0,
                    [247, 127, 11, 255],
                )
            },
            recorded(
                [356.25, 322.279296875, 478.125, 498.814453125],
                "雾海孤舟夜渡寒江\n编创组：甲乙丙 tab 东篱 秋水\n总编剧：方圆\n主笔：糯米 橘子√ 小舟\n助理：南风 青梅竹马 不吃香菜\n雨后 PC 墨鱼丸 山茶\n责编：汽水 北窗月\n监制：周一 吴同学\n出品人：某某",
                "A Lone Boat Crosses the Cold River by Night\nCreation Team: Jia Yi Bing tab Dong Li Qiu Shui\nHead Writer: Fang Yuan\nMain Artist: Nuo Mi Ju Zi Xiao Zhou\nAssistants: Nan Feng Qing Mei Zhu Ma Bu Chi Xiang Cai\nYu Hou PC Mo Yu Wan Shan Cha\nEditor: Qi Shui Bei Chuang Yue\nSupervisors: Zhou Yi Wu Tong Xue\nProducer: Mou Mou",
                0.52734375,
                0.9958314,
                92.0,
                [255, 255, 255, 255],
            ),
            recorded(
                [459.375, 1904.9375, 84.375, 32.84375],
                "穆叶",
                "Mu Ye",
                0.27734375,
                0.7307869,
                34.0,
                [255, 205, 230, 255],
            ),
        ];

        let page_007 = [
            recorded(
                [513.28125, 570.96875, 236.71875, 52.9375],
                "称号：青衫",
                "Title: Qing Shan",
                0.43359375,
                0.9994684,
                51.0,
                [248, 141, 33, 255],
            ),
            RegionOut {
                occluded_by: Some("unread-box"),
                ..recorded(
                    [508.59375, 733.5625, 208.59375, 52.9375],
                    "状态激活！",
                    "Status Activated!",
                    0.25585938,
                    0.99700874,
                    51.0,
                    [247, 137, 27, 255],
                )
            },
            recorded(
                [173.4375, 786.5, 900.0, 181.5],
                "当前效果：每当你在夜晚独自赶路时，脚下会自动亮起一盏引路灯笼，照亮前方十步之内的道路。在灯笼熄灭之前，你的移动速度将会",
                "Current Effect: Whenever you travel alone at night, a guiding lantern automatically lights up at your feet, illuminating the road within ten paces ahead. Until the lantern goes out, your movement speed will",
                0.42382812,
                0.9867962,
                52.0,
                [0, 67, 65, 255],
            ),
        ];

        let pages: [(&str, &str, &[RegionOut]); 2] = [
            (
                "001",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/fixtures/wire/001.json")),
                &page_001,
            ),
            (
                "007",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/fixtures/wire/007.json")),
                &page_007,
            ),
        ];
        for (name, fixture, built) in pages {
            let golden: serde_json::Value =
                serde_json::from_str(fixture).expect("the recorded wire file parses");
            let golden_regions = golden["regions"]
                .as_array()
                .expect("the recorded body has a regions array");
            assert_eq!(
                golden_regions.len(),
                built.len(),
                "page {name}: one literal per recorded region"
            );
            for (i, region) in built.iter().enumerate() {
                let actual: serde_json::Value = serde_json::from_str(
                    &serde_json::to_string(region).expect("RegionOut serializes"),
                )
                .expect("its own output parses");
                assert_eq!(
                    actual, golden_regions[i],
                    "page {name}, region {i}: the serializer no longer writes what the server shipped"
                );
            }
        }
    }
}
