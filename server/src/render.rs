//! Compositing the finished scene into PNG bytes.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context as _, Result};
use image::{
    RgbaImage,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use koharu_renderer::{
    Compositor, HyphenationPolicy, RasterOptions, Rasterizer, RenderDiagnostic, RenderRequest,
    SceneRenderer, VisualLayerKind,
};
use koharu_scene::{EntityId, Snapshot};
use serde::Serialize;

/// Built once at startup and shared behind an `Arc`.
///
/// All three take `&self` and guard their own caches, and koharu itself keeps a
/// `Rasterizer` in a `static` and shares all three across a rayon iterator, so
/// sharing them is sound. Rebuilding them per request would throw away the
/// scene renderer's image and frame caches and, worse, spin up a new wgpu
/// device every time.
pub struct Renderers {
    compositor: Compositor,
    scenes: SceneRenderer,
    rasterizer: Rasterizer,
    /// Sticky, and set the first time a render fails while a font list was in
    /// force. See `render_png`.
    bundled_fonts_unavailable: AtomicBool,
}

impl Renderers {
    /// Blocking: `Rasterizer::new` parks the calling thread while it acquires a
    /// wgpu adapter. Call it from `spawn_blocking`, at startup, so a machine
    /// without a usable adapter fails loudly instead of mid-request.
    pub fn new() -> Result<Self> {
        Ok(Self {
            compositor: Compositor::new(),
            scenes: SceneRenderer::new(),
            rasterizer: Rasterizer::new().context("failed to initialize the rasterizer")?,
            bundled_fonts_unavailable: AtomicBool::new(false),
        })
    }

    /// Blocking and GPU-touching. Run it under `spawn_blocking`, holding the
    /// server's GPU permit.
    ///
    /// The result is exactly the upload's pixel dimensions: the page was seeded
    /// from the decoded image, the composition surface is the page size, and
    /// `RasterOptions::default()` is a strict 1:1 render.
    /// Falls back to the installed fonts rather than failing the page.
    ///
    /// The default list leads with "CCWildWords", which is not installed and so
    /// is resolved from Koharu's bundled catalog -- a Hugging Face dataset read
    /// at `Revision::Latest`, with no offline path. Font resolution happens at
    /// *render* time, after detection, OCR, translation and inpainting have all
    /// run, so an unreachable catalog would throw away a whole GPU page.
    ///
    /// That is not hypothetical. Upstream moved `mayocream/fonts` to index
    /// schema 3 while `koharu-renderer` still pinned 2, and every CCWildWords
    /// render died with `unsupported bundled font index schema 3` -- 23s in.
    /// Our patch teaches it schema 3; this is the belt to that braces, for the
    /// next schema bump, an offline machine, or a family that simply is not in
    /// the catalog.
    ///
    /// The retry is deliberately not conditioned on the error text. It costs
    /// one extra render (~0.1s) at most **once per process**, because the flag
    /// is sticky: a non-font failure would mean lettering in Arial until the
    /// next restart, which is a far better trade than parsing error strings.
    pub fn render_png(
        &self,
        snapshot: &Snapshot,
        page: EntityId,
        font_families: &[String],
        hyphenation: Option<HyphenationPolicy>,
        size_coherence: Option<f32>,
        collision_relief: bool,
        edge_anchored_lettering: bool,
    ) -> Result<Rendered> {
        if font_families.is_empty() || self.bundled_fonts_unavailable.load(Ordering::Relaxed) {
            return self.render_with(snapshot, page, &[], hyphenation, size_coherence, collision_relief, edge_anchored_lettering);
        }
        match self.render_with(snapshot, page, font_families, hyphenation, size_coherence, collision_relief, edge_anchored_lettering) {
            Ok(rendered) => Ok(rendered),
            Err(error) => {
                tracing::warn!(
                    %error,
                    "render failed with {font_families:?}; falling back to the installed fonts \
                     for the rest of this process"
                );
                self.bundled_fonts_unavailable
                    .store(true, Ordering::Relaxed);
                self.render_with(snapshot, page, &[], hyphenation, size_coherence, collision_relief, edge_anchored_lettering)
            }
        }
    }

    fn render_with(
        &self,
        snapshot: &Snapshot,
        page: EntityId,
        font_families: &[String],
        hyphenation: Option<HyphenationPolicy>,
        size_coherence: Option<f32>,
        collision_relief: bool,
        edge_anchored_lettering: bool,
    ) -> Result<Rendered> {
        let request = render_request(page, font_families, hyphenation, size_coherence, collision_relief, edge_anchored_lettering);
        let composition = self.compositor.compile(snapshot, &request)?;
        let frame = self.scenes.render(snapshot, &composition)?;
        // Read BEFORE the rasterize, because that consumes nothing but there is
        // no reason to hold the frame longer than needed.
        let warnings = layout_warnings(frame.diagnostics());
        let text = solved_text(&frame);
        let raster = self.rasterizer.rasterize(&frame, RasterOptions::default())?;
        // A full-page frame always starts at the origin; only per-entity crops
        // move it.
        debug_assert_eq!((raster.left, raster.top), (0, 0));
        Ok(Rendered {
            png: encode_png(&raster.image)?,
            warnings,
            text,
        })
    }
}

/// One finished page, and what the renderer has to say about it.
pub struct Rendered {
    pub png: Vec<u8>,
    pub warnings: Vec<LayoutWarning>,
    /// Every text layer as the renderer actually **set** it.
    ///
    /// `font_size` here is not the same number as a region's `font_size`, which
    /// is what detection *authored* from the Japanese em -- auto-fit then
    /// re-solves it against the box, and for bubble text the balloon auto-fit
    /// ignores the authored value outright. Reading the authored figure and
    /// calling it the rendered one is a real trap: a whole rendering change once
    /// measured as a no-op through it.
    ///
    /// The box travels with the size because a size on its own cannot be
    /// judged. "86px" is a defect in a speech balloon and correct in a burst
    /// balloon holding two characters, and the only way to tell those apart
    /// after the fact is to see what the layer was holding and how big its box
    /// was.
    pub text: Vec<RenderedText>,
}

/// One text layer, as set.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RenderedText {
    /// The text layer, so a caller can match it back to a region.
    #[serde(skip)]
    pub entity: EntityId,
    /// Which region this layer is, as an index into the response's `regions`
    /// array. Filled in by [`crate::regions::attribute_rendered_text`], for the
    /// same reason [`LayoutWarning::region`] is: the renderer sees layers, and a
    /// reader holding only the JSON cannot recover the pairing.
    ///
    /// Without it this array is unusable for anything per-region. The layers
    /// arrive in composition order, which is not the region order, and matching
    /// on `font_size` collides by construction -- most solved layers on a dense
    /// page share the 9px floor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<usize>,
    pub font_size: f32,
    /// Non-whitespace characters of the text that was set.
    pub chars: usize,
    /// The layout box, which is the same rectangle a region reports as `fit_*`.
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    /// Where the text was actually **placed**, which is not the box above.
    ///
    /// # Why this is the whole point
    ///
    /// Every collision instrument this project has built compares fit *boxes*,
    /// and box overlap has been retired as a predictor: boxes overlapping
    /// does not mean the rendered text overlaps, because auto-fit routinely sets
    /// a translation far smaller than the box it was given and then centres it.
    /// A region can honour its box on paper and violate it in pixels -- and,
    /// measured here for the first time, the reverse is just as common: a box
    /// can overlap a neighbour while the ink inside it comes nowhere near.
    ///
    /// The renderer has always known the answer. `text_renderer` solves the
    /// layout, then places it with `placement(bounds, layout.width,
    /// layout.height, vertical_alignment)` and records the result as
    /// `VisualText::rendered_bounds` -- the true placed extent of the text that
    /// was set. `solved_text` read `layout_bounds` (the box) beside it and threw
    /// `rendered_bounds` away. This is the same shape as the
    /// `x-birelate-untranslated` bug and the `LayoutWarning` one above: the
    /// pipeline computes the answer and BireLate drops it.
    ///
    /// This carries **no behaviour change** -- it is the measurement that decides
    /// whether a post-layout collision rule is worth building at all.
    pub placed_x: f32,
    pub placed_y: f32,
    pub placed_width: f32,
    pub placed_height: f32,
    /// The rotation the text was set at, about the placed rectangle's own centre.
    ///
    /// Carried because without it the rectangle above is un-rotated and therefore
    /// wrong on exactly the population that owns the defect: the pipeline gives a
    /// free-standing text with a real angle a rotated layout box, and sound
    /// effects are 88% of severe ink collisions. Reading `placed_*` as an
    /// axis-aligned box on an angled effect re-inflates it toward the loose bbox
    /// this pair exists to escape.
    ///
    /// `text_renderer` rotates about `layout_rect.center()`, so the four corners
    /// of the true oriented box are recoverable from these five numbers alone.
    pub placed_angle: f32,
}

/// A layer the renderer could not set properly, in the shape the extension and
/// `format=json` want.
///
/// The renderer has always computed these -- `text_renderer` emits
/// `TextOverflow` when a layout does not fit its box and
/// `TextBelowReadableSize` when auto-fit bottoms out at `minimum_font_size` --
/// and `render_png` threw the whole `Frame`'s diagnostics away, using it only to
/// rasterize. That is the same shape as the `x-birelate-untranslated` bug: the
/// pipeline computes the answer and BireLate drops it. Reporting them turns
/// "does the text look wrong on this page?" from an eyeballing exercise into a
/// number per region.
///
/// `MissingBaseAsset` and `UsedSourceText` are deliberately not carried: the
/// first cannot happen on a page we seeded ourselves, and the second is
/// near-unreachable in shipping -- the translation stage writes every skipped
/// segment back as a `Translation` holding the source verbatim
/// (`stages/translation.rs`), so the compositor's source-text fallback never
/// sees a non-blank candidate; a census over 127,571 stored regions found zero.
/// What a reader experiences as "untranslated" is that echoed
/// source, which `untranslated` already reports better.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayoutWarning {
    /// `overflow` or `too_small`.
    pub kind: &'static str,
    /// The text layer, so a caller can match it back to a region.
    #[serde(skip)]
    pub entity: EntityId,
    /// Which region this warning is about, as an index into the response's
    /// `regions` array -- the same denominator `untranslated` and `dropped` use.
    ///
    /// The comment above `entity` has always said a caller "can match it back to
    /// a region", and that was never true through the wire: `entity` is
    /// `serde(skip)` and no other field identified the layer, so a reader holding
    /// only the JSON had to *guess* -- walk the warnings in order taking the next
    /// solved layer whose `font_size` matches -- and that pairing is not unique.
    /// It cannot be: most overflowing layers bottom out at the same 9px floor, so
    /// the checksum collides across a whole page.
    ///
    /// `None` means the join genuinely failed rather than that it was not
    /// attempted -- a layer the region walk did not produce, which should not
    /// happen because both read the same finished scene. A caller seeing `None`
    /// should treat the warning as unattributed, not silently drop it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<usize>,
    pub font_size: f32,
    /// Only on `too_small`: the floor auto-fit refused to go below.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_font_size: Option<f32>,
    /// Only on `overflow`: how far past its box the text actually ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_width: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_height: Option<f32>,
}

fn solved_text(frame: &koharu_renderer::Frame) -> Vec<RenderedText> {
    frame
        .layers()
        .iter()
        .filter(|layer| layer.kind == VisualLayerKind::Text)
        .filter_map(|layer| {
            let text = layer.text.as_ref()?;
            Some(RenderedText {
                entity: layer.entity,
                region: None,
                font_size: text.font_size,
                chars: text.text.chars().filter(|c| !c.is_whitespace()).count(),
                x: text.layout_bounds.x,
                y: text.layout_bounds.y,
                width: text.layout_bounds.width,
                height: text.layout_bounds.height,
                placed_x: text.rendered_bounds.x,
                placed_y: text.rendered_bounds.y,
                placed_width: text.rendered_bounds.width,
                placed_height: text.rendered_bounds.height,
                placed_angle: text.angle_degrees,
            })
        })
        .collect()
}

fn layout_warnings(diagnostics: &[RenderDiagnostic]) -> Vec<LayoutWarning> {
    diagnostics
        .iter()
        .filter_map(|diagnostic| match diagnostic {
            RenderDiagnostic::TextOverflow {
                entity,
                actual_width,
                actual_height,
                font_size,
                ..
            } => Some(LayoutWarning {
                kind: "overflow",
                entity: *entity,
                // Filled in by the caller, which is the only place that knows the
                // region order; the renderer sees layers, not the response array.
                region: None,
                font_size: *font_size,
                minimum_font_size: None,
                actual_width: Some(*actual_width),
                actual_height: Some(*actual_height),
            }),
            RenderDiagnostic::TextBelowReadableSize {
                entity,
                font_size,
                minimum_font_size,
            } => Some(LayoutWarning {
                kind: "too_small",
                entity: *entity,
                region: None,
                font_size: *font_size,
                minimum_font_size: Some(*minimum_font_size),
                actual_width: None,
                actual_height: None,
            }),
            _ => None,
        })
        .collect()
}

/// Builds the render request, overwriting the theme's font list.
///
/// The list is assigned rather than merged, so `--font-family Arial` really
/// does drop CCWildWords instead of leaving it in front where it would still
/// win.
///
/// An empty list is the fallback, not the default. It leaves the resolver to
/// append the installed "Arial" -- which is what this function used to be
/// *for*, back when the point was to keep "CCWildWords" out of the request and
/// avoid the bundled-catalog fetch entirely. That trade is now made the other
/// way round in `cli.rs`, and `Renderers::render_png` is what falls back here
/// when the catalog cannot be reached.
///
/// The theme's `text_stroke` is deliberately left at `None`. It looks like the
/// cheap way to halo the text that loses its inpainted fill on a kept-art page --
/// `text_renderer` resolves a halo as `layer.stroke.or(theme.text_stroke)`, so
/// the theme is reached only by layers carrying no stroke of their own, which
/// in this fork is exactly the in-bubble ones. It targets correctly and is
/// still wrong, because the theme holds one *absolute* width for every layer
/// while the renderer then scales each layer's stroke by
/// `layout.font_size / layer.font_size`. The effective ratio is therefore
/// inversely proportional to the layer's own size: measured on the test pages, a
/// theme width of 0.72px lands at ratio 0.007 on a 98px layer -- invisible -- and
/// at 0.144 on a 5px one, which is over twice the width that closes Arial's
/// counters. It simultaneously does nothing to large bubbles and destroys small
/// ones. `lettering::halo_bubble_text` writes the width per layer instead, where
/// a ratio survives that scaling exactly.
#[must_use]
pub fn render_request(
    page: EntityId,
    font_families: &[String],
    hyphenation: Option<HyphenationPolicy>,
    size_coherence: Option<f32>,
    collision_relief: bool,
    edge_anchored_lettering: bool,
) -> RenderRequest {
    let mut request = RenderRequest::new(page);
    request.theme.font_families = font_families.to_vec();
    request.theme.hyphenation = hyphenation;
    request.theme.size_coherence = size_coherence;
    request.theme.collision_relief = collision_relief;
    request.theme.edge_anchored_lettering = edge_anchored_lettering;
    request
}

/// `PngEncoder` needs only `Write`, so no `Cursor` and no `Seek`.
///
/// `CompressionType::Fast` is also `image` 0.25's own default, so naming it
/// changes nothing today. It is named anyway so that a future `image` bump
/// cannot silently move this onto `Balanced`, which deflates a multi-megapixel
/// page through flate2 rather than fdeflate for no visible gain on a response
/// that is decoded and thrown away.
///
/// `FilterType::Up` instead of the default `Adaptive` is the real saving.
/// `Adaptive` filters every row four ways, sums each result and then re-filters
/// the winner -- around nine passes over the row against one. The file is
/// slightly larger; it travels over loopback.
fn encode_png(image: &RgbaImage) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(image.as_raw().len() / 3);
    image
        .write_with_encoder(PngEncoder::new_with_quality(
            &mut bytes,
            CompressionType::Fast,
            FilterType::Up,
        ))
        .context("failed to encode the rendered page as PNG")?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_produces_a_png() {
        let image = RgbaImage::from_pixel(2, 2, image::Rgba([12, 34, 56, 255]));
        let bytes = encode_png(&image).unwrap();
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn the_fast_encoder_still_round_trips() {
        // The extension assigns the response to an <img src>, so the pairing of
        // the fastest deflate mode with a non-default filter has to decode as an
        // ordinary PNG rather than merely carry the right magic bytes.
        let mut image = RgbaImage::from_pixel(37, 11, image::Rgba([12, 34, 56, 255]));
        for (index, pixel) in image.pixels_mut().enumerate() {
            pixel[0] = (index % 251) as u8;
        }
        let decoded = image::load_from_memory(&encode_png(&image).unwrap())
            .unwrap()
            .into_rgba8();
        assert_eq!(decoded, image);
    }

    #[test]
    fn an_empty_list_is_passed_through_so_the_resolver_appends_arial() {
        // This is the shape `render_png` falls back to when the bundled
        // catalog cannot be reached. It must stay empty rather than inherit
        // RenderTheme::default(), whose first entry is the very family that
        // could not be resolved.
        let request = render_request(EntityId::new(), &[], None, None, false, false);
        assert!(request.theme.font_families.is_empty());
    }

    #[test]
    fn a_configured_font_family_is_used_verbatim() {
        let request = render_request(EntityId::new(), &["Comic Sans MS".to_owned()], None, None, false, false);
        assert_eq!(request.theme.font_families, vec!["Comic Sans MS".to_owned()]);
    }

    #[test]
    fn the_theme_never_carries_a_page_wide_stroke() {
        /* A page-wide width cannot be right for every layer on the page: the
         * renderer divides each layer's stroke by that layer's own font size, so one
         * absolute number becomes a ratio inversely proportional to the text it
         * outlines. Haloing belongs in `lettering`, per layer. */
        assert!(
            render_request(EntityId::new(), &[], None, None, false, false)
                .theme
                .text_stroke
                .is_none()
        );
    }
}
