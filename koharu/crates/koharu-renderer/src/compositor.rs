//! Compilation of semantic scene components into a reusable composition.

use std::collections::BTreeSet;

use koharu_scene::{
    Asset, BlobId, EntityId, FILL_GRADIENT_TO_EXTENSION, Geometry, Group, LanguageTag,
    OcrAnalysis, Origin, Page, RasterLayer, RasterLayerKind, RelationId, Revision,
    STRIKE_COLOR_EXTENSION, Snapshot, SourceText, TextAlignment, TextDirection,
    TextLayout as SceneTextLayout, TextLayoutKind,
    Translation, Typography, Visibility,
};

use crate::{
    Error, LayerPresentation, RenderRequest, Result, StrokeOptions, TextAlign, WritingMode,
    bubble::{LayoutBox, geometry_bounds, geometry_frame},
    script::{is_cjk_text, shaping_direction_for_text},
};

const MAX_SURFACE_DIMENSION: u32 = 32_768;
const MAX_SURFACE_PIXELS: u64 = 268_435_456;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RenderDependency {
    Entity(EntityId),
    Relation(RelationId),
    Blob(BlobId),
}

#[derive(Clone, Debug, PartialEq)]
pub enum RenderDiagnostic {
    MissingBaseAsset {
        roles: Vec<String>,
    },
    UsedSourceText {
        entity: EntityId,
    },
    TextOverflow {
        entity: EntityId,
        available: RenderBounds,
        actual_width: f32,
        actual_height: f32,
        font_size: f32,
    },
    TextBelowReadableSize {
        entity: EntityId,
        font_size: f32,
        minimum_font_size: f32,
    },
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RenderBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl From<LayoutBox> for RenderBounds {
    fn from(value: LayoutBox) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Composition {
    pub(crate) revision: Revision,
    pub(crate) page: EntityId,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layers: Vec<Layer>,
    pub(crate) dependencies: Vec<RenderDependency>,
    pub(crate) diagnostics: Vec<RenderDiagnostic>,
    pub(crate) request: RenderRequest,
}

impl Composition {
    #[must_use]
    pub const fn revision(&self) -> Revision {
        self.revision
    }

    #[must_use]
    pub const fn page(&self) -> EntityId {
        self.page
    }

    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[must_use]
    pub fn dependencies(&self) -> &[RenderDependency] {
        &self.dependencies
    }

    #[must_use]
    pub fn diagnostics(&self) -> &[RenderDiagnostic] {
        &self.diagnostics
    }
}

/// Compiles a semantic Koharu page into ordered visual layers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Compositor;

impl Compositor {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Resolves scene capabilities into an owned, backend-independent composition.
    pub fn compile(&self, snapshot: &Snapshot, request: &RenderRequest) -> Result<Composition> {
        compile(snapshot, request)
    }
}

fn compile(snapshot: &Snapshot, request: &RenderRequest) -> Result<Composition> {
    validate_request(request)?;
    let page = snapshot.page(request.page)?.page()?;
    let (width, height) = surface_size(&page)?;
    let page_entities = snapshot
        .subtree(request.page)?
        .map(|entity| entity.id())
        .collect::<BTreeSet<_>>();
    let mut dependencies = BTreeSet::from([RenderDependency::Entity(request.page)]);
    let mut diagnostics = Vec::new();
    let mut layers = Vec::new();

    let mut base_found = false;
    for role in &request.base_assets {
        let Some(asset) = snapshot.asset(request.page, role)? else {
            continue;
        };
        dependencies.insert(RenderDependency::Blob(asset.blob));
        layers.push(Layer::Image(ImageLayer {
            entity: request.page,
            asset,
            name: None,
            kind: crate::VisualLayerKind::Image,
            bounds: LayoutBox {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
            },
            opacity: 1.0,
            is_base: true,
        }));
        base_found = true;
        break;
    }
    if !request.base_assets.is_empty() && !base_found {
        diagnostics.push(RenderDiagnostic::MissingBaseAsset {
            roles: request
                .base_assets
                .iter()
                .map(|role| role.as_str().to_owned())
                .collect(),
        });
    }

    let mut visual_entities = Vec::new();
    VisualTraversal {
        snapshot,
        presentation: request.presentation,
        dependencies: &mut dependencies,
        output: &mut visual_entities,
    }
    .collect(request.page, 1.0)?;
    for (entity, opacity) in visual_entities {
        if let Some(raster) = snapshot.component::<RasterLayer>(entity)? {
            if request.include_images
                && let Some(asset) = snapshot.asset(entity, &request.image_asset)?
            {
                dependencies.insert(RenderDependency::Blob(asset.blob));
                layers.push(Layer::Image(ImageLayer {
                    entity,
                    asset,
                    name: Some(raster.name),
                    kind: match raster.kind {
                        RasterLayerKind::Cleanup => crate::VisualLayerKind::Cleanup,
                        RasterLayerKind::Paint => crate::VisualLayerKind::Paint,
                    },
                    bounds: LayoutBox {
                        x: 0.0,
                        y: 0.0,
                        width: width as f32,
                        height: height as f32,
                    },
                    opacity,
                    is_base: false,
                }));
            }
            continue;
        }
        if let Some(text_layout) = snapshot.component::<SceneTextLayout>(entity)? {
            if request
                .text_entities
                .as_ref()
                .is_some_and(|entities| !entities.contains(&entity))
            {
                continue;
            }
            let text_layer = snapshot.text_layer(entity)?;
            let content = text_layer.content()?.id();
            let presents = snapshot
                .relation_from::<koharu_scene::Presents>(entity)?
                .expect("validated text layers present content");
            dependencies.insert(RenderDependency::Relation(presents.id()));
            dependencies.insert(RenderDependency::Entity(content));

            let geometry = snapshot.component::<Geometry>(entity)?;
            let fit = crate::bubble::resolve(snapshot, entity, &page_entities)?;
            if let Some(fit) = fit.as_ref() {
                dependencies.insert(RenderDependency::Relation(fit.relation));
                dependencies.insert(RenderDependency::Entity(fit.region));
            }
            // Read before `fit` is consumed below; the geometry branch keeps
            // only a borrow of it and the other moves it.
            let fit_region = fit.as_ref().map(|fit| fit.region);
            let (text_frame, balloon_contour) = if let Some(geometry) = geometry.as_ref() {
                let Some(frame) = geometry_frame(geometry) else {
                    continue;
                };
                /* A READER-AUTHORED frame (`Origin::User` -- the box editor's
                 * placements and its resized/drawn boxes) is a hard
                 * boundary, so it gets a contour even when the layer fits no
                 * bubble: the contour is what routes layout through the
                 * balanced balloon search, which re-WRAPS the text for the
                 * box's shape -- the free arm only re-sizes, which reads as
                 * "zooming" -- and edge clearance then keeps
                 * every glyph outline inside the rectangle. Pipeline-authored
                 * frames (`Origin::Generated` -- scream bands, turned columns)
                 * keep the old behaviour exactly. */
                let contour = if fit
                    .as_ref()
                    .and_then(|fit| fit.balloon_contour.as_ref())
                    .is_some()
                    || matches!(geometry.origin, Origin::User)
                {
                    Some(crate::bubble::contour(geometry, frame))
                } else {
                    None
                };
                (frame, contour)
            } else {
                let Some(fit) = fit else {
                    continue;
                };
                (fit.frame, fit.balloon_contour)
            };

            let Some((text, language)) =
                resolve_text(snapshot, content, entity, request, &mut diagnostics)?
            else {
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }
            let typography = text_layer.typography()?;
            let mut ink_bounds = None;
            let analysis = if let Some(recognized) =
                snapshot.relation_from::<koharu_scene::RecognizedFrom>(content)?
            {
                dependencies.insert(RenderDependency::Relation(recognized.id()));
                dependencies.insert(RenderDependency::Entity(recognized.value().target));
                // The recognized-from region is the tight box the source text
                // was read out of -- the one position known to sit right in the
                // artist's own balloon, cut or whole.
                if let Some(ink) = snapshot.component::<Geometry>(recognized.value().target)? {
                    ink_bounds = geometry_bounds(&ink);
                }
                snapshot.component::<OcrAnalysis>(recognized.value().target)?
            } else {
                None
            };
            let writing_mode = resolve_writing_mode(
                &text,
                text_frame.bounds,
                typography.as_ref(),
                analysis.as_ref(),
            );
            let (direction, _) = shaping_direction_for_text(&text, writing_mode);
            let rtl = direction == harfrust::Direction::RightToLeft;
            let alignment =
                resolve_alignment(typography.as_ref().and_then(|value| value.alignment), rtl);
            let is_bubble_text = balloon_contour.is_some();
            let anchor_center_y = if request.theme.edge_anchored_lettering && is_bubble_text {
                edge_anchored_center_y(text_frame.bounds, height as f32, ink_bounds)
            } else {
                None
            };
            layers.push(Layer::Text(TextLayer {
                entity,
                text,
                language,
                bounds: text_frame.bounds,
                balloon_contour,
                opacity,
                preferred_font: typography
                    .as_ref()
                    .and_then(|value| value.preferred_font.clone()),
                font_weight: typography.as_ref().and_then(|value| value.font_weight),
                font_size: typography.as_ref().and_then(|value| value.size),
                auto_fit: typography.as_ref().is_none_or(|value| value.auto_fit),
                alignment,
                writing_mode,
                foreground_color: typography.as_ref().and_then(|value| value.color),
                fill_gradient_to: typography.as_ref().and_then(resolve_fill_gradient_to),
                strike_color: typography.as_ref().and_then(resolve_strike_color),
                stroke: resolve_stroke(typography.as_ref()),
                angle_degrees: text_frame.angle_degrees,
                point_text: !is_bubble_text && text_layout.kind == TextLayoutKind::Point,
                fit_region,
                size_ceiling: None,
                anchor_center_y,
            }));
            continue;
        }

        let Some(geometry) = snapshot.component::<Geometry>(entity)? else {
            continue;
        };
        let Some(bounds) = geometry_bounds(&geometry) else {
            continue;
        };
        if request.include_images
            && let Some(asset) = snapshot.asset(entity, &request.image_asset)?
        {
            dependencies.insert(RenderDependency::Blob(asset.blob));
            layers.push(Layer::Image(ImageLayer {
                entity,
                asset,
                name: None,
                kind: crate::VisualLayerKind::Image,
                bounds,
                opacity,
                is_base: false,
            }));
        }
    }

    Ok(Composition {
        revision: snapshot.revision(),
        page: request.page,
        width,
        height,
        layers,
        dependencies: dependencies.into_iter().collect(),
        diagnostics,
        request: request.clone(),
    })
}

struct VisualTraversal<'a> {
    snapshot: &'a Snapshot,
    presentation: LayerPresentation,
    dependencies: &'a mut BTreeSet<RenderDependency>,
    output: &'a mut Vec<(EntityId, f32)>,
}

impl VisualTraversal<'_> {
    fn collect(&mut self, parent: EntityId, inherited_opacity: f32) -> Result<()> {
        for entity in self.snapshot.children(parent)? {
            self.dependencies.insert(RenderDependency::Entity(entity));
            let visibility = self
                .snapshot
                .component::<Visibility>(entity)?
                .unwrap_or(Visibility {
                    origin: koharu_scene::Origin::User,
                    visible: true,
                    opacity: 1.0,
                });
            let (visible, opacity) = if self.presentation == LayerPresentation::Deferred {
                (true, 1.0)
            } else {
                (visibility.visible, inherited_opacity * visibility.opacity)
            };
            if self.snapshot.component::<Group>(entity)?.is_some() {
                if visible && opacity > 0.0 {
                    self.collect(entity, opacity)?;
                }
            } else if visible && opacity > 0.0 {
                self.output.push((entity, opacity));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Layer {
    Image(ImageLayer),
    Text(TextLayer),
}

#[derive(Clone, Debug)]
pub(crate) struct ImageLayer {
    pub entity: EntityId,
    pub asset: Asset,
    pub name: Option<String>,
    pub kind: crate::VisualLayerKind,
    pub bounds: LayoutBox,
    pub opacity: f32,
    pub is_base: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct TextLayer {
    pub entity: EntityId,
    pub text: String,
    pub language: Option<LanguageTag>,
    pub bounds: LayoutBox,
    pub balloon_contour: Option<Vec<(f32, f32)>>,
    pub opacity: f32,
    pub preferred_font: Option<String>,
    pub font_weight: Option<u16>,
    pub font_size: Option<f32>,
    pub auto_fit: bool,
    pub alignment: TextAlign,
    pub writing_mode: WritingMode,
    pub foreground_color: Option<[u8; 4]>,
    /// A second fill colour from the `FILL_GRADIENT_TO_EXTENSION` typography
    /// extension: the fill letters as a linear gradient from
    /// `foreground_color` at the layout's top to this at its bottom, tilting
    /// with `angle_degrees`. `None` -- the overwhelmingly common case -- is a
    /// solid fill, byte-identical to before the field existed.
    pub fill_gradient_to: Option<[u8; 4]>,
    /// A strike-through colour from the `STRIKE_COLOR_EXTENSION` typography
    /// extension: one stroke along the reading direction through the middle of
    /// the laid-out text, drawn after the glyphs and inside this layer, so it
    /// lands on its own text whatever `angle_degrees` does. `None` -- almost
    /// always -- draws nothing at all.
    pub strike_color: Option<[u8; 4]>,
    pub stroke: Option<StrokeOptions>,
    pub angle_degrees: f32,
    pub point_text: bool,
    /// The region this layer was fitted to, and so the identity of the thing
    /// speaking.
    ///
    /// Two text layers sharing one bubble are two halves of one utterance --
    /// detection gives each its own cell of the balloon, and each cell then
    /// solves its size independently. `FitsTo` already
    /// carries exactly this grouping: a bubble for dialogue, the text's own
    /// region for free-standing text, which makes every caption its own group
    /// and the rule a no-op there.
    pub fit_region: Option<EntityId>,
    /// An upper bound on the auto-fit size, imposed after the page has been
    /// laid out once. `None` on the first pass.
    pub size_ceiling: Option<f32>,
    /// Where the block's vertical center should land, in canvas coordinates,
    /// for a balloon whose mask runs to the canvas edge -- the balloon is cut
    /// by the slice, so centering in its visible part drifts the text away
    /// from the cut. `None` everywhere the theme flag is off or the balloon is
    /// whole, which keeps every other layer byte-identical.
    pub anchor_center_y: Option<f32>,
}

/// The anchored vertical center for a balloon the slice boundary cut, or
/// `None` for every balloon it did not.
///
/// A cut balloon is recognized by its fit frame touching the canvas top or
/// bottom: the segmentation mask is exactly canvas-sized, so a contour that
/// reaches the edge means the drawn balloon continues past it. The anchor is
/// the source ink's own vertical center -- the artist centered that text in
/// the WHOLE balloon, so it is the one position that still reads as centered
/// on the joined page. Clamped into the fit frame so a stray detection cannot
/// anchor outside the balloon.
///
/// A displacement a reader could not see is not worth a re-solve: anchoring
/// repositions the block inside the contour, so the auto-fit can settle on a
/// different wrap, and on one test slice a 15.8 px anchor turned a clean
/// two-line balloon into a hyphenated three-line one while fixing nothing
/// visible. Two independent reviews had called that case sub-visible (14 px
/// measured against 40-158 px on the cut balloons that needed the anchor),
/// so displacements under `max(16 px, 6%)` of the fit height keep today's
/// centering.
pub(crate) fn edge_anchored_center_y(
    fit: LayoutBox,
    canvas_height: f32,
    ink: Option<LayoutBox>,
) -> Option<f32> {
    const EDGE_EPSILON_PX: f32 = 2.0;
    const DISPLACEMENT_FLOOR_PX: f32 = 16.0;
    const DISPLACEMENT_FLOOR_SHARE: f32 = 0.06;
    let ink = ink?;
    let touches_top = fit.y <= EDGE_EPSILON_PX;
    let touches_bottom = fit.y + fit.height >= canvas_height - EDGE_EPSILON_PX;
    if !touches_top && !touches_bottom {
        return None;
    }
    let center = (ink.y + ink.height * 0.5).clamp(fit.y, fit.y + fit.height);
    let fit_center = fit.y + fit.height * 0.5;
    let floor = DISPLACEMENT_FLOOR_PX.max(fit.height * DISPLACEMENT_FLOOR_SHARE);
    if (center - fit_center).abs() < floor {
        return None;
    }
    Some(center)
}

fn validate_request(request: &RenderRequest) -> Result<()> {
    let theme = &request.theme;
    let valid = theme.font_size.is_finite()
        && theme.font_size > 0.0
        && theme.minimum_font_size.is_finite()
        && theme.minimum_font_size > 0.0
        && theme.minimum_font_size <= theme.font_size
        && theme.line_height.is_finite()
        && theme.line_height > 0.0
        && theme.letter_spacing.is_finite()
        && theme.word_spacing.is_finite()
        && theme
            .text_inset
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && theme
            .text_stroke
            .is_none_or(|stroke| stroke.width_px.is_finite() && stroke.width_px >= 0.0);
    if valid {
        Ok(())
    } else {
        Err(Error::invalid("render theme contains invalid dimensions"))
    }
}

fn surface_size(page: &Page) -> Result<(u32, u32)> {
    let width = page.width.ceil() as u32;
    let height = page.height.ceil() as u32;
    if width == 0
        || height == 0
        || width > MAX_SURFACE_DIMENSION
        || height > MAX_SURFACE_DIMENSION
        || u64::from(width) * u64::from(height) > MAX_SURFACE_PIXELS
    {
        return Err(Error::invalid(format!(
            "page surface {width}x{height} exceeds renderer limits"
        )));
    }
    Ok((width, height))
}

fn resolve_text(
    snapshot: &Snapshot,
    content: EntityId,
    layer: EntityId,
    request: &RenderRequest,
    diagnostics: &mut Vec<RenderDiagnostic>,
) -> Result<Option<(String, Option<LanguageTag>)>> {
    if let Some(translation) = snapshot.component::<Translation>(content)? {
        return Ok(Some((translation.text.value, translation.language)));
    }
    if !request.fallback_to_source_text {
        return Ok(None);
    }
    if let Some(source) = snapshot.component::<SourceText>(content)? {
        diagnostics.push(RenderDiagnostic::UsedSourceText { entity: layer });
        return Ok(Some((source.text.value, source.language)));
    }
    Ok(None)
}

fn resolve_writing_mode(
    text: &str,
    bounds: LayoutBox,
    typography: Option<&Typography>,
    analysis: Option<&OcrAnalysis>,
) -> WritingMode {
    if !is_cjk_text(text) {
        return WritingMode::Horizontal;
    }
    if let Some(mode) = typography.and_then(|value| value.writing_mode) {
        return match mode {
            koharu_scene::WritingMode::Horizontal => WritingMode::Horizontal,
            koharu_scene::WritingMode::Vertical => WritingMode::VerticalRl,
        };
    }
    match analysis.map(|value| value.direction) {
        Some(TextDirection::Vertical) => WritingMode::VerticalRl,
        Some(TextDirection::Horizontal) => WritingMode::Horizontal,
        Some(TextDirection::Auto) | None if bounds.height > bounds.width => WritingMode::VerticalRl,
        Some(TextDirection::Auto) | None => WritingMode::Horizontal,
    }
}

/// Parse the `FILL_GRADIENT_TO_EXTENSION` value, `"r,g,b"` or `"r,g,b,a"`.
/// A malformed value is a solid fill, never an error -- the extension is
/// styling, and a page must not fail to render over a colour string.
fn resolve_fill_gradient_to(typography: &Typography) -> Option<[u8; 4]> {
    let value = typography.extensions.get(FILL_GRADIENT_TO_EXTENSION)?;
    let mut parts = value.split(',').map(|part| part.trim().parse::<u8>());
    let mut color = [0_u8, 0, 0, u8::MAX];
    for slot in color.iter_mut().take(3) {
        *slot = parts.next()?.ok()?;
    }
    match parts.next() {
        None => Some(color),
        Some(Ok(alpha)) if parts.next().is_none() => {
            color[3] = alpha;
            Some(color)
        }
        _ => None,
    }
}

/// Parse the `STRIKE_COLOR_EXTENSION` value, `"r,g,b"` or `"r,g,b,a"`.
/// Malformed is no strike, never an error -- same contract as the gradient
/// above, and for the same reason: a page must not fail to render over a
/// colour string.
fn resolve_strike_color(typography: &Typography) -> Option<[u8; 4]> {
    let value = typography.extensions.get(STRIKE_COLOR_EXTENSION)?;
    let mut parts = value.split(',').map(|part| part.trim().parse::<u8>());
    let mut color = [0_u8, 0, 0, u8::MAX];
    for slot in color.iter_mut().take(3) {
        *slot = parts.next()?.ok()?;
    }
    match parts.next() {
        None => Some(color),
        Some(Ok(alpha)) if parts.next().is_none() => {
            color[3] = alpha;
            Some(color)
        }
        _ => None,
    }
}

fn resolve_stroke(typography: Option<&Typography>) -> Option<StrokeOptions> {
    let typography = typography?;
    let width_px = typography.stroke_width.filter(|width| *width > 0.0)?;
    Some(StrokeOptions {
        color: typography.stroke_color.unwrap_or([u8::MAX; 4]),
        width_px,
    })
}

fn resolve_alignment(alignment: Option<TextAlignment>, rtl: bool) -> TextAlign {
    match alignment.unwrap_or(TextAlignment::Center) {
        TextAlignment::Start if rtl => TextAlign::Right,
        TextAlignment::Start => TextAlign::Left,
        TextAlignment::Center => TextAlign::Center,
        TextAlignment::End if rtl => TextAlign::Left,
        TextAlignment::End => TextAlign::Right,
        TextAlignment::Justify => TextAlign::Justify,
    }
}

#[cfg(test)]
mod tests {
    use koharu_scene::{
        AssetInput, AssetMetadata, AssetRole, At, Authored, BubbleRegion, FitsTo, Geometry, Origin,
        PageDraft, Point, RasterLayer, RasterLayerKind, RecognizedFrom, Session, TextAlignment,
        TextLayout, TextLayoutKind, TextRegion, Translation, Typography, Visibility,
    };

    use super::*;

    /// The fill-gradient extension: well-formed values parse, and every
    /// malformed value is a SOLID fill rather than an error -- a page must
    /// never fail to render over a colour string.
    #[test]
    fn fill_gradient_extension_parses_or_falls_back_to_solid() {
        let typography = |value: Option<&str>| Typography {
            origin: Origin::User,
            preferred_font: None,
            font_weight: None,
            size: None,
            auto_fit: true,
            color: Some([78, 32, 31, 255]),
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: None,
            extensions: value
                .map(|value| {
                    [(
                        koharu_scene::FILL_GRADIENT_TO_EXTENSION.to_owned(),
                        value.to_owned(),
                    )]
                    .into_iter()
                    .collect()
                })
                .unwrap_or_default(),
        };
        assert_eq!(
            resolve_fill_gradient_to(&typography(Some("43,42,43,255"))),
            Some([43, 42, 43, 255])
        );
        assert_eq!(
            resolve_fill_gradient_to(&typography(Some("43, 42, 43"))),
            Some([43, 42, 43, 255]),
            "three channels default the alpha"
        );
        assert_eq!(resolve_fill_gradient_to(&typography(None)), None);
        assert_eq!(resolve_fill_gradient_to(&typography(Some(""))), None);
        assert_eq!(resolve_fill_gradient_to(&typography(Some("1,2"))), None);
        assert_eq!(resolve_fill_gradient_to(&typography(Some("1,2,3,4,5"))), None);
        assert_eq!(resolve_fill_gradient_to(&typography(Some("256,0,0"))), None);
        assert_eq!(resolve_fill_gradient_to(&typography(Some("red,0,0"))), None);
    }

    struct Fixture {
        snapshot: Snapshot,
        page: EntityId,
        text: EntityId,
        bubble: EntityId,
        relation: RelationId,
    }

    fn fixture(include_translation: bool) -> Fixture {
        fixture_with_visibility(include_translation, None, Some(18.0), false)
    }

    fn rotated_geometry(x: f64, y: f64, width: f64, height: f64, degrees: f64) -> Geometry {
        let center_x = x + width * 0.5;
        let center_y = y + height * 0.5;
        let (sin, cos) = degrees.to_radians().sin_cos();
        Geometry {
            origin: Origin::User,
            points: [
                (-width * 0.5, -height * 0.5),
                (width * 0.5, -height * 0.5),
                (width * 0.5, height * 0.5),
                (-width * 0.5, height * 0.5),
            ]
            .map(|(x, y)| Point {
                x: center_x + x * cos - y * sin,
                y: center_y + x * sin + y * cos,
            })
            .into(),
        }
    }

    fn fixture_with_visibility(
        include_translation: bool,
        visibility: Option<Visibility>,
        font_size: Option<f32>,
        manual_frame: bool,
    ) -> Fixture {
        let mut session = Session::memory().unwrap();
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 200.0, 120.0), At::End)?;
                let bubble = edit.add_analysis_region::<BubbleRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(20.0, 30.0, 100.0, 50.0),
                    None,
                )?;
                let source_region = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &rotated_geometry(30.0, 35.0, 80.0, 40.0, 12.5),
                    None,
                )?;
                let content = edit.add_text_content(page, At::End)?;
                edit.set(
                    content,
                    &SourceText {
                        text: Authored::user("原文".to_owned()),
                        language: Some(LanguageTag::new("ja")?),
                    },
                )?;
                if include_translation {
                    edit.set(
                        content,
                        &Translation {
                            text: Authored::user("مرحبا".to_owned()),
                            language: Some(LanguageTag::new("ar")?),
                        },
                    )?;
                }
                let text = edit.add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )?;
                if manual_frame {
                    edit.set(text, &rotated_geometry(30.0, 35.0, 80.0, 40.0, 12.5))?;
                }
                edit.set(
                    text,
                    &Typography {
                        origin: Origin::User,
                        preferred_font: None,
                        font_weight: Some(600),
                        size: font_size.filter(|size| *size > 0.0),
                        auto_fit: font_size.is_none_or(|size| size <= 0.0),
                        color: Some([0x12, 0x34, 0x56, 0xff]),
                        stroke_color: Some([0xff, 0xff, 0xff, 0xff]),
                        stroke_width: Some(1.5),
                        alignment: Some(TextAlignment::Start),
                        writing_mode: None,
                        extensions: Default::default(),
                    },
                )?;
                if let Some(visibility) = &visibility {
                    edit.set(text, visibility)?;
                }
                let relation = edit.relate::<FitsTo>(text, bubble)?;
                edit.relate::<RecognizedFrom>(content, source_region)?;
                ids = Some((page, text, bubble, relation));
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).unwrap().snapshot;
        let (page, text, bubble, relation) = ids.unwrap();
        Fixture {
            snapshot,
            page,
            text,
            bubble,
            relation,
        }
    }

    /// A fixture whose bubble runs to the canvas bottom edge, the way a
    /// slice-cut balloon's mask contour does, with the ink well inside it.
    /// Page 200x120; bubble y 30..119 (119 >= 120 - 2, touching); ink y 40..60,
    /// center 50.
    fn edge_fixture(bubble_touches_edge: bool) -> Fixture {
        let mut session = Session::memory().unwrap();
        let mut ids = None;
        let bubble_height = if bubble_touches_edge { 89.0 } else { 60.0 };
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 200.0, 120.0), At::End)?;
                let bubble = edit.add_analysis_region::<BubbleRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(20.0, 30.0, 100.0, bubble_height),
                    None,
                )?;
                let source_region = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(30.0, 40.0, 80.0, 20.0),
                    None,
                )?;
                let content = edit.add_text_content(page, At::End)?;
                edit.set(
                    content,
                    &SourceText {
                        text: Authored::user("原文".to_owned()),
                        language: Some(LanguageTag::new("ja")?),
                    },
                )?;
                edit.set(
                    content,
                    &Translation {
                        text: Authored::user("translated".to_owned()),
                        language: Some(LanguageTag::new("en")?),
                    },
                )?;
                let text = edit.add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )?;
                let relation = edit.relate::<FitsTo>(text, bubble)?;
                edit.relate::<RecognizedFrom>(content, source_region)?;
                ids = Some((page, text, bubble, relation));
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).unwrap().snapshot;
        let (page, text, bubble, relation) = ids.unwrap();
        Fixture {
            snapshot,
            page,
            text,
            bubble,
            relation,
        }
    }

    fn compiled_anchor(fixture: &Fixture, edge_anchored_lettering: bool) -> Option<f32> {
        let mut request = RenderRequest::transparent(fixture.page);
        request.theme.edge_anchored_lettering = edge_anchored_lettering;
        let resolved = Compositor::new()
            .compile(&fixture.snapshot, &request)
            .unwrap();
        let Some(Layer::Text(text)) = resolved.layers.first() else {
            panic!("expected a text layer");
        };
        assert_eq!(text.entity, fixture.text);
        text.anchor_center_y
    }

    /// The composed path the renderer actually runs: flag on + bubble at the
    /// canvas edge + a recognized-from ink box -> the layer carries the ink's
    /// vertical center as its anchor.
    #[test]
    fn a_cut_bubble_anchors_its_layer_at_the_ink_center() {
        let anchor = compiled_anchor(&edge_fixture(true), true);
        assert_eq!(anchor, Some(50.0), "ink y 40..60 anchors at 50");
    }

    #[test]
    fn the_anchor_is_off_by_default_even_for_a_cut_bubble() {
        assert_eq!(compiled_anchor(&edge_fixture(true), false), None);
    }

    #[test]
    fn a_whole_bubble_never_anchors_even_with_the_flag_on() {
        assert_eq!(compiled_anchor(&edge_fixture(false), true), None);
    }

    #[test]
    fn edge_anchor_requires_the_fit_to_touch_a_canvas_edge() {
        let interior = LayoutBox {
            x: 20.0,
            y: 30.0,
            width: 100.0,
            height: 60.0,
        };
        let ink = LayoutBox {
            x: 30.0,
            y: 40.0,
            width: 80.0,
            height: 20.0,
        };
        assert_eq!(edge_anchored_center_y(interior, 120.0, Some(ink)), None);
        let touching_bottom = LayoutBox {
            height: 89.0,
            ..interior
        };
        assert_eq!(
            edge_anchored_center_y(touching_bottom, 120.0, Some(ink)),
            Some(50.0)
        );
        let touching_top = LayoutBox { y: 0.0, ..interior };
        assert_eq!(
            edge_anchored_center_y(touching_top, 120.0, Some(ink)),
            Some(50.0)
        );
        assert_eq!(
            edge_anchored_center_y(touching_bottom, 120.0, None),
            None,
            "no ink, no anchor"
        );
    }

    /// A trap seen on a real slice: the detection bbox is not clamped to the
    /// canvas, so an ink center can sit below the fit frame. The anchor must be clamped
    /// into the fit, never trusted raw.
    #[test]
    fn an_out_of_range_ink_center_is_clamped_into_the_fit() {
        let fit = LayoutBox {
            x: 385.0,
            y: 615.0,
            width: 634.0,
            height: 292.0,
        };
        let overflowing_ink = LayoutBox {
            x: 400.0,
            y: 900.0,
            width: 100.0,
            height: 80.0,
        };
        assert_eq!(
            edge_anchored_center_y(fit, 908.0, Some(overflowing_ink)),
            Some(907.0),
            "ink center 940 clamps to the fit bottom 907"
        );
    }

    /// A measured slice, verbatim: a 15.8 px displacement on a 292 px fit is
    /// under the 6% floor, and anchoring it anyway turned a clean two-line
    /// balloon into a hyphenated three-line one on the first render. A
    /// sub-visible displacement must keep today's centering.
    #[test]
    fn a_sub_visible_displacement_never_anchors() {
        let fit = LayoutBox {
            x: 385.0,
            y: 615.0,
            width: 634.0,
            height: 292.0,
        };
        let ink = LayoutBox {
            x: 400.0,
            y: 622.4,
            width: 500.0,
            height: 308.8,
        };
        assert_eq!(
            edge_anchored_center_y(fit, 908.0, Some(ink)),
            None,
            "displacement 15.8 px sits under the 17.5 px floor for this fit"
        );

        let small_fit = LayoutBox {
            x: 0.0,
            y: 0.0,
            width: 200.0,
            height: 100.0,
        };
        let near_center_ink = LayoutBox {
            x: 10.0,
            y: 42.0,
            width: 100.0,
            height: 40.0,
        };
        assert_eq!(
            edge_anchored_center_y(small_fit, 400.0, Some(near_center_ink)),
            None,
            "12 px on a small balloon is under the absolute 16 px floor"
        );
        let displaced_ink = LayoutBox {
            y: 50.0,
            ..near_center_ink
        };
        assert_eq!(
            edge_anchored_center_y(small_fit, 400.0, Some(displaced_ink)),
            Some(70.0),
            "20 px clears the absolute floor and anchors"
        );
    }

    #[test]
    fn deferred_presentation_retains_hidden_text_at_full_opacity() {
        let fixture = fixture_with_visibility(
            true,
            Some(Visibility {
                origin: Origin::User,
                visible: false,
                opacity: 0.25,
            }),
            Some(18.0),
            false,
        );
        let compositor = Compositor::new();

        let resolved = compositor
            .compile(&fixture.snapshot, &RenderRequest::transparent(fixture.page))
            .unwrap();
        assert!(resolved.layers.is_empty());

        let mut deferred = RenderRequest::transparent(fixture.page);
        deferred.presentation = LayerPresentation::Deferred;
        let deferred = compositor.compile(&fixture.snapshot, &deferred).unwrap();
        let Layer::Text(text) = &deferred.layers[0] else {
            panic!("expected retained text layer");
        };
        assert_eq!(text.entity, fixture.text);
        assert_eq!(text.opacity, 1.0);
    }

    #[test]
    fn text_group_visibility_is_inherited_by_its_layers() {
        let fixture = fixture(true);
        let group = fixture
            .snapshot
            .page(fixture.page)
            .unwrap()
            .text_group()
            .unwrap()
            .unwrap()
            .id();
        let patch = fixture
            .snapshot
            .patch(|edit| {
                edit.set(
                    group,
                    &Visibility {
                        origin: Origin::User,
                        visible: true,
                        opacity: 0.4,
                    },
                )
            })
            .unwrap();
        let snapshot = fixture.snapshot.preview([&patch]).unwrap();
        let composition = Compositor::new()
            .compile(&snapshot, &RenderRequest::transparent(fixture.page))
            .unwrap();
        let Layer::Text(text) = &composition.layers[0] else {
            panic!("expected a text layer");
        };
        assert_eq!(text.opacity, 0.4);
    }

    #[test]
    fn compiles_translation_and_explicit_bubble_relation() {
        let fixture = fixture(true);
        let request = RenderRequest::transparent(fixture.page);

        let composition = Compositor::new()
            .compile(&fixture.snapshot, &request)
            .unwrap();
        let Layer::Text(text) = &composition.layers[0] else {
            panic!("expected a text layer");
        };

        assert_eq!(text.entity, fixture.text);
        assert_eq!(text.text, "مرحبا");
        assert_eq!(text.language.as_ref().unwrap().as_str(), "ar");
        assert_eq!(text.alignment, TextAlign::Right);
        assert_eq!(text.writing_mode, WritingMode::Horizontal);
        assert_eq!(text.font_size, Some(18.0));
        assert_eq!(text.font_weight, Some(600));
        assert!(text.balloon_contour.is_some());
        assert_eq!(text.foreground_color, Some([0x12, 0x34, 0x56, 0xff]));
        assert_eq!(
            text.stroke,
            Some(StrokeOptions {
                color: [0xff; 4],
                width_px: 1.5,
            })
        );
        assert_eq!(text.angle_degrees, 0.0);
        assert!((text.bounds.x - 20.0).abs() < 1e-5);
        assert!((text.bounds.y - 30.0).abs() < 1e-5);
        assert!((text.bounds.width - 100.0).abs() < 1e-5);
        assert!((text.bounds.height - 50.0).abs() < 1e-5);
        assert_eq!(text.balloon_contour.as_ref().unwrap().len(), 4);
        assert!(
            composition
                .dependencies
                .contains(&RenderDependency::Relation(fixture.relation))
        );
        assert!(
            composition
                .dependencies
                .contains(&RenderDependency::Entity(fixture.bubble))
        );
    }

    #[test]
    fn automatic_font_size_is_an_explicit_layout_mode() {
        for size in [None, Some(0.0)] {
            let fixture = fixture_with_visibility(true, None, size, false);
            let composition = Compositor::new()
                .compile(&fixture.snapshot, &RenderRequest::transparent(fixture.page))
                .unwrap();
            let Layer::Text(text) = &composition.layers[0] else {
                panic!("expected a text layer");
            };
            assert_eq!(text.font_size, None);
            assert!(text.auto_fit);
        }
    }

    #[test]
    fn compiles_raster_paint_as_its_own_visual_layer() {
        let mut session = Session::memory().unwrap();
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 100.0, 100.0), At::End)?;
                let drawing = edit.add_entity(page, At::End)?;
                edit.set(
                    drawing,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Drawing 1".to_owned(),
                        kind: RasterLayerKind::Paint,
                    },
                )?;
                edit.set_asset(
                    drawing,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        std::sync::Arc::<[u8]>::from(&b"png"[..]),
                        "image/png",
                        AssetMetadata {
                            width: Some(100),
                            height: Some(100),
                            attributes: Default::default(),
                        },
                    ),
                )?;
                ids = Some((page, drawing));
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).unwrap().snapshot;
        let (page, drawing) = ids.unwrap();

        let composition = Compositor::new()
            .compile(&snapshot, &RenderRequest::transparent(page))
            .unwrap();
        let Layer::Image(layer) = &composition.layers[0] else {
            panic!("expected paint layer");
        };
        assert_eq!(layer.entity, drawing);
        assert_eq!(layer.name.as_deref(), Some("Drawing 1"));
        assert_eq!(layer.kind, crate::VisualLayerKind::Paint);
        assert_eq!(layer.bounds.width, 100.0);
        assert_eq!(layer.bounds.height, 100.0);
    }

    #[test]
    fn records_when_a_missing_translation_falls_back_to_source() {
        let fixture = fixture(false);
        let request = RenderRequest::transparent(fixture.page);

        let composition = Compositor::new()
            .compile(&fixture.snapshot, &request)
            .unwrap();
        let Layer::Text(text) = &composition.layers[0] else {
            panic!("expected a text layer");
        };

        assert_eq!(text.text, "原文");
        assert_eq!(text.language.as_ref().unwrap().as_str(), "ja");
        assert_eq!(
            composition.diagnostics,
            vec![RenderDiagnostic::UsedSourceText {
                entity: fixture.text,
            }]
        );
    }

    #[test]
    fn missing_translation_can_be_skipped_without_losing_other_capabilities() {
        let fixture = fixture(false);
        let mut request = RenderRequest::transparent(fixture.page);
        request.fallback_to_source_text = false;

        let composition = Compositor::new()
            .compile(&fixture.snapshot, &request)
            .unwrap();

        assert!(composition.layers.is_empty());
        assert!(composition.diagnostics.is_empty());
    }

    #[test]
    fn writing_mode_is_only_applied_to_cjk_text() {
        let typography = Typography {
            origin: Origin::User,
            preferred_font: None,
            font_weight: None,
            size: None,
            auto_fit: true,
            color: None,
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: Some(koharu_scene::WritingMode::Vertical),
            extensions: Default::default(),
        };
        let bounds = LayoutBox {
            x: 0.0,
            y: 0.0,
            width: 20.0,
            height: 80.0,
        };

        assert_eq!(
            resolve_writing_mode("Latin", bounds, Some(&typography), None),
            WritingMode::Horizontal
        );
        assert_eq!(
            resolve_writing_mode("日本語", bounds, Some(&typography), None),
            WritingMode::VerticalRl
        );
        assert_eq!(
            resolve_writing_mode("한국어", bounds, Some(&typography), None),
            WritingMode::VerticalRl
        );
    }

    #[test]
    fn authored_text_geometry_preserves_bubble_shaped_layout() {
        let fixture = fixture_with_visibility(true, None, Some(18.0), true);
        let composition = Compositor::new()
            .compile(&fixture.snapshot, &RenderRequest::transparent(fixture.page))
            .unwrap();
        let Layer::Text(text) = &composition.layers[0] else {
            panic!("expected a text layer");
        };

        assert_eq!(text.font_size, Some(18.0));
        assert!(text.balloon_contour.is_some());
        assert_eq!(text.bounds.width, 80.0);
        assert_eq!(text.angle_degrees, 12.5);
    }

    /// A READER-authored frame is a hard boundary, so a free-standing layer
    /// carrying one gets a rectangle contour -- and with it the balanced
    /// balloon search, which re-WRAPS for the box's shape. The free arm only
    /// re-sizes, so resizing a translated box would look like zooming.
    #[test]
    fn a_reader_authored_frame_gets_a_contour_without_any_bubble() {
        let mut session = Session::memory().unwrap();
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 200.0, 120.0), At::End)?;
                let source_region = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(30.0, 35.0, 80.0, 40.0),
                    None,
                )?;
                let content = edit.add_text_content(page, At::End)?;
                edit.set(
                    content,
                    &SourceText {
                        text: Authored::user("原文".to_owned()),
                        language: Some(LanguageTag::new("ja")?),
                    },
                )?;
                edit.set(
                    content,
                    &Translation {
                        text: Authored::user("MOVED".to_owned()),
                        language: Some(LanguageTag::new("en")?),
                    },
                )?;
                let text = edit.add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )?;
                edit.set(text, &Geometry::rectangle(10.0, 10.0, 120.0, 60.0))?;
                edit.relate::<FitsTo>(text, source_region)?;
                edit.relate::<RecognizedFrom>(content, source_region)?;
                ids = Some((page, text));
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).unwrap().snapshot;
        let (page, text_id) = ids.unwrap();
        let composition = Compositor::new()
            .compile(&snapshot, &RenderRequest::transparent(page))
            .unwrap();
        let Layer::Text(text) = &composition.layers[0] else {
            panic!("expected a text layer");
        };
        assert_eq!(text.entity, text_id);
        assert!(
            text.balloon_contour.is_some(),
            "the reader's rectangle is a hard boundary, contour and all"
        );
        assert_eq!(text.balloon_contour.as_ref().unwrap().len(), 4);
        assert_eq!(
            text.angle_degrees, 0.0,
            "an axis-aligned frame never turns the lettering"
        );
        assert_eq!(text.bounds.width, 120.0);
    }
}
