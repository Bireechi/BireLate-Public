//! Rendering of compiled compositions into reusable vector frames.

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use koharu_scene::{
    Asset, BlobId, Change, ComponentOwner, EntityChange, EntityId, LanguageTag, RelationChange,
    Revision, Snapshot,
};
use parking_lot::Mutex;
use rayon::prelude::*;
use vello::{
    Scene,
    kurbo::{Affine, Rect, Vec2},
    peniko::{Blob, Fill, ImageAlphaType, ImageData, ImageFormat, Mix},
};

use crate::{
    Composition, Error, FontFamilyInfo, RenderDependency, RenderDiagnostic, RenderRequest,
    RenderTheme, Result, TextLayout, TextRenderOptions, WritingMode,
    compositor::{ImageLayer, Layer, RenderBounds},
    fonts::Fonts,
};

const DEFAULT_CACHED_FRAMES: usize = 8;
const DEFAULT_IMAGE_CACHE_BYTES: usize = 512 * 1024 * 1024;

/// Resolves composition layers into retained Vello frames.
pub struct SceneRenderer {
    fonts: Arc<Fonts>,
    images: Mutex<DecodedImageCache>,
    text_renderer: crate::TextRenderer,
    frames: Mutex<FrameCache>,
}

impl SceneRenderer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            fonts: Fonts::shared(),
            images: Mutex::new(DecodedImageCache::new(DEFAULT_IMAGE_CACHE_BYTES)),
            text_renderer: crate::TextRenderer::new(),
            frames: Mutex::new(FrameCache::new(DEFAULT_CACHED_FRAMES)),
        }
    }

    pub async fn available_fonts() -> Result<Vec<FontFamilyInfo>> {
        Fonts::shared()
            .families()
            .await
            .map_err(Error::FontResource)
    }

    pub async fn font_preview(post_script_name: &str) -> Result<FontPreview> {
        const FONT_SIZE: f32 = 24.0;
        const PADDING: f32 = 6.0;

        let fonts = Fonts::shared();
        let font = fonts
            .by_post_script_name(post_script_name)
            .await
            .map_err(Error::FontResource)?;
        let label = font.family_name().to_owned();
        let preview_fonts = if font.covers(&label) {
            vec![font]
        } else {
            fonts
                .resolve(Some("Arial"), Some(400), &[], &label, None)
                .map_err(Error::FontResource)?
        };
        let layout = TextLayout::new(&preview_fonts[0])
            .with_fallback_fonts(&preview_fonts[1..])
            .with_font_size(FONT_SIZE)
            .run(&label)
            .map_err(Error::FontResource)?;
        let width = (layout.width + PADDING * 2.0).ceil().max(1.0) as u32;
        let height = (layout.height + PADDING * 2.0).ceil().max(1.0) as u32;
        let mut scene = Scene::new();
        crate::TextRenderer::new().render(
            &mut scene,
            &layout,
            WritingMode::Horizontal,
            &TextRenderOptions::default(),
            Affine::translate((f64::from(PADDING), f64::from(PADDING))),
        );
        Ok(FontPreview {
            scene,
            width,
            height,
        })
    }

    #[must_use]
    pub const fn text_renderer(&self) -> &crate::TextRenderer {
        &self.text_renderer
    }

    pub fn render(&self, snapshot: &Snapshot, composition: &Composition) -> Result<Arc<Frame>> {
        if composition.revision() != snapshot.revision() {
            return Err(Error::invalid(format!(
                "composition revision {} does not match scene revision {}",
                composition.revision(),
                snapshot.revision()
            )));
        }
        let key = FrameKey::new(
            composition.revision(),
            self.fonts.generation(),
            &composition.request,
        );
        if let Some(frame) = self.frames.lock().get(&key) {
            return Ok(frame);
        }
        let frame = Arc::new(Frame::render(
            composition,
            snapshot,
            self,
            &composition.request.theme,
            &self.text_renderer,
        )?);
        Ok(self.frames.lock().insert(key, frame))
    }

    pub fn clear_cache(&self) {
        self.frames.lock().clear();
        self.images.lock().clear();
    }

    /// Advances unaffected cached frames to the new revision and removes stale ones.
    pub fn apply_changes(&self, changes: &Change) {
        self.frames.lock().apply_changes(changes);
    }
}

impl Default for SceneRenderer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FontPreview {
    scene: Scene,
    width: u32,
    height: u32,
}

impl FontPreview {
    #[must_use]
    pub const fn scene(&self) -> &Scene {
        &self.scene
    }

    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum VisualLayerKind {
    Image,
    Cleanup,
    Paint,
    Text,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisualLayer {
    pub entity: EntityId,
    pub kind: VisualLayerKind,
    pub name: Option<String>,
    pub bounds: RenderBounds,
    pub font_size: Option<f32>,
    pub text: Option<VisualText>,
}

/// Text presentation resolved by the renderer for downstream editable exports.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualText {
    pub text: String,
    pub language: Option<LanguageTag>,
    /// The resolved text run before rotation, in page coordinates.
    pub rendered_bounds: RenderBounds,
    pub layout_bounds: RenderBounds,
    pub post_script_fonts: Vec<String>,
    pub font_size: f32,
    pub color: [u8; 4],
    pub alignment: crate::TextAlign,
    pub writing_mode: WritingMode,
    pub angle_degrees: f32,
}

pub struct Frame {
    revision: Revision,
    page: EntityId,
    width: u32,
    height: u32,
    left: i32,
    top: i32,
    scene: Arc<Scene>,
    entity_scenes: HashMap<EntityId, Vec<Arc<Scene>>>,
    layers: Vec<VisualLayer>,
    dependencies: Vec<RenderDependency>,
    diagnostics: Vec<RenderDiagnostic>,
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Frame")
            .field("revision", &self.revision)
            .field("page", &self.page)
            .field("size", &(self.width, self.height))
            .field("origin", &(self.left, self.top))
            .field("layers", &self.layers)
            .field("dependencies", &self.dependencies)
            .field("diagnostics", &self.diagnostics)
            .finish_non_exhaustive()
    }
}

impl Frame {
    fn render(
        composition: &Composition,
        snapshot: &Snapshot,
        renderer: &SceneRenderer,
        theme: &RenderTheme,
        text_renderer: &crate::TextRenderer,
    ) -> Result<Self> {
        let mut rendered = composition
            .layers
            .par_iter()
            .map(|layer| render_layer(layer, composition, snapshot, renderer, theme, text_renderer))
            .collect::<Result<Vec<_>>>()?;
        reconcile_text_sizes(composition, renderer, theme, text_renderer, &mut rendered)?;
        reconcile_text_collisions(composition, renderer, theme, text_renderer, &mut rendered)?;
        let mut scene = Scene::new();
        let mut entity_scenes = HashMap::<EntityId, Vec<Arc<Scene>>>::new();
        let mut layers = Vec::with_capacity(rendered.len());
        let mut diagnostics = composition.diagnostics.clone();
        for layer in rendered {
            let entity = layer.layer.entity;
            let layer_scene = Arc::new(layer.scene);
            scene.append(&layer_scene, None);
            entity_scenes.entry(entity).or_default().push(layer_scene);
            layers.push(layer.layer);
            diagnostics.extend(layer.diagnostics);
        }
        Ok(Self {
            revision: composition.revision,
            page: composition.page,
            width: composition.width,
            height: composition.height,
            left: 0,
            top: 0,
            scene: Arc::new(scene),
            entity_scenes,
            layers,
            dependencies: composition.dependencies.clone(),
            diagnostics,
        })
    }

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
    pub const fn origin(&self) -> (i32, i32) {
        (self.left, self.top)
    }

    #[must_use]
    pub fn layers(&self) -> &[VisualLayer] {
        &self.layers
    }

    #[must_use]
    pub fn dependencies(&self) -> &[RenderDependency] {
        &self.dependencies
    }

    #[must_use]
    pub fn diagnostics(&self) -> &[RenderDiagnostic] {
        &self.diagnostics
    }

    /// Appends the rendered frame to a caller-owned Vello scene without rasterization.
    pub fn append_to(&self, scene: &mut Scene, transform: Option<Affine>) {
        scene.append(&self.scene, transform);
    }

    /// Appends every visual layer owned by one entity.
    ///
    /// Interactive consumers use this to apply a transient transform without
    /// repeating image decoding, text shaping, or scene rendering.
    pub fn append_entity_to(
        &self,
        entity: EntityId,
        scene: &mut Scene,
        transform: Option<Affine>,
    ) -> bool {
        let Some(layers) = self.entity_scenes.get(&entity) else {
            return false;
        };
        for layer in layers {
            scene.append(layer, transform);
        }
        true
    }

    pub(crate) fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Returns a tightly cropped vector frame for one entity.
    ///
    /// The returned origin remains in page coordinates while its Vello scene
    /// is translated into frame-local coordinates.
    pub fn entity(&self, entity: EntityId) -> Result<Option<Self>> {
        let mut bounds = self
            .layers
            .iter()
            .filter(|layer| layer.entity == entity)
            .map(|layer| layer.bounds);
        let Some(first) = bounds.next() else {
            return Ok(None);
        };
        let (mut left, mut top) = (first.x, first.y);
        let (mut right, mut bottom) = (first.x + first.width, first.y + first.height);
        for bounds in bounds {
            left = left.min(bounds.x);
            top = top.min(bounds.y);
            right = right.max(bounds.x + bounds.width);
            bottom = bottom.max(bounds.y + bounds.height);
        }
        if ![left, top, right, bottom].into_iter().all(f32::is_finite) {
            return Err(Error::invalid(format!(
                "visual layer bounds are not finite for entity {entity}"
            )));
        }
        let left = left.floor() as i32;
        let top = top.floor() as i32;
        let right = right.ceil() as i32;
        let bottom = bottom.ceil() as i32;
        let width = u32::try_from((i64::from(right) - i64::from(left)).max(1))
            .map_err(|_| Error::invalid("visual layer width exceeds u32"))?;
        let height = u32::try_from((i64::from(bottom) - i64::from(top)).max(1))
            .map_err(|_| Error::invalid("visual layer height exceeds u32"))?;
        let mut scene = Scene::new();
        self.append_entity_to(
            entity,
            &mut scene,
            Some(Affine::translate((
                f64::from(self.left) - f64::from(left),
                f64::from(self.top) - f64::from(top),
            ))),
        );
        let layer_scene = Arc::new(scene);
        Ok(Some(Self {
            revision: self.revision,
            page: self.page,
            width,
            height,
            left,
            top,
            scene: layer_scene.clone(),
            entity_scenes: HashMap::from([(entity, vec![layer_scene])]),
            layers: self
                .layers
                .iter()
                .filter(|layer| layer.entity == entity)
                .cloned()
                .collect(),
            dependencies: self.dependencies.clone(),
            diagnostics: self.diagnostics.clone(),
        }))
    }

    pub(crate) fn at_revision(&self, revision: Revision) -> Self {
        Self {
            revision,
            page: self.page,
            width: self.width,
            height: self.height,
            left: self.left,
            top: self.top,
            scene: self.scene.clone(),
            entity_scenes: self.entity_scenes.clone(),
            layers: self.layers.clone(),
            dependencies: self.dependencies.clone(),
            diagnostics: self.diagnostics.clone(),
        }
    }

    #[cfg(test)]
    pub(crate) fn empty_for_test(
        revision: Revision,
        page: EntityId,
        dependencies: Vec<RenderDependency>,
    ) -> Self {
        Self {
            revision,
            page,
            width: 100,
            height: 100,
            left: 0,
            top: 0,
            scene: Arc::new(Scene::new()),
            entity_scenes: HashMap::new(),
            layers: Vec::new(),
            dependencies,
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct FrameKey {
    revision: Revision,
    font_generation: u64,
    request: RenderRequest,
}

impl FrameKey {
    fn new(revision: Revision, font_generation: u64, request: &RenderRequest) -> Self {
        Self {
            revision,
            font_generation,
            request: request.clone(),
        }
    }
}

struct FrameEntry {
    key: FrameKey,
    frame: Arc<Frame>,
}

struct FrameCache {
    entries: VecDeque<FrameEntry>,
    capacity: usize,
}

impl FrameCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn get(&mut self, key: &FrameKey) -> Option<Arc<Frame>> {
        let position = self.entries.iter().position(|entry| entry.key == *key)?;
        let entry = self.entries.remove(position)?;
        let frame = entry.frame.clone();
        self.entries.push_back(entry);
        Some(frame)
    }

    fn insert(&mut self, key: FrameKey, frame: Arc<Frame>) -> Arc<Frame> {
        if let Some(existing) = self.get(&key) {
            return existing;
        }
        if self.capacity == 0 {
            return frame;
        }
        while self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(FrameEntry {
            key,
            frame: frame.clone(),
        });
        frame
    }

    fn apply_changes(&mut self, changes: &Change) {
        let invalidate_all = !changes.relations.is_empty()
            || changes
                .components
                .iter()
                .any(|change| change.owner == ComponentOwner::Project)
            || changes
                .entities
                .iter()
                .any(|change| matches!(change, EntityChange::Inserted(_)));
        if invalidate_all {
            self.entries
                .retain(|entry| entry.key.revision != changes.from);
            return;
        }
        self.entries.retain_mut(|entry| {
            if entry.key.revision != changes.from {
                return true;
            }
            let depends_on_entity = |entity| {
                entry
                    .frame
                    .dependencies()
                    .contains(&RenderDependency::Entity(entity))
            };
            let entity_changed = changes.entities.iter().any(|change| match *change {
                EntityChange::Inserted(entity) | EntityChange::Removed(entity) => {
                    depends_on_entity(entity)
                }
            }) || changes.hierarchy.iter().copied().any(depends_on_entity)
                || changes.components.iter().any(|change| match change.owner {
                    ComponentOwner::Project => false,
                    ComponentOwner::Entity(entity) => depends_on_entity(entity),
                    ComponentOwner::Relation(relation) => entry
                        .frame
                        .dependencies()
                        .contains(&RenderDependency::Relation(relation)),
                });
            let relation_changed = changes.relations.iter().any(|change| {
                let relation = match *change {
                    RelationChange::Inserted(id)
                    | RelationChange::Removed(id)
                    | RelationChange::Changed(id) => id,
                };
                entry
                    .frame
                    .dependencies()
                    .contains(&RenderDependency::Relation(relation))
            });
            if entity_changed || relation_changed {
                false
            } else {
                entry.key.revision = changes.to;
                entry.frame = Arc::new(entry.frame.at_revision(changes.to));
                true
            }
        });
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

pub(crate) struct RenderedLayer {
    pub(crate) scene: Scene,
    pub(crate) layer: VisualLayer,
    pub(crate) diagnostics: Vec<RenderDiagnostic>,
}

/// A size difference too small for a reader to see, and small enough to be
/// arithmetic noise from the auto-fit bisection.
const SIZE_COHERENCE_EPSILON: f32 = 0.25;

/// Longest text still read as an interjection rather than as speech.
///
/// A balloon holding two or three words is not a balloon that happened to come
/// out roomy -- the artist drew it that way for one short cry, and filling it is
/// the correct lettering. Capping those is how a first attempt at this rule set
/// a burst balloon's `!?` at the page's ordinary reading size, leaving a small
/// mark floating in a large spiky outline: measurably more coherent, visibly
/// wrong, and a straight undoing of the balloon fill (auto-fit sizing balloon
/// text from the balloon rather than from the source em) on exactly the
/// balloons it was for.
///
/// 12 rather than a rounder number because that is where the measured gap is.
/// Over 20 pages, the bubble layers sitting more than 1.25x above their page's
/// prevailing size ran 2, 3, 3, 3, 5, 5, 10, 11, 12, then **18**, 19, 20, 22 --
/// and reading them back, everything at or below 12 characters is an
/// interjection (`?!`, `Oh my,`, `Ah~`, and a 12-character three-word cry) while
/// everything from 18 up is ordinary speech. Any cut in 12..=17 gives
/// the same partition on this material; it is one series, so the flag that turns
/// the whole rule off is the escape hatch.
const SIZE_COHERENCE_INTERJECTION_CHARS: usize = 12;

/// Brings a balloon that solved far above the page's prevailing dialogue size
/// back down towards it.
///
/// Auto-fit is per-layer: it returns the largest size that fits the box it was
/// given, so the size a balloon settles at is a fact about *that balloon's
/// area*, not about the voice. A page therefore ends up lettered in whatever
/// sizes its bubble geometry happened to imply -- measured 4:1 inside one page
/// -- and a reader reads a size change as emphasis.
///
/// The correction can only ever be downward, and that is not a design choice:
/// every layer is already at the largest size that fits, so the small ones
/// cannot be raised without overflowing. The ceiling one layer is given is
///
/// ```text
/// group_minimum.clamp(base, base * ratio)
/// ```
///
/// which does two jobs in one expression. The `clamp`'s upper end is the page
/// band: nothing may sit more than `ratio` above the prevailing size. The
/// `group_minimum` reconciles the cells detection cuts out of one bubble --
/// two halves of one sentence set at 44px and 19px is the worst instance
/// measured -- by bringing the roomy half down towards the cramped one.
///
/// The `clamp`'s *lower* end is what keeps the balloon fill: a ceiling is never
/// below the prevailing size, so no balloon can be pushed under the page's own
/// norm, which is precisely the under-filled state the balloon fill fixed. That
/// makes "this does not undo the balloon fill" a property of the expression
/// rather than a hope about the constant.
///
/// Interjections are exempt outright -- see
/// `SIZE_COHERENCE_INTERJECTION_CHARS`. That exemption is not a refinement, it
/// is the difference between the rule working and the rule being wrong: without
/// it the three largest corrections on the measured pages were all short cries
/// in balloons drawn big to hold them.
fn reconcile_text_sizes(
    composition: &Composition,
    renderer: &SceneRenderer,
    theme: &RenderTheme,
    text_renderer: &crate::TextRenderer,
    rendered: &mut [RenderedLayer],
) -> Result<()> {
    let Some(ratio) = theme.size_coherence.filter(|ratio| *ratio >= 1.0) else {
        return Ok(());
    };
    /* Dialogue only. Free-standing text takes its size from the ink of the
     * Japanese it replaces rather than from the box it landed in, so it is
     * already tied to something on the page and does not carry this defect --
     * and there are rarely enough captions on one page for a page statistic
     * over them to mean anything. Every caption is its own `fit_region` group,
     * so the rule would be a no-op for it in any case. */
    let solved = composition
        .layers
        .iter()
        .enumerate()
        .filter_map(|(index, layer)| {
            let Layer::Text(text) = layer else {
                return None;
            };
            if text.balloon_contour.is_none() || !text.auto_fit || text.point_text {
                return None;
            }
            let size = rendered[index].layer.font_size?;
            (size > 0.0).then_some((index, text, size))
        })
        .collect::<Vec<_>>();
    if solved.len() < 2 {
        return Ok(());
    }

    let base = prevailing_font_size(
        &solved
            .iter()
            .map(|(_, layer, size)| (*size, non_whitespace(&layer.text) as f32))
            .collect::<Vec<_>>(),
    );
    let mut group_minimum: HashMap<EntityId, f32> = HashMap::new();
    for (_, layer, size) in &solved {
        group_minimum
            .entry(layer.fit_region.unwrap_or(layer.entity))
            .and_modify(|minimum| *minimum = minimum.min(*size))
            .or_insert(*size);
    }

    let capped = solved
        .iter()
        .filter(|(_, layer, _)| !is_interjection(&layer.text))
        .filter_map(|(index, layer, size)| {
            let group = group_minimum[&layer.fit_region.unwrap_or(layer.entity)];
            let ceiling = size_ceiling(group, base, ratio);
            (*size > ceiling + SIZE_COHERENCE_EPSILON).then_some((*index, ceiling))
        })
        .collect::<Vec<_>>();
    if capped.is_empty() {
        return Ok(());
    }
    tracing::info!(
        base,
        ratio,
        capped = capped.len(),
        of = solved.len(),
        "size coherence"
    );

    let recomputed = capped
        .par_iter()
        .map(|&(index, ceiling)| {
            let Layer::Text(layer) = &composition.layers[index] else {
                unreachable!("only text layers are collected above");
            };
            let mut layer = layer.clone();
            layer.size_ceiling = Some(ceiling);
            text_renderer
                .render_layer(&layer, &renderer.fonts, theme)
                .map(|rendered| (index, rendered))
        })
        .collect::<Result<Vec<_>>>()?;
    for (index, layer) in recomputed {
        rendered[index] = layer;
    }
    Ok(())
}

/// How much of its size a layer gives up per round of relief.
///
/// Iterative rather than solved in one step because shrinking a layer also
/// MOVES it: `placement` re-centres the smaller run inside the same box, so the
/// overlap does not fall in proportion to the size and there is no closed form
/// to divide by. Each round costs one re-layout of one layer, which is the same
/// operation `reconcile_text_sizes` already pays for every capped balloon.
const COLLISION_RELIEF_STEP: f32 = 0.85;

/// How many rounds before the pass gives up and leaves the page as it is.
///
/// 0.85^6 is 0.377, so a layer can lose at most ~62% of its size. A collision
/// that survives that is not a sizing problem and shrinking further would trade
/// one unreadable layer for two.
const COLLISION_RELIEF_ROUNDS: usize = 6;

/// How much of the smaller layer's own area the overlap must cover before it
/// counts as a collision rather than a graze.
///
/// A rectangle is a coarse stand-in for a run of glyphs, so two placed rects can
/// clip corners while no ink is within tens of pixels of anything. Measured on 40
/// manga pages, every pair whose placed rects overlap at all, ranked by this
/// quantity:
///
/// | overlap of the smaller rect | ink in the zone | judged by eye |
/// |---|---|---|
/// | 0.97% | 2px | clean |
/// | 21.2% | 15px | clean |
/// | 35.4% .. 100% (7 pairs) | 411 .. 41,277px | all seven real collisions |
///
/// The gap between 21.2% and 35.4% is empty, and a five-judge vision panel split
/// those nine pairs exactly there. 0.30 sits in the gap. **n = 9**, so this is a
/// separator measured on one volume rather than a constant with a theory behind
/// it; the two bounds are recorded above so the next person can see how much room
/// it has rather than re-deriving them.
const COLLISION_RELIEF_MIN_SHARE: f32 = 0.30;

/// The placed extent of a solved text layer, as `(x0, y0, x1, y1)`.
///
/// `rendered_bounds` rather than `layout_bounds`: the first is where the text was
/// actually set, the second is the box it was given, and the whole point of this
/// pass is that those are different by about 4:1 in area.
fn placed_extent(layer: &RenderedLayer) -> Option<(f32, f32, f32, f32)> {
    let text = layer.layer.text.as_ref()?;
    let bounds = &text.rendered_bounds;
    (bounds.width > 0.0 && bounds.height > 0.0).then(|| {
        (
            bounds.x,
            bounds.y,
            bounds.x + bounds.width,
            bounds.y + bounds.height,
        )
    })
}

fn extents_overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> bool {
    a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3
}

/// The shared area as a fraction of the SMALLER rectangle's own area.
///
/// Of the smaller, because that is the layer with the most to lose: a caption
/// half-buried under a huge effect has most of itself contested while the effect
/// has barely any of itself, and taking the larger denominator would score the
/// worst cases lowest.
fn overlap_share(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let shared = (a.2.min(b.2) - a.0.max(b.0)).max(0.0) * (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
    let smaller = ((a.2 - a.0) * (a.3 - a.1)).min((b.2 - b.0) * (b.3 - b.1));
    if smaller <= 0.0 { 0.0 } else { shared / smaller }
}

/// Which of two colliding layers gives up size, by index into the pair.
///
/// **Dialogue never yields, and between two free-standing texts the LARGER one
/// does.** That second half is deliberately the opposite of the rule an earlier
/// pass used when it was *dropping* a layer, and the difference is the whole reason
/// this pass is worth having. Dropping asks "which loss hurts least", so it took
/// the smaller. Shrinking asks "which one has room to give", and that is the big
/// one: the measured case is a giant sound effect painted across a line of
/// dialogue, where taking size off the effect leaves both readable and taking it
/// off the already-small dialogue leaves neither.
///
/// Ties fall to the second index so the answer never depends on float equality
/// and a page renders the same way twice.
fn collision_loser(
    left_dialogue: bool,
    right_dialogue: bool,
    left_size: f32,
    right_size: f32,
) -> Option<usize> {
    match (left_dialogue, right_dialogue) {
        // Two balloons overlapping is a detection or layout fault this pass
        // cannot fix by shrinking, and shrinking speech is the one thing a
        // reader notices immediately. Left alone on purpose.
        (true, true) => None,
        (true, false) => Some(1),
        (false, true) => Some(0),
        (false, false) => Some(usize::from(right_size >= left_size)),
    }
}

/// Shrink text that auto-fit placed on top of other text.
///
/// Runs after `reconcile_text_sizes` and before any scene is appended, which is
/// the one point in the pipeline where every layer's layout is solved and nothing
/// has been drawn. See `RenderTheme::collision_relief` for why the decision cannot
/// be made earlier.
fn reconcile_text_collisions(
    composition: &Composition,
    renderer: &SceneRenderer,
    theme: &RenderTheme,
    text_renderer: &crate::TextRenderer,
    rendered: &mut [RenderedLayer],
) -> Result<()> {
    if !theme.collision_relief {
        return Ok(());
    }

    /* Only auto-fit layers can be relieved: a layer that asked for an explicit
     * size was told what to be, and point text has no box to shrink within. */
    let eligible = composition
        .layers
        .iter()
        .enumerate()
        .filter_map(|(index, layer)| {
            let Layer::Text(text) = layer else {
                return None;
            };
            (text.auto_fit && !text.point_text).then_some((index, text))
        })
        .collect::<Vec<_>>();
    if eligible.len() < 2 {
        return Ok(());
    }

    // The size each layer is currently set at, which the loop lowers.
    let mut ceiling: HashMap<usize, f32> = HashMap::new();
    let mut relieved = 0usize;

    for _round in 0..COLLISION_RELIEF_ROUNDS {
        let mut shrink: Vec<usize> = Vec::new();
        for left in 0..eligible.len() {
            for right in (left + 1)..eligible.len() {
                let (li, ltext) = eligible[left];
                let (ri, rtext) = eligible[right];
                let (Some(lbox), Some(rbox)) =
                    (placed_extent(&rendered[li]), placed_extent(&rendered[ri]))
                else {
                    continue;
                };
                if !extents_overlap(lbox, rbox) {
                    continue;
                }
                if overlap_share(lbox, rbox) < COLLISION_RELIEF_MIN_SHARE {
                    continue;
                }
                let (Some(lsize), Some(rsize)) =
                    (rendered[li].layer.font_size, rendered[ri].layer.font_size)
                else {
                    continue;
                };
                let Some(which) = collision_loser(
                    ltext.balloon_contour.is_some(),
                    rtext.balloon_contour.is_some(),
                    lsize,
                    rsize,
                ) else {
                    continue;
                };
                let index = if which == 0 { li } else { ri };
                let size = if which == 0 { lsize } else { rsize };
                /* Never below the readable floor. A layer already at the floor
                 * has nothing left to give, and shrinking it further would swap a
                 * collision for an illegible line -- strictly worse, and it is
                 * the trade `--no-translate-sfx` already makes by lettering
                 * nothing. */
                let next = size * COLLISION_RELIEF_STEP;
                if next + f32::EPSILON < theme.minimum_font_size {
                    continue;
                }
                if !shrink.contains(&index) {
                    shrink.push(index);
                    ceiling.insert(index, next);
                }
            }
        }
        if shrink.is_empty() {
            break;
        }
        let recomputed = shrink
            .par_iter()
            .map(|&index| {
                let Layer::Text(layer) = &composition.layers[index] else {
                    unreachable!("only text layers are collected above");
                };
                let mut layer = layer.clone();
                /* The SMALLER of the two ceilings, so this pass can never undo
                 * the page-level coherence cap `reconcile_text_sizes` applied a
                 * few lines earlier. That rule brought an outlier down; this one
                 * may only bring it down further. */
                layer.size_ceiling = Some(match layer.size_ceiling {
                    Some(existing) => existing.min(ceiling[&index]),
                    None => ceiling[&index],
                });
                text_renderer
                    .render_layer(&layer, &renderer.fonts, theme)
                    .map(|rendered| (index, rendered))
            })
            .collect::<Result<Vec<_>>>()?;
        for (index, layer) in recomputed {
            rendered[index] = layer;
            relieved += 1;
        }
    }

    if relieved > 0 {
        tracing::info!(
            relieved,
            of = eligible.len(),
            "collision relief"
        );
    }
    Ok(())
}

/// How much text a layer holds, ignoring the spaces between the words.
fn non_whitespace(text: &str) -> usize {
    text.chars().filter(|character| !character.is_whitespace()).count()
}

/// Whether a balloon is holding a cry rather than a line of speech, and so
/// whether its size is the artist's emphasis rather than an accident of area.
fn is_interjection(text: &str) -> bool {
    non_whitespace(text) <= SIZE_COHERENCE_INTERJECTION_CHARS
}

/// The ceiling one balloon is allowed, given the smallest size solved anywhere
/// in its own bubble and the page's prevailing size.
///
/// The whole rule, so it can be read and tested as one thing.
fn size_ceiling(group_minimum: f32, base: f32, ratio: f32) -> f32 {
    group_minimum.clamp(base, base * ratio)
}

/// The size at which half of the page's dialogue *characters* are set, given
/// `(size, character count)` for every balloon.
///
/// Weighted by character count rather than by balloon, because the balloons
/// that solve far too large are exactly the ones holding two or three words: an
/// unweighted median lets a page of interjections set the norm for the page of
/// speech around it.
fn prevailing_font_size(sizes: &[(f32, f32)]) -> f32 {
    let mut weighted = sizes.to_vec();
    weighted.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total: f32 = weighted.iter().map(|(_, weight)| weight).sum();
    if total <= 0.0 {
        // Nothing to weigh by: fall back to the plain median, and to nothing at
        // all when there is not even that.
        return weighted.get(weighted.len() / 2).map_or(0.0, |&(size, _)| size);
    }
    let mut accumulated = 0.0;
    for (size, weight) in &weighted {
        accumulated += weight;
        if accumulated * 2.0 >= total {
            return *size;
        }
    }
    weighted.last().map_or(0.0, |(size, _)| *size)
}

fn render_layer(
    layer: &Layer,
    composition: &Composition,
    snapshot: &Snapshot,
    renderer: &SceneRenderer,
    theme: &RenderTheme,
    text_renderer: &crate::TextRenderer,
) -> Result<RenderedLayer> {
    match layer {
        Layer::Image(layer) => render_image(layer, composition, snapshot, renderer),
        Layer::Text(layer) => text_renderer.render_layer(layer, &renderer.fonts, theme),
    }
}

fn render_image(
    layer: &ImageLayer,
    composition: &Composition,
    snapshot: &Snapshot,
    renderer: &SceneRenderer,
) -> Result<RenderedLayer> {
    let image = renderer.image(snapshot, &layer.asset)?;
    if layer.is_base && (image.width != composition.width || image.height != composition.height) {
        return Err(Error::invalid(format!(
            "base image for page {} is {}x{}, expected {}x{}",
            composition.page, image.width, image.height, composition.width, composition.height
        )));
    }
    let pixels: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(ImageBytes(image.pixels.clone()));
    let data = ImageData {
        data: Blob::new(pixels),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width: image.width,
        height: image.height,
    };
    let transform = Affine::scale_non_uniform(
        f64::from(layer.bounds.width) / f64::from(image.width),
        f64::from(layer.bounds.height) / f64::from(image.height),
    )
    .then_translate(Vec2::new(
        f64::from(layer.bounds.x),
        f64::from(layer.bounds.y),
    ));
    let mut scene = Scene::new();
    if layer.opacity < 1.0 {
        scene.push_layer(
            Fill::NonZero,
            Mix::Normal,
            layer.opacity,
            transform,
            &Rect::new(0.0, 0.0, f64::from(image.width), f64::from(image.height)),
        );
    }
    scene.draw_image(&data, transform);
    if layer.opacity < 1.0 {
        scene.pop_layer();
    }
    Ok(RenderedLayer {
        scene,
        layer: VisualLayer {
            entity: layer.entity,
            kind: layer.kind,
            name: layer.name.clone(),
            bounds: layer.bounds.into(),
            font_size: None,
            text: None,
        },
        diagnostics: Vec::new(),
    })
}

struct ImageBytes(Arc<[u8]>);

impl AsRef<[u8]> for ImageBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl SceneRenderer {
    fn image(&self, snapshot: &Snapshot, asset: &Asset) -> Result<Arc<DecodedImage>> {
        if let Some(image) = self.images.lock().get(asset.blob) {
            return Ok(image);
        }
        let bytes = snapshot.read_blob(asset.blob)?;
        let decoded = image::load_from_memory(&bytes)
            .map_err(|source| Error::Image {
                blob: asset.blob,
                source,
            })?
            .into_rgba8();
        if let (Some(expected_width), Some(expected_height)) =
            (asset.metadata.width, asset.metadata.height)
            && (decoded.width() != expected_width || decoded.height() != expected_height)
        {
            return Err(Error::invalid(format!(
                "blob {} decoded as {}x{}, expected {}x{}",
                asset.blob,
                decoded.width(),
                decoded.height(),
                expected_width,
                expected_height
            )));
        }
        let image = Arc::new(DecodedImage {
            width: decoded.width(),
            height: decoded.height(),
            pixels: Arc::from(decoded.into_raw()),
        });
        self.images.lock().insert(asset.blob, image.clone());
        Ok(image)
    }
}

struct DecodedImage {
    width: u32,
    height: u32,
    pixels: Arc<[u8]>,
}

impl DecodedImage {
    fn byte_len(&self) -> usize {
        self.pixels.len()
    }
}

struct CachedImage {
    image: Arc<DecodedImage>,
    last_used: u64,
}

struct DecodedImageCache {
    entries: HashMap<BlobId, CachedImage>,
    max_bytes: usize,
    bytes: usize,
    clock: u64,
}

impl DecodedImageCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_bytes,
            bytes: 0,
            clock: 0,
        }
    }

    fn get(&mut self, id: BlobId) -> Option<Arc<DecodedImage>> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(&id)?;
        entry.last_used = self.clock;
        Some(entry.image.clone())
    }

    fn insert(&mut self, id: BlobId, image: Arc<DecodedImage>) {
        let image_bytes = image.byte_len();
        if self.max_bytes == 0 || image_bytes > self.max_bytes {
            return;
        }
        if let Some(previous) = self.entries.remove(&id) {
            self.bytes = self.bytes.saturating_sub(previous.image.byte_len());
        }
        while self.bytes.saturating_add(image_bytes) > self.max_bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(id, _)| *id)
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.bytes = self.bytes.saturating_sub(removed.image.byte_len());
            }
        }
        self.clock = self.clock.wrapping_add(1);
        self.bytes += image_bytes;
        self.entries.insert(
            id,
            CachedImage {
                image,
                last_used: self.clock,
            },
        );
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}

/// The page-level size rule, tested as arithmetic.
///
/// Deliberately separate from `mod tests` below, which shares one helper that
/// builds a whole scene through `Compositor::compile` -- that helper fails
/// upstream, at the forked revision, with every change of ours reverted, and these
/// assertions are about the rule rather than about the scene graph.
#[cfg(test)]
mod size_coherence_tests {
    use super::{is_interjection, prevailing_font_size, size_ceiling};

    #[test]
    fn a_cry_is_told_from_a_line_of_speech() {
        /* Every one of these stands for a bubble that solved more than 1.25x
         * above its page's prevailing size in the 20-page measurement, listed
         * shortest first. Each string has the non-whitespace count of the
         * measured line it stands for, which is all `is_interjection` reads.
         * The rule has to keep the top group
         * and cap the bottom one, and several are the same page, so this is the
         * partition itself rather than an example of it. */
        for cry in ["?!", "\u{30fb}r\u{30fb}", "Ah~", "Oh my,", "I see.", "Not a chance!!"] {
            assert!(is_interjection(cry), "{cry:?} should keep its balloon fill");
        }
        for speech in [
            "Have you seen my keys?",
            "Why is the kettle still on?",
            "The morning fog finally lifted,",
            "--This is basically a thunderstorm!!",
            "We have three more stations to go before the lake!",
        ] {
            assert!(!is_interjection(speech), "{speech:?} is a line, not a cry");
        }
        // Spaces are not what makes a line long: "Not a chance!!" is three words
        // and 12 characters, and it is the closest call in the set.
        assert!(is_interjection("A B C D E F G H I J K L"));
    }

    #[test]
    fn the_prevailing_size_is_where_half_the_characters_are_set() {
        // Three roomy interjections against one dense balloon. By balloon the
        // median is 40; by character it is 18, which is where the page's
        // reading actually happens.
        let page = [(44.0, 3.0), (40.0, 4.0), (38.0, 3.0), (18.0, 60.0)];
        assert!((prevailing_font_size(&page) - 18.0).abs() < f32::EPSILON);

        // Degenerate inputs must still answer something usable rather than
        // panicking on an empty slice or a zero total.
        assert!((prevailing_font_size(&[(21.0, 1.0)]) - 21.0).abs() < f32::EPSILON);
        assert!((prevailing_font_size(&[(21.0, 0.0), (13.0, 0.0)]) - 21.0).abs() < f32::EPSILON);
        assert!(prevailing_font_size(&[]).abs() < f32::EPSILON);
    }

    #[test]
    fn a_ceiling_is_never_below_the_prevailing_size() {
        /* This is the property that keeps the balloon fill. The defect it fixed
         * was balloons set BELOW what the page could hold; a rule that
         * can never push one under the page's own norm cannot recreate it, at
         * any ratio and for any group. */
        for &group in &[1.0_f32, 9.0, 18.0, 24.0, 44.0, 300.0] {
            for &ratio in &[1.0_f32, 1.1, 1.25, 1.5, 4.0] {
                let ceiling = size_ceiling(group, 20.0, ratio);
                assert!(ceiling >= 20.0, "group={group} ratio={ratio} -> {ceiling}");
                assert!(
                    ceiling <= 20.0 * ratio,
                    "group={group} ratio={ratio} -> {ceiling}"
                );
            }
        }
    }

    #[test]
    fn the_two_cells_of_one_bubble_are_brought_together() {
        /* The worst instance measured: one continuous sentence cut across two
         * cells of the same bubble and solved at 44px and 19px. Both cells see
         * the group minimum, so the roomy half comes down -- but only to the
         * page's prevailing size, not all the way to the cramped half, because
         * dragging a whole bubble down to its most crowded corner would waste
         * the room that corner does not have. */
        let (base, ratio) = (22.0, 1.25);
        let cramped = size_ceiling(19.0, base, ratio);
        let roomy = size_ceiling(19.0, base, ratio);
        assert!((roomy - 22.0).abs() < f32::EPSILON);
        assert!((cramped - 22.0).abs() < f32::EPSILON);
        // 44:19 was 2.3:1. What the reader now sees is 22 against 19.
        assert!(roomy.min(19.0) / roomy.max(19.0) > 0.8);

        // A balloon alone in its group is bounded by the band and nothing else.
        assert!((size_ceiling(44.0, base, ratio) - 27.5).abs() < f32::EPSILON);
    }
}

#[cfg(test)]
mod tests {
    use koharu_scene::{
        At, Authored, BubbleRegion, FitsTo, Geometry, LanguageTag, Origin, PageDraft, Session,
        SourceText, TextAlignment, TextLayout, TextLayoutKind, Typography,
    };

    use super::*;
    use crate::{Compositor, RenderRequest};

    fn render_frame(
        composition: &Composition,
        snapshot: &Snapshot,
        renderer: &SceneRenderer,
        theme: &RenderTheme,
    ) -> Result<Frame> {
        Frame::render(
            composition,
            snapshot,
            renderer,
            theme,
            &crate::TextRenderer::new(),
        )
    }

    fn cached_frame(
        revision: Revision,
        page: EntityId,
        dependencies: Vec<RenderDependency>,
    ) -> Arc<Frame> {
        Arc::new(Frame::empty_for_test(revision, page, dependencies))
    }

    #[test]
    fn change_sets_reuse_only_unaffected_frames() {
        let mut session = Session::memory().unwrap();
        let mut ids = None;
        let create = session
            .snapshot()
            .patch(|edit| {
                let first_page = edit.add_page(PageDraft::new("first", 100.0, 100.0), At::End)?;
                let first_entity = edit.add_entity(first_page, At::End)?;
                edit.set(first_entity, &Geometry::rectangle(0.0, 0.0, 10.0, 10.0))?;
                let second_page = edit.add_page(PageDraft::new("second", 100.0, 100.0), At::End)?;
                let second_entity = edit.add_entity(second_page, At::End)?;
                edit.set(second_entity, &Geometry::rectangle(0.0, 0.0, 10.0, 10.0))?;
                ids = Some((first_page, first_entity, second_entity));
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(create).unwrap().snapshot;
        let (first_page, first_entity, second_entity) = ids.unwrap();
        let request = RenderRequest::transparent(first_page);
        let mut cache = FrameCache::new(2);
        cache.insert(
            FrameKey::new(snapshot.revision(), 0, &request),
            cached_frame(
                snapshot.revision(),
                first_page,
                vec![
                    RenderDependency::Entity(first_page),
                    RenderDependency::Entity(first_entity),
                ],
            ),
        );

        let unrelated = snapshot
            .patch(|edit| edit.set(second_entity, &Geometry::rectangle(1.0, 1.0, 10.0, 10.0)))
            .unwrap();
        let commit = session.commit(unrelated).unwrap();
        cache.apply_changes(&commit.changes);
        let reused = cache
            .get(&FrameKey::new(commit.snapshot.revision(), 0, &request))
            .expect("unaffected frame should advance to the new revision");
        assert_eq!(reused.revision(), commit.snapshot.revision());

        let relevant = commit
            .snapshot
            .patch(|edit| edit.set(first_entity, &Geometry::rectangle(2.0, 2.0, 10.0, 10.0)))
            .unwrap();
        let commit = session.commit(relevant).unwrap();
        cache.apply_changes(&commit.changes);

        assert!(
            cache
                .get(&FrameKey::new(commit.snapshot.revision(), 0, &request))
                .is_none()
        );
    }

    fn text_fixture(
        balloon_width: f64,
        balloon_height: f64,
        text: &str,
        font_size: f32,
    ) -> (koharu_scene::Snapshot, Composition, EntityId) {
        let mut session = Session::memory().unwrap();
        // The PAGE is captured out of the closure, not derived afterwards.
        //
        // `snapshot.parent(text_layer)` does NOT answer the page:
        // `Edit::add_text_layer` calls `ensure_text_group(page)` and parents the
        // layer to that group, so the parent is always the group and
        // `RenderRequest::transparent(group)` fails on the first line of
        // `compile` -- `snapshot.page(request.page)?` -- with
        // `EntityNotFound(<the text group>)`. This is what took all six tests in
        // this module red, and `compositor.rs`'s own `fixture` never had the bug
        // because it already captures its ids out of the closure.
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 300.0, 200.0), At::End)?;
                let bubble = edit.add_analysis_region::<BubbleRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(10.0, 10.0, balloon_width, balloon_height),
                    None,
                )?;
                let content = edit.add_text_content(page, At::End)?;
                edit.set(
                    content,
                    &SourceText {
                        text: Authored::user(text.to_owned()),
                        language: Some(LanguageTag::new("en")?),
                    },
                )?;
                let entity = edit.add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )?;
                edit.set(
                    entity,
                    &Geometry::rectangle(10.0, 10.0, balloon_width, balloon_height),
                )?;
                edit.set(
                    entity,
                    &Typography {
                        origin: Origin::User,
                        preferred_font: None,
                        font_weight: None,
                        size: Some(font_size),
                        auto_fit: false,
                        color: None,
                        stroke_color: None,
                        stroke_width: None,
                        alignment: Some(TextAlignment::Center),
                        writing_mode: Some(koharu_scene::WritingMode::Horizontal),
                        extensions: Default::default(),
                    },
                )?;
                edit.relate::<FitsTo>(entity, bubble)?;
                ids = Some((page, entity));
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).unwrap().snapshot;
        let (page, entity) = ids.unwrap();
        let composition = Compositor::new()
            .compile(&snapshot, &RenderRequest::transparent(page))
            .unwrap();
        (snapshot, composition, entity)
    }

    #[test]
    fn explicit_font_size_skips_auto_fit_and_reports_overflow() {
        let (snapshot, composition, entity) =
            text_fixture(40.0, 18.0, "This dialogue cannot fit", 18.0);
        let theme = RenderTheme {
            text_inset: [0.0; 4],
            ..RenderTheme::default()
        };

        let frame = render_frame(&composition, &snapshot, &SceneRenderer::new(), &theme).unwrap();
        let rendered = frame
            .layers()
            .iter()
            .find(|layer| layer.entity == entity)
            .unwrap();

        assert_eq!(rendered.font_size, Some(18.0));
        assert!(frame.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            RenderDiagnostic::TextOverflow { entity: found, .. } if *found == entity
        )));
        assert_eq!(frame.entity_scenes[&entity][0].encoding().n_clips, 0);
    }

    #[test]
    fn free_text_auto_fits_the_exact_original_block_without_balloon_air() {
        let (snapshot, mut composition, entity) =
            text_fixture(40.0, 18.0, "This free text must shrink", 18.0);
        let Layer::Text(layer) = &mut composition.layers[0] else {
            panic!("expected a text layer");
        };
        layer.balloon_contour = None;
        layer.font_size = None;
        // Without this the test does not exercise the thing it is named after.
        // The fixture's `Typography` sets `auto_fit: false`, so `text_renderer`
        // takes the `with_font_size(...)` arm, the size never moves off
        // `automatic_maximum()`, and `rendered.font_size < theme.font_size`
        // cannot hold. Identical under upstream: its branch is `layer.auto_fit`
        // where ours is `layer.auto_fit && is_bubble_text`, and with `auto_fit`
        // false both take the same else arm -- so this is stale, not a
        // divergence of ours.
        layer.auto_fit = true;
        let original_bounds = layer.bounds;
        let theme = RenderTheme {
            minimum_font_size: 1.0,
            text_inset: [100.0; 4],
            ..RenderTheme::default()
        };

        let frame = render_frame(&composition, &snapshot, &SceneRenderer::new(), &theme).unwrap();
        let rendered = frame
            .layers()
            .iter()
            .find(|rendered| rendered.entity == entity)
            .unwrap();

        assert!(rendered.font_size.unwrap() < theme.font_size);
        assert!(rendered.bounds.x >= original_bounds.x - f32::EPSILON);
        assert!(rendered.bounds.y >= original_bounds.y - f32::EPSILON);
        assert!(rendered.bounds.width <= original_bounds.width + f32::EPSILON);
        assert!(rendered.bounds.height <= original_bounds.height + f32::EPSILON);
        assert!(!frame.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            RenderDiagnostic::TextOverflow { entity: found, .. } if *found == entity
        )));
    }

    /// Diagnostic, not an assertion -- the same shape as
    /// `layout::tests::probe_width_against_max_width`, and ignored for the
    /// same reason: it prints what the solver did rather than claiming what it
    /// should do.
    ///
    /// The measured case: a skill name in a 145x626 vertical column of four Han
    /// glyphs, whose glyph pitch is ~156 px, lettered as a three-word English
    /// name at **41.626236**. Two readings of that number survived the source:
    /// the English is fitted to the column's WIDTH (the 7 characters of its
    /// longest word into 145 px at ~0.5 em is ~41.4, which matches almost
    /// exactly), or it is fitted to the widened 354 px box the wire reports as
    /// `fit_width` and something else binds. Those imply different fixes, so
    /// print all three and stop guessing. The English strings below are
    /// invented stand-ins with the measured names' word lengths.
    ///
    /// Run with:
    /// `cargo test -p koharu-renderer probe_what_binds_a_vertical_column -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic probe for vertical-column fitting, not an assertion"]
    fn probe_what_binds_a_vertical_column() {
        // Two slices joined across a seam, measured: the server reports
        // `font_size` 201.0 RAW (the region is refused, so the fitter never
        // runs) and 61.671852 once it does. This asks what the fitter is
        // obeying to give up 139px of em on a box with 1693px of unused height.
        for (label, width, height, text, em) in [
            ("column as detected      145x626", 145.0, 626.0, "Crimson Moon Palm", 156.0),
            ("widened box             354x626", 354.0, 626.0, "Crimson Moon Palm", 156.0),
            ("same area, landscape    626x145", 626.0, 145.0, "Crimson Moon Palm", 156.0),
            ("joined slices         204x1693", 204.0, 1693.0, "Duskmere Ash Vale \u{b7} Duskmere Ro", 201.0),
            ("joined, ROTATED       1693x204", 1693.0, 204.0, "Duskmere Ash Vale \u{b7} Duskmere Ro", 201.0),
            // The longest single word alone. A word cannot be broken, so if
            // this solves at the same size the binding constraint is the WORD
            // and no amount of line breaking can help.
            ("joined, longest word   204x1693", 204.0, 1693.0, "Duskmere", 201.0),
        ] {
            let (snapshot, mut composition, entity) = text_fixture(width, height, text, em);
            let Layer::Text(layer) = &mut composition.layers[0] else {
                panic!("expected a text layer");
            };
            layer.balloon_contour = None;
            layer.font_size = Some(em);
            layer.auto_fit = true;
            let theme = RenderTheme {
                minimum_font_size: 9.0,
                ..RenderTheme::default()
            };
            let frame =
                render_frame(&composition, &snapshot, &SceneRenderer::new(), &theme).unwrap();
            let rendered = frame
                .layers()
                .iter()
                .find(|rendered| rendered.entity == entity)
                .unwrap();
            let b = rendered.bounds;
            println!(
                "{label} -> solved {:>7.2}  ink {:>7.1}x{:<7.1} of {:.0}x{:.0}  uses {:>5.1}% W {:>5.1}% H{}{}",
                rendered.font_size.unwrap_or(f32::NAN),
                b.width,
                b.height,
                width,
                height,
                100.0 * b.width / width as f32,
                100.0 * b.height / height as f32,
                if b.width > width as f32 + 0.5 { "  OVERFLOWS-W" } else { "" },
                if b.height > height as f32 + 0.5 { "  OVERFLOWS-H" } else { "" },
            );
        }
    }

    #[test]
    fn automatic_balloon_text_can_grow_beyond_the_theme_font_size() {
        let (snapshot, mut composition, entity) = text_fixture(240.0, 120.0, "Hi", 18.0);
        let Layer::Text(layer) = &mut composition.layers[0] else {
            panic!("expected a text layer");
        };
        layer.font_size = None;
        let theme = RenderTheme {
            text_inset: [0.0; 4],
            ..RenderTheme::default()
        };

        let frame = render_frame(&composition, &snapshot, &SceneRenderer::new(), &theme).unwrap();
        let rendered = frame
            .layers()
            .iter()
            .find(|rendered| rendered.entity == entity)
            .unwrap();

        assert!(rendered.font_size.unwrap() > theme.font_size);
    }

    #[test]
    fn entity_frame_is_cropped_once_in_page_coordinates() {
        let (snapshot, composition, entity) = text_fixture(240.0, 120.0, "Cropped", 18.0);
        let frame = render_frame(
            &composition,
            &snapshot,
            &SceneRenderer::new(),
            &RenderTheme::default(),
        )
        .unwrap();
        let bounds = frame
            .layers()
            .iter()
            .find(|layer| layer.entity == entity)
            .unwrap()
            .bounds;
        let expected_origin = (bounds.x.floor() as i32, bounds.y.floor() as i32);
        let expected_size = (
            (bounds.x + bounds.width).ceil() as u32 - expected_origin.0 as u32,
            (bounds.y + bounds.height).ceil() as u32 - expected_origin.1 as u32,
        );

        let entity_frame = frame.entity(entity).unwrap().unwrap();
        assert_eq!(entity_frame.origin(), expected_origin);
        assert_eq!(entity_frame.size(), expected_size);

        let nested = entity_frame.entity(entity).unwrap().unwrap();
        assert_eq!(nested.origin(), entity_frame.origin());
        assert_eq!(nested.size(), entity_frame.size());
    }

    #[test]
    fn rendering_reports_text_below_the_readability_floor() {
        let (snapshot, composition, entity) = text_fixture(240.0, 120.0, "Small dialogue", 8.0);
        let theme = RenderTheme {
            minimum_font_size: 9.0,
            text_inset: [0.0; 4],
            ..RenderTheme::default()
        };

        let frame = render_frame(&composition, &snapshot, &SceneRenderer::new(), &theme).unwrap();

        assert!(frame.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            RenderDiagnostic::TextBelowReadableSize {
                entity: found,
                font_size,
                minimum_font_size,
            } if *found == entity && *font_size == 8.0 && *minimum_font_size == 9.0
        )));
    }

    #[test]
    fn rendering_rotates_text_and_reported_bounds() {
        let (snapshot, composition, entity) = text_fixture(240.0, 120.0, "Rotated text", 18.0);
        let theme = RenderTheme {
            text_inset: [0.0; 4],
            ..RenderTheme::default()
        };
        let renderer = SceneRenderer::new();
        let baseline = render_frame(&composition, &snapshot, &renderer, &theme).unwrap();
        let baseline_bounds = baseline
            .layers()
            .iter()
            .find(|rendered| rendered.entity == entity)
            .unwrap()
            .bounds;

        let mut rotated_composition = composition.clone();
        let Layer::Text(layer) = &mut rotated_composition.layers[0] else {
            panic!("expected a text layer");
        };
        layer.angle_degrees = 90.0;
        let rotated = render_frame(&rotated_composition, &snapshot, &renderer, &theme).unwrap();
        let rotated_bounds = rotated
            .layers()
            .iter()
            .find(|rendered| rendered.entity == entity)
            .unwrap()
            .bounds;

        assert!((rotated_bounds.width - baseline_bounds.height).abs() < 1e-4);
        assert!((rotated_bounds.height - baseline_bounds.width).abs() < 1e-4);
        assert!(
            (rotated_bounds.x + rotated_bounds.width * 0.5
                - baseline_bounds.x
                - baseline_bounds.width * 0.5)
                .abs()
                < 1e-4
        );
        assert!(
            (rotated_bounds.y + rotated_bounds.height * 0.5
                - baseline_bounds.y
                - baseline_bounds.height * 0.5)
                .abs()
                < 1e-4
        );
    }
}

#[cfg(test)]
mod collision_relief_tests {
    use super::{
        COLLISION_RELIEF_MIN_SHARE, COLLISION_RELIEF_STEP, collision_loser, extents_overlap,
        overlap_share,
    };

    /// Dialogue never gives up size, and between two free-standing texts the
    /// LARGER one does -- the opposite of the order an earlier pass used to
    /// decide which effect to DROP. The measured case is a huge sound effect painted
    /// across a line of dialogue: the effect has room to give and the dialogue
    /// does not.
    #[test]
    fn the_larger_free_standing_text_yields_and_dialogue_never_does() {
        assert_eq!(collision_loser(true, false, 10.0, 90.0), Some(1));
        assert_eq!(collision_loser(false, true, 90.0, 10.0), Some(0));
        // Bigger effect loses, whichever side it is on.
        assert_eq!(collision_loser(false, false, 90.0, 10.0), Some(0));
        assert_eq!(collision_loser(false, false, 10.0, 90.0), Some(1));
        // Two balloons are not this pass's business.
        assert_eq!(collision_loser(true, true, 10.0, 90.0), None);
    }

    /// The separator is bounded on BOTH sides by measurement, and this asserts
    /// the bounds rather than restating the constant: 21.2% was judged clean by
    /// eye and 35.4% was a real collision, so the threshold has to lie strictly
    /// between them or it re-decides one of the nine measured pairs.
    #[test]
    fn the_share_threshold_sits_in_the_measured_gap() {
        assert!(COLLISION_RELIEF_MIN_SHARE > 0.2118);
        assert!(COLLISION_RELIEF_MIN_SHARE < 0.3544);
    }

    /// A step that did not shrink would spin the loop for six rounds and change
    /// nothing, and one at or below zero would invert the size.
    #[test]
    fn the_step_really_shrinks() {
        assert!(COLLISION_RELIEF_STEP < 1.0 && COLLISION_RELIEF_STEP > 0.0);
        // Six rounds must not collapse a layer to nothing.
        assert!(COLLISION_RELIEF_STEP.powi(6) > 0.3);
    }

    #[test]
    fn share_is_measured_against_the_smaller_rectangle() {
        // A small rect wholly inside a big one is 100% contested, not 4%.
        let small = (10.0, 10.0, 20.0, 20.0);
        let big = (0.0, 0.0, 50.0, 50.0);
        assert!((overlap_share(small, big) - 1.0).abs() < 1e-6);
        assert!((overlap_share(big, small) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rectangles_that_only_touch_do_not_overlap() {
        assert!(extents_overlap((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)));
        assert!(!extents_overlap((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0)));
        assert!((overlap_share((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0))).abs() < 1e-6);
    }
}
