use std::collections::BTreeSet;

use koharu_scene::{AssetRole, EntityId};

use crate::StrokeOptions;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum VerticalAlignment {
    Top,
    #[default]
    Center,
    Bottom,
}

/// Whether layer visibility and opacity are resolved into the rendered frame.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum LayerPresentation {
    /// Omit hidden layers and bake opacity into each rendered layer.
    #[default]
    Resolved,
    /// Retain every layer at full opacity so an interactive compositor can
    /// apply visibility and opacity without repeating layout or shaping.
    Deferred,
}

/// Non-persistent visual policy applied after scene intent has been resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderTheme {
    pub font_families: Vec<String>,
    pub font_size: f32,
    pub minimum_font_size: f32,
    pub text_color: [u8; 4],
    pub text_stroke: Option<StrokeOptions>,
    pub line_height: f32,
    pub letter_spacing: f32,
    pub word_spacing: f32,
    /// Insets in top, right, bottom, left order.
    pub text_inset: [f32; 4],
    pub vertical_alignment: VerticalAlignment,
    /// Overrides the per-layer hyphenation choice for the whole page.
    ///
    /// `None` keeps the built-in rule, which is not uniform: the default policy
    /// is `Normal` (hyphens weighed during ordinary line optimisation) and only
    /// *horizontal English inside a bubble* is softened to `LastResort`. So
    /// free-standing captions and signs -- the narrowest boxes on a page -- get
    /// the most aggressive of the three settings.
    ///
    /// Exposed because it was unreachable from outside the renderer, and
    /// hyphenation was the most frequent complaint in the 20-page render audit:
    /// ~17 of 20 pages, names split, already-hyphenated words broken a second
    /// time, two-letter orphans.
    pub hyphenation: Option<crate::HyphenationPolicy>,
    /// How far above the page's prevailing dialogue size one balloon may be set,
    /// as a multiple of it. `None` leaves every balloon to solve alone.
    ///
    /// Auto-fit is per-layer and knows nothing of the page, so the size a
    /// balloon settles at encodes *its own area* rather than the voice speaking
    /// -- measured 4:1 spreads inside one page, and one continuous sentence
    /// split across two cells of the same bubble solved at 44px and 19px. A
    /// reader reads a size change as emphasis, so incidental ones read as
    /// shouting.
    ///
    /// Only an upper bound is on offer: auto-fit already returns the *largest*
    /// size that fits, so a small balloon is small because nothing bigger fits
    /// and no page-level rule can raise it. Coherence is therefore bought by
    /// bringing the outliers down. See `Frame::reconcile_text_sizes`.
    pub size_coherence: Option<f32>,
    /// Shrink a text layer that would be SET on top of another one, rather than
    /// letting the two be painted over each other. `false` leaves every layer at
    /// the size auto-fit solved.
    ///
    /// This is the only rule in the renderer that can see one layer from
    /// another, and it is the reason it lives here rather than in detection.
    /// Detection knows bounding boxes, and bounding-box overlap was measured and
    /// retired as a collision predictor: a fit box is about four times the area
    /// of the text auto-fit ends up setting inside it, so two boxes routinely
    /// overlap while the ink comes nowhere near. After the layout is solved the
    /// real extents exist -- `VisualText::rendered_bounds` -- and on 40 manga
    /// pages the same ink rule fires on 67 box pairs and 7 placed pairs, with a
    /// five-judge vision panel finding all 7 genuine and none of the 60 discarded
    /// pairs a real collision.
    ///
    /// **Shrink rather than drop, because a dropped layer cannot be recovered.**
    /// By the time layout runs the inpainter has already erased the artwork under
    /// the text, so removing a layer leaves a blank fill where the drawing was.
    /// Shrinking keeps every word on the page. See `Frame::reconcile_text_collisions`.
    pub collision_relief: bool,
    /// Anchor a cut balloon's lettering at the source ink's own center instead
    /// of the center of the visible part.
    ///
    /// A webtoon slice boundary can cut a balloon whose text it does not cut.
    /// The balloon's segmentation mask is exactly slice-sized, so the fit frame
    /// of a cut balloon ends at the canvas edge, and centering in it drifts the
    /// block away from the cut -- measured on a 179-slice chapter: 23 of 24
    /// cut-crossing dialogue balloons letter displaced away from the cut,
    /// median 77.8 px, while the block sits within 1 px of the fit center. The
    /// source ink center is the one position known to read as centered on the
    /// JOINED page: un-clipped balloons letter a median 6.6 px from it, clipped
    /// ones 46.3 px. `false` keeps every page byte-identical to before the flag
    /// existed. See `edge_anchored_center_y` in the compositor.
    pub edge_anchored_lettering: bool,
}

impl Default for RenderTheme {
    fn default() -> Self {
        Self {
            font_families: vec!["CCWildWords".to_owned(), "Arial".to_owned()],
            font_size: 24.0,
            minimum_font_size: 9.0,
            text_color: [0, 0, 0, 255],
            text_stroke: None,
            line_height: 1.2,
            letter_spacing: 0.0,
            word_spacing: 0.0,
            text_inset: [4.0; 4],
            vertical_alignment: VerticalAlignment::Center,
            hyphenation: None,
            size_coherence: None,
            collision_relief: false,
            edge_anchored_lettering: false,
        }
    }
}

/// Everything needed to deterministically render one page revision.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderRequest {
    pub page: EntityId,
    /// Page asset roles in preference order. An empty list renders transparently.
    pub base_assets: Vec<AssetRole>,
    /// Asset role used by image entities.
    pub image_asset: AssetRole,
    pub include_images: bool,
    /// Restricts text composition to specific entities while retaining page context.
    pub text_entities: Option<BTreeSet<EntityId>>,
    pub presentation: LayerPresentation,
    pub fallback_to_source_text: bool,
    pub theme: RenderTheme,
}

impl RenderRequest {
    #[must_use]
    pub fn new(page: EntityId) -> Self {
        Self {
            page,
            base_assets: vec![asset_role("source")],
            image_asset: asset_role("source"),
            include_images: true,
            text_entities: None,
            presentation: LayerPresentation::Resolved,
            fallback_to_source_text: true,
            theme: RenderTheme::default(),
        }
    }

    #[must_use]
    pub fn transparent(page: EntityId) -> Self {
        let mut request = Self::new(page);
        request.base_assets.clear();
        request
    }
}

fn asset_role(value: &str) -> AssetRole {
    AssetRole::new(value).expect("the built-in asset role is valid")
}
