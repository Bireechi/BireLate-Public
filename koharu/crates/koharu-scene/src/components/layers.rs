use std::collections::BTreeMap;

use revision::revisioned;
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::{
    Error, Result,
    component::{Component, ValidationContext},
    id::validate_namespaced,
};

use super::Origin;

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum RasterLayerKind {
    Cleanup,
    Paint,
}

/// A full-page transparent pixel layer composited above the page source.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct RasterLayer {
    pub origin: Origin,
    pub name: String,
    pub kind: RasterLayerKind,
}

impl Component for RasterLayer {
    const KIND: &'static str = "dev.koharu.layer.raster";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        self.origin.validate()?;
        if self.name.is_empty() || self.name.len() > 4096 || self.name.contains('\0') {
            return Err(Error::invalid("raster layer name is invalid"));
        }
        Ok(())
    }

    fn origin(&self) -> Option<&Origin> {
        Some(&self.origin)
    }

    fn set_origin(&mut self, origin: Origin) -> bool {
        self.origin = origin;
        true
    }
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum TextLayoutKind {
    Point,
    Paragraph,
}

#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct TextLayout {
    pub origin: Origin,
    pub kind: TextLayoutKind,
}

impl Component for TextLayout {
    const KIND: &'static str = "dev.koharu.layer.text";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        self.origin.validate()
    }

    fn origin(&self) -> Option<&Origin> {
        Some(&self.origin)
    }

    fn set_origin(&mut self, origin: Origin) -> bool {
        self.origin = origin;
        true
    }
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
pub enum TextAlignment {
    Start,
    Center,
    End,
    Justify,
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
pub enum WritingMode {
    Horizontal,
    Vertical,
}

/// `Typography.extensions` key carrying a second fill colour as `"r,g,b,a"`:
/// the renderer letters the fill as a linear gradient from `color` at the
/// layout's top to this value at its bottom, tilting with the layer's frame.
/// An extension rather than a field so the component schema and every stored
/// scene stay byte-compatible; a reader that predates it ignores it.
pub const FILL_GRADIENT_TO_EXTENSION: &str = "dev.koharu.text.fill-gradient-to";

/// `Typography.extensions` key asking for a strike-through mark across the
/// layer, as `"r,g,b"` or `"r,g,b,a"`: the renderer draws one stroke along the
/// reading direction, through the middle of the laid-out text, in this colour.
///
/// It is drawn AFTER the glyphs and inside the same layer, which is the whole
/// point of it. The author's own strike is ink on the page, so the inpainter
/// either erases it or the English is composited on top of it -- measured on
/// one test chapter at 75% erased on one page and 93% overpainted on another. A mark
/// that must cross the replacement text cannot be a surviving piece of the
/// original; it has to be re-drawn over what replaced it.
///
/// An extension rather than a field for `FILL_GRADIENT_TO_EXTENSION`'s reason:
/// the component schema and every stored scene stay byte-compatible, and a
/// reader that predates it ignores the key.
pub const STRIKE_COLOR_EXTENSION: &str = "dev.koharu.text.strike";

#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct Typography {
    pub origin: Origin,
    pub preferred_font: Option<String>,
    pub font_weight: Option<u16>,
    pub size: Option<f32>,
    pub auto_fit: bool,
    pub color: Option<[u8; 4]>,
    pub stroke_color: Option<[u8; 4]>,
    pub stroke_width: Option<f32>,
    pub alignment: Option<TextAlignment>,
    pub writing_mode: Option<WritingMode>,
    pub extensions: BTreeMap<String, String>,
}

impl Component for Typography {
    const KIND: &'static str = "dev.koharu.text.typography";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        if self
            .preferred_font
            .as_ref()
            .is_some_and(|font| font.len() > 4096)
            || self
                .font_weight
                .is_some_and(|weight| !(1..=1000).contains(&weight))
            || self
                .size
                .is_some_and(|size| !size.is_finite() || size <= 0.0)
            || (!self.auto_fit && self.size.is_none())
            || self
                .stroke_width
                .is_some_and(|width| !width.is_finite() || width < 0.0)
        {
            return Err(Error::invalid("typography intent is invalid"));
        }
        self.origin.validate()?;
        if self.extensions.len() > 1024
            || self.extensions.iter().any(|(key, value)| {
                validate_namespaced(key, "typography extension").is_err()
                    || value.len() > 64 * 1024
                    || value.contains('\0')
            })
        {
            return Err(Error::invalid("typography extensions are invalid"));
        }
        Ok(())
    }

    fn origin(&self) -> Option<&Origin> {
        Some(&self.origin)
    }

    fn set_origin(&mut self, origin: Origin) -> bool {
        self.origin = origin;
        true
    }
}
