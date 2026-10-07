//! Text layout and Vello glyph recording.

use anyhow::Result;
use vello::{
    FontEmbolden, Glyph, Scene,
    kurbo::{Affine, Diagonal2, Join, Line, Rect, Stroke},
    peniko::{Fill, Gradient},
};

use crate::{
    Error, HyphenationPolicy, LayoutRun, RenderDiagnostic, RenderTheme, Result as RenderResult,
    TextLayout, VerticalAlignment, WritingMode,
    bubble::LayoutBox,
    compositor::{RenderBounds, TextLayer},
    fonts::{Fonts, font_key},
    rasterizer::rgba,
    scene_renderer::{RenderedLayer, VisualLayer, VisualLayerKind, VisualText},
    script::is_cjk_text,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeOptions {
    pub color: [u8; 4],
    pub width_px: f32,
}

/// How thick a strike mark is, as a fraction of the solved font size.
///
/// Measured against the author's own marks rather than chosen: on one test
/// page the drawn strike is 12-13 px against a solved font of 73.9, and on
/// another 18 px against 59.8 -- 17% and 30%. The licensed release draws its own at
/// ~1.0% of page width. This sits at the thin end of that range: a strike
/// that reads as a deliberate cancellation rather than a smear, and one that
/// still leaves the struck word legible, which is the entire point of the
/// device.
const STRIKE_WIDTH_RATIO: f32 = 0.15;

/// Below this a strike is an artefact rather than a mark. Guards the tiny-font
/// end, where the ratio above would ask for a sub-pixel stroke.
const STRIKE_MIN_WIDTH_PX: f32 = 2.0;

/// How far above the baseline a strike sits, as a fraction of the font size.
///
/// Cap height runs about 0.7 of the size in the faces this letters with, so
/// ~0.32 puts the mark just under the middle of the capitals -- the licensed
/// reference draws its own through the upper-to-middle third of the cap
/// height, which is what a strike has to do to read as a cancellation instead
/// of an underline.
const STRIKE_BASELINE_RATIO: f32 = 0.32;

/// Paint options used when recording one laid-out text run into a Vello scene.
#[derive(Debug, Clone)]
pub struct TextRenderOptions {
    pub color: [u8; 4],
    /// A second fill colour: the fill letters as a linear gradient from
    /// `color` at the layout's top to this at its bottom, in the layout's own
    /// frame -- so it tilts with the run's transform exactly as the glyphs
    /// do. `None` is a solid fill, byte-identical to before.
    pub fill_gradient_to: Option<[u8; 4]>,
    /// Draw a strike-through mark across the run in this colour, after the
    /// glyphs. `None` draws nothing.
    pub strike_color: Option<[u8; 4]>,
    pub hint_glyphs: bool,
    pub padding: f32,
    pub baseline_shift: f32,
    pub stroke: Option<StrokeOptions>,
}

impl Default for TextRenderOptions {
    fn default() -> Self {
        Self {
            color: [0, 0, 0, 255],
            fill_gradient_to: None,
            strike_color: None,
            hint_glyphs: true,
            padding: 0.0,
            baseline_shift: 0.0,
            stroke: None,
        }
    }
}

/// Shapes text and records the resulting glyphs into vector scenes.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextRenderer;

impl TextRenderer {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub fn layout<'a>(&self, builder: &TextLayout<'a>, text: &str) -> Result<LayoutRun<'a>> {
        builder.run(text)
    }

    pub fn render(
        &self,
        scene: &mut Scene,
        layout: &LayoutRun<'_>,
        writing_mode: WritingMode,
        options: &TextRenderOptions,
        transform: Affine,
    ) {
        if let Some(stroke) = options
            .stroke
            .filter(|stroke| stroke.width_px > 0.0 && stroke.color[3] > 0)
        {
            draw_layout(
                scene,
                layout,
                writing_mode,
                options,
                transform,
                DrawStyle::Stroke(stroke),
            );
        }
        draw_layout(
            scene,
            layout,
            writing_mode,
            options,
            transform,
            DrawStyle::Fill,
        );
    }

    pub(crate) fn render_layer(
        &self,
        layer: &TextLayer,
        fonts: &Fonts,
        theme: &RenderTheme,
    ) -> RenderResult<RenderedLayer> {
        let is_bubble_text = layer.balloon_contour.is_some();
        let bounds = if is_bubble_text {
            inset(layer.bounds, theme.text_inset)
        } else {
            layer.bounds
        };
        if bounds.width <= 0.0 || bounds.height <= 0.0 {
            return Err(Error::invalid(format!(
                "text inset leaves no layout area for entity {}",
                layer.entity
            )));
        }
        let fonts = fonts
            .resolve(
                layer.preferred_font.as_deref(),
                layer.font_weight,
                &theme.font_families,
                &layer.text,
                layer
                    .language
                    .as_ref()
                    .map(koharu_scene::LanguageTag::as_str),
            )
            .map_err(|source| Error::Font {
                entity: layer.entity,
                source,
            })?;
        // The auto-fit ceiling is named rather than inlined: it had been wrong in two
        // opposite directions and no test failed either time.
        // See `auto_fit_ceiling` for both, and for the tests that now pin each end.
        let maximum = auto_fit_ceiling(
            layer.auto_fit,
            layer.font_size,
            layer.writing_mode.is_vertical(),
            bounds.width,
            bounds.height,
        );
        // A ceiling the page imposed after seeing every balloon solve alone. It
        // only ever lowers the search range, so the layout still fits by exactly
        // the same test -- a capped balloon is a balloon with more air in it,
        // never one that now overflows.
        let maximum = match layer.size_ceiling {
            Some(ceiling) if layer.auto_fit && !layer.point_text => maximum.min(ceiling.max(0.5)),
            _ => maximum,
        };
        let minimum = theme.minimum_font_size.min(maximum);
        let mut layout = TextLayout::new(&fonts[0])
            .with_fallback_fonts(&fonts[1..])
            .with_writing_mode(layer.writing_mode)
            .with_alignment(layer.alignment)
            .with_line_height(theme.line_height)
            .with_spacing(theme.letter_spacing, theme.word_spacing)
            .with_compact_emphasis_punctuation(
                is_cjk_text(&layer.text)
                    || layer
                        .language
                        .as_ref()
                        .is_some_and(|language| is_cjk_language(language.as_str())),
            );
        if !layer.point_text {
            layout = layout
                .with_max_width(bounds.width)
                .with_max_height(bounds.height);
        }
        if let Some(contour) = &layer.balloon_contour {
            let [top, _, _, left] = theme.text_inset;
            layout = layout
                .with_comic_balloon(
                    bounds.width,
                    bounds.height,
                    contour.iter().map(|&(x, y)| (x - left, y - top)).collect(),
                    match theme.vertical_alignment {
                        VerticalAlignment::Top => 0.0,
                        VerticalAlignment::Center => 0.5,
                        VerticalAlignment::Bottom => 1.0,
                    },
                    theme.text_inset.into_iter().fold(0.0, f32::max),
                )
                // Canvas coordinates to balloon-local: `bounds` is already the
                // inset frame, so its origin is the balloon solver's (0, 0).
                .with_balloon_anchor_y(layer.anchor_center_y.map(|center| center - bounds.y));
        }
        if let Some(language) = &layer.language {
            layout = layout.with_hyphenation_language_tag(language.as_str());
            if is_bubble_text
                && layer.writing_mode == WritingMode::Horizontal
                && is_english(language.as_str())
            {
                layout = layout.with_hyphenation_policy(HyphenationPolicy::LastResort);
            }
        }
        // Applied last so it beats the rule above, and outside the `language`
        // block so it still reaches a layer whose language was never resolved --
        // which is every layer the built-in rule silently leaves at `Normal`.
        if let Some(policy) = theme.hyphenation {
            layout = layout.with_hyphenation_policy(policy);
        }
        let layout = if layer.auto_fit && !layer.point_text {
            layout
                .with_max_font_size(maximum)
                .with_min_font_size(minimum)
                .with_min_line_height(1.0)
        } else {
            layout.with_font_size(layer.font_size.unwrap_or(maximum))
        };
        let layout = self
            .layout(&layout, &layer.text)
            .map_err(|source| Error::Layout {
                entity: layer.entity,
                source,
            })?;
        let (mut x, mut y) = if layer.point_text {
            (bounds.x, bounds.y)
        } else {
            placement(
                bounds,
                layout.width,
                layout.height,
                theme.vertical_alignment,
            )
        };
        x += layout.placement_offset_x();
        y += layout.placement_offset_y();
        let layout_rect = Rect::new(
            f64::from(x),
            f64::from(y),
            f64::from(x + layout.width),
            f64::from(y + layout.height),
        );
        let angle = f64::from(layer.angle_degrees).to_radians();
        let center = layout_rect.center();
        let rotation = Affine::rotate_about(angle, center);
        let transform =
            Affine::translate((f64::from(x), f64::from(y))).then_rotate_about(angle, center);
        let color = with_alpha(
            layer.foreground_color.unwrap_or(theme.text_color),
            layer.opacity,
        );
        let mut options = TextRenderOptions {
            color,
            fill_gradient_to: layer
                .fill_gradient_to
                .map(|value| with_alpha(value, layer.opacity)),
            strike_color: layer
                .strike_color
                .map(|value| with_alpha(value, layer.opacity)),
            stroke: None,
            ..TextRenderOptions::default()
        };
        let mut scene = Scene::new();
        if let Some(mut stroke) = layer.stroke.or(theme.text_stroke) {
            stroke.color = with_alpha(stroke.color, layer.opacity);
            /* A stroke width is authored against the size the layer asked for,
             * but auto-fit solves the layout first and routinely renders far
             * smaller -- a translation is longer than the source it replaces, so
             * `layout.font_size` can be a third of `layer.font_size`. Carrying an
             * absolute width across that gap triples the halo relative to the
             * stem it is outlining, and since `render` draws
             * `Stroke::new(width_px * 2.0)`, that is enough to close the counter
             * of every `a`, `e` and `g` and hand back a solid blob. Scale it by
             * the ratio actually used, so the halo stays the fraction of the
             * glyph it was authored as. */
            if let Some(authored) = layer.font_size
                && authored > 0.0
                && layout.font_size > 0.0
            {
                stroke.width_px *= layout.font_size / authored;
            }
            options.stroke = Some(stroke);
        }
        self.render(&mut scene, &layout, layer.writing_mode, &options, transform);
        /* THE STRIKE, DRAWN LAST AND INTO THIS LAYER'S OWN SCENE.
         *
         * After `self.render` above, so it lands ON the glyphs rather than
         * under them -- which is the whole reason the mark is re-drawn instead
         * of preserved. The author's own strike is ink on the page and the
         * English is composited over the raster, so a surviving mark sits
         * behind the replacement text forever; measured on a test page, only
         * 6.6% of the lost strike was ever erased and the rest is simply
         * painted over.
         *
         * Inside this scene rather than a page-level pass, and that is enough:
         * a strike only ever has to be above its OWN text, and this layer is
         * appended to the page after the raster like every other text layer.
         *
         * It shares `transform` with the glyphs, which is what makes it
         * angle-agnostic. The line is computed in the layout's own unrotated
         * frame and carried through the same rotation the text got, so on a
         * turned column it comes out running down the words instead of across
         * the page. Nothing here reads `angle_degrees`: this never chooses an
         * angle to letter at, it only follows the one already chosen. */
        if let Some(strike) = options.strike_color {
            let width = (layout.font_size * STRIKE_WIDTH_RATIO).max(STRIKE_MIN_WIDTH_PX);
            let offset = f64::from(layout.font_size * STRIKE_BASELINE_RATIO);
            /* ONE STROKE PER LINE, off each line's OWN baseline and advance.
             *
             * A single stroke through the layout's mid-height was the first
             * build, and the render refuted it: on a two-line block the middle
             * of the BLOCK is the gap BETWEEN the lines, so the mark crossed
             * nothing and merely underlined the first line. Both test pages
             * showed it.
             *
             * `baseline` and `advance` are the same numbers `draw_layout` walks
             * to place the glyphs, so the mark cannot drift from the text it
             * cancels however the layout was solved. */
            for line in &layout.lines {
                let (x, y) = line.baseline;
                let (x, y, advance) = (f64::from(x), f64::from(y), f64::from(line.advance));
                let (start, end) = match layer.writing_mode {
                    WritingMode::VerticalRl | WritingMode::VerticalLr => {
                        ((x + offset, y), (x + offset, y + advance))
                    }
                    _ => ((x, y - offset), (x + advance, y - offset)),
                };
                scene.stroke(
                    &Stroke::new(f64::from(width)),
                    transform,
                    rgba(strike),
                    None,
                    &Line::new(start, end),
                );
            }
        }
        let rendered_bounds = rotation.transform_rect_bbox(layout_rect);
        let mut diagnostics = Vec::new();
        if layout.font_size + f32::EPSILON < theme.minimum_font_size {
            diagnostics.push(RenderDiagnostic::TextBelowReadableSize {
                entity: layer.entity,
                font_size: layout.font_size,
                minimum_font_size: theme.minimum_font_size,
            });
        }
        if layout.overflowed() {
            diagnostics.push(RenderDiagnostic::TextOverflow {
                entity: layer.entity,
                available: bounds.into(),
                actual_width: layout.width,
                actual_height: layout.height,
                font_size: layout.font_size,
            });
        }
        Ok(RenderedLayer {
            scene,
            layer: VisualLayer {
                entity: layer.entity,
                kind: VisualLayerKind::Text,
                name: None,
                bounds: RenderBounds {
                    x: rendered_bounds.x0 as f32,
                    y: rendered_bounds.y0 as f32,
                    width: rendered_bounds.width() as f32,
                    height: rendered_bounds.height() as f32,
                },
                font_size: Some(layout.font_size),
                text: Some(VisualText {
                    text: layer.text.clone(),
                    language: layer.language.clone(),
                    rendered_bounds: RenderBounds {
                        x,
                        y,
                        width: layout.width,
                        height: layout.height,
                    },
                    layout_bounds: if layer.point_text {
                        RenderBounds {
                            x,
                            y,
                            width: layout.width,
                            height: layout.height,
                        }
                    } else {
                        bounds.into()
                    },
                    post_script_fonts: fonts
                        .iter()
                        .map(|font| font.post_script_name().to_owned())
                        .collect(),
                    font_size: layout.font_size,
                    color,
                    alignment: layer.alignment,
                    writing_mode: layer.writing_mode,
                    angle_degrees: layer.angle_degrees,
                }),
            },
            diagnostics,
        })
    }
}

fn is_english(language: &str) -> bool {
    language
        .split(['-', '_'])
        .next()
        .is_some_and(|primary| primary.eq_ignore_ascii_case("en"))
}

fn is_cjk_language(language: &str) -> bool {
    language
        .split(['-', '_'])
        .next()
        .is_some_and(|primary| matches!(primary.to_ascii_lowercase().as_str(), "ja" | "ko" | "zh"))
}

fn inset(rect: LayoutBox, [top, right, bottom, left]: [f32; 4]) -> LayoutBox {
    LayoutBox {
        x: rect.x + left,
        y: rect.y + top,
        width: (rect.width - left - right).max(0.0),
        height: (rect.height - top - bottom).max(0.0),
    }
}

fn placement(rect: LayoutBox, width: f32, height: f32, vertical: VerticalAlignment) -> (f32, f32) {
    let x = rect.x + (rect.width - width) * 0.5;
    let remaining = rect.height - height;
    let y = rect.y
        + match vertical {
            VerticalAlignment::Top => 0.0,
            VerticalAlignment::Center => remaining * 0.5,
            VerticalAlignment::Bottom => remaining,
        };
    (x, y)
}

fn with_alpha(mut color: [u8; 4], opacity: f32) -> [u8; 4] {
    color[3] = (f32::from(color[3]) * opacity.clamp(0.0, 1.0)).round() as u8;
    color
}

#[derive(Clone, Copy)]
enum DrawStyle {
    Stroke(crate::StrokeOptions),
    Fill,
}

fn draw_layout(
    scene: &mut Scene,
    layout: &LayoutRun<'_>,
    writing_mode: WritingMode,
    options: &TextRenderOptions,
    transform: Affine,
    style: DrawStyle,
) {
    for line in &layout.lines {
        let (baseline_x, baseline_y) = match writing_mode {
            WritingMode::Horizontal | WritingMode::VerticalRl | WritingMode::VerticalLr => {
                line.baseline
            }
        };
        let mut pen_x = 0.0;
        let mut pen_y = 0.0;
        let mut start = 0;

        while start < line.glyphs.len() {
            let font = line.glyphs[start].font;
            let key = font_key(font);
            let mut end = start + 1;
            while end < line.glyphs.len() && font_key(line.glyphs[end].font) == key {
                end += 1;
            }

            let mut glyphs = Vec::with_capacity(end - start);
            for glyph in &line.glyphs[start..end] {
                glyphs.push(Glyph {
                    id: glyph.glyph_id,
                    x: options.padding + baseline_x + pen_x + glyph.x_offset,
                    y: options.padding + baseline_y + pen_y
                        - glyph.y_offset
                        - options.baseline_shift,
                });
                pen_x += glyph.x_advance;
                pen_y -= glyph.y_advance;
            }

            let font_data = font.vello_data();
            let normalized_coords = font.normalized_coords();
            let mut run = scene
                .draw_glyphs(&font_data)
                .font_size(layout.font_size)
                .transform(transform)
                .hint(options.hint_glyphs);
            if !normalized_coords.is_empty() {
                run = run.normalized_coords(normalized_coords);
            }
            if let Some(angle) = font.synthetic_skew() {
                run = run
                    .glyph_transform(Some(Affine::skew(-(angle.to_radians().tan() as f64), 0.0)));
            }
            if font.synthetic_bold() {
                run = run.font_embolden(FontEmbolden::new(Diagonal2::new(1.0, 1.0)));
            }

            match style {
                DrawStyle::Fill => match options.fill_gradient_to {
                    // The gradient runs the layout's own top to bottom in
                    // run-local coordinates, so the run transform tilts it
                    // with the glyphs -- a rotated scream's ramp follows the
                    // cascade, not the page.
                    Some(tail) => run
                        .brush(
                            &Gradient::new_linear(
                                (0.0, f64::from(options.padding)),
                                (0.0, f64::from(options.padding + layout.height)),
                            )
                            .with_stops([(0.0, rgba(options.color)), (1.0, rgba(tail))]),
                        )
                        .draw(Fill::NonZero, glyphs.into_iter()),
                    None => run
                        .brush(rgba(options.color))
                        .draw(Fill::NonZero, glyphs.into_iter()),
                },
                DrawStyle::Stroke(stroke) => {
                    let outline =
                        Stroke::new((stroke.width_px * 2.0) as f64).with_join(Join::Round);
                    run.brush(rgba(stroke.color))
                        .draw(&outline, glyphs.into_iter());
                }
            }
            start = end;
        }
    }
}

/// The span an auto-fit search may grow into: the box, on the axis the text runs along.
fn auto_fit_ceiling_span(vertical: bool, width: f32, height: f32) -> f32 {
    if vertical { height } else { width }
}

/// The auto-fit CEILING for a layer -- the largest size the search may consider.
///
/// Named and pulled out of `render_layer` because it had been wrong in two opposite
/// directions and no test failed either time.
///
///  - It used to be `layer.font_size` for free-standing text: the **source Japanese glyph
///    height**. A vertical kana measured 9 px tall gave a search with exactly one
///    candidate and forced the English to 9 px. Measured, `solved == authored` to the
///    digit, and no amount of extra box WIDTH could move it.
///  - Replacing it with the flat `theme.font_size` (24.0) fixed that end and **capped the
///    other**: on a test chapter's first page a chapter title fell 41.5 -> 24.0, and 45 of
///    48 free-text layers in the chapter piled onto 24.0 exactly.
///
/// What ships is the rule a balloon has always used: **the space available**. A title gets
/// a title's ceiling and an interjection gets a legible one, because both are sized by
/// their own box rather than by a constant or by the source glyphs.
///
/// A layer that is NOT auto-fitting keeps its authored size, which is what point text and
/// explicitly-sized layers rely on.
fn auto_fit_ceiling(
    auto_fit: bool,
    authored: Option<f32>,
    vertical: bool,
    width: f32,
    height: f32,
) -> f32 {
    let span = auto_fit_ceiling_span(vertical, width, height);
    if auto_fit {
        span
    } else {
        authored.unwrap_or(span)
    }
}

#[cfg(test)]
mod tests {
    use super::{auto_fit_ceiling, auto_fit_ceiling_span};

    #[test]
    fn an_auto_fit_layer_is_ceilinged_by_its_box_and_not_by_the_source_glyphs() {
        // The measured failure: `Eh?` in a 60 x 25.4 box whose Japanese was
        // measured at 9 px. Honouring the authored size gave the search ONE candidate
        // and lettered the English at 9 px.
        let ceiling = auto_fit_ceiling(true, Some(9.0), false, 60.0, 25.4);
        assert_eq!(ceiling, 60.0, "the box span is the ceiling, not the 9px glyph height");
        assert!(ceiling > 9.0, "a 9px ceiling is the defect this pins");
    }

    #[test]
    fn an_auto_fit_layer_is_not_capped_by_the_theme_size() {
        // The opposite regression: a chapter title
        // authored at 51 px in a 290.6 x 51.3 box came back at 24.0 -- the theme's
        // `font_size` -- when the ceiling was a flat constant. 24.0 must NOT appear here.
        let ceiling = auto_fit_ceiling(true, Some(51.0), false, 290.6, 51.3);
        assert_eq!(ceiling, 290.6);
        assert!(
            ceiling > 24.0,
            "a flat theme-size cap crushes display text: {ceiling}"
        );
    }

    #[test]
    fn vertical_text_is_ceilinged_by_height_and_horizontal_by_width() {
        assert_eq!(auto_fit_ceiling_span(true, 120.0, 400.0), 400.0);
        assert_eq!(auto_fit_ceiling_span(false, 120.0, 400.0), 120.0);
        assert_eq!(auto_fit_ceiling(true, None, true, 120.0, 400.0), 400.0);
        assert_eq!(auto_fit_ceiling(true, None, false, 120.0, 400.0), 120.0);
    }

    #[test]
    fn a_layer_that_is_not_auto_fitting_keeps_its_authored_size() {
        // Point text and explicitly-sized layers depend on this; the ceiling change must
        // not reach them.
        assert_eq!(auto_fit_ceiling(false, Some(18.0), false, 300.0, 40.0), 18.0);
        // ...and with nothing authored there is still a sane answer.
        assert_eq!(auto_fit_ceiling(false, None, false, 300.0, 40.0), 300.0);
    }
}
