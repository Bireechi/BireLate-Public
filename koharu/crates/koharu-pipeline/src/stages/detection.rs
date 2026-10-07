use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use image::{DynamicImage, GrayImage, ImageFormat, Luma, RgbImage};
use imageproc::{
    contours::{BorderType, find_contours_with_threshold},
    contrast::otsu_level,
    distance_transform::Norm,
    geometry::{approximate_polygon_dp, arc_length, contour_area},
    morphology::dilate,
    region_labelling::{Connectivity, connected_components},
};
use koharu_ml::{
    koharu_layout_rfdetr_seg_2xl::{
        KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask, KoharuLayoutRFDetrSeg2XL,
        KoharuLayoutThresholds,
    },
    manga_text_mask::{MangaTextMaskCleaningOptions, MangaTextMaskGenerator},
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, BubbleRegion, DetectionAnalysis, DetectionLabel,
    EntityId, EntityOrigin, FitsTo, Generation, Geometry, Inside, Origin, PanelRegion, Point,
    Presents, RecognizedFrom, Region, RegionKind, RegionSpec, RemovePolicy,
    STRIKE_COLOR_EXTENSION, TextLayout, TextLayoutKind, TextRegion, TextRole, Typography,
    WritingMode,
};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::{ModelRef, StageInput, StageProcessor, finish, generation};
use crate::{DetectionModel, EdgeHint, ModelCell, PageEdge, Progress};

const MODEL_ID: &str = "mayocream/koharu-layout-rfdetr-seg-2xl-1152";
const MODEL_NAME: &str = "koharu-layout-rfdetr-seg-2xl";
const PRODUCER: &str = "dev.koharu.pipeline.detection";
const ANGLE_SNAP_DEGREES: f32 = 3.0;
const ANGLE_SEARCH_HALF_STEPS: i32 = 90;
const ANGLE_SEARCH_STEP_DEGREES: f64 = 0.5;
const COLOR_SNAP_CHROMA: u8 = 24;
/// The least chroma (`max - min` over RGB) a strike mark's own colour may have.
///
/// Set to `COLOR_SNAP_CHROMA`'s value and NOT to that constant, deliberately:
/// the two mean the same thing today -- "this colour is grey enough that the
/// pipeline treats it as black or white" -- and a future lettering change to the
/// snap must not silently move what counts as a drawn mark.
const STRIKE_MIN_CHROMA: u8 = 24;
const COLOR_SNAP_DARK_LUMINANCE: u16 = 64;
const COLOR_SNAP_LIGHT_LUMINANCE: u16 = 191;
const NMS_CONTAINMENT_THRESHOLD: f32 = 0.9;
/// Two boxes "share" an axis when their extents on it are within this ratio.
/// Of the two measured exhibits, one pair shares width at 1.0433 and the other
/// shares height at 1.0119 -- both well inside 1.15.
const AXIS_SHARE_MAX_RATIO: f32 = 1.15;
/// ...and "differ" on an axis at or above this ratio. The first pair differs
/// 2.2664 in height, the second 1.6203 in width, so both clear 1.30 with margin. The
/// 1.15..1.30 gap is a deliberate dead band: a pair that neither shares nor
/// clearly differs gets today's answer.
const AXIS_DIFFER_MIN_RATIO: f32 = 1.30;
/// A candidate may not evict a kept box it is much less confident than. Both
/// exhibits pass with room (0.2383/0.2715 = 0.878; 0.2832/0.3730 = 0.759); the
/// guard exists to stop a barely-admitted column evicting a confident
/// merged-balloon box, which is the manga failure mode a census of manga pages
/// was run to bound.
const AXIS_TIE_BREAK_MIN_SCORE_RATIO: f32 = 0.60;
const CELL_MINIMUM_EXTENT: f32 = 1.0;
/// Least width-to-height ratio a free-standing *vertical* text's layout box is
/// grown to.
///
/// Such a text has no bubble to inherit a frame from, so it lays its
/// translation out in its own tight bounding box -- for vertical Japanese, a
/// column perhaps 40px wide and 300px tall. `compositor.rs` `resolve_writing_mode`
/// returns `Horizontal` for the English that replaces it, auto-fit binary-searches
/// down to `theme.minimum_font_size`, and `layout.rs` `run_auto` returns the
/// overflowing layout rather than erroring. The reader gets a stack of ~10px
/// fragments. Bubble text never hits this because it inherits the balloon.
const FREE_TEXT_MIN_ASPECT: f32 = 0.6;
/// Least ABSOLUTE width a *short* free-standing text's box is grown to, and the
/// height at or below which that applies.
///
/// The ratio above is height-driven, and that is the whole of its trouble: it
/// gives a tall column far too much -- a 204.5px column 1692.5px tall was
/// measured handed a **1015.5px** frame, 4.97x -- and a short one almost nothing. A
/// 21.7 x 36.1 box keeps 21.7, because `36.1 * 0.6` is 21.66 and the `max` picks
/// the text's own width, while the English needs 34.4. So the rule fires, does
/// nothing, and still leaves the reader the ~10px fragments the comment above
/// promises to prevent.
///
/// MEASURED BEFORE IT WAS WRITTEN, over 2,860 free-text layers on disk. Of the
/// layers pinned at the floor *and* overflowing, the ratio they need is a median
/// of **0.94** against the 0.6 that ships. Raising the ratio to 0.95 would reach
/// **1,214 boxes -- 43% of every free-text box in the corpus** -- at a median
/// 1.58x and a worst 8.98x. A height-gated absolute floor is far more targeted.
///
/// **THE VALUE WAS 40.0 FIRST AND THAT WAS SIZED AGAINST THE WRONG TARGET.** 40 px
/// is the width the failing text needed *at the 9 px floor*, so it bought fitting
/// and not legibility: rendered, it widened 7 boxes and raised the font on only
/// **3**. Overflow fell 6 -> 3 while text pinned at 9 px fell just 9 -> 8. The
/// number that matters is the size the box can then CARRY, and for a single
/// unbroken token -- which is 24 of the 29 measured failures -- that scales
/// linearly with width.
///
/// Over the 68 short boxes lettered at 12 px or less, the median size each floor
/// reaches, against a page whose own prevailing lettering is ~21.5 px:
///
///     40px -> 14.1    50px -> 17.6    60px -> 21.1    70px -> 24.0 (ceiling)
///
/// and the short free-text boxes each would widen, of 528:
///
///     40px -> 127 (1.48x)   50px -> 167 (1.67x)   60px -> 174 (2.00x)
///
/// **60 is the knee.** It matches the page's own size; 70 reaches the same layers
/// and only clips at the theme ceiling while widening more.
///
/// THE HEIGHT GATE IS THE LOAD-BEARING HALF. Without it this is the same
/// height-blind widening in another costume and would make that tall column worse.
/// At or below `FREE_TEXT_SHORT_BOX_HEIGHT` a box holds an interjection --
/// `*Gulp*`, `Eh?`, `Well...` -- not a column; above it nothing changes at all.
const FREE_TEXT_MIN_WIDTH: f32 = 60.0;
const FREE_TEXT_SHORT_BOX_HEIGHT: f32 = 60.0;
/// Glyph halo width as a fraction of the font size.
///
/// Bounded by the counters, not by taste. `TextRenderer::render` draws
/// `Stroke::new(width_px * 2.0)`, a centred stroke, so the halo reaches a full
/// `width_px` OUTWARD from every contour -- including the inner ones. The fill
/// pass is `Fill::NonZero`, so a counter the stroke has closed is not reopened
/// by it. Arial's `e` eye is about 0.12 em, so it survives only while
/// `width_px < 0.06 * size`, and stays legibly open below about half of that.
/// 0.12 closed it with 8px to spare at every size, since halo and counter scale
/// together -- not a small-text edge case but every page.
const FREE_TEXT_STROKE_RATIO: f32 = 0.03;
/// Floor for that halo, so a small glyph still gets a visible edge. Kept under
/// the same bound: at the 9.0 `minimum_font_size` the ratio alone would give
/// 0.27px, which disappears.
const FREE_TEXT_STROKE_MINIMUM_WIDTH: f32 = 0.5;
/// Least Rec.601 luma gap between glyph and local background for the halo to
/// stay the BACKGROUND's colour; under it the halo flips to the opposite tone
/// (white under dark ink, black under light). The measured case: black
/// glyphs took a (95,61,245) halo -- 65k of RGB
/// distance, which sails past the gate above, but luma 92 against fill luma 0,
/// so glyph, halo and the plate's darker band were all one TONE and the edge
/// dissolved exactly where the plate drifted toward the ink. A halo separates
/// by tone, and RGB distance cannot see tone.
///
/// 128 is half scale AND sits in the hole the chapter's own population leaves:
/// measured over all 73 lettered free-text regions of one test chapter, the
/// gaps run ...
/// 116.0, 117.6, 118.6, 120.8, 122.5 | 134.6, 135.3, 138.1, 147.0 ... -- the
/// dark-plate family below the line, every bright-page region above it keeping
/// today's halo untouched.
const FREE_TEXT_STROKE_TONE_GAP: f32 = 128.0;
/// Chebyshev erosion depth at which a pixel counts as CORE for the ink sample
/// -- applied twice, to the mask (whose eroded interior is the PAPER) and to
/// the differing set (whose eroded interior is the INK; see `infer_ink_core`).
/// At 2, the outermost ring -- anti-aliasing, and any drawn halo's own edge --
/// is excluded. Depth 3 was rejected because a 5px drawn stroke -- one measured
/// exhibit's thickness -- peaks at depth 3 and would keep almost nothing.
const INK_CORE_DEPTH: u32 = 2;
/// Least squared RGB distance from the region's own paper for a pixel to count
/// as ink-or-halo. The value is `text_stroke`'s old RGB gate reborn in a new
/// role: 48 per channel was measured as "the glyphs match the plate" there,
/// and "differs from the paper" is the same boundary read from the other side.
const INK_MIN_CONTRAST: u32 = 3 * 48 * 48;
/// Fewer core pixels than this and the ink sample abstains, leaving the legacy
/// contrast sample in force -- a hairline mask erodes to nothing and a median
/// over a handful of pixels is noise, not a measurement.
const INK_CORE_MINIMUM_PIXELS: usize = 12;
/// Drawn stroke thickness as a share of the measured glyph size at or above
/// which free-standing lettering takes a bold face. Set between the two
/// measured exhibits -- a heavy brush scream at 0.149 and a thin squiggle at
/// 0.05 -- and provisional until a corpus measurement
/// tightens it; the render adjudicates.
const HEAVY_INK_RATIO: f32 = 0.11;
/// The weight heavy drawn ink letters at. 700 rather than 900 because the
/// catalog face may not carry a true black and `fontique`'s synthetic embolden
/// fattens whatever it resolves; the counters close well before 900 lands.
pub(super) const HEAVY_INK_WEIGHT: u16 = 700;
/// Thinner measured ink than this abstains. At `INK_CORE_DEPTH` 2 a stroke
/// under ~8px erodes to a one-pixel centerline whose median is an
/// anti-aliased BLEND, not the ink -- measured shipping muddy grey over a
/// white-on-black caption and a black-on-white plate (checked in the rendered
/// pages), both strictly worse than the legacy sample. The drawn marks this
/// exists for clear it with room: a stroke-plus-glow measures ~11px, a
/// stroke-plus-rim ~18px.
const INK_SAMPLE_MINIMUM_THICKNESS_PX: f32 = 8.0;
/// The sampled ink must be at least this much DARKER than the paper (Rec.601)
/// or the sample abstains. The paper-then-ink split has no way to know which
/// of its two clusters is the ink when the mask's eroded interior lands on the
/// glyphs -- on a page of dense black-on-white text it did exactly that,
/// called the white gaps "ink", and lettered the page hollow (checked in the
/// rendered page). Dark-over-light is the one polarity
/// where the roles cannot have swapped, so it is the only one the override
/// ships; genuine light-on-dark ink -- inverted narration above all -- keeps
/// the legacy sample, which reads it correctly. One snap-chroma unit of
/// margin, so a near-tie never flips a page.
const INK_DARKER_THAN_PAPER_MARGIN: f32 = 24.0;
/// Least share of a detection blob the refined mask must cover before that blob
/// is sharpened rather than left as the dilated box.
const MINIMUM_REFINED_COVERAGE: f64 = 0.10;
/// The detector's class name for a drawn sound effect.
/// How far outside a detection's box the ink search looks, in pixels.
/// `content.py` uses 10; the same value here for the same reason -- a glyph's
/// anti-aliased skirt sits just outside the box the detector drew.
const INK_MASK_PAD: u32 = 10;
/// How far from its balloon's paper a pixel must be to count as INK, in mean-channel
/// luminance.
///
/// Generous on purpose. The two things it must separate are a balloon's paper and the
/// glyphs printed on it, which on real pages are near-white against near-black -- a
/// gap of roughly 200. Anything near the middle is an anti-aliased rim, and including
/// or excluding those does not decide the erase, because `write_mask` dilates the text
/// mask afterwards and closes them either way.
const SYNTHESISED_INK_LUMA_DELTA: u16 = 60;

/// How coarsely colours are binned when hunting a strike mark: 4 bits a channel,
/// 4,096 buckets.
///
/// **Measured, and 5 bits is too fine.** At 5 bits JPEG noise splits the drawn
/// stroke across neighbouring buckets and a test page that visibly carries one
/// produces no candidate at all, while another (whose stroke is half again as
/// thick) still fires. A detector that finds the fat instance and misses the thin
/// one is worse than none: the thin one is where the device is hardest to see.
const STRIKE_COLOR_BITS: u8 = 4;

/// Fewest pixels of one colour before a bucket is worth measuring. Shares
/// `SYNTHESISED_INK_MIN_PIXELS`' reasoning and its value.
const STRIKE_MIN_PIXELS: usize = 200;

/// How elongated a colour's footprint must be to be a stroke and not a blob.
///
/// The two drawn strikes measure 64.6 and 36.6. Every negative measured -- the
/// gloss column beside one of them, and three WHOLE pages including one carrying
/// 39,851 pixels of red artwork -- produces no bucket clearing 8 on all three
/// tests together. Set far below the positives rather than between the two
/// populations, because n=2 does not earn a tight fit.
const STRIKE_MIN_ASPECT: f32 = 8.0;

/// How much of its own bounding box a stroke's pixels must occupy.
///
/// A projection profile was found unable to pass this test, and the reason it
/// can pass here is SCOPE: that attempt failed at page scale, where a red art mass
/// dwarfs the mark. Inside one region the mark is the dominant object of its own
/// colour -- 0.51 and 0.58 measured -- while artwork is blobby and sparse within
/// any single bucket.
const STRIKE_MIN_FILL: f32 = 0.30;

/// How far along the region a stroke must run. A cancellation crosses what it
/// cancels; a short streak of colour is decoration. Measured 0.99 and 0.52.
const STRIKE_MIN_SPAN: f32 = 0.40;
/// How many ink pixels a textless balloon must hold before a region is synthesised
/// for it at all.
///
/// **It does NOT gate the known false positive.** One test page's spurious bubble
/// clears this floor with **795** ink pixels -- the character's own mouth line and collar edges
/// are real ink, just not glyphs. An absolute count cannot separate them;
/// [`SYNTHESISED_INK_MIN_FRACTION`] is what does. Kept as a cheap guard against a
/// degenerate mask, not as the discriminator it was described as.
///
/// It is also the WRONG SHAPE, measured: it rejects a real balloon on a test page
/// holding `!`, for having 195 ink pixels in a 2,217-pixel mask -- an ink
/// fraction of **0.0880**, one of the highest in the corpus. A small balloon is
/// penalised for being small. Recorded rather than changed here, because widening
/// it is a separate question from the false positive this gate was written for.
const SYNTHESISED_INK_MIN_PIXELS: usize = 200;
/// What share of a balloon's own mask must be ink before a region is synthesised.
///
/// **This is the gate on the false positive, and it is scale-free where the
/// absolute count above is not.** Derived from a census of **2,519 bubbles over 773
/// pages** across four test corpora.
///
/// # The separation it rests on
///
/// Twelve bubbles in that corpus hold no detected text, which is the only
/// population this floor gates. **All twelve were inspected in pixels** and
/// eleven are genuine balloons -- mostly an ellipsis or a small interjection the
/// text detector skips. The twelfth is a chin and a collar, no balloon at all.
///
///     the only known false positive        ink_frac 0.0213   <- rejected
///     the lowest GENUINE text-less balloon ink_frac 0.0495   <- kept
///
/// A 2.32x gap with nothing inside it. This sits at 0.035: **1.64x above the false
/// positive and 1.41x below the nearest true positive.**
///
/// # What it costs, and what it cannot promise
///
/// Of 2,495 bubbles that DO hold text -- known-real balloons, and the reference
/// distribution for the accept side -- three sit below this floor, **0.12%**. None
/// of them reaches this gate today, because a bubble holding text is never
/// synthesised for; that number is the cost only in the case where such a balloon's
/// text is also missed.
///
/// It cannot promise anything about false positives nobody has seen. One example is
/// one example, and no sweep manufactures more. What is measured is that this
/// rejects the one known and costs nothing on the other eleven.
///
/// # Why it is not redundant with the OCR-side refusal
///
/// The OCR stage refuses the READ -- a lone prolongation mark, which is what OCR
/// made of that mouth line. This refuses the DETECTION, before OCR is ever asked. A
/// future spurious bubble landing on artwork that happens to read as something
/// legible would defeat that refusal and still be caught here.
const SYNTHESISED_INK_MIN_FRACTION: f64 = 0.035;
/// The marker a synthesised region carries as a SECOND `DetectionLabel`, so a later
/// stage can tell it from a region the detector actually returned.
///
/// **It is `dev.birelate.` and not `dev.koharu.` on purpose:** the synthesis is
/// ours, `--read-textless-bubbles` is ours, and a kind in Koharu's namespace would
/// claim an upstream that does not define it.
///
/// Read by `stages::ocr`, which is the only consumer. Do not gate behaviour on the
/// FIRST label being this -- it never is; the real class stays first so that
/// `regions.rs`'s `detection_confidence` is unchanged on the wire.
pub(in crate::stages) const SYNTHESISED_REGION_KIND: &str = "dev.birelate.region.synthesised";
/// The detector's class name for a drawn sound effect.
///
/// Shared with `ocr.rs` rather than re-spelled there. It is the ONLY thing that
/// separates a sound effect from ordinary lettering once `region_kind` has run:
/// that function collapses both into `TextRegion` whenever `translate_sfx` is on,
/// so a second copy of this literal drifting from this one would not fail a test
/// -- it would silently select every text region.
pub(in crate::stages) const ONOMATOPOEIA: &str = "onomatopoeia";
/// The detector's class name for ordinary lettering.
const TEXT: &str = "text";
/// Slices along a blob's height that must each contain some refined ink. Catches
/// a tall caption the segmenter resolved at one end and lost at the other.
const COVERAGE_BANDS: usize = 6;

/// Confidence a `text` detection needs before the run will MENTION it to the
/// caller, as an edge hint that nothing on this page acts on.
///
/// **This is not a second text threshold and must never be used as one.** The
/// checkpoint's own `text` floor is 0.25 and it still decides every region: a
/// box between this value and that one is reported through
/// `Progress::EdgeHints` and then *dropped*, before NMS, before the erase mask,
/// before OCR. Admitting it instead would strip the artwork under it on every
/// ordinary page read while nothing ever letters it -- see `mask_includes`,
/// which builds the erase mask from raw detections, and the slice-fragment
/// refusal, which refuses a fragment in two places precisely so that cannot happen.
///
/// **0.12.** The previous value is recorded here, because the reasoning for it
/// is still the reasoning against going lower again. It was **0.20**: the checkpoint's own
/// `onomatopoeia` recommendation, i.e. the lowest confidence this detector is
/// already trusted at for any class. The box this constant was first written for
/// -- a display column on one slice of a 179-slice test chapter -- scores **0.2363**
/// on the GPU against the 0.25 floor, so 0.20 already admitted it with margin rather
/// than by a hair. A CPU-proxy sweep of that whole chapter put the extra
/// traffic at 9 boxes, about 0.05 a page; treat that as shape and not as truth,
/// since the proxy disagreed with the GPU on the very box in question (0.2585
/// against 0.2363).
///
/// **What 0.12 buys, measured over that chapter and not predicted.** Text genuinely
/// crosses 25 slice boundaries in that chapter. At 0.20 the seam plans 13 joins
/// and all 13 are real -- precision 13/13, recall about half. At 0.12 it plans
/// 21, of which **17 are real and 4 are false**. The trade has no free point;
/// the four extra real joins were judged worth the four false ones.
///
/// **The false joins were rendered and LOOKED AT before the value was adopted**,
/// which is the only reason adopting it is defensible. Every one lands on
/// *decorative artwork* -- huge white-keylined ink brush flourishes, and an impact
/// starburst -- rather than on text. None of them marks the page. The composite
/// each produces is refused downstream for being impossible rather than lettered:
/// one box is `1219x1211` on a `1200x961` crop, **1.28 of the page**, and comes
/// back `layers=0`. The cost of a false join here is wasted work, not pixels.
const EDGE_HINT_MIN_SCORE: f32 = 0.12;
/// Fraction of a page's height that counts as its top or bottom EDGE BAND, with
/// [`EDGE_BAND_FLOOR_PX`] as the floor.
///
/// The fraction is `extension/seam.js`'s `SEAM_EDGE_FRACTION` verbatim, and that
/// is the point rather than a coincidence: the joiner decides which slices to
/// stitch by asking whether ink reaches a slice's band, and if this file drew
/// the band anywhere else the two would disagree about the same pixel row. The
/// band is deliberately tight -- it was chosen so seaming does not fire on an
/// ordinary vertical-scroll manga reader, where pages are also stacked flush and
/// equal-width but bubbles sit inside a margin.
///
/// **The FLOOR does not match.** `seam.js` sets `SEAM_EDGE_FLOOR_PX` to 16 while
/// this stays 8, so the two files disagree about short pages. Moving the
/// constant is a behaviour change that needs its own sweep, and nothing has
/// measured 16 here.
const EDGE_BAND_FRACTION: f32 = 0.006;
/// See [`EDGE_BAND_FRACTION`]. 8px, so a short page still has a usable band --
/// and 8 is NOT `seam.js`'s 16; see the note above.
const EDGE_BAND_FLOOR_PX: u32 = 8;
/// Ink pixels a row must hold before the walk counts it as ink rather than as
/// paper speckle or a screentone dot.
const INK_WALK_MIN_ROW_PIXELS: u32 = 3;
/// Consecutive paper rows the walk will step over before it decides the column
/// has ended. The largest real inter-glyph gap measured on a test chapter's
/// display column is **87 rows**, so this is that plus headroom, not a guess.
const INK_WALK_MAX_GAP_ROWS: u32 = 96;
/// How far past its own box the walk will follow ink, as a multiple of the box's
/// height.
///
/// A bound rather than "to the page edge" because the question the walk asks is
/// whether THIS column continues, and ink two box-heights below is another
/// column, a caption or a signature. It is also the guard that makes the rule
/// falsifiable: raise it and the negative control stops declining.
const INK_WALK_REACH_RATIO: f32 = 1.0;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default, deny_unknown_fields)]
pub struct KoharuLayoutRFDetrSeg2XLConfig {
    pub text_threshold: Option<f32>,
    pub bubble_threshold: Option<f32>,
    pub panel_threshold: Option<f32>,

    /// Confidence a sound-effect detection needs. `None` keeps the checkpoint's
    /// own recommendation, which is **0.2**.
    ///
    /// That is the lowest of the four classes -- text is 0.25, bubble and panel
    /// 0.5 -- and it was the right shape while `onomatopoeia` was consumed
    /// destructively and only destructively: the class existed to build an erase
    /// mask, where recall is cheap and a false positive merely erases a little
    /// extra.
    ///
    /// `translate_sfx` changed what a false positive costs. A low-confidence box
    /// is now erased *and lettered*, so the failure mode is no longer a slightly
    /// wide erase: it is artwork destroyed and English painted over it. Measured
    /// on 40 pages, the two worst cases -- a playing card whose pips were erased,
    /// and a cityscape panel -- scored **0.232 and 0.241**, while the effects that
    /// letter cleanly sit at 0.89-0.91.
    ///
    /// Raising it is a real trade rather than a free win, which is why the
    /// default is left alone: the same band holds stylised narration that
    /// translates well.
    pub onomatopoeia_threshold: Option<f32>,

    /// Sharpen `text-mask` with `manga-text-segmentation-2025` before it reaches
    /// the inpainter. **Defaults to OFF, on measurement**; `Some(true)` enables it.
    ///
    /// Off because the benefit measured far smaller than the theory predicted
    /// while the harm is real. Enabling it changes **0.19%** of the rendered
    /// pixels on an ordinary bubble page -- because in-bubble text never reaches
    /// an inpainter at all: `fill_uniform_regions` paints the balloon's median
    /// colour over the masked pixels and zeroes them in `pending_mask` before
    /// LaMa runs, so a glyph-tight mask and a 10px blob yield the same flat
    /// fill. Mask precision only bites where text sits on textured artwork --
    /// and that is precisely where this segmenter's recall fails. On an
    /// art-heavy title page the intersection left a vertical caption legible
    /// under the English that the box mask had erased cleanly.
    ///
    /// **That regression is now fixed** and the flag is still off, which is the
    /// honest position rather than a contradictory one. The per-blob rule
    /// originally asked only that each height band be non-empty, and the caption
    /// passed it -- the segmenter finds scattered pixels all the way down a
    /// column while missing most of its glyphs. Each band must now *meet* the
    /// coverage ratio, which rejects that blob and falls it back to the box
    /// mask; re-measured, the caption is erased exactly as it is with refinement
    /// off.
    ///
    /// So the harm is gone and the benefit is still 0.19%. A flag that costs
    /// ~1.5s of detection per page for no measured gain does not earn a default,
    /// but it is now safe to turn on, which it was not before.
    ///
    /// The detection mask is a 288x288 seg head bilinearly upsampled to page
    /// size and then hard-thresholded, so its edge lands wherever a ~5px
    /// quantisation puts it, and `write_mask` then grows it by
    /// `round(max_dim / 1024 * 6)` -- 10px on a 1200x1700 page, unfeathered.
    /// The segmenter runs at true page resolution with a stride-1 output, so it
    /// resolves individual strokes.
    ///
    /// The refined mask is INTERSECTED with the dilated box mask, never unioned.
    /// That makes it a strict subset of what ships today: no page can gain
    /// erasure, only lose it. It also keeps the segmenter's lack of any
    /// detection gate harmless -- run alone it would erase credits, page numbers
    /// and free-standing onomatopoeia, all of which `mask_for` deliberately
    /// leaves alone.
    pub refine_text_mask: Option<bool>,

    /// Probability above which a pixel is text. Default 0.5.
    pub text_mask_threshold: Option<f32>,

    /// Chebyshev dilation applied to the refined mask. Default 2px.
    ///
    /// This is the knob the blanket ~10px dilation never had. It should not be
    /// dropped to 0 without checking the render: measured on real pages, a
    /// per-pixel mask with no growth leaves anti-aliased glyph rims behind and
    /// LaMa reconstructs the Japanese from them as legible ghost strokes.
    pub text_mask_padding_iterations: Option<u32>,

    /// Letter the `onomatopoeia` class instead of consuming it destructively.
    /// **Defaults to ON**; `Some(false)` restores the old behaviour for an A/B.
    ///
    /// RF-DETR emits four classes and this was the only one with no product.
    /// `write_region` returned early for any non-`"text"` label so no text layer
    /// was ever built, `ocr.rs` visits `TextRegion` only, and `mask_for` folded
    /// an onomatopoeia into the *text* mask whenever a bubble contained at least
    /// half of it -- so an in-bubble sound effect was dilated, inpainted away and
    /// never painted back, while a free-standing one was left untouched. The
    /// class was read exactly once, and only to delete.
    ///
    /// Enabling this promotes the class into the pipeline that already exists:
    /// the region becomes a `TextRegion`, so OCR reads it, the translator
    /// receives it as one more segment, and the renderer letters it. **The
    /// unwired `koharu-ml/src/comic_onomatopoeia/` detector is not needed** --
    /// RF-DETR has already localised the effect, and that model's own detector
    /// half would only find the same boxes again.
    ///
    /// The mask rule changes with it, and has to. Once an effect is lettered,
    /// every one of them must be erased rather than only the in-bubble half:
    /// leaving the artwork's kana under fresh English is worse than either
    /// behaviour on its own.
    pub translate_sfx: Option<bool>,

    /// Grow each region's erase mask to this multiple of its own bounding box,
    /// instead of by the page-flat `round(max_dim / 1024 * 6)`.
    ///
    /// `None` keeps the flat rule, which is what has always shipped. See
    /// `scaled_dilated_mask` for why the two are not the same shape of rule: the
    /// flat radius is a constant number of pixels, so as a *scale* it varies
    /// from about 2.0x on 20px text to 1.07x on a 300px effect.
    ///
    /// 1.37 is the optimum reported for LaMa text removal in arXiv 2511.22499,
    /// which also found masks that are too small leave residue while overly
    /// large ones degrade the reconstruction. Their masks are character-wise and
    /// ours are region-wise, so this is an interpretation of that result rather
    /// than a reproduction of it.
    pub mask_scale: Option<f32>,

    /// Add the ink actually found inside each box to the erase mask, using Otsu
    /// plus connected components rather than a model. Off by default.
    ///
    /// The counterpart to `refine_text_mask` and the reason it is worth a
    /// separate flag: that one INTERSECTS and so can only erase less, which is
    /// why it measured negative on sound effects, where the defect is ink left
    /// behind. This one is UNIONed and can only erase more. See `ink_mask_for`.
    pub ink_mask: Option<bool>,
}

pub(super) struct Processor {
    config: DetectionModel,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
    /// The fraction of the page's own area at which a detection stops
    /// contributing to the **erase** mask, or `None` to erase whatever the
    /// detector claimed. See `ProcessorConfig::skip_implausible_masks`.
    ///
    /// Not on `KoharuLayoutRFDetrSeg2XLConfig` beside `mask_scale` and
    /// `ink_mask`, even though it is a mask knob, because the fraction it
    /// compares against is `ProcessorConfig::large_crop_ocr_max_area` -- the same
    /// number the OCR router uses. Splitting the pair across two structs is how
    /// the two ends drift.
    implausible_mask_area: Option<f32>,
    /// Whether a tall vertical free-text column is TURNED into its own ink rather
    /// than widened. **ON by default**; see
    /// `free_text_column_geometry`, and note the caller ANDs this with
    /// `joined_page` before it arrives here, so a manga page never reaches the
    /// turn whatever this says. `false` restores the widened cell exactly, which
    /// is what keeps the A/B available.
    rotate_free_text_columns: bool,
    /// Whether the turn is also offered to a column on an UNJOINED page.
    /// **ON by default** at this layer; the server scopes it per request (below).
    ///
    /// This widens the scope of the JOINED-page turn, where the turn hands back
    /// the ink's own bbox and the exemplar measures `fit_width / width == 1.00`
    /// against the 2.92x the widened cell gave it.
    ///
    /// The population it reaches on an unjoined page is not only display
    /// columns. `free_text_column_geometry` claims any free-standing column
    /// taller than `1 / FREE_TEXT_MIN_ASPECT` times its width at inferred angle
    /// zero, and on manga that includes ordinary upright vertical Japanese --
    /// 366 height-driven free-text regions on one manga baseline, 38.3% of
    /// manga-ja free-text. Turning those would letter English sideways down a
    /// column that should read across. **Detection is never told the source
    /// language** -- there is no such field on this stage -- so the gate cannot
    /// separate them here. **The SERVER separates them one layer up**:
    /// `birelate-server`'s `desired_config` resolves this flag per request to
    /// `false` unless the declared script is Chinese or Korean, because on a
    /// dense Japanese text page the turn letters every free-text column (5 of 5)
    /// sideways. The field's meaning here is unchanged; what changed is who
    /// decides its value.
    ///
    /// On the pages it still reaches, three measured display columns go from
    /// 2.29x / 2.12x / 1.40x their own ink to exactly 1.00. No balloon moves at
    /// any setting, which is what bounds it: dialogue takes the bubble path and
    /// never reaches this function.
    turn_unjoined_columns: bool,
    /// Whether a box the detector truncated at a slice edge is grown along its
    /// own ink, and whether a sub-floor box against an edge is reported as an
    /// [`settle::EdgeHint`]. **ON by default**, on the rendered evidence.
    ///
    /// One flag for both, because they are one mechanism seen at two scores: a
    /// hint only exists because `Model::network_thresholds` lowers the request,
    /// and lowering the request is only worth doing to feed the repair. Splitting
    /// them would let a caller ask for hints that nothing can use.
    ///
    /// `cli.rs` is the authority for the default -- `unwrap_or(true)`, and it
    /// really is reached here, because the arg declares `default_missing_value`
    /// rather than `default_value`.
    ///
    /// On an UNJOINED page this changes what a reader sees. Growing a column to
    /// span its slice puts it past `CROSS_SLICE_HEIGHT_SHARE`, so the
    /// slice-fragment refusal then refuses it on a single-slice request -- the
    /// artwork survives and the artist's Chinese stays standing rather than a
    /// wrong string being lettered over it. By design, that is the preferred
    /// answer.
    ///
    /// Swept over a whole test chapter, hints ON against OFF: **LOST 0, NEW 4**
    /// (178 boundaries), and **0 / 0 on 221 manga pages**, which carry no hints
    /// at all.
    repair_clipped_columns: bool,

    /// Whether a BUBBLE holding no text region of its own is read as one.
    /// **OFF by default.** `ProcessorConfig` carries the numbers.
    read_textless_bubbles: bool,

    /// `Some(floor)` opens a replacement-only score band `[floor, text)` on
    /// JOINED pages only. **`None` by default.** On one measured slice-pair
    /// composite the detector returns the whole cut column at 0.2383
    /// against the 0.25 floor, while a top-only fragment at 0.2715 survives and
    /// reads half the text. A band box is never a region in its own right --
    /// it survives only by replacing a kept box through the axis tie-break.
    joined_page_text_floor: Option<f32>,
    /// Whether the NMS containment/IoU suppression prefers the column-shaped
    /// box of a pair sharing one axis, instead of always the higher score.
    /// **OFF by default**, and the floor above is inert without it: a band box
    /// survives only through the replace outcome this flag enables. The two
    /// ship as a pair.
    axis_aware_nms: bool,
    nms_residue_regions: bool,
    strike_through_devices: bool,
    /// Whether free-standing lettering takes its fill, weight and
    /// contrast-picked outline from the drawn ink's sampled colour.
    /// `cli.rs` is the authority for the default.
    sampled_ink_lettering: bool,
    /// `Some(dir)` writes each settled detection's mask as a PNG there.
    /// See `ProcessorConfig::debug_mask_dir`. `None` is the shipped state.
    debug_mask_dir: Option<PathBuf>,
}

impl Processor {
    pub(super) fn new(
        config: DetectionModel,
        device: koharu_ml::Device,
        implausible_mask_area: Option<f32>,
        rotate_free_text_columns: bool,
        turn_unjoined_columns: bool,
        repair_clipped_columns: bool,
        read_textless_bubbles: bool,
        joined_page_text_floor: Option<f32>,
        axis_aware_nms: bool,
        nms_residue_regions: bool,
        strike_through_devices: bool,
        sampled_ink_lettering: bool,
        debug_mask_dir: Option<String>,
    ) -> Result<Self> {
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(settings) = &config;
        for (name, value) in [
            ("text", settings.text_threshold),
            ("onomatopoeia", settings.onomatopoeia_threshold),
            ("bubble", settings.bubble_threshold),
            ("panel", settings.panel_threshold),
        ] {
            if let Some(value) = value {
                ensure!(
                    value.is_finite() && (0.0..=1.0).contains(&value),
                    "{name} confidence threshold must be finite and between zero and one"
                );
            }
        }

        Ok(Self {
            config,
            device,
            model: ModelCell::new(),
            implausible_mask_area,
            rotate_free_text_columns,
            turn_unjoined_columns,
            repair_clipped_columns,
            read_textless_bubbles,
            joined_page_text_floor,
            axis_aware_nms,
            nms_residue_regions,
            strike_through_devices,
            sampled_ink_lettering,
            debug_mask_dir: debug_mask_dir.map(PathBuf::from),
        })
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> ModelRef<'_> {
        ModelRef::new(MODEL_NAME, &self.model)
    }

    async fn load(&self) -> Result<()> {
        self.model
            .ensure(|| {
                Model::load(
                    self.device.clone(),
                    &self.config,
                    self.implausible_mask_area,
                    self.rotate_free_text_columns,
                    self.turn_unjoined_columns,
                    self.repair_clipped_columns,
                    self.read_textless_bubbles,
                    self.joined_page_text_floor,
                    self.axis_aware_nms,
                    self.nms_residue_regions,
                    self.strike_through_devices,
                    self.sampled_ink_lettering,
                    self.debug_mask_dir.clone(),
                )
            })
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("detection model is not loaded"))?
            .run(input)
            .await
    }
}

struct Model {
    network: Arc<Mutex<KoharuLayoutRFDetrSeg2XL>>,
    thresholds: KoharuLayoutThresholds,
    /// `None` when refinement is switched off, or when the segmenter could not
    /// be loaded. Both mean "use the box mask", which is what shipped before.
    text_mask: Option<Arc<Mutex<MangaTextMaskGenerator>>>,
    cleaning: MangaTextMaskCleaningOptions,
    /// Resolved once at load: `translate_sfx` defaults to on.
    translate_sfx: bool,
    /// `None` keeps the page-flat dilation.
    mask_scale: Option<f32>,
    /// Union the algorithmic ink mask into the erase mask.
    ink_mask: bool,
    /// `None` erases whatever the detector claimed, which is what has always
    /// shipped. See `Processor::implausible_mask_area`.
    implausible_mask_area: Option<f32>,
    /// See `Processor::rotate_free_text_columns`.
    rotate_free_text_columns: bool,
    /// See `Processor::turn_unjoined_columns`. **ON by default**
    /// (`cli.rs`, `unwrap_or(true)`). `cli.rs` is the authority for a default,
    /// never a doc comment on the struct that receives it.
    turn_unjoined_columns: bool,
    /// See `Processor::repair_clipped_columns`. **ON is the shipping default**,
    /// chosen after both arms were rendered.
    ///
    /// FALSE restores the previous behaviour op for op -- no lowered network
    /// request, so no sub-floor box exists to become a hint, and no growth.
    repair_clipped_columns: bool,

    /// Whether a BUBBLE holding no text region of its own is read as one.
    /// **OFF by default.** `ProcessorConfig` carries the numbers.
    read_textless_bubbles: bool,

    /// `Some(floor)` opens the replacement-only band on JOINED pages only;
    /// `None` is the shipping default, and `cli.rs` is the authority for the
    /// recommended value.
    joined_page_text_floor: Option<f32>,
    /// Whether NMS prefers the column-shaped box of a contained pair over the
    /// higher-scored one. **OFF by default.**
    axis_aware_nms: bool,
    nms_residue_regions: bool,
    strike_through_devices: bool,
    /// See `Processor::sampled_ink_lettering`.
    sampled_ink_lettering: bool,
    /// See `Processor::debug_mask_dir`.
    debug_mask_dir: Option<PathBuf>,
}

impl Model {
    async fn load(
        device: koharu_ml::Device,
        config: &DetectionModel,
        implausible_mask_area: Option<f32>,
        rotate_free_text_columns: bool,
        turn_unjoined_columns: bool,
        repair_clipped_columns: bool,
        read_textless_bubbles: bool,
        joined_page_text_floor: Option<f32>,
        axis_aware_nms: bool,
        nms_residue_regions: bool,
        strike_through_devices: bool,
        sampled_ink_lettering: bool,
        debug_mask_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(config) = config;
        let network = KoharuLayoutRFDetrSeg2XL::load(device.clone()).await?;
        let mut thresholds = network.recommended_thresholds();
        thresholds.text = config.text_threshold.unwrap_or(thresholds.text);
        thresholds.onomatopoeia = config
            .onomatopoeia_threshold
            .unwrap_or(thresholds.onomatopoeia);
        thresholds.bubble = config.bubble_threshold.unwrap_or(thresholds.bubble);
        thresholds.panel = config.panel_threshold.unwrap_or(thresholds.panel);

        let mut cleaning = MangaTextMaskCleaningOptions::default();
        if let Some(threshold) = config.text_mask_threshold {
            cleaning.threshold = threshold;
        }
        if let Some(iterations) = config.text_mask_padding_iterations {
            cleaning.padding_iterations = iterations;
        }

        // Refinement failing must not fail detection. `MangaTextMaskGenerator`
        // has a `TryFrom<Device>` that `KoharuLayoutRFDetrSeg2XL::load` does
        // not, so it can reject a backend the detector accepted -- and losing
        // the whole stage over a mask that is only ever an intersection would
        // be a strict regression on a page that works today.
        let text_mask = if config.refine_text_mask.unwrap_or(false) {
            match MangaTextMaskGenerator::load(device).await {
                Ok(generator) => Some(Arc::new(Mutex::new(generator))),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "manga text segmentation unavailable; \
                         falling back to the detection mask"
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            network: Arc::new(Mutex::new(network)),
            thresholds,
            text_mask,
            cleaning,
            translate_sfx: config.translate_sfx.unwrap_or(true),
            mask_scale: config.mask_scale,
            ink_mask: config.ink_mask.unwrap_or(false),
            implausible_mask_area,
            rotate_free_text_columns,
            turn_unjoined_columns,
            repair_clipped_columns,
            read_textless_bubbles,
            joined_page_text_floor,
            axis_aware_nms,
            nms_residue_regions,
            strike_through_devices,
            sampled_ink_lettering,
            debug_mask_dir,
        })
    }

    async fn run(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let page = input.page;
        let image = input
            .images
            .get(&input.scene, page, "source")?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;
        let output = self.detect(image.clone(), input.joined_page()).await?;
        // Nothing to sharpen when nothing was detected as text, and the forward
        // pass is a full-page one -- worth the branch.
        let refined = if output.detections.iter().any(|value| value.label == "text") {
            self.segment(image.clone()).await?
        } else {
            None
        };
        build_patch(
            &input,
            &image,
            output,
            &generation(PRODUCER, MODEL_ID)?,
            refined.as_ref(),
            self.translate_sfx,
            self.mask_scale,
            self.ink_mask,
            self.implausible_mask_area,
            self.rotate_free_text_columns,
            self.turn_unjoined_columns,
            self.thresholds.text,
            self.joined_floor(input.joined_page()),
            self.repair_clipped_columns,
            self.read_textless_bubbles,
            self.axis_aware_nms,
            self.nms_residue_regions,
            self.strike_through_devices,
            self.sampled_ink_lettering,
            self.debug_mask_dir.as_deref(),
        )
    }

    /// The replacement-only band's lower edge, or `None` off a joined page and
    /// whenever the flag is unset. `.min()`, not assignment: the flag may only
    /// ever *lower* the checkpoint's own recommendation, so a fat-fingered 0.9
    /// cannot silently raise the shipping floor -- the same idiom as
    /// `network_thresholds`' edge-hint clamp. `joined_page` is the caller
    /// stating what it built, never a format guess.
    fn joined_floor(&self, joined_page: bool) -> Option<f32> {
        match self.joined_page_text_floor {
            Some(floor) if joined_page => Some(self.thresholds.text.min(floor)),
            _ => None,
        }
    }

    /// The lowest score this run wants the NETWORK to return `text` at.
    fn text_floor(&self, joined_page: bool) -> f32 {
        self.joined_floor(joined_page)
            .unwrap_or(self.thresholds.text)
    }

    /// What the NETWORK is asked for, which is not what this stage admits.
    ///
    /// **The sub-floor boxes an edge hint is made of do not exist at the call
    /// site by default, and this is the only place they can be created.**
    /// `KoharuLayoutRFDetrSeg2XL` applies the thresholds itself, inside
    /// `processor.rs` `postprocess` -- `keep = scores.gt_tensor(&thresholds)` --
    /// so by the time `KoharuLayoutDetections` reaches `write_page` every box
    /// under the `text` floor has already been dropped on the device. Lowering
    /// the request is the mechanism; `write_page` then re-applies
    /// `self.thresholds.text` verbatim, so which boxes become REGIONS is
    /// unchanged, op for op.
    ///
    /// Only the `text` class moves. `onomatopoeia`, `bubble` and `panel` keep
    /// whatever the config resolved, so a sub-floor detection reaching the
    /// pipeline can only be a `text` one.
    ///
    /// `min`, so a caller who has already set `text_threshold` below the hint
    /// floor is not silently raised. Their run then has no sub-floor band at all
    /// and reports no hints, which is right: everything is already a region.
    ///
    /// **Gated, and this is the whole of the off switch on the device side.**
    /// With `repair_clipped_columns` false the request is `self.thresholds`
    /// untouched, so the network drops every sub-floor box exactly as it would
    /// with no hint machinery, and there is nothing for `edge_hint` to find. The flag
    /// is therefore not a filter applied after the fact -- off means the work is
    /// never done, not that its results are discarded.
    fn network_thresholds(&self, joined_page: bool) -> KoharuLayoutThresholds {
        let mut thresholds = self.thresholds;
        /* The joined floor moves the REQUEST too, not only the admission above.
         * Under the shipping `repair_clipped_columns` default the `min` below
         * already asks lower than any plausible joined floor, so this line is a
         * present-day no-op -- but leaving it out would couple the floor lever
         * to a DIFFERENT flag staying on, silently. */
        thresholds.text = self.text_floor(joined_page);
        if self.repair_clipped_columns {
            thresholds.text = thresholds.text.min(EDGE_HINT_MIN_SCORE);
        }
        thresholds
    }

    async fn detect(
        &self,
        image: Arc<DynamicImage>,
        joined_page: bool,
    ) -> Result<KoharuLayoutDetections> {
        let network = self.network.clone();
        let thresholds = self.network_thresholds(joined_page);
        tokio::task::spawn_blocking(move || {
            let network = network
                .lock()
                .map_err(|_| anyhow!("layout model lock is poisoned"))?;
            network.inference_with_thresholds(&image, thresholds)
        })
        .await
        .context("layout detection task panicked")?
    }

    /// Per-pixel text probabilities for the whole page, cleaned into a mask.
    ///
    /// Returns `Ok(None)` rather than an error when the segmenter is absent, so
    /// the caller has one code path for "no refinement available".
    async fn segment(&self, image: Arc<DynamicImage>) -> Result<Option<GrayImage>> {
        let Some(generator) = self.text_mask.clone() else {
            return Ok(None);
        };
        let cleaning = self.cleaning.clone();
        tokio::task::spawn_blocking(move || {
            let generator = generator
                .lock()
                .map_err(|_| anyhow!("manga text segmentation lock is poisoned"))?;
            Ok(Some(generator.inference(&image)?.process(&cleaning)?))
        })
        .await
        .context("manga text segmentation task panicked")?
    }
}

#[derive(Clone, Copy)]
struct DetectedRegion {
    entity: EntityId,
    bounds: [f32; 4],
    /// Bounds of the `Geometry` actually written for the region, which for a
    /// bubble is its mask polygon rather than `bounds`. This is the frame the
    /// renderer lays text out in: `bubble::resolve` reads the target's geometry
    /// and `geometry_frame` falls back to its axis-aligned bounds for a polygon
    /// that is not a rectangle.
    frame: [f32; 4],
}

#[derive(Clone)]
struct DetectedText {
    region: DetectedRegion,
    content: EntityId,
    layer: EntityId,
    /// What the glyph mask said about this text, or `None` when the mask carried
    /// no foreground pixel. Retained past `write_region` because the layout box
    /// and the halo both need it, and neither can be decided until every text on
    /// the page is known.
    inferred: Option<InferredTypography>,
    /// The colour of a strike-through mark the author drew across this text, or
    /// `None` -- almost always.
    ///
    /// Measured in `write_region`, which has the page pixels, and spent in
    /// `write_typography`, which does not: the same reason `inferred` is carried
    /// rather than recomputed.
    strike: Option<[u8; 4]>,
    /// Whether this text is the RESIDUE of a box the axis tie-break evicted --
    /// `--nms-residue-regions`. On the measured exhibit that is the red
    /// replacement name standing beside the struck one: the device's gloss.
    ///
    /// Carried because the gloss cannot recognise itself. `strike_ink` is a
    /// measured NEGATIVE on the gloss column -- no colour bucket there clears
    /// the aspect, fill and span tests -- so the gloss never gets a `strike` of
    /// its own, and a rule phrased as "read my own strike extension" would be a
    /// silent no-op. Provenance is the ONLY thing separating it from an ordinary
    /// detection, whose label, score and mask it otherwise clones exactly.
    residue: bool,
    /// The `Typography` this run authors for `layer`, or `None` when a user owns
    /// the existing one and the pipeline must not write at all.
    ///
    /// Held rather than written, because the halo is only decidable once it is
    /// known whether a bubble contains the text -- and a component may be `set`
    /// only ONCE per `Edit` on an entity the base snapshot already holds. Each
    /// `set` records an `Observation::Component` carrying the fingerprint it read
    /// *at that moment*, so a second one observes the first write's value, which
    /// can never match the base and rejects the whole patch as a conflict
    /// (`edit.rs` `validate_removal_authorship`, `patch.rs` `Observation::validate`).
    typography: Option<Typography>,
}

#[derive(Clone)]
struct PreviousText {
    bounds: [f32; 4],
    geometry: Geometry,
    content: EntityId,
    layer: EntityId,
}

#[derive(Default)]
struct PageRegions {
    bubbles: Vec<DetectedRegion>,
    texts: Vec<DetectedText>,
}

#[derive(Clone, Copy)]
struct ImageSize {
    width: u32,
    height: u32,
}

fn build_patch(
    input: &StageInput,
    image: &DynamicImage,
    output: KoharuLayoutDetections,
    generation: &Generation,
    refined_text_mask: Option<&GrayImage>,
    translate_sfx: bool,
    mask_scale: Option<f32>,
    ink_mask: bool,
    implausible_mask_area: Option<f32>,
    rotate_free_text_columns: bool,
    turn_unjoined_columns: bool,
    text_floor: f32,
    joined_text_floor: Option<f32>,
    repair_clipped_columns: bool,
    read_textless_bubbles: bool,
    axis_aware_nms: bool,
    nms_residue_regions: bool,
    strike_through_devices: bool,
    sampled_ink_lettering: bool,
    debug_mask_dir: Option<&Path>,
) -> Result<koharu_scene::Patch> {
    let page = input.page;
    let mut previous_texts = previous_texts(input, generation)?;
    let mut reused_contents = BTreeSet::new();
    let mut edit = input.scene.edit_as(generation.clone());
    edit.observe_subtree(page)?;
    remove_previous_regions(input, &mut edit, generation)?;
    write_page(
        input,
        &mut edit,
        page,
        image,
        output,
        generation,
        &mut previous_texts,
        &mut reused_contents,
        refined_text_mask,
        translate_sfx,
        mask_scale,
        ink_mask,
        implausible_mask_area,
        rotate_free_text_columns,
        turn_unjoined_columns,
        text_floor,
        joined_text_floor,
        repair_clipped_columns,
        read_textless_bubbles,
        axis_aware_nms,
        nms_residue_regions,
        strike_through_devices,
        sampled_ink_lettering,
        debug_mask_dir,
    )?;
    remove_unmatched_texts(
        input,
        &mut edit,
        generation,
        previous_texts,
        &reused_contents,
    )?;
    finish(edit)
}

fn remove_previous_regions(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    generation: &Generation,
) -> Result<()> {
    let mut remove = Vec::new();
    for entity in input.scene.descendants(input.page)? {
        let id = entity.id();
        if !input.contains_entity(id)? {
            continue;
        }
        let owned_region = entity
            .component::<EntityOrigin>()?
            .is_some_and(|origin| {
                matches!(origin.origin, Origin::Generated(ref owner) if owner.producer == generation.producer)
            })
            && entity.component::<Region>()?.is_some();
        if owned_region {
            remove.push(id);
        }
    }
    for entity in remove {
        if input.scene.entity(entity).is_ok() {
            edit.remove_entity(entity, RemovePolicy::Cascade)?;
        }
    }
    Ok(())
}

fn previous_texts(input: &StageInput, generation: &Generation) -> Result<Vec<PreviousText>> {
    let mut previous = Vec::new();
    for entity in input.scene.descendants(input.page)? {
        let region = entity.id();
        if !input.contains_entity(region)? {
            continue;
        }
        let owned_text_region = entity
            .component::<EntityOrigin>()?
            .is_some_and(|origin| {
                matches!(origin.origin, Origin::Generated(ref owner) if owner.producer == generation.producer)
            })
            && entity
                .component::<Region>()?
                .is_some_and(|value| value.kind == TextRegion::kind());
        if !owned_text_region {
            continue;
        }
        let Some(geometry) = entity.component::<Geometry>()? else {
            continue;
        };
        let Some(bounds) = geometry_bounds(&geometry) else {
            continue;
        };
        for recognized in input.scene.relations_to_as::<RecognizedFrom>(region) {
            let content = recognized.value().source;
            for presentation in input.scene.relations_to_as::<Presents>(content) {
                previous.push(PreviousText {
                    bounds,
                    geometry: geometry.clone(),
                    content,
                    layer: presentation.value().source,
                });
            }
        }
    }
    Ok(previous)
}

fn remove_unmatched_texts(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    generation: &Generation,
    previous: Vec<PreviousText>,
    reused_contents: &BTreeSet<EntityId>,
) -> Result<()> {
    let mut contents = BTreeSet::new();
    for previous in previous {
        contents.insert(previous.content);
        let layer_generated = input
            .scene
            .component::<EntityOrigin>(previous.layer)?
            .is_some_and(|origin| {
                matches!(origin.origin, Origin::Generated(ref owner) if owner.producer == generation.producer)
            });
        if layer_generated {
            edit.remove_entity(previous.layer, RemovePolicy::Cascade)?;
        } else if input.scene.component::<Geometry>(previous.layer)?.is_none() {
            let mut geometry = previous.geometry;
            geometry.origin = Origin::User;
            edit.set(previous.layer, &geometry)?;
        }
    }
    for content in contents {
        if reused_contents.contains(&content) {
            continue;
        }
        let content_generated = input
            .scene
            .component::<EntityOrigin>(content)?
            .is_some_and(|origin| {
                matches!(origin.origin, Origin::Generated(ref owner) if owner.producer == generation.producer)
            });
        if content_generated {
            edit.remove_entity(content, RemovePolicy::Cascade)?;
        }
    }
    Ok(())
}

fn write_page(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    image: &DynamicImage,
    output: KoharuLayoutDetections,
    generation: &Generation,
    previous_texts: &mut Vec<PreviousText>,
    reused_contents: &mut BTreeSet<EntityId>,
    refined_text_mask: Option<&GrayImage>,
    translate_sfx: bool,
    mask_scale: Option<f32>,
    ink_mask: bool,
    implausible_mask_area: Option<f32>,
    rotate_free_text_columns: bool,
    turn_unjoined_columns: bool,
    text_floor: f32,
    joined_text_floor: Option<f32>,
    repair_clipped_columns: bool,
    read_textless_bubbles: bool,
    axis_aware_nms: bool,
    nms_residue_regions: bool,
    strike_through_devices: bool,
    sampled_ink_lettering: bool,
    debug_mask_dir: Option<&Path>,
) -> Result<()> {
    let KoharuLayoutDetections {
        mut detections,
        image_width,
        image_height,
    } = output;
    let size = ImageSize {
        width: image_width,
        height: image_height,
    };
    if let Some(region) = input.region {
        detections.retain(|detection| intersects(detection.bbox, region));
    }
    /* HOISTED ABOVE THE SUPPRESSION, and it used to sit below it.
     *
     * `settle_detections` walks the page's own ink to decide whether a box the
     * detector truncated at a slice edge is a column that carries on, so it
     * needs the decoded page here rather than three lines later. */
    let image = image.to_rgb8();
    let (mut detections, hints) = settle::settle_detections(
        detections,
        0.5,
        translate_sfx,
        text_floor,
        joined_text_floor,
        size,
        &image,
        repair_clipped_columns,
        read_textless_bubbles,
        axis_aware_nms,
        nms_residue_regions,
    );
    if !hints.is_empty() {
        tracing::info!(
            hints = hints.len(),
            page = %page,
            "reporting sub-floor text against a page edge"
        );
        input.report(Progress::EdgeHints { page, hints });
    }
    /* The reader's edits, in this order and this place on purpose:
     * removal AFTER the settle pass, so a synthesised textless-bubble read
     * (a known false-positive class) is deletable too; admission after
     * removal, so a delete-and-redraw in one request nets to the new box; and
     * both before the sort, the region writes and `write_masks`, so reading
     * order, the lettering and the ERASE all see the reader's answer -- an
     * added box is erased and lettered, and a deleted one's artwork survives
     * untouched. Counted against what was ASKED, because an admission can be
     * refused for holding no ink and a removal rect can cover nothing. */
    /* Donors snapshot FIRST: a resize is remove(old)+add(nudged),
     * and the region being replaced must still be visible when the admission
     * decides whether the model's mask can be inherited. */
    let donors = detections.caller_donors(input.added_regions());
    let removed = detections.remove_caller_boxes(input.removed_regions());
    let admitted = detections.admit_caller_boxes(&image, input.added_regions(), &donors);
    if !input.removed_regions().is_empty() || !input.added_regions().is_empty() {
        tracing::info!(
            removed,
            admitted,
            asked_removed = input.removed_regions().len(),
            asked_added = input.added_regions().len(),
            page = %page,
            "applied the reader's box edits"
        );
    }
    detections.sort_by_layout();

    /* SETTLED masks on purpose, not the network's raw output: post-NMS, post
     * column repair, post textless-bubble synthesis, in reading order -- the
     * same objects `infer_ink_core` and the erase assembly consume, which is
     * the blindness this dump exists to lift. */
    if let Some(dir) = debug_mask_dir {
        if let Err(error) = dump_settled_masks(dir, page, &detections) {
            tracing::warn!(%error, page = %page, "debug mask dump failed");
        }
    }

    let regions = write_regions(
        &input.scene,
        edit,
        page,
        &image,
        &detections,
        generation,
        previous_texts,
        reused_contents,
        translate_sfx,
        strike_through_devices,
    )?;
    /* SCOPED TO PAGES THE CALLER ASSEMBLED, and that is what keeps it off manga.
     *
     * `joined_page` is not a format classifier -- "webtoon or manga?" fails
     * as a discriminator, and its counterexample was a
     * JAPANESE webtoon, so a format gate would rotate that page's captions too.
     * This is the caller stating what it built, which is the distinction
     * `ocr.rs` already draws: "a fact the caller has to state rather than one
     * the geometry can infer".
     *
     * And the seam that sets it is ALREADY scoped away from manga on purpose.
     * `extension/seam.js` uses a tight edge band -- `SEAM_EDGE_FLOOR_PX` 16
     * plus `SEAM_EDGE_FRACTION` 0.006 -- chosen so it does not fire "on an ordinary
     * vertical-scroll manga reader, where pages are also stacked flush and
     * equal-width but bubbles sit inside a margin". So a manga page is never
     * joined, and therefore never turned.
     *
     * The cost, stated plainly: this is NARROWER than "all webtoons". Display
     * text sitting wholly inside one slice is not joined either, so it keeps the
     * widened cell. Widening the scope is a separate decision and wants its own
     * render. */
    link_dialogue_regions(
        &input.scene,
        edit,
        &regions,
        size,
        generation,
        turn_is_offered(
            rotate_free_text_columns,
            input.joined_page(),
            turn_unjoined_columns,
        ),
        sampled_ink_lettering,
    )?;
    write_masks(
        input,
        edit,
        page,
        &detections,
        size,
        refined_text_mask,
        translate_sfx,
        mask_scale,
        ink_mask,
        implausible_mask_area,
        &image,
    )
}

fn write_regions(
    snapshot: &koharu_scene::Snapshot,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    image: &RgbImage,
    // `&settle::Settled` and NOT `&[KoharuLayoutDetection]`. See the `settle`
    // module: this parameter type is what makes an unwired `write_page` a
    // compile error rather than a silent regression.
    detections: &settle::Settled,
    generation: &Generation,
    previous_texts: &mut Vec<PreviousText>,
    reused_contents: &mut BTreeSet<EntityId>,
    translate_sfx: bool,
    strike_through_devices: bool,
) -> Result<PageRegions> {
    let mut regions = PageRegions::default();
    let text_group = snapshot.page(page)?.text_group()?;
    let managed_text_group = if let Some(group) = text_group {
        snapshot
            .component::<EntityOrigin>(group.id())?
            .is_some_and(|origin| origin.origin != Origin::User)
            .then_some(group.id())
    } else {
        None
    };
    for (index, detection) in detections.iter().enumerate() {
        // An effect that would be lettered on top of something else is spared
        // ENTIRELY -- not lettered, not read, not erased. Passing the flag down
        // as false for this one detection reuses the whole `--no-translate-sfx`
        // path rather than inventing a second way to skip a region: it stops
        // `letters_text`, and through it `region_kind`, which is what keeps the
        // region out of OCR (`ocr.rs`) and out of the flat fill
        // (`inpainting.rs`). `mask_includes` recomputes the same predicate for
        // the erase mask, from this same slice.
        let effects_here = translate_sfx
            && !(spare_overshadowed_effects()
                && overshadowed_effect(detection, detections, translate_sfx));
        let letters = letters_text(&detection.label, effects_here);
        let previous = letters
            .then(|| take_previous_text(previous_texts, detection.bbox))
            .flatten();
        let (detected, text) = write_region(
            snapshot,
            edit,
            page,
            image,
            detection,
            generation,
            previous,
            reused_contents,
            effects_here,
            // Asked of `Settled` by index rather than recomputed from the label,
            // because nothing in a synthesised detection distinguishes it from a
            // `text` box the network returned -- that is the point of it.
            detections.synthesised(index),
            // Same reason as `synthesised` above: nothing in a residue strip
            // distinguishes it from a `text` box the network returned.
            detections.residue(index),
            strike_through_devices,
        )?;
        if detection.label == "bubble" {
            regions.bubbles.push(detected);
        } else if letters {
            let text = text.expect("lettered detections create text semantics");
            if let Some(group) = managed_text_group {
                edit.move_entity(text.layer, Some(group), At::End)?;
            }
            regions.texts.push(text);
        }
    }
    Ok(regions)
}

fn write_region(
    snapshot: &koharu_scene::Snapshot,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    image: &RgbImage,
    detection: &KoharuLayoutDetection,
    generation: &Generation,
    previous: Option<PreviousText>,
    reused_contents: &mut BTreeSet<EntityId>,
    translate_sfx: bool,
    /* Whether this detection was SYNTHESISED from a bubble holding no text region
     * of its own -- `--read-textless-bubbles`. `Settled` is the source;
     * see `oriented_ink_box` for what it changes and why it cannot be inferred
     * here from the label. False for everything the network returned, which is
     * everything at all with the flag off. */
    synthesised: bool,
    /* Whether this detection is an evicted box's readmitted residue --
     * `--nms-residue-regions`. Asked of `Settled` rather than inferred
     * here, because a residue strip clones its loser's label, score and mask and
     * so is indistinguishable from an ordinary detection by inspection. */
    residue: bool,
    /* `--strike-through-devices`. With it false `strike_ink` is never
     * called, so the histogram is not merely discarded -- the work is never done,
     * and the off arm cannot differ by a pixel. */
    strike_through_devices: bool,
) -> Result<(DetectedRegion, Option<DetectedText>)> {
    let entity = edit.add_entity(page, At::End)?;
    let letters = letters_text(&detection.label, translate_sfx);
    let kind = region_kind(&detection.label, translate_sfx)?;
    // A sound effect is lettering too, so it needs the same measurement the
    // `text` class gets: the glyph height, the ink colour and the angle all come
    // off the segmentation mask, and without them an effect would be laid out
    // from an unrotated box at the theme's default size.
    let inferred = letters
        .then(|| infer_typography(image, detection))
        .flatten();
    let geometry = region_geometry(detection, letters, synthesised, inferred);
    edit.set(entity, &geometry)?;
    edit.set(
        entity,
        &Region {
            origin: Origin::Generated(generation.clone()),
            kind: kind.clone(),
            label: Some(detection.label.clone()),
        },
    )?;
    /* A SYNTHESISED REGION CARRIES A SECOND LABEL, and that is the only way the
     * later stages can tell it apart.
     *
     * The pipeline's stages communicate through the SCENE and nothing else --
     * `PageRegions` dies with this function -- and a synthesised region is
     * otherwise indistinguishable from a detected one: `Region.label` is `text`
     * for both, the kind is `TextRegion` for both, and `link_dialogue_regions`
     * gives both the role `dialogue` because containment against its own bubble is
     * 1.0. So `write_region`'s `synthesised` argument, which is the truth, had no
     * way of reaching the stage that needs it.
     *
     * `DetectionAnalysis.labels` is a `Vec` and its validator constrains only that
     * each kind parses and each confidence is finite and in `0..=1` -- no
     * uniqueness, no count. So an extra entry costs no component, no schema
     * revision and no wire change: `regions.rs` reports `detection_confidence`
     * from the FIRST label, which is still the real one.
     *
     * The confidence is copied rather than invented, because the validator demands
     * a real one and a sentinel like 0.0 would be a number somebody later reads. */
    let mut labels = vec![DetectionLabel {
        kind,
        confidence: detection.score,
    }];
    if synthesised {
        labels.push(DetectionLabel {
            kind: RegionKind::new(SYNTHESISED_REGION_KIND)?,
            confidence: detection.score,
        });
    }
    edit.set(
        entity,
        &DetectionAnalysis {
            origin: Origin::Generated(generation.clone()),
            labels,
        },
    )?;
    let region = DetectedRegion {
        entity,
        bounds: detection.bbox,
        frame: geometry_bounds(&geometry).unwrap_or(detection.bbox),
    };
    if !letters {
        return Ok((region, None));
    }

    let (content, layer, created) = previous.map_or_else(
        || -> Result<_> {
            let content = edit.add_text_content(page, At::End)?;
            let layer = edit.add_text_layer(
                page,
                At::End,
                content,
                &TextLayout {
                    origin: Origin::Generated(generation.clone()),
                    kind: TextLayoutKind::Paragraph,
                },
            )?;
            Ok((content, layer, true))
        },
        |previous| Ok((previous.content, previous.layer, false)),
    )?;
    if !created {
        reused_contents.insert(content);
    }
    if created
        || snapshot
            .component::<TextRole>(content)?
            .is_none_or(|value| value.origin != Origin::User)
    {
        write_text_role(edit, content, "dev.koharu.text.free-text", generation)?;
    }
    if created || snapshot.component::<TextLayout>(layer)?.is_none() {
        edit.set(
            layer,
            &TextLayout {
                origin: Origin::Generated(generation.clone()),
                kind: TextLayoutKind::Paragraph,
            },
        )?;
    }
    // Built here and written by `link_dialogue_regions`, which is the first point
    // that knows whether a bubble contains this text and so whether its glyphs
    // need a halo. See `DetectedText::typography` for why it must not be written
    // twice.
    let typography = (created
        || snapshot
            .component::<Typography>(layer)?
            .is_none_or(|value| value.origin != Origin::User))
    .then(|| Typography {
        origin: Origin::Generated(generation.clone()),
        preferred_font: None,
        font_weight: None,
        size: inferred.map(|value| value.font_size),
        auto_fit: true,
        color: inferred.map(|value| [value.color[0], value.color[1], value.color[2], u8::MAX]),
        stroke_color: None,
        stroke_width: None,
        alignment: None,
        writing_mode: inferred.map(|value| value.writing_mode),
        extensions: Default::default(),
    });
    edit.relate::<RecognizedFrom>(content, entity)?;

    Ok((
        region,
        Some(DetectedText {
            region,
            content,
            layer,
            inferred,
            /* Measured HERE because this is where the page pixels are; spent in
             * `write_typography`, which has no image. Gated so the off arm never
             * runs the histogram at all. */
            strike: strike_through_devices
                .then(|| strike_ink(image, detection))
                .flatten(),
            residue,
            typography,
        }),
    ))
}

/// Relates each detected text to the bubble that contains it, and gives every
/// text in a shared bubble its own share of that bubble's frame.
///
/// A generated text layer carries no `Geometry`, so the compositor falls back
/// to the frame of whatever the layer `FitsTo`. Relating several texts to one
/// bubble therefore hands them all the same bounds and paints their
/// translations on top of each other. The relations still have to be written --
/// they are what keeps the region a bubble and the balloon contour in force --
/// so the fix is an explicit `Geometry` per layer, which `compositor.rs` prefers
/// over the fit frame and still derives a contour from. That is the same
/// mechanism the desktop editor's `set_geometry` uses, which is why a human
/// dragging text boxes apart never sees this.
///
/// It is also the first point that knows whether a bubble contains a text, so
/// two things that turn on that answer settle here: a free-standing vertical
/// text gets a box wide enough to read its translation in (`free_text_cell`),
/// and only free-standing text gets a glyph halo (`text_stroke`). Every text
/// layer's `Typography` is written here for that reason, once.
fn link_dialogue_regions(
    snapshot: &koharu_scene::Snapshot,
    edit: &mut koharu_scene::Edit,
    regions: &PageRegions,
    size: ImageSize,
    generation: &Generation,
    rotate_free_text_columns: bool,
    sampled_ink_lettering: bool,
) -> Result<()> {
    let mut containing = Vec::with_capacity(regions.texts.len());
    let mut sharing: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (index, text) in regions.texts.iter().enumerate() {
        let bubble = containing_bubble(&regions.bubbles, text.region.bounds).copied();
        if let Some(bubble) = bubble {
            sharing.entry(bubble.entity).or_default().push(index);
        }
        containing.push(bubble);
    }

    let device_ink = device_strike_ink(regions.texts.iter().map(|text| text.strike));

    for (index, text) in regions.texts.iter().enumerate() {
        let containing_bubble = containing[index];
        match containing_bubble {
            // Free-standing text: no bubble to inherit a frame from. Its own
            // bbox is the layout box, and for a vertical column that box is far
            // too narrow to read the horizontal translation in, so widen it.
            //
            // Horizontal free text -- a signboard, a caption -- already fits its
            // own bbox, and `max` below leaves it at exactly its own width, so
            // this only ever fires where the column shape is the problem.
            None => {
                edit.relate::<FitsTo>(text.layer, text.region.entity)?;
                // This used to be gated on `writing_mode == Vertical`, which
                // conflated the two jobs `free_text_cell` does. Widening is
                // already self-gating -- `half` takes the `max` of the box's own
                // width, so a wide horizontal sign grows by exactly nothing --
                // but the gate also denied horizontal text the neighbour CUTS,
                // and those are what stop two free-standing texts being painted
                // over each other. Measured on 32 real pages: 12 of the 14
                // regions caught in an overlapping pair are horizontal
                // free-text, i.e. precisely the ones the gate excluded.
                // A tall column is turned rather than widened, when the caller
                // asked for it and `free_text_column_geometry` claims the region.
                // Tried FIRST because the widened cell is the thing it replaces;
                // it returns `None` for everything it does not claim, so the
                // fall-through below is the unchanged path for every other region.
                let turned = rotate_free_text_columns
                    .then(|| free_text_column_geometry(text.region.bounds, text.inferred))
                    .flatten();
                // A layer the reader positioned by hand already carries the
                // geometry the compositor prefers, and the pipeline is not
                // allowed to overwrite a user-owned component anyway.
                if let Some(geometry) = turned {
                    if layer_geometry_is_writable(snapshot, text.layer, generation)? {
                        edit.set(text.layer, &geometry)?;
                    }
                } else if let Some((cell, cut)) =
                    free_text_cell(text.region.bounds, size, &other_text_bounds(regions, index))
                    && layer_geometry_is_writable(snapshot, text.layer, generation)?
                {
                    edit.set(text.layer, &free_text_geometry(cell, cut, text.inferred))?;
                }
            }
            Some(bubble) => {
                let shared = sharing
                    .get(&bubble.entity)
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                let (fit, cell) = if shared.len() < 2 {
                    (bubble.entity, None)
                } else {
                    let neighbours = shared
                        .iter()
                        .filter(|other| **other != index)
                        .map(|other| regions.texts[*other].region.bounds)
                        .collect::<Vec<_>>();
                    match bubble_text_cell(bubble.frame, text.region.bounds, &neighbours) {
                        Some(cell) => {
                            let writable =
                                layer_geometry_is_writable(snapshot, text.layer, generation)?;
                            (bubble.entity, writable.then_some(cell))
                        }
                        // No usable cell: the text's own region is a worse frame
                        // than the bubble but at least it is this text's alone.
                        //
                        // "Worse" is an understatement. The text's own region is
                        // the SOURCE INK COLUMN, and for vertical Japanese that
                        // is a strip a few characters wide. The horizontal
                        // English is laid into it, auto-fit walks down to
                        // `minimum_font_size` and still overflows, and the
                        // fitter's only remaining lever is to hyphenate -- which
                        // is where the shredded two-letter line fragments come
                        // from. Measured over 31 real pages: 21 of 254 regions
                        // were dialogue that landed here, and every layout
                        // warning on the whole set was an overflow at the 9px
                        // floor.
                        //
                        // The widening below is a PARTIAL mitigation and the
                        // measurement says so: it took those 21 to 18. Do not
                        // read it as the fix.
                        //
                        // Why it only goes so far, since the next person will
                        // want to know before trying again. `free_text_cell`
                        // grows a box to `max(own_width, own_height *
                        // FREE_TEXT_MIN_ASPECT)`, which does nothing unless the
                        // box is markedly taller than it is wide -- and of the
                        // 18 still landing here, 11 are not that narrow by the
                        // aspect rule (a 107x134 column widens to 107, i.e. not
                        // at all) while only 7 are blocked by a neighbour. The
                        // measured 107x134 case sits inside a balloon several
                        // times its size, so the box is wrong relative to the
                        // BUBBLE, not relative to its own aspect.
                        //
                        // Which means the real fix is not here: it is making
                        // `bubble_text_cell` yield a bubble-bounded cell instead
                        // of refusing, so the frame comes from the balloon minus
                        // the siblings' territory. A page-global widening is the
                        // wrong tool for text that has a bubble. This arm is
                        // kept because it is a strict improvement and regresses
                        // nothing, not because it is sufficient.
                        None => {
                            let widened =
                                if layer_geometry_is_writable(snapshot, text.layer, generation)? {
                                    /* The cut flag is dropped here on purpose:
                                     * this arm is text INSIDE a balloon, and it
                                     * stays axis-aligned whatever the flag says.
                                     * Dialogue sits on a flat inpainted fill
                                     * inside a contour the compositor also
                                     * derives from this geometry, so rotating it
                                     * would tilt the lettering against the
                                     * balloon that frames it -- a different and
                                     * much less obviously wanted change than
                                     * matching a drawn sound effect to the
                                     * artwork it is painted over. The angle loss
                                     * this file's `free_text_geometry` fixes is
                                     * about free-STANDING text. */
                                    free_text_cell(
                                        text.region.bounds,
                                        size,
                                        &other_text_bounds(regions, index),
                                    )
                                    .map(|(cell, _cut)| cell)
                                } else {
                                    None
                                };
                            (text.region.entity, widened)
                        }
                    }
                };
                edit.relate::<Inside>(text.region.entity, bubble.entity)?;
                edit.relate::<FitsTo>(text.layer, fit)?;
                write_text_role(edit, text.content, "dev.koharu.text.dialogue", generation)?;
                if let Some(cell) = cell {
                    edit.set(text.layer, &rectangle_geometry(cell))?;
                }
            }
        }
        let in_bubble = containing_bubble.is_some();
        let ink = sampled_ink(text.inferred, in_bubble, sampled_ink_lettering);
        /* The halo's contrast decision must see the fill the reader will see.
         * Detection's own sample and the eroded-core sample can disagree by the
         * full tone range -- one measured exhibit reads white one way and dark
         * red the other -- and a stroke chosen against the losing one is
         * decided against a colour that is no longer on the page. */
        let gloss = gloss_ink(text.residue, text.strike, device_ink);
        /* The device ink WINS over the eroded-core sample, and the halo is
         * re-derived against whichever fill actually lands -- the same move the
         * sampled-ink arm below already makes, for the same reason.
         *
         * Winning is the point: the sampler is measured to LOSE this red. The
         * ink's luminance sits inside the paper median's band, so most red
         * pixels fail `ink_within`, the background median becomes the red
         * itself, the "ink" becomes the white outline, and the fill snaps to
         * white. Red survived 1 of 5 plausible boxes. Hoping the sampler
         * recovers an authored colour is exactly what this override replaces. */
        let fill = gloss.or_else(|| ink.as_ref().map(|sample| sample.color));
        let stroke_inferred = match fill {
            Some(color) => text.inferred.map(|mut value| {
                value.color = [color[0], color[1], color[2]];
                value
            }),
            None => text.inferred,
        };
        write_typography(edit, text, text_stroke(stroke_inferred, in_bubble), ink, gloss)?;
    }
    Ok(())
}

/// The own-bounds of every text on the page but this one.
///
/// Every text, not only the ones sharing a bubble: a free-standing text competes
/// for space with whatever else is painted on the artwork around it, in-bubble or
/// not, and its widened box has no balloon edge to stop at.
fn other_text_bounds(regions: &PageRegions, index: usize) -> Vec<[f32; 4]> {
    regions
        .texts
        .iter()
        .enumerate()
        .filter(|(other, _)| *other != index)
        .map(|(_, text)| text.region.bounds)
        .collect()
}

/// A layout box a free-standing text's translation can actually be read in.
///
/// The width -- and only the width -- grows symmetrically about the text's own
/// centre until the box is at least `FREE_TEXT_MIN_ASPECT` as wide as it is tall,
/// then clips to the page. Growing only, never shrinking, keeps the box
/// containing its own ink, which is what lets `bubble_text_cell` be reused
/// verbatim: it cuts the result back at the midpoint between this text and every
/// other on the page, re-centres it on its own ink, and refuses outright rather
/// than hand back a box that no longer covers the glyphs it stands in for.
///
/// `None` when no usable box exists, and the caller must then leave the text
/// exactly as it is today: `FitsTo` its own region, no authored `Geometry`.
/// The geometry to author for a free-standing text's layer.
///
/// **The inferred glyph angle reaches the renderer through this component and
/// through nothing else**, which is the whole reason this function exists.
/// `compositor.rs` prefers a layer's own `Geometry` over the fit frame and takes
/// the angle off it via `geometry_frame`, which reads the top edge -- exactly
/// `atan2(0, width)` = 0.0 for an axis-aligned rectangle. So every free-standing
/// text that received a cell was lettered flat, and the angle `infer_typography`
/// searched +/-45 degrees to find was discarded. `translate_sfx` made that
/// population the drawn sound effects, which are the one class routinely set at an angle:
/// measured on 213 real pages, 577 of 636 onomatopoeia carry `role: free-text`.
///
/// **Rotation is offered only to an UNCUT cell, and that restriction is the
/// entire safety argument.** A cut edge is the line between this text and its
/// neighbour -- it is what stops two free-standing texts being painted over each
/// other, which is what the neighbour cuts were built to fix, and measured (12 of 14
/// overlapping pairs were horizontal free-text). Rotating a rectangle about its centre moves
/// every edge, so rotating a cut cell would gamble that fix. An uncut cell
/// answers to no neighbour: `usable` was empty, nothing was clipped against
/// anything, and the box is simply the ink's own bounds widened. Moving it
/// cannot collide with a text that is not there.
///
/// The population this reaches is 284 of the 949 free-standing regions on that
/// volume, about 30%. The other 70% keep today's flat lettering, deliberately --
/// changing them needs an overlap measurement, not an argument.
///
/// A zero angle takes the same path as before, byte for byte: `infer_typography`
/// snaps anything under `ANGLE_SNAP_DEGREES` to exactly 0.0, and
/// `rotated_rectangle_geometry` at 0.0 is `rectangle_geometry`. The branch is
/// written explicitly anyway so the intent survives a refactor of either.
/// Narrowest column that may be turned on its side.
///
/// In the rotated frame the column's WIDTH becomes the layout box's height, so it
/// has to hold one line: `theme.minimum_font_size` 9.0 times `line_height` 1.2.
/// Below that the rotated box cannot contain a single line at the floor and the
/// overflow is painted sideways across the artwork -- strictly worse than the
/// widened box it replaced. Five of the 366 height-driven regions on one manga
/// baseline are under it, two of them 1.0px wide.
const FREE_TEXT_ROTATED_MIN_WIDTH: f32 = 21.6;

/// The ink's own box, turned 90 degrees, for a tall vertical column.
///
/// **Why this exists.** `free_text_cell` widens a column so horizontal English
/// has room, and on a display column that is ruinous: `FREE_TEXT_MIN_ASPECT`
/// demanded 1015.5px for a 204.5px column purely because it was 1692.5px tall.
/// Turning the box instead makes the constraint the column's HEIGHT rather than
/// its width -- which is what the artist did -- and the footprint becomes the
/// ink's own bbox, so the type lands inside the area the eraser already cleaned.
/// Measured: the size cap is NOT the lever here (2.10x in the JSON and zero
/// pixels in the render); the box is.
///
/// **ANGLE ZERO ONLY, AND THAT RESTRICTION IS THE ENTIRE SAFETY ARGUMENT.**
/// Rotating an `h x w` box by `90 + theta` gives an axis-aligned width of
/// `h*|sin theta| + w*|cos theta|`, which equals the ink only at `theta == 0`: on
/// the exemplar 1.43x the ink at 3 degrees, 2.42x at 10, and WIDER than the
/// 598.10 box it replaces beyond about 14. At zero the rotated quad's hull is the
/// ink bbox exactly, so it can collide with nothing the ink did not already
/// collide with -- which is what lets this run on a cell `free_text_geometry`
/// refuses to rotate. Composing 90 with a non-zero inferred angle would gamble
/// the neighbour-cut overlap fix: on its own test fixture the two rotated hulls
/// overlap by 16.3px at 15 degrees and 140.4px at 45.
///
/// Returns `None` for anything it does not claim, so the caller falls straight
/// through to the widened cell and every other region is untouched.
/// Whether the free-text column TURN is offered on this page at all.
///
/// **Named rather than spelled inline at the call site, because the call site is
/// where this file has been burned before.** A bare `a && (b || c)` in an argument
/// list is testable only by driving the whole stage, so the arm that widens the
/// scope could be deleted and every test here would stay green -- the same hole
/// `stages/ocr.rs` records at length above `withdraw_from_mask`.
///
/// `rotate_free_text_columns` is the master switch and stays first: with it off
/// nothing turns, whatever the page is, which is what keeps the original A/B
/// available.
const fn turn_is_offered(rotate: bool, joined_page: bool, turn_unjoined: bool) -> bool {
    rotate && (joined_page || turn_unjoined)
}

pub(super) fn free_text_column_geometry(
    bounds: [f32; 4],
    inferred: Option<InferredTypography>,
) -> Option<Geometry> {
    if !bounds.iter().all(|value| value.is_finite()) {
        return None;
    }
    let [left, top, right, bottom] = bounds;
    let width = right - left;
    let height = bottom - top;

    // The same predicate `free_text_cell` widens on, so this claims exactly the
    // population the widening would have distorted and not one region more.
    if height * FREE_TEXT_MIN_ASPECT <= width {
        return None;
    }
    let angle = inferred.map_or(0.0, |inferred| inferred.angle_degrees);
    if angle != 0.0 || !angle.is_finite() {
        return None;
    }
    if width < FREE_TEXT_ROTATED_MIN_WIDTH {
        return None;
    }

    // Swap the extents about the ink's own centre, then turn it back. The hull of
    // the result is the ink bbox, to the float.
    let (center_x, center_y) = box_center(bounds);
    Some(rotated_rectangle_geometry(
        [
            center_x - height * 0.5,
            center_y - width * 0.5,
            center_x + height * 0.5,
            center_y + width * 0.5,
        ],
        90.0,
    ))
}

fn free_text_geometry(
    cell: [f32; 4],
    cut: bool,
    inferred: Option<InferredTypography>,
) -> Geometry {
    let angle = inferred.map_or(0.0, |inferred| inferred.angle_degrees);
    if cut || angle == 0.0 || !angle.is_finite() {
        return rectangle_geometry(cell);
    }
    rotated_rectangle_geometry(cell, angle)
}

/// The layout box for a free-standing text, and **whether a neighbour cut it**.
///
/// The flag exists so the caller can tell a cell that is protecting something
/// from one that is merely a widened box. A cut cell's edges are load-bearing --
/// they are the line between this text and the next -- and nothing may move
/// them. An uncut one answers to no neighbour at all, which is what makes it
/// safe to rotate. See the write site.
fn free_text_cell(
    bounds: [f32; 4],
    size: ImageSize,
    neighbours: &[[f32; 4]],
) -> Option<([f32; 4], bool)> {
    if !bounds.iter().all(|value| value.is_finite()) {
        return None;
    }
    let [left, top, right, bottom] = bounds;
    let (center_x, center_y) = box_center(bounds);
    /* The absolute floor applies ONLY to a short box. On a tall column the ratio
     * already gives more than the floor ever could, and applying it there would
     * be the height-blind widening `FREE_TEXT_MIN_WIDTH` describes. Written as
     * a separate `max` so
     * the two rules stay legible and either can be read off a failing case. */
    let height = bottom - top;
    let grown = (height * FREE_TEXT_MIN_ASPECT).max(if height <= FREE_TEXT_SHORT_BOX_HEIGHT {
        FREE_TEXT_MIN_WIDTH
    } else {
        0.0
    });
    let half = (right - left).max(grown) * 0.5;

    /* Only the width grows, so the frame's top and bottom already ARE the ink's.
     * A vertical cut can therefore never widen anything: it either lands beyond
     * the ink and does nothing, or lands inside it and the containment guard
     * throws the whole cell away. That is not a rare corner -- every text on the
     * page is fed in here, so any one of them sitting more below than beside,
     * within a column height, silently refused the widening and left the reader
     * the same 10px text. Keep only the neighbours whose cut can actually do
     * work, plus any whose ink genuinely shares these rows, where a vertical
     * relation is real and refusing is the safe answer. */
    let usable = neighbours
        .iter()
        .filter(|neighbour| {
            let (neighbour_x, neighbour_y) = box_center(**neighbour);
            let shares_rows = neighbour[1] < bottom && neighbour[3] > top;
            (neighbour_x - center_x).abs() >= (neighbour_y - center_y).abs() || shares_rows
        })
        .copied()
        .collect::<Vec<_>>();

    /* Grown symmetrically and clipped to the page only at the END. Clipping
     * first makes the frame asymmetric about the ink, and the centring clamp
     * inside `bubble_text_cell` then keeps the SMALLER half and throws away the
     * side that still had room -- so a column flush against the margin grew by
     * nothing whatsoever. Vertical narration is classically set in the margin,
     * which is precisely the case this exists for. Clipping last cannot break
     * containment, since the ink is inside the page to begin with. */
    let cell = bubble_text_cell([center_x - half, top, center_x + half, bottom], bounds, &usable)?;
    let clipped = [
        cell[0].max(0.0),
        cell[1].max(0.0),
        cell[2].min(size.width as f32),
        cell[3].min(size.height as f32),
    ];

    /* RE-CENTRE ON THE INK, because the clip above only ever bites ONE side.
     *
     * `bubble_text_cell`'s clamp leaves a cut cell symmetric about the ink, and
     * the widening above is symmetric by construction -- but a page clip takes
     * the overhang off whichever side hangs over, and `placement` in
     * `text_renderer.rs` centres the translation in whatever cell it is handed.
     * So a one-sided clip silently walks the English away from the ink it stands
     * in for. Measured on a joined manhua skill-name column: the cell arrived
     * [-268.21, 598.10] symmetric about the ink at x 164.94, the clip made it
     * [0.0, 598.10] centred at 299.05, and the English was lettered 134.10px to
     * the RIGHT of the column -- over artwork the eraser never touched, because
     * the erase mask keys off the raw detection and not this cell.
     *
     * THIS IS NOT THE CLIP-FIRST ORDERING THE COMMENT ABOVE REFUSES, and the
     * difference is the number. Clipping first hands the clamp the UNCLIPPED
     * frame, whose far half is enormous, so the min() picks the bitten side and
     * the column keeps nothing. Re-centring here clamps against the CLIPPED
     * bounds, so the growth that survived the clip survives this too: the
     * exemplar keeps [0.0, 329.88], still 1.61x its own ink, rather than
     * collapsing to the 204.49 the old ordering produced.
     *
     * Floored at the ink's own half-width for the same reason the clamp is: it
     * can only ever SHRINK the cell, and shrinking must not pull an edge inside
     * the text it stands for.
     *
     * ORDER MATTERS AND IT COST A RED TEST TO LEARN IT. The re-centring runs
     * AFTER the containment guard, never before. That floor re-expands the box
     * to the ink's own extent, and where the INK ITSELF hangs off the page --
     * `a_widened_column_is_clipped_to_the_page`'s third case, a 940..980 column
     * on a 977px page -- it walks the right edge back out to 980 and turns a
     * correct refusal into an acceptance. Guarding first keeps every
     * accept/refuse decision exactly as it was, and once the guard has passed
     * `bounds` is known to sit inside both the page and the cell, so the floor
     * cannot reach outside either. */
    if !(clipped[0] <= bounds[0]
        && clipped[1] <= bounds[1]
        && clipped[2] >= bounds[2]
        && clipped[3] >= bounds[3]
        && clipped[2] - clipped[0] > CELL_MINIMUM_EXTENT
        && clipped[3] - clipped[1] > CELL_MINIMUM_EXTENT)
    {
        return None;
    }

    let own_half_width = (bounds[2] - bounds[0]) * 0.5;
    let half_width = (center_x - clipped[0])
        .min(clipped[2] - center_x)
        .max(own_half_width);
    let centred = [
        center_x - half_width,
        clipped[1],
        center_x + half_width,
        clipped[3],
    ];

    /* A degenerate detection keeps the wider box rather than becoming one. Two
     * regions in one manga baseline are 1.0px wide (both refused as
     * punctuation), and re-centring one collapses the
     * cell to its own 1.0px, under `CELL_MINIMUM_EXTENT`. Refusing there would
     * be a THIRD behaviour for a case neither the guard above nor this pass is
     * about, so the cell simply stays where the clip left it. */
    let cell = if centred[2] - centred[0] > CELL_MINIMUM_EXTENT {
        centred
    } else {
        clipped
    };

    Some((cell, !usable.is_empty()))
}

/// The glyph halo for a text, as `(colour, width)`.
///
/// Only free-standing text gets one. Text in a bubble sits on a flat inpainted
/// fill that already separates it from the artwork, and fattening every glyph in
/// every balloon would be a visible regression on pages that are correct today.
///
/// The colour is the local background median `infer_text_color` measures and
/// would otherwise discard -- the non-glyph pixels inside the detection window,
/// which is exactly the thing the glyphs have to stand out from. `None` only
/// when there is no such measurement.
///
/// **Unless the background shares the ink's TONE**: a
/// background-coloured halo works by cutting a band of clean plate around the
/// glyph, but where the plate itself drifts toward the ink -- black brushwork
/// under black lettering -- the only edge left is fill-against-halo, and a halo
/// in the ink's own tone defines nothing. Under `FREE_TEXT_STROKE_TONE_GAP` the
/// halo flips to the opposite tone (white under dark ink, black under light),
/// the published-edition convention for effects over dark art.
///
/// An RGB gate (`3 * 48 * 48` squared distance) used to sit ahead of the flip
/// and answer "fill matches the plate" with NO halo at all. That was tuned for
/// the contrast-ranked fill, which is background-distant by construction, so
/// the gate almost never fired. With the fill genuinely sampled from drawn ink
/// ink-coloured-plate is the COMMON case for effects over dark
/// art, and "no halo" letters invisible text -- the exact defect class on
/// record as "black lettering on a near-black plate". The gate is gone rather
/// than rerouted because it was already subsumed: a squared distance under
/// `3 * 48 * 48` bounds every channel gap under ~83, whose Rec.601 combination
/// is at most 83 -- inside `FREE_TEXT_STROKE_TONE_GAP` -- so every pair the
/// gate refused now takes the tone flip.
pub(super) fn text_stroke(inferred: Option<InferredTypography>, in_bubble: bool) -> Option<([u8; 3], f32)> {
    if in_bubble {
        return None;
    }
    let inferred = inferred?;
    let background = inferred.background?;
    let fill_tone = rec601_luma(inferred.color);
    let halo = if (fill_tone - rec601_luma(background)).abs() < FREE_TEXT_STROKE_TONE_GAP {
        if fill_tone < 128.0 { [255, 255, 255] } else { [0, 0, 0] }
    } else {
        background
    };
    let width = (inferred.font_size * FREE_TEXT_STROKE_RATIO).max(FREE_TEXT_STROKE_MINIMUM_WIDTH);
    width.is_finite().then_some((halo, width))
}

/// Rec.601 luma -- the tone axis `FREE_TEXT_STROKE_TONE_GAP` is measured on.
fn rec601_luma(rgb: [u8; 3]) -> f32 {
    0.299 * f32::from(rgb[0]) + 0.587 * f32::from(rgb[1]) + 0.114 * f32::from(rgb[2])
}

/// What the eroded-core ink sample asks the lettering to carry: its fill, and a
/// bold face when the drawn stroke is heavy.
pub(super) struct SampledInk {
    pub(super) color: [u8; 4],
    pub(super) font_weight: Option<u16>,
}

/// The fill override for free-standing text: letter in the drawn
/// ink's own colour, at a weight matched to its stroke.
///
/// Free-standing only, mirroring `text_stroke`'s scope and for the same reason
/// in reverse: in a bubble the glyphs sit on a flat fill the contrast sample
/// reads correctly, and the standing constraint on this build is that balloon
/// dialogue does not change anywhere. `None` -- no override, the legacy sample
/// stands -- whenever the core was too thin to read, so a region the erosion
/// cannot measure letters exactly as it does today. `enabled` is
/// `--sampled-ink-lettering`, and `false` is the byte-exact control arm.
pub(super) fn sampled_ink(
    inferred: Option<InferredTypography>,
    in_bubble: bool,
    enabled: bool,
) -> Option<SampledInk> {
    if !enabled || in_bubble {
        return None;
    }
    let inferred = inferred?;
    let ink = inferred.ink_color?;
    Some(SampledInk {
        color: [ink[0], ink[1], ink[2], u8::MAX],
        font_weight: inferred
            .ink_stroke_ratio
            .is_some_and(|ratio| ratio >= HEAVY_INK_RATIO)
            .then_some(HEAVY_INK_WEIGHT),
    })
}

/// Writes the `Typography` `write_region` prepared, with `stroke` applied.
///
/// The single write for this layer -- see `DetectedText::typography`. It carries
/// the generation `write_region` stamped on it, and `Edit::set` re-stamps it
/// anyway, so there is nothing further to pass in.
/// The single strike ink this page's device uses, or `None`.
///
/// ONE INK FOR THE WHOLE DEVICE. The licensed reference draws the
/// mark and the replacement name in a single colour, so taking the gloss's fill
/// from the mark makes that structural rather than a constant matching one
/// release. It is read off the page, so it travels to a series inked differently.
///
/// `None` when the page carries no strike -- and, deliberately, when it carries
/// two DIFFERENT ones. This rule assumes one device per page; deciding which
/// mark a given gloss belongs to is a geometry problem nobody has measured, and
/// guessing would letter a name in another device's colour. Failing closed
/// leaves the gloss lettering exactly as it does today: a wrong colour rather
/// than a wrong pairing.
fn device_strike_ink(strikes: impl Iterator<Item = Option<[u8; 4]>>) -> Option<[u8; 4]> {
    let mut inks = strikes.flatten();
    match inks.next() {
        Some(ink) if inks.all(|other| other == ink) => Some(ink),
        Some(_) => {
            tracing::debug!("refusing the gloss ink: the page carries more than one strike colour");
            None
        }
        None => None,
    }
}

/// The ink a text must letter in as a device's gloss, or `None` to letter as it
/// otherwise would.
///
/// The gloss, and ONLY the gloss: a residue region carrying no strike of its
/// own. A struck name keeps the colour it was lettered in, because the device
/// draws red OVER black glyphs and their white outline rather than recolouring
/// them -- so a region with its own `strike` is skipped, and that skip is the
/// difference between completing the device and destroying it.
///
/// `residue` rather than anything read off the region itself: `strike_ink` is a
/// measured negative on the gloss column, so the gloss can never recognise
/// itself, and a rule phrased as "read my own extension" would be a silent
/// no-op.
fn gloss_ink(
    residue: bool,
    own_strike: Option<[u8; 4]>,
    device: Option<[u8; 4]>,
) -> Option<[u8; 4]> {
    (residue && own_strike.is_none()).then_some(device).flatten()
}

fn write_typography(
    edit: &mut koharu_scene::Edit,
    text: &DetectedText,
    stroke: Option<([u8; 3], f32)>,
    ink: Option<SampledInk>,
    /* The strike's own ink, for a gloss that must letter in it.
     * Applied AFTER `ink` so it wins, because the eroded-core sampler is
     * measured to lose this red; see the caller for the measurement. */
    gloss_ink: Option<[u8; 4]>,
) -> Result<()> {
    let Some(typography) = text.typography.as_ref() else {
        return Ok(());
    };
    let mut typography = typography.clone();
    /* THE STRIKE RIDES AS AN EXTENSION, not a field.
     *
     * `Typography` is `#[revisioned]` and every stored scene has to stay
     * byte-compatible, so a new device gets a namespaced key in the open map
     * rather than a new column -- the pattern `FILL_GRADIENT_TO_EXTENSION`
     * already ships and the renderer already knows how to read.
     *
     * Set before the fill below, so a strike and a sampled ink can both land on
     * one layer; they are independent devices and neither should silence the
     * other. */
    if let Some(strike) = text.strike {
        typography.extensions.insert(
            STRIKE_COLOR_EXTENSION.to_owned(),
            format!("{},{},{},{}", strike[0], strike[1], strike[2], strike[3]),
        );
    }
    if let Some(ink) = ink {
        typography.color = Some(ink.color);
        typography.font_weight = ink.font_weight;
    }
    /* Last, so it beats the sampled ink above rather than racing it. The weight
     * is deliberately NOT touched: the authored colour says what ink the device
     * uses, not how heavy the replacement lettering should be. */
    if let Some(color) = gloss_ink {
        typography.color = Some(color);
    }
    typography.stroke_color = stroke.map(|(color, _)| [color[0], color[1], color[2], u8::MAX]);
    typography.stroke_width = stroke.map(|(_, width)| width);
    edit.set(text.layer, &typography)?;
    Ok(())
}

/// The share of `frame` that belongs to the text at `bounds` rather than to any
/// of its `neighbours`: an axis-aligned Voronoi cell over box centres.
///
/// Each neighbour cuts the cell exactly once, on whichever axis its centre
/// offset dominates. Cutting both axes for every neighbour shrinks a cell far
/// below the space its text owns once a bubble holds five or six of them, and
/// cutting only one fixed axis cannot describe a cluster whose horizontal and
/// vertical spreads are comparable.
///
/// Cuts are clamped so a cell always covers the text it belongs to, which makes
/// the result bounded below by that text's own bbox and above by `frame`. It is
/// therefore never worse than the caller's fallback (the bbox alone) and usually
/// much better, so the common case is `Some`.
///
/// `None` only for the two situations no cell can describe: a degenerate box, and
/// coincident centres, where both texts would be handed the identical cell.
fn bubble_text_cell(
    frame: [f32; 4],
    bounds: [f32; 4],
    neighbours: &[[f32; 4]],
) -> Option<[f32; 4]> {
    if !frame.iter().chain(&bounds).all(|value| value.is_finite()) {
        return None;
    }
    let [mut left, mut top, mut right, mut bottom] = frame;
    let (center_x, center_y) = box_center(bounds);
    let mut cut = false;
    for neighbour in neighbours {
        if !neighbour.iter().all(|value| value.is_finite()) {
            continue;
        }
        let (neighbour_x, neighbour_y) = box_center(*neighbour);
        let horizontal = neighbour_x - center_x;
        let vertical = neighbour_y - center_y;
        // Coincident centres -- a horizontal run crossing a vertical one, which
        // NMS keeps because their IoU is tiny -- have no dominant axis. Both
        // sides would then take the same `else` arm at the same midpoint and get
        // the *same* cell, which is precisely the stacking this exists to stop.
        // Refuse, so both fall back to their own regions: separate frames.
        if horizontal == 0.0 && vertical == 0.0 {
            return None;
        }
        cut = true;
        // Each cut is CLAMPED so it can never cross this text's own ink.
        //
        // It used to cut at the raw midpoint and then throw the whole cell away
        // if the result failed to cover the text -- which is a real hazard, but
        // refusing was the wrong answer to it. The caller's fallback is the
        // text's own bbox, and for vertical Japanese that is the source ink
        // column: a strip a few characters wide, which the horizontal English
        // then cannot fit. Measured over 31 real pages, 21 of 254 regions landed
        // there and every layout warning on the set was an overflow at the 9px
        // floor.
        //
        // A clamped cut is strictly better than that fallback at both ends. The
        // cell can never shrink past the ink, so it always contains its own text
        // -- the property the guard below was defending -- and it is still
        // bounded by the frame, so it is never larger than the balloon. Where a
        // neighbour is far away it keeps the room it always kept; where one is
        // close it gives up only the space between the two inks, instead of
        // surrendering the entire balloon and collapsing to the column.
        if horizontal.abs() >= vertical.abs() {
            let middle = (center_x + neighbour_x) * 0.5;
            if horizontal > 0.0 {
                right = right.min(middle.max(bounds[2]));
            } else {
                left = left.max(middle.min(bounds[0]));
            }
        } else {
            let middle = (center_y + neighbour_y) * 0.5;
            if vertical > 0.0 {
                bottom = bottom.min(middle.max(bounds[3]));
            } else {
                top = top.max(middle.min(bounds[1]));
            }
        }
    }

    // Centre the cell on the text it replaces.
    //
    // A neighbour only ever cuts ONE side, so any side no neighbour reached keeps
    // the whole frame. The renderer centres the translation in the cell it is
    // given (`text_renderer.rs` `placement`), so a cell that runs to the far edge
    // of a wide balloon drags the English away from the Japanese it stands in
    // for -- measured at 108px on a six-item cloud balloon, which put one item's
    // translation outside its lobe and over the artwork.
    //
    // Taking the smaller half-extent on each axis makes the cell symmetric about
    // the source text's own centre, so the rendered line lands where the original
    // was. It only ever SHRINKS the cell, and a subset of a non-overlapping cell
    // cannot start overlapping, so the guarantee above is untouched.
    // Only once something actually competed for the bubble. A text with no rival
    // owns the whole balloon and should keep filling it, which is the behaviour
    // every correct page depends on today.
    //
    // Floored at the ink's own half-extents, for the same reason the cuts are
    // clamped: taking the smaller half on each axis is what makes the cell
    // symmetric, but unfloored it can pull an edge back inside the text and undo
    // the containment the clamped cuts just guaranteed. `center` is the centre of
    // `bounds`, so the ink's two half-extents are equal and the floor is simply
    // half its own width and height.
    if cut {
        let own_half_width = (bounds[2] - bounds[0]) * 0.5;
        let own_half_height = (bounds[3] - bounds[1]) * 0.5;
        let half_width = (center_x - left)
            .min(right - center_x)
            .max(own_half_width);
        let half_height = (center_y - top)
            .min(bottom - center_y)
            .max(own_half_height);
        left = center_x - half_width;
        right = center_x + half_width;
        top = center_y - half_height;
        bottom = center_y + half_height;
    }

    let width = right - left;
    let height = bottom - top;
    if !width.is_finite()
        || !height.is_finite()
        || width <= CELL_MINIMUM_EXTENT
        || height <= CELL_MINIMUM_EXTENT
        // Big enough is not the same as over the right pixels. A cut lands at the
        // midpoint of two centres, so it slices into this text's own ink whenever
        // the centre gap is smaller than the ink's extent -- while the far edge
        // still runs to the frame, keeping the cell wide enough to pass a size
        // test. The translation would then be centred well away from the glyphs
        // it replaces. Demand that the cell actually cover its own text.
        || left > bounds[0]
        || top > bounds[1]
        || right < bounds[2]
        || bottom < bounds[3]
    {
        return None;
    }
    Some([left, top, right, bottom])
}

fn box_center([left, top, right, bottom]: [f32; 4]) -> (f32, f32) {
    ((left + right) * 0.5, (top + bottom) * 0.5)
}

/// Whether the pipeline may author this layer's `Geometry`.
///
/// `Edit::set` rejects an overwrite of a user-owned component, so writing a cell
/// over a hand-placed text box would fail the whole detection stage rather than
/// the one layer.
fn layer_geometry_is_writable(
    snapshot: &koharu_scene::Snapshot,
    layer: EntityId,
    generation: &Generation,
) -> Result<bool> {
    // A layer this run has just minted exists only in the in-flight `Edit`, and
    // `Snapshot::component` resolves through the entity's page -- so for one the
    // base snapshot has never seen it answers `EntityNotFound`, not `None`, and
    // the `?` below would fail the whole detection stage. Every text layer is
    // new whenever the scene is built per request, which is exactly what the
    // server does, so this is the ordinary path rather than an edge case. The
    // writes just above encode the same rule as `created || snapshot...`.
    if snapshot.entity(layer).is_err() {
        return Ok(true);
    }
    Ok(snapshot
        .component::<Geometry>(layer)?
        .is_none_or(|geometry| match &geometry.origin {
            Origin::Generated(owner) => owner.producer == generation.producer,
            Origin::User => false,
        }))
}

fn containing_bubble(bubbles: &[DetectedRegion], bounds: [f32; 4]) -> Option<&DetectedRegion> {
    bubbles
        .iter()
        .filter(|bubble| containment(bubble.bounds, bounds) >= 0.5)
        .min_by(|left, right| area(left.bounds).total_cmp(&area(right.bounds)))
}

fn take_previous_text(previous: &mut Vec<PreviousText>, bounds: [f32; 4]) -> Option<PreviousText> {
    let (index, overlap) = previous
        .iter()
        .enumerate()
        .map(|(index, previous)| (index, overlap_over_smaller(previous.bounds, bounds)))
        .max_by(|left, right| left.1.total_cmp(&right.1))?;
    (overlap >= 0.5).then(|| previous.swap_remove(index))
}

fn geometry_bounds(geometry: &Geometry) -> Option<[f32; 4]> {
    let first = geometry.points.first()?;
    let (mut left, mut top, mut right, mut bottom) = (first.x, first.y, first.x, first.y);
    for point in &geometry.points[1..] {
        left = left.min(point.x);
        top = top.min(point.y);
        right = right.max(point.x);
        bottom = bottom.max(point.y);
    }
    Some([left as f32, top as f32, right as f32, bottom as f32])
}

fn write_text_role(
    edit: &mut koharu_scene::Edit,
    entity: EntityId,
    role: &str,
    generation: &Generation,
) -> Result<()> {
    edit.set(
        entity,
        &TextRole {
            origin: Origin::Generated(generation.clone()),
            role: role.to_owned(),
        },
    )?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct InferredTypography {
    pub(super) font_size: f32,
    pub(super) color: [u8; 3],
    /// Median of the non-glyph pixels inside the detection window -- what the
    /// text is painted on. `None` when the mask covered the whole window and
    /// there was nothing to measure. `infer_text_color` computes this to find
    /// the glyph core; it is also the colour a halo has to be to separate the
    /// two, so it is kept rather than dropped.
    pub(super) background: Option<[u8; 3]>,
    pub(super) angle_degrees: f32,
    pub(super) writing_mode: WritingMode,
    /// Median colour of the mask's eroded core (`INK_CORE_DEPTH`), snapped by
    /// `normalize_text_color` like `color`. `None` when the core is thinner
    /// than the erosion or smaller than `INK_CORE_MINIMUM_PIXELS`. Unlike
    /// `color` it cannot be captured by a light rim: the ranking heuristic
    /// keeps the pixels farthest from the background, which on ink whose panel
    /// shares its family is the anti-aliased skirt rather than the ink.
    pub(super) ink_color: Option<[u8; 3]>,
    /// Drawn stroke thickness -- twice the 95th-percentile erosion depth of the
    /// mask -- as a share of `font_size`. `None` whenever `ink_color` is.
    pub(super) ink_stroke_ratio: Option<f32>,
}

#[derive(Clone, Copy)]
struct MaskPoint {
    x: f64,
    y: f64,
}

// BallonsTranslator defines font size as the text-line cross-axis span and
// normalizes vertical-line angles relative to upright vertical text:
// https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/utils/textblock.py#L576-L608
// RF-DETR provides foreground pixels rather than line quadrilaterals, so
// projection-profile sharpness supplies the line axis and the mask projection
// supplies its cross span. Whole-block PCA is deliberately avoided because a
// tall multiline horizontal block otherwise looks vertical.
/// `pub(super)` for the spot rescue: the OCR stage mints a region for a
/// spot-rescued display run and must measure its glyph size, colour and
/// writing mode the same way a detected region's are measured, or the renderer
/// falls back to opaque theme black -- on a black banner that is black on
/// black. The caller synthesises a coarse ink mask INSIDE the rescued box only;
/// this function is never handed a whole page, and it is not a detector.
pub(super) fn infer_typography(
    image: &RgbImage,
    detection: &KoharuLayoutDetection,
) -> Option<InferredTypography> {
    let mask = &detection.mask;
    let width = image.width().min(mask.width);
    let height = image.height().min(mask.height);
    let [left, top, right, bottom] = mask_window(detection.bbox, width, height)?;
    let mut points = Vec::new();
    let mut foreground = Vec::new();
    let mut background = Vec::new();
    // The window kept in place, for the erosion below: the contrast lists lose
    // position, and the core is a spatial fact.
    let window_width = (right - left) as usize;
    let window_height = (bottom - top) as usize;
    let mut window_inside = vec![false; window_width * window_height];
    let mut window_colors = vec![[0u8; 3]; window_width * window_height];
    for y in top..bottom {
        let row = y as usize * mask.width as usize;
        for x in left..right {
            let color = image.get_pixel(x, y).0;
            let window_index =
                (y - top) as usize * window_width + (x - left) as usize;
            window_colors[window_index] = color;
            if mask.pixels.get(row + x as usize).copied().unwrap_or(0) == 0 {
                background.push(color);
                continue;
            }
            window_inside[window_index] = true;
            points.push(MaskPoint {
                x: f64::from(x) + 0.5,
                y: f64::from(y) + 0.5,
            });
            foreground.push(color);
        }
    }
    if points.is_empty() {
        return None;
    }

    let (angle_degrees, vertical) = mask_angle(&points, detection.bbox);
    let font_size = mask_font_size(&points, angle_degrees, vertical)?;
    let (color, background) = infer_text_color(&foreground, &background);
    let (ink_color, ink_stroke_ratio) = infer_ink_core(
        &window_inside,
        window_width,
        window_height,
        &window_colors,
        font_size,
        [left, top, right, bottom],
    );
    Some(InferredTypography {
        font_size,
        color,
        background,
        angle_degrees,
        writing_mode: if vertical {
            WritingMode::Vertical
        } else {
            WritingMode::Horizontal
        },
        ink_color,
        ink_stroke_ratio,
    })
}

/// The ink sample the contrast ranking cannot take: paper first, then the
/// deepest pixels of whatever differs from it.
///
/// `infer_text_color` keeps the quartile of mask pixels FARTHEST from the
/// background median. That heuristic answers "which pixels are not paper", and
/// it inverts exactly when the ink's own colour sits near the panel it is drawn
/// on: the anti-aliased rim is then the most background-distant thing in the
/// mask, and the sample comes back as the rim. One measured scream shipped a
/// `[255,255,255,255]` fill over ink measured at RGB(94,45,45) that way.
///
/// **Eroding the MASK and taking the median of its core does not work, and a
/// rendered chapter showed it**: this detector's mask is REGION-shaped, not
/// glyph-shaped -- the fact `infer_text_color`'s own doc comment states -- so a
/// blob's eroded interior is the paper it stands on, not the ink. One exhibit
/// medianed the light banner and snapped white (a no-op), another medianed a
/// wood-panel/glow mix to a grey-teal the page never drew. So the erosion is
/// applied twice, to two
/// different questions:
///
/// 1. **The paper** is the median of the mask's eroded interior -- a region is
///    mostly paper by area, the same fact `mask_for`'s doc records for
///    balloons. (Whole mask when the region is too thin to erode.)
/// 2. **The ink-plus-halo** is every mask pixel whose colour differs from that
///    paper by at least `INK_MIN_CONTRAST`.
/// 3. **The ink** is the eroded core of THAT set: a drawn halo hugs the stroke
///    from outside, so the stroke is the interior. Anti-aliasing cannot be two
///    pixels deep.
/// 4. **Dark over light only** (`INK_DARKER_THAN_PAPER_MARGIN`): when the
///    mask's interior lands on the glyphs instead of the paper, steps 1-3 run
///    with the roles swapped and nominate the gaps -- so a core LIGHTER than
///    its paper abstains rather than shipping the inverse.
///
/// A mask that is all ink (a synthesised region's mask IS its ink)
/// finds nothing differing from its own "paper" and abstains -- the legacy
/// sample already reads those correctly. The window border counts as outside,
/// so a glyph the window clips erodes from that edge too.
fn infer_ink_core(
    inside: &[bool],
    width: usize,
    height: usize,
    colors: &[[u8; 3]],
    font_size: f32,
    /* Page-space window `[left, top, right, bottom]`, for the diagnostic
     * lines only: the sampler's intermediates are invisible from the wire
     * (masks are not stored), so the gate design needs these numbers read
     * from a real render rather than guessed. */
    window: [u32; 4],
) -> (Option<[u8; 3]>, Option<f32>) {
    debug_assert_eq!(inside.len(), width * height);
    debug_assert_eq!(colors.len(), width * height);
    if inside.is_empty() {
        tracing::debug!(?window, verdict = "empty_window", "ink core");
        return (None, None);
    }

    let mask_depth = chebyshev_depth(inside, width, height);
    let mut paper_core = Vec::new();
    for (index, inked) in inside.iter().enumerate() {
        if *inked && mask_depth[index] >= INK_CORE_DEPTH {
            paper_core.push(colors[index]);
        }
    }
    let paper = if paper_core.len() >= INK_CORE_MINIMUM_PIXELS {
        median_color(&paper_core)
    } else {
        let all = inside
            .iter()
            .enumerate()
            .filter(|(_, inked)| **inked)
            .map(|(index, _)| colors[index])
            .collect::<Vec<_>>();
        median_color(&all)
    };

    let differs = inside
        .iter()
        .enumerate()
        .map(|(index, inked)| {
            *inked && color_distance_squared(colors[index], paper) >= INK_MIN_CONTRAST
        })
        .collect::<Vec<_>>();
    let ink_depth = chebyshev_depth(&differs, width, height);

    let mut core = Vec::new();
    let mut depths = Vec::new();
    for (index, differing) in differs.iter().enumerate() {
        if !differing {
            continue;
        }
        depths.push(ink_depth[index]);
        if ink_depth[index] >= INK_CORE_DEPTH {
            core.push(colors[index]);
        }
    }
    let differing = depths.len();
    if core.len() < INK_CORE_MINIMUM_PIXELS {
        tracing::debug!(
            ?window,
            ?paper,
            differing,
            core = core.len(),
            verdict = "sparse_core",
            "ink core"
        );
        return (None, None);
    }
    // p95 rather than max: a single blob-shaped defect in the differing set
    // would otherwise set the thickness for the whole region.
    depths.sort_unstable();
    let thickness = 2.0 * depths[(depths.len() - 1) * 95 / 100] as f32;
    // Thin strokes erode to an anti-aliased blend, not the ink. See
    // `INK_SAMPLE_MINIMUM_THICKNESS_PX` for the pages that earned this.
    if thickness < INK_SAMPLE_MINIMUM_THICKNESS_PX {
        tracing::debug!(
            ?window,
            ?paper,
            differing,
            core = core.len(),
            thickness,
            verdict = "thin_stroke",
            "ink core"
        );
        return (None, None);
    }
    let color = normalize_text_color(median_color(&core));
    // Dark-over-light only -- the one polarity where the paper/ink roles
    // cannot have swapped. See `INK_DARKER_THAN_PAPER_MARGIN` for the page
    // that earned this.
    if rec601_luma(color) + INK_DARKER_THAN_PAPER_MARGIN > rec601_luma(paper) {
        tracing::debug!(
            ?window,
            ?paper,
            ?color,
            differing,
            core = core.len(),
            thickness,
            verdict = "light_over_dark",
            "ink core"
        );
        return (None, None);
    }
    let ratio = (font_size.is_finite() && font_size > 0.0).then(|| thickness / font_size);
    tracing::debug!(
        ?window,
        ?paper,
        ?color,
        differing,
        core = core.len(),
        thickness,
        ?ratio,
        verdict = "sampled",
        "ink core"
    );
    (Some(color), ratio)
}

/// Two-pass Chebyshev distance to the nearest pixel outside `inside`, with the
/// window border counting as outside. A member's depth is at least 1; a
/// non-member's is 0.
fn chebyshev_depth(inside: &[bool], width: usize, height: usize) -> Vec<u32> {
    let mut depth = vec![0u32; inside.len()];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if !inside[index] {
                continue;
            }
            let up = if y > 0 { depth[index - width] } else { 0 };
            let left = if x > 0 { depth[index - 1] } else { 0 };
            let up_left = if y > 0 && x > 0 { depth[index - width - 1] } else { 0 };
            let up_right = if y > 0 && x + 1 < width { depth[index - width + 1] } else { 0 };
            depth[index] = up.min(left).min(up_left).min(up_right).saturating_add(1);
        }
    }
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let index = y * width + x;
            if !inside[index] {
                continue;
            }
            let down = if y + 1 < height { depth[index + width] } else { 0 };
            let right = if x + 1 < width { depth[index + 1] } else { 0 };
            let down_left = if y + 1 < height && x > 0 { depth[index + width - 1] } else { 0 };
            let down_right = if y + 1 < height && x + 1 < width { depth[index + width + 1] } else { 0 };
            let through = down.min(right).min(down_left).min(down_right).saturating_add(1);
            depth[index] = depth[index].min(through);
        }
    }
    depth
}

/// Write each settled detection's mask, cropped to its own bbox window, as a
/// PNG under `dir` — `p<page>_r<index>_<label>_x<left>y<top>w<w>h<h>.png`, so
/// the page-space placement survives in the name after the crop discards it.
/// Debug-only (`ProcessorConfig::debug_mask_dir`); the caller downgrades any
/// error to a `tracing::warn!` because an instrument must not kill the run it
/// is instrumenting. Cropped rather than page-sized because the RF-DETR
/// mask is full-page resolution, and a per-region dump
/// of whole pages is dozens of mostly-black images per page.
fn dump_settled_masks(
    dir: &Path,
    page: impl std::fmt::Display,
    detections: &settle::Settled,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    // `EntityId`'s Display is not promised filename-safe; ':' alone would
    // break every name on Windows.
    let page: String = page
        .to_string()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    for (index, detection) in detections.iter().enumerate() {
        let mask = &detection.mask;
        let Some([left, top, right, bottom]) = mask_window(detection.bbox, mask.width, mask.height)
        else {
            continue;
        };
        let (w, h) = (right - left, bottom - top);
        let mut crop = GrayImage::new(w, h);
        for y in top..bottom {
            for x in left..right {
                let value = mask.pixels[(y * mask.width + x) as usize];
                crop.put_pixel(x - left, y - top, Luma([value]));
            }
        }
        let name = format!(
            "p{page}_r{index:02}_{label}_x{left}y{top}w{w}h{h}.png",
            label = detection.label,
        );
        crop.save(dir.join(name))
            .with_context(|| format!("saving debug mask for region {index}"))?;
    }
    Ok(())
}

fn mask_window([left, top, right, bottom]: [f32; 4], width: u32, height: u32) -> Option<[u32; 4]> {
    if width == 0 || height == 0 {
        return None;
    }
    let left = left.floor().clamp(0.0, width as f32) as u32;
    let top = top.floor().clamp(0.0, height as f32) as u32;
    let right = right.ceil().clamp(0.0, width as f32) as u32;
    let bottom = bottom.ceil().clamp(0.0, height as f32) as u32;
    (right > left && bottom > top).then_some([left, top, right, bottom])
}

fn mask_angle(points: &[MaskPoint], [left, top, right, bottom]: [f32; 4]) -> (f32, bool) {
    let mut horizontal = (f64::NEG_INFINITY, 0.0);
    let mut vertical = (f64::NEG_INFINITY, 0.0);
    for step in -ANGLE_SEARCH_HALF_STEPS..=ANGLE_SEARCH_HALF_STEPS {
        let angle_degrees = f64::from(step) * ANGLE_SEARCH_STEP_DEGREES;
        let (sin, cos) = angle_degrees.to_radians().sin_cos();
        let horizontal_score = projection_score(points, -sin, cos);
        if horizontal_score > horizontal.0 {
            horizontal = (horizontal_score, angle_degrees);
        }
        let vertical_score = projection_score(points, cos, sin);
        if vertical_score > vertical.0 {
            vertical = (vertical_score, angle_degrees);
        }
    }
    let maximum_score = horizontal.0.max(vertical.0);
    let scores_are_close = (horizontal.0 - vertical.0).abs() <= maximum_score * 0.02;
    let is_vertical = if scores_are_close {
        bottom - top > right - left
    } else {
        vertical.0 > horizontal.0
    };
    let mut angle = if is_vertical {
        vertical.1
    } else {
        horizontal.1
    } as f32;
    if angle.abs() < ANGLE_SNAP_DEGREES {
        angle = 0.0;
    }
    (angle, is_vertical)
}

fn projection_score(points: &[MaskPoint], axis_x: f64, axis_y: f64) -> f64 {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for point in points {
        let projection = point.x * axis_x + point.y * axis_y;
        minimum = minimum.min(projection);
        maximum = maximum.max(projection);
    }
    let origin = minimum.floor();
    let length = (maximum.ceil() - origin).max(0.0) as usize + 2;
    let mut profile = vec![0.0; length];
    for point in points {
        let projection = point.x * axis_x + point.y * axis_y - origin;
        let index = projection.floor() as usize;
        let fraction = projection - index as f64;
        profile[index] += 1.0 - fraction;
        profile[index + 1] += fraction;
    }
    profile.iter().map(|value| value * value).sum::<f64>() / points.len() as f64
}

fn mask_font_size(points: &[MaskPoint], angle_degrees: f32, vertical: bool) -> Option<f32> {
    let line_angle = f64::from(angle_degrees).to_radians()
        + if vertical {
            std::f64::consts::FRAC_PI_2
        } else {
            0.0
        };
    let cross_x = -line_angle.sin();
    let cross_y = line_angle.cos();
    let projections = points
        .iter()
        .map(|point| point.x * cross_x + point.y * cross_y)
        .collect::<Vec<_>>();
    let minimum = projections.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = projections
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let length = (maximum.ceil() - minimum.floor()).max(0.0) as usize + 1;
    let mut occupied = vec![false; length];
    for projection in projections {
        let index = (projection - minimum.floor()).floor() as usize;
        occupied[index.min(length - 1)] = true;
    }
    close_short_projection_gaps(&mut occupied, 2);

    let mut spans = Vec::new();
    let mut index = 0;
    while index < occupied.len() {
        if !occupied[index] {
            index += 1;
            continue;
        }
        let start = index;
        while index < occupied.len() && occupied[index] {
            index += 1;
        }
        spans.push((index - start) as u32);
    }
    let maximum = spans.iter().copied().max()?;
    spans.retain(|span| *span * 2 >= maximum);
    spans.sort_unstable();
    let middle = spans.len() / 2;
    let size = if spans.len().is_multiple_of(2) {
        (spans[middle - 1] + spans[middle]) as f32 * 0.5
    } else {
        spans[middle] as f32
    };
    Some(size.max(1.0))
}

fn close_short_projection_gaps(occupied: &mut [bool], maximum_gap: usize) {
    let mut index = 0;
    while index < occupied.len() {
        if occupied[index] {
            index += 1;
            continue;
        }
        let start = index;
        while index < occupied.len() && !occupied[index] {
            index += 1;
        }
        if start > 0 && index < occupied.len() && index - start <= maximum_gap {
            occupied[start..index].fill(true);
        }
    }
}

fn median_channel(values: &mut [u8]) -> u8 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        ((u16::from(values[middle - 1]) + u16::from(values[middle])) / 2) as u8
    } else {
        values[middle]
    }
}

// BallonsTranslator normally receives foreground colors predicted per OCR line:
// https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/modules/ocr/mit48px.py#L202-L210
// This detector has only a segmentation mask, so the most background-distant
// mask pixels approximate the solid glyph core without antialiasing bias.
//
// Returns the background median alongside the glyph colour. It has to be
// computed to rank the foreground at all, and it is the only measurement of what
// the text is painted on that this stage ever takes -- so a caller wanting a
// halo would otherwise have to walk the mask a second time to recover it.
fn infer_text_color(foreground: &[[u8; 3]], background: &[[u8; 3]]) -> ([u8; 3], Option<[u8; 3]>) {
    if background.is_empty() {
        return (normalize_text_color(median_color(foreground)), None);
    }
    let background = median_color(background);
    let mut foreground = foreground.to_vec();
    foreground.sort_unstable_by(|left, right| {
        color_distance_squared(*right, background).cmp(&color_distance_squared(*left, background))
    });
    let core = foreground.len().div_ceil(4).max(1);
    (
        normalize_text_color(median_color(&foreground[..core])),
        Some(background),
    )
}

fn median_color(colors: &[[u8; 3]]) -> [u8; 3] {
    std::array::from_fn(|channel| {
        let mut values = colors
            .iter()
            .map(|color| color[channel])
            .collect::<Vec<_>>();
        median_channel(&mut values)
    })
}

/// The INK inside a region's mask: the pixels that differ from the paper around them.
///
/// **A synthesised region's mask is its BALLOON, and that one fact caused both of the
/// defects a render exposed.** `mask_for` unions the mask into the erase
/// mask, so the balloon body was inpainted away -- 79.7% of its white destroyed on
/// one test page -- and `infer_typography` measures colour and size from the same
/// pixels, so it reported the *paper*: `font_size 322.0` and `color [255,255,255]`,
/// white lettering on a white balloon. One mask, two consumers, one wrong answer each.
///
/// So the mask becomes the ink, and both consumers are right for the same reason.
/// They read one source and **cannot disagree** -- which is the property
/// `settle_detections` was already reaching for when it cloned the bubble's mask.
///
/// The paper is the MEDIAN luminance of the masked pixels, not the mean: a balloon is
/// mostly paper by area, so the median lands on it however much ink is present, while
/// a mean is dragged by the glyphs it is supposed to be measuring against.
///
/// Returns `None` when there is not enough ink to be text -- see
/// [`SYNTHESISED_INK_MIN_PIXELS`], which is what keeps a spurious bubble on flat
/// artwork from being handed a region.
fn ink_within(image: &RgbImage, detection: &KoharuLayoutDetection) -> Option<KoharuLayoutMask> {
    let mask = &detection.mask;
    let width = image.width().min(mask.width);
    let height = image.height().min(mask.height);
    let [left, top, right, bottom] = mask_window(detection.bbox, width, height)?;
    let luminance = |color: [u8; 3]| -> u16 {
        color.iter().copied().map(u16::from).sum::<u16>() / 3
    };

    let mut paper = Vec::new();
    for y in top..bottom {
        let row = y as usize * mask.width as usize;
        for x in left..right {
            if mask.pixels.get(row + x as usize).copied().unwrap_or(0) != 0 {
                paper.push(luminance(image.get_pixel(x, y).0));
            }
        }
    }
    if paper.is_empty() {
        return None;
    }
    // The balloon's own area in mask pixels -- the denominator for `ink_frac`
    // below. Taken before `paper` is shadowed by its own median.
    let masked = paper.len();
    paper.sort_unstable();
    let paper = paper[paper.len() / 2];

    let mut pixels = vec![0u8; mask.pixels.len()];
    let mut ink = 0usize;
    for y in top..bottom {
        let row = y as usize * mask.width as usize;
        for x in left..right {
            let index = row + x as usize;
            if mask.pixels.get(index).copied().unwrap_or(0) == 0 {
                continue;
            }
            if luminance(image.get_pixel(x, y).0).abs_diff(paper) >= SYNTHESISED_INK_LUMA_DELTA {
                pixels[index] = u8::MAX;
                ink += 1;
            }
        }
    }
    /* EVERY BUBBLE MEASURED, ACCEPTED OR NOT.
     *
     * If only what was ACCEPTED were logged, a refusal by the floor would be
     * silent -- indistinguishable from "no bubble was there" and from "the filter
     * is off" -- and the question *"is the ink floor refusing this bubble?"* could
     * not be answered from the log; it would have to be reconstructed from
     * containment arithmetic off a separate census.
     *
     * `ink_frac` is the scale-free form, and the quantity the fraction floor gates.
     * Measured on the two test pages that reach this: the genuine balloon is
     * **40,946 / 473,442 = 8.65%**, the spurious bubble over bare skin is
     * **795 / 58,400 = 1.36%**. The absolute floor sits at 200, which is 51x below
     * the false positive -- it separates neither. */
    let ink_frac = if masked == 0 {
        0.0
    } else {
        ink as f64 / masked as f64
    };
    let accepted = ink >= SYNTHESISED_INK_MIN_PIXELS && ink_frac >= SYNTHESISED_INK_MIN_FRACTION;
    tracing::debug!(
        score = detection.score,
        bbox = ?detection.bbox,
        ink,
        masked,
        ink_frac,
        accepted,
        "measured the ink inside a bubble"
    );
    accepted.then(|| KoharuLayoutMask {
        width: mask.width,
        height: mask.height,
        pixels,
    })
}

/// The colour of a strike-through mark drawn across this detection, if there is
/// one.
///
/// ## Why a colour histogram and not a line finder
///
/// A page-scale projection profile cannot separate parallel blades from glyph
/// rows, and one measured page shows why: it carries a 25,597 px red art mass and a
/// 17,631 px red drip, both far larger than its 6,219 px strike. Both sit OUTSIDE
/// the region box, though -- 6,219 in-box against 55,767 on the page -- so a test
/// scoped to one region is not the page-scale test that failed.
///
/// ## Why not "find the saturated pixels"
///
/// Tried and refuted in pixels before this was written. These pages are washed in
/// purple and magenta: a chroma floor returns 17,020 pixels inside one page's name
/// column against the strike's 4,132, and their median colour comes back PURPLE.
/// Saturation is not the signal on artwork. Being one colour, in a thin line, is.
///
/// That stays true, and `STRIKE_MIN_CHROMA` at the bottom of this function is not
/// it coming back. The refuted idea used chroma to CHOOSE the pixels, and chose
/// artwork. This tests the colour of the bucket the geometry has already picked,
/// and only to throw away marks that have no colour at all -- the black borders
/// and white gutters the shape test cannot tell from a strike. One is a search,
/// the other is a refusal, and they fail in opposite directions.
///
/// ## The test
///
/// Bin the window's colours coarsely, then ask of every bucket holding enough
/// pixels: is its footprint long and thin, solid within itself, and does it run
/// most of the way along the region? Artwork fails the last two together even
/// when it is redder and more plentiful than the mark.
///
/// The colour returned is the MEDIAN of the bucket's real pixels rather than the
/// bucket's centre -- a bucket is 16 levels wide a channel, and the mark is
/// re-drawn in whatever this returns.
fn strike_ink(image: &RgbImage, detection: &KoharuLayoutDetection) -> Option<[u8; 4]> {
    let [left, top, right, bottom] = mask_window(detection.bbox, image.width(), image.height())?;
    let (span_x, span_y) = ((right - left) as f32, (bottom - top) as f32);
    if span_x <= 0.0 || span_y <= 0.0 {
        return None;
    }
    let shift = 8 - STRIKE_COLOR_BITS;
    let bucket = |color: [u8; 3]| -> u32 {
        color.iter().fold(0_u32, |key, channel| {
            (key << STRIKE_COLOR_BITS) | u32::from(channel >> shift)
        })
    };
    /* One pass to bin, tracking each bucket's footprint as it goes: the count and
     * the four extremes are all the tests below need, so nothing keeps a pixel
     * list and the window is walked once. */
    let mut buckets: BTreeMap<u32, (usize, u32, u32, u32, u32)> = BTreeMap::new();
    for y in top..bottom {
        for x in left..right {
            let entry = buckets
                .entry(bucket(image.get_pixel(x, y).0))
                .or_insert((0, u32::MAX, u32::MAX, 0, 0));
            entry.0 += 1;
            entry.1 = entry.1.min(x);
            entry.2 = entry.2.min(y);
            entry.3 = entry.3.max(x);
            entry.4 = entry.4.max(y);
        }
    }
    let mut best: Option<(usize, u32)> = None;
    for (key, (count, x0, y0, x1, y1)) in &buckets {
        if *count < STRIKE_MIN_PIXELS {
            continue;
        }
        let width = (x1 - x0 + 1) as f32;
        let height = (y1 - y0 + 1) as f32;
        let long = width.max(height);
        let short = width.min(height);
        if short <= 0.0 {
            continue;
        }
        let fits = long / short >= STRIKE_MIN_ASPECT
            && *count as f32 / (width * height) >= STRIKE_MIN_FILL
            && long / span_x.max(span_y) >= STRIKE_MIN_SPAN;
        if fits && best.is_none_or(|(seen, _)| *count > seen) {
            best = Some((*count, *key));
        }
    }
    let (count, key) = best?;
    /* The bucket's real median, channel by channel. A second pass, and it runs
     * only for the one bucket that already cleared every test. */
    let mut channels: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for y in top..bottom {
        for x in left..right {
            let pixel = image.get_pixel(x, y).0;
            if bucket(pixel) == key {
                for (slot, value) in channels.iter_mut().zip(pixel) {
                    slot.push(value);
                }
            }
        }
    }
    let mut color = [0_u8, 0, 0, u8::MAX];
    for (slot, values) in color.iter_mut().zip(channels.iter_mut()) {
        values.sort_unstable();
        *slot = values[values.len() / 2];
    }
    /* THE MARK MUST BE DRAWN IN A COLOUR, and this is the last gate rather than
     * the first for a reason -- see the docstring's "Why not find the saturated
     * pixels", which is about the opposite thing and is still true.
     *
     * Measured over a whole 179-slice test chapter before this existed: the
     * geometry alone found FOURTEEN marks, of which TWO were the author's --
     * `(249, 3, 85)` and `(251, 1, 86)` -- and twelve were page
     * furniture. Black panel borders up to 297,162 px, white gutters 56 px wide,
     * and a 138 x 4 px sliver lying along the top edge of a page. Long, thin,
     * solid and one colour describes a border exactly as well as it describes a
     * strike.
     *
     * The separation is not delicate: every false mark came back with a chroma of
     * 0 or 1, both real ones above 245. So the test is whether the mark has a
     * colour at all, not how saturated it is.
     *
     * WHAT THIS GIVES UP, stated rather than discovered later: an authored strike
     * drawn in black on a monochrome page is now out of scope. That is the right
     * direction to fail. A miss leaves the page exactly as it ships today, while
     * a false positive paints a line THROUGH a reader's lettering -- observed on
     * a test page, a black stroke across "GALE KING", a name nobody struck. */
    let chroma = color[..3].iter().copied().max().unwrap_or_default()
        - color[..3].iter().copied().min().unwrap_or_default();
    if chroma <= STRIKE_MIN_CHROMA {
        /* Logged, never silent: this file already paid for a floor that refused
         * without saying so, where "refused" and "nothing was there" and "the
         * filter is off" all looked identical from the log. */
        tracing::debug!(
            bbox = ?detection.bbox,
            count,
            color = ?color,
            chroma,
            "refusing a strike mark: it is drawn in no colour, so it is page furniture"
        );
        return None;
    }
    tracing::info!(
        bbox = ?detection.bbox,
        count,
        color = ?color,
        chroma,
        "found a strike mark drawn across a region"
    );
    Some(color)
}

fn color_distance_squared(left: [u8; 3], right: [u8; 3]) -> u32 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| i32::from(left) - i32::from(right))
        .map(|difference| difference.unsigned_abs().pow(2))
        .sum()
}

fn normalize_text_color(color: [u8; 3]) -> [u8; 3] {
    let minimum = color.iter().copied().min().unwrap_or_default();
    let maximum = color.iter().copied().max().unwrap_or_default();
    let luminance = color.iter().copied().map(u16::from).sum::<u16>() / 3;
    if maximum - minimum <= COLOR_SNAP_CHROMA && luminance <= COLOR_SNAP_DARK_LUMINANCE {
        [0, 0, 0]
    } else if maximum - minimum <= COLOR_SNAP_CHROMA && luminance >= COLOR_SNAP_LIGHT_LUMINANCE {
        [u8::MAX; 3]
    } else {
        color
    }
}

/// The shape a region carries downstream -- and the one the OCR gate measures.
///
/// **Free-standing, and named, because a test of the branch INSIDE `write_region`
/// cannot fail for the reason the caller fails.** The predecessor of this function
/// was three arms of an `if` inside `write_region`, and the test that pinned it
/// (`the_synthesised_region_survives_being_turned_again`) re-implemented those arms
/// op for op in a local closure. It stayed green through every change to the real
/// branch, which is the classic trap: name the composed predicate and
/// assert on what the caller actually calls.
///
/// # A SYNTHESISED region and its bubble are ONE OBJECT, so they take ONE SHAPE
///
/// `settle_detections` clones the bubble's mask deliberately -- *"so the eraser and
/// the rule cannot disagree about the shape"* -- and the first arm below finishes
/// that thought. The same mask now yields the same polygon on both, instead of a
/// polygon on the bubble and a rotated rectangle on the text standing in front of it.
///
/// **It is the polygon, not a better rectangle, that clears the OCR gate.**
/// `region_extent` measures the axis-aligned HULL of whatever geometry arrives, and
/// a polygon's hull is its own AABB -- there is no rectangle left over to inflate it.
/// Turning the box to the balloon's own axes reached an oriented 1113.5 x 320.8 =
/// 0.3279, and STILL measured **0.5883** of the test page, because a hull is not a
/// box: any rectangle covering this mask spends more than the 0.0658 of margin the
/// mask's own AABB leaves.
///
/// **The fallback is the plain bbox, NOT the turned one, and that is a change.**
/// `mask_geometry` returns None only for a mask with no usable contour -- and
/// `oriented_ink_box` returns None on that same mask, so the old fallback chain
/// collapsed to `rotated_rectangle_geometry(bbox, angle)`: the DOUBLE-COUNTED
/// rotation that measured 0.6957, which turning the box exists to remove, reinstated in
/// precisely the case no render ever covers.
///
/// NOT extended to ordinary rotated regions. That is a known open defect, it
/// costs 19 erasures on a measured baseline, and it is a shipping default changing
/// what every reader sees -- it wants its own measurement rather than a free ride on
/// a flag that ships OFF. `oriented_ink_box` is kept for it.
fn region_geometry(
    detection: &KoharuLayoutDetection,
    letters: bool,
    synthesised: bool,
    inferred: Option<InferredTypography>,
) -> Geometry {
    if detection.label == "bubble" {
        mask_geometry(&detection.mask).unwrap_or_else(|| rectangle_geometry(detection.bbox))
    } else if synthesised {
        /* THE BALLOON'S OWN BOX, PLAIN AND UNTURNED.
         *
         * The mask's polygon was right here only while the mask WAS the balloon.
         * It no longer is -- `ink_within` makes it the glyphs -- and
         * `mask_geometry` keeps only the largest contour, so taking it now would hand
         * OCR a crop of the single biggest letter.
         *
         * The bbox is still the balloon's, unshrunk, and it still clears the gate:
         * 1075.8 x 439.8 = **0.4342** of the test page, under the 0.5
         * `--large-crop-ocr-max-area` ceiling. That is the measured FLOOR -- "any
         * box covering the mask has a hull of at least 0.4342" -- reached here
         * directly instead of through a polygon. What fails is the ROTATED
         * rectangle, whose hull is 0.5883 because turning an axis-aligned box can
         * only inflate it. This is not that: it is not turned at all, so there is
         * nothing to inflate.
         *
         * The lettering still follows the balloon, and not this box -- the angle
         * travels on `Typography` and the text `FitsTo` the BUBBLE, whose geometry is
         * its own mask polygon. Shape and gate are simply different questions. */
        rectangle_geometry(detection.bbox)
    } else if letters {
        inferred.map_or_else(
            || rectangle_geometry(detection.bbox),
            |typography| rotated_rectangle_geometry(detection.bbox, typography.angle_degrees),
        )
    } else {
        rectangle_geometry(detection.bbox)
    }
}

fn rectangle_geometry([left, top, right, bottom]: [f32; 4]) -> Geometry {
    Geometry::rectangle(
        f64::from(left),
        f64::from(top),
        f64::from((right - left).max(1.0)),
        f64::from((bottom - top).max(1.0)),
    )
}

/// The ink measured along **its own axes**, as the box whose turn by
/// `angle_degrees` REPRODUCES that ink -- rather than the axis-aligned hull of a
/// shape that is already tilted.
///
/// # Why a synthesised region cannot just keep the bubble's bbox
///
/// A synthesised region is labelled `text`, so [`build_region`] takes the
/// `letters` branch and turns its bbox about its centre. **Turning an
/// already-axis-aligned rectangle can only INFLATE its hull** -- this file says so
/// at `a_rotated_region_the_reader_refuses_for_size_is_erased_anyway` -- and
/// `ocr.rs`'s `region_extent` measures exactly that hull.
///
/// Measured on the test page this exists for:
///
/// | | box | share of page |
/// |---|---|---|
/// | bubble detection, axis-aligned | 1075.78 x 439.81 | 0.4342 |
/// | that box handed over, turned 12.5 degrees | 1145.47 x 662.23 | **0.6957** |
/// | this box, turned the same 12.5 degrees | 1032 x 392 | **0.3713** |
///
/// 0.6957 is over the 0.5 `large_crop_ocr_max_area` ceiling, so
/// `skip_implausible_regions` refused the region before any engine read it and the
/// render came back byte-identical to the OFF arm. That figure is not an estimate:
/// it is the rendered run's own JSON, to seven decimals.
///
/// # Shrinking to the ink's HULL does not fix it, and that is the trap
///
/// The obvious repair -- hand over the balloon's white body, **1032 x 392 =
/// 0.3713**, comfortably under the ceiling -- **does not reach**. 0.3713 is the
/// hull of a balloon that is ALREADY turned; feeding it back as an axis-aligned
/// box applies the same 12.5 degrees a second time and arrives at 1092.3 x 605.8 =
/// **0.6073**, still refused. The rotation would be counted twice.
///
/// So what this returns is the extent along the ink's own axes -- about 1018 x 176
/// -- positioned so that turning it lands the hull back on 1032 x 392. The idiom is
/// [`free_text_column_geometry`]'s, *"the hull of the result is the ink bbox, to
/// the float"*, and the arithmetic is the exact inverse of
/// [`rotated_rectangle_geometry`]'s, so the two cannot drift apart without one of
/// them failing its test.
///
/// # THE ANGLE IS A PARAMETER, AND THAT IS THE WHOLE DESIGN
///
/// It would be far tidier to measure the angle here. **It was built that way
/// first, in `settle_detections`, and it does not work.** `infer_typography`
/// re-measures the angle downstream through `mask_window` on whatever box it is
/// given, so a box already shrunk to the ink's own axes hands that pass a
/// **clipped** balloon -- and the clip is axis-aligned, which biases the
/// measurement back toward zero. Measured on the fixture: **12.5 degrees on the
/// whole mask against 6.5 on the shrunken box**, for a hull of 1031 x 290 where
/// the ink is 1032 x 392. That clears the ceiling and cuts 102px off the balloon,
/// and since the hull is also the OCR crop, those are glyphs lost before any
/// engine sees them.
///
/// So the caller measures once, on the whole mask, and passes that one angle here
/// and to [`rotated_rectangle_geometry`]. One measurement used twice cannot
/// disagree with itself; two measurements of two different windows did.
///
/// Returns `None` -- meaning "keep the box you have" -- when there is nothing to
/// win: no ink, a non-finite or snapped-to-zero angle (at zero the oriented extent
/// IS the axis-aligned bbox, so the round trip could only lose floats), or a
/// result that does not actually come out smaller. That last guard is cheap and
/// the alternative is arguing; a case where this does not shrink is a case this
/// reasoning does not cover.
///
/// # OFF THE SHIPPING PATH, AND KEPT ON PURPOSE
///
/// Its one production caller was `region_geometry`'s synthesised arm, which no
/// longer turns a box at all. This is therefore dead code, and it is retained
/// rather than deleted because **an open defect still needs it**: ordinary
/// rotated regions still inflate their hulls, at a measured cost of 19 erasures,
/// and that fix is a shipping default that wants its own render. Deleting a
/// measured, tested tool an open item needs would cost more than the attribute
/// below.
///
/// It is pinned by `the_oriented_box_still_measures_the_recorded_extent` so it
/// cannot rot silently while unused.
#[allow(dead_code)]
fn oriented_ink_box(
    image: &RgbImage,
    detection: &KoharuLayoutDetection,
    angle_degrees: f32,
) -> Option<[f32; 4]> {
    if angle_degrees == 0.0 || !angle_degrees.is_finite() {
        return None;
    }
    let mask = &detection.mask;
    // The same window `infer_typography` takes, so the ink measured here is the
    // ink whose angle was handed in.
    let width = image.width().min(mask.width);
    let height = image.height().min(mask.height);
    let [window_left, window_top, window_right, window_bottom] =
        mask_window(detection.bbox, width, height)?;

    let (sin, cos) = f64::from(angle_degrees).to_radians().sin_cos();
    let (mut min_u, mut min_v) = (f64::INFINITY, f64::INFINITY);
    let (mut max_u, mut max_v) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut any = false;
    for y in window_top..window_bottom {
        let row = y as usize * mask.width as usize;
        for x in window_left..window_right {
            if mask.pixels.get(row + x as usize).copied().unwrap_or(0) == 0 {
                continue;
            }
            any = true;
            let (point_x, point_y) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            // The inverse of `rotated_rectangle_geometry`'s forward turn, and the
            // same local frame the `masked_text` fixture builds its rectangle in.
            let u = point_x * cos + point_y * sin;
            let v = -point_x * sin + point_y * cos;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !any {
        return None;
    }

    let (oriented_width, oriented_height) = (max_u - min_u, max_v - min_v);
    let [left, top, right, bottom] = detection.bbox;
    if oriented_width * oriented_height >= f64::from((right - left) * (bottom - top)) {
        return None;
    }

    // Back to the page, so that turning this box by `angle_degrees` about its own
    // centre puts its four corners on the ink's oriented hull.
    let (center_u, center_v) = ((min_u + max_u) * 0.5, (min_v + max_v) * 0.5);
    let center_x = center_u * cos - center_v * sin;
    let center_y = center_u * sin + center_v * cos;
    Some([
        (center_x - oriented_width * 0.5) as f32,
        (center_y - oriented_height * 0.5) as f32,
        (center_x + oriented_width * 0.5) as f32,
        (center_y + oriented_height * 0.5) as f32,
    ])
}

pub(super) fn rotated_rectangle_geometry(
    [left, top, right, bottom]: [f32; 4],
    angle_degrees: f32,
) -> Geometry {
    let width = f64::from((right - left).max(1.0));
    let height = f64::from((bottom - top).max(1.0));
    let center_x = f64::from(left + right) * 0.5;
    let center_y = f64::from(top + bottom) * 0.5;
    let (sin, cos) = f64::from(angle_degrees).to_radians().sin_cos();
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

fn mask_geometry(mask: &KoharuLayoutMask) -> Option<Geometry> {
    let mask = GrayImage::from_raw(mask.width, mask.height, mask.pixels.clone())?;
    let mut padded = GrayImage::new(mask.width() + 2, mask.height() + 2);
    image::imageops::replace(&mut padded, &mask, 1, 1);
    let contours = find_contours_with_threshold::<i32>(&padded, 0);
    let contour = contours
        .iter()
        .filter(|contour| contour.border_type == BorderType::Outer)
        .max_by(|left, right| {
            contour_area(&left.points)
                .partial_cmp(&contour_area(&right.points))
                .unwrap_or(Ordering::Equal)
        })?;
    if contour.points.len() < 3 {
        return None;
    }

    let epsilon = (arc_length(&contour.points, true) * 0.001).max(f64::EPSILON);
    let points = approximate_polygon_dp(&contour.points, epsilon, true)
        .into_iter()
        .map(|point| Point {
            x: f64::from(point.x - 1),
            y: f64::from(point.y - 1),
        })
        .collect::<Vec<_>>();
    (points.len() >= 3).then_some(Geometry {
        origin: Origin::User,
        points,
    })
}

fn write_masks(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    // `&settle::Settled` for the reason `write_regions` gives.
    detections: &settle::Settled,
    size: ImageSize,
    refined_text_mask: Option<&GrayImage>,
    translate_sfx: bool,
    mask_scale: Option<f32>,
    ink_mask: bool,
    implausible_mask_area: Option<f32>,
    image: &RgbImage,
) -> Result<()> {
    for spec in [
        MaskSpec {
            role: "text-mask",
            label: TEXT,
            dilate: true,
        },
        MaskSpec {
            role: "bubble-mask",
            label: "bubble",
            dilate: false,
        },
    ] {
        // Only the text mask is refined. `bubble-mask` is a balloon polygon
        // and is not what the segmenter looks for.
        let refined = spec.dilate.then_some(refined_text_mask).flatten();
        // Same discriminator, same reason: `text-mask` is the one the inpainter
        // erases from, and it is the only one where a mis-segmented box costs
        // artwork. `bubble-mask` is read by the editor for layout, so refusing a
        // detection there would change what a human can select and repair
        // nothing.
        let implausible = spec.dilate.then_some(implausible_mask_area).flatten();
        write_mask(
            input,
            edit,
            page,
            detections,
            spec,
            size,
            refined,
            translate_sfx,
            mask_scale,
            ink_mask,
            implausible,
            input.joined_page(),
            image,
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct MaskSpec {
    role: &'static str,
    label: &'static str,
    dilate: bool,
}

fn write_mask(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    detections: &[KoharuLayoutDetection],
    spec: MaskSpec,
    size: ImageSize,
    refined: Option<&GrayImage>,
    translate_sfx: bool,
    mask_scale: Option<f32>,
    ink_mask: bool,
    implausible: Option<f32>,
    // The caller assembled this page so its text ends inside it. Travels
    // beside `implausible` because it lands in the same predicate.
    joined_page: bool,
    image: &RgbImage,
) -> Result<()> {
    let mut mask = mask_for(detections, spec.label, size, translate_sfx, implausible, joined_page);
    if spec.dilate && size.width > 0 && size.height > 0 {
        mask = match mask_scale {
            Some(scale) => scaled_dilated_mask(
                detections,
                spec.label,
                size,
                translate_sfx,
                scale,
                implausible,
                joined_page,
            ),
            None => {
                // The shared rule pre-clamp; this site's own tail stays here
                // (`super::dilation_radius`'s doc owns the why).
                let radius =
                    (super::dilation_radius(size.width.max(size.height)) as f32).clamp(1.0, 255.0) as u8;
                dilate(&mask, Norm::L2, radius)
            }
        };
    }
    // Unioned after dilation so it adds ink the box mask missed, and before the
    // refinement intersection so an operator asking for both still gets a subset
    // of what was asked for rather than a contradiction.
    if ink_mask && spec.dilate {
        let ink = ink_mask_for(
            image,
            detections,
            spec.label,
            size,
            translate_sfx,
            INK_MASK_PAD,
            implausible,
            joined_page,
        );
        for (target, source) in mask.iter_mut().zip(ink.iter()) {
            if *source != 0 {
                *target = u8::MAX;
            }
        }
    }
    if let Some(refined) = refined {
        mask = intersect_masks(&mask, refined);
    }
    if let Some(bounds) = input.region {
        preserve_mask_outside_region(input, page, spec.role, bounds, &mut mask)?;
    }

    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageLuma8(mask).write_to(&mut bytes, ImageFormat::Png)?;
    edit.set_asset(
        page,
        &AssetRole::new(spec.role)?,
        AssetInput::new(
            Arc::<[u8]>::from(bytes.into_inner()),
            "image/png",
            AssetMetadata {
                width: Some(size.width),
                height: Some(size.height),
                attributes: BTreeMap::new(),
            },
        ),
    )?;
    Ok(())
}

/// Ink actually present inside each detection's box, found without a model.
///
/// Ported behaviour, not code, from `comic-translate`'s
/// `modules/detection/utils/content.py`, which pairs a box-only detector with no
/// segmenter at all: expand the box, Otsu-threshold the crop, take connected
/// components at **both polarities**, and drop the ones that are not lettering.
///
/// **Why this is worth trying when `refine_text_mask` was measured negative.**
/// That one intersects the segmenter's output with the dilated box mask, so it
/// can only ever erase *less* -- and the defect on real pages is ink left
/// *behind*, which LaMa then reconstructs into legible ghost katakana. This is
/// the opposite operation: it is UNIONed in, so it can only ever erase *more*,
/// and it is looking for exactly the ink the segmentation head under-covered.
///
/// Both polarities matter and are not symmetry for its own sake: manga sound
/// effects are routinely white-on-black, and a dark-ink-only rule finds nothing
/// on the very panels where the residue is worst.
///
/// The 50%-of-crop guard is what stops it swallowing the page. A component that
/// large is a filled panel, a black background or a coloured narration box
/// rather than a glyph, and erasing it would be far worse than the residue this
/// exists to remove.
fn ink_mask_for(
    image: &RgbImage,
    detections: &[KoharuLayoutDetection],
    label: &str,
    size: ImageSize,
    translate_sfx: bool,
    pad: u32,
    implausible: Option<f32>,
    // The caller assembled this page so its text ends inside it. Travels
    // beside `implausible` because it lands in the same predicate.
    joined_page: bool,
) -> GrayImage {
    let mut mask = GrayImage::new(size.width, size.height);
    for detection in detections
        .iter()
        .filter(|value| mask_includes(detections, value, label, translate_sfx, size, implausible, joined_page))
    {
        let [left, top, right, bottom] = detection.bbox;
        let x0 = (left.floor().max(0.0) as u32).saturating_sub(pad);
        let y0 = (top.floor().max(0.0) as u32).saturating_sub(pad);
        let x1 = ((right.ceil().max(0.0) as u32) + pad).min(size.width);
        let y1 = ((bottom.ceil().max(0.0) as u32) + pad).min(size.height);
        if x1 <= x0 + 2 || y1 <= y0 + 2 {
            continue;
        }
        for (x, y, value) in ink_components(image, x0, y0, x1, y1).enumerate_pixels() {
            if value[0] != 0 {
                mask.put_pixel(x0 + x, y0 + y, Luma([u8::MAX]));
            }
        }
    }
    mask
}

/// The ink inside one crop of the page, as a mask of the crop's own size.
///
/// Lifted out of [`ink_mask_for`] unchanged so that [`ink_reaches_far_band`] can
/// ask the same question, and **that sharing is the point rather than tidiness**.
/// A column is grown because ink was found under this rule; the eraser then
/// decides what to strip under this rule. Two definitions of ink would let a box
/// be grown on one footprint and erased on another, which is a blank hole in the
/// artwork with nothing to show for it.
///
/// It also rules out the cheap version of the same idea. A brightness probe --
/// "is there anything dark down there?" -- fires on the white paper of an
/// ordinary manga page, and that was measured: 0 manga joins
/// became 22.
fn ink_components(image: &RgbImage, x0: u32, y0: u32, x1: u32, y1: u32) -> GrayImage {
    let (width, height) = (x1 - x0, y1 - y0);
    let mut mask = GrayImage::new(width, height);
    let mut crop = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x0 + x, y0 + y).0;
            // Rec. 601 luma, matching the rest of this file's greyscale.
            let luma = (0.299 * f32::from(pixel[0])
                + 0.587 * f32::from(pixel[1])
                + 0.114 * f32::from(pixel[2])) as u8;
            crop.put_pixel(x, y, Luma([luma]));
        }
    }

    let level = otsu_level(&crop);
    let area = (width * height) as f32;
    for dark in [true, false] {
        let mut binary = GrayImage::new(width, height);
        for (x, y, pixel) in crop.enumerate_pixels() {
            let ink = if dark {
                pixel[0] < level
            } else {
                pixel[0] > level
            };
            if ink {
                binary.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let labels = connected_components(&binary, Connectivity::Eight, Luma([0u8]));
        let count = labels.iter().copied().max().unwrap_or(0) as usize + 1;
        let mut sizes = vec![0u32; count];
        for label in labels.iter() {
            if *label != 0 {
                sizes[*label as usize] += 1;
            }
        }
        for (x, y, value) in labels.enumerate_pixels() {
            let component = value[0] as usize;
            if component == 0 {
                continue;
            }
            let share = sizes[component] as f32 / area;
            // Too large is background; a couple of pixels is sensor noise or
            // a screentone dot, and erasing those buys nothing.
            if share > 0.5 || sizes[component] < 4 {
                continue;
            }
            mask.put_pixel(x, y, Luma([u8::MAX]));
        }
    }
    mask
}

/// Grows each detection's own mask toward `scale` times its bounding box,
/// instead of growing the whole page's mask by one flat radius.
///
/// The shipped rule is `round(max_dim / 1024 * 6)` -- about 10px on a 1200px
/// page -- and it depends only on the PAGE size, so every region on a page is
/// grown by the same number of pixels regardless of how big the region is.
/// Restated as a scale factor, that flat 10px is:
///
/// | region's smaller side | effective scale |
/// |---|---|
/// | 20px | 2.00x |
/// | 50px | 1.40x |
/// | 300px | 1.07x |
///
/// So it lands near the reported optimum only for regions of about 50px, and is
/// far too aggressive on small text while barely growing large effects at all.
/// This computes the radius from the *region*, so the scale is constant instead.
///
/// Per detection rather than per page because a single radius cannot be right
/// for both: the dilation is applied before the union, so neighbouring regions
/// of different sizes each get their own growth.
fn scaled_dilated_mask(
    detections: &[KoharuLayoutDetection],
    label: &str,
    size: ImageSize,
    translate_sfx: bool,
    scale: f32,
    implausible: Option<f32>,
    // The caller assembled this page so its text ends inside it. Travels
    // beside `implausible` because it lands in the same predicate.
    joined_page: bool,
) -> GrayImage {
    let mut mask = GrayImage::new(size.width, size.height);
    for detection in detections.iter().filter(|value| {
        mask_includes(detections, value, label, translate_sfx, size, implausible, joined_page)
    }) {
        let [left, top, right, bottom] = detection.bbox;
        let shorter = (right - left).min(bottom - top).max(0.0);
        // Scaling a box by `s` about its centre grows each side by
        // `(s - 1) / 2` of that side, which is the radius wanted here.
        let radius = (((scale - 1.0) / 2.0) * shorter).round().clamp(1.0, 255.0) as u8;

        // Dilating the whole page once per detection would be quadratic in the
        // number of regions for no benefit -- the growth cannot reach beyond the
        // bbox plus the radius, so only that window is built and dilated.
        let margin = u32::from(radius) + 1;
        let x0 = (left.floor().max(0.0) as u32).saturating_sub(margin);
        let y0 = (top.floor().max(0.0) as u32).saturating_sub(margin);
        let x1 = ((right.ceil().max(0.0) as u32) + margin).min(size.width);
        let y1 = ((bottom.ceil().max(0.0) as u32) + margin).min(size.height);
        if x1 <= x0 || y1 <= y0 {
            continue;
        }

        let mut window = GrayImage::new(x1 - x0, y1 - y0);
        for y in y0..y1 {
            for x in x0..x1 {
                let index = (y * size.width + x) as usize;
                if detection.mask.pixels.get(index).is_some_and(|value| *value != 0) {
                    window.put_pixel(x - x0, y - y0, Luma([u8::MAX]));
                }
            }
        }
        let grown = dilate(&window, Norm::L2, radius);
        for y in 0..grown.height() {
            for x in 0..grown.width() {
                if grown.get_pixel(x, y)[0] != 0 {
                    mask.put_pixel(x + x0, y + y0, Luma([u8::MAX]));
                }
            }
        }
    }
    mask
}

/// How much of a candidate effect's **own** box may be covered by another
/// lettered region before it yields. Measured against the candidate, not against
/// the smaller of the pair -- see `overshadowed_effect`, which divides by the
/// candidate's area.
const OVERSHADOW_SHARE: f32 = 0.25;

/// Whether to spare an effect that would be lettered on top of something else.
///
/// # OFF, and the measurement says why
///
/// It does what it claims on the catastrophic cases: over 40 manga pages, same
/// binary, the 5k-pixel-and-worse collision band goes 11 -> 1 and the worst single
/// case 59,579px -> 5,273px, keeping 129 of 149 effects.
///
/// It also fires where nothing was wrong. A five-judge vision panel over 20
/// before/after sheets returned **3 better, 1 worse, 15 same**, and every "worse"
/// names the same thing: a perfectly legible effect removed where the baseline had
/// no visible collision at all. On the Chinese corpus it is pure cost -- 8 effects
/// spared and the collision counts byte-identical to the off arm.
///
/// THE PROXY IS THE PROBLEM, and three discriminators were tried and refuted:
/// overlap share (complaints 0.50-0.81 against wins 0.27-1.00), absolute box area
/// (21k-98k against 1.9k-315k), and what the effect overlaps (both groups are
/// effect-on-effect). None separates a real collision from a harmless one, because
/// bounding boxes overlapping does not mean the RENDERED text overlaps: effects
/// carry sparse ink and two overlapping boxes routinely letter without touching.
///
/// The decision needs the laid-out text extents, and detection does not have them
/// -- layout is solved later, in the renderer. Doing this properly means deciding
/// after layout, not here. Kept because the erase half is correct and the panel
/// confirmed it (zero smear defects flagged: a spared effect leaves clean
/// artwork), and because re-deriving the refutation costs another GPU day.
fn spare_overshadowed_effects() -> bool {
    matches!(
        std::env::var("KOHARU_SPARE_OVERSHADOWED_SFX").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Would lettering this effect land on top of another region's lettering?
///
/// # The defect
///
/// `translate_sfx` letters sound effects. Effects are drawn large and free-form over
/// artwork with no balloon to bound them, and two of them frequently occupy the
/// same pixels -- so both get lettered and both become unreadable. Measured over
/// 4,456 drawn regions on two Japanese volumes: onomatopoeia is 30% of regions
/// but carries **88% of the severe ink collisions**, and the worst single case is
/// 59,579px of two English effects stacked on each other. The same holds on the
/// webtoon formats, where 100% of the severe collisions involve an effect.
///
/// # Why layout cannot fix it, and this has to be suppression
///
/// `bubble_text_cell` clamps every cut so a cell always contains its own ink
/// (`right.min(middle.max(bounds[2]))`). That guarantee is correct and load
/// bearing -- it is what stops a cut shredding a column into two-letter lines --
/// but it means that when two effects' SOURCE INK overlaps, which is 55.9% of
/// contested manga pairs, their cells must overlap too. No cell-cutting scheme
/// reaches ink that was drawn overlapping in the first place. Something has to
/// not be drawn.
///
/// # Who yields
///
/// Dialogue never yields: a reader loses more from an unreadable line than from
/// an untranslated sound effect, and an effect is usually legible as artwork
/// anyway. Between two effects the smaller yields, because the larger one is the
/// one the artist drew big. The comparison is a TOTAL order -- area, then score,
/// then the bbox itself -- so exactly one of any pair loses and two effects can
/// never suppress each other into a page with neither.
///
/// # It is computed, never stored, and that is deliberate
///
/// Both callers derive it from the same `detections` slice: `write_regions` to
/// decide lettering, `mask_includes` to decide erasing. A spared effect must be
/// left BOTH un-lettered and un-erased or it is worse than the collision -- an
/// erased effect with nothing drawn back is a large LaMa inpaint over artwork,
/// and this project has already measured that failure as smearing and residue.
/// Threading a decision through both paths invites them to disagree; recomputing
/// a pure function of the same input cannot.
fn overshadowed_effect(
    detection: &KoharuLayoutDetection,
    detections: &[KoharuLayoutDetection],
    translate_sfx: bool,
) -> bool {
    // With effects off entirely there is nothing to spare, and the global arm
    // already decides what happens to them.
    // The env gate lives in the CALLERS, not here: a predicate that reads the
    // environment cannot be unit-tested without mutating global process state,
    // and both call sites have to agree on the gate anyway.
    if !translate_sfx || detection.label != ONOMATOPOEIA {
        return false;
    }
    let area = |b: [f32; 4]| ((b[2] - b[0]) * (b[3] - b[1])).max(0.0);
    let mine = area(detection.bbox);
    if mine <= 0.0 {
        return false;
    }
    for other in detections {
        if other.label != TEXT && other.label != ONOMATOPOEIA {
            continue;
        }
        let theirs = area(other.bbox);
        if theirs <= 0.0 {
            continue;
        }
        // Identity by value: `mask_includes` is handed a detection by reference
        // but makes no promise it points into this slice.
        if other.bbox == detection.bbox && other.label == detection.label {
            continue;
        }
        let ix = (detection.bbox[2].min(other.bbox[2]) - detection.bbox[0].max(other.bbox[0]))
            .max(0.0);
        let iy = (detection.bbox[3].min(other.bbox[3]) - detection.bbox[1].max(other.bbox[1]))
            .max(0.0);
        // Share of the CANDIDATE'S OWN box, not of the smaller of the two.
        //
        // Using the smaller box was the first rule and an existing test refuted
        // it: a page-wide effect that merely CONTAINS a small speech bubble
        // scores 1.0 against the bubble's area and the whole effect is dropped,
        // for a collision covering a sixth of it. Measuring against its own area
        // asks the proportionate question -- is a substantial part of THIS
        // region contested -- so a big effect keeps its lettering when a bubble
        // clips a corner of it, while two effects of comparable size still both
        // clear the bar and the tie-break decides.
        if ix * iy < mine * OVERSHADOW_SHARE {
            continue;
        }
        if other.label == TEXT {
            return true;
        }
        // Two effects: a total order, so exactly one of the pair yields.
        let key = |b: [f32; 4], s: f32, a: f32| (a, s, b[0], b[1], b[2], b[3]);
        if partial_less(
            key(detection.bbox, detection.score, mine),
            key(other.bbox, other.score, theirs),
        ) {
            return true;
        }
    }
    false
}

/// Lexicographic `<` over the tuple `overshadowed_effect` orders effects by.
/// Written out because `f32` is not `Ord` and a NaN must not make the relation
/// non-antisymmetric -- if any comparison is undecidable, neither side yields.
fn partial_less(a: (f32, f32, f32, f32, f32, f32), b: (f32, f32, f32, f32, f32, f32)) -> bool {
    for (left, right) in [
        (a.0, b.0),
        (a.1, b.1),
        (a.2, b.2),
        (a.3, b.3),
        (a.4, b.4),
        (a.5, b.5),
    ] {
        if left < right {
            return true;
        }
        if left > right {
            return false;
        }
        if left.is_nan() || right.is_nan() {
            return false;
        }
    }
    false
}

/// Whether this detection contributes to `label`'s mask. Shared so the flat and
/// scaled paths cannot drift apart on the sound-effect rule.
///
/// `implausible` is the fraction of the page's own area at which a box stops
/// being a credible region and becomes a detector error -- `None` accepts every
/// box, which is what has always shipped. It is applied **here** rather than in
/// `mask_for` alone, and that placement is the point: `mask_scale` swaps
/// `mask_for` out for `scaled_dilated_mask` entirely, and `ink_mask` unions
/// `ink_mask_for` back in afterwards, so a gate in `mask_for` would be silently
/// bypassed by the first flag and silently undone by the second. All three
/// consult this predicate, so all three refuse the same boxes.
///
/// # It measures a DIFFERENT BOX than the OCR router does, and that is a defect
///
/// `crate::stages::implausible_region` is shared with `ocr.rs` precisely so the
/// reader and the eraser cannot disagree, and its doc says so. **The predicate is
/// shared; the argument is not.** Here it is handed `detection.bbox` raw. In
/// `ocr.rs` `region_extent` hands it `geometry_extents(target.geometry)`, and for
/// a lettering region that geometry is `rotated_rectangle_geometry(bbox, angle)`
/// -- the same rectangle turned about its centre and re-boxed, which can only be
/// larger. So a rotated box can be refused by the reader and erased here.
///
/// Measured over a full manga baseline run: **23 regions refused by
/// the reader, 19 erased anyway**, 4.8%-47.4% of the pixels inside the box, 11 of
/// them on pages with no other region to blame. The four that were spared are the
/// four with the largest reported boxes, which is what the inflation predicts.
/// Pinned by `a_rotated_region_the_reader_refuses_for_size_is_erased_anyway`.
/// **Not yet fixed** -- which of the two boxes is the right one is still open.
fn mask_includes(
    detections: &[KoharuLayoutDetection],
    detection: &KoharuLayoutDetection,
    label: &str,
    translate_sfx: bool,
    size: ImageSize,
    implausible: Option<f32>,
    // The caller assembled this page so its text ends inside it. Travels
    // beside `implausible` because it lands in the same predicate.
    joined_page: bool,
) -> bool {
    // Asked before the label rules and not after: the question is whether the
    // detector can be believed at all about this box, which does not depend on
    // what it called it or on whether effects are being lettered.
    let [left, top, right, bottom] = detection.bbox;
    /* THE SAME PREDICATE THE OCR ROUTER CALLS, and it has to be the same one.
     *
     * A slice-height fragment is refused in two places on purpose: the
     * router, so it is never read, and here, so it is never ERASED -- a fragment
     * stripped and then lettered with nothing leaves a blank column, which is
     * worse than leaving the original glyphs alone.
     *
     * That pairing is exactly why this call needs `joined_page` too. Lifting the
     * guard for the router alone was measured on a joined slice pair and it is a
     * REGRESSION: the joined name is read and lettered while its artwork is
     * spared, so the English lands on top of intact Chinese. Both halves move
     * together or neither does. */
    if crate::stages::refuse_region(
        (f64::from(right - left), f64::from(bottom - top)),
        (size.width, size.height),
        implausible,
        joined_page,
    ) {
        return false;
    }
    // A spared effect is not lettered, so it must not be erased either. Erasing
    // it and drawing nothing back is a large LaMa inpaint over artwork, which
    // this project has already measured as smearing and residue -- strictly
    // worse than the collision this is avoiding.
    if spare_overshadowed_effects() && overshadowed_effect(detection, detections, translate_sfx) {
        return false;
    }
    detection.label == label
        || (label == TEXT
            && detection.label == ONOMATOPOEIA
            && (translate_sfx
                || detections.iter().any(|bubble| {
                    bubble.label == "bubble" && containment(bubble.bbox, detection.bbox) >= 0.5
                })))
}

/// Keeps only the pixels both masks agree on.
///
/// Deliberately an intersection and not a union. `coarse` is the dilated box
/// mask that ships today, so the result can only ever be a subset of it: a page
/// that erases correctly now cannot start erasing *more*, and everything the
/// segmenter sees which detection did not -- credits, page numbers,
/// free-standing onomatopoeia -- is discarded rather than becoming new damage.
///
/// A size mismatch keeps `coarse` untouched. Both masks come from the same page
/// image but by different routes (`KoharuLayoutDetections` reports the size it
/// inferred at, `MangaTextMask` the size it was handed), and a disagreement
/// would otherwise become a hard stage failure downstream in
/// `preserve_mask_outside_region`.
fn intersect_masks(coarse: &GrayImage, fine: &GrayImage) -> GrayImage {
    if coarse.dimensions() != fine.dimensions() {
        tracing::warn!(
            coarse = ?coarse.dimensions(),
            fine = ?fine.dimensions(),
            "refined text mask has the wrong size; keeping the detection mask"
        );
        return coarse.clone();
    }

    // Per connected blob, not per page. Sharpening is only safe where the
    // segmenter actually saw the text: measured on an art-heavy title page, it
    // resolves clean balloon text well but finds little of a vertical caption
    // running over dark hair, and a whole-page intersection there left the
    // Japanese legible under the English. A blob it did not see keeps the
    // dilated box mask, which is exactly today's output.
    let labels = connected_components(coarse, Connectivity::Eight, Luma([0u8]));
    let count = labels.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut coarse_area = vec![0u32; count];
    let mut fine_area = vec![0u32; count];
    let mut fine_bands = vec![[0u32; COVERAGE_BANDS]; count];
    let mut coarse_bands = vec![[0u32; COVERAGE_BANDS]; count];
    let mut top = vec![u32::MAX; count];
    let mut bottom = vec![0u32; count];

    for (x, y, label) in labels.enumerate_pixels() {
        let label = label[0] as usize;
        if label == 0 {
            continue;
        }
        coarse_area[label] += 1;
        top[label] = top[label].min(y);
        bottom[label] = bottom[label].max(y);
        if fine.get_pixel(x, y)[0] != 0 {
            fine_area[label] += 1;
        }
    }
    for (x, y, label) in labels.enumerate_pixels() {
        let label = label[0] as usize;
        if label == 0 {
            continue;
        }
        let height = bottom[label] - top[label] + 1;
        let band = ((y - top[label]) as usize * COVERAGE_BANDS / height as usize)
            .min(COVERAGE_BANDS - 1);
        coarse_bands[label][band] += 1;
        if fine.get_pixel(x, y)[0] != 0 {
            fine_bands[label][band] += 1;
        }
    }

    let sharpen = (0..count)
        .map(|label| {
            if coarse_area[label] == 0 {
                return false;
            }
            // Two tests, because either alone is fooled. The ratio rejects a
            // blob the segmenter barely touched; the bands reject one it saw
            // only at one end -- a tall caption whose top half resolves and
            // whose bottom half crosses black artwork passes the ratio easily
            // and is the case that actually went wrong.
            let covered = f64::from(fine_area[label]) / f64::from(coarse_area[label]);
            /* Each band must MEET the ratio, not merely be non-empty. Requiring
             * only "some ink somewhere in this band" passed the caption that
             * broke the first version: the segmenter finds scattered pixels all
             * the way down a column while missing most of the glyphs, so every
             * band was non-empty and the blob was sharpened into
             * under-erasure. */
            covered >= MINIMUM_REFINED_COVERAGE
                && (0..COVERAGE_BANDS).all(|band| {
                    let coarse = coarse_bands[label][band];
                    // A band the blob barely occupies carries no evidence
                    // either way; judging it would reject on noise.
                    coarse == 0
                        || f64::from(fine_bands[label][band]) / f64::from(coarse)
                            >= MINIMUM_REFINED_COVERAGE
                })
        })
        .collect::<Vec<_>>();

    let mut out = coarse.clone();
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let label = labels.get_pixel(x, y)[0] as usize;
        if label != 0 && sharpen[label] {
            pixel[0] = pixel[0].min(fine.get_pixel(x, y)[0]);
        }
    }
    out
}

fn preserve_mask_outside_region(
    input: &StageInput,
    page: EntityId,
    role: &str,
    bounds: crate::Bounds,
    mask: &mut GrayImage,
) -> Result<()> {
    let previous = input
        .images
        .get(&input.scene, page, role)?
        .map(|image| image.to_luma8());
    if previous
        .as_ref()
        .is_some_and(|image| image.dimensions() != mask.dimensions())
    {
        bail!("existing {role} dimensions do not match page {page}");
    }
    for (x, y, pixel) in mask.enumerate_pixels_mut() {
        if f64::from(x + 1) <= bounds.x
            || f64::from(y + 1) <= bounds.y
            || f64::from(x) >= bounds.x + bounds.width
            || f64::from(y) >= bounds.y + bounds.height
        {
            *pixel = previous
                .as_ref()
                .map_or(Luma([0]), |image| *image.get_pixel(x, y));
        }
    }
    Ok(())
}

/// Whether a detection of this class becomes a lettered text layer.
///
/// The one place the sound-effect decision is made. Everything downstream --
/// which regions get a `Typography`, which get a text content entity, which the
/// OCR stage will visit, and which pixels the inpainter erases -- follows from
/// this predicate rather than re-testing the label.
fn letters_text(label: &str, translate_sfx: bool) -> bool {
    label == TEXT || (translate_sfx && label == ONOMATOPOEIA)
}

fn region_kind(label: &str, translate_sfx: bool) -> Result<RegionKind> {
    RegionKind::new(match label {
        // `TextRegion` is not cosmetic here: `stages/ocr.rs` selects the regions
        // it reads by `kind == TextRegion::kind()`, so this is what puts a sound
        // effect in front of the OCR model at all.
        _ if letters_text(label, translate_sfx) => TextRegion::KIND,
        "bubble" => BubbleRegion::KIND,
        "panel" => PanelRegion::KIND,
        _ => "dev.koharu.region.unknown",
    })
    .map_err(Into::into)
}

fn mask_for(
    detections: &[KoharuLayoutDetection],
    label: &str,
    size: ImageSize,
    translate_sfx: bool,
    implausible: Option<f32>,
    // The caller assembled this page so its text ends inside it. Travels
    // beside `implausible` because it lands in the same predicate.
    joined_page: bool,
) -> GrayImage {
    let mut mask = GrayImage::new(size.width, size.height);
    // Lettering an effect makes erasing it unconditional -- see `mask_includes`,
    // which both this and the scaled path share so they cannot disagree.
    for detection in detections
        .iter()
        .filter(|value| mask_includes(detections, value, label, translate_sfx, size, implausible, joined_page))
    {
        for (target, source) in mask.as_mut().iter_mut().zip(&detection.mask.pixels) {
            if *source != 0 {
                *target = u8::MAX;
            }
        }
    }
    mask
}

fn intersects([left, top, right, bottom]: [f32; 4], region: crate::Bounds) -> bool {
    left < (region.x + region.width) as f32
        && right > region.x as f32
        && top < (region.y + region.height) as f32
        && bottom > region.y as f32
}

fn detection_order(left: &KoharuLayoutDetection, right: &KoharuLayoutDetection) -> Ordering {
    left.bbox[1]
        .total_cmp(&right.bbox[1])
        .then_with(|| right.bbox[0].total_cmp(&left.bbox[0]))
        .then_with(|| left.label.cmp(&right.label))
}

/// The composed detection pass, and the only type its consumers accept.
///
/// **A module, and `Settled`'s field is private to it, because a call site is
/// not a contract.** The first version of this change folded the suppression
/// into one function and deleted the old free one, exactly as planned -- and
/// replacing the call in `write_page` with `Vec::new()` still compiled, with one
/// `unused variable` warning and every test green. That is the unwired-fix failure
/// reproduced inside its own countermeasure. `write_regions` and `write_masks`
/// now demand a [`Settled`], nothing outside this module can construct one, and
/// a `write_page` that skips the pass therefore does not build at all.
mod settle {
    use image::RgbImage;
    use koharu_ml::koharu_layout_rfdetr_seg_2xl::{KoharuLayoutDetection, KoharuLayoutMask};

    use super::{
        AXIS_DIFFER_MIN_RATIO, AXIS_SHARE_MAX_RATIO, AXIS_TIE_BREAK_MIN_SCORE_RATIO,
        EDGE_BAND_FLOOR_PX, EDGE_BAND_FRACTION, EDGE_HINT_MIN_SCORE, EdgeHint,
        FREE_TEXT_MIN_ASPECT, FREE_TEXT_ROTATED_MIN_WIDTH, INK_MASK_PAD, INK_WALK_MAX_GAP_ROWS,
        INK_WALK_MIN_ROW_PIXELS, INK_WALK_REACH_RATIO, ImageSize, NMS_CONTAINMENT_THRESHOLD,
        PageEdge, TEXT, box_height, box_width, containment, detection_order, extent_ratio,
        ink_components, ink_within, intersection_over_union, letters_text, overlap_over_smaller,
    };

    /// Detections that have been through [`settle_detections`].
    ///
    /// The field is private to this module and there is no other constructor, so
    /// possessing one of these is proof the pass ran. See the module's own doc.
    #[derive(Debug)]
    pub(in crate::stages::detection) struct Settled {
        detections: Vec<KoharuLayoutDetection>,
        /// Whether the detection at the same index was SYNTHESISED from a bubble
        /// that held no text region of its own, rather than returned by the
        /// network. One entry per detection, always.
        ///
        /// **A parallel vector and not a trailing count, because
        /// [`Settled::sort_by_layout`] reorders the detections.** The synthesis
        /// appends, so "the last N are synthesised" is true for exactly as long as
        /// nothing sorts -- and then it silently names different regions. That is
        /// a known defect class: a positional pairing that survives review
        /// because it is right when it is written.
        synthesised: Vec<bool>,
        /// Whether each detection is an evicted box's readmitted residue --
        /// `--nms-residue-regions`. Parallel to `synthesised`, and
        /// permuted with it for the reason written above it: a positional
        /// pairing is right when written and wrong after the first sort.
        ///
        /// NOTE for whoever next touches `settle`: that function binds `residue`
        /// to the FLAG, so field-init shorthand here would silently store a bool
        /// where a vector belongs. Always write `residue: residue_flags`.
        residue: Vec<bool>,
    }

    impl Settled {
        /// Reading order, applied in place. Kept as a method rather than letting
        /// the caller reach the vector, so the wrapper stays unforgeable.
        ///
        /// **Permutes the provenance with it.** Sorting one of two parallel
        /// vectors is how a pairing rots, so the order is computed once, as
        /// indices, and applied to both -- see
        /// `sorting_carries_the_provenance_with_the_detections`.
        pub(in crate::stages::detection) fn sort_by_layout(&mut self) {
            // The same permutation `super::sort_by_layout` applies, taken once and
            // applied to both vectors rather than sorting one and hoping.
            let order = super::layout_order(&self.detections);
            self.synthesised = order.iter().map(|&index| self.synthesised[index]).collect();
            self.residue = order.iter().map(|&index| self.residue[index]).collect();
            let mut values = std::mem::take(&mut self.detections)
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>();
            self.detections = order
                .into_iter()
                .map(|index| {
                    values[index]
                        .take()
                        .expect("layout order contains each detection once")
                })
                .collect();
        }

        /// Whether the detection at `index` was synthesised from a text-less
        /// bubble. `false` for anything the network actually returned, and for an
        /// index this does not have.
        pub(in crate::stages::detection) fn synthesised(&self, index: usize) -> bool {
            self.synthesised.get(index).copied().unwrap_or(false)
        }

        /// Whether the detection at `index` is an evicted box's readmitted
        /// residue. `false` for anything the network returned, and for an index
        /// this does not have.
        pub(in crate::stages::detection) fn residue(&self, index: usize) -> bool {
            self.residue.get(index).copied().unwrap_or(false)
        }

        /// The reader's own boxes, admitted AFTER the pass on
        /// purpose: a caller box is an assertion, not a candidate, so the NMS
        /// loop may not evict it and the column repair may not reshape it.
        ///
        /// Each box's mask is its INK, not its body -- `ink_within`, the same
        /// distillation the textless-bubble synthesis below uses, and for the
        /// same two consumers: the erase assembly would otherwise inpaint the
        /// whole rectangle away (79.7% of one balloon's white, measured), and
        /// `infer_typography` would measure the paper. A box holding no ink
        /// is REFUSED rather than invented -- `ink_within`'s own floors --
        /// which is also what spares the artwork under a misdrawn rectangle;
        /// each refusal is logged with its rect.
        ///
        /// `score` 1.0: the one detection whose confidence is the reader's
        /// own assertion, and the wire's `detection_confidence` then reports
        /// it as exactly that. `label_id` 0 -- nothing on this crate's path
        /// reads it (the synthesis note below), and the label is `TEXT`,
        /// which is what every consumer keys on.
        /// The detections a caller box may inherit its mask and kind from:
        /// everything whose bbox centre lies inside one of the
        /// reader's ADDED rects, cloned BEFORE `remove_caller_boxes` runs so
        /// a resize (remove old + add nudged) can still see the region it is
        /// replacing. Cloned narrowly -- a page carries at most a few donors
        /// per edit -- because borrowing across the removal is the aliasing
        /// this file keeps refusing.
        pub(in crate::stages::detection) fn caller_donors(
            &self,
            added: &[crate::CallerRegion],
        ) -> Vec<KoharuLayoutDetection> {
            if added.is_empty() {
                return Vec::new();
            }
            self.detections
                .iter()
                .filter(|detection| {
                    let center_x = (detection.bbox[0] + detection.bbox[2]) / 2.0;
                    let center_y = (detection.bbox[1] + detection.bbox[3]) / 2.0;
                    added.iter().any(|rect| {
                        center_x >= rect.x
                            && center_x <= rect.x + rect.width
                            && center_y >= rect.y
                            && center_y <= rect.y + rect.height
                    })
                })
                .cloned()
                .collect()
        }

        pub(in crate::stages::detection) fn admit_caller_boxes(
            &mut self,
            image: &RgbImage,
            boxes: &[crate::CallerRegion],
            donors: &[KoharuLayoutDetection],
        ) -> usize {
            let mut admitted = 0;
            for region in boxes {
                let Some(probe) = caller_probe(image, *region) else {
                    tracing::info!(?region, "caller box lies off the page; refused");
                    continue;
                };
                /* Measured on a real page: `ink_within` is a
                 * threshold, not a text model, so a rect that overlaps drawn
                 * art distills the art's strokes into the erase mask -- 3.2x
                 * the detector's footprint on the exhibit, and the extra was
                 * the character's hair. Where the detector already SAW the
                 * text, its model mask is the right answer: the reader's rect
                 * decides what is read, the model decides what is erased, and
                 * the donor's kind rides along so a resized sound effect stays
                 * on the SFX path instead of becoming ordinary text. */
                if let Some(inherited) = inherit_from_donors(&probe, donors) {
                    self.detections.push(inherited);
                    self.synthesised.push(false);
                    admitted += 1;
                    continue;
                }
                let Some(mask) = super::ink_within(image, &probe) else {
                    tracing::info!(?region, "caller box holds no ink to read; refused");
                    continue;
                };
                let area = mask.pixels.iter().filter(|value| **value != 0).count() as u32;
                self.detections.push(KoharuLayoutDetection { mask, area, ..probe });
                // Asserted by the reader, not synthesised from a bubble: the
                // flip-reread and its siblings must treat it as ordinary text.
                self.synthesised.push(false);
                admitted += 1;
            }
            admitted
        }

        /// Every detection whose bbox CENTER lies inside one of the reader's
        /// rectangles goes -- applied AFTER the pass, so a synthesised
        /// textless-bubble read (a known false-positive class) is deletable
        /// too, and before `write_masks`, which is what
        /// genuinely un-erases the artwork underneath. Center-inside rather
        /// than an overlap ratio: the editor sends back the exact rectangle
        /// it displayed, so the test only has to be predictable, and a
        /// center cannot half-match.
        pub(in crate::stages::detection) fn remove_caller_boxes(
            &mut self,
            rects: &[crate::CallerRegion],
        ) -> usize {
            if rects.is_empty() {
                return 0;
            }
            let doomed: Vec<bool> = self
                .detections
                .iter()
                .map(|detection| {
                    let center_x = (detection.bbox[0] + detection.bbox[2]) / 2.0;
                    let center_y = (detection.bbox[1] + detection.bbox[3]) / 2.0;
                    rects.iter().any(|rect| {
                        center_x >= rect.x
                            && center_x <= rect.x + rect.width
                            && center_y >= rect.y
                            && center_y <= rect.y + rect.height
                    })
                })
                .collect();
            /* Both vectors filtered by the ONE mask -- the parallel-pairing
             * rule `sort_by_layout` above already documents: filtering one of
             * two parallel vectors is how a pairing rots. */
            let mut keep = doomed.iter().map(|gone| !gone);
            self.detections.retain(|_| keep.next().unwrap());
            let mut keep = doomed.iter().map(|gone| !gone);
            self.synthesised.retain(|_| keep.next().unwrap());
            doomed.iter().filter(|gone| **gone).count()
        }
    }

    /// The inherited admission: the union of every donor's model
    /// mask **whole**, with the largest donor's kind. The masks share the page
    /// coordinate frame (clamped to each mask's own dims), so this is a plain
    /// copy. `None` when no donor matched -- the caller falls through to
    /// `ink_within`, which stays the answer for text the detector never saw.
    ///
    /// **This copy is NOT clipped to the caller's rect.** A clip would
    /// contradict `admit_caller_boxes`' own stated rule -- *"the reader's rect
    /// decides what is READ, the model decides what is ERASED"* -- by letting
    /// the rect veto the model. A reader who SHRANK a box below the donor's own
    /// glyphs would keep every stroke outside their rectangle out of
    /// `text-mask`, so that source ink would never be inpainted; it would stay
    /// invisible under the English lettered into the smaller box, and appear
    /// the moment they dragged the placement box off it. Pinned by
    /// `a_shrunk_box_still_erases_every_glyph_its_donor_saw`.
    ///
    /// Widening the erase past the reader's rectangle is deliberate and is
    /// NOT a "my rect wins" violation: that semantics governs what is READ
    /// and re-translated, which the probe's own bbox still decides. A donor
    /// only matches when its CENTRE lies inside the reader's rect, so a
    /// neighbouring region's text is never dragged in by this.
    ///
    /// The probe's own bbox and score survive untouched: "my rect wins" is
    /// the resize semantics by design, and 1.0 is the wire fingerprint
    /// of a caller-asserted box that the extension's tests pin.
    fn inherit_from_donors(
        probe: &KoharuLayoutDetection,
        donors: &[KoharuLayoutDetection],
    ) -> Option<KoharuLayoutDetection> {
        let [left, top, right, bottom] = probe.bbox;
        let matched: Vec<&KoharuLayoutDetection> = donors
            .iter()
            .filter(|donor| {
                let center_x = (donor.bbox[0] + donor.bbox[2]) / 2.0;
                let center_y = (donor.bbox[1] + donor.bbox[3]) / 2.0;
                center_x >= left && center_x <= right && center_y >= top && center_y <= bottom
            })
            /* TEXT-BEARING donors only. A text box centred in its
             * balloon matches BOTH, their centres coincide, and the balloon is
             * always the larger: taking the largest donor's label handed the
             * reader's box kind `bubble`, which took it off the text path
             * entirely and shipped the bubble untranslated AND un-erased.
             *
             * The mask half is the same filter and matters as much: a balloon's
             * model mask is the whole balloon SHAPE, so unioning it into
             * `text-mask` would inpaint the balloon away -- the same
             * destruction class arriving by a new road. A container donates
             * neither its kind nor its pixels. */
            .filter(|donor| donor.label == super::TEXT || donor.label == super::ONOMATOPOEIA)
            .collect();
        if matched.is_empty() {
            // Only containers matched, or nothing did. `ink_within` decides,
            // which also keeps the probe's own `text` label for a box the
            // reader drew over a balloon the detector found no text in.
            return None;
        }
        let width = probe.mask.width;
        let height = probe.mask.height;
        let mut pixels = vec![0u8; (width as usize) * (height as usize)];
        for donor in &matched {
            /* NO clip to the caller's rect. The donor's model mask
             * IS the text the detector saw for this region, and all of it has
             * to be erased or the reader's shrink leaves source ink on the
             * page. Bounds are each mask's own dims, nothing else. */
            let cols = width.min(donor.mask.width);
            let rows = height.min(donor.mask.height);
            for row in 0..rows {
                for col in 0..cols {
                    let from = (row * donor.mask.width + col) as usize;
                    if donor.mask.pixels[from] != 0 {
                        pixels[(row * width + col) as usize] = donor.mask.pixels[from];
                    }
                }
            }
        }
        let area = pixels.iter().filter(|value| **value != 0).count() as u32;
        if area == 0 {
            // A matched donor with an empty mask. The ink fallback decides,
            // exactly as with no donor at all.
            return None;
        }
        let largest = matched
            .iter()
            .max_by_key(|donor| donor.area)
            .expect("matched is non-empty");
        Some(KoharuLayoutDetection {
            label_id: largest.label_id,
            label: largest.label.clone(),
            mask: KoharuLayoutMask { width, height, pixels },
            area,
            ..probe.clone()
        })
    }

    /// The probe `admit_caller_boxes` hands `ink_within`: the caller's rect as
    /// corner coordinates clamped to the page, with a full-rect mask for the
    /// ink pass to distill. `None` when nothing usable of the rect lies on the
    /// page -- off-page, degenerate, or non-finite.
    fn caller_probe(
        image: &RgbImage,
        region: crate::CallerRegion,
    ) -> Option<KoharuLayoutDetection> {
        if ![region.x, region.y, region.width, region.height]
            .iter()
            .all(|value| value.is_finite())
        {
            return None;
        }
        let left = region.x.max(0.0);
        let top = region.y.max(0.0);
        let right = (region.x + region.width).min(image.width() as f32);
        let bottom = (region.y + region.height).min(image.height() as f32);
        if right - left < 1.0 || bottom - top < 1.0 {
            return None;
        }
        let width = image.width() as usize;
        let mut pixels = vec![0u8; width * image.height() as usize];
        for y in top as u32..bottom as u32 {
            let row = y as usize * width;
            for x in left as u32..right as u32 {
                pixels[row + x as usize] = u8::MAX;
            }
        }
        Some(KoharuLayoutDetection {
            label_id: 0,
            label: TEXT.to_owned(),
            score: 1.0,
            bbox: [left, top, right, bottom],
            area: ((right - left) * (bottom - top)) as u32,
            mask: KoharuLayoutMask {
                width: image.width(),
                height: image.height(),
                pixels,
            },
        })
    }

    impl std::ops::Deref for Settled {
        type Target = [KoharuLayoutDetection];

        fn deref(&self) -> &Self::Target {
            &self.detections
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(in crate::stages::detection) enum NmsOutcome {
        /// Not comparable, or not overlapping: both boxes stand.
        Independent,
        /// The plain NMS answer: the later, lower-scored
        /// box goes.
        SuppressCandidate,
        /// The axis evidence says the kept box is the wrong read of this ink.
        ReplaceKept,
    }

    /// Whether `candidate` is the better read of the same ink as `kept`, on the
    /// axis evidence alone. Two buckets, mutually exclusive by construction:
    ///
    ///   share width, differ height  -> the column is the TALLER box.
    ///     First exhibit: a whole 1023 px column at 0.2383 against a 451 px
    ///     top-only fragment at 0.2715 that shares its width to 4.3%.
    ///   share height, differ width  -> the column is the NARROWER box.
    ///     Second exhibit: a 197 px box that swallows the gloss and merges two
    ///     columns (losing the fourth glyph) against a 122 px tight name column
    ///     that shares its height to 1.2%.
    ///
    /// Both readings are the same underlying fact -- a CJK column is the more
    /// elongated-vertical of two boxes over the same ink -- but the rule is
    /// written as two axis-gated buckets, NOT as "prefer the higher h/w
    /// aspect", because the bare aspect rule would also fire on pairs that
    /// differ on both axes, where there is no evidence either way and today's
    /// answer must stand.
    fn prefers_candidate_on_axis(kept: [f32; 4], candidate: [f32; 4]) -> bool {
        let width = extent_ratio(box_width(kept), box_width(candidate));
        let height = extent_ratio(box_height(kept), box_height(candidate));

        if width <= AXIS_SHARE_MAX_RATIO && height >= AXIS_DIFFER_MIN_RATIO {
            return box_height(candidate) > box_height(kept);
        }
        if height <= AXIS_SHARE_MAX_RATIO && width >= AXIS_DIFFER_MIN_RATIO {
            return box_width(candidate) < box_width(kept);
        }
        false
    }

    /// The whole suppression decision for one `(kept, candidate)` pair, as ONE
    /// predicate. The NMS loop in [`settle_detections`] calls nothing else, so a
    /// test that calls this calls exactly what the caller calls -- including the
    /// `||` between the IoU arm and the containment arm, which the two measured
    /// exhibits separate: the first pair is caught by containment alone (IoU
    /// 0.4229, containment 1.0000) while the second is caught by BOTH (IoU
    /// 0.6100, containment 1.0000). A tie-break wired inside either arm alone
    /// fixes one exhibit and not the other.
    pub(in crate::stages::detection) fn nms_outcome(
        kept: &KoharuLayoutDetection,
        candidate: &KoharuLayoutDetection,
        iou_threshold: f32,
        translate_sfx: bool,
        axis_aware: bool,
    ) -> NmsOutcome {
        /* Same label, or two different labels that will both be lettered. The
         * second arm is deliberately not "any two labels": a bubble and the text
         * inside it overlap almost totally and must both survive. */
        let comparable = kept.label == candidate.label
            || (letters_text(&kept.label, translate_sfx)
                && letters_text(&candidate.label, translate_sfx));
        if !comparable {
            return NmsOutcome::Independent;
        }
        let overlapping = intersection_over_union(kept.bbox, candidate.bbox) >= iou_threshold
            || overlap_over_smaller(kept.bbox, candidate.bbox) >= NMS_CONTAINMENT_THRESHOLD;
        if !overlapping {
            return NmsOutcome::Independent;
        }
        let confident_enough =
            kept.score > 0.0 && candidate.score / kept.score >= AXIS_TIE_BREAK_MIN_SCORE_RATIO;
        if axis_aware && confident_enough && prefers_candidate_on_axis(kept.bbox, candidate.bbox)
        {
            NmsOutcome::ReplaceKept
        } else {
            NmsOutcome::SuppressCandidate
        }
    }

    /// The shortest side a residue strip may have and still be worth reading.
    ///
    /// **Not tuned to the case that motivated it, and the margin says so.** The
    /// measured gloss strip is 73.2 px on its short side and the sliver the
    /// same eviction leaves on the other end is 2.3 px -- a factor of 30 apart,
    /// so any value in a wide band separates them and this one is not sitting in
    /// a gap it was fitted to. What it encodes is the floor below which a strip
    /// cannot hold a legible glyph at all: the theme letters no smaller than 9 px
    /// and a CJK column needs some multiple of that before a reader gets a
    /// character rather than a smudge.
    ///
    /// It is deliberately the CHEAP gate. `ink_within` is the one that decides
    /// whether a strip has anything in it, and it decodes pixels; this exists so
    /// that a hairline never reaches it.
    const RESIDUE_MIN_SIDE: f32 = 24.0;

    /// The parts of `loser` that `winner` does not cover, as whole rectangles.
    ///
    /// Called only from the `ReplaceKept` arm, where `loser` is about to be
    /// overwritten and its uncovered ink would otherwise leave the page with it.
    ///
    /// Up to four strips. Left and right span the loser's full height; top and
    /// bottom span only the horizontal overlap, so a corner is claimed by one
    /// strip rather than two -- overlapping strips would each be measured against
    /// the other by the admission's conflict test and both would lose.
    ///
    /// The axis tie-break's two buckets make one pair or the other empty in
    /// practice (a narrower winner leaves side strips, a shorter one leaves top
    /// and bottom), but nothing here depends on which bucket fired: the geometry
    /// is computed from the two boxes and an empty strip fails the floor.
    ///
    /// Everything but the box is inherited by `..loser.clone()`. The MASK comes
    /// with it, which is the point -- it is page-sized and windowed by the bbox
    /// downstream, so a strip carries the segmentation the detector really
    /// produced under it instead of a synthetic full rectangle whose median
    /// would flip the ink test's polarity on dark artwork.
    pub(in crate::stages::detection) fn residue_strips(
        loser: &KoharuLayoutDetection,
        winner: &KoharuLayoutDetection,
    ) -> Vec<KoharuLayoutDetection> {
        let [lx0, ly0, lx1, ly1] = loser.bbox;
        let [wx0, wy0, wx1, wy1] = winner.bbox;
        let inner0 = lx0.max(wx0);
        let inner1 = lx1.min(wx1);
        [
            [lx0, ly0, lx1.min(wx0), ly1],
            [lx0.max(wx1), ly0, lx1, ly1],
            [inner0, ly0, inner1, ly1.min(wy0)],
            [inner0, ly0.max(wy1), inner1, ly1],
        ]
        .into_iter()
        .filter(|bbox| {
            (bbox[2] - bbox[0]).min(bbox[3] - bbox[1]) >= RESIDUE_MIN_SIDE
        })
        .map(|bbox| KoharuLayoutDetection {
            bbox,
            ..loser.clone()
        })
        .collect()
    }

    /// Per-class NMS, **plus one cross-class rule between the labels that letter**.
    ///
    /// Upstream suppresses only within a label, which is right for a detector whose
    /// classes describe different objects: a `bubble` containing a `text` is the
    /// ordinary case and neither should suppress the other.
    ///
    /// `text` and `onomatopoeia` stopped being different objects when
    /// `translate_sfx` made both letter. RF-DETR readily returns the *same ink
    /// twice*, once under each label, and per-class NMS keeps both by construction.
    /// Both then become `TextRegion`s, both are read by OCR, both are translated,
    /// and both are painted into the same box -- which is the same illegible stack
    /// the shared-bubble cell split exists to prevent, arriving by a route it cannot
    /// see because the two layers have no containing bubble to share.
    ///
    /// **Measured on one test page: four such pairs, byte-identical source on
    /// every one.** The twins are
    /// one region that the detector described twice, so suppressing one loses no
    /// text at all -- and it saves the OCR read and the translation segment the
    /// duplicate was costing.
    ///
    /// **It was invisible until it was not, which is why it survived.** The reply is
    /// one batched call over both segments, so the model usually gives the twins the
    /// same English and painting one over the other looks like nothing. It shows
    /// only when they differ: one pair came back as two different English lines on
    /// the same 26x93 box. Rendered, that is unreadable; in the JSON it is two ordinary
    /// regions. So the defect's visibility is luck, and a run that looks clean is no
    /// evidence of absence.
    ///
    /// **Gated on `translate_sfx` so it cannot fire where it would lose text.** With
    /// effects off, an `onomatopoeia` does not letter, so a `text` twin suppressed in
    /// its favour would erase the words instead of translating them. Off, the set
    /// collapses to `text` alone and no cross-class comparison can happen -- the
    /// function is then upstream's, op for op.
    ///
    /// # This function IS the suppression, and it is written that way on purpose
    ///
    /// The NMS body below was a free `non_maximum_suppression(&mut detections, 0.5,
    /// translate_sfx)` that `write_page` called on one line, and the two passes
    /// around it were added by folding them into it rather than by adding calls
    /// beside it. That is the first half of the defence against this repository's
    /// characteristic failure: a fix once shipped **completely unwired** with all
    /// 128 tests green, because the tests exercised the new function
    /// while the caller still called the old one. The second half is
    /// [`Settled`] -- folding alone was measured insufficient, and the module's
    /// own doc records what it left standing.
    ///
    /// # The order of the three passes is load-bearing
    ///
    /// 1. **Collect the hints**, from the sub-floor boxes, *before* anything
    ///    discards them -- and then discard them here, so nothing downstream can
    ///    mistake a hint for a region. See [`EDGE_HINT_MIN_SCORE`].
    /// 2. **Suppress.**
    /// 3. **Extend**, and only on the survivors. Extending first would let a grown
    ///    box swallow an unrelated region through
    ///    [`NMS_CONTAINMENT_THRESHOLD`] -- the containment arm compares over the
    ///    *smaller* box, so a column grown across half a slice contains every small
    ///    detection in that column's x-range and would delete them.
    pub(in crate::stages::detection) fn settle_detections(
        mut detections: Vec<KoharuLayoutDetection>,
        threshold: f32,
        translate_sfx: bool,
        text_floor: f32,
        /* `Some(band floor)` on a JOINED page with `--joined-page-text-floor`
         * set, `None` otherwise. A `text` box scoring in `[band, text_floor)`
         * is NOT admitted as a region: it enters the suppression below as a
         * REPLACEMENT-ONLY candidate, survives only through
         * `NmsOutcome::ReplaceKept`, and is otherwise dropped exactly as the
         * retain would have dropped it, edge-hint chance included. That
         * containment is the design: a plain lowered floor would have admitted
         * every band box on every composite as a new region -- one measured
         * composite's log alone carries an unrelated 0.2266 strip in the band --
         * which is the broad joined-page admission already measured and rejected. */
        joined_text_floor: Option<f32>,
        size: ImageSize,
        image: &RgbImage,
        /* `Processor::repair_clipped_columns`, threaded rather than read from a
         * constant so the OFF arm is exercised by the same tests as the ON one.
         *
         * With it false this function is the plain suppression exactly: the
         * retain below cannot fire (the network already dropped every sub-floor
         * box, because `network_thresholds` did not lower the request), and the
         * extend pass is skipped outright. Both are asserted, not assumed --
         * `the_off_arm_is_the_old_behaviour_op_for_op`. */
        repair: bool,
        /* `ProcessorConfig::read_textless_bubbles`, threaded for `repair`'s reason:
         * the OFF arm is then exercised by the same tests as the ON one. */
        read_textless: bool,
        /* `ProcessorConfig::axis_aware_nms`, threaded for the same reason again.
         * OFF is the shipping default, and with it false the loop below cannot
         * produce `NmsOutcome::ReplaceKept`, so the OFF arm is today's
         * suppression op for op -- by construction, not by inspection. */
        axis_aware: bool,
        /* `ProcessorConfig::nms_residue_regions`, threaded for that reason a
         * third time.
         *
         * WHAT IT IS FOR. `ReplaceKept` below discards the evicted box WHOLE,
         * and on the measured exhibit that box is the one `prefers_candidate_on_axis`
         * already calls "a 197 px box that swallows the gloss": the winner is
         * the 122 px tight name column, and the 73 px the loser held to its
         * left carry the author's red replacement name -- measured 3,898 red
         * px at x54-119 y176-308, against none of it inside the winner. So
         * dropping the loser takes a whole authored device off the page.
         *
         * The design rule here is to SHRINK a loser rather than drop it, so no
         * text leaves the page: `koharu-renderer`'s `request.rs`
         * (`collision_relief`) argues it in writing and its relief pass
         * implements it, and the same design was deferred for the duplicate gate
         * until a firing was observed that loses a mark the artist drew. This is
         * that firing, and it is measured rather than argued.
         *
         * WHY IT IS GATED SEPARATELY rather than folded into `axis_aware`: a
         * census refuted shipping the tie-break globally, and that census
         * needed a byte-exact control arm. A change riding the same flag would
         * have none. With this false the branch is the plain axis tie-break
         * exactly, op for op, which `the_off_arm_admits_no_residue` asserts.
         *
         * WHY IT CANNOT FIRE ON JAPANESE: it acts only inside `ReplaceKept`,
         * which only `axis_aware` can produce, and `desired_config` scopes that
         * to a positively declared zh/ko. The ja arm is inert by construction --
         * the property that census had to buy with 91 pages. */
        residue: bool,
    ) -> (Settled, Vec<EdgeHint>) {
        /* WHAT THE NETWORK ACTUALLY RETURNED, before a line of this function's
         * filtering touches it. The sub-floor log below covers the `text` boxes
         * this function DROPS, but it fires only for `text`, so a `bubble` or
         * `panel` detection over the same artwork would otherwise be invisible.
         *
         * The distinction matters. If the model finds a balloon's SHAPE on a
         * page while finding none of its text, that is a far
         * smaller problem than "the model is blind on this page" -- and nothing
         * anywhere could tell those apart, because `bubble` and `panel` never become
         * regions on their own and appear on no wire field.
         *
         * THE COUNT IS EMITTED SEPARATELY AND UNCONDITIONALLY, and that is not
         * redundant with the loop. An empty loop prints nothing, which is
         * indistinguishable from a filter that was never switched on -- an
         * ambiguity that would otherwise need a control run. A `count=0` line says
         * "asked, and the answer was none".
         *
         * Note what this is NOT: the network has already applied its own per-class
         * thresholds by the time it returns, so this is everything that cleared
         * them -- `text` at `network_thresholds`' lowered floor, `bubble` at 0.5,
         * `panel` at 0.5, `onomatopoeia` at 0.2. It is not the raw head output
         * and cannot say what sat below those.
         *
         * THOSE NUMBERS ARE READ OFF THE MODEL'S OWN `inference_config.json`, not
         * off the TOML-parsing unit-test fixture in `config.rs`, whose 0.45/0.55
         * would make a measured bubble look like it cleared the floor by 0.058
         * when it clears it by 0.0078. Two quantisation ticks (scores are
         * multiples of 1/256; 0.5 is 128/256 and 0.5078125 is 130/256). The
         * true-positive population sits ON the floor, so there is no headroom
         * underneath it to lower. */
        tracing::debug!(
            count = detections.len(),
            "detections the network returned, before settling"
        );
        for detection in &detections {
            tracing::debug!(
                label = %detection.label,
                score = detection.score,
                bbox = ?detection.bbox,
                "network returned a detection"
            );
        }

        let mut hints = Vec::new();
        /* The lowest score that may PASS the retain. Everything in
         * `[admission_floor, text_floor)` is a replacement-only candidate for
         * the loop below, never a region in its own right. */
        let admission_floor = joined_text_floor
            .map(|floor| floor.min(text_floor))
            .unwrap_or(text_floor);
        detections.retain(|detection| {
            /* Only `text` can be here at all -- `Model::network_thresholds` lowers
             * that one class and no other -- but the test is on the score against
             * the floor this run resolved, not on the label, so a future caller who
             * lowers a second class does not silently admit it as a region. */
            if detection.label != TEXT || detection.score >= admission_floor {
                return true;
            }
            /* Dropped either way. A sub-floor box is not a region under any flag;
             * `repair` decides only whether it is REPORTED on the way out. Keeping
             * the drop unconditional means the off arm cannot start admitting
             * regions if the network is ever asked for a lower floor by some other
             * caller. */
            let hint = if repair {
                edge_hint(detection, size, image, translate_sfx)
            } else {
                None
            };
            /* THE ONLY RECORD A SUB-FLOOR BOX EVER LEAVES, and before this line there
             * was none at all.
             *
             * `edge_hint` reports a box that touches exactly one of the two edge
             * bands. Every other sub-floor box -- including one sitting in the middle
             * of the page, and one touching BOTH bands, which `edge_touched` also
             * returns `None` for -- vanished here silently: not in `regions`, not in
             * `edge_hints`, not in `dropped` or `skipped`, and not in any log. So a
             * page whose text the detector saw only weakly was indistinguishable from
             * a page with no text on it, and that was measured on a test page --
             * TEN counters at zero, every one of them correct, over a balloon a
             * reader can plainly read.
             *
             * `reported` is the field that matters: it separates "the detector saw
             * nothing here" from "the detector saw it and the edge test declined to
             * pass it on", which are different defects with different fixes and which
             * nothing else can tell apart.
             *
             * At `debug`, so it costs nothing under the shipping filter
             * (`birelate_server=info,tower_http=info,info`) and needs
             * `BIRELATE_LOG=info,koharu_pipeline::stages::detection=debug` to appear.
             * Deliberately NOT a counter on the wire: this exists to answer a
             * question, and a wire field would be a shipping-surface change made on
             * the strength of one page. */
            tracing::debug!(
                label = %detection.label,
                score = detection.score,
                bbox = ?detection.bbox,
                reported = hint.is_some(),
                "dropping a sub-floor detection"
            );
            if let Some(hint) = hint {
                hints.push(hint);
            }
            false
        });

        detections.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| detection_order(left, right))
        });
        /* Residue strips harvested from evicted boxes, admitted after the walk.
         * Empty whenever `residue` is false, so the off arm allocates and does
         * nothing. */
        let mut residues: Vec<KoharuLayoutDetection> = Vec::new();
        let mut kept: Vec<KoharuLayoutDetection> = Vec::with_capacity(detections.len());
        // Provenance, one entry per kept detection, grown with `kept` so the two
        // cannot fall out of step. Only the synthesis below writes `true`.
        let mut synthesised_flags: Vec<bool> = Vec::with_capacity(detections.len());
        let mut residue_flags: Vec<bool> = Vec::with_capacity(detections.len());
        for candidate in detections.drain(..) {
            let mut replaceable: Option<usize> = None;
            let mut suppressed = false;
            for (index, existing) in kept.iter().enumerate() {
                match nms_outcome(existing, &candidate, threshold, translate_sfx, axis_aware) {
                    NmsOutcome::Independent => {}
                    /* A suppression from ANY kept box outranks a replacement
                     * offered by another: every multi-conflict case keeps
                     * today's behaviour. */
                    NmsOutcome::SuppressCandidate => {
                        suppressed = true;
                        break;
                    }
                    NmsOutcome::ReplaceKept => {
                        /* A candidate that would evict two kept boxes is a
                         * merge, not a better read of one column. Take today's
                         * answer rather than guessing. */
                        if replaceable.is_some() {
                            suppressed = true;
                            break;
                        }
                        replaceable = Some(index);
                    }
                }
            }
            /* A band box may only ever be a REPLACEMENT. One that was
             * suppressed, or that replaced nothing, is dropped exactly as the
             * retain would have dropped it -- edge-hint chance included, so
             * the repair mechanism is unchanged on joined pages. The band is
             * empty unless `joined_text_floor` was passed, and without
             * `axis_aware` no `ReplaceKept` exists, so the floor flag alone
             * admits NOTHING -- the census control arm is inert by
             * construction, not by measurement. */
            let band = candidate.label == TEXT && candidate.score < text_floor;
            match (suppressed, replaceable) {
                (false, Some(index)) => {
                    /* THE LOSER IS MEASURED BEFORE IT IS OVERWRITTEN.
                     *
                     * `kept[index]` is about to be discarded whole. Whatever it
                     * covered that the winner does not cover is authored ink
                     * with nothing left to carry it -- on the measured exhibit that is the red
                     * replacement name, and the erase mask is built from the
                     * SETTLED list, so today it is not even erased: it ships as
                     * un-translated Chinese beside the English (measured 3,898
                     * red px surviving at 3,908 in the render).
                     *
                     * Admitting the residue is therefore a change with TWO
                     * directions, and the second is why `ink_within` is not
                     * optional here: a strip that becomes a region also enters
                     * the erase mask, so an EMPTY strip would erase artwork to
                     * letter nothing -- a known defect class, bought for free.
                     * The ink test is the safety property, not a refinement.
                     *
                     * Deferred rather than pushed, because the walk's invariant
                     * is that `kept` stays pairwise conflict-free at every step
                     * (the comment below the loop states it and explains why no
                     * re-settle pass exists). A strip admitted here could
                     * conflict with a candidate arriving later; one admitted
                     * after the walk is measured against the final list. */
                    if residue {
                        residues.extend(residue_strips(&kept[index], &candidate));
                    }
                    kept[index] = candidate;
                    synthesised_flags[index] = false;
                    residue_flags[index] = false;
                }
                (false, None) if !band => {
                    kept.push(candidate);
                    synthesised_flags.push(false);
                    residue_flags.push(false);
                }
                (true, _) | (false, None) => {
                    if band {
                        let hint = if repair {
                            edge_hint(&candidate, size, image, translate_sfx)
                        } else {
                            None
                        };
                        tracing::debug!(
                            label = %candidate.label,
                            score = candidate.score,
                            bbox = ?candidate.bbox,
                            reported = hint.is_some(),
                            "dropping a band box that was not a replacement"
                        );
                        if let Some(hint) = hint {
                            hints.push(hint);
                        }
                    }
                }
            }
        }

        /* THE RESIDUE ADMISSION, after the walk and against the final list.
         * Empty and therefore inert whenever the flag is off.
         *
         * Two gates, in cost order. A strip must clear the geometry floor
         * first (that is arithmetic on four floats), and only what survives it
         * pays for `ink_within`, which decodes the strip's pixels.
         *
         * The conflict test is the walk's own `nms_outcome`, so a strip that
         * overlaps something already kept is refused by the same rule that
         * settled the rest of the page rather than by a second opinion. Only
         * `Independent` against EVERY kept box admits.
         *
         * `..loser.clone()` carries the loser's label, score and mask across;
         * the mask is page-sized and `ink_within` windows it by the bbox, so a
         * strip inherits the segmentation the detector actually produced there
         * rather than a synthetic full rectangle. That matters: a full-rect
         * mask makes `paper` the median of the whole box and the ink test then
         * runs in reverse polarity on dark artwork. */
        for strip in residues {
            if ink_within(image, &strip).is_none() {
                tracing::debug!(
                    bbox = ?strip.bbox,
                    "refusing an evicted box's residue: no ink in it"
                );
                continue;
            }
            let clear = kept.iter().all(|existing| {
                matches!(
                    nms_outcome(existing, &strip, threshold, translate_sfx, axis_aware),
                    NmsOutcome::Independent
                )
            });
            if !clear {
                tracing::debug!(
                    bbox = ?strip.bbox,
                    "refusing an evicted box's residue: it conflicts with a kept box"
                );
                continue;
            }
            tracing::debug!(
                label = %strip.label,
                score = strip.score,
                bbox = ?strip.bbox,
                "admitting the residue of a box the axis tie-break evicted"
            );
            kept.push(strip);
            synthesised_flags.push(false);
            residue_flags.push(true);
        }

        /* No re-settle pass is needed after a replacement, and that is an
         * invariant rather than an oversight: `nms_outcome`'s conflict test is
         * SYMMETRIC in its two boxes, the walk above checks the LIVE `kept`
         * list (a candidate arriving after a replacement is measured against
         * the replacement, not the box it evicted), and a candidate is admitted
         * only when it conflicts with at most one kept box -- so `kept` is
         * pairwise conflict-free at every step. The first draft of this change
         * carried a defensive re-sweep here; it was removed because no fixture
         * could make it fire, and a pass that cannot go red is not a check. */

        /* Skipped WHOLE when the flag is off, rather than computed and thrown
         * away. The walk decodes ink over a full page-height strip per candidate,
         * so this is also the only part of the column repair with a cost worth
         * not paying. */
        if repair {
            for detection in &mut kept {
                if let Some(grown) = repaired_column(detection, size, image, translate_sfx) {
                    tracing::debug!(
                        label = %detection.label,
                        score = detection.score,
                        was = ?detection.bbox,
                        now = ?grown,
                        "growing a column the detector truncated at a page edge"
                    );
                    detection.bbox = grown;
                }
            }
        }

        /* A BUBBLE THAT HOLDS NO TEXT REGION IS READ AS ONE.
         *
         * The detector finds a balloon's shape and its text INDEPENDENTLY, and on
         * one test page it found the shape at 0.5078 and none of the text --
         * so the balloon reached a reader in Chinese while every counter on the
         * wire read zero and every one of them was correct. This is the only rule
         * in this file that acts UPSTREAM of that failure rather than downstream of
         * it, which is why nothing else could reach that page.
         *
         * ## Why the whole bubble, and not an inset of it
         *
         * There is no text box to inset from -- that is the defect. The bubble's
         * own geometry and its own segmentation mask are what exist, and they are
         * the right answer for a balloon: a balloon takes a flat fill rather than
         * an inpaint, so erasing its interior costs nothing a reader can see. The
         * mask is CLONED rather than rebuilt, so the eraser and this rule cannot
         * disagree about the shape.
         *
         * ## Why the measured font size does not matter here
         *
         * `mask_font_size` would read typography off the BALLOON rather than off
         * glyphs and return nonsense. It is never consulted: `text_renderer.rs`
         * takes `automatic_maximum()` instead of `layer.font_size` whenever
         * `auto_fit && is_bubble_text`, and a region sitting inside a detected
         * bubble is bubble text by construction.
         *
         * ## The false positive, and what pays for it
         *
         * A detection census over 750 pages and 1,565 bubbles found 4 bubbles
         * holding no `text` DETECTION, but that is not a rate for this rule: the
         * census ran with OCR disabled, so it counted bubbles holding no `text`
         * detection -- not bubbles this synthesis fires on. Rendering all four on
         * one build added a region on exactly **two** pages, one true and one
         * false. The other two are byte-identical or one pixel. The census also
         * counted only the `text` class, while the gate above counts
         * `onomatopoeia` too when `translate_sfx` is on -- one of the four
         * balloons holds an SFX detection, so the census called it text-less and
         * this code correctly does not.
         * **On the population this rule acts on, it is 1 true and 1 false.**
         *
         * **`withdraw_unread_masks` does NOT pay for it.** It fires only where
         * `refuse_region` refused the box **for size** (`ocr.rs`); the false
         * positive's region is 0.0455 of its page, so it is read normally and
         * `unread` stays empty. What it covers is a read
         * that is ABSENT. This is a read that is WRONG: OCR returns `ー` off the
         * character's own mouth line, `illegible_text` sees one non-Latin letter and
         * passes it, and the em dash is lettered across her face. Measured: of the
         * 9,290 px that change on that page's band, the glyph is 755 -- **8,535 px
         * are artwork destroyed by the erase alone, 92% of the damage.**
         *
         * The absolute ink floor does not gate it either: the false positive
         * clears 200 with **795** ink pixels, against the true positive's
         * **40,946**. The floor is 51x too low to separate them, not incapable of
         * it; `SYNTHESISED_INK_MIN_FRACTION` is the gate that does.
         *
         * `label_id` is copied from the bubble rather than invented. Nothing on
         * this crate's path reads it -- its only consumers are two debug binaries
         * in `koharu-ml` -- and minting an id for a class this detection is not
         * would be the worse lie.
         */
        if read_textless {
            /* THE CENSUS ARM, and it does NOT run in a shipping run.
             *
             * `ink_within` is only ever called on a TEXT-LESS bubble, so the only
             * ink fractions this pipeline can observe are the handful of pages the
             * synthesis fires on -- one true and one false in the whole 750-page
             * corpus. That is not a population, and a floor derived from it would
             * be two points with a line through them.
             *
             * A bubble that HOLDS detected text is a known-real balloon, and there
             * are thousands of those. Measuring the same statistic over them gives
             * the accept side its reference distribution, using the real function
             * rather than a Python re-implementation that could drift from it --
             * which is the divergence trap this file already carries two warnings
             * about.
             *
             * Gated on the debug level actually being enabled, so a shipping run
             * pays nothing: no extra per-pixel pass, no log. The result is
             * discarded; `ink_within`'s own line is the output. */
            if tracing::enabled!(tracing::Level::DEBUG) {
                for bubble in kept.iter().filter(|bubble| bubble.label == "bubble") {
                    let holds_text = kept.iter().any(|other| {
                        letters_text(&other.label, translate_sfx)
                            && containment(bubble.bbox, other.bbox) >= 0.5
                    });
                    tracing::debug!(holds_text, "census: about to measure a bubble");
                    let _ = ink_within(image, bubble);
                }
            }

            let synthesised: Vec<KoharuLayoutDetection> = kept
                .iter()
                .filter(|bubble| bubble.label == "bubble")
                .filter(|bubble| {
                    !kept.iter().any(|other| {
                        letters_text(&other.label, translate_sfx)
                            && containment(bubble.bbox, other.bbox) >= 0.5
                    })
                })
                .filter_map(|bubble| {
                    /* THE MASK IS THE BALLOON'S INK, NOT ITS BODY, AND A RENDER
                     * SHOWED WHY. Cloning the body put
                     * the balloon into the erase mask (79.7% of one balloon's white
                     * destroyed) and made `infer_typography` measure the paper
                     * (`font_size 322.0`, `color [255,255,255]` -- white on white).
                     * One mask feeds both, so one correction fixes both.
                     *
                     * `None` here means the balloon holds no ink, so no region is
                     * invented for it -- which is also the first thing this rule has
                     * ever had against its known false positive, a spurious
                     * bubble over flat skin. */
                    let mask = ink_within(image, bubble)?;
                    let area = mask.pixels.iter().filter(|value| **value != 0).count() as u32;
                    tracing::debug!(
                        score = bubble.score,
                        bbox = ?bubble.bbox,
                        ink = area,
                        "reading a bubble that holds no text region of its own"
                    );
                    Some(KoharuLayoutDetection {
                        label_id: bubble.label_id,
                        label: TEXT.to_owned(),
                        score: bubble.score,
                        bbox: bubble.bbox,
                        // The INK's area, so the mask and the count describe one
                        // object rather than the count still describing the balloon.
                        area,
                        mask,
                    })
                })
                .collect();
            /* THE BBOX IS THE BUBBLE'S OWN, UNSHRUNK, AND THAT IS DELIBERATE.
             *
             * The region this becomes must be turned to the balloon's angle and
             * must NOT inflate when it is -- see `oriented_ink_box`. The obvious
             * place to do that is right here, handing over a smaller box; it was
             * built that way first and it does not work. `infer_typography`
             * re-measures the angle downstream through `mask_window` on whatever
             * box arrives here, so a box shrunk to the balloon's own axes gives
             * that pass a CLIPPED balloon to measure and the angle drifts:
             * measured, 12.5 degrees on the whole mask against **6.5** on the
             * shrunken box, for a hull of 1031 x 290 where the ink is 1032 x 392.
             * Under the ceiling, and 102px short of the glyphs it exists to read.
             *
             * So the shrink happens in `build_region` instead, at the angle that
             * pass has just measured off the WHOLE mask -- one measurement, used
             * for both the extent and the turn, which is the only arrangement in
             * which the two cannot disagree. What this pass records is which
             * detections are synthesised; `Settled` carries that. */
            synthesised_flags.resize(kept.len() + synthesised.len(), true);
            residue_flags.resize(kept.len() + synthesised.len(), false);
            kept.extend(synthesised);
        }

        debug_assert_eq!(
            kept.len(),
            synthesised_flags.len(),
            "the provenance vector must hold one entry per kept detection"
        );
        debug_assert_eq!(
            kept.len(),
            residue_flags.len(),
            "the residue vector must hold one entry per kept detection"
        );
        (
            Settled {
                detections: kept,
                synthesised: synthesised_flags,
                residue: residue_flags,
            },
            hints,
        )
    }

    /// Whether a sub-floor `text` box is worth mentioning to a caller that assembles
    /// pages out of slices, and where. `None` for every box that is not.
    ///
    /// The gate is deliberately **not** the column aspect rule the repair uses. The
    /// box this exists for is one measured slice's `[52.1, 712.9, 243.8, 904.5]`, which is
    /// 191.6 x 191.5 -- near-square, and it fails
    /// [`FREE_TEXT_MIN_ASPECT`] outright. It is still the top of a display column;
    /// it is square because the detector only found part of one. A hint is a report,
    /// not a decision, so it costs nothing to be generous here and everything to be
    /// wrong in the other direction.
    ///
    /// What it does refuse is a speck. Both sides must clear
    /// [`FREE_TEXT_ROTATED_MIN_WIDTH`], because a 4px mark against the top of a page
    /// tells a joiner nothing it can use.
    /// # The walk runs on sub-floor boxes too
    ///
    /// [`repaired_column`] is called here, on a box that will never be a region,
    /// purely to answer "does this box's own ink cross the slice?". When it does,
    /// the hint carries the **grown** geometry and `spans: true`, which is the only
    /// thing that lets the browser's run detection step across a slice the detector
    /// refused outright.
    ///
    /// **It is the same call the repair loop makes on survivors**, not a copy: the
    /// guards that keep this off manga -- exactly one edge band, column-shaped, and
    /// the ink must really arrive -- are therefore identical by construction rather
    /// than by review. Measured need: one test slice has no admitted box at
    /// all, and its best candidate scores **0.2295** against a 0.25 floor while its
    /// ink runs the full 907 px of the slice. Without this, that slice could become a
    /// TOP hint and nothing more, so no run could pass through it and the column was
    /// read as two fragments -- the confabulation shape the slice-fragment refusal
    /// exists to stop.
    ///
    /// A box whose ink does NOT reach keeps its raw geometry and `spans: false`, and
    /// is exactly the plain one-sided hint.
    fn edge_hint(
        detection: &KoharuLayoutDetection,
        size: ImageSize,
        image: &RgbImage,
        translate_sfx: bool,
    ) -> Option<EdgeHint> {
        if detection.score < EDGE_HINT_MIN_SCORE {
            return None;
        }
        let [left, top, right, bottom] = detection.bbox;
        let (width, height) = (right - left, bottom - top);
        if width < FREE_TEXT_ROTATED_MIN_WIDTH || height < FREE_TEXT_ROTATED_MIN_WIDTH {
            return None;
        }
        let edge = edge_touched(detection.bbox, size)?;

        /* Grown, or left exactly as it was. `repaired_column` re-derives the edge
         * and re-checks the label and the aspect itself, so this cannot admit a
         * box the repair loop would have refused. */
        if let Some(grown) = repaired_column(detection, size, image, translate_sfx) {
            let [gleft, gtop, gright, gbottom] = grown;
            return Some(EdgeHint {
                x: gleft,
                y: gtop,
                width: gright - gleft,
                height: gbottom - gtop,
                edge,
                score: detection.score,
                spans: true,
            });
        }

        Some(EdgeHint {
            x: left,
            y: top,
            width,
            height,
            edge,
            // The raw score, from the same detection the geometry above came from,
            // so a caller can tell 0.2295 from 0.0102 rather than being told only
            // that something was refused. See `EdgeHint::score`.
            score: detection.score,
            // The ink did not reach the far band, so this is a fragment that really
            // does end inside its slice. It stays a one-sided hint and cannot reach
            // `seamSpansSlice`.
            spans: false,
        })
    }

    /// The box this detection should have had, or `None` to leave it exactly alone.
    ///
    /// A box that touches EXACTLY ONE edge band, is shaped like a column, and whose
    /// **own ink continues to the opposite band**, is a column crossing this slice
    /// rather than one that ends inside it -- so it is grown to the page edge, and
    /// the size guards downstream then see it for what it is.
    ///
    /// Every clause is doing work:
    ///
    /// - *Exactly one band*, and this is what keeps the whole mechanism off manga.
    ///   1,221 boxes on one Japanese test volume are tall and narrow, because
    ///   vertical Japanese is that shape -- the aspect rule is no guard at all there.
    ///   Measured over 213 manga pages and 2,337 regions: **3 qualify, 0 grow.** Over
    ///   a 179-slice manhua test chapter: 9 qualify, 3 grow.
    /// - *Touching both* is excluded by the same clause and deliberately: such a box
    ///   already spans its slice, there is no direction to walk in, and the
    ///   slice-fragment refusal already has a rule for it.
    /// - *The ink must actually reach*. Without it this would invent a column
    ///   wherever the detector clipped a heading, which is the failure mode of a
    ///   brightness probe -- measured firing on the white paper of manga and
    ///   turning 0 manga joins into 22.
    ///   The ink here is [`ink_components`], the eraser's own rule, so a box grown
    ///   on one definition of ink cannot be erased on another.
    fn repaired_column(
        detection: &KoharuLayoutDetection,
        size: ImageSize,
        image: &RgbImage,
        translate_sfx: bool,
    ) -> Option<[f32; 4]> {
        if !letters_text(&detection.label, translate_sfx) {
            return None;
        }
        if !column_shaped(detection.bbox) {
            return None;
        }
        let edge = edge_touched(detection.bbox, size)?;
        if !ink_reaches_far_band(image, detection.bbox, size, edge) {
            return None;
        }
        let mut bbox = detection.bbox;
        match edge {
            // Only the far side moves. The near one is already inside the band, and
            // snapping it too would widen the erase footprint for nothing.
            PageEdge::Top => bbox[3] = size.height as f32,
            PageEdge::Bottom => bbox[1] = 0.0,
        }
        Some(bbox)
    }

    /// How deep into a page its top and bottom edge bands run. See
    /// [`EDGE_BAND_FRACTION`].
    fn edge_band(height: u32) -> u32 {
        EDGE_BAND_FLOOR_PX.max((height as f32 * EDGE_BAND_FRACTION).round() as u32)
    }

    /// Which single edge band this box is up against, or `None` for neither and for
    /// both. See [`repaired_column`] for why "both" is a `None` and not a third arm.
    fn edge_touched(bbox: [f32; 4], size: ImageSize) -> Option<PageEdge> {
        let band = edge_band(size.height) as f32;
        let touches_top = bbox[1] <= band;
        let touches_bottom = bbox[3] >= size.height as f32 - band;
        match (touches_top, touches_bottom) {
            (true, false) => Some(PageEdge::Top),
            (false, true) => Some(PageEdge::Bottom),
            _ => None,
        }
    }

    /// Whether this box is the shape a vertical display column is.
    ///
    /// The same predicate `free_text_column_geometry` turns on, reused rather than
    /// re-derived so the repair claims exactly the population the turn does.
    fn column_shaped(bbox: [f32; 4]) -> bool {
        let width = bbox[2] - bbox[0];
        let height = bbox[3] - bbox[1];
        height * FREE_TEXT_MIN_ASPECT > width && width >= FREE_TEXT_ROTATED_MIN_WIDTH
    }

    /// Walks this box's own ink away from the edge it already touches, and answers
    /// whether it arrives at the opposite band.
    ///
    /// The walk starts at the box's far side and steps a row at a time, over the
    /// box's x-range padded by [`INK_MASK_PAD`] -- the eraser's own padding, for the
    /// eraser's own reason, that a glyph's anti-aliased skirt sits outside the box
    /// the detector drew. A row holding at least [`INK_WALK_MIN_ROW_PIXELS`] ink
    /// pixels is ink; [`INK_WALK_MAX_GAP_ROWS`] consecutive paper rows end the walk;
    /// [`INK_WALK_REACH_RATIO`] bounds how far it may go at all.
    ///
    /// The crop handed to [`ink_components`] deliberately spans the BOX as well as
    /// the walk window. Otsu needs to see both ink and paper to put its level
    /// anywhere sensible, and a window past the end of a column that really has
    /// ended is nothing but paper -- thresholded alone it would resolve into noise
    /// and the walk would follow it.
    fn ink_reaches_far_band(
        image: &RgbImage,
        bbox: [f32; 4],
        size: ImageSize,
        edge: PageEdge,
    ) -> bool {
        let [left, top, right, bottom] = bbox;
        let x0 = (left.floor().max(0.0) as u32).saturating_sub(INK_MASK_PAD);
        let x1 = ((right.ceil().max(0.0) as u32) + INK_MASK_PAD).min(size.width);
        let reach = ((bottom - top).max(0.0) * INK_WALK_REACH_RATIO).max(0.0);
        let box_top = top.floor().max(0.0).min(size.height as f32) as u32;
        let box_bottom = bottom.ceil().max(0.0).min(size.height as f32) as u32;
        let (crop_top, crop_bottom) = match edge {
            PageEdge::Top => (
                box_top,
                ((bottom + reach).ceil().max(0.0) as u32).min(size.height),
            ),
            PageEdge::Bottom => (
                ((top - reach).floor().max(0.0)) as u32,
                box_bottom,
            ),
        };
        if x1 <= x0 + 2 || crop_bottom <= crop_top + 2 {
            return false;
        }

        let ink = ink_components(image, x0, crop_top, x1, crop_bottom);
        let inked = |row: u32| -> bool {
            let mut count = 0u32;
            for x in 0..ink.width() {
                if ink.get_pixel(x, row)[0] != 0 {
                    count += 1;
                    if count >= INK_WALK_MIN_ROW_PIXELS {
                        return true;
                    }
                }
            }
            false
        };

        let band = edge_band(size.height);
        let mut gap = 0u32;
        match edge {
            PageEdge::Top => {
                // Away from the top band means downward, from the box's own bottom.
                for y in box_bottom.max(crop_top)..crop_bottom {
                    if inked(y - crop_top) {
                        gap = 0;
                        if y >= size.height.saturating_sub(band) {
                            return true;
                        }
                    } else {
                        gap += 1;
                        if gap > INK_WALK_MAX_GAP_ROWS {
                            return false;
                        }
                    }
                }
            }
            PageEdge::Bottom => {
                for y in (crop_top..box_top.min(crop_bottom)).rev() {
                    if inked(y - crop_top) {
                        gap = 0;
                        if y <= band {
                            return true;
                        }
                    } else {
                        gap += 1;
                        if gap > INK_WALK_MAX_GAP_ROWS {
                            return false;
                        }
                    }
                }
            }
        }
        false
    }
}

/* `sort_by_layout` stood here and is GONE, deliberately rather than by accident.
 *
 * It took `&mut Vec<KoharuLayoutDetection>` and permuted it in place, which is
 * exactly the signature that cannot reach the provenance vector `Settled` now
 * carries beside those detections. Its one caller is `Settled::sort_by_layout`,
 * which takes `layout_order` directly and applies the SAME permutation to both --
 * because sorting one of two parallel vectors is how a pairing rots silently.
 *
 * Left as free, it was dead code that still looked like the way to sort settled
 * detections, and the next caller to reach for it would have dropped the
 * provenance on the floor with no warning. */

fn layout_order(detections: &[KoharuLayoutDetection]) -> Vec<usize> {
    let panels = indices_with_label(detections, "panel");
    let bubbles = indices_with_label(detections, "bubble");
    let texts = indices_with_label(detections, "text");
    let panels = spatial_order(detections, panels);

    let mut panel_for_bubble = vec![None; detections.len()];
    for &bubble in &bubbles {
        panel_for_bubble[bubble] = best_container(detections, bubble, &panels);
    }
    let mut bubble_for_text = vec![None; detections.len()];
    let mut panel_for_text = vec![None; detections.len()];
    for &text in &texts {
        let bubble = best_container(detections, text, &bubbles);
        bubble_for_text[text] = bubble;
        panel_for_text[text] = bubble
            .and_then(|bubble| panel_for_bubble[bubble])
            .or_else(|| best_container(detections, text, &panels));
    }

    let mut order = Vec::with_capacity(detections.len());
    let mut included = vec![false; detections.len()];
    for &panel in &panels {
        append_once(panel, &mut order, &mut included);
        for bubble in spatial_order(
            detections,
            bubbles
                .iter()
                .copied()
                .filter(|&bubble| panel_for_bubble[bubble] == Some(panel))
                .collect(),
        ) {
            append_once(bubble, &mut order, &mut included);
            append_texts(detections, &texts, &mut order, &mut included, |text| {
                bubble_for_text[text] == Some(bubble)
            });
        }
        append_texts(detections, &texts, &mut order, &mut included, |text| {
            bubble_for_text[text].is_none() && panel_for_text[text] == Some(panel)
        });
    }

    for bubble in spatial_order(
        detections,
        bubbles
            .iter()
            .copied()
            .filter(|&bubble| panel_for_bubble[bubble].is_none())
            .collect(),
    ) {
        append_once(bubble, &mut order, &mut included);
        append_texts(detections, &texts, &mut order, &mut included, |text| {
            bubble_for_text[text] == Some(bubble)
        });
    }

    append_texts(detections, &texts, &mut order, &mut included, |text| {
        bubble_for_text[text].is_none() && panel_for_text[text].is_none()
    });
    for index in spatial_order(
        detections,
        (0..detections.len())
            .filter(|&index| !included[index])
            .collect(),
    ) {
        append_once(index, &mut order, &mut included);
    }
    order
}

fn indices_with_label(detections: &[KoharuLayoutDetection], label: &str) -> Vec<usize> {
    detections
        .iter()
        .enumerate()
        .filter_map(|(index, detection)| (detection.label == label).then_some(index))
        .collect()
}

fn append_texts(
    detections: &[KoharuLayoutDetection],
    texts: &[usize],
    order: &mut Vec<usize>,
    included: &mut [bool],
    belongs: impl Fn(usize) -> bool,
) {
    for text in spatial_order(
        detections,
        texts
            .iter()
            .copied()
            .filter(|text| belongs(*text))
            .collect(),
    ) {
        append_once(text, order, included);
    }
}

fn append_once(index: usize, order: &mut Vec<usize>, included: &mut [bool]) {
    if !included[index] {
        included[index] = true;
        order.push(index);
    }
}

fn spatial_order(detections: &[KoharuLayoutDetection], mut indices: Vec<usize>) -> Vec<usize> {
    indices.sort_by(|&left, &right| {
        detection_order(&detections[left], &detections[right])
            .then_with(|| detections[right].score.total_cmp(&detections[left].score))
            .then_with(|| left.cmp(&right))
    });
    indices
}

fn best_container(
    detections: &[KoharuLayoutDetection],
    value: usize,
    candidates: &[usize],
) -> Option<usize> {
    candidates
        .iter()
        .copied()
        .filter(|&candidate| containment(detections[candidate].bbox, detections[value].bbox) >= 0.5)
        .min_by(|&left, &right| {
            area(detections[left].bbox)
                .total_cmp(&area(detections[right].bbox))
                .then_with(|| detection_order(&detections[left], &detections[right]))
                .then_with(|| left.cmp(&right))
        })
}

fn containment(container: [f32; 4], value: [f32; 4]) -> f32 {
    let value_area = area(value);
    if value_area <= 0.0 {
        return 0.0;
    }
    intersection_area(container, value) / value_area
}

fn intersection_over_union(left: [f32; 4], right: [f32; 4]) -> f32 {
    let intersection = intersection_area(left, right);
    let union = area(left) + area(right) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn overlap_over_smaller(left: [f32; 4], right: [f32; 4]) -> f32 {
    let smaller = area(left).min(area(right));
    if smaller <= 0.0 {
        0.0
    } else {
        intersection_area(left, right) / smaller
    }
}

fn box_width(bbox: [f32; 4]) -> f32 {
    bbox[2] - bbox[0]
}

fn box_height(bbox: [f32; 4]) -> f32 {
    bbox[3] - bbox[1]
}

/// The larger of two extents over the smaller, so the answer is >= 1.0 and
/// order-free. A degenerate extent is infinitely far from anything.
fn extent_ratio(left: f32, right: f32) -> f32 {
    let (large, small) = if left >= right { (left, right) } else { (right, left) };
    if small <= 0.0 { f32::INFINITY } else { large / small }
}

fn intersection_area(left: [f32; 4], right: [f32; 4]) -> f32 {
    (left[2].min(right[2]) - left[0].max(right[0])).max(0.0)
        * (left[3].min(right[3]) - left[1].max(right[1])).max(0.0)
}

fn area(bounds: [f32; 4]) -> f32 {
    (bounds[2] - bounds[0]).max(0.0) * (bounds[3] - bounds[1]).max(0.0)
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};
    use koharu_ml::koharu_layout_rfdetr_seg_2xl::{KoharuLayoutDetection, KoharuLayoutMask};
    use koharu_scene::{RegionSpec, TextRegion, WritingMode};

    use super::{
        ANGLE_SNAP_DEGREES, EdgeHint, FREE_TEXT_MIN_ASPECT, FREE_TEXT_MIN_WIDTH,
        FREE_TEXT_SHORT_BOX_HEIGHT, FREE_TEXT_STROKE_MINIMUM_WIDTH,
        FREE_TEXT_STROKE_RATIO, Geometry, HEAVY_INK_RATIO, HEAVY_INK_WEIGHT, ImageSize,
        InferredTypography, PageEdge,
        SYNTHESISED_INK_MIN_FRACTION, SYNTHESISED_INK_MIN_PIXELS, TEXT,
        box_center, bubble_text_cell, device_strike_ink, free_text_cell,
        free_text_column_geometry, free_text_geometry, gloss_ink,
        infer_typography, intersection_area,
        layout_order, letters_text, mask_for, mask_geometry, mask_includes,
        normalize_text_color, oriented_ink_box, rectangle_geometry, region_geometry,
        region_kind, rotated_rectangle_geometry, sampled_ink, scaled_dilated_mask,
        settle::{NmsOutcome, Settled, nms_outcome, settle_detections},
        strike_ink,
        turn_is_offered,
        text_stroke,
    };

    /// Regions are reported as `(x, y, width, height)`; the cell arithmetic
    /// works in `[left, top, right, bottom]`.
    fn rect(x: f32, y: f32, width: f32, height: f32) -> [f32; 4] {
        [x, y, x + width, y + height]
    }

    fn cells(frame: [f32; 4], texts: &[[f32; 4]]) -> Vec<[f32; 4]> {
        texts
            .iter()
            .enumerate()
            .map(|(index, bounds)| {
                let neighbours = texts
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| *other != index)
                    .map(|(_, value)| *value)
                    .collect::<Vec<_>>();
                bubble_text_cell(frame, *bounds, &neighbours)
                    .unwrap_or_else(|| panic!("text {index} keeps a usable cell"))
            })
            .collect()
    }

    fn assert_cells_partition(frame: [f32; 4], texts: &[[f32; 4]]) -> Vec<[f32; 4]> {
        let cells = cells(frame, texts);
        for (index, cell) in cells.iter().enumerate() {
            let bounds = texts[index];
            assert!(
                cell[0] <= bounds[0]
                    && cell[1] <= bounds[1]
                    && cell[2] >= bounds[2]
                    && cell[3] >= bounds[3],
                "cell {index} {cell:?} does not contain its own text {bounds:?}"
            );
            assert!(
                cell[0] >= frame[0]
                    && cell[1] >= frame[1]
                    && cell[2] <= frame[2]
                    && cell[3] <= frame[3],
                "cell {index} {cell:?} escapes the bubble frame {frame:?}"
            );
            for (other, neighbour) in cells.iter().enumerate().skip(index + 1) {
                assert_eq!(
                    intersection_area(*cell, *neighbour),
                    0.0,
                    "cells {index} and {other} overlap: {cell:?} against {neighbour:?}"
                );
            }
        }
        cells
    }

    #[test]
    fn a_lone_text_keeps_the_whole_bubble_frame() {
        let frame = rect(75.0, 638.0, 118.0, 416.0);
        assert_eq!(
            bubble_text_cell(frame, rect(87.0, 659.0, 97.0, 150.0), &[]),
            Some(frame)
        );
    }

    #[test]
    fn stacked_texts_in_one_bubble_split_across() {
        // Measured on a real page: two texts sharing fit(75, 638, 118, 416).
        let frame = rect(75.0, 638.0, 118.0, 416.0);
        let texts = [
            rect(87.0, 659.0, 97.0, 150.0),
            rect(98.0, 880.0, 70.0, 115.0),
        ];

        let cells = assert_cells_partition(frame, &texts);

        // Centres are 203px apart vertically and 2px apart horizontally, so the
        // cut is horizontal. The cells do NOT meet at the cut and do not span the
        // frame: each is pulled in symmetrically about the text it replaces, so
        // the rendered line lands where the Japanese was rather than in the
        // middle of whatever space happened to be free.
        for (cell, text) in cells.iter().zip(&texts) {
            assert_eq!(box_center(*cell), box_center(*text));
        }
    }

    #[test]
    fn side_by_side_texts_in_one_bubble_split_down() {
        // Measured on a real page: two texts sharing fit(292, 537, 187, 97).
        let frame = rect(292.0, 537.0, 187.0, 97.0);
        let texts = [
            rect(401.0, 545.0, 49.0, 64.0),
            rect(309.0, 545.0, 38.0, 76.0),
        ];

        let cells = assert_cells_partition(frame, &texts);

        // Same rule on the other axis: centred on its own text, not filling the
        // half of the balloon the cut left free.
        for (cell, text) in cells.iter().zip(&texts) {
            assert_eq!(box_center(*cell), box_center(*text));
        }
    }

    #[test]
    fn a_two_dimensional_cluster_splits_on_both_axes() {
        // Measured on a real page: six texts sharing fit(56, 0, 350, 390),
        // whose 250px horizontal spread and 270px vertical spread are close
        // enough that neither axis alone can separate them.
        let frame = rect(56.0, 0.0, 350.0, 390.0);
        let texts = [
            rect(166.0, 35.0, 38.0, 50.0),
            rect(343.0, 50.0, 33.0, 65.0),
            rect(234.0, 61.0, 72.0, 82.0),
            rect(93.0, 108.0, 57.0, 49.0),
            rect(111.0, 197.0, 68.0, 50.0),
            rect(104.0, 305.0, 39.0, 50.0),
        ];

        let cells = assert_cells_partition(frame, &texts);

        assert!(
            cells.iter().any(|cell| cell[0] > frame[0]),
            "no cell was cut from the left"
        );
        assert!(
            cells.iter().any(|cell| cell[2] < frame[2]),
            "no cell was cut from the right"
        );
        assert!(
            cells.iter().any(|cell| cell[1] > frame[1]),
            "no cell was cut from the top"
        );
        assert!(
            cells.iter().any(|cell| cell[3] < frame[3]),
            "no cell was cut from the bottom"
        );
    }

    #[test]
    fn a_cell_squeezed_by_a_close_neighbour_stops_at_its_own_text() {
        // Two near-identical boxes: the midpoint cut would leave less width than
        // the text itself occupies.
        //
        // This used to return `None`, and the caller then fell back to the text's
        // own region -- which for vertical Japanese is the source ink column, far
        // too narrow for the horizontal English, and measurably the largest
        // render defect on real pages. The cut is clamped instead, so the cell
        // bottoms out at exactly the text's own bounds: the same layout the old
        // fallback produced, reached without giving up the whole balloon in every
        // less extreme case. It is no worse for stacking either, since refusing
        // also handed both texts overlapping frames.
        let frame = rect(0.0, 0.0, 100.0, 20.0);
        let bounds = rect(10.0, 0.0, 80.0, 20.0);
        let cell = bubble_text_cell(frame, bounds, &[rect(14.0, 0.0, 80.0, 20.0)])
            .expect("a clamped cut always leaves a cell");
        assert_eq!(cell, bounds, "squeezed to exactly its own ink, no further");
    }

    #[test]
    fn a_cell_always_covers_its_own_text_and_never_escapes_the_frame() {
        // The invariant the clamping buys, stated directly: whatever the
        // neighbours do, the cell contains the text it belongs to and stays
        // inside the balloon. Those two bounds are what make it never worse than
        // the caller's fallback and never larger than the frame.
        let frame = rect(0.0, 0.0, 200.0, 200.0);
        let bounds = rect(80.0, 80.0, 40.0, 40.0);
        for neighbour in [
            rect(84.0, 80.0, 40.0, 40.0),   // almost on top of it
            rect(0.0, 80.0, 40.0, 40.0),    // hard left
            rect(160.0, 80.0, 40.0, 40.0),  // hard right
            rect(80.0, 0.0, 40.0, 40.0),    // above
            rect(80.0, 160.0, 40.0, 40.0),  // below
            rect(150.0, 150.0, 40.0, 40.0), // diagonal
        ] {
            let cell = bubble_text_cell(frame, bounds, &[neighbour])
                .expect("a clamped cut always leaves a cell");
            assert!(
                cell[0] <= bounds[0]
                    && cell[1] <= bounds[1]
                    && cell[2] >= bounds[2]
                    && cell[3] >= bounds[3],
                "cell {cell:?} does not cover its own text {bounds:?}"
            );
            assert!(
                cell[0] >= frame[0]
                    && cell[1] >= frame[1]
                    && cell[2] <= frame[2]
                    && cell[3] <= frame[3],
                "cell {cell:?} escaped the frame {frame:?}"
            );
        }
    }

    #[test]
    fn non_finite_coordinates_never_reach_a_geometry() {
        let frame = rect(0.0, 0.0, 100.0, 100.0);
        let bounds = rect(10.0, 10.0, 20.0, 20.0);
        assert_eq!(
            bubble_text_cell(frame, bounds, &[[f32::NAN, 0.0, 1.0, 1.0]]),
            Some(frame)
        );
        assert_eq!(
            bubble_text_cell([f32::INFINITY, 0.0, 1.0, 1.0], bounds, &[]),
            None
        );
        assert_eq!(bubble_text_cell(frame, [f32::NAN; 4], &[]), None);
    }

    /// The real page both defects were measured on.
    const PAGE: ImageSize = ImageSize {
        width: 977,
        height: 1400,
    };

    fn assert_close(actual: f32, expected: f32, what: &str) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "{what}: {actual} is not {expected}"
        );
    }

    fn assert_within_page(cell: [f32; 4]) {
        assert!(
            cell[0] >= 0.0
                && cell[1] >= 0.0
                && cell[2] <= PAGE.width as f32
                && cell[3] <= PAGE.height as f32,
            "cell {cell:?} escapes the {}x{} page",
            PAGE.width,
            PAGE.height
        );
    }

    fn assert_contains(cell: [f32; 4], bounds: [f32; 4]) {
        assert!(
            cell[0] <= bounds[0]
                && cell[1] <= bounds[1]
                && cell[2] >= bounds[2]
                && cell[3] >= bounds[3],
            "cell {cell:?} does not contain its own text {bounds:?}"
        );
    }

    #[test]
    fn a_vertical_free_text_column_is_widened_around_its_own_ink() {
        // Measured shape: free-standing vertical Japanese is a column about
        // 40px wide and 300px tall, which laid the English out at ~10px.
        let bounds = rect(300.0, 500.0, 40.0, 300.0);

        let (cell, cut) = free_text_cell(bounds, PAGE, &[]).unwrap();
        assert!(!cut, "no neighbours were given, so nothing can have cut");

        assert_close(
            cell[2] - cell[0],
            300.0 * FREE_TEXT_MIN_ASPECT,
            "grown width",
        );
        assert!(
            cell[2] - cell[0] > bounds[2] - bounds[0],
            "the box did not grow"
        );
        // Height is untouched, so the box still stands exactly where the
        // Japanese did rather than drifting up or down the page.
        assert_eq!(cell[1], bounds[1]);
        assert_eq!(cell[3], bounds[3]);
        assert_eq!(box_center(cell), box_center(bounds));
        assert_contains(cell, bounds);
        assert_within_page(cell);
    }

    #[test]
    fn a_short_narrow_free_text_reaches_the_absolute_floor() {
        /* The measured failure, off disk: a 21.7 x 36.1 box whose English needed
         * 34.4px and was lettered at the 9px floor instead. The aspect rule fires
         * and does NOTHING here -- 36.1 * 0.6 = 21.66, under the box's own 21.7 --
         * which is why an absolute floor was needed rather than a bigger ratio. */
        let bounds = rect(300.0, 500.0, 21.7, 36.1);
        assert!(
            36.1 * FREE_TEXT_MIN_ASPECT < 21.7,
            "the aspect rule must be inert on this shape, or the test proves nothing"
        );

        let (cell, _cut) = free_text_cell(bounds, PAGE, &[]).unwrap();
        assert_close(cell[2] - cell[0], FREE_TEXT_MIN_WIDTH, "grown width");
        assert!(
            cell[2] - cell[0] >= 34.4,
            "the box is still too narrow for the English that failed here: {}",
            cell[2] - cell[0]
        );
        // Height untouched and still centred on the ink, exactly as the aspect
        // rule leaves it -- the floor changes width and nothing else.
        assert_eq!(cell[1], bounds[1]);
        assert_eq!(cell[3], bounds[3]);
        assert_eq!(box_center(cell), box_center(bounds));
        assert_contains(cell, bounds);
        assert_within_page(cell);
    }

    #[test]
    fn a_tall_column_is_untouched_by_the_absolute_floor() {
        /* The tall-column hazard, pinned. A column is already given far more by the ratio
         * than the floor could, and the height gate must keep the floor out of it
         * entirely -- a height-blind floor would be the same defect in a new
         * costume. 300px is well above FREE_TEXT_SHORT_BOX_HEIGHT. */
        let bounds = rect(300.0, 500.0, 40.0, 300.0);
        assert!(300.0 > FREE_TEXT_SHORT_BOX_HEIGHT, "gate must exclude this box");

        let (cell, _cut) = free_text_cell(bounds, PAGE, &[]).unwrap();
        assert_close(
            cell[2] - cell[0],
            300.0 * FREE_TEXT_MIN_ASPECT,
            "a tall column must still be sized by the ASPECT rule alone",
        );
    }

    #[test]
    fn a_short_box_already_wide_enough_is_left_alone() {
        /* The floor is a floor, not a target. A short box that already clears it
         * must come back byte-identical, or every wide caption on the page moves
         * for nothing. */
        let bounds = rect(300.0, 500.0, 90.0, 30.0);
        assert!(90.0 > FREE_TEXT_MIN_WIDTH && 30.0 <= FREE_TEXT_SHORT_BOX_HEIGHT);

        let (cell, _cut) = free_text_cell(bounds, PAGE, &[]).unwrap();
        assert_close(cell[2] - cell[0], 90.0, "an already-wide short box must not move");
        assert_eq!(box_center(cell), box_center(bounds));
    }

    #[test]
    fn a_wide_free_text_is_left_exactly_as_it_is() {
        // The signboard: 119x129 is already wider than 129 * 0.6, so the max()
        // picks the text's own width and the box is its own bounds, op for op.
        let bounds = rect(430.0, 210.0, 119.0, 129.0);

        let (cell, cut) = free_text_cell(bounds, PAGE, &[]).unwrap();
        assert!(!cut, "no neighbours were given, so nothing can have cut");

        assert_eq!(cell[2] - cell[0], bounds[2] - bounds[0]);
        assert_eq!(cell, bounds);
    }

    /// An upright vertical column, which is the only shape the turn may claim.
    fn upright() -> Option<InferredTypography> {
        Some(InferredTypography {
            font_size: 40.0,
            color: [0, 0, 0],
            background: None,
            angle_degrees: 0.0,
            writing_mode: WritingMode::Vertical,
            ink_color: None,
            ink_stroke_ratio: None,
        })
    }

    /// The turned column's FOOTPRINT is the ink, to the float.
    ///
    /// This is the whole safety argument, so it is asserted rather than argued. A
    /// rotated cell is safe only where its hull is what the un-rotated ink already
    /// occupied -- then it collides with nothing new and the neighbour-cut overlap
    /// guarantee is untouched. The exemplar's own measured bounds.
    #[test]
    fn a_turned_column_occupies_exactly_its_own_ink() {
        let column = [62.6953125, 10.0546875, 267.1875, 1702.59375];
        let geometry = free_text_column_geometry(column, upright()).expect("claimed");
        let (min_x, min_y, max_x, max_y) =
            crate::scope::geometry_extents(&geometry).expect("has extents");
        for (got, want) in [
            (min_x, column[0]),
            (min_y, column[1]),
            (max_x, column[2]),
            (max_y, column[3]),
        ] {
            assert!(
                (got - f64::from(want)).abs() < 1e-3,
                "hull {got} is not the ink's {want}"
            );
        }
    }

    /// A NON-ZERO angle is refused, and the neighbour-cut overlap guarantee depends on it.
    ///
    /// Composing 90 with an inferred angle makes the hull `h*|sin| + w*|cos|` --
    /// 1.43x the ink at 3 degrees, 2.42x at 10, and wider than the widened box it
    /// replaces beyond about 14. There is no safe composition, so the only correct
    /// answer is to decline the region and let it widen as before.
    /// The composed gate the caller actually passes, on all eight inputs.
    ///
    /// Without this the `|| turn_unjoined` arm can be deleted and every
    /// other test in this file stays green — the turn tests below exercise
    /// `free_text_column_geometry`, which never sees the page.
    #[test]
    fn the_turn_is_offered_exactly_when_the_master_switch_and_one_scope_agree() {
        // Master switch off: nothing turns, whatever the page is. This is what
        // keeps `--rotate-free-text-columns false` a true restore.
        assert!(!turn_is_offered(false, false, false));
        assert!(!turn_is_offered(false, true, false));
        assert!(!turn_is_offered(false, false, true));
        assert!(!turn_is_offered(false, true, true));
        // The shipped arm: joined pages turn, unjoined ones do not.
        assert!(turn_is_offered(true, true, false));
        assert!(
            !turn_is_offered(true, false, false),
            "an unjoined page must keep the widened cell until the flag says otherwise"
        );
        // The new scope, and it must not need a joined page to take effect.
        assert!(turn_is_offered(true, false, true));
        assert!(turn_is_offered(true, true, true));
    }

    #[test]
    fn a_tilted_column_is_not_turned_at_all() {
        let column = [62.6953125, 10.0546875, 267.1875, 1702.59375];
        for angle in [3.0_f32, -3.0, 10.0, 45.0, f32::NAN] {
            let inferred = Some(InferredTypography {
                angle_degrees: angle,
                ..upright().expect("fixture")
            });
            assert!(
                free_text_column_geometry(column, inferred).is_none(),
                "a column at {angle} degrees must fall through to the widened cell"
            );
        }
    }

    /// The two populations it must never claim, for opposite reasons.
    #[test]
    fn a_wide_or_hairline_region_falls_through_to_the_widened_cell() {
        // Horizontal: `free_text_cell`'s max() already leaves it at its own
        // width, so turning it would be a change with no defect behind it.
        assert!(
            free_text_column_geometry(rect(430.0, 210.0, 119.0, 129.0), upright()).is_none(),
            "a signboard is not a column"
        );
        // Too narrow to hold one 9px line on its side. Two regions in one
        // manga baseline are exactly 1.0px wide.
        assert!(
            free_text_column_geometry(rect(100.0, 100.0, 1.0, 300.0), upright()).is_none(),
            "a 1px detection must not be turned into a box that cannot hold a line"
        );
        // Just over the bar is still claimed, so the guard is a floor and not a
        // blanket refusal.
        assert!(
            free_text_column_geometry(rect(100.0, 100.0, 22.0, 300.0), upright()).is_some(),
            "22px clears the 21.6 floor and must still be turned"
        );
    }

    /// The exemplar's own measured numbers.
    ///
    /// A seam-joined manhua skill-name column and a site watermark, exactly as a
    /// recorded run measured them. The watermark cuts the
    /// column's right edge at the midpoint of the two centres, 598.095703125, and
    /// the page clip then takes the left overhang off at 0.0 -- leaving a cell
    /// whose centre is 299.05 while its ink's is 164.94, so `placement` in
    /// `text_renderer.rs` letters the English 134.10px to the RIGHT of the column,
    /// over artwork the eraser never touched.
    ///
    /// **This asserts the CENTRE, not the width, because the centre is the
    /// defect.** A test that pinned the width would pass on a box of the right
    /// size in the wrong place, which is precisely what shipped.
    #[test]
    fn a_clipped_cell_is_re_centred_on_the_ink_it_replaces() {
        let page = ImageSize {
            width: 1200,
            height: 1716,
        };
        let column = [62.6953125, 10.0546875, 267.1875, 1702.59375];
        let watermark = [890.625, 28.69775390625, 1171.875, 116.466796875];

        let (cell, cut) = free_text_cell(column, page, &[watermark]).unwrap();
        assert!(cut, "the watermark is to the right and must cut this cell");

        let ink_centre = (column[0] + column[2]) * 0.5;
        let cell_centre = (cell[0] + cell[2]) * 0.5;
        assert!(
            (cell_centre - ink_centre).abs() < 1e-3,
            "the English is lettered at the cell's centre, so a cell centred at \
             {cell_centre} letters it {} px from ink centred at {ink_centre}: {cell:?}",
            (cell_centre - ink_centre).abs()
        );

        // The growth the clip-last ordering exists to protect must SURVIVE the
        // re-centring, or this fix has simply reintroduced clip-first. 1.61x here.
        assert!(
            cell[2] - cell[0] > column[2] - column[0],
            "re-centring cost the box all of its growth: {cell:?}"
        );
        assert_contains(cell, column);
        assert!(
            cell[0] >= 0.0 && cell[2] <= page.width as f32,
            "cell {cell:?} escapes the page"
        );
    }

    #[test]
    fn a_widened_column_is_clipped_to_the_page() {
        for bounds in [
            rect(10.0, 500.0, 40.0, 300.0),
            rect(930.0, 500.0, 40.0, 300.0),
        ] {
            let (cell, cut) = free_text_cell(bounds, PAGE, &[]).unwrap();
        assert!(!cut, "no neighbours were given, so nothing can have cut");

            assert_within_page(cell);
            assert_contains(cell, bounds);
            assert!(
                cell[2] - cell[0] > bounds[2] - bounds[0],
                "clipping cost the box all of its growth: {cell:?}"
            );
        }
        // A column that starts outside the page cannot be grown into one that
        // still covers its own ink, so the caller keeps today's behaviour.
        assert_eq!(
            free_text_cell(rect(940.0, 500.0, 40.0, 300.0), PAGE, &[]),
            None
        );
    }

    /// The angle survives to the layer, but only where moving the box is free.
    ///
    /// `geometry_frame` in the renderer reads the top edge, so an axis-aligned
    /// rectangle reports exactly 0 degrees however the glyphs actually run. That
    /// is the loss this covers. The restriction to an uncut cell is the safety
    /// argument, and it is asserted here rather than described: a cut cell keeps
    /// its axis-aligned box whatever the angle says.
    #[test]
    fn only_an_uncut_cell_carries_the_glyph_angle_to_the_layer() {
        let cell = rect(300.0, 500.0, 180.0, 300.0);
        let tilted = |angle: f32| {
            let mut inferred = inferred([0, 0, 0], Some([255, 255, 255]));
            inferred.angle_degrees = angle;
            Some(inferred)
        };

        // Uncut and genuinely tilted: the frame's top edge is no longer flat, so
        // the renderer can finally see the angle.
        let rotated = free_text_geometry(cell, false, tilted(18.0));
        let flat = free_text_geometry(cell, false, tilted(0.0));
        assert_ne!(
            geometry_bounds_debug(&rotated),
            geometry_bounds_debug(&flat),
            "an 18 degree angle produced the same geometry as no angle at all"
        );

        // Cut: the neighbour owns these edges. Identical to the flat case.
        assert_eq!(
            geometry_bounds_debug(&free_text_geometry(cell, true, tilted(18.0))),
            geometry_bounds_debug(&flat),
            "a cut cell was rotated, which is what puts two texts back on top of \
             each other"
        );

        // No inference at all, and a non-finite angle: both take the old path.
        assert_eq!(
            geometry_bounds_debug(&free_text_geometry(cell, false, None)),
            geometry_bounds_debug(&flat)
        );
        assert_eq!(
            geometry_bounds_debug(&free_text_geometry(cell, false, tilted(f32::NAN))),
            geometry_bounds_debug(&flat)
        );
    }

    /// Compares two geometries by their point list, since `Geometry` is not
    /// `PartialEq` and the bounds hull of a rotated box is not enough to tell a
    /// rotation from a resize.
    fn geometry_bounds_debug(geometry: &Geometry) -> String {
        format!("{geometry:?}")
    }

    #[test]
    fn two_vertical_free_texts_side_by_side_do_not_overlap() {
        let left = rect(300.0, 500.0, 40.0, 300.0);
        let right = rect(400.0, 500.0, 40.0, 300.0);

        let (left_cell, left_cut) = free_text_cell(left, PAGE, &[right]).unwrap();
        let (right_cell, right_cut) = free_text_cell(right, PAGE, &[left]).unwrap();

        /* Both boxes were cut by the other, and this is exactly the pair whose
         * edges may not move: the assertion below is that they do not overlap,
         * and rotating either cell about its centre would put them back on top
         * of each other. `free_text_geometry` reads this flag for that reason. */
        assert!(left_cut && right_cut, "the neighbour cut must be reported");

        assert_eq!(
            intersection_area(left_cell, right_cell),
            0.0,
            "{left_cell:?} overlaps {right_cell:?}"
        );
        for (cell, bounds) in [(left_cell, left), (right_cell, right)] {
            assert_contains(cell, bounds);
            assert_within_page(cell);
            assert_eq!(box_center(cell), box_center(bounds));
            assert!(
                cell[2] - cell[0] > bounds[2] - bounds[0],
                "the box gave up all its growth to its neighbour: {cell:?}"
            );
        }
    }

    fn inferred(color: [u8; 3], background: Option<[u8; 3]>) -> InferredTypography {
        InferredTypography {
            font_size: 24.0,
            color,
            background,
            angle_degrees: 0.0,
            writing_mode: WritingMode::Vertical,
            ink_color: None,
            ink_stroke_ratio: None,
        }
    }

    /// The `inferred` fixture with an ink sample present, for the override's
    /// own tests -- the legacy-shaped fixture above deliberately has none, so
    /// every pre-existing assertion keeps describing the fallback path.
    fn inked(ratio: Option<f32>) -> InferredTypography {
        InferredTypography {
            ink_color: Some([94, 45, 45]),
            ink_stroke_ratio: ratio,
            ..inferred([255, 255, 255], Some([150, 60, 60]))
        }
    }

    #[test]
    fn only_free_standing_text_gets_a_halo() {
        let over_artwork = inferred([0, 0, 0], Some([214, 198, 170]));

        let (color, width) = text_stroke(Some(over_artwork), false).unwrap();
        assert_eq!(color, [214, 198, 170]);
        assert_close(width, 24.0 * FREE_TEXT_STROKE_RATIO, "stroke width");

        // In a bubble the glyphs sit on a flat inpainted fill, so a halo would
        // only fatten every line on pages that render correctly today.
        assert_eq!(text_stroke(Some(over_artwork), true), None);
    }

    /// An RGB gate used to refuse any halo when fill and plate matched. With the fill
    /// genuinely sampled from drawn ink, that pairing is the COMMON case for
    /// effects over dark art, and "no halo" letters invisible text -- so the
    /// gate is gone and the matched pair falls through to the tone flip, which
    /// its low RGB distance guarantees it takes (every channel gap under ~83,
    /// tone gap at most 83, inside `FREE_TEXT_STROKE_TONE_GAP`).
    #[test]
    fn a_background_the_glyphs_already_match_flips_the_halo_instead_of_dropping_it() {
        let (color, width) = text_stroke(Some(inferred([0, 0, 0], Some([12, 9, 14]))), false)
            .expect("the matched pair now takes the flip rather than None");
        assert_eq!(color, [255, 255, 255], "dark ink on its own plate takes a white halo");
        assert_close(width, 24.0 * FREE_TEXT_STROKE_RATIO, "the flip must not touch the width");

        // No background measured, or nothing inferred at all: still no halo.
        assert_eq!(text_stroke(Some(inferred([0, 0, 0], None)), false), None);
        assert_eq!(text_stroke(None, false), None);
    }

    /// The composed override `link_dialogue_regions` actually calls -- the
    /// flag, the in-bubble refusal, the abstention, and the weight mapping are
    /// one function on purpose, so a test cannot green two halves around a
    /// missing `&&` (the unwired-fix lesson).
    #[test]
    fn only_free_standing_text_takes_the_sampled_ink() {
        let sample = sampled_ink(Some(inked(Some(0.149))), false, true)
            .expect("free-standing text with a core sample takes the override");
        assert_eq!(sample.color, [94, 45, 45, 255]);
        assert_eq!(
            sample.font_weight,
            Some(HEAVY_INK_WEIGHT),
            "the measured 0.149 stroke ratio is the heavy exemplar"
        );

        // The flag's off arm is the byte-exact control: no override, whatever
        // the sample says.
        assert!(sampled_ink(Some(inked(Some(0.149))), false, false).is_none());

        // In a bubble the flat fill IS the background, the contrast sample
        // reads it correctly, and the standing constraint is that balloon
        // dialogue changes nowhere.
        assert!(sampled_ink(Some(inked(Some(0.149))), true, true).is_none());

        // Thin drawn ink letters at the default weight -- the measured thin squiggle.
        let thin = sampled_ink(Some(inked(Some(0.05))), false, true).expect("still overridden");
        assert_eq!(thin.color, [94, 45, 45, 255]);
        assert_eq!(thin.font_weight, None);
        assert_eq!(
            sampled_ink(Some(inked(None)), false, true).expect("no ratio").font_weight,
            None
        );

        // No core sample: the caller keeps the legacy fill untouched.
        assert!(sampled_ink(Some(inferred([0, 0, 0], Some([250, 250, 250]))), false, true).is_none());
        assert!(sampled_ink(None, false, true).is_none());
    }

    /// The measured numbers: black glyphs over a plate whose
    /// median is (95,61,245) -- 65k of RGB distance (the gate passes), but luma
    /// 92 against fill luma 0, one TONE. The halo must flip to white or the
    /// glyph edge dissolves wherever the plate's darker band meets the ink.
    /// Both flip directions and the keep side are pinned; the width rule is
    /// untouched by the flip.
    #[test]
    fn a_plate_of_the_inks_own_tone_flips_the_halo_to_the_opposite_tone() {
        // Dark ink, dark-toned plate: flip to white.
        let (color, width) = text_stroke(Some(inferred([0, 0, 0], Some([95, 61, 245]))), false)
            .expect("the RGB gate passes this pair; only the colour changes");
        assert_eq!(
            color,
            [255, 255, 255],
            "a dark-toned halo under dark ink defines no edge on a dark band"
        );
        assert_close(width, 24.0 * FREE_TEXT_STROKE_RATIO, "flip must not touch the width");

        // Light ink, light-toned plate: flip to black. (A status-window
        // shape: light fill over a bright green plate, tone gap ~25.)
        let (color, _) = text_stroke(Some(inferred([230, 230, 230], Some([55, 172, 143]))), false)
            .expect("the RGB gate passes this pair too");
        assert_eq!(color, [0, 0, 0], "light ink on a light-toned plate takes a black halo");

        // The keep side, one step past the threshold: a bright plate under dark
        // ink keeps the background-median halo exactly as it ships today.
        let (color, _) = text_stroke(Some(inferred([0, 0, 0], Some([254, 254, 254]))), false)
            .expect("the ordinary white-page case");
        assert_eq!(
            color,
            [254, 254, 254],
            "tone gap 254 is far above the line: the median halo is untouched"
        );
    }

    #[test]
    fn a_small_glyph_still_gets_a_visible_edge() {
        let mut small = inferred([255, 255, 255], Some([20, 20, 20]));
        small.font_size = 6.0;

        let (_, width) = text_stroke(Some(small), false).unwrap();

        assert_eq!(width, FREE_TEXT_STROKE_MINIMUM_WIDTH);
    }

    #[test]
    fn the_halo_never_closes_a_lowercase_counter() {
        /* The renderer draws `Stroke::new(width_px * 2.0)` -- centred, so the
         * halo reaches `width_px` outward from every contour, inner ones
         * included, and `Fill::NonZero` does not reopen a counter it has closed.
         * Arial's `e` eye is roughly 0.12 em, so it shuts once the halo reaches
         * half of that from both sides. Asserting the bound rather than
         * restating the multiplication is the point: the previous value passed a
         * test that only checked `size * RATIO` while rendering every `a`, `e`
         * and `g` as a blob. */
        const EYE_EM: f32 = 0.12;
        assert!(
            FREE_TEXT_STROKE_RATIO < EYE_EM * 0.5,
            "halo {FREE_TEXT_STROKE_RATIO} closes a {EYE_EM} em counter"
        );
        // And half the bound again, so the counter stays visibly open.
        assert!(FREE_TEXT_STROKE_RATIO <= EYE_EM * 0.25);

        for size in [9.0_f32, 24.0, 68.0, 140.0] {
            let mut glyph = inferred([0, 0, 0], Some([255, 255, 255]));
            glyph.font_size = size;
            let (_, width) = text_stroke(Some(glyph), false).unwrap();
            assert!(
                width < size * EYE_EM * 0.5 || width == FREE_TEXT_STROKE_MINIMUM_WIDTH,
                "{size}px glyph got a {width}px halo"
            );
        }
    }

    /// The exemplar is a measured test page: `bubble 0.5078` at
    /// `[114.84, 457.55, 1190.62, 897.36]` and NOT ONE text detection on the page.
    ///
    /// **Asserted on the COMPOSED pass, never on the synthesis alone**, because a
    /// test that reaches past the composition cannot see the wiring -- a fix once
    /// shipped completely unwired with 128 tests green for exactly that
    /// reason, and `settle_with`/`settle_reading_bubbles` exist to make the two
    /// arms reachable without bypassing it.
    #[test]
    fn a_bubble_holding_no_text_is_read_as_text_when_the_flag_is_on() {
        let mut page = RgbImage::from_pixel(1200, 908, Rgb([255, 255, 255]));
        let bubble = inked_bubble(&mut page, 0.5078125, [114.84, 457.55, 1190.62, 897.36], true);

        let (off, _) = settle_with(vec![bubble.clone()], &page, false, true);
        assert!(
            off.iter().all(|d| d.label != TEXT),
            "the OFF arm invented a text region: {:?}",
            off.iter().map(|d| &d.label).collect::<Vec<_>>()
        );

        let (on, _) = settle_reading_bubbles(vec![bubble.clone()], &page, false);
        let texts: Vec<&KoharuLayoutDetection> =
            on.iter().filter(|d| d.label == TEXT).collect();
        assert_eq!(texts.len(), 1, "expected exactly one synthesised text region");
        // The BOX is the bubble's own, unshrunk -- it is what OCR is handed and what
        // the size gate measures.
        assert_eq!(texts[0].bbox, bubble.bbox);
        assert_eq!(texts[0].score, bubble.score);
        // But the MASK is the ink, not the body, which is why the
        // balloon survives the eraser. Asserted as a strict shrink rather than an
        // exact count: the count is a property of the fixture's glyph bars, the
        // shrink is the property of the code.
        let body = bubble.mask.pixels.iter().filter(|v| **v != 0).count();
        let ink = texts[0].mask.pixels.iter().filter(|v| **v != 0).count();
        assert!(
            ink > 0 && ink < body / 4,
            "the synthesised mask is still the balloon BODY ({ink} of {body} px), so the \
             eraser will take the balloon with the text"
        );
        // And the bubble itself survives -- it is not consumed.
        assert!(on.iter().any(|d| d.label == "bubble"));
    }

    /// **The other half of the ink-mask rule, and the first gate this rule has on
    /// its own false positive.** The known false positive is a spurious `bubble` over
    /// flat skin and a gold collar. It holds no paper and no glyphs, so there is nothing
    /// for `ink_within` to find and no region is invented for it.
    ///
    /// Proved red for its own reason by the test above, which is the same fixture with
    /// `inked: true` and expects exactly one region. The pair is the assertion: the
    /// discriminator is the ink and nothing else about the two differs.
    #[test]
    fn a_bubble_over_flat_artwork_is_given_no_region_at_all() {
        let mut page = RgbImage::from_pixel(1200, 908, Rgb([255, 255, 255]));
        let bubble = inked_bubble(&mut page, 0.5078125, [114.84, 457.55, 1190.62, 897.36], false);

        let (on, _) = settle_reading_bubbles(vec![bubble.clone()], &page, false);
        assert!(
            on.iter().all(|d| d.label != TEXT),
            "a balloon with no ink in it was given a region anyway -- this is the shape of \
             the known false positive, lettering junk onto artwork"
        );
        // And the bubble is still a bubble; nothing else about it changed.
        assert!(on.iter().any(|d| d.label == "bubble"));
    }

    /// The other half, and the one that decides whether the rule is safe: a bubble
    /// that DOES hold text must be left alone, or every balloon in the corpus grows
    /// a second region stacked on its own dialogue.
    #[test]
    fn a_bubble_that_already_holds_text_gains_nothing() {
        let page = RgbImage::from_pixel(1200, 908, Rgb([255, 255, 255]));
        // A real measured pairing, from the same run as the textless exemplar.
        let detections = vec![
            detection("bubble", 0.8203125, [628.12, 22.37, 1106.25, 618.09]),
            detection(TEXT, 0.9179688, [703.12, 109.79, 1021.88, 540.83]),
        ];
        let (on, _) = settle_reading_bubbles(detections, &page, false);
        assert_eq!(
            on.iter().filter(|d| d.label == TEXT).count(),
            1,
            "a bubble that already holds text was given a second region"
        );
    }

    /// A filled balloon, turned, at the size and angle the measured test page has it.
    ///
    /// `local_width` x `local_height` are the balloon's extents in **its own
    /// frame**; the returned `bbox` is the mask's axis-aligned hull, which is what
    /// the detector reports and what the synthesis hands straight over.
    ///
    /// **The MASK is the filled body and carries no glyphs** -- that is what a
    /// `bubble` detection's mask is, and on the exemplar it is all the detector returns.
    /// **The IMAGE underneath it DOES carry glyphs**, as dark bars along the
    /// balloon's own axis, and that is load-bearing rather than
    /// decorative: `ink_within` reads the image through the mask to find the ink, and
    /// a balloon with none is no longer given a region at all.
    ///
    /// The two together are the real case exactly. The detector finds the shape and
    /// misses the text; the text is nonetheless *there*,
    /// printed on the paper, which is why reading pixels recovers what the network
    /// could not.
    fn tilted_balloon(
        page: (u32, u32),
        local_width: f64,
        local_height: f64,
        angle_degrees: f64,
    ) -> (RgbImage, KoharuLayoutDetection) {
        let (width, height) = page;
        let center_x = f64::from(width) * 0.5;
        let center_y = f64::from(height) * 0.5;
        let (sin, cos) = angle_degrees.to_radians().sin_cos();
        let mut image = RgbImage::from_pixel(width, height, Rgb([40, 40, 40]));
        let mut pixels = vec![0u8; width as usize * height as usize];
        let (mut left, mut top) = (f32::INFINITY, f32::INFINITY);
        let (mut right, mut bottom) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for y in 0..height {
            for x in 0..width {
                let dx = f64::from(x) + 0.5 - center_x;
                let dy = f64::from(y) + 0.5 - center_y;
                let local_x = dx * cos + dy * sin;
                let local_y = -dx * sin + dy * cos;
                if local_x.abs() <= local_width * 0.5 && local_y.abs() <= local_height * 0.5 {
                    pixels[y as usize * width as usize + x as usize] = u8::MAX;
                    /* Paper, and then GLYPHS on it: four dark bars along the
                     * balloon's own axis, inset from its edges so the outline stays
                     * paper. The mask does not know about them -- a bubble mask never
                     * does -- but `ink_within` finds them through it, which is the
                     * whole mechanism this fixture now has to exercise. Eight percent
                     * of the balloon's own width per bar is a plausible glyph density
                     * and lands far above `SYNTHESISED_INK_MIN_PIXELS`. */
                    let inked = local_y.abs() <= local_height * 0.35
                        && (0..4).any(|glyph| {
                            let center = local_width * (f64::from(glyph) * 0.2 - 0.3);
                            (local_x - center).abs() <= local_width * 0.04
                        });
                    image.put_pixel(x, y, if inked { Rgb([20, 20, 20]) } else { Rgb([250, 250, 250]) });
                    left = left.min(x as f32);
                    top = top.min(y as f32);
                    right = right.max(x as f32 + 1.0);
                    bottom = bottom.max(y as f32 + 1.0);
                }
            }
        }
        (
            image,
            KoharuLayoutDetection {
                label_id: 0,
                label: "bubble".to_owned(),
                score: 0.5078125,
                bbox: [left, top, right, bottom],
                area: pixels.iter().filter(|value| **value != 0).count() as u32,
                mask: KoharuLayoutMask {
                    width,
                    height,
                    pixels,
                },
            },
        )
    }

    /// **The assertion is on the hull THE READER MEASURES.**
    ///
    /// The two tests above pin the synthesis; neither can see this defect, because
    /// both drive it with the 1x1 dummy mask `detection` builds -- and with no ink
    /// to measure, `infer_typography` returns `None`, no rotation is ever applied
    /// and the inflation this test exists for cannot occur. **They were green
    /// throughout the render that proved a plain turned bbox does not reach the page**,
    /// which is exactly the hole `stages/ocr.rs` records: a
    /// test that stops short of what the caller measures cannot fail for the
    /// reason the caller fails.
    ///
    /// So this drives the whole path `ocr.rs`'s `region_extent` sees -- settle,
    /// then `build_region`'s `letters` branch op for op -- and measures the axis-
    /// aligned hull of the resulting geometry against the 0.5 ceiling.
    ///
    /// **It is proved red for its own reason by the first assertion**, which is
    /// the plain-bbox behaviour built from the same fixture: the bubble's own bbox,
    /// handed over and turned a second time, measures 0.61 of the page and is
    /// refused. Point the synthesised arm back at a rectangle and the third
    /// assertion fails; return something larger than the mask and the second does.
    ///
    /// # How the arm changed, and why the assertions survived it
    ///
    /// The synthesised arm was once `oriented_ink_box` -- a rectangle turned
    /// from the balloon's own axes -- and this test measured its **drift**, because
    /// the angle had to be measured once and used twice or the hull came back 102 px
    /// short of the glyphs. The arm no longer turns a rectangle at all (see
    /// `region_geometry`), so there is no angle to drift and no corners to inflate,
    /// and the third assertion is stronger than the 0.01 tolerance it is written to
    /// -- the hull is the mask's AABB, and the fixture builds `raw` from that same
    /// AABB.
    #[test]
    fn the_synthesised_region_survives_being_turned_again() {
        let page = (1200u32, 908u32);
        // The exemplar balloon, to the pixel: ~1018 x 176 in its own frame at
        // 12.5 degrees, whose hull is the 1032 x 392 = 0.3713 measured from
        // the white body alone.
        let (image, bubble) = tilted_balloon(page, 1018.0, 176.0, 12.5);
        let page_area = f64::from(page.0) * f64::from(page.1);
        let raw = bubble.bbox;

        let (on, _) = settle_reading_bubbles(vec![bubble.clone()], &image, false);
        let texts: Vec<&KoharuLayoutDetection> = on.iter().filter(|d| d.label == TEXT).collect();
        assert_eq!(texts.len(), 1, "expected exactly one synthesised text region");
        let synthesised = texts[0];

        // **`region_geometry` is the function `write_region` CALLS**, with
        // `synthesised` the only difference between the two arms below -- so a
        // difference in the result can only be that branch. A closure that
        // re-implemented the branch op for op would stay green through every
        // change to the real one; calling the real function is what makes the
        // assertion mean anything. The angle comes back with the share because the angle is
        // what a failure here is ABOUT: a wrong hull is wrong by exactly the degrees
        // a re-measurement drifted, and reporting only the area would make the next
        // reader re-derive that.
        let hull_share = |detection: &KoharuLayoutDetection, synthesised: bool| -> (f64, f32) {
            let typography = infer_typography(&image, detection)
                .expect("a filled balloon has ink for the typography pass to measure");
            let geometry = region_geometry(detection, true, synthesised, Some(typography));
            let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
            let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for point in &geometry.points {
                min_x = min_x.min(point.x);
                max_x = max_x.max(point.x);
                min_y = min_y.min(point.y);
                max_y = max_y.max(point.y);
            }
            (
                (max_x - min_x) * (max_y - min_y) / page_area,
                typography.angle_degrees,
            )
        };

        // The bubble's own bbox reaches `build_region` untouched, which is what
        // the eraser measures.
        assert_eq!(synthesised.bbox, raw);

        let (before, angle) = hull_share(synthesised, false);
        assert!(
            before > 0.5,
            "the fixture no longer reproduces the defect: the turned bbox measures {before} of \
             the page at {angle} degrees, so the assertions below would pass with the shrink \
             deleted"
        );

        let (after, _) = hull_share(synthesised, true);
        assert!(
            after < 0.5,
            "the synthesised region is still refused before any engine reads it: {after} of the \
             page against a 0.5 ceiling"
        );

        // Under the ceiling is necessary and not sufficient -- a box shrunk to a
        // speck would clear it too, and would letter nothing. The hull has to land
        // back on the balloon's own ink, because the hull is also the OCR CROP:
        // short of the ink is glyphs cut off before any engine sees them. This is
        // the assertion that failed at 0.2746 when the angle was re-measured on a
        // shrunken window instead of being passed in -- 6.5 degrees against 12.5,
        // for a hull 102px short of the balloon.
        let ink = f64::from((raw[2] - raw[0]) * (raw[3] - raw[1])) / page_area;
        assert!(
            (after - ink).abs() < 0.01,
            "the hull should reconstruct the balloon's ink ({ink}), not merely clear the \
             ceiling ({after}), measured at {angle} degrees"
        );
    }

    /// **`oriented_ink_box` is off the shipping path and held for an open
    /// defect.** This pins its recorded measurement, so an unused tool cannot rot
    /// into a wrong one before the item that needs it is picked up.
    ///
    /// It deliberately asserts on the ORIENTED box, which is the thing that is
    /// correct about it, and not on the hull -- the hull measured 0.5883 and is
    /// exactly why this is no longer called.
    #[test]
    fn the_oriented_box_still_measures_the_recorded_extent() {
        let page = (1200u32, 908u32);
        let (image, bubble) = tilted_balloon(page, 1018.0, 176.0, 12.5);
        let typography = infer_typography(&image, &bubble)
            .expect("a filled balloon has ink for the typography pass to measure");
        let turned = oriented_ink_box(&image, &bubble, typography.angle_degrees)
            .expect("a tilted balloon has an oriented box smaller than its own bbox");

        let width = f64::from(turned[2] - turned[0]);
        let height = f64::from(turned[3] - turned[1]);
        let share = width * height / (f64::from(page.0) * f64::from(page.1));
        // The oriented box clears the ceiling comfortably; it is its HULL, once
        // `rotated_rectangle_geometry` turns it back, that does not. Both halves of
        // that sentence are why the synthesised arm no longer turns a box.
        assert!(
            share < 0.5,
            "the oriented box no longer clears the OCR ceiling: {share} of the page \
             ({width} x {height}) at {} degrees",
            typography.angle_degrees
        );
        assert!(
            width > height,
            "the oriented box lost the balloon's long axis: {width} x {height}"
        );
    }

    /// **One object, one shape.** The invariant, asserted as an EQUALITY rather
    /// than as an area, because an area
    /// is exactly what three previous attempts satisfied while still handing OCR a
    /// different shape from the one the eraser used.
    ///
    /// `settle_detections` clones the bubble's mask into the synthesised detection
    /// *"so the eraser and the rule cannot disagree about the shape"*. That was true
    /// of the MASK and false of the GEOMETRY: the bubble took `mask_geometry` and the
    /// synthesised text took a rotated rectangle built from the same mask, so the two
    /// regions describing one balloon disagreed by 0.5883 against 0.4342 of the page.
    /// This pins them equal.
    ///
    /// **It is proved red for its own reason**, and the first attempt at it was not.
    /// That attempt asserted the outline had MORE THAN FOUR points, and it failed
    /// against the shipping code: `tilted_balloon` builds a *perfect* rotated
    /// rectangle, so `approximate_polygon_dp` correctly reduces its contour to
    /// exactly four. **Four points is the right answer for this fixture** -- do not
    /// "fix" that assertion back in. A real balloon's outline has many more, but
    /// pinning that here would be pinning the fixture and not the code.
    ///
    /// What separates the two paths is therefore the COORDINATES, not the count, and
    /// the last assertion says so directly: `rotated_rectangle_geometry` turns the
    /// AXIS-ALIGNED bbox, so it lands on four corners the mask never had -- which is
    /// the 0.6957 and 0.5883 inflations measured on earlier versions. Point the
    /// synthesised arm back at either and that assertion fails.
    /// **`SYNTHESISED_REGION_KIND` must PARSE, or `write_region` fails at runtime.**
    ///
    /// It reaches `RegionKind::new(..)?` on every synthesised region, and
    /// `validate_namespaced` is the only thing standing between a typo and a `?`
    /// that aborts the detection stage on the exact pages `--read-textless-bubbles`
    /// exists for. A compile cannot catch it: the constant is a `&str`.
    ///
    /// The second assertion is the load-bearing one. The marker must NOT collide
    /// with a real region kind, because `is_synthesised` matches on any label and a
    /// collision would mark ordinary detected regions as synthesised -- which would
    /// stop lettering real balloons.
    #[test]
    fn the_synthesised_marker_parses_and_is_not_a_real_region_kind() {
        let marker = koharu_scene::RegionKind::new(super::SYNTHESISED_REGION_KIND)
            .expect("the synthesised marker must parse as a RegionKind");
        assert_eq!(marker.as_str(), super::SYNTHESISED_REGION_KIND);
        assert_ne!(marker.as_str(), TextRegion::KIND);
        assert_ne!(marker.as_str(), koharu_scene::BubbleRegion::KIND);
        // Ours, not upstream's: a kind in Koharu's namespace would claim an
        // upstream that does not define it.
        assert!(super::SYNTHESISED_REGION_KIND.starts_with("dev.birelate."));
    }

    /// **The measured gap the ink-fraction floor lives in, pinned so it cannot be
    /// tuned away without a failure.**
    ///
    /// A census of 2,519 bubbles over 773 pages found twelve that hold no detected
    /// text -- the only population this floor gates -- and all twelve were inspected
    /// in pixels. Eleven are genuine balloons; one is bare skin. Their ink
    /// fractions are `0.0213` for the false positive and `0.0495` for the lowest
    /// genuine one, with nothing in between.
    ///
    /// This does not exercise `ink_within`; it asserts the CONSTANT sits inside the
    /// gap the corpus measured. A test that re-derived the fractions from a fixture
    /// would be pinning the fixture, which this file already has a warning about
    /// four tests below. Move the constant outside these bounds and this fails,
    /// which is the point: the number is evidence-backed, not chosen.
    #[test]
    fn the_ink_fraction_floor_sits_in_the_measured_gap() {
        const FALSE_POSITIVE: f64 = 0.0213; // the known false positive, bare skin
        const LOWEST_TRUE: f64 = 0.0495; // the lowest genuine text-less balloon
        assert!(
            SYNTHESISED_INK_MIN_FRACTION > FALSE_POSITIVE,
            "the floor must REJECT bare skin: {SYNTHESISED_INK_MIN_FRACTION} <= {FALSE_POSITIVE}"
        );
        assert!(
            SYNTHESISED_INK_MIN_FRACTION < LOWEST_TRUE,
            "the floor must KEEP every genuine text-less balloon measured: \
             {SYNTHESISED_INK_MIN_FRACTION} >= {LOWEST_TRUE}"
        );
        // The absolute floor is a degenerate-mask guard, not the discriminator --
        // the false positive clears it with 795 pixels. Pinned so nobody restores
        // it to that role.
        assert!(
            795 >= SYNTHESISED_INK_MIN_PIXELS,
            "the false positive clears the absolute floor; only the fraction rejects it"
        );
    }

    #[test]
    fn a_synthesised_region_is_boxed_by_its_balloon_and_never_turned() {
        let page = (1200u32, 908u32);
        // The exemplar balloon, as `the_synthesised_region_survives_being_turned_again`
        // builds it -- the same fixture, so a divergence between the two tests is a
        // divergence in the code and not in the setup.
        let (image, bubble) = tilted_balloon(page, 1018.0, 176.0, 12.5);
        let page_area = f64::from(page.0) * f64::from(page.1);

        let (on, _) = settle_reading_bubbles(vec![bubble.clone()], &image, false);
        let synthesised = on
            .iter()
            .find(|d| d.label == TEXT)
            .expect("the flag should synthesise a text region for a lonely bubble");

        // Exactly as `write_region` calls it.
        let typography = infer_typography(&image, synthesised)
            .expect("an inked balloon has glyphs for the typography pass to measure");
        let shape = region_geometry(synthesised, true, true, Some(typography));

        // 1. It is the balloon's own box, unturned -- four points on the bbox.
        assert_eq!(shape.points, rectangle_geometry(bubble.bbox).points);

        // 2. Its hull clears the OCR ceiling, which is the whole reason for this
        //    arm. 0.6957 (bbox turned) -> 0.5883 (oriented box turned) -> 0.4342,
        //    the measured FLOOR.
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for point in &shape.points {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }
        let share = (max_x - min_x) * (max_y - min_y) / page_area;
        assert!(
            share < 0.5,
            "the synthesised region is refused before any engine reads it: {share} of the \
             page against a 0.5 ceiling"
        );

        // 3. And it is NOT turned. This is the assertion that reddens if the arm is
        //    pointed back at either turned-rectangle geometry, both of which turn an
        //    axis-aligned rectangle and so inflate the very hull just measured.
        let turned = rotated_rectangle_geometry(synthesised.bbox, typography.angle_degrees);
        assert_ne!(
            shape.points, turned.points,
            "the synthesised region is a rectangle turned about the bbox centre again -- \
             that is the double-counted rotation"
        );
    }

    /// **`Settled` pairs each detection with its provenance BY INDEX, and a sort
    /// runs between the synthesis and `write_regions`.** So the pairing has to
    /// survive being reordered, and this is the test that says so.
    ///
    /// It is not hypothetical bookkeeping. The synthesis APPENDS, so "the last N
    /// entries are synthesised" is true at the moment it is written and false the
    /// moment anything sorts -- and it then silently names *different* regions
    /// rather than failing. The same defect has been found in a log split; this
    /// is the same shape one layer down, where nothing would print.
    #[test]
    fn sorting_carries_the_provenance_with_the_detections() {
        let page = RgbImage::from_pixel(1200, 908, Rgb([255, 255, 255]));
        /* THE LONELY BUBBLE IS THE ONE THAT SORTS FIRST, AND THAT IS THE WHOLE
         * FIXTURE. The synthesis APPENDS, so with the pair reading first this test
         * would end with the synthesised region still sitting last -- and a broken
         * "the final N entries are the synthesised ones" scheme would satisfy every
         * assertion below while naming regions by position. Putting the lonely
         * bubble in the top band moves its twin to index 1, where only a pairing
         * that actually travelled with the sort can find it. */
        let lonely = [640.0, 64.0, 1120.0, 448.0];
        let mut page = page;
        let detections = vec![
            // A bubble that already holds text, lower down: neither is synthesised.
            detection("bubble", 0.82, [96.0, 512.0, 480.0, 800.0]),
            detection(TEXT, 0.91, [140.0, 560.0, 440.0, 760.0]),
            // And one that holds none, which gains a synthesised twin. It must carry
            // real ink, or nothing is synthesised and this test can no longer see
            // the sort it exists to pin.
            inked_bubble(&mut page, 0.5078125, lonely, true),
        ];

        let (mut settled, _) = settle_reading_bubbles(detections, &page, false);
        settled.sort_by_layout();

        let flagged: Vec<usize> = (0..settled.len())
            .filter(|&index| settled.synthesised(index))
            .collect();
        assert_eq!(flagged.len(), 1, "exactly one region was synthesised");
        assert_ne!(
            flagged[0],
            settled.len() - 1,
            "the fixture stopped exercising the sort: the synthesised region is still last, \
             where naming it by position would also have worked"
        );

        let region = &settled[flagged[0]];
        assert_eq!(
            (region.label.as_str(), region.bbox),
            (TEXT, lonely),
            "the flag followed the sort onto the wrong region"
        );
        // And the converse, which is the half a positive assertion cannot see: no
        // region the network returned may come back flagged.
        for index in 0..settled.len() {
            if index == flagged[0] {
                continue;
            }
            assert!(
                !settled.synthesised(index),
                "{} at {:?} was flagged as synthesised",
                settled[index].label,
                settled[index].bbox
            );
        }
    }

    /// A `bubble` detection whose mask is a filled box and whose IMAGE carries ink.
    ///
    /// **A textless balloon with no ink is not given a region**, so
    /// a fixture built from [`detection`]'s 1x1 dummy mask can no longer drive the
    /// synthesis at all -- there is nothing for `ink_within` to find. This paints the
    /// balloon onto a shared page and returns the detection that describes it.
    ///
    /// Three dark bars, inset well inside the box so the fixture never depends on
    /// where the boundary falls. Their area is far above
    /// `SYNTHESISED_INK_MIN_PIXELS`, and `inked = false` is the honest way to build
    /// the OTHER case -- a balloon over flat artwork, which is what the known
    /// false positive actually is.
    fn inked_bubble(
        page: &mut RgbImage,
        score: f32,
        bbox: [f32; 4],
        inked: bool,
    ) -> KoharuLayoutDetection {
        let (width, height) = (page.width(), page.height());
        let mut pixels = vec![0u8; width as usize * height as usize];
        let [left, top, right, bottom] = bbox;
        let mut area = 0u32;
        for y in (top.max(0.0) as u32)..(bottom.min(height as f32) as u32) {
            for x in (left.max(0.0) as u32)..(right.min(width as f32) as u32) {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
                area += 1;
                let across = (f32::from(x as u16) - left) / (right - left);
                let down = (f32::from(y as u16) - top) / (bottom - top);
                let on_glyph = inked
                    && (0.35..=0.65).contains(&down)
                    && [0.25f32, 0.45, 0.65]
                        .iter()
                        .any(|center| (across - center).abs() <= 0.05);
                page.put_pixel(
                    x,
                    y,
                    if on_glyph {
                        Rgb([20, 20, 20])
                    } else {
                        Rgb([250, 250, 250])
                    },
                );
            }
        }
        KoharuLayoutDetection {
            label_id: 0,
            label: "bubble".to_owned(),
            score,
            bbox,
            area,
            mask: KoharuLayoutMask {
                width,
                height,
                pixels,
            },
        }
    }

    fn detection(label: &str, score: f32, bbox: [f32; 4]) -> KoharuLayoutDetection {
        KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score,
            bbox,
            area: 0,
            mask: KoharuLayoutMask {
                width: 1,
                height: 1,
                pixels: vec![0],
            },
        }
    }

    /// Dialogue never yields. A reader loses more from an unreadable line than
    /// from an untranslated sound effect, and the effect stays legible as the
    /// artwork it already is.
    #[test]
    fn a_sound_effect_yields_to_dialogue_and_never_the_other_way() {
        let all = vec![
            detection("text", 0.9, rect(0.0, 0.0, 100.0, 100.0)),
            detection("onomatopoeia", 0.9, rect(20.0, 20.0, 100.0, 100.0)),
        ];
        assert!(
            super::overshadowed_effect(&all[1], &all, true),
            "the effect overlapping dialogue should yield"
        );
        assert!(
            !super::overshadowed_effect(&all[0], &all, true),
            "dialogue must never be spared"
        );
    }

    /// Exactly ONE of any pair yields. If both did, a page could lose both
    /// effects and the reader would be worse off than without the rule.
    #[test]
    fn exactly_one_of_two_overlapping_effects_yields() {
        for (a, b) in [
            (rect(0.0, 0.0, 200.0, 200.0), rect(50.0, 50.0, 100.0, 100.0)),
            // identical area and score: the tie-break must still decide
            (rect(0.0, 0.0, 100.0, 100.0), rect(10.0, 10.0, 100.0, 100.0)),
        ] {
            let all = vec![
                detection("onomatopoeia", 0.5, a),
                detection("onomatopoeia", 0.5, b),
            ];
            let first = super::overshadowed_effect(&all[0], &all, true);
            let second = super::overshadowed_effect(&all[1], &all, true);
            assert!(
                first ^ second,
                "exactly one of {a:?} / {b:?} must yield, got {first} and {second}"
            );
        }
    }

    /// The smaller effect is the one that yields: the larger is what the artist
    /// drew big, and shrinking the page's loudest sound is the wrong trade.
    #[test]
    fn the_smaller_effect_is_the_one_spared() {
        let big = rect(0.0, 0.0, 300.0, 300.0);
        let small = rect(50.0, 50.0, 80.0, 80.0);
        let all = vec![
            detection("onomatopoeia", 0.5, big),
            detection("onomatopoeia", 0.5, small),
        ];
        assert!(super::overshadowed_effect(&all[1], &all, true));
        assert!(!super::overshadowed_effect(&all[0], &all, true));
    }

    /// A glancing overlap is not a collision. Effects touch constantly on a busy
    /// page and sparing on contact would gut the feature.
    #[test]
    fn a_glancing_overlap_spares_nothing() {
        let all = vec![
            detection("onomatopoeia", 0.9, rect(0.0, 0.0, 100.0, 100.0)),
            detection("onomatopoeia", 0.9, rect(95.0, 95.0, 100.0, 100.0)),
        ];
        assert!(!super::overshadowed_effect(&all[0], &all, true));
        assert!(!super::overshadowed_effect(&all[1], &all, true));
    }

    /// With effects switched off globally there is nothing to spare, and the
    /// existing arm already decides what happens to them.
    #[test]
    fn nothing_is_spared_when_effects_are_not_lettered_at_all() {
        let all = vec![
            detection("text", 0.9, rect(0.0, 0.0, 100.0, 100.0)),
            detection("onomatopoeia", 0.9, rect(20.0, 20.0, 100.0, 100.0)),
        ];
        assert!(!super::overshadowed_effect(&all[1], &all, false));
    }

    #[test]
    fn bubble_geometry_follows_the_instance_mask_polygon() {
        let width = 9;
        let height = 9;
        let mut pixels = vec![0; width * height];
        for y in 0..height {
            for x in 0..width {
                if x.abs_diff(4) + y.abs_diff(4) <= 4 {
                    pixels[y * width + x] = u8::MAX;
                }
            }
        }

        let geometry = mask_geometry(&KoharuLayoutMask {
            width: width as u32,
            height: height as u32,
            pixels,
        })
        .unwrap();

        assert!(geometry.points.len() >= 4);
        assert!(
            geometry
                .points
                .iter()
                .all(|point| !(point.x == 0.0 && point.y == 0.0))
        );
        assert!(
            geometry
                .points
                .iter()
                .any(|point| point.x == 4.0 && point.y == 0.0)
        );
    }

    fn masked_text(
        local_width: f64,
        local_height: f64,
        angle_degrees: f64,
        color: [u8; 3],
    ) -> (RgbImage, KoharuLayoutDetection) {
        let width = 96;
        let height = 96;
        let center_x = f64::from(width) * 0.5;
        let center_y = f64::from(height) * 0.5;
        let (sin, cos) = angle_degrees.to_radians().sin_cos();
        let mut image = RgbImage::from_pixel(width, height, Rgb([200, 180, 160]));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in 0..height {
            for x in 0..width {
                let dx = f64::from(x) + 0.5 - center_x;
                let dy = f64::from(y) + 0.5 - center_y;
                let local_x = dx * cos + dy * sin;
                let local_y = -dx * sin + dy * cos;
                if local_x.abs() <= local_width * 0.5 && local_y.abs() <= local_height * 0.5 {
                    pixels[y as usize * width as usize + x as usize] = u8::MAX;
                    image.put_pixel(x, y, Rgb(color));
                }
            }
        }
        (
            image,
            KoharuLayoutDetection {
                label_id: 0,
                label: "text".to_owned(),
                score: 1.0,
                bbox: [0.0, 0.0, width as f32, height as f32],
                area: pixels.iter().filter(|value| **value != 0).count() as u32,
                mask: KoharuLayoutMask {
                    width,
                    height,
                    pixels,
                },
            },
        )
    }

    /// The composed pass `write_page` actually calls.
    ///
    /// **Every test in this file goes through here and none through its
    /// halves.** A fix once shipped into this crate completely unwired
    /// with 128 tests green, because the tests called the new function and the
    /// caller still called the old one; a test that reaches past the composition
    /// structurally cannot see that. The page is blank white and large, so the
    /// ink walk finds nothing and the suppression is measured alone -- a test
    /// that wants the walk builds its own page.
    fn settle(
        detections: Vec<KoharuLayoutDetection>,
        translate_sfx: bool,
    ) -> (Settled, Vec<EdgeHint>) {
        let image = RgbImage::from_pixel(1200, 1200, Rgb([255, 255, 255]));
        settle_on(detections, &image, translate_sfx)
    }

    /// The same composed pass, over a page a test built on purpose. `0.25` is
    /// the checkpoint's own `text` floor, which is what a real run resolves
    /// unless the config overrides it.
    fn settle_on(
        detections: Vec<KoharuLayoutDetection>,
        image: &RgbImage,
        translate_sfx: bool,
    ) -> (Settled, Vec<EdgeHint>) {
        settle_with(detections, image, translate_sfx, true)
    }

    /// The same again with the flag exposed, so the OFF arm is reachable from a
    /// test rather than only from a config. `settle_on` is the ON arm and stays
    /// the default the other tests call, because ON is what they are about.
    fn settle_with(
        detections: Vec<KoharuLayoutDetection>,
        image: &RgbImage,
        translate_sfx: bool,
        repair: bool,
    ) -> (Settled, Vec<EdgeHint>) {
        let size = ImageSize {
            width: image.width(),
            height: image.height(),
        };
        settle_detections(
            detections, 0.5, translate_sfx, 0.25, None, size, image, repair, false, false, false,
        )
    }

    /// The composed pass with `read_textless_bubbles` ON, and `repair` ON as it
    /// ships. Separate from `settle_with` so every existing test stays on the OFF
    /// arm -- OFF is the shipping default and what their assertions describe.
    fn settle_reading_bubbles(
        detections: Vec<KoharuLayoutDetection>,
        image: &RgbImage,
        translate_sfx: bool,
    ) -> (Settled, Vec<EdgeHint>) {
        let size = ImageSize {
            width: image.width(),
            height: image.height(),
        };
        settle_detections(
            detections, 0.5, translate_sfx, 0.25, None, size, image, true, true, false, false,
        )
    }

    fn caller(x: f32, y: f32, width: f32, height: f32) -> crate::CallerRegion {
        crate::CallerRegion {
            x,
            y,
            width,
            height,
        }
    }

    /// A reader's deletion removes the settled box -- and with it
    /// its mask, since `write_masks` consumes the same list -- while a rect
    /// covering nothing removes nothing. Proven able to fail by making
    /// `remove_caller_boxes` retain everything: the count assert went red at
    /// once.
    #[test]
    fn a_readers_deletion_removes_the_settled_box_and_a_miss_removes_nothing() {
        let (mut settled, _) = settle(
            vec![
                detection("text", 0.9, rect(10.0, 10.0, 100.0, 40.0)),
                detection("text", 0.8, rect(10.0, 300.0, 100.0, 40.0)),
            ],
            true,
        );
        let removed = settled.remove_caller_boxes(&[caller(0.0, 0.0, 200.0, 100.0)]);
        assert_eq!(removed, 1, "the rect covers the first box's center and only it");
        assert_eq!(settled.len(), 1);
        assert_eq!(settled[0].bbox, rect(10.0, 300.0, 100.0, 40.0));
        assert_eq!(
            settled.remove_caller_boxes(&[caller(1000.0, 1000.0, 50.0, 50.0)]),
            0,
            "a rect over nothing is a no-op, not a nearest-match guess"
        );
        assert_eq!(settled.len(), 1);
    }

    /// The deletion reaches a SYNTHESISED read too -- the textless-bubble
    /// false-positive class is exactly what a reader wants to delete -- and
    /// the provenance vector stays parallel through the removal, which is the
    /// pairing rule `sort_by_layout` documents.
    #[test]
    fn a_readers_deletion_reaches_a_synthesised_read_and_keeps_provenance_parallel() {
        let mut page = RgbImage::from_pixel(1200, 1200, Rgb([255, 255, 255]));
        let bubble = inked_bubble(&mut page, 0.9, rect(600.0, 600.0, 400.0, 300.0), true);
        let (mut settled, _) = settle_reading_bubbles(
            vec![detection("text", 0.9, rect(10.0, 10.0, 100.0, 40.0)), bubble],
            &page,
            true,
        );
        // Network text, the bubble, and its synthesised read.
        assert_eq!(settled.len(), 3);
        assert!(settled.synthesised(2));

        // Deleting the network text box: the synthesised flag must FOLLOW the
        // surviving entries, not stay glued to index 2.
        assert_eq!(
            settled.remove_caller_boxes(&[caller(0.0, 0.0, 150.0, 100.0)]),
            1
        );
        assert_eq!(settled.len(), 2);
        assert!(!settled.synthesised(0), "the bubble is the network's own");
        assert!(settled.synthesised(1), "the synthesised read kept its provenance");

        // And the synthesised read itself is deletable: its bbox is the
        // bubble's own, so one rect over the balloon takes both.
        assert_eq!(
            settled.remove_caller_boxes(&[caller(600.0, 600.0, 400.0, 300.0)]),
            2
        );
        assert_eq!(settled.len(), 0);
    }

    /// The reader's box is admitted with its INK as its
    /// mask -- the textless-bubble lesson, same distillation -- and a
    /// rectangle over bare paper is refused rather than invented, which is
    /// what spares the artwork under a misdrawn box. Off-page is refused
    /// outright.
    #[test]
    fn a_readers_box_is_admitted_on_its_ink_and_refused_on_bare_paper() {
        let mut page = RgbImage::from_pixel(1200, 1200, Rgb([255, 255, 255]));
        for y in 320..380 {
            for x in 120..360 {
                page.put_pixel(x, y, Rgb([20, 20, 20]));
            }
        }
        let (mut settled, _) = settle_on(
            vec![detection("text", 0.9, rect(700.0, 700.0, 100.0, 40.0))],
            &page,
            true,
        );
        let boxes = [
            caller(100.0, 300.0, 300.0, 120.0), // over the ink block
            caller(600.0, 100.0, 200.0, 100.0), // bare paper
        ];
        let donors = settled.caller_donors(&boxes);
        assert!(donors.is_empty(), "no detection's centre lies in either rect");
        let admitted = settled.admit_caller_boxes(&page, &boxes, &donors);
        assert_eq!(admitted, 1, "ink admits, paper refuses");
        assert_eq!(settled.len(), 2);
        let added = &settled[1];
        assert_eq!(added.label, "text");
        assert_eq!(
            added.score, 1.0,
            "the reader's own assertion, reported as exactly that"
        );
        let set = added.mask.pixels.iter().filter(|value| **value != 0).count();
        assert_eq!(added.area, set as u32, "the count describes the mask");
        assert!(set >= 200, "clears ink_within's absolute floor");
        assert!(
            (set as f32) < 300.0 * 120.0 * 0.9,
            "the mask is the ink, not the cloned body"
        );
        assert!(
            !settled.synthesised(1),
            "a caller box is asserted, not synthesised"
        );
        assert_eq!(
            settled.admit_caller_boxes(&page, &[caller(2000.0, 2000.0, 50.0, 50.0)], &[]),
            0,
            "off the page is refused outright"
        );
    }

    /// Measured on a real page: a caller rect that lands where the
    /// detector already saw text inherits the MODEL's mask and kind, so drawn
    /// art inside the rect never joins the erase and a resized sound effect
    /// stays a sound effect. Proven able to fail against a plain `ink_within`
    /// admit: it distilled the art block into the mask (the art assert) and
    /// the label fell to the probe's plain "text" (the kind assert).
    #[test]
    fn a_resized_box_inherits_the_detectors_mask_and_kind_and_spares_the_art() {
        let mut page = RgbImage::from_pixel(1200, 1200, Rgb([255, 255, 255]));
        // The glyph ink the donor's model mask covers...
        for y in 320..380 {
            for x in 120..240 {
                page.put_pixel(x, y, Rgb([20, 20, 20]));
            }
        }
        // ...and the ART inside the reader's grown rect that it does not.
        for y in 430..470 {
            for x in 120..300 {
                page.put_pixel(x, y, Rgb([25, 25, 25]));
            }
        }
        let mut donor = detection("onomatopoeia", 0.9, rect(110.0, 310.0, 150.0, 90.0));
        let mut pixels = vec![0u8; 1200 * 1200];
        for y in 320..380 {
            for x in 120..240 {
                pixels[y * 1200 + x] = 255;
            }
        }
        donor.mask = KoharuLayoutMask {
            width: 1200,
            height: 1200,
            pixels,
        };
        donor.area = 120 * 60;
        let (mut settled, _) = settle_on(vec![donor], &page, true);
        assert_eq!(settled.len(), 1, "the donor survives the settle");

        // The reader's resize gesture, exactly as the editor sends it:
        // remove the old rect, add the grown one that now reaches the art.
        let added = [caller(100.0, 300.0, 220.0, 200.0)];
        let donors = settled.caller_donors(&added);
        assert_eq!(donors.len(), 1, "the donor's centre lies in the new rect");
        assert_eq!(
            settled.remove_caller_boxes(&[caller(100.0, 300.0, 180.0, 110.0)]),
            1
        );
        let admitted = settled.admit_caller_boxes(&page, &added, &donors);
        assert_eq!(admitted, 1);

        let added_detection = &settled[0];
        assert_eq!(
            added_detection.label, "onomatopoeia",
            "the kind rides along: a resized sound effect stays on the SFX path"
        );
        assert_eq!(
            added_detection.score, 1.0,
            "the reader's own assertion fingerprint survives the inheritance"
        );
        let width = added_detection.mask.width as usize;
        let art_pixels = (430..470)
            .flat_map(|y| (120..300).map(move |x| (x, y)))
            .filter(|&(x, y): &(usize, usize)| added_detection.mask.pixels[y * width + x] != 0)
            .count();
        assert_eq!(art_pixels, 0, "not one art pixel joins the erase");
        let glyph_pixels = (320..380)
            .flat_map(|y| (120..240).map(move |x| (x, y)))
            .filter(|&(x, y): &(usize, usize)| added_detection.mask.pixels[y * width + x] != 0)
            .count();
        assert_eq!(
            glyph_pixels,
            120 * 60,
            "every stroke of the model's own mask is kept"
        );
        assert_eq!(added_detection.area, 120 * 60, "the count describes the mask");
    }

    /// The other half of the inheritance: a caller rect that
    /// SHRINKS below the donor's own glyphs must still erase all of them.
    ///
    /// If `inherit_from_donors` clipped the donor's model mask to the reader's
    /// rect, guarding only the total wipeout (`area == 0` falls back to
    /// `ink_within`), a PARTIAL shrink would silently drop the strokes outside,
    /// so the source glyphs beyond the reader's rectangle would never be erased
    /// -- invisible while the English is lettered over them, and revealed the
    /// moment the reader moves the placement box off them: the original
    /// untranslated text shows at the spot the box used to be.
    ///
    /// A clip would also contradict `admit_caller_boxes`' own stated rule --
    /// **"the reader's rect decides what is READ, the model decides what is
    /// ERASED"** -- by letting the rect veto the model.
    ///
    /// Proven able to fail against a clipping implementation.
    #[test]
    fn a_shrunk_box_still_erases_every_glyph_its_donor_saw() {
        let mut page = RgbImage::from_pixel(1200, 1200, Rgb([255, 255, 255]));
        // One wide line of glyph ink, x 120..300.
        for y in 320..380 {
            for x in 120..300 {
                page.put_pixel(x, y, Rgb([20, 20, 20]));
            }
        }
        let mut donor = detection("text", 0.9, rect(110.0, 310.0, 200.0, 90.0));
        let mut pixels = vec![0u8; 1200 * 1200];
        for y in 320..380 {
            for x in 120..300 {
                pixels[y * 1200 + x] = 255;
            }
        }
        donor.mask = KoharuLayoutMask { width: 1200, height: 1200, pixels };
        donor.area = 180 * 60;
        let (mut settled, _) = settle_on(vec![donor], &page, true);
        assert_eq!(settled.len(), 1, "the donor survives the settle");

        // The reader SHRINKS the box: x 110..250, so the donor's glyphs from
        // x 250..300 fall outside the rectangle they drew.
        let added = [caller(110.0, 310.0, 140.0, 90.0)];
        let donors = settled.caller_donors(&added);
        assert_eq!(donors.len(), 1, "the donor's centre still lies in the shrunk rect");
        assert_eq!(
            settled.remove_caller_boxes(&[caller(110.0, 310.0, 200.0, 90.0)]),
            1
        );
        assert_eq!(settled.admit_caller_boxes(&page, &added, &donors), 1);

        let admitted = &settled[0];
        let width = admitted.mask.width as usize;
        let outside = (320..380)
            .flat_map(|y| (250..300).map(move |x| (x, y)))
            .filter(|&(x, y): &(usize, usize)| admitted.mask.pixels[y * width + x] != 0)
            .count();
        assert_eq!(
            outside,
            50 * 60,
            "every glyph the DETECTOR saw is erased, including the strokes \
             outside the reader's shrunk rect -- the model decides what is \
             erased, and a shrink that leaves source ink on the page is the \
             defect the reader sees when they move the English off it"
        );
    }

    /// Found by rendering the no-clip fix: a text box centred in its
    /// balloon has TWO donors -- the text and the balloon enclosing it, whose
    /// centres coincide -- and `inherit_from_donors` took the LARGEST donor's
    /// label. The balloon is always larger, so the reader's resized box came
    /// back kind `bubble`, was never OCR'd as text, and shipped the bubble
    /// **fully untranslated AND un-erased**. Measured on the exhibit page: the
    /// baseline dumped `r18_text_x90y776w114h320`, the resized arm dumped
    /// `r20_bubble_x90y776w114h241`, and the render showed the Japanese intact
    /// with no English at all.
    ///
    /// The mask half matters as much as the label: a balloon's model mask is
    /// the whole balloon SHAPE, so unioning it into `text-mask` would inpaint
    /// the entire balloon -- the same destruction class, arriving by a new
    /// road. Text-bearing donors only, for the kind AND for the mask.
    ///
    /// Proven able to fail on the label assert (`bubble` for `text`).
    #[test]
    fn a_box_in_a_balloon_inherits_the_texts_kind_and_never_the_balloons() {
        let mut page = RgbImage::from_pixel(1200, 1200, Rgb([255, 255, 255]));
        for y in 390..610 {
            for x in 210..290 {
                page.put_pixel(x, y, Rgb([20, 20, 20]));
            }
        }
        // The balloon: a big donor whose mask is the whole balloon body.
        let mut balloon = detection("bubble", 0.9, rect(100.0, 300.0, 300.0, 400.0));
        let mut body = vec![0u8; 1200 * 1200];
        for y in 300..700 {
            for x in 100..400 {
                body[y * 1200 + x] = 255;
            }
        }
        balloon.mask = KoharuLayoutMask { width: 1200, height: 1200, pixels: body };
        balloon.area = 300 * 400;
        // The text inside it, sharing the balloon's centre exactly.
        let mut text = detection("text", 0.9, rect(200.0, 380.0, 100.0, 240.0));
        let mut glyphs = vec![0u8; 1200 * 1200];
        for y in 390..610 {
            for x in 210..290 {
                glyphs[y * 1200 + x] = 255;
            }
        }
        text.mask = KoharuLayoutMask { width: 1200, height: 1200, pixels: glyphs };
        text.area = 80 * 220;
        let (mut settled, _) = settle_on(vec![balloon, text], &page, true);
        assert_eq!(settled.len(), 2, "both the balloon and its text survive the settle");

        // The reader shrinks the text box; both donors' centres are inside it.
        let added = [caller(200.0, 380.0, 100.0, 180.0)];
        let donors = settled.caller_donors(&added);
        assert_eq!(donors.len(), 2, "the balloon donates alongside its text");
        assert_eq!(settled.remove_caller_boxes(&[caller(200.0, 380.0, 100.0, 240.0)]), 2);
        assert_eq!(settled.admit_caller_boxes(&page, &added, &donors), 1);

        let admitted = &settled[0];
        assert_eq!(
            admitted.label, TEXT,
            "a text box in a balloon stays TEXT -- inheriting the balloon's kind \
             takes the region off the text path entirely and ships the bubble \
             untranslated and un-erased"
        );
        let width = admitted.mask.width as usize;
        // The balloon's own body, where no glyph is: never in the erase mask.
        assert_eq!(
            admitted.mask.pixels[320 * width + 120], 0,
            "the balloon's SHAPE never joins the erase -- inpainting it would \
             destroy the balloon, the same destruction class by another road"
        );
        // And the no-clip rule still holds: glyphs below the reader's shrunk rect stay.
        let below = (560..610)
            .flat_map(|y| (210..290).map(move |x| (x, y)))
            .filter(|&(x, y): &(usize, usize)| admitted.mask.pixels[y * width + x] != 0)
            .count();
        assert_eq!(below, 50 * 80, "every glyph the detector saw is still erased");
    }

    /// A white page with dark blocks stacked down one column.
    ///
    /// `runs` are `(top, height)` in page coordinates. Deliberately blocks
    /// rather than glyphs: `ink_components` runs Otsu over the crop and keeps
    /// connected components under half its area, so what matters to the walk is
    /// which ROWS carry ink and how far apart they are, and a block says that
    /// without pretending to be lettering.
    ///
    /// **The ink is a ramp and not a flat tone, and that is not decoration.**
    /// `imageproc`'s `otsu_level` returns the LOW end of the plateau of equally
    /// good thresholds, so on a strictly two-tone crop of 16 and 250 it answers
    /// **16** -- the ink's own value -- and `pixel < level` then selects nothing
    /// at all. Measured, on the first version of these fixtures: 398 crop rows,
    /// **0** of them inked, which would have made every growth assertion below
    /// pass for the wrong reason and every refusal assertion vacuous. A real
    /// page never has the problem, because anti-aliasing puts values either side
    /// of the level; the ramp is that, minimally.
    fn column_page(width: u32, height: u32, column: (u32, u32), runs: &[(u32, u32)]) -> RgbImage {
        let mut image = RgbImage::from_pixel(width, height, Rgb([250, 250, 250]));
        let (left, column_width) = column;
        for &(top, run) in runs {
            for y in top..(top + run).min(height) {
                for x in left..(left + column_width).min(width) {
                    let value = 10 + ((x - left) % 32) as u8;
                    image.put_pixel(x, y, Rgb([value, value, value]));
                }
            }
        }
        image
    }

    /// The defect, in the shape two measured test slices have it: the
    /// detector stopped the box well inside the slice while the column's own ink
    /// runs on to the far edge. On the real page that truncation is what shipped
    /// 55% of a display column; here it is a box that must come back spanning.
    #[test]
    fn a_column_whose_ink_reaches_the_far_edge_is_grown_to_span_the_page() {
        // Blocks every 50 rows down the column, the last one running into the
        // bottom band. Every gap is 20 rows, well under `INK_WALK_MAX_GAP_ROWS`.
        let mut runs: Vec<(u32, u32)> = (0..7).map(|index| (4 + index * 50, 30)).collect();
        runs.push((354, 46));
        let image = column_page(200, 400, (60, 40), &runs);
        let detections = vec![detection("text", 0.29, [60.0, 2.0, 100.0, 210.0])];

        let (detections, hints) = settle_on(detections, &image, true);

        assert!(hints.is_empty(), "an admitted region is not a hint");
        assert_eq!(
            detections[0].bbox,
            [60.0, 2.0, 100.0, 400.0],
            "the column must be grown to the page edge it was truncated short of"
        );
    }

    /// THE OFF ARM IS THE OLD BEHAVIOUR, OP FOR OP -- asserted rather than assumed.
    ///
    /// A default-off flag whose off arm nothing tests is a flag that only looks
    /// safe. This drives the SAME fixture as the grow test above and the same
    /// sub-floor box as the hint test, through the same entry point the caller
    /// calls, and demands both halves stay put: the box is not grown and no hint
    /// is reported.
    ///
    /// Both halves in ONE test on purpose. They are one flag, and this repo has
    /// already shipped a fix whose two halves were each tested while the `||`
    /// between them was gone.
    #[test]
    fn the_off_arm_is_the_old_behaviour_op_for_op() {
        let mut runs: Vec<(u32, u32)> = (0..7).map(|index| (4 + index * 50, 30)).collect();
        runs.push((354, 46));
        let image = column_page(200, 400, (60, 40), &runs);
        let truncated = [60.0, 2.0, 100.0, 210.0];
        let detections = vec![
            detection("text", 0.29, truncated),
            // Sub-floor and against the bottom band: the measured display-column shape.
            detection("text", 0.21, [130.0, 300.0, 190.0, 399.0]),
        ];

        let (settled, hints) = settle_with(detections, &image, true, false);

        assert!(
            hints.is_empty(),
            "with the repair off nothing may be reported as an edge hint"
        );
        assert_eq!(
            settled[0].bbox, truncated,
            "with the repair off a truncated column must keep the box the detector gave it"
        );
    }

    /// A measured slice-pair composite and a single slice, verbatim from a
    /// server log -- the axis tie-break's two exhibits. The
    /// FRAGMENT survives today and reads half the text; the COLUMN is the whole
    /// cut column the composite squashed to 0.2383. The STRIP is the unrelated
    /// band box on the same composite that a plain lowered floor would have
    /// admitted as a brand-new region. On the single slice the WIDE box merges two columns
    /// and loses the fourth glyph of a name; the TIGHT box is the name column.
    const COMPOSITE_FRAGMENT: [f32; 4] = [112.5, 33.75, 234.375, 485.15625];
    const COMPOSITE_COLUMN: [f32; 4] = [111.91406, 23.203125, 239.0625, 1046.25];
    const COMPOSITE_STRIP: [f32; 4] = [885.9375, 51.679688, 1157.8125, 129.72656];
    const SLICE_WIDE: [f32; 4] = [48.632813, 17.96875, 246.09375, 477.96875];
    const SLICE_TIGHT: [f32; 4] = [121.875, 19.765625, 243.75, 474.375];

    /// The axis tie-break's entry point: ordinary floor 0.25, `repair` and
    /// `read_textless` off so nothing but the suppression is in play, the band
    /// and the tie-break exposed.
    fn settle_axis(
        detections: Vec<KoharuLayoutDetection>,
        image: &RgbImage,
        joined_text_floor: Option<f32>,
        axis_aware: bool,
        residue: bool,
    ) -> (Settled, Vec<EdgeHint>) {
        let size = ImageSize {
            width: image.width(),
            height: image.height(),
        };
        settle_detections(
            detections,
            0.5,
            true,
            0.25,
            joined_text_floor,
            size,
            image,
            false,
            false,
            axis_aware,
            residue,
        )
    }

    fn white_page(width: u32, height: u32) -> RgbImage {
        RgbImage::from_pixel(width, height, Rgb([250, 250, 250]))
    }

    /// The composite pair is caught by the CONTAINMENT arm alone -- its IoU is
    /// under the 0.5 threshold -- so deleting that arm turns this Independent.
    /// The single-slice pair cannot prove the containment arm load-bearing (it clears both arms),
    /// which is why this fixture exists.
    #[test]
    fn the_composite_pair_is_caught_by_containment_alone() {
        let fragment = detection("text", 0.271484375, COMPOSITE_FRAGMENT);
        let column = detection("text", 0.23828125, COMPOSITE_COLUMN);
        assert!(
            super::intersection_over_union(COMPOSITE_FRAGMENT, COMPOSITE_COLUMN) < 0.5,
            "the fixture must sit under the IoU threshold or it proves nothing"
        );
        assert!(
            super::overlap_over_smaller(COMPOSITE_FRAGMENT, COMPOSITE_COLUMN) >= 0.9,
            "the fragment is wholly inside the column"
        );
        assert_eq!(
            nms_outcome(&fragment, &column, 0.5, true, false),
            NmsOutcome::SuppressCandidate,
            "without the tie-break the column dies on arrival -- the no-op the \
             floor-only census arm encodes"
        );
    }

    /// A pair the IoU arm catches and the containment arm does not, so deleting
    /// the IoU arm turns THIS one Independent.
    #[test]
    fn an_overlap_without_containment_is_still_suppressed() {
        let kept = detection("text", 0.9, rect(0.0, 0.0, 100.0, 100.0));
        let candidate = detection("text", 0.8, rect(0.0, 15.0, 100.0, 100.0));
        assert!(
            super::intersection_over_union(kept.bbox, candidate.bbox) >= 0.5,
            "the fixture must clear the IoU threshold"
        );
        assert!(
            super::overlap_over_smaller(kept.bbox, candidate.bbox) < 0.9,
            "neither box may contain the other or this collapses into the other arm"
        );
        assert_eq!(
            nms_outcome(&kept, &candidate, 0.5, true, true),
            NmsOutcome::SuppressCandidate,
            "an equal-shape overlap is not an axis case; today's answer stands"
        );
    }

    /// The dead band, both ways: a contained pair that differs on BOTH axes has
    /// no evidence which box is the column, and a near-identical pair has no
    /// column at all. A bare "prefer the higher aspect" rule would fire on the
    /// first; this pins that it must not.
    #[test]
    fn an_axis_ambiguous_pair_keeps_todays_answer() {
        let kept = detection("text", 0.6, rect(0.0, 0.0, 300.0, 1000.0));
        let differs_on_both = detection("text", 0.5, rect(50.0, 50.0, 100.0, 600.0));
        assert_eq!(
            nms_outcome(&kept, &differs_on_both, 0.5, true, true),
            NmsOutcome::SuppressCandidate,
            "differing on both axes is a merge question, not a column question"
        );
        let near_identical = detection("text", 0.5, rect(2.0, 2.0, 296.0, 994.0));
        assert_eq!(
            nms_outcome(&kept, &near_identical, 0.5, true, true),
            NmsOutcome::SuppressCandidate,
            "a twin is a duplicate read, and the higher score keeps it"
        );
    }

    /// The confidence rail: the single slice's geometry with the wide box's score raised to
    /// 0.90 puts the challenger at 0.31x of the incumbent, under the 0.60
    /// floor. Removing the `confident_enough` term turns this ReplaceKept.
    #[test]
    fn a_much_weaker_candidate_may_not_evict_a_confident_box() {
        let wide = detection("text", 0.90, SLICE_WIDE);
        let tight = detection("text", 0.283203125, SLICE_TIGHT);
        assert_eq!(
            nms_outcome(&wide, &tight, 0.5, true, true),
            NmsOutcome::SuppressCandidate,
            "a barely-detected column may not evict a confident box"
        );
    }

    /// THE CENSUS CONTROL, ENCODED: the floor alone must change nothing. The
    /// column enters the band, conflicts with the fragment, and without the
    /// tie-break no replacement outcome exists, so the fragment survives alone
    /// -- exactly today's answer. Setting `axis_aware` true here changes the
    /// survivor, which is what proves the tie-break is wired (the unwired-fix
    /// failure mode, countered by construction).
    #[test]
    fn the_floor_alone_still_loses_the_composite_column() {
        let image = white_page(1200, 1100);
        let detections = vec![
            detection("text", 0.271484375, COMPOSITE_FRAGMENT),
            detection("text", 0.23828125, COMPOSITE_COLUMN),
        ];
        let (settled, hints) = settle_axis(detections, &image, Some(0.20), false, false);
        assert_eq!(settled.len(), 1, "exactly one of the pair survives");
        assert_eq!(
            settled[0].bbox, COMPOSITE_FRAGMENT,
            "without the tie-break the fragment wins, exactly as it does today"
        );
        assert!(hints.is_empty(), "repair is off; a dropped band box is silent");
    }

    /// The paired lever on the composite: the whole column replaces the
    /// fragment. This is the first half of the acceptance target.
    #[test]
    fn the_pair_keeps_the_whole_column_on_the_composite() {
        let image = white_page(1200, 1100);
        let detections = vec![
            detection("text", 0.271484375, COMPOSITE_FRAGMENT),
            detection("text", 0.23828125, COMPOSITE_COLUMN),
        ];
        let (settled, _) = settle_axis(detections, &image, Some(0.20), true, false);
        assert_eq!(settled.len(), 1, "a replacement is a swap, never an addition");
        assert_eq!(
            settled[0].bbox, COMPOSITE_COLUMN,
            "the whole cut column must be the box that survives"
        );
    }

    /// The tie-break on an ORDINARY slice, no band at all: the single slice's
    /// tight name column evicts the wide merge. Scoping the tie-break to joined
    /// pages turns this red -- that slice is not a composite, and the second half
    /// of the acceptance target lives there.
    #[test]
    fn the_tie_break_reaches_an_ordinary_slice() {
        let image = white_page(260, 500);
        let detections = vec![
            detection("text", 0.373046875, SLICE_WIDE),
            detection("text", 0.283203125, SLICE_TIGHT),
        ];
        let (settled, _) = settle_axis(detections, &image, None, true, false);
        assert_eq!(settled.len(), 1);
        assert_eq!(
            settled[0].bbox, SLICE_TIGHT,
            "the tight name column must evict the wide two-column merge"
        );
    }

    /// A detection whose mask actually covers its box, page-sized the way the
    /// network returns them. `detection` above hands back a 1x1 mask, which is
    /// fine for geometry-only tests and useless here: `ink_within` windows the
    /// mask by the bbox and would see a single pixel.
    fn masked_detection(
        label: &str,
        score: f32,
        bbox: [f32; 4],
        width: u32,
        height: u32,
    ) -> KoharuLayoutDetection {
        let mut pixels = vec![0u8; (width * height) as usize];
        for y in bbox[1].max(0.0) as u32..(bbox[3] as u32).min(height) {
            for x in bbox[0].max(0.0) as u32..(bbox[2] as u32).min(width) {
                pixels[(y * width + x) as usize] = u8::MAX;
            }
        }
        KoharuLayoutDetection {
            mask: KoharuLayoutMask {
                width,
                height,
                pixels,
            },
            ..detection(label, score, bbox)
        }
    }

    /// A page carrying one solid rectangle of `ink` colour, for `strike_ink`.
    fn page_with_mark(
        width: u32,
        height: u32,
        mark: [u32; 4],
        color: [u8; 3],
    ) -> RgbImage {
        let mut page = RgbImage::from_pixel(width, height, Rgb([40, 40, 60]));
        let [x0, y0, x1, y1] = mark;
        for y in y0..y1 {
            for x in x0..x1 {
                page.put_pixel(x, y, Rgb(color));
            }
        }
        page
    }

    /// A DRAWN STRIKE IS FOUND, AND ITS OWN COLOUR COMES BACK.
    ///
    /// The proportions are the measured exhibit's: a ~13 px mark down a ~122 px column, running
    /// the full height. The colour is the page's measured `#FA0355`, and the
    /// assertion is on the returned value rather than on `is_some`, because the
    /// mark is RE-DRAWN in whatever this returns -- a detector that finds the
    /// stroke and reports the wrong ink would letter a red cancellation in some
    /// other colour and no count would notice.
    #[test]
    fn a_drawn_strike_is_found_with_its_own_ink() {
        let image = page_with_mark(200, 500, [90, 20, 103, 480], [250, 3, 85]);
        let found = strike_ink(&image, &detection("text", 0.3, [40.0, 10.0, 162.0, 490.0]));
        assert_eq!(
            found,
            Some([250, 3, 85, 255]),
            "the strike's own ink, not a quantisation bucket's centre"
        );
    }

    /// PAGE FURNITURE IS NOT A STRIKE, and geometry alone cannot tell them
    /// apart.
    ///
    /// The same rectangle as the test above, in black. Long, thin, solid and one
    /// colour describes a panel border exactly as well as it describes a
    /// cancellation, so before the colour gate this returned `Some([0,0,0,255])`
    /// and the renderer drew a black line through the region's lettering.
    ///
    /// Measured over all 179 slices of a test chapter: fourteen marks found, TWO of them
    /// the author's. The other twelve were borders up to 297,162 px, 56 px
    /// gutters, and a 138 x 4 px sliver along a page's top edge.
    #[test]
    fn a_black_mark_of_the_same_shape_is_page_furniture() {
        let image = page_with_mark(200, 500, [90, 20, 103, 480], [0, 0, 0]);
        let found = strike_ink(&image, &detection("text", 0.3, [40.0, 10.0, 162.0, 490.0]));
        assert_eq!(found, None, "a black border is not a cancellation");
    }

    /// The other half of the same population: a white gutter.
    #[test]
    fn a_white_gutter_of_the_same_shape_is_page_furniture() {
        let image = page_with_mark(200, 500, [90, 20, 103, 480], [255, 255, 255]);
        let found = strike_ink(&image, &detection("text", 0.3, [40.0, 10.0, 162.0, 490.0]));
        assert_eq!(found, None, "a white gutter is not a cancellation");
    }

    /// THE GATE ASKS WHETHER THE MARK HAS A COLOUR, NOT HOW SATURATED IT IS, and
    /// this pins that it is not accidentally a saturation floor.
    ///
    /// Two muted browns either side of `STRIKE_MIN_CHROMA`. Neither is anything
    /// like the author's `#FA0355`, and the faintly coloured one is still kept --
    /// a series that cancels a name in a dull ink is in scope, which is the whole
    /// difference between this and the saturation floor the docstring refutes.
    #[test]
    fn a_faintly_coloured_mark_is_kept_where_a_greyer_one_is_not() {
        let region = detection("text", 0.3, [40.0, 10.0, 162.0, 490.0]);

        let kept = page_with_mark(200, 500, [90, 20, 103, 480], [120, 100, 90]);
        assert_eq!(
            strike_ink(&kept, &region),
            Some([120, 100, 90, 255]),
            "chroma 30 is a colour, however muted"
        );

        let refused = page_with_mark(200, 500, [90, 20, 103, 480], [110, 100, 90]);
        assert_eq!(
            strike_ink(&refused, &region),
            None,
            "chroma 20 is grey, and grey marks are furniture"
        );
    }

    /// ARTWORK IS NOT A STRIKE, and this is the test that keeps the device off
    /// every page that merely has colour on it.
    ///
    /// Same colour, same region, far MORE of it -- a blob rather than a stroke.
    /// A page-scale test failed for exactly this reason: one measured page carries two
    /// red masses several times larger than its actual strike, so "there is red
    /// here" cannot be the test and "it is long, thin and solid" has to be.
    #[test]
    fn a_blob_of_the_same_colour_is_not_a_strike() {
        let image = page_with_mark(200, 500, [50, 100, 150, 400], [250, 3, 85]);
        assert_eq!(
            strike_ink(&image, &detection("text", 0.3, [40.0, 10.0, 162.0, 490.0])),
            None,
            "a wide mass of the device's own colour must not read as a mark"
        );
    }

    /// A short streak is decoration, not a cancellation: a strike crosses what it
    /// cancels. Same thin shape as the positive, a fifth of the length.
    #[test]
    fn a_short_streak_is_not_a_strike() {
        let image = page_with_mark(200, 500, [90, 200, 103, 290], [250, 3, 85]);
        assert_eq!(
            strike_ink(&image, &detection("text", 0.3, [40.0, 10.0, 162.0, 490.0])),
            None,
            "a mark that crosses only part of the region is not a cancellation"
        );
    }

    /// A white page with one dark block on it, so a residue strip can be given
    /// ink or denied it without anything else about the fixture changing.
    fn page_with_ink(width: u32, height: u32, ink: Option<[u32; 4]>) -> RgbImage {
        let mut page = white_page(width, height);
        if let Some([x0, y0, x1, y1]) = ink {
            for y in y0..y1 {
                for x in x0..x1 {
                    page.put_pixel(x, y, Rgb([20, 20, 20]));
                }
            }
        }
        page
    }

    /// THE OFF ARM IS THE PLAIN AXIS TIE-BREAK OP FOR OP, and this is the whole
    /// control arm.
    ///
    /// Same fixture as the ON test below, ink and all -- only the flag differs.
    /// If this goes red the residue admission has leaked out of its gate, and
    /// the tie-break's census loses the byte-exact baseline it was measured against.
    #[test]
    fn the_off_arm_admits_no_residue() {
        let image = page_with_ink(260, 500, Some([54, 176, 119, 308]));
        let detections = vec![
            masked_detection("text", 0.373046875, SLICE_WIDE, 260, 500),
            masked_detection("text", 0.283203125, SLICE_TIGHT, 260, 500),
        ];
        let (settled, _) = settle_axis(detections, &image, None, true, false);
        assert_eq!(
            settled.len(),
            1,
            "with the flag off an evicted box must still go whole"
        );
        assert_eq!(settled[0].bbox, SLICE_TIGHT);
    }

    /// THE GLOSS COMES BACK.
    ///
    /// The measured exhibit's real geometry, with ink where the author's red replacement name is
    /// actually drawn (x54-119, y176-308, measured off the source page rather
    /// than invented for the fixture). The tie-break still evicts the wide box;
    /// what changes is that the 73 px it held left of the winner comes back as a
    /// region instead of leaving with it.
    ///
    /// Note WHICH strip is asserted. Four are offered and three are slivers --
    /// 2.3 px, 1.8 px and 3.6 px -- so this pins the geometry floor too: a test
    /// that only counted regions would pass on a build that admitted all four.
    #[test]
    fn an_evicted_boxs_inked_residue_becomes_its_own_region() {
        let image = page_with_ink(260, 500, Some([54, 176, 119, 308]));
        let detections = vec![
            masked_detection("text", 0.373046875, SLICE_WIDE, 260, 500),
            masked_detection("text", 0.283203125, SLICE_TIGHT, 260, 500),
        ];
        let (settled, _) = settle_axis(detections, &image, None, true, true);
        assert_eq!(
            settled.len(),
            2,
            "the winner plus exactly one residue strip, not all four offered"
        );
        assert_eq!(settled[0].bbox, SLICE_TIGHT, "the winner is unchanged");
        let residue = settled[1].bbox;
        assert_eq!(
            residue,
            [
                SLICE_WIDE[0],
                SLICE_WIDE[1],
                SLICE_TIGHT[0],
                SLICE_WIDE[3]
            ],
            "the admitted strip is the loser left of the winner, at full height"
        );
        assert!(
            residue[2] - residue[0] > 70.0,
            "the gloss column is ~73 px wide; a sliver here means the wrong strip"
        );
    }

    /// The measured exhibit's strike ink, and a second one that is not it.
    const STRIKE: [u8; 4] = [250, 3, 85, 255];
    const OTHER_STRIKE: [u8; 4] = [12, 200, 40, 255];

    /// THE COMPOSITION THE CALLER RUNS, and the reason these tests use it rather
    /// than calling either half alone: the rule is the `&&` between "this is a
    /// residue strip" and "it carries no mark of its own", joined to a page-level
    /// decision computed from every OTHER region. Asserting on the halves would
    /// leave that join untested, which is how a fix ships green and unwired.
    ///
    /// Mirrors `link_dialogue_regions` exactly: page ink once, then this text.
    fn ink_for(page: &[(bool, Option<[u8; 4]>)], index: usize) -> Option<[u8; 4]> {
        let device = device_strike_ink(page.iter().map(|&(_, strike)| strike));
        let (residue, own_strike) = page[index];
        gloss_ink(residue, own_strike, device)
    }

    /// The measured device: the struck name carries the ink, and the residue gloss
    /// standing beside it letters in it. One ink for the whole device.
    #[test]
    fn the_gloss_letters_in_the_strikes_own_ink() {
        let page = [(false, Some(STRIKE)), (true, None)];
        assert_eq!(
            ink_for(&page, 1),
            Some(STRIKE),
            "the gloss takes the mark's own colour, read off the page"
        );
    }

    /// THE SKIP THAT SEPARATES COMPLETING THE DEVICE FROM DESTROYING IT. The
    /// author drew the red OVER the name's black glyphs and their white outline;
    /// recolouring the name would erase the contrast the strike is drawn against.
    #[test]
    fn a_struck_name_is_never_recoloured_by_its_own_mark() {
        let page = [(false, Some(STRIKE)), (true, None)];
        assert_eq!(ink_for(&page, 0), None, "the struck name keeps its own fill");
    }

    /// Over-firing here would repaint ordinary dialogue in a device colour, on
    /// every page that happens to carry a strike anywhere.
    #[test]
    fn an_ordinary_region_never_takes_the_device_ink() {
        let page = [(false, Some(STRIKE)), (false, None)];
        assert_eq!(ink_for(&page, 1), None);
    }

    /// A residue strip that carries a mark of its own is a struck name, whatever
    /// admitted it -- provenance decides that it MAY be a gloss, the absent mark
    /// decides that it is.
    #[test]
    fn a_residue_that_carries_its_own_mark_keeps_its_colour() {
        let page = [(true, Some(STRIKE))];
        assert_eq!(ink_for(&page, 0), None);
    }

    /// Two devices on one page. Pairing a gloss to the nearer mark is a geometry
    /// problem nobody has measured, so this refuses rather than lettering a name
    /// in the other device's colour -- a wrong colour is recoverable, a wrong
    /// pairing reads as a different character.
    #[test]
    fn two_different_strike_inks_refuse_rather_than_guess() {
        let page = [
            (false, Some(STRIKE)),
            (false, Some(OTHER_STRIKE)),
            (true, None),
        ];
        assert_eq!(ink_for(&page, 2), None);
    }

    /// But the same ink twice is ONE device, not two, and must still apply --
    /// otherwise a name struck across two lines would silently disable the gloss.
    #[test]
    fn one_ink_on_two_marks_is_still_one_device() {
        let page = [(false, Some(STRIKE)), (false, Some(STRIKE)), (true, None)];
        assert_eq!(ink_for(&page, 2), Some(STRIKE));
    }

    /// THE OFF ARM. With `--strike-through-devices` false no region carries a
    /// strike at all, so there is no device ink and the gloss letters exactly as
    /// it does today -- the control arm must be byte-exact.
    #[test]
    fn with_no_strike_on_the_page_the_gloss_is_left_alone() {
        let page = [(false, None), (true, None)];
        assert_eq!(ink_for(&page, 1), None);
    }

    /// THE PROVENANCE MUST TRAVEL WITH THE SORT, not with the index. That is the
    /// positional-pairing defect exactly, and the reason `Settled` permutes both vectors by one
    /// order rather than sorting one of them.
    ///
    /// The strip is admitted LAST and sits LEFT of the winner, so a scheme that
    /// remembered "the last entry is the residue" would satisfy every count-based
    /// assertion here and name the struck name instead. The assertion is
    /// therefore on the flagged region's GEOMETRY, which only a pairing that
    /// actually travelled can satisfy.
    #[test]
    fn sorting_carries_the_residue_flag_with_the_detections() {
        let image = page_with_ink(260, 500, Some([54, 176, 119, 308]));
        let detections = vec![
            masked_detection("text", 0.373046875, SLICE_WIDE, 260, 500),
            masked_detection("text", 0.283203125, SLICE_TIGHT, 260, 500),
        ];
        let (mut settled, _) = settle_axis(detections, &image, None, true, true);
        assert!(settled.residue(1), "the strip is admitted last, before sorting");
        assert!(!settled.residue(0), "the winner is not a residue strip");

        settled.sort_by_layout();

        let flagged: Vec<usize> = (0..settled.len())
            .filter(|&index| settled.residue(index))
            .collect();
        assert_eq!(flagged.len(), 1, "exactly one region is a residue strip");
        let strip = settled[flagged[0]].bbox;
        assert!(
            strip[2] - strip[0] > 70.0,
            "the flag must name the ~73 px gloss column, not the struck name"
        );
        assert_eq!(
            strip[0], SLICE_WIDE[0],
            "the gloss column starts at the evicted box's left edge"
        );
    }

    /// THE SAFETY PROPERTY, and it guards ARTWORK rather than text.
    ///
    /// A region is not only lettered, it is ERASED. A residue strip with nothing
    /// in it would cost a reader real artwork to letter nothing, which is a
    /// known defect class bought for free. Same fixture as the test above with
    /// the ink removed, so the only thing measured is whether `ink_within` is
    /// consulted at all.
    #[test]
    fn an_evicted_boxs_blank_residue_is_refused() {
        let image = page_with_ink(260, 500, None);
        let detections = vec![
            masked_detection("text", 0.373046875, SLICE_WIDE, 260, 500),
            masked_detection("text", 0.283203125, SLICE_TIGHT, 260, 500),
        ];
        let (settled, _) = settle_axis(detections, &image, None, true, true);
        assert_eq!(
            settled.len(),
            1,
            "an empty residue must not become a region, because a region is erased"
        );
        assert_eq!(settled[0].bbox, SLICE_TIGHT);
    }

    /// The reason the band is replacement-only: the same composite carries an
    /// unrelated 0.2266 strip inside `[0.20, 0.25)`. A plain lowered floor
    /// admits it as a brand-new region -- the broad joined-page admission
    /// already measured and rejected, and the end of the control arm's byte-identity. Here it
    /// must vanish without trace while the column still wins.
    #[test]
    fn a_band_box_that_replaces_nothing_is_not_admitted() {
        let image = white_page(1200, 1100);
        let detections = vec![
            detection("text", 0.271484375, COMPOSITE_FRAGMENT),
            detection("text", 0.23828125, COMPOSITE_COLUMN),
            detection("text", 0.2265625, COMPOSITE_STRIP),
        ];
        let (settled, hints) = settle_axis(detections, &image, Some(0.20), true, false);
        assert_eq!(
            settled.len(),
            1,
            "the strip conflicts with nothing, so it may replace nothing, so it \
             must not exist"
        );
        assert_eq!(settled[0].bbox, COMPOSITE_COLUMN);
        assert!(hints.is_empty(), "repair is off; the strip drops silently");
    }

    /// A candidate that conflicts with TWO kept boxes keeps today's answer,
    /// whatever the axis evidence says about either conflict: a suppression
    /// from any kept box outranks a replacement offered by another. The small
    /// box sits in the column's lower half, so the column arrives conflicting
    /// with both it and the fragment.
    #[test]
    fn a_candidate_conflicting_with_two_kept_boxes_keeps_todays_answer() {
        let image = white_page(1200, 1100);
        let lower_cell = rect(130.0, 600.0, 70.0, 100.0);
        let detections = vec![
            detection("text", 0.30, lower_cell),
            detection("text", 0.271484375, COMPOSITE_FRAGMENT),
            detection("text", 0.23828125, COMPOSITE_COLUMN),
        ];
        let (settled, _) = settle_axis(detections, &image, Some(0.20), true, false);
        assert_eq!(
            settled.len(),
            2,
            "the column conflicts with two kept boxes and must not land"
        );
        assert_eq!(settled[0].bbox, lower_cell);
        assert_eq!(settled[1].bbox, COMPOSITE_FRAGMENT);
    }

    /// The band clamp: a floor ABOVE the ordinary one may not raise it. A 0.30
    /// text box is an ordinary region and must survive a fat-fingered
    /// `--joined-page-text-floor 0.9` untouched.
    #[test]
    fn a_joined_floor_above_the_ordinary_one_cannot_raise_it() {
        let image = white_page(300, 300);
        let box_bbox = rect(10.0, 10.0, 80.0, 200.0);
        let detections = vec![detection("text", 0.30, box_bbox)];
        let (settled, _) = settle_axis(detections, &image, Some(0.9), true, false);
        assert_eq!(settled.len(), 1, "an ordinary region must survive the clamp");
        assert_eq!(settled[0].bbox, box_bbox);
    }

    /// The negative control, and the one that decides whether this rule invents
    /// columns. One measured slice's box is the real instance: `[167.6, 0, 202.7, 207.5]`,
    /// against the top band, tall and narrow, and its ink genuinely STOPS at 207.
    /// Growing it would claim a column the artist did not draw.
    #[test]
    fn a_column_whose_ink_stops_inside_the_page_is_left_alone() {
        let image = column_page(200, 400, (60, 40), &[(4, 206)]);
        let detections = vec![detection("text", 0.29, [60.0, 2.0, 100.0, 210.0])];

        let (detections, hints) = settle_on(detections, &image, true);

        assert!(hints.is_empty());
        assert_eq!(
            detections[0].bbox,
            [60.0, 2.0, 100.0, 210.0],
            "a column that ends inside the page must not be grown"
        );
    }

    /// Ink further from the box than the box is tall is a DIFFERENT column, a
    /// caption or a signature -- not this one continuing. Same page, same
    /// unbroken chain of ink down to the bottom band; the only thing that
    /// refuses it is [`INK_WALK_REACH_RATIO`], and every gap in the chain is
    /// under [`INK_WALK_MAX_GAP_ROWS`] so nothing else can be doing the work.
    #[test]
    fn a_column_whose_ink_resumes_beyond_its_own_reach_is_left_alone() {
        let mut runs: Vec<(u32, u32)> = (0..3).map(|index| (4 + index * 50, 30)).collect();
        runs.extend((0..14).map(|index| (200 + index * 50, 30)));
        runs.push((886, 12));
        let image = column_page(200, 900, (60, 40), &runs);
        let detections = vec![detection("text", 0.29, [60.0, 2.0, 100.0, 150.0])];

        let (detections, _) = settle_on(detections, &image, true);

        assert_eq!(
            detections[0].bbox,
            [60.0, 2.0, 100.0, 150.0],
            "ink beyond one box-height is another region, not this one"
        );
    }

    /// A box already against both bands has no direction to be grown in, and
    /// the slice-fragment refusal already has a rule for it. `edge_touched` answers `None` for
    /// "both" as well as for "neither", and this is the assertion that keeps the
    /// two arms together: the ink here runs into the bottom band, so a rule that
    /// treated "both" as "top" would walk four rows and grow it.
    #[test]
    fn a_column_already_spanning_both_bands_is_not_touched() {
        let mut runs: Vec<(u32, u32)> = (0..7).map(|index| (4 + index * 50, 30)).collect();
        runs.push((354, 46));
        let image = column_page(200, 400, (60, 40), &runs);
        let detections = vec![detection("text", 0.29, [60.0, 2.0, 100.0, 396.0])];

        let (detections, _) = settle_on(detections, &image, true);

        assert_eq!(detections[0].bbox, [60.0, 2.0, 100.0, 396.0]);
    }

    /// The box this whole channel exists for, at the score the GPU gave it.
    ///
    /// One measured slice's `[52.1, 712.9, 243.8, 904.5]` scores **0.2363** against the
    /// 0.25 floor, so the seam never hears about it and joins four slices
    /// instead of five -- losing a whole glyph of a name. It must come back as a
    /// hint, and it must NOT come back as a detection: `mask_includes` builds the
    /// erase mask from raw detections, so admitting it would strip the artwork
    /// under it on every ordinary page read while nothing ever letters it.
    #[test]
    fn a_sub_floor_box_against_one_edge_is_reported_and_never_admitted() {
        let image = RgbImage::from_pixel(1200, 908, Rgb([250, 250, 250]));
        let detections = vec![
            detection("text", 0.2363, [52.1, 712.9, 243.8, 904.5]),
            detection("text", 0.62, [400.0, 100.0, 460.0, 300.0]),
        ];

        let (detections, hints) = settle_on(detections, &image, true);

        assert_eq!(
            hints,
            vec![EdgeHint {
                x: 52.1,
                y: 712.9,
                width: 243.8 - 52.1,
                height: 904.5 - 712.9,
                edge: PageEdge::Bottom,
                // The score is the reason this box is a hint rather than a region,
                // so it is asserted rather than ignored: reported VERBATIM, not
                // clamped to the floor it failed and not rounded. This fixture is
                // a measured slice and 0.2363 is its measured score, so a
                // change here is a change in what the detector said.
                score: 0.2363,
                // This fixture's page is blank paper, so the ink walk has nothing
                // to follow and the box is reported exactly as the detector drew
                // it. `false` here is the plain one-sided hint, asserted so the sub-floor
                // growth added later cannot start firing on every hint unnoticed.
                spans: false,
            }]
        );
        assert_eq!(
            detections.len(),
            1,
            "a hint must never survive into the detections: {detections:?}"
        );
        assert_eq!(detections[0].score, 0.62);
    }

    /// A SUB-FLOOR column whose own ink crosses the slice is grown, and
    /// says so -- while still never becoming a region.
    ///
    /// This is a measured test slice in miniature, and it is the case the whole
    /// run-spanning join was blocked on. That slice has no admitted box at all: its
    /// best candidate scores **0.2295** against a 0.25 floor while its ink runs the
    /// full height of the slice. Without this it could become a TOP hint and nothing
    /// more, so no run could pass through it and the column was read as two
    /// fragments -- the confabulation the slice-fragment refusal exists to refuse.
    ///
    /// **The score is the real one, and it sits in the live window on purpose.**
    /// 0.2295 is above `EDGE_HINT_MIN_SCORE` (0.12, and also above the previous
    /// 0.20) and below the floor (0.25), so
    /// moving either constant past it turns this test red rather than quietly
    /// changing which boxes the mechanism sees.
    ///
    /// Three things are asserted together because they are one behaviour: the box
    /// is **grown**, it is **flagged** `spans`, and it is **still not a region**.
    /// Asserting the growth alone would pass with the flag never set, which is the
    /// half of the wire the browser actually reads.
    #[test]
    fn a_sub_floor_column_whose_ink_crosses_the_slice_is_grown_and_says_so() {
        let mut runs: Vec<(u32, u32)> = (0..7).map(|index| (4 + index * 50, 30)).collect();
        runs.push((354, 46));
        let image = column_page(200, 400, (60, 40), &runs);
        // Sub-floor, column-shaped, against the TOP band, ink reaching the bottom.
        let detections = vec![detection("text", 0.2295, [60.0, 2.0, 100.0, 210.0])];

        let (settled, hints) = settle_on(detections, &image, true);

        assert!(
            settled.is_empty(),
            "a sub-floor box must never be admitted as a region, grown or not: {settled:?}"
        );
        assert_eq!(
            hints,
            vec![EdgeHint {
                x: 60.0,
                y: 2.0,
                width: 40.0,
                // Grown from 208 to the page edge: 400 - 2.
                height: 398.0,
                edge: PageEdge::Top,
                score: 0.2295,
                spans: true,
            }]
        );
    }

    /// The other side of the same rule: ink that STOPS inside the slice leaves the
    /// hint exactly as the plain one-sided hint, unable to declare a crossing.
    ///
    /// Without this, "grow every sub-floor edge box" would pass the test above and
    /// invent a spanning column wherever the detector clipped a heading -- the
    /// failure already measured once, when a brightness probe
    /// fired on the white paper of manga and turned 0 manga joins into 22.
    #[test]
    fn a_sub_floor_column_whose_ink_stops_inside_the_slice_is_reported_unchanged() {
        // Ink only in the top third; the walk runs out long before the far band.
        let runs: Vec<(u32, u32)> = (0..3).map(|index| (4 + index * 40, 30)).collect();
        let image = column_page(200, 400, (60, 40), &runs);
        let detections = vec![detection("text", 0.2295, [60.0, 2.0, 100.0, 130.0])];

        let (settled, hints) = settle_on(detections, &image, true);

        assert!(settled.is_empty(), "still never a region: {settled:?}");
        assert_eq!(hints.len(), 1, "a refused edge box is still worth reporting");
        assert!(
            !hints[0].spans,
            "ink that stops inside the slice must not declare a crossing"
        );
        assert_eq!(
            (hints[0].y, hints[0].height),
            (2.0, 128.0),
            "an unverified hint keeps the geometry the detector drew"
        );
    }

    /// A sub-floor box in the middle of the page tells a joiner nothing, so it
    /// is dropped in silence -- exactly as it is today. Without this the hint
    /// channel would carry every doubtful detection on every page.
    #[test]
    fn a_sub_floor_box_away_from_every_edge_is_no_hint_at_all() {
        let image = RgbImage::from_pixel(1200, 908, Rgb([250, 250, 250]));
        let detections = vec![detection("text", 0.2363, [52.1, 400.0, 243.8, 591.6])];

        let (detections, hints) = settle_on(detections, &image, true);

        assert!(hints.is_empty(), "{hints:?}");
        assert!(detections.is_empty(), "{detections:?}");
    }

    /// **The manga guard, asserted where it actually lives.**
    ///
    /// It is the edge precondition and NOT the aspect rule: 1,221 boxes on
    /// one Japanese test volume are tall and narrow, because vertical Japanese is that
    /// shape. The column below is that shape, sits in the middle of the page,
    /// and has an unbroken run of ink from the top of the page to the bottom of
    /// it -- everything the walk looks for -- and it must still come back
    /// untouched and unmentioned. Measured over 213 manga pages and 2,337
    /// regions: 3 qualify on shape, 0 grow.
    ///
    /// The box is tall enough that its reach covers the rest of the page and the
    /// ink chain runs into the bottom band, so nothing but the edge precondition
    /// can be refusing it. Sized that way on measurement: a shorter box was
    /// refused by [`INK_WALK_REACH_RATIO`] instead, and the test stayed green
    /// when the precondition was deleted -- which is a test that pins nothing.
    #[test]
    fn a_tall_narrow_box_in_mid_page_is_neither_grown_nor_hinted() {
        let mut runs: Vec<(u32, u32)> = (0..34).map(|index| (index * 50, 25)).collect();
        runs.push((1690, 10));
        let image = column_page(1200, 1700, (600, 40), &runs);
        let detections = vec![detection("text", 0.9, [600.0, 300.0, 640.0, 1100.0])];

        let (detections, hints) = settle_on(detections, &image, true);

        assert!(hints.is_empty(), "{hints:?}");
        assert_eq!(
            detections[0].bbox,
            [600.0, 300.0, 640.0, 1100.0],
            "vertical Japanese is tall and narrow; shape alone must not grow it"
        );
    }

    /// A wide banner along the top edge is not a column, whatever its ink does.
    /// The aspect rule is the second half of the precondition and this is the
    /// half of it the edge test cannot cover. Sized so the reach and the ink
    /// chain would both carry it to the bottom band, for the reason the previous
    /// test gives.
    #[test]
    fn a_wide_band_along_an_edge_is_not_a_column() {
        let mut runs: Vec<(u32, u32)> = (0..8).map(|index| (index * 50, 20)).collect();
        runs.push((388, 12));
        let image = column_page(200, 400, (10, 180), &runs);
        let detections = vec![detection("text", 0.29, [10.0, 2.0, 190.0, 210.0])];

        let (detections, _) = settle_on(detections, &image, true);

        assert_eq!(detections[0].bbox, [10.0, 2.0, 190.0, 210.0]);
    }

    #[test]
    fn nms_removes_lower_scored_overlapping_regions_per_class() {
        let detections = vec![
            detection("text", 0.8, [5.0, 5.0, 105.0, 105.0]),
            detection("bubble", 0.7, [0.0, 0.0, 100.0, 100.0]),
            detection("text", 0.9, [0.0, 0.0, 100.0, 100.0]),
            detection("text", 0.5, [20.0, 20.0, 80.0, 80.0]),
            detection("text", 0.6, [200.0, 0.0, 250.0, 50.0]),
        ];

        let (detections, _) = settle(detections, true);

        let text_scores = detections
            .iter()
            .filter(|detection| detection.label == "text")
            .map(|detection| detection.score)
            .collect::<Vec<_>>();
        assert_eq!(text_scores, [0.9, 0.6]);
        assert!(
            detections
                .iter()
                .any(|detection| detection.label == "bubble")
        );
    }

    /// The same ink returned twice under two labels is one region, not two.
    ///
    /// Measured on one test page: four boxes came back as both `text`
    /// and `onomatopoeia` with byte-identical OCR, and both halves were
    /// lettered into the same rectangle. The pair below is a real one --
    /// a 26x93 box at (649,1021), scored 0.494 as text and 0.264 as
    /// an effect -- which is also why the assertion is that the TEXT survives
    /// here: it outscored the effect on this instance.
    #[test]
    fn the_same_ink_under_two_lettering_labels_is_suppressed_to_one() {
        let twins = || {
            vec![
                detection("text", 0.494, [649.0, 1021.0, 675.0, 1115.0]),
                detection("onomatopoeia", 0.264, [649.0, 1021.0, 675.0, 1115.0]),
            ]
        };

        let (detections, _) = settle(twins(), true);
        assert_eq!(detections.len(), 1, "the duplicate must not survive");
        assert_eq!(detections[0].label, "text");

        /* With effects off the two labels are no longer both lettered, so
         * suppressing across them would erase the words instead of translating
         * them. The rule must be inert here, and this is the assertion that
         * keeps it inert -- upstream's behaviour, op for op. */
        let (detections, _) = settle(twins(), false);
        assert_eq!(
            detections.len(),
            2,
            "with --no-translate-sfx a text twin must not be suppressed by an effect"
        );
    }

    /// A bubble and the text inside it overlap almost totally and must both
    /// survive, which is the reason the cross-class rule names the lettering
    /// labels rather than allowing any two labels to suppress each other.
    #[test]
    fn a_bubble_never_suppresses_the_text_inside_it() {
        let detections = vec![
            detection("bubble", 0.9, [0.0, 0.0, 100.0, 100.0]),
            detection("text", 0.4, [5.0, 5.0, 95.0, 95.0]),
            detection("panel", 0.8, [0.0, 0.0, 100.0, 100.0]),
        ];

        let (detections, _) = settle(detections, true);

        assert_eq!(detections.len(), 3, "{detections:?}");
    }

    #[test]
    fn text_mask_includes_only_onomatopoeia_contained_by_a_bubble() {
        let detection = |label: &str, bbox: [f32; 4], pixels: [u8; 4]| KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score: 1.0,
            bbox,
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                width: 4,
                height: 1,
                pixels: pixels.to_vec(),
            },
        };
        let detections = vec![
            detection("bubble", [0.0, 0.0, 3.0, 1.0], [0, 0, 0, 0]),
            detection("onomatopoeia", [0.0, 0.0, 1.0, 1.0], [255, 0, 0, 0]),
            detection("text", [1.0, 0.0, 2.0, 1.0], [0, 255, 0, 0]),
            detection("onomatopoeia", [3.0, 0.0, 4.0, 1.0], [0, 0, 0, 255]),
        ];

        let mask = mask_for(
            &detections,
            "text",
            ImageSize {
                width: 4,
                height: 1,
            },
            false,
            None,
            false,
        );

        assert_eq!(mask.as_raw(), &[255, 255, 0, 0]);
    }

    /// Lettering an effect makes erasing it unconditional, which is the half of
    /// the change that is easy to leave out: promote the class into the text
    /// path but keep the containment rule, and a free-standing effect gets
    /// English painted straight over the artist's kana.
    #[test]
    fn every_onomatopoeia_is_erased_once_effects_are_lettered() {
        let detection = |label: &str, bbox: [f32; 4], pixels: [u8; 4]| KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score: 1.0,
            bbox,
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                width: 4,
                height: 1,
                pixels: pixels.to_vec(),
            },
        };
        let detections = vec![
            detection("bubble", [0.0, 0.0, 3.0, 1.0], [0, 0, 0, 0]),
            detection("onomatopoeia", [0.0, 0.0, 1.0, 1.0], [255, 0, 0, 0]),
            detection("text", [1.0, 0.0, 2.0, 1.0], [0, 255, 0, 0]),
            // Outside every bubble: left as artwork before, erased now.
            detection("onomatopoeia", [3.0, 0.0, 4.0, 1.0], [0, 0, 0, 255]),
        ];

        let mask = mask_for(
            &detections,
            "text",
            ImageSize {
                width: 4,
                height: 1,
            },
            true,
            None,
            false,
        );

        assert_eq!(mask.as_raw(), &[255, 255, 0, 255]);
    }

    /// The erase mask and the OCR router move TOGETHER on a joined page.
    ///
    /// Lifting the guard for the router alone was measured on a joined slice
    /// pair and it is a regression the wrong way: the joined name is read and
    /// lettered while its artwork is spared, so English lands on top of intact
    /// Chinese. Worse for a reader than either the fabrication before it or the
    /// untranslated column after it. This asserts the mask half specifically,
    /// because that is the half that was missing.
    #[test]
    fn a_joined_page_erases_the_very_column_it_was_cut_to_hold() {
        // The joined seam: one column 1693px tall on a 1716px page, 0.987 --
        // past the 0.95 the fragment arm refuses at.
        let size = ImageSize {
            width: 1200,
            height: 1716,
        };
        let detections = vec![detection(TEXT, 0.9, [63.0, 10.0, 267.0, 1703.0])];

        // On an ordinary slice: spared, so a refused fragment keeps its artwork
        // rather than being stripped and lettered with nothing.
        assert!(
            !mask_includes(&detections, &detections[0], TEXT, true, size, Some(0.5), false),
            "an ordinary slice must still spare a fragment's artwork"
        );
        // On a page assembled to hold exactly that column: erased, because the
        // router is about to read it and letter the result into the hole.
        assert!(
            mask_includes(&detections, &detections[0], TEXT, true, size, Some(0.5), true),
            "a joined page must erase the column it was cut to hold"
        );
    }

    /// The erase mask is written from raw detections before OCR runs, so a box
    /// the reader refuses is still erased and inpainted. This is the arm that
    /// stops that -- a measured 1210x1247 `onomatopoeia` on an 844x1200 page is
    /// a whole artwork panel, and it is 1.49x the area of the page it was found
    /// on, which no text region can be.
    ///
    /// Scaled to a 4x1 test page rather than run at 844x1200: the rule is a
    /// ratio, so a box 1.49x of a 4-pixel page tests exactly the same predicate,
    /// and the neighbouring ordinary detection has to survive it.
    #[test]
    fn an_implausible_region_contributes_nothing_to_the_erase_mask() {
        let detection = |label: &str, bbox: [f32; 4], pixels: [u8; 4]| KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score: 0.43,
            bbox,
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                width: 4,
                height: 1,
                pixels: pixels.to_vec(),
            },
        };
        let size = ImageSize {
            width: 4,
            height: 1,
        };
        let detections = vec![
            // 6.0 x 1.0 on a 4 x 1 page: 1.5x the page's own area, and it starts
            // off the left edge exactly as the real one does at x = -185.
            detection("onomatopoeia", [-1.0, 0.0, 5.0, 1.0], [255, 255, 255, 0]),
            detection("text", [3.0, 0.0, 4.0, 1.0], [0, 0, 0, 255]),
        ];

        // Off: today's behaviour, the panel is erased along with the text.
        assert_eq!(
            mask_for(&detections, "text", size, true, None, false).as_raw(),
            &[255, 255, 255, 255]
        );

        // On: only the ordinary region survives.
        assert_eq!(
            mask_for(&detections, "text", size, true, Some(0.5), false).as_raw(),
            &[0, 0, 0, 255]
        );
    }

    /// **The reader and the eraser are handed DIFFERENT BOXES, and this is the
    /// assertion that says so out loud.**
    ///
    /// `stages/mod.rs`'s `implausible_region` is deliberately shared by both
    /// stages, and its own doc comment says why: *"one copy that drifts would let
    /// a box be refused by the reader and erased by the mask, or the reverse"*.
    /// **The predicate never drifted. Its ARGUMENT did.**
    ///
    /// - The **eraser** calls it from `mask_includes` on `detection.bbox` raw --
    ///   `(right - left, bottom - top)`.
    /// - The **reader** calls it from `ocr.rs`'s `region_extent`, which measures
    ///   `geometry_extents(target.geometry)`. For a lettering region that geometry
    ///   is `rotated_rectangle_geometry(bbox, angle)` -- the same rectangle turned
    ///   about its centre and re-boxed. It is also exactly what reaches the wire,
    ///   via `birelate-server`'s `axis_aligned_bounds`.
    ///
    /// Rotating a rectangle about its centre can only **inflate** its axis-aligned
    /// hull (`W|cos| + H|sin|` by `W|sin| + H|cos|`, equal only at multiples of
    /// 90 degrees), so the reader always sees a box at least as large as the
    /// eraser does -- and between the two lies a band where **the reader refuses
    /// and the eraser erases anyway**.
    ///
    /// Measured consequence over a full manga baseline run: 23 regions
    /// were refused by the reader as too large, and **19 of them were erased
    /// regardless** -- artwork inpainted away with nothing painted back, 4.8% to
    /// 47.4% of the pixels inside the box, 11 of them on pages carrying no other
    /// region at all. The four the eraser *did* spare are the four with the
    /// largest reported boxes, which is what this divergence predicts.
    ///
    /// **Asserted on `mask_includes` -- the predicate the eraser actually calls --
    /// and not on `implausible_region` and `rotated_rectangle_geometry`
    /// separately, because the defect lives BETWEEN them.** Testing the two halves
    /// is precisely how a fix once shipped with 128 tests green and no behaviour.
    #[test]
    fn a_rotated_region_the_reader_refuses_for_size_is_erased_anyway() {
        let size = ImageSize {
            width: 100,
            height: 100,
        };
        let page_area = f64::from(size.width) * f64::from(size.height);

        // 70 x 64 = 4,480 px on a 100 x 100 page: **0.448**, comfortably under the
        // shipping 0.5 ceiling, so nothing about this box is implausible on its
        // own terms.
        let bbox = rect(15.0, 18.0, 70.0, 64.0);
        // Well clear of the snap floor, so the rotation survives to the geometry.
        // Under it the two boxes are identical and there is no divergence at all,
        // which is why the oversized-panel case -- horizontal, angle 0 -- is one of the four
        // the eraser spares.
        let angle = 25.0_f32;
        assert!(angle > ANGLE_SNAP_DEGREES);

        let raw = (
            f64::from(bbox[2] - bbox[0]),
            f64::from(bbox[3] - bbox[1]),
        );
        assert!((raw.0 * raw.1 / page_area - 0.448).abs() < 1e-9);

        // THE ERASER. `mask_includes` measures the raw box, finds 0.448, and
        // includes the region in the erase mask.
        let detections = vec![detection("onomatopoeia", 0.43, bbox)];
        assert!(
            mask_includes(&detections, &detections[0], TEXT, true, size, Some(0.5), false),
            "the eraser measures the raw 0.448 box and erases it"
        );

        // THE READER. The same rectangle, rotated and re-boxed, is what `ocr.rs`
        // measures and what the JSON reports.
        let geometry = rotated_rectangle_geometry(bbox, angle);
        let (min_x, min_y, max_x, max_y) =
            crate::scope::geometry_extents(&geometry).expect("a rotated rectangle has extents");
        let reader = (max_x - min_x, max_y - min_y);
        assert!(
            crate::stages::implausible_region(reader, (size.width, size.height), Some(0.5)),
            "the reader measures the rotated hull and refuses to read it"
        );

        // The whole defect in one line: one box, two stages, opposite verdicts.
        // The inflation is what carries it across the ceiling.
        assert!(reader.0 * reader.1 > raw.0 * raw.1);
        assert!(reader.0 * reader.1 / page_area >= 0.5);
    }

    /// `mask_scale` replaces `mask_for` outright and `ink_mask` unions pixels
    /// back in after it, so a gate written into `mask_for` alone would be
    /// bypassed by the first flag and undone by the second. Both go through
    /// `mask_includes`; this is the assertion that keeps them there.
    #[test]
    fn the_scaled_mask_refuses_the_same_implausible_region() {
        let detection = |label: &str, bbox: [f32; 4], pixels: [u8; 4]| KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score: 0.43,
            bbox,
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                width: 4,
                height: 1,
                pixels: pixels.to_vec(),
            },
        };
        let size = ImageSize {
            width: 4,
            height: 1,
        };
        let detections = vec![detection(
            "onomatopoeia",
            [-1.0, 0.0, 5.0, 1.0],
            [255, 255, 255, 0],
        )];

        assert!(
            scaled_dilated_mask(&detections, "text", size, true, 1.37, None, false)
                .as_raw()
                .iter()
                .any(|value| *value != 0)
        );
        assert!(
            scaled_dilated_mask(&detections, "text", size, true, 1.37, Some(0.5), false)
                .as_raw()
                .iter()
                .all(|value| *value == 0)
        );
    }

    /// The point of the scaled rule: growth follows the REGION, where the flat
    /// radius follows the page and so means a different scale for every region.
    #[test]
    fn a_scaled_mask_grows_small_and_large_regions_proportionally() {
        let detection = |bbox: [f32; 4], pixels: Vec<u8>| KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox,
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                width: 200,
                height: 40,
                pixels,
            },
        };
        // Two regions on one page: a 20px-tall one and a 40px-tall one.
        let mut small = vec![0u8; 200 * 40];
        let mut large = vec![0u8; 200 * 40];
        for y in 10..30 {
            small[y * 200 + 20] = 255; // 20px tall
        }
        for y in 0..40 {
            large[y * 200 + 150] = 255; // 40px tall
        }
        let detections = vec![
            detection([15.0, 10.0, 25.0, 30.0], small),
            detection([145.0, 0.0, 155.0, 40.0], large),
        ];

        let size = ImageSize {
            width: 200,
            height: 40,
        };
        let scaled = scaled_dilated_mask(&detections, "text", size, false, 1.37, None, false);

        // radius = (scale - 1) / 2 * shorter side. Both bboxes are 10 wide, so
        // the shorter side is 10 and both get the same 2px -- the scale is
        // constant, which is exactly the property the flat rule lacks.
        let width_at = |row: u32, centre: u32| {
            (0..200)
                .filter(|x| scaled.get_pixel(*x, row)[0] != 0)
                .filter(|x| x.abs_diff(centre) < 30)
                .count()
        };
        assert_eq!(width_at(20, 20), 5, "small region: 1px ink + 2px each side");
        assert_eq!(width_at(20, 150), 5, "large region: same scale, same growth");

        // And a bigger scale really does grow further.
        let wider = scaled_dilated_mask(&detections, "text", size, false, 2.0, None, false);
        assert!(
            (0..200)
                .filter(|x| wider.get_pixel(*x, 20)[0] != 0)
                .filter(|x| x.abs_diff(20) < 30)
                .count()
                > 5
        );
    }

    /// The whole sound-effect decision, in one predicate. Bubbles and panels
    /// must never letter in either arm -- a bubble that became a text region
    /// would be handed to OCR as a balloon-sized crop.
    #[test]
    fn only_text_and_enabled_effects_letter() {
        assert!(letters_text("text", false));
        assert!(letters_text("text", true));
        assert!(!letters_text("onomatopoeia", false));
        assert!(letters_text("onomatopoeia", true));
        for translate_sfx in [false, true] {
            assert!(!letters_text("bubble", translate_sfx));
            assert!(!letters_text("panel", translate_sfx));
            assert!(!letters_text("", translate_sfx));
        }
    }

    /// The line that actually puts an effect in front of the OCR model:
    /// `stages/ocr.rs` selects on `TextRegion`, so a sound effect that keeps the
    /// unknown kind is detected, erased and never read.
    #[test]
    fn a_lettered_onomatopoeia_becomes_a_text_region() {
        let text = TextRegion::kind();
        assert_eq!(region_kind("onomatopoeia", true).unwrap(), text);
        assert_eq!(region_kind("text", true).unwrap(), text);
        assert_eq!(region_kind("text", false).unwrap(), text);
        // Off, it stays the class nothing consumes.
        assert_ne!(region_kind("onomatopoeia", false).unwrap(), text);
        // Never a bubble or a panel, in either arm.
        for translate_sfx in [false, true] {
            assert_ne!(region_kind("bubble", translate_sfx).unwrap(), text);
            assert_ne!(region_kind("panel", translate_sfx).unwrap(), text);
        }
    }

    #[test]
    fn layout_order_follows_panels_then_bubbles_then_their_text() {
        let detections = vec![
            detection("text", 0.63, [20.0, 30.0, 70.0, 70.0]),
            detection("bubble", 0.7, [10.0, 20.0, 80.0, 80.0]),
            detection("panel", 0.9, [100.0, 0.0, 200.0, 200.0]),
            detection("text", 0.62, [130.0, 110.0, 180.0, 150.0]),
            detection("bubble", 0.8, [120.0, 20.0, 190.0, 80.0]),
            detection("panel", 0.9, [0.0, 0.0, 95.0, 200.0]),
            detection("text", 0.61, [130.0, 30.0, 180.0, 70.0]),
            detection("bubble", 0.7, [120.0, 100.0, 190.0, 160.0]),
        ];

        let text_scores = layout_order(&detections)
            .into_iter()
            .filter_map(|index| {
                (detections[index].label == "text").then_some(detections[index].score)
            })
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.61, 0.62, 0.63]);
    }

    #[test]
    fn typography_comes_from_horizontal_text_mask() {
        let (image, detection) = masked_text(52.0, 12.0, 12.0, [24, 80, 160]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert!((inferred.angle_degrees - 12.0).abs() < 1.0);
        assert!((11.0..=14.0).contains(&inferred.font_size));
        assert_eq!(inferred.color, [24, 80, 160]);
        // The colour the glyphs are painted on, kept rather than discarded --
        // `masked_text` fills the page with it outside the mask.
        assert_eq!(inferred.background, Some([200, 180, 160]));
        assert_eq!(inferred.writing_mode, WritingMode::Horizontal);
    }

    #[test]
    fn vertical_text_angle_is_relative_to_upright_vertical() {
        let (image, detection) = masked_text(12.0, 52.0, 9.0, [120, 80, 40]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert!((inferred.angle_degrees - 9.0).abs() < 1.0);
        assert!((11.0..=14.0).contains(&inferred.font_size));
        assert_eq!(inferred.writing_mode, WritingMode::Vertical);
    }

    #[test]
    fn tall_multiline_mask_uses_text_lines_instead_of_block_aspect() {
        let width = 96;
        let height = 96;
        let angle_degrees = 8.0_f64;
        let (sin, cos) = angle_degrees.to_radians().sin_cos();
        let mut image = RgbImage::from_pixel(width, height, Rgb([240, 240, 240]));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in 0..height {
            for x in 0..width {
                let dx = f64::from(x) + 0.5 - f64::from(width) * 0.5;
                let dy = f64::from(y) + 0.5 - f64::from(height) * 0.5;
                let local_x = dx * cos + dy * sin;
                let local_y = -dx * sin + dy * cos;
                let inside = [-24.0, 0.0, 24.0]
                    .into_iter()
                    .any(|line_y| local_x.abs() <= 15.0 && (local_y - line_y).abs() <= 2.0);
                if inside {
                    pixels[y as usize * width as usize + x as usize] = u8::MAX;
                    image.put_pixel(x, y, Rgb([8, 8, 8]));
                }
            }
        }
        let detection = KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox: [0.0, 0.0, width as f32, height as f32],
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                width,
                height,
                pixels,
            },
        };

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.writing_mode, WritingMode::Horizontal);
        assert!((inferred.angle_degrees - 8.0).abs() < 1.0);
    }

    #[test]
    fn antialiased_dark_text_uses_the_high_contrast_core() {
        let (mut image, detection) = masked_text(52.0, 12.0, 0.0, [96, 94, 92]);
        for (index, &mask) in detection.mask.pixels.iter().enumerate() {
            if mask != 0 && index.is_multiple_of(3) {
                let x = index as u32 % image.width();
                let y = index as u32 / image.width();
                image.put_pixel(x, y, Rgb([8, 9, 7]));
            }
        }

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
    }

    /// A REGION-shaped mask, which is what this detector actually returns --
    /// the fact `infer_text_color`'s doc comment records, and the fact a
    /// mask-eroding `infer_ink_core` would miss (eroding the mask samples the
    /// paper). A 60x24 blob strip painted `paper`, on a
    /// `page`-coloured panel, carrying one 40x6 ink bar with a 2px `rim` ring.
    fn blob_text(
        page: [u8; 3],
        paper: [u8; 3],
        rim: [u8; 3],
        ink: [u8; 3],
    ) -> (RgbImage, KoharuLayoutDetection) {
        let width = 96u32;
        let height = 96u32;
        let mut image = RgbImage::from_pixel(width, height, Rgb(page));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in 36u32..60 {
            for x in 18u32..78 {
                let color = if (28..68).contains(&x) && (45..51).contains(&y) {
                    ink
                } else if (26..70).contains(&x) && (43..53).contains(&y) {
                    rim
                } else {
                    paper
                };
                image.put_pixel(x, y, Rgb(color));
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
            }
        }
        let detection = KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox: [0.0, 0.0, width as f32, height as f32],
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                width,
                height,
                pixels,
            },
        };
        (image, detection)
    }

    /// In the shape a measured scream has it: dark red brush ink on
    /// a light banner, blob-masked, on a panel of the ink's own family. The
    /// contrast ranking keeps the quartile FARTHEST from the background median,
    /// and against a red panel that is the light banner -- so the legacy sample
    /// comes back near-white and the light snap makes it pure white, which is
    /// the wire colour the real region shipped (`[255,255,255,255]`, under the
    /// contrast ranking and again under a mask-eroding sampler). The
    /// paper-then-ink sample cannot be
    /// captured either way: the paper is the blob's own eroded interior, and
    /// the ink is the deepest thing that differs from it. BOTH halves are
    /// pinned: if the legacy assertion ever fails, the ranking heuristic
    /// changed and the override may no longer be needed; if the ink assertion
    /// fails, the fix regressed.
    #[test]
    fn a_light_banner_inverts_the_contrast_sample_and_the_paper_then_ink_core_does_not() {
        let (image, detection) =
            blob_text([150, 60, 60], [220, 220, 220], [139, 136, 136], [94, 45, 45]);

        let inferred = infer_typography(&image, &detection).unwrap();

        // The defect, reproduced: the banner is ~70% of the mask and the
        // farthest thing from the red panel, so the quartile is pure banner and
        // (220,220,220) snaps to pure white.
        assert_eq!(inferred.color, [255, 255, 255]);
        // The fix: paper = the blob's interior median (the banner); what
        // differs from it is the ink bar and its rim; the deepest of THAT is
        // the ink, and chroma 49 survives the snap.
        assert_eq!(inferred.ink_color, Some([94, 45, 45]));
        // A 6px bar in a 2px rim reads heavy against this blob's measured line.
        let ratio = inferred.ink_stroke_ratio.expect("a sampled core carries a ratio");
        assert!(ratio >= HEAVY_INK_RATIO, "the brush bar must read heavy, got {ratio}");
    }

    /// The property that keeps this build a no-op on the ordinary population:
    /// dark glyphs blob-masked over light paper on an ordinary page. The
    /// ranking and the paper-then-ink core agree, so the override rewrites the
    /// fill to the value it already had.
    #[test]
    fn the_ink_core_matches_the_legacy_sample_on_ordinary_dark_text() {
        let (image, detection) =
            blob_text([200, 180, 160], [250, 250, 250], [120, 120, 120], [10, 10, 10]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
        assert_eq!(inferred.ink_color, Some([0, 0, 0]));
    }

    /// The inversion a render caught: dense black text
    /// blob-masked on white, where the mask's eroded interior is majority INK
    /// -- so the split calls the glyphs "paper" and the white gaps "ink", and
    /// without the polarity guard the page letters hollow white-on-white.
    /// A differing core LIGHTER than the
    /// paper abstains, and the legacy sample -- which reads this page
    /// correctly -- stands. The same guard is what keeps genuine inverted
    /// narration out of the override's reach.
    #[test]
    fn a_light_differing_core_is_not_ink_and_the_sample_abstains() {
        // Roles deliberately swapped against blob_text's naming: the blob's
        // bulk is near-black glyph ink, and the small bar is the white gap.
        let (image, detection) =
            blob_text([245, 245, 245], [15, 15, 15], [140, 140, 140], [250, 250, 250]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.ink_color, None, "a light core over dark paper is a swapped split");
        assert_eq!(inferred.ink_stroke_ratio, None);
    }

    /// The muddiness a render caught: ordinary-sized
    /// text whose strokes erode to a one-pixel centerline of anti-aliased
    /// blend, shipping grey where legacy ships crisp ink. Measured ink thinner than
    /// `INK_SAMPLE_MINIMUM_THICKNESS_PX` abstains.
    #[test]
    fn thin_strokes_abstain_rather_than_sampling_the_blend() {
        // blob_text's geometry with the ink bar thinned to a 2px stroke in a
        // 1px rim -- a 4px differing blob, well under the floor.
        let width = 96u32;
        let height = 96u32;
        let mut image = RgbImage::from_pixel(width, height, Rgb([200, 180, 160]));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in 36u32..60 {
            for x in 18u32..78 {
                let color = if (28..68).contains(&x) && (47..49).contains(&y) {
                    [10, 10, 10]
                } else if (27..69).contains(&x) && (46..50).contains(&y) {
                    [120, 120, 120]
                } else {
                    [250, 250, 250]
                };
                image.put_pixel(x, y, Rgb(color));
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
            }
        }
        let detection = KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox: [0.0, 0.0, width as f32, height as f32],
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                width,
                height,
                pixels,
            },
        };

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.ink_color, None, "a 4px blob erodes to a blend, not ink");
        assert_eq!(inferred.ink_stroke_ratio, None);
    }

    /// A mask that is ALL ink -- a synthesised region's mask is its ink --
    /// finds nothing differing from its own paper and abstains, and the
    /// legacy sample stays in force. Nothing is lost where the split cannot be
    /// read.
    #[test]
    fn an_all_ink_mask_abstains_and_the_legacy_sample_stands() {
        let (image, detection) = masked_text(52.0, 12.0, 0.0, [10, 10, 10]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.ink_color, None);
        assert_eq!(inferred.ink_stroke_ratio, None);
        assert_eq!(inferred.color, [0, 0, 0]);
    }

    /// A hairline mask erodes to nothing at the PAPER stage too, and the
    /// whole-mask fallback median is the ink itself -- so nothing differs and
    /// the sample abstains rather than guessing.
    #[test]
    fn a_hairline_mask_declines_to_sample_ink() {
        let (image, detection) = masked_text(52.0, 2.0, 0.0, [10, 10, 10]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.ink_color, None);
        assert_eq!(inferred.ink_stroke_ratio, None);
        assert_eq!(inferred.color, [0, 0, 0]);
    }

    #[test]
    fn near_neutral_extremes_snap_to_full_black_or_white() {
        assert_eq!(normalize_text_color([20, 31, 24]), [0, 0, 0]);
        assert_eq!(normalize_text_color([230, 240, 250]), [255, 255, 255]);
        assert_eq!(normalize_text_color([33, 33, 33]), [0, 0, 0]);
        assert_eq!(normalize_text_color([205, 210, 216]), [255, 255, 255]);
        assert_eq!(normalize_text_color([40, 70, 40]), [40, 70, 40]);
    }

    /// **The snap DOES flatten a recognisable colour, and that is a trade rather
    /// than a bug -- so it is pinned here where it is visible.**
    ///
    /// The concern is that it snaps to black at chroma <= 24, flattening
    /// low-saturation dark tints (muted brown, navy). Stated that way
    /// it sounds unbounded. It is not: the predicate is chroma <= 24 **and**
    /// luminance <= 64 together, so it can only reach colours that are near-grey
    /// AND very dark. Enumerated over the whole RGB cube, that is
    /// **102,965 of 16,777,216 colours -- 0.614%**.
    ///
    /// But the corner of that set is real. `[77, 64, 53]` is `#4D4035`, a dark warm
    /// brown, sitting on both caps at once, and it goes to pure black. Each of the
    /// three colours below is one step over a different edge, so this fails if
    /// either constant moves in either direction.
    ///
    /// Whether flattening it is *right* is an open product question. This test
    /// only stops the behaviour changing silently -- the cases above are all grey.
    #[test]
    fn the_darkest_muted_colour_is_flattened_and_its_neighbours_are_not() {
        // On both caps: chroma exactly 24, luminance exactly 64.
        assert_eq!(normalize_text_color([77, 64, 53]), [0, 0, 0]);
        // One step over on CHROMA alone (25) -- kept.
        assert_eq!(normalize_text_color([78, 64, 53]), [78, 64, 53]);
        // One step over on LUMINANCE alone (65), chroma still 24 -- kept.
        assert_eq!(normalize_text_color([78, 65, 54]), [78, 65, 54]);
    }
}

