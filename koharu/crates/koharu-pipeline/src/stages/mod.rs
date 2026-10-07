mod detection;
mod inpainting;
mod ocr;
mod translation;

use std::{collections::BTreeSet, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use koharu_scene::{Edit, EntityId, Generation, Patch, ProducerId, Snapshot};
use koharu_translator::Language;

pub use detection::KoharuLayoutRFDetrSeg2XLConfig;
pub use inpainting::{Flux2KleinConfig, RoremMixedConfig};

use crate::{
    Bounds, ImageCache, InpaintingMask, ModelCell, PipelineConfig, Progress, ProgressSink, Stage,
    progress,
};

#[derive(Clone)]
pub(crate) struct StageInput {
    scene: koharu_scene::Snapshot,
    page: EntityId,
    entities: Option<Arc<BTreeSet<EntityId>>>,
    region: Option<Bounds>,
    images: Arc<ImageCache>,
    inpainting_mask: Option<InpaintingMask>,
    /// Prior translations to offer the model as precedent. Empty for every
    /// stage but translation, and empty there too unless the caller supplied it.
    context: Arc<[koharu_translator::TranslationContext]>,
    /// Pinned per-series term renderings. Same carriage as `context`, read only
    /// by the translation stage. See `Request::glossary`.
    glossary: Arc<[koharu_translator::TranslationContext]>,
    /// The caller's sampler seed for this run's translation stage. `None`
    /// keeps the config's own generation, seed included -- the fixed upstream
    /// constant on the shipping server. See `Request::translation_seed`.
    translation_seed: Option<u32>,
    /// The reader's own boxes and deletions for this run, read only by the
    /// detection stage. See `Request::added_regions` / `removed_regions`.
    added_regions: Arc<[crate::CallerRegion]>,
    removed_regions: Arc<[crate::CallerRegion]>,
    /// This page was assembled so its text does not run past its own edges.
    /// See `Request::joined_page`; read only by the OCR stage's size guard.
    joined_page: bool,
    /// See `Request::joined_boundaries`; read only by the spot rescue's
    /// composite gate.
    joined_boundaries: Arc<[f64]>,
    /// The run's sink, so a processor can report something the `Patch` it
    /// returns has no way to express. `StageRunner` emits the lifecycle events
    /// around a stage; this is for what only the stage itself can see.
    progress: Option<ProgressSink>,
}

impl StageInput {
    pub(crate) fn new(
        scene: Snapshot,
        page: EntityId,
        entities: Option<Arc<BTreeSet<EntityId>>>,
        region: Option<Bounds>,
        images: Arc<ImageCache>,
        inpainting_mask: Option<InpaintingMask>,
        context: Arc<[koharu_translator::TranslationContext]>,
        glossary: Arc<[koharu_translator::TranslationContext]>,
        translation_seed: Option<u32>,
        added_regions: Arc<[crate::CallerRegion]>,
        removed_regions: Arc<[crate::CallerRegion]>,
        joined_page: bool,
        joined_boundaries: Arc<[f64]>,
        progress: Option<ProgressSink>,
    ) -> Self {
        Self {
            scene,
            page,
            entities,
            region,
            images,
            inpainting_mask,
            context,
            glossary,
            translation_seed,
            added_regions,
            removed_regions,
            joined_page,
            joined_boundaries,
            progress,
        }
    }

    fn report(&self, event: Progress) {
        progress::emit(self.progress.as_ref(), event);
    }

    pub(crate) fn page(&self) -> EntityId {
        self.page
    }

    pub(crate) fn translation_context(&self) -> &[koharu_translator::TranslationContext] {
        &self.context
    }

    pub(crate) fn translation_glossary(&self) -> &[koharu_translator::TranslationContext] {
        &self.glossary
    }

    /// The caller's per-run sampler seed, `None` for every caller that did not
    /// ask for a re-roll. See `Request::translation_seed` for why this is not
    /// a config field.
    pub(crate) fn translation_seed(&self) -> Option<u32> {
        self.translation_seed
    }

    /// Boxes the reader drew over text the detector missed. Empty for every
    /// caller but the box editor. See `Request::added_regions`.
    pub(crate) fn added_regions(&self) -> &[crate::CallerRegion] {
        &self.added_regions
    }

    /// Rectangles the reader marked for deletion. See
    /// `Request::removed_regions`.
    pub(crate) fn removed_regions(&self) -> &[crate::CallerRegion] {
        &self.removed_regions
    }

    /// Whether the caller assembled this page so its text ends inside it.
    /// See `Request::joined_page` for why this is not a config field.
    pub(crate) fn joined_page(&self) -> bool {
        self.joined_page
    }

    /// Where the cuts sit inside a joined page. Empty when the caller did not
    /// say, and every rule reading it must treat empty as "do not fire".
    pub(crate) fn joined_boundaries(&self) -> &[f64] {
        &self.joined_boundaries
    }

    fn contains_entity(&self, entity: EntityId) -> Result<bool> {
        crate::scope::contains_entity(
            &self.scene,
            self.page,
            self.entities.as_deref(),
            self.region,
            entity,
        )
    }
}

/// Whether a detection box is too large to be a text region on this page.
///
/// **This is a fourth discriminator for the mis-segmented-artwork class, and it
/// is a different kind of thing from the three already rejected.** Ink density,
/// detection confidence and text-segmenter fraction all failed because they ask
/// *what is inside the box* -- and the measured cases are a playing card whose
/// ink is denser than the clear win, two boxes 0.009 apart in confidence, and a
/// segmenter that finds 31% text in the bad box and 0% in the good one. This
/// asks nothing about the content. A box covering 1.49x the area of the page it
/// was found on is a **detector error by construction**: no text region can be
/// larger than the image it was detected in, whatever it contains. The rule is
/// geometry against geometry, so there is no content for it to be wrong about.
///
/// Below 1.0 it stops being an impossibility and becomes a plausibility bound,
/// which is why the number is a flag. The separation is wide: of the 24 routed
/// regions in the run that latched, the largest legitimate one is **0.30x** of
/// the page and the outlier is **1.49x**, with nothing in between.
///
/// **A fraction of the page, not an absolute pixel count.** Pages in one volume
/// are one size and pages across sources are not -- the manga volume is
/// 844x1200 and the dense fixture is 1492x1118 -- so a pixel count would fire
/// at different severities on each, while the whole claim being made ("this box
/// is most of its page") is a ratio.
///
/// The box's own extent is measured rather than the crop's, and the difference
/// matters twice. `crop` clamps to the source image, so a crop can never exceed
/// 1.0 and the 1.49 that identified this defect would be unreadable in the rule
/// meant to catch it. And a clamped crop makes the verdict depend on *where* a
/// box lands: the identical bad box would be refused mid-page and accepted in a
/// corner. Since the crop is always contained by the box, measuring the box is
/// also strictly the more protective of the two.
///
/// Free-standing rather than a method so it is testable without a `Device`.
///
/// **It lives here rather than in `ocr.rs` because two stages now ask it.** The
/// OCR router asks whether to *read* the box; the detection stage asks whether
/// to *erase* it. Those are separate decisions with separate flags, but they
/// must be the same test -- one copy that drifts would let a box be refused by
/// the reader and erased by the mask, or the reverse, and the pair of arms would
/// no longer describe a single phenomenon.
fn implausible_region(region: (f64, f64), page: (u32, u32), max_area: Option<f32>) -> bool {
    let Some(max_area) = max_area else {
        return false;
    };
    // Written as a negated `>` so NaN takes the same branch as zero: an
    // unusable fraction reads as "off" rather than being clamped into meaning
    // something the caller did not ask for.
    if !(max_area > 0.0) {
        return false;
    }
    let page_area = f64::from(page.0) * f64::from(page.1);
    if page_area <= 0.0 {
        return false;
    }
    region.0.max(0.0) * region.1.max(0.0) >= page_area * f64::from(max_area)
}

/// The share of a slice's HEIGHT at which a region stops being a piece of text
/// and starts being a piece of a piece of text.
///
/// A webtoon chapter is delivered as slices, and the host cuts them wherever it
/// likes. A region that reaches both the top and the bottom of its slice is
/// therefore not a short thing that happens to be tall -- it is the middle of
/// something that continues off both ends, and no amount of reading the slice can
/// recover the rest.
///
/// **Measured on a 241-region Chinese test chapter, and the threshold sits on a
/// plateau rather than a cliff edge.** Every value from **0.90 to 0.98** selects
/// exactly the same four lettered regions, and the next candidate below them is
/// at **0.886**. 0.95 is the middle of that plateau, so the rule
/// does not depend on where in it the number is put.
///
/// Position is deliberately not consulted. A box taller than 0.95 of its page
/// cannot avoid reaching within 0.05 of both edges, so the height share already
/// carries "touches both cuts" without needing an origin -- which matters because
/// a rotated box reports a hull whose origin can be negative.
const CROSS_SLICE_HEIGHT_SHARE: f32 = 0.95;

/// The shortest page on which a height share means anything. See the floor's own
/// comment in `cross_slice_fragment` -- an area ratio survives a 4x1 test page and
/// a height ratio does not.
const CROSS_SLICE_MIN_PAGE_HEIGHT: u32 = 128;

/// Is this region a fragment of text cut by the slice boundary above AND below?
///
/// **This exists because the failure it prevents is FABRICATION, not loss.** A
/// fragment is short, meaningless and stylised, and the translator is
/// grammar-constrained to return something for it, so it returns something
/// confident and wrong. Measured on a Chinese test chapter, each with a control
/// elsewhere in the same chapter that was not cut:
///
/// | cut fragment | lettered as | the same name, uncut |
/// |---|---|---|
/// | `霜真诀` | `Shuang Zhenjue` | `冰霜真诀` -> `Frost True Art` |
/// | `迎接之王` | `Lord of Ashes` | `裂风之王！` -> `King Gale!` |
/// | `1号云守！` | `No. 1: Kumomori!` | -- |
///
/// Three fragments, three wrong names, two of them contradicting the chapter's own
/// translation of the identical characters. A reader cannot detect any of it: it
/// is fluent English in the right place, which is worse than leaving the Chinese.
///
/// **So the refusal is the point, and it must happen HERE rather than downstream.**
/// The erase mask is written during detection; a refusal that arrives after it
/// would remove the artwork and letter nothing, leaving a blank column -- the
/// "right and too late" failure `illegible_text` and `watermark_text` both
/// describe. Refused here, the original glyphs stay on the page untouched, which
/// is the honest outcome until the seam can span a run of slices.
///
/// Free-standing and taking only the extent, for the same reason
/// `implausible_region` is: testable without a `Device`, and asked by both the
/// reader and the mask.
fn cross_slice_fragment(region: (f64, f64), page: (u32, u32)) -> bool {
    // **Unlike `implausible_region`, this rule is NOT scale-free, and the
    // difference is load-bearing.** An area ratio tests the same predicate on a
    // 4x1 page as on 844x1200, which is why the erase-mask fixtures next door are
    // deliberately one pixel tall. A HEIGHT ratio degenerates there: on a 1px page
    // every region spans 100% of it, so without this floor the rule would refuse
    // every detection in every one of those fixtures -- and it did, which is how
    // the floor was found.
    //
    // The claim being made is "this is a piece of text the host's cut ran
    // through", and that needs a page big enough for a cut line of text to exist.
    // Slices in these corpora are 900-1300 px tall and the shortest plausible
    // webtoon slice is in the low hundreds, so 128 sits an order of magnitude
    // below real material and clear of every synthetic page in the suite.
    if page.1 < CROSS_SLICE_MIN_PAGE_HEIGHT {
        return false;
    }
    let page_height = f64::from(page.1);
    if page_height <= 0.0 {
        return false;
    }
    region.1.max(0.0) >= page_height * f64::from(CROSS_SLICE_HEIGHT_SHARE)
}

/// Whether a region is refused before any engine reads it.
///
/// **Named rather than spelled inline, because a test of the two halves does not
/// test the `||` between them.** `withdraw_from_mask` in `ocr.rs` carries the same
/// warning for the same reason, and it is not hypothetical there: both its arms
/// had passing tests while the `||` was deleted and all 128 stayed green. Assert
/// on **this** function, not on `implausible_region` and `cross_slice_fragment`
/// separately.
///
/// The two arms refuse for unrelated reasons -- one box is too big to be a line of
/// text, the other is a slice-sized piece of a longer one -- but they earn the
/// same treatment at both call sites: do not read it, and do not erase it.
fn refuse_before_reading(region: (f64, f64), page: (u32, u32), max_area: Option<f32>) -> bool {
    implausible_region(region, page, max_area) || cross_slice_fragment(region, page)
}

/// Whether this region is refused before reading, ON A PAGE OF THIS KIND.
///
/// **This is the predicate the OCR router actually calls, and the reason it has a
/// name is the same reason `refuse_before_reading` above has one:** a test of the
/// two branches does not test the choice between them. Written inline at the call
/// site, the `joined_page` arm would be a conditional nobody asserts on, and the
/// failure mode is silent -- every seam would still be refused and the suite
/// would still be green, which is exactly the state this project has shipped
/// before.
///
/// `cross_slice_fragment` asks "is this a slice-sized piece of something taller?"
/// On an ordinary slice the answer earns a refusal: the translator fabricates
/// from a fragment, and one became `No. 1: Kumomori!`. On a page
/// the caller ASSEMBLED from a run of slices the same shape means the opposite --
/// the text reaches both edges because the crop was cut to hold exactly it, and
/// measured there the joined column is 0.987 of the seam against a 0.95
/// threshold. So the fragment arm is dropped for a joined page -- which is
/// exactly what `Request::joined_page` says the flag exists to do -- and the
/// configured area ceiling applies here as everywhere else.
///
/// **The area arm is deliberately NOT raised on joined pages.** Raising it to 1.0
/// once looked right: a refused joined box lettered NOTHING, so a successful
/// join was discarded and the reader kept the two wrong halves. The upright pass
/// removed that premise: a refused free-standing box is stashed and swept at 16
/// angles, and it reads such a balloon at 0.9831. With the raise in place the
/// same box instead took the ordinary single-orientation read at 0.5417 and was
/// lettered with a wrong phrase -- and a census of 26 composites on a Chinese
/// test chapter measured the raise buying ZERO wins and three regressions: that
/// fabrication, a one-character read at 0.19 lettered over a 214,119 px erase of
/// drawn brush art the refusal spares, and a garbled sound effect at 0.75 where
/// the sweep reads it correctly at 0.98. The reads the flag genuinely buys come
/// from the fragment-arm drop and the mint path, and neither passes through the
/// area arm at all.
///
/// **Both call sites pass the same `joined_page`, and that is load-bearing.**
/// `detection.rs`'s erase-mask call reaches this too, and its own comment records
/// what happens when only one of them is relaxed: on one joined pair the name
/// was read and lettered while its artwork was spared, so English landed on top
/// of intact Chinese. Reader and mask move together or neither does.
fn refuse_region(
    region: (f64, f64),
    page: (u32, u32),
    max_area: Option<f32>,
    joined_page: bool,
) -> bool {
    if joined_page {
        implausible_region(region, page, max_area)
    } else {
        refuse_before_reading(region, page, max_area)
    }
}

/// Did OCR read anything that could be text at all?
///
/// **This is asked so the ERASE MASK can be withdrawn, and that is the whole
/// point of it existing here rather than downstream.** `birelate-server`'s
/// `labels::hide_implausible` already asks a superset of this question, and asks
/// it well -- but it runs after `Pipeline::execute` returns, and the mask is
/// written during DETECTION, before OCR has read a character. So the refusal
/// arrives correct and too late: the region is kept off the page and the artwork
/// under it is already gone.
///
/// Measured on three Chinese webtoon slices before this existed: 15 regions, 0
/// lettered, 15 erased. RF-DETR labelled five birds `onomatopoeia`, PaddleOCR-VL
/// read them as `↓ Y V √ 1`, the server refused all five, and the birds were
/// erased out of the sky with nothing drawn in their place.
///
/// **Only the language-INDEPENDENT rules live here**, deliberately. The script
/// rules (kana on a Chinese page, Han on a Korean one) need a declared source
/// language, which the pipeline is never told -- it is a request field the server
/// resolves from the extension's per-host latch. Those stay downstream, where the
/// answer is known. What is here is the part that needs no language at all:
///
/// 1. Nothing that can be a letter in any script -- empty, or only punctuation,
///    symbols, arrows, box drawing and digits.
/// 2. One or two Latin letters and nothing else, which on CJK material is a
///    detector error rather than a word.
///
/// A single CJK character is NOT refused, and that separation is the reason the
/// rule works rather than a lucky threshold: on the measured pages the false
/// positives were `↓ Y V √ 1 ^ A 5` and the genuine sound effects were the single
/// characters `米 共 大 明 鸣 嫩`. Length does not separate them; script does.
/// Site watermarks, kept BYTE-IDENTICAL to `birelate-server`'s `labels.rs` copy.
///
/// The same convention `is_symbol_or_punctuation` already follows below, and for
/// the same reason: the two answer one question at two points in the pipeline,
/// and a divergence would let a region be erased by one and lettered by the
/// other.
const WATERMARKS: &[&str] = &[
    "最新免费漫画",
    // The traditional twin. Every other CJK entry here carries both variants and
    // these two did not, which matters more than it looks: `watermark_text` is a
    // literal `contains`, so a traditional read never reaches `story_text` at
    // all -- and `site_prose`, which would otherwise catch `漫畫`, is only
    // reachable from inside it. Prophylactic: no test corpus contains either
    // string, so neither claims a measured effect.
    "最新免費漫畫",
    "本漫畫由",
    "本漫画由",
    "腾讯动漫",
    "騰訊動漫",
];

/// A site watermark, which is never worth erasing artwork for.
///
/// **This is the language-independent half of the downstream rule, and moving it
/// here is what stops the damage.** `labels::hide_implausible` already refuses a
/// watermark, correctly, but it runs after `Pipeline::execute` returns while the
/// erase mask is written during DETECTION -- so the refusal has always arrived
/// right and too late, exactly as the comment on `illegible_text` describes for
/// the illegible case.
///
/// The erase-dilation radius, `round(max_dim / 1024 * 6)`, BEFORE each call
/// site's own clamp/cast. The one rule `write_mask` dilates the erase mask by
/// and both of OCR's mask-asset writers (`write_illegible_veto`,
/// `write_rescue_mask`) grow their boxes by -- previously three inline copies,
/// two byte-identical, all keyed off the original page size.
///
/// **The TAILS deliberately stay at the call sites**, because they genuinely
/// differ and unifying them would be a behavior change: detection clamps to
/// `1.0..=255.0` and casts `u8` for `imageproc::dilate`; the two OCR writers
/// floor at `0.0` and cast `i64` for box growth. Measured divergence of the
/// tails: 156,481 of the dimensions in `0..200_000` (everything below 86 and
/// everything at or above 43,606). The pre-clamp value itself is exact in
/// either float width for every dimension below ~5.6M -- `3d/512` needs at
/// most 24 significand bits there -- proven exhaustively over `0..2^22`
/// during extraction.
fn dilation_radius(max_dim: u32) -> f64 {
    ((f64::from(max_dim) / 1024.0) * 6.0).round()
}

/// Measured on 60 manhua pages before this existed: **31 of 67 refusals were
/// watermarks, and every one still had its erase applied**. One page's
/// site-address plate came back as LaMa's invention of a garbled address;
/// another page's logo lost its face to a white blob; 25 pages carried a smear
/// where a clean watermark had been. The change measured inside the refused box
/// reached 60.1%.
///
/// A watermark is a literal string in any language, so nothing stopped this one
/// being asked here even before the pipeline knew what it was reading.
///
/// The script rules need a declared source language as well:
/// `PipelineConfig::translation.source_language` carries it, and
/// `script_mismatch` asks that question here too, at the point where the erase
/// mask is still changeable, rather than downstream where the refusal arrives
/// right and too late.
fn watermark_text(text: &str) -> bool {
    let lowered = text.to_lowercase();
    WATERMARKS
        .iter()
        .any(|mark| lowered.contains(&mark.to_lowercase()))
}

/// Fold fullwidth ASCII (U+FF01..U+FF5E) onto its halfwidth twin.
///
/// PaddleOCR-VL returns a plate's address in fullwidth forms often enough to
/// matter. Every character is non-ASCII, so [`site_address`]'s trim eats the whole
/// token and the `.` test finds nothing to split on — the address is kept as
/// residue and the plate is lettered.
///
/// **MEASURED, and the first estimate was wrong by an order of magnitude.** Of the
/// 8 distinct fullwidth garbles across 93,588 stored regions, exactly **one**
/// folds into a well-formed address, and it occurred twice (a plate the earlier,
/// longer list marked). The rest fail on no
/// dot at all, a TLD too long, a non-alphabetic TLD (`８ｍ`), or an empty head.
/// **Claim two, not nineteen.** Folding [`orphaned_address_tail`] as well was
/// measured and buys nothing, so it is deliberately not done.
fn fold_fullwidth(token: &str) -> String {
    token
        .chars()
        .map(|character| match character as u32 {
            point @ 0xFF01..=0xFF5E => char::from_u32(point - 0xFEE0).unwrap_or(character),
            _ => character,
        })
        .collect()
}

/// One character folded for a case-insensitive compare.
fn lower_char(character: char) -> char {
    character.to_lowercase().next().unwrap_or(character)
}

/// Remove every occurrence of `mark`, comparing case-insensitively.
///
/// Walks CHARACTERS rather than bytes. A byte search over `to_lowercase()` would
/// be shorter, but it is only correct while folding preserves byte length -- true
/// for the Han in `WATERMARKS` today and not a property to rely on.
fn strip_mark(text: &str, mark: &str) -> String {
    let haystack: Vec<char> = text.chars().collect();
    let needle: Vec<char> = mark.chars().map(lower_char).collect();
    if needle.is_empty() || haystack.len() < needle.len() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < haystack.len() {
        let matches = index + needle.len() <= haystack.len()
            && (0..needle.len()).all(|step| lower_char(haystack[index + step]) == needle[step]);
        if matches {
            index += needle.len();
        } else {
            out.push(haystack[index]);
            index += 1;
        }
    }
    out
}

/// A token shaped like a site address, however badly the recogniser mangled it.
///
/// **A literal list cannot do this job and that is the whole reason this exists.**
/// The same plate read twice can come back as two different misspellings of its
/// address (say `www.paperleef.com` and `www.papcrleaf.com`), and another page as
/// a third (`Pagerleaf.com`). No literal list can hold them all, because the
/// garbling differs on every read. Shape is stable where the spelling is not.
///
/// **The fullwidth fold below gives up a guarantee.** Without it this rule was
/// ASCII-only, so its risk on ordinary text was structurally zero rather than
/// merely small. A fullwidth token can now satisfy the rule, so the risk is no
/// longer structural. It is **measured** instead — over 93,588 stored regions
/// the fold changes the translator's effective input on **2 occurrences, both
/// watermark-bearing under the earlier list, 0 non-watermark** — with a
/// byte-identical control arm
/// measuring 0 everywhere.
///
/// The exchange is deliberate: without the fold, a plate whose address came
/// back in fullwidth forms keeps that address as residue and the plate is
/// translated and lettered onto the page.
fn site_address(token: &str) -> bool {
    let folded = fold_fullwidth(token);
    let token = folded.as_str();
    let trimmed = token.trim_matches(|character: char| !character.is_ascii_alphanumeric());
    if !trimmed
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '.' || character == '-')
    {
        return false;
    }
    trimmed.rsplit_once('.').is_some_and(|(head, tld)| {
        !head.is_empty()
            && (2..=4).contains(&tld.len())
            && tld.chars().all(|character| character.is_ascii_alphabetic())
    })
}

/// A domain's TAIL, orphaned when [`story_text`] removed the mark that was its head.
///
/// **This exists because the refusal was defeated by the plate being read
/// CORRECTLY.** When a listed mark is written straight onto a domain's tail --
/// `最新免费漫画.com` -- stripping the mark leaves `.com`. [`site_address`]
/// does not match that: it trims the leading dot, finds `com`, and `rsplit_once('.')`
/// then returns `None` for want of a second dot. The residue is non-empty, so the
/// region reads as story text and is translated and lettered. A MISREAD of the same
/// plate survives mark removal intact, is a well-formed address, and is correctly
/// refused. Measured over all 180 pages of a Chinese test chapter, with an earlier
/// list that also named the hosting site: 31 bare plates were lettered this way,
/// and one plate carrying genuine story text beside it was correctly kept. The
/// shipped list names no site, so a plate carrying only a site's own name is not
/// refused here.
///
/// **Trimming only the END is what keeps a fused skill name alive.** `苍炎之王.com`
/// -- a skill name the recogniser fused to the plate -- must NOT match, and it does
/// not, because the leading dot it is tested for is not there. A rule that trimmed
/// both ends would see `com` and throw the name away.
fn orphaned_address_tail(token: &str) -> bool {
    let trimmed = token.trim_end_matches(|character: char| !character.is_ascii_alphanumeric());
    trimmed.strip_prefix('.').is_some_and(|tld| {
        (2..=4).contains(&tld.len()) && tld.chars().all(|character| character.is_ascii_alphabetic())
    })
}

/// Words a site's own sentence is built from, and a skill name is not.
///
/// **This exists because stripping the literal marks is NOT enough, and a test
/// caught it rather than a corpus sweep.** A credit line such as
/// `本漫畫由某某漫畫收集整理，更多免費漫畫請訪問` loses only its [`WATERMARKS`] prefix,
/// leaving text which is still entirely the site talking. Rescuing
/// that would letter site prose onto the artwork -- the failure mode that makes
/// this whole idea net-harmful, so it is tested for directly.
///
/// **A ratio cannot separate these and it was tried.** The prose survivor is 64%
/// of its region; the skill name is 40% of its own. The survivors differ in
/// vocabulary, not in proportion.
///
/// The known cost, stated rather than discovered later: a genuine line ABOUT
/// comics -- a character saying `漫画` -- is refused. That is a narrow population
/// and the refusal is today's behaviour for it, not a new loss.
const SITE_PROSE: &[&str] = &[
    "漫畫", "漫画", "免費", "免费", "訪問", "访问", "收集整理",
];

/// Whether what is left of a line is still the site talking.
fn site_prose(line: &str) -> bool {
    let lowered = line.to_lowercase();
    SITE_PROSE.iter().any(|word| lowered.contains(word))
}

/// What a region says once the SITE's own text is removed from it.
///
/// **This is the line-vs-region fix, and it is scoped to SUBSTRINGS rather than
/// lines because the measurement forced it.** A vertical display column read
/// upright returns the names and the watermark on three separate lines; read with
/// the crop turned -- which the rotated re-read does -- the recogniser returns all
/// of it on ONE line, `苍炎之王·冥霜之王·裂风之王最新免费漫画 www.paperleaf.com`,
/// because after the turn the plate genuinely does sit beside the name. A rule
/// that dropped whole lines would refuse that entire string and throw the skill
/// name away.
///
/// Empty means "nothing here but the site's furniture", which is the ordinary case
/// and which every caller still refuses. Non-empty is the case this exists for: a
/// region the detector drew correctly around artwork the site happened to stamp
/// on.
fn story_text(text: &str) -> String {
    let mut cleaned = text.to_string();
    for mark in WATERMARKS {
        cleaned = strip_mark(&cleaned, mark);
    }
    // GATED ON THE REGION ACTUALLY BEARING A MARK, and that gate is not tidiness.
    // This function has TWO consumers and only one is the refusal: `targets()` in
    // `stages/translation.rs` applies it to EVERY region's source before
    // translation. Ungated, the decoration rule below does not just tidy watermark
    // residue -- it takes punctuation off ordinary pages. Measured over 93,588
    // stored regions, ungated it changed the translator's effective input on
    // 30 occurrences, 9 distinct,
    // EVERY ONE non-watermark, a Korean bubble losing its `!` among them. Gated:
    // 0, with the refusal behaviour unchanged.
    let drop_decoration = watermark_text(text);
    cleaned
        .lines()
        .map(|line| {
            line.split_whitespace()
                .filter(|token| !site_address(token) && !orphaned_address_tail(token))
                .collect::<Vec<_>>()
                .join(" ")
        })
        // A line of pure DECORATION is not story text. The motivating
        // plate is `👇最新免费漫画👇`, and with the mark stripped the two pointing
        // hands survived as residue, so `only_site_furniture`'s second half was
        // false and the banner was lettered onto the page as dialogue.
        //
        // `scripted()`, never `letters()`. `letters()` sums `other`, and
        // `is_symbol_or_punctuation` tops out at 0xFF64, so an emoji lands in
        // `other` and `letters("👇👇") == 2` -- a rule written on `letters` is a
        // no-op against the exact string this exists for.
        .filter(|line| {
            !line.trim().is_empty()
                && !site_prose(line)
                && !(drop_decoration && Scripts::of(line).scripted() == 0)
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Whether a region is nothing but site furniture.
///
/// The composed predicate the callers call, named so a test can assert on it.
/// With `scoped` off this is [`watermark_text`] exactly -- op for op, so the
/// shipped arm is unchanged -- and with it on a region survives whenever any
/// story text is left after the furniture is removed.
fn only_site_furniture(text: &str, scoped: bool) -> bool {
    if !scoped {
        return watermark_text(text);
    }
    watermark_text(text) && story_text(text).is_empty()
}

/// Whether this read will not be lettered whatever the translator answers, so
/// sending it buys nothing and costs the good text beside it.
///
/// **The composed predicate the caller calls, named so a test can assert on the
/// `||`s rather than on their halves** -- the reason [`only_site_furniture`] and
/// [`withdraw_from_mask`] are shaped the same way.
///
/// Every arm is a fact the pipeline ALREADY holds before translation runs; not
/// one of them is a new signal, which is deliberate. Every cheap new signal that
/// was tried failed: OCR sequence confidence (AUC 0.407 -- *anti*-correlated),
/// the whole class of score-monotone rules (by Pareto dominance), and length
/// gates. A gate here that invented a threshold would be re-deriving a rejected
/// approach.
///
/// - `hidden` is the OCR stage's own lettering veto, `Visibility { visible:
///   false }` written at `ocr.rs:2764` and committed with that stage's patch
///   before this stage's input exists. It is the identical fact the server
///   later reports as `HIDDEN_LAYER_REFUSAL` (`regions.rs:328`). It collapses
///   three vetoes -- the spotting gate, the illegible-read floor and the
///   synthesised-region withdrawal -- which the scene cannot tell apart.
/// - [`illegible_text`] is nothing-that-can-be-a-letter, or one-or-two Latin
///   letters on CJK material. **45 of 331 regions on the measured chapter, 0 of
///   them ever lettered.**
/// - [`only_site_furniture`] is the hosting site's stamp: **116 of 331** (measured
///   with an earlier list that also named the hosting site). Narrower
///   than the `story_text(..).is_empty()` test sitting in `targets` already,
///   because it also demands a [`watermark_text`] literal -- and that gate is
///   load-bearing, not decoration. Ungated, the decoration rule takes
///   punctuation off ordinary pages: 30 occurrences over 93,588 regions, every
///   one non-watermark, a Korean bubble losing its `!` among them.
///
/// **What this predicate is NOT allowed to do is decide a page.** The caller
/// must keep every target when nothing would survive -- see
/// `ProcessorConfig::skip_unlettered_reads` for why an emptied page turns a 502
/// into a silently untranslated render.
fn unlettered_read(text: &str, hidden: bool, scoped: bool) -> bool {
    hidden || illegible_text(text) || only_site_furniture(text, scoped)
}

/// The share of a region's letters that must be in the wrong script before it
/// counts as a bad read rather than a stray character.
///
/// Kept byte-identical to `birelate-server`'s copy in `labels.rs`, for the same
/// reason `is_symbol_or_punctuation` below is: the two answer the same question at
/// two points in the pipeline, and a divergence would let a region be erased by
/// one and lettered by the other.
const FOREIGN_SHARE: f64 = 0.34;

/// Letters by script, counted exactly as `labels.rs`'s `Scripts` counts them.
///
/// **Punctuation is tested FIRST, and that ordering is load-bearing.** `・`
/// (U+30FB, the katakana middle dot) lives INSIDE the kana block, so a test that
/// reaches the block first counts a row of interpuncts as kana -- "100% foreign
/// script" on a Korean page, a refusal for entirely the wrong reason. Order, not
/// ranges. `illegible_text` below already carries the same warning.
#[derive(Clone, Copy, Debug, Default)]
struct Scripts {
    han: u32,
    kana: u32,
    hangul: u32,
    latin: u32,
    other: u32,
}

impl Scripts {
    fn of(text: &str) -> Self {
        let mut counts = Self::default();
        for character in text.chars() {
            let point = character as u32;
            if is_symbol_or_punctuation(point)
                || character.is_whitespace()
                || character.is_ascii_punctuation()
                || character.is_ascii_digit()
            {
                continue;
            }
            if matches!(point, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF) {
                counts.han += 1;
            } else if matches!(point, 0x3040..=0x30FF | 0xFF66..=0xFF9F) {
                // The halfwidth katakana at FF66-FF9F are kana and are
                // `is_alphabetic`, so without them they would have been counted
                // as LATIN -- escaping the one rule they should trip.
                counts.kana += 1;
            } else if matches!(point, 0xAC00..=0xD7AF | 0x1100..=0x11FF) {
                counts.hangul += 1;
            } else if character.is_alphabetic() {
                counts.latin += 1;
            } else {
                counts.other += 1;
            }
        }
        counts
    }

    /// Letters in a *named* script. `other` is excluded on purpose: it is the
    /// bucket for things nobody classified, and dividing by it would move the
    /// threshold around for reasons unrelated to the script question.
    const fn scripted(self) -> u32 {
        self.han + self.kana + self.hangul + self.latin
    }
}

/// Is this read in a script the page is not written in?
///
/// **The pipeline had to be TOLD the source language for this to be askable at
/// all**, which is what `PipelineConfig::translation.source_language` is for and
/// what the comment on `illegible_text` below is describing when it says the
/// script rules "need a declared source language the pipeline is never told".
/// They are told now.
///
/// The arms are byte-identical to `labels::classify`'s: Korean is written in
/// Hangul, so Han and kana are foreign; Chinese has no kana; Japanese mixes kana
/// and Han by design so nothing fires on it. **`None` fires nothing**, and that
/// asymmetry is deliberate -- inferring "not Japanese" from silence would drop
/// genuine Japanese dialogue.
/// **Takes the enum, NOT a string, and that is the whole of a bug this cost.**
/// The first version took `Option<&str>` and parsed a language tag out of it. In
/// production the caller had a `Language`, so it passed `Language::to_string()` --
/// and strum renders `ChineseSimplified` as **"Simplified Chinese"**, not `"zh"`.
/// The tag parser split that on `-`, got `"simplified chinese"`, matched nothing,
/// and returned false for every page. The feature was dead in the render while
/// the unit test stayed green, because the test passed the literal `"zh-CN"` and
/// never went through the conversion. Matching the variant cannot drift with a
/// `to_string` attribute.
fn script_mismatch(text: &str, source: Option<Language>) -> bool {
    let Some(language) = source else {
        return false;
    };
    let counts = Scripts::of(text);
    let scripted = f64::from(counts.scripted());
    if scripted <= 0.0 {
        return false;
    }
    match language {
        Language::Korean => {
            f64::from(counts.han + counts.kana) / scripted >= FOREIGN_SHARE
        }
        Language::ChineseSimplified | Language::ChineseTraditional => {
            f64::from(counts.kana) / scripted >= FOREIGN_SHARE
        }
        // Japanese mixes kana and Han by design, and every other target language
        // says nothing about what the SOURCE page is written in.
        _ => false,
    }
}

/// A declared-KOREAN read carrying no hangul at all.
///
/// The ratio arm above cannot see this population: `scripted()` includes latin,
/// so an all-Latin misread of drawn hangul ("Hwak", "OUBA") divides to zero and
/// `ga 大` to 1/3 -- both under `FOREIGN_SHARE`. Korean is written in hangul; a
/// read of `STRICT_HANGUL_MIN`+ scripted letters with none is not Korean. The
/// floor keeps one- and two-letter reads with the rules that already own them
/// (the junk rule, the ratio arm at 1/1).
///
/// Kept byte-identical to `birelate-server/src/labels.rs`'s copy, the same
/// contract `is_symbol_or_punctuation` names: two answers to one question is a
/// region erased by one half and lettered by the other.
const STRICT_HANGUL_MIN: u32 = 3;

fn strict_korean_mismatch(text: &str, source: Option<Language>) -> bool {
    if !matches!(source, Some(Language::Korean)) {
        return false;
    }
    let counts = Scripts::of(text);
    counts.hangul == 0 && counts.scripted() >= STRICT_HANGUL_MIN
}

fn illegible_text(text: &str) -> bool {
    let mut latin = 0_u32;
    let mut letters = 0_u32;
    let mut prolongations = 0_u32;
    for character in text.chars() {
        let point = character as u32;
        // Punctuation FIRST, exactly as `labels.rs` does and for the same reason
        // recorded there: `・` (U+30FB) sits inside the kana block, so a test
        // that reaches the block first counts a row of interpuncts as kana.
        if character.is_whitespace()
            || character.is_ascii_punctuation()
            || character.is_ascii_digit()
            || is_symbol_or_punctuation(point)
        {
            continue;
        }
        letters += 1;
        if is_prolongation(point) {
            prolongations += 1;
        }
        let cjk = matches!(point,
            0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF   // han
            | 0x3040..=0x30FF | 0xFF66..=0xFF9F                   // kana, incl. halfwidth
            | 0xAC00..=0xD7AF | 0x1100..=0x11FF);                 // hangul
        if !cjk && character.is_alphabetic() {
            latin += 1;
        }
    }
    letters == 0 || (latin == letters && letters <= 2) || letters == prolongations
}

/// `ー` (U+30FC) and its halfwidth twin, which lengthen the vowel BEFORE them.
///
/// **They have no sound of their own**, so a read that is nothing but prolongation
/// marks is not an utterance -- there is no preceding vowel for them to lengthen.
/// This is deliberately NOT folded into `is_symbol_or_punctuation`: that function
/// is contractually byte-identical to `birelate-server`'s copy, and `ー` really is
/// a letter the moment anything precedes it. What is meaningless is a read
/// consisting of NOTHING else, which is why the caller compares the count rather
/// than skipping the character.
///
/// # The page this was measured on
///
/// A webtoon test page. RF-DETR returns a `bubble` at **0.6211** over
/// flat skin -- no balloon at all -- and with `--read-textless-bubbles` on, the
/// synthesis hands the whole thing to OCR, which reads the character's own MOUTH
/// LINE as `ー`. It survived every existing gate: `illegible_text` counted one
/// non-Latin letter, `only_site_furniture` matches watermarks alone, and
/// `script_mismatch`'s arm is unreachable because a synthesised region always
/// carries the role `dialogue`. So an em dash was lettered across her face.
///
/// **92% of that damage is the ERASE, not the lettering** -- of the 9,290 px that
/// change on the band, the glyph box is 755, leaving **8,535 px of her face and
/// collar destroyed**. The erase is written from raw detections before OCR runs,
/// so a rule that only suppressed the lettering would have fixed 8% of it. This
/// one withdraws the mask through `write_illegible_veto`, which subtracts from
/// both masks before inpainting, and so pays for the whole of it.
///
/// # What it must NOT catch
///
/// Ordinary dialogue that merely carries a prolongation (`すごーい！`, `そうだよー`),
/// and a genuine balloon whose whole read is ONE character (`ん…`). That last
/// shape is why this rule is about the mark rather than about length: a "too few
/// characters" rule was one of only three fields that separated the false
/// positive from the true one, and it would have thrown a real read away.
/// Rendered before the rule was written, not after.
const fn is_prolongation(point: u32) -> bool {
    matches!(point, 0x30FC | 0xFF70)
}

/// Ranges Unicode calls punctuation or symbol that `is_ascii_punctuation` misses.
/// Kept byte-identical to `birelate-server`'s copy in `labels.rs`: the two answer
/// the same question at two points in the pipeline, and a divergence here would
/// let a region be erased by one and lettered by the other -- the precise failure
/// `implausible_region` above already carries a warning about.
const fn is_symbol_or_punctuation(point: u32) -> bool {
    matches!(point,
        0x30FB | 0xFF65   // katakana middle dot, full and half width
        // The LATIN middle dot, the mark above's Latin-1 twin. Missing, it made
        // a pupil: a 16x16 detection on a character's EYE read `·`, and because
        // U+00B7 fell to the `other` bucket the read counted as a letter --
        // no PunctuationOnly refusal, no letters==0 withdrawal -- so the iris
        // was erased and an interpunct lettered onto it.
        // Contractually byte-identical to birelate-server's copy; changed there
        // in the same commit.
        | 0x00B7
        | 0x309B | 0x309C // spacing voiced sound marks
        | 0x2000..=0x206F // general punctuation, including the ellipsis
        | 0x2190..=0x21FF // arrows
        | 0x2200..=0x22FF // mathematical operators
        | 0x2500..=0x257F // box drawing
        | 0x25A0..=0x25FF // geometric shapes
        | 0x2600..=0x27BF // misc symbols and dingbats
        | 0x3000..=0x303F // CJK symbols and punctuation
        | 0xFE30..=0xFE4F // CJK compatibility forms
        | 0xFF01..=0xFF20 // fullwidth ASCII punctuation and digits
        | 0xFF3B..=0xFF40
        | 0xFF5B..=0xFF64)
}

trait ModelState: Send + Sync {
    fn loaded(&self) -> bool;
    fn unload(&self) -> bool;
    fn touch(&self, sequence: u64);
    fn last_used(&self) -> u64;
}

impl<M: Send> ModelState for ModelCell<M> {
    fn loaded(&self) -> bool {
        ModelCell::loaded(self)
    }

    fn unload(&self) -> bool {
        ModelCell::unload(self)
    }

    fn touch(&self, sequence: u64) {
        ModelCell::touch(self, sequence);
    }

    fn last_used(&self) -> u64 {
        ModelCell::last_used(self)
    }
}

struct ModelRef<'a> {
    name: &'static str,
    state: &'a dyn ModelState,
}

impl<'a> ModelRef<'a> {
    fn new(name: &'static str, state: &'a dyn ModelState) -> Self {
        Self { name, state }
    }
}

#[async_trait]
trait StageProcessor: Send + Sync {
    fn model(&self) -> ModelRef<'_>;

    /// Whether this page holds anything for the stage to do, asked before its
    /// weights are paged in.
    ///
    /// Default `true`, so a stage that does not override this behaves exactly as
    /// it did before the question existed. Override it only where the answer can
    /// be read off the scene alone -- no device, no model -- and only by reusing
    /// the very walk `process` uses. Two walks that can disagree is how a page
    /// silently loses its dialogue: a `false` here skips the stage outright, and
    /// an untranslated bubble renders as its own source text, which nothing
    /// downstream can tell apart from a stage that ran and found nothing.
    ///
    /// An `Err` is not a refusal. The caller falls back to loading, and the same
    /// walk fails again inside `process` a moment later with the stage's own
    /// error context around it -- which is where it belongs.
    fn has_work(&self, _input: &StageInput) -> Result<bool> {
        Ok(true)
    }

    async fn load(&self) -> Result<()>;
    async fn process(&self, input: StageInput) -> Result<Patch>;
}

pub(crate) struct Stages {
    detection: detection::Processor,
    ocr: ocr::Processor,
    translation: translation::Processor,
    inpainting: inpainting::Processor,
    /// Whether `has_work` is consulted at all. Off restores the load-then-ask
    /// order for every stage at once, which is what makes the two arms
    /// comparable over one page set. See `ProcessorConfig::skip_empty_stages`.
    skip_empty_stages: bool,
}

impl Stages {
    pub(crate) fn new(
        config: &PipelineConfig,
        translator: koharu_translator::Translator,
        device: &koharu_ml::Device,
    ) -> Result<Self> {
        Ok(Self {
            detection: detection::Processor::new(
                config.detection()?,
                device.clone(),
                // Resolved to one `Option<f32>` here rather than handed to the
                // stage as a flag plus a fraction, because the stage has exactly
                // one question to answer and there is no state in which it wants
                // half of this: a fraction with the flag off means "keep
                // erasing", and the flag on without a fraction has no ceiling to
                // test against. Both are `None`.
                config
                    .processor
                    .skip_implausible_masks
                    .then_some(config.processor.large_crop_ocr_max_area)
                    .flatten(),
                config.processor.rotate_free_text_columns,
                config.processor.turn_unjoined_columns,
                config.processor.repair_clipped_columns,
                config.processor.read_textless_bubbles,
                config.processor.joined_page_text_floor,
                config.processor.axis_aware_nms,
                config.processor.nms_residue_regions,
                config.processor.strike_through_devices,
                config.processor.sampled_ink_lettering,
                config.processor.debug_mask_dir.clone(),
            )?,
            ocr: ocr::Processor::new(
                config.ocr.clone(),
                device.clone(),
                config.processor.large_crop_ocr_px,
                ocr::ImplausibleRegions {
                    max_area: config.processor.large_crop_ocr_max_area,
                    skip: config.processor.skip_implausible_regions,
                    veto_mask: config.processor.withdraw_unread_masks,
                },
                config.processor.withdraw_illegible_masks,
                // Read off `translation` because that is where the server already
                // resolves a per-request language, NOT because the translator
                // uses it. `stages/translation.rs` must never read this field:
                // naming the source language in the prompt measured worse.
                //
                // Passed as the `Language` itself. It was `.map(ToString::to_string)`
                // for one render, and that render silently did nothing: strum
                // renders ChineseSimplified as "Simplified Chinese".
                config.translation.source_language,
                config.processor.reread_rotated_columns,
                config.processor.scope_watermark_refusals,
                config.processor.orientation_confidence_margin,
                config.processor.perturb_reread_grow_px,
                config.processor.upright_pass,
                config.processor.flip_reread_bubbles,
                config.processor.spot_rescue,
                config.processor.spot_rescue_erase,
                config.processor.replace_scream_marks,
                config.processor.spot_rescue_joined,
                config.processor.leave_misread_bubbles,
                config.processor.korean_script_strict,
                config.processor.reread_refused_dialogue,
            ),
            translation: translation::Processor::new(
                config.translation.clone(),
                translator,
                config.processor.scope_watermark_refusals,
                config.processor.skip_unlettered_reads,
            ),
            inpainting: inpainting::Processor::new(
                config.inpainting()?,
                device.clone(),
                config.processor.seam_safe_erase,
                config.processor.debug_mask_dir.clone(),
            )?,
            skip_empty_stages: config.processor.skip_empty_stages,
        })
    }

    fn processor(&self, stage: Stage) -> &dyn StageProcessor {
        match stage {
            Stage::Detection => &self.detection,
            Stage::Ocr => &self.ocr,
            Stage::Translation => &self.translation,
            Stage::Inpainting => &self.inpainting,
        }
    }

    pub(crate) fn model(&self, stage: Stage) -> &'static str {
        self.processor(stage).model().name
    }

    pub(crate) async fn load(&self, stage: Stage) -> Result<()> {
        self.processor(stage).load().await
    }

    /// Whether `stage` has anything to do for `input`, before any weights move.
    ///
    /// Reports `true` whenever the feature is off, so the flag is one branch in
    /// one place rather than a condition every processor has to remember. A
    /// `true` can only ever be right: it restores the load-then-ask order the
    /// caller had before.
    pub(crate) fn has_work(&self, stage: Stage, input: &StageInput) -> Result<bool> {
        if !self.skip_empty_stages {
            return Ok(true);
        }
        self.processor(stage).has_work(input)
    }

    pub(crate) async fn process(&self, stage: Stage, input: StageInput) -> Result<Patch> {
        self.processor(stage).process(input).await
    }

    pub(crate) fn loaded(&self, stage: Stage) -> bool {
        self.processor(stage).model().state.loaded()
    }

    pub(crate) fn unload(&self, stage: Stage) -> bool {
        self.processor(stage).model().state.unload()
    }

    pub(crate) fn touch(&self, stage: Stage, sequence: u64) {
        self.processor(stage).model().state.touch(sequence);
    }

    pub(crate) fn last_used(&self, stage: Stage) -> u64 {
        self.processor(stage).model().state.last_used()
    }
}

fn generation(producer: &str, model: &str) -> Result<Generation> {
    let mut generation = Generation::new(ProducerId::new(producer)?);
    generation.model = Some(model.to_owned());
    Ok(generation)
}

fn finish(edit: Edit) -> Result<Patch> {
    edit.finish().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first test this arithmetic has ever had: each call site's FINAL
    /// radius, tails replicated here exactly as they appear at the sites, over
    /// the boundary dimensions where the tails diverge (85/86 is where the
    /// rounded value first reaches 1; 43,605/43,606 is where detection's u8
    /// clamp pins while the OCR writers keep growing). Proven able to fail by
    /// changing the helper's 6.0 to 7.0 (five rows go red) during extraction.
    #[test]
    fn the_dilation_radius_tails_stay_at_their_sites() {
        for (dim, detection_tail, ocr_tail) in [
            (0_u32, 1_u8, 0_i64),
            (85, 1, 0),
            (86, 1, 1),
            (1200, 7, 7),
            (1700, 10, 10),
            (43_605, 255, 255),
            (43_606, 255, 256),
        ] {
            let radius = dilation_radius(dim);
            assert_eq!(
                (radius as f32).clamp(1.0, 255.0) as u8,
                detection_tail,
                "detection's dilate tail at max_dim {dim}"
            );
            assert_eq!(
                radius.max(0.0) as i64,
                ocr_tail,
                "the OCR writers' box-growth tail at max_dim {dim}"
            );
        }
    }

    fn asset(bytes: &'static [u8]) -> koharu_scene::AssetInput {
        koharu_scene::AssetInput::new(
            bytes,
            "image/png",
            koharu_scene::AssetMetadata {
                width: Some(1),
                height: Some(1),
                attributes: std::collections::BTreeMap::new(),
            },
        )
    }

    #[tokio::test]
    async fn registry_contains_every_stage() {
        let translator = koharu_translator::Translator::from_config(
            koharu_ml::Device::cpu(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let stages = Stages::new(
            &PipelineConfig::default(),
            translator,
            &koharu_ml::Device::cpu(),
        )
        .unwrap();

        assert_eq!(
            Stage::ALL.map(|stage| stages.model(stage)),
            [
                "koharu-layout-rfdetr-seg-2xl",
                // hunyuan-ocr-1.5 is the default; paddleocr-vl-1.6 is the
                // reserve, wired but not default.
                "hunyuan-ocr-1.5",
                "local",
                "lama",
            ]
        );
    }

    #[test]
    fn translation_and_inpainting_compose_without_weakening_text_guards() {
        let mut session = koharu_scene::Session::memory().unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(
                koharu_scene::PageDraft::new("page", 1.0, 1.0),
                koharu_scene::At::End,
            )
            .unwrap();
        let text = setup.add_text_content(page, koharu_scene::At::End).unwrap();
        setup
            .set(
                text,
                &koharu_scene::SourceText {
                    text: koharu_scene::Authored::user("before".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        setup
            .set_asset(
                page,
                &koharu_scene::AssetRole::new("source").unwrap(),
                asset(b"source"),
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).unwrap();
        let base = session.snapshot();

        let mut text_edit = base.edit();
        text_edit.observe::<koharu_scene::SourceText>(text).unwrap();
        text_edit
            .observe::<koharu_scene::Translation>(text)
            .unwrap();
        text_edit
            .set(
                text,
                &koharu_scene::Translation {
                    text: koharu_scene::Authored::user("after".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        let text_patch = text_edit.finish().unwrap();

        let mut image_edit = base.edit();
        image_edit.observe_assets(page).unwrap();
        let cleanup = image_edit
            .add_entity(page, koharu_scene::At::Start)
            .unwrap();
        image_edit
            .set(
                cleanup,
                &koharu_scene::RasterLayer {
                    origin: koharu_scene::Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: koharu_scene::RasterLayerKind::Cleanup,
                },
            )
            .unwrap();
        image_edit
            .set_asset(
                cleanup,
                &koharu_scene::AssetRole::new("source").unwrap(),
                asset(b"clean"),
            )
            .unwrap();
        let image_patch = image_edit.finish().unwrap();

        let image_first = base.preview([&image_patch]).unwrap();
        assert!(text_patch.rebase_on(&image_first).is_ok());
        let text_first = base.preview([&text_patch]).unwrap();
        assert!(image_patch.rebase_on(&text_first).is_ok());

        let changed_source = base
            .patch(|edit| {
                edit.set(
                    text,
                    &koharu_scene::SourceText {
                        text: koharu_scene::Authored::user("changed".to_owned()),
                        language: None,
                    },
                )
            })
            .unwrap();
        let changed_source = base.preview([&changed_source]).unwrap();
        assert!(text_patch.rebase_on(&changed_source).is_err());
    }

    /// **The eight strings that were actually erasing artwork.** All real
    /// PaddleOCR-VL output from four slices of a test webtoon chapter, where
    /// RF-DETR labelled birds and motion marks `onomatopoeia`.
    #[test]
    fn the_measured_false_positives_are_all_illegible() {
        for text in ["↓", "Y", "V", "√", "1", "^", "A", "5"] {
            assert!(illegible_text(text), "{text:?} should be refused");
        }
    }

    /// **The six that must survive, and they are why the rule is about script
    /// rather than length.** Every genuine sound effect on those same pages was
    /// a SINGLE character, so any rule keyed on how much text was read would
    /// have thrown all of these away together with the birds.
    #[test]
    fn a_single_cjk_character_is_legible() {
        for text in ["米", "共", "大", "明", "鸣", "嫩"] {
            assert!(!illegible_text(text), "{text:?} should be kept");
        }
    }

    /// Kana and hangul are letters too, and this is the reason the short rule is
    /// Latin-ONLY. A Japanese effect is very often one or two kana -- exactly
    /// the length the Latin rule refuses -- so a script-blind version of it
    /// would erase nothing on Chinese pages and delete the effects on Japanese
    /// ones.
    #[test]
    fn one_or_two_kana_or_hangul_are_kept() {
        for text in ["ド", "ドン", "あ", "탕"] {
            assert!(!illegible_text(text), "{text:?} should be kept");
        }
    }

    /// The trivial arm, and the commonest refusal of all: a region OCR returned
    /// nothing for was previously erased in full.
    #[test]
    fn empty_and_blank_are_illegible() {
        for text in ["", " ", "\n", "\u{3000}"] {
            assert!(illegible_text(text), "{text:?} should be refused");
        }
    }

    /// Longer Latin is real text -- a title, an effect already in English, a
    /// translator credit. The rule refuses one or two letters, not the script.
    #[test]
    fn three_or_more_latin_letters_are_legible() {
        assert!(!illegible_text("BAM"));
        assert!(!illegible_text("Hey"));
        assert!(illegible_text("Hi"));
    }

    /// Digits and punctuation are never letters, so a page number or a row of
    /// interpuncts is refused however long it is.
    #[test]
    fn digits_and_punctuation_are_never_letters() {
        for text in ["1", "5", "123", "!?", "・・・", "...", "…"] {
            assert!(illegible_text(text), "{text:?} should be refused");
        }
    }

    /// Punctuation around a word changes nothing either way: the verdict follows
    /// the letters. `「大」` is a real effect in quotes; `(↓)` is still an arrow.
    #[test]
    fn punctuation_neither_rescues_nor_condemns_a_word() {
        assert!(!illegible_text("大!"));
        assert!(!illegible_text("「大」"));
        assert!(illegible_text("Y!"));
        assert!(illegible_text("(↓)"));
    }
}
