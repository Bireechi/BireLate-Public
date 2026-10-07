use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

use super::{
    ModelRef, Scripts, StageInput, StageProcessor,
    detection::{
        HEAVY_INK_WEIGHT, InferredTypography, ONOMATOPOEIA as ONOMATOPOEIA_LABEL,
        SYNTHESISED_REGION_KIND, free_text_column_geometry, infer_typography,
        rotated_rectangle_geometry, text_stroke,
    },
    finish, generation, illegible_text, only_site_furniture, refuse_region, script_mismatch,
    strict_korean_mismatch,
};
use crate::{ModelCell, OcrModel, scope::geometry_extents};
use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_ml::{
    baberu_ocr::BaberuOcr,
    koharu_layout_rfdetr_seg_2xl::{KoharuLayoutDetection, KoharuLayoutMask},
    manga_ocr::MangaOcr,
    paddle_ocr_vl::{PaddleOCRVL, PaddleOCRVLTask},
};
use koharu_scene::{
    At, Authored, DetectionAnalysis, DetectionLabel, EntityId, FILL_GRADIENT_TO_EXTENSION, FitsTo,
    Generation, Geometry, LanguageTag, OcrAnalysis, Origin, RecognizedFrom, Region, RegionKind,
    RegionSpec, Snapshot, SourceText, TextDirection, TextLayout, TextLayoutKind, TextRegion,
    TextRole, Typography, Visibility, WritingMode,
};
use koharu_translator::Language;

const PRODUCER: &str = "dev.koharu.pipeline.ocr";

/// The engine oversized crops are routed to. Not configurable: it is the
/// only one of the three whose preprocessing preserves aspect ratio, which
/// is the entire reason routing exists.
const FALLBACK_MODEL: &str = "paddleocr-vl-1.6";

/// Padding for a reserve re-read of a lever-refused dialogue crop.
///
/// The value the recovery was MEASURED at: on a 19-region refusal population,
/// the reserve engine on +24 px padded crops read 4 of the 6 genuine dialogue
/// targets EXACTLY where the same crops cut TIGHT gave garbage -- three of the
/// four are edge slivers, and the padding is load-bearing. `crop_grown` clamps
/// to the posted image, so a box at a slice boundary pads as far as the slice
/// allows; the measurement's unclamped stacking is not reproducible in-pipeline.
const REREAD_PAD_PX: u32 = 24;
/// The wire name of the Ollama-served vision engine. `ModelRef::new` wants a
/// `&'static str`, so the STAGE name is fixed even though the Ollama tag behind
/// it is overridable -- do not fold the tag in here, it would have to leak.
const OLLAMA_VISION_NAME: &str = "ollama-vision";

/// Ollama's NATIVE chat endpoint, not the `/v1` OpenAI shim the translator uses.
/// The shim cannot carry an image; this one can, in a separate `images` field.
const OLLAMA_VISION_ENDPOINT: &str = "http://127.0.0.1:11434/api/chat";

/// Overridable so a quant can be swapped without a rebuild.
/// The DEFAULT tag, not an endorsement. This engine names a ROUTE, not a model:
/// point `BIRELATE_OLLAMA_VISION_MODEL` at any vision model Ollama serves.
const OLLAMA_VISION_MODEL: &str = "hf.co/ggml-org/MiniCPM-V-4.6-GGUF:Q8_0";

/// **The prompt is the point of this engine and it is deliberately settable.**
/// PaddleOCR-VL ships a fixed set of task prompts and cannot be directed;
/// this one can, which is the whole reason it is worth an experiment.
///
/// **This default was chosen on a bake-off, and the FIRST one shipped here was
/// wrong in a way no counter could see.** That one read "Read the text in this
/// image ... if there is no text, output nothing" -- reasonable-sounding, never
/// measured on artwork, and it described the picture instead: over a 179-slice
/// chapter it produced **20 English descriptions and all 20 were LETTERED**, one
/// page carrying eleven paragraphs about pink shapes across the artwork. The page
/// reported 11 regions lettered and zero warnings, so only looking at it showed it.
///
/// Measured over the 9 ARTWORK crops (correct answer: nothing) and 3 text crops
/// that must survive the same wording:
///
///     shipped-ocr    artwork silent 0/9    text kept 1/3
///     strict         artwork silent 0/9    text kept 1/3
///     plain          artwork silent 8/9    text kept 1/3
///     refuse-first   artwork silent 8/9    text kept 1/3
///
/// **AND THEN THE BAKE-OFF WINNER LOST ON THE CHAPTER, which is the lesson worth
/// keeping.** `plain` looked strictly better on those 12 crops, so it was adopted
/// and the chapter re-rendered. Over 258 real regions it was WORSE on every axis
/// that matters: all-Latin reads 50 -> **85**, median read length 16 -> **5**, and
/// long dialogue came back wrapped in English ("The characters are: ...") or
/// mangled. Descriptions barely moved, 20 -> 16. **A 12-crop bake-off did not
/// represent a 258-region chapter**, and the population it under-represented --
/// ordinary dialogue -- is the bulk of the work.
///
/// So this is back to the wording above, which reads long passages essentially as
/// well as PaddleOCR-VL. **Its artwork descriptions are a GATE problem, not a
/// prompt problem:** an all-Latin prose read on a page declared `zh` is trivially
/// detectable, and `labels.rs` only refuses all-Latin reads UNDER 3 letters
/// (`Junk`), so an English sentence sails through. Fix it there, not here.
///
/// **Score BOTH columns when changing this, and score them on a CHAPTER.** A prompt
/// that silences artwork by silencing everything scores 9/9 on artwork and is
/// useless. A crop-level probe is a guide; a full render is the one that decides.
/// Override with `BIRELATE_OLLAMA_VISION_PROMPT` and re-measure before trusting it.
const OLLAMA_VISION_PROMPT: &str = "Read the text in this image. Output only the text \nexactly as written, with no explanation, no translation and no description. If \nthere is no text, output nothing.";

/// The wire name of the HunyuanOCR engine -- the default, with `paddleocr-vl-1.6`
/// as the reserve. Served by a local sidecar (`scripts/hunyuan-llamacpp-shim.py`
/// in front of llama-server) speaking the same Ollama-native protocol this file's
/// HTTP client already speaks; the model is `tencent/HunyuanOCR` v1.5, 1B, with no
/// ONNX export, which is why it cannot ride the in-process engine path.
const HUNYUAN_NAME: &str = "hunyuan-ocr-1.5";

/// The sidecar's port. 11436, NOT 11434: Ollama owns 11434, and a sidecar squatting
/// on a free 11434 would be mistaken for Ollama by every other local tool the
/// moment Ollama started.
const HUNYUAN_ENDPOINT: &str = "http://127.0.0.1:11436/api/chat";

/// The tag the sidecar announces on `/api/tags`; the load-time probe checks it.
const HUNYUAN_MODEL: &str = "hunyuan-ocr-1.5";

/// Chinese, not English, and measured before being chosen: this exact string read
/// all three control crops of a test page byte-exactly. Tencent's official
/// `structured_parse` task (`提取图中的文字。`) is the alternative, unmeasured on
/// these crops. Override with `BIRELATE_HUNYUAN_PROMPT` and re-measure on a
/// CHAPTER before trusting it -- the MiniCPM prompt above records why a crop
/// bake-off is not enough.
const HUNYUAN_PROMPT: &str = "请识别图中的所有文字，只输出文字本身。";

/// Tencent's official `spotting_json` instruction, byte-exact from their client
/// (`inference/utils/tasks.py`). Boxes come back normalised 0..1000 against the
/// ORIGINAL image -- confirmed by overlay and against Tencent's own
/// `denormalize_coordinates`.
const HUNYUAN_SPOT_PROMPT: &str = "检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。 输出格式为 JSON 数组，每个元素必须包含：\"box\": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；\"text\": \"识别出的文字内容\"。 注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。";

/// Pixel budget for the image handed to the page-level spot call.
///
/// Measured on the llama.cpp sidecar: a 2130x8000 page (17.0 MP) encodes to
/// 16,712 visual tokens against serve.ps1's hard `-c 10240` -- roughly a token per
/// 1,020 px². 8 MP keeps the biggest spot request near ~7.9k tokens, inside
/// the context with room for the prompt and the 1536-token reply. Spot boxes
/// are normalized to `[0, 1000]`, so scaling the INPUT changes no consumer;
/// box precision at 8 MP is far finer than the admission predicate's floors.
const SPOT_MAX_PIXELS: u64 = 8_000_000;

/// The dimensions to downscale a page to before the spot call, or `None` when
/// it already fits the budget. Aspect-preserving, never upscales, and both
/// sides stay at least 1. A named function rather than inline arithmetic so
/// the decision has somewhere to be tested -- the context-size refusal was
/// only ever reachable through the one caller that feeds whole pages in.
fn spot_scale_dimensions(width: u32, height: u32) -> Option<(u32, u32)> {
    let pixels = u64::from(width) * u64::from(height);
    if pixels <= SPOT_MAX_PIXELS || pixels == 0 {
        return None;
    }
    let scale = (SPOT_MAX_PIXELS as f64 / pixels as f64).sqrt();
    let scaled_width = ((f64::from(width) * scale) as u32).max(1);
    let scaled_height = ((f64::from(height) * scale) as u32).max(1);
    Some((scaled_width, scaled_height))
}

/// The spotting reply's boxes, tolerant of a code fence, empty on anything else.
/// Failing OPEN is deliberate: a parse failure must never refuse a read.
fn parse_spot_boxes(raw: &str) -> Vec<[f64; 4]> {
    let mut body = raw.trim();
    if let Some(stripped) = body.strip_prefix("```") {
        body = stripped
            .split_once('\n')
            .map_or(stripped, |(_, rest)| rest)
            .trim_end_matches('`')
            .trim();
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return Vec::new();
    };
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let b = item.get("box")?.as_array()?;
            if b.len() != 4 {
                return None;
            }
            let mut out = [0.0; 4];
            for (slot, v) in out.iter_mut().zip(b) {
                *slot = v.as_f64()?;
            }
            Some(out)
        })
        .collect()
}

/// How much of the REGION the best spotting box covers, in [0, 1]. Boxes are
/// norm-1000 against the page; the region's extents are page pixels.
fn spot_coverage(
    extents: (f64, f64, f64, f64),
    boxes: &[[f64; 4]],
    width: u32,
    height: u32,
) -> f64 {
    let (ax0, ay0, ax1, ay1) = extents;
    let area = ((ax1 - ax0) * (ay1 - ay0)).max(1e-6);
    let (w, h) = (f64::from(width), f64::from(height));
    boxes
        .iter()
        .map(|b| {
            let (bx0, by0, bx1, by1) =
                (b[0] / 1000.0 * w, b[1] / 1000.0 * h, b[2] / 1000.0 * w, b[3] / 1000.0 * h);
            let ix = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
            let iy = (ay1.min(by1) - ay0.max(by0)).max(0.0);
            ix * iy / area
        })
        .fold(0.0, f64::max)
}

/// A read short enough to be a candidate artwork fabrication. The measured
/// population is 1-3 glyphs; 4 is head room, and every real 4-glyph read on the
/// test chapter is spotting-covered >= 0.97.
/// Length picks who gets CHECKED, never who gets refused -- real single-glyph
/// effects (`嘭`, `轰`) are exactly what the coverage test protects.
fn spot_suspect(text: &str) -> bool {
    let glyphs = text.chars().filter(|c| !c.is_whitespace()).count();
    glyphs > 0 && glyphs <= 4
}

/// An all-Latin read on a CJK page is a fabrication candidate at ANY length.
///
/// `spot_suspect`'s ceiling assumes fabrications are short, and that was true of
/// the transformers-served sidecar's -- every one measured was 1-3 CJK glyphs.
/// The llama.cpp serving of the same model fabricates LONGER, direct-to-English
/// strings: one test page's clean sky read a six-letter romanized name
/// (`Ren Hao` in the tests), six glyphs, walked past
/// the ceiling, and was lettered across the artwork with the artist's hatch
/// strokes inpainted out from under it. A read whose every scripted letter
/// is Latin on a page declared Chinese, Japanese or Korean is suspicious by
/// construction, so it becomes ELIGIBLE for the coverage check -- which decides
/// nothing by itself: real Latin ink (a brand plate, a URL, a drawn effect) is
/// spotting-covered and letters exactly as before. `None` fires nothing, the
/// same silence rule as `script_mismatch`; digits are unscripted, so a page
/// number is judged on its letters alone. Measured over two chapter arms before
/// shipping: the incremental population of this arm is EXACTLY ONE region in 410
/// page results -- the defect itself.
fn latin_on_cjk(text: &str, source_language: Option<Language>) -> bool {
    if !matches!(
        source_language,
        Some(
            Language::ChineseSimplified
                | Language::ChineseTraditional
                | Language::Japanese
                | Language::Korean
        )
    ) {
        return false;
    }
    let counts = Scripts::of(text);
    counts.latin > 0 && counts.scripted() == counts.latin
}

/// Who actually pays for a spotting call: a short read that is NOT already dead,
/// or an all-Latin read on a CJK page at any length (`latin_on_cjk`, above).
/// The first chapter run fired the gate 40 times and 26 of those were junk
/// (`1`, `H`, `↗`, `？`) that `illegible_text` and the server's own rules refuse
/// anyway -- and because the call fires per PAGE, junk-only pages were paying the
/// whole round-trip for nothing: +18% median page time. Composing with
/// `illegible_text` keeps the call for pages carrying a PLAUSIBLE short read.
///
/// The composed predicate is what both call sites use, and the test calls it too
/// -- testing the halves separately is how a fix once shipped unwired.
///
/// **The BOM case is deliberate and pinned**: a BOM before a bare `！` slips
/// `is_symbol_or_punctuation` in BOTH copies (server and pipeline, contractually
/// identical), so it lettered a bare "!" once. It slips `illegible_text` the same
/// way, therefore STAYS a candidate here, and the spotting gate is what catches
/// it -- the two predicates failing in the same direction is load-bearing.
fn spot_candidate(text: &str, source_language: Option<Language>) -> bool {
    (spot_suspect(text) || latin_on_cjk(text, source_language)) && !illegible_text(text)
}

/// The gate's eligibility exactly as the CALL SITES compose it: a candidate read
/// on a free-standing region. Both loops -- the page-level arming and the
/// per-region refusal -- ask this question, and naming it is what lets a test
/// assert on what the caller calls rather than on the halves.
fn spot_eligible(text: &str, source_language: Option<Language>, role: Option<&str>) -> bool {
    spot_candidate(text, source_language) && free_standing(role)
}

/// The whole refusal decision for one region, exactly as the refusal loop runs
/// it: eligible, geometry readable, and the best spotting box covering less
/// than `SPOT_COVERAGE_FLOOR` of the region. A region with no readable extents
/// is NOT refused -- the loop's original `continue`, preserved.
fn spot_refuses(
    text: &str,
    source_language: Option<Language>,
    role: Option<&str>,
    geometry: &Geometry,
    boxes: &[[f64; 4]],
    width: u32,
    height: u32,
) -> bool {
    spot_eligible(text, source_language, role)
        && geometry_extents(geometry).is_some_and(|extents| {
            spot_coverage(extents, boxes, width, height) < SPOT_COVERAGE_FLOOR
        })
}

/// The region must overlap a spotted text line by at least this fraction of its
/// own area to letter. Measured chapter-wide before being chosen: the twelve
/// fabrications score 0.00 -- literally zero, not merely low -- and 24 of 26 real
/// short reads score >= 0.51. The two exceptions are a BOM-carrying bare `！`
/// (junk either way) and a wrong-word read of a cross-slice glyph the spotting
/// could not see whole -- both adjudicated in pixels as acceptable losses
/// before this constant was written.
const SPOT_COVERAGE_FLOOR: f64 = 0.2;

/// An ILLEGIBLE free-standing read -- too low-confidence to letter or to erase
/// for. Not an artwork discriminator: confidence was measured NOT to separate
/// artwork from real text on PaddleOCR-VL. This IS the legibility signal, on
/// the shipping engine's distribution, and it deliberately accepts one known
/// cost -- a real drawn effect MISREAD scores here too, and refusing that
/// misread is the desired outcome: the drawn glyph STANDS, which is what
/// the test chapter's official edition does with SFX. `None` fires nothing --
/// NOT MEASURED is never zero (manga-ocr, baberu-ocr and Ollama all report `None`), the same silence
/// rule every other consumer of this field keeps.
fn illegible_read(confidence: Option<f32>, role: Option<&str>) -> bool {
    free_standing(role) && confidence.is_some_and(|c| c < ILLEGIBLE_READ_FLOOR)
}

/// **0.55, calibrated across two test chapters; 0.50's empty band turned out to
/// be chapter-local.** On the first chapter alone (79 lettered free-text reads)
/// the illegible class that lettered -- among them `选` 0.167 ("SELECT" over a
/// drawn `轰`) and another `选` at 0.194 -- topped out at 0.456, and 0.50 sat in
/// that chapter's empty band. The FIRST render under 0.50 on the second chapter
/// then found drawn burst art read as a romanized two-word name at **0.5228**,
/// lettered on the artwork. The two-chapter census of lettered free-standing
/// reads in 0.45-0.80: junk at 0.5228 and 0.6642 (a meta-answer, junk by CLASS
/// -- a separate refusal owns it, not this score); real at **0.7308**, 0.7531,
/// 0.7692. 0.55 clears the leak by 0.027 and the lowest real read by 0.15,
/// still under nothing real on either chapter; the lowest correct DIALOGUE read
/// (0.5043) stays out of scope by role. An ABSOLUTE confidence floor is a NEW
/// design decision, not a re-fit: none existed on this path before this one.
const ILLEGIBLE_READ_FLOOR: f32 = 0.55;

/// Kill-switch for the artwork-spotting gate, for A/B runs. Default ON; set
/// `BIRELATE_SPOT_ARTWORK_GATE=off` to disable. An env rather than a CLI flag so
/// the change stays engine-local -- the gate only exists on the VLM route.
fn spot_gate_enabled() -> bool {
    !matches!(
        std::env::var("BIRELATE_SPOT_ARTWORK_GATE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "off" | "0" | "false" | "no"
    )
}

/// MiniCPM-V 4.6 reached over HTTP through Ollama's native `/api/chat`.
///
/// **Determinism is pinned here and it is not decoration.** Every other OCR engine
/// on this path is byte-identical across runs, and that zero A/A floor is what makes
/// every A/B on this project readable. An Ollama-reached model has no such guarantee
/// by default. Measured with exactly these options: **1.000 over 6 crops
/// x 2 requests**, for both MiniCPM-V generations -- the model this route was first
/// built against, and now only one of the tags it can carry.
pub(crate) struct OllamaVision {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    prompt: String,
    /// Whether the probed endpoint advertised the `rotate_ccw` capability.
    /// Set once at load from `/api/tags`; the upright pass is disabled (with a
    /// warning) when it is false, because a silently-ignored rotation option
    /// degrades the sweep into sixteen identical reads with nothing to show it.
    rotate_capable: bool,
}

#[derive(serde::Serialize)]
struct OllamaMessage<'a> {
    role: &'a str,
    content: &'a str,
    /// **This field is how the image reaches the model.** The
    /// translator's OpenAI-shaped `Message` is `{ role, content: &str }` with no
    /// content-parts array, so it cannot carry an image. Ollama's native API does
    /// not use content parts at all -- the image rides here, beside the text.
    images: Vec<String>,
}

#[derive(serde::Serialize)]
struct OllamaOptions {
    temperature: f32,
    seed: u32,
    top_k: u32,
    top_p: f32,
    num_predict: i32,
    /// Degrees the SIDECAR rotates the crop counter-clockwise before reading it,
    /// for the upright pass. The rotation lives sidecar-side, in PIL,
    /// deliberately: `Image.rotate(angle, expand, fillcolor=white, BICUBIC)` is
    /// byte-identical to the apparatus every selection number was measured on,
    /// and no Rust resampler is. `None` is omitted from the wire
    /// entirely, so an ordinary read's request stays byte-identical to what it
    /// was before this field existed -- the A/A floor is untouched.
    #[serde(skip_serializing_if = "Option::is_none")]
    rotate_ccw: Option<f64>,
}

#[derive(serde::Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage<'a>>,
    stream: bool,
    think: bool,
    options: OllamaOptions,
}

#[derive(serde::Deserialize)]
struct OllamaReply {
    message: OllamaReplyMessage,
    /// The sidecar's length-normalised greedy log-prob (the confidence quantity
    /// PaddleOCR-VL reports, for this engine). Absent from real Ollama and from
    /// sidecars predating the extension; `None` then, never 0.0 -- "not
    /// measured" and "measured as zero"
    /// must not collapse. Consumed only by the upright pass's selection rule.
    #[serde(default)]
    score: Option<SidecarScore>,
}

#[derive(serde::Deserialize)]
struct SidecarScore {
    mean_logprob: f64,
}

#[derive(serde::Deserialize)]
struct OllamaReplyMessage {
    #[serde(default)]
    content: String,
}

#[derive(serde::Deserialize)]
struct OllamaTags {
    #[serde(default)]
    models: Vec<OllamaTag>,
    /// Capability markers the sidecar advertises (`"rotate_ccw"`, `"score"`).
    /// Real Ollama sends none. The upright pass REQUIRES `rotate_ccw` here: a
    /// sidecar that ignores the option would silently read the same unrotated
    /// crop sixteen times, and the pass would report a sweep that never happened.
    #[serde(default)]
    capabilities: Vec<String>,
}

#[derive(serde::Deserialize)]
struct OllamaTag {
    #[serde(default)]
    name: String,
}

impl OllamaVision {
    fn new() -> Self {
        let env = |key: &str, fallback: &str| {
            std::env::var(key)
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| fallback.to_string())
        };
        Self {
            client: reqwest::Client::new(),
            endpoint: env("BIRELATE_OLLAMA_VISION_ENDPOINT", OLLAMA_VISION_ENDPOINT),
            model: env("BIRELATE_OLLAMA_VISION_MODEL", OLLAMA_VISION_MODEL),
            prompt: env("BIRELATE_OLLAMA_VISION_PROMPT", OLLAMA_VISION_PROMPT),
            rotate_capable: false,
        }
    }

    /// The same HTTP client pointed at the HunyuanOCR sidecar instead of Ollama.
    /// One wire protocol, two tenants -- the struct is engine-agnostic on purpose,
    /// and a third HTTP-served reader should follow this same two-line pattern.
    fn hunyuan() -> Self {
        let env = |key: &str, fallback: &str| {
            std::env::var(key)
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| fallback.to_string())
        };
        Self {
            client: reqwest::Client::new(),
            endpoint: env("BIRELATE_HUNYUAN_ENDPOINT", HUNYUAN_ENDPOINT),
            model: env("BIRELATE_HUNYUAN_MODEL", HUNYUAN_MODEL),
            prompt: env("BIRELATE_HUNYUAN_PROMPT", HUNYUAN_PROMPT),
            rotate_capable: false,
        }
    }

    /// Fail at LOAD if Ollama is down or the tag is absent, rather than as one
    /// error per region. A 179-slice chapter would otherwise report ~250 identical
    /// failures and bury the one fact that matters. Records the endpoint's
    /// advertised capabilities on the way through, so the upright pass can know
    /// at run time whether `rotate_ccw` would actually rotate anything.
    async fn probe(&mut self) -> Result<()> {
        let base = self
            .endpoint
            .strip_suffix("/api/chat")
            .unwrap_or(&self.endpoint);
        let tags: OllamaTags = self
            .client
            .get(format!("{base}/api/tags"))
            .send()
            .await
            .with_context(|| format!("ollama is not reachable at {}", self.endpoint))?
            .json()
            .await
            .context("ollama /api/tags did not return the expected JSON")?;
        self.rotate_capable = tags.capabilities.iter().any(|c| c == "rotate_ccw");
        if tags.models.iter().any(|tag| tag.name == self.model) {
            return Ok(());
        }
        let installed = tags
            .models
            .iter()
            .map(|tag| tag.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "ollama has no model {:?}; installed: [{}]. Pull it with: ollama pull {}",
            self.model,
            installed,
            self.model
        )
    }

    /// White border added around every crop before it is sent. **This is the fix
    /// for a tight-crop misread, and it is measured, not guessed.** The
    /// pipeline crops at margin ZERO (`crop` = `crop_grown(.., 0)`), which the
    /// in-process engines were tuned on -- PaddleOCR-VL reads the same tight crop
    /// correctly. A VLM does not: a 47x205 px column read `仓炎之王！`
    /// from the exact pipeline crop and `苍炎之王！` from the same pixels with ANY
    /// of 8/16/24 px of margin, white or real-page (a battery of five regions x
    /// seven treatments, zero movement on the four controls).
    ///
    /// **WHITE, not context growth, deliberately**: growing into the page pulls
    /// neighbouring ink into a reader whose instruction is "read ALL text in the
    /// image" -- the battery's context arms appended to one read a drawn `！` that
    /// sits outside the region. A white border can never add foreign text. 16 is
    /// the middle of the three widths that all fixed it, not a tuned edge.
    ///
    /// Engine-local on purpose: geometry, the erase mask, and every other engine
    /// see nothing. Applies to both HTTP tenants (`ollama-vision` too) -- the
    /// margin need is a property of VLM-class readers, not of one model.
    const CROP_PAD_PX: u32 = 16;

    /// Read one crop. **Synchronous on purpose**: `infer_text` runs its closure
    /// inside `tokio::task::spawn_blocking`, so this is already on a blocking
    /// thread and `block_on` is legal there. Going async instead would mean
    /// bypassing `infer_text` and re-implementing the orientation and margin
    /// handling every other engine gets for free.
    fn inference(&self, image: &DynamicImage) -> Result<String> {
        Ok(self
            .request(image, &self.prompt, 128, Self::CROP_PAD_PX, None)?
            .0)
    }

    /// One crop, read WITH the decode's confidence beside it -- `exp(mean_logprob)`,
    /// the same 0..1 quantity PaddleOCR-VL reports, so `choose_orientation`'s
    /// margin compares like with like. `None` when the endpoint returns no score
    /// (real Ollama, or a sidecar without the score extension) -- and an absent
    /// confidence DISABLES the ranking rather than scoring zero, which
    /// `choose_orientation` already guarantees. Without this the margin flag
    /// would be structurally dead on the shipping engine: every turn would log
    /// `upright_confidence=NaN`.
    fn inference_scored(&self, image: &DynamicImage) -> Result<(String, Option<f32>)> {
        let (text, mlp) = self.request(image, &self.prompt, 128, Self::CROP_PAD_PX, None)?;
        Ok((text, mlp.map(|value| value.exp() as f32)))
    }

    /// One crop, read with the sidecar rotating it `rotate_ccw` degrees first.
    /// The upright pass's read primitive: same prompt, same token budget and
    /// the same 16 px white pad as `inference` -- pad-then-rotate is the order
    /// every selection number was measured under. Returns the score beside the
    /// text because the selection rule ranks a crop's OWN angles by it; it must
    /// never be compared across crops.
    fn inference_rotated(
        &self,
        image: &DynamicImage,
        rotate_ccw: f64,
    ) -> Result<(String, Option<f64>)> {
        self.request(image, &self.prompt, 128, Self::CROP_PAD_PX, Some(rotate_ccw))
    }

    /// One page, the official spotting task, boxes back. **This is the artwork
    /// gate's other half** -- see `spot_refusals` in `Model::run` for the rule and
    /// the measurement. Whole page, so NO padding; a dense page can name many
    /// regions, so the token budget is the chapter sweep's, not a crop's.
    ///
    /// The page is DOWNSCALED to [`SPOT_MAX_PIXELS`] first when it is larger:
    /// the spot call is the biggest single request this stack ever makes, its
    /// visual tokens scale with pixels, and the llama.cpp sidecar measured a
    /// 2130x8000 page at **16,712 tokens against `-c 10240`** -- `request
    /// (16712 tokens) exceeds the available context size (10240 tokens)`. The
    /// scale-down is lossless to every consumer BY THE PROTOCOL: the reply's
    /// boxes are normalized to `[0, 1000]` and `denormalize_spot_box` maps
    /// them onto the ORIGINAL page dimensions, which this function never
    /// changes. Both call sites fail open on an error regardless; this makes
    /// the refusal unreachable rather than merely survivable.
    fn spot_page(&self, page: &DynamicImage) -> Result<Vec<[f64; 4]>> {
        let raw = match spot_scale_dimensions(page.width(), page.height()) {
            Some((width, height)) => {
                let scaled = page.resize_exact(width, height, image::imageops::FilterType::Triangle);
                self.request(&scaled, HUNYUAN_SPOT_PROMPT, 1536, 0, None)?.0
            }
            None => self.request(page, HUNYUAN_SPOT_PROMPT, 1536, 0, None)?.0,
        };
        Ok(parse_spot_boxes(&raw))
    }

    fn request(
        &self,
        image: &DynamicImage,
        prompt: &str,
        num_predict: i32,
        pad: u32,
        rotate_ccw: Option<f64>,
    ) -> Result<(String, Option<f64>)> {
        let mut canvas = image::RgbImage::from_pixel(
            image.width() + 2 * pad,
            image.height() + 2 * pad,
            image::Rgb([255, 255, 255]),
        );
        image::imageops::overlay(&mut canvas, &image.to_rgb8(), i64::from(pad), i64::from(pad));
        let mut png = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(canvas)
            .write_to(&mut png, image::ImageFormat::Png)
            .context("encoding the crop as PNG for ollama")?;
        let encoded = {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.encode(png.into_inner())
        };

        let body = OllamaRequest {
            model: &self.model,
            messages: vec![OllamaMessage {
                role: "user",
                content: prompt,
                images: vec![encoded],
            }],
            stream: false,
            // 4.6 is a thinking model. Reasoning is OFF because it was measured and
            // it does not help: 0 of 12 on the drawn glyphs with it enabled, while
            // costing hundreds of tokens a region across ~250 regions a chapter.
            think: false,
            options: OllamaOptions {
                temperature: 0.0,
                seed: 42,
                top_k: 1,
                top_p: 1.0,
                num_predict,
                rotate_ccw,
            },
        };

        let request = self.client.post(&self.endpoint).json(&body);
        let reply: OllamaReply = tokio::runtime::Handle::current().block_on(async move {
            let response = request.send().await.context("posting the crop to ollama")?;
            let status = response.status();
            let text = response
                .text()
                .await
                .context("reading ollama's response body")?;
            if !status.is_success() {
                bail!(
                    "ollama returned {status}: {}",
                    text.chars().take(512).collect::<String>()
                );
            }
            serde_json::from_str(&text).with_context(|| {
                format!(
                    "ollama returned unparseable JSON: {}",
                    text.chars().take(512).collect::<String>()
                )
            })
        })?;

        Ok((
            reply.message.content.trim().to_string(),
            reply.score.map(|s| s.mean_logprob),
        ))
    }
}


pub(super) struct Processor {
    config: OcrModel,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
    large_crop_px: Option<u32>,
    implausible: ImplausibleRegions,
    /// Whether a region OCR could not read is taken back out of the erase mask
    /// before the inpainter sees it. See `withdraw_illegible_from_mask`.
    withdraw_illegible_masks: bool,
    /// What the page is written in, when the caller knows. Feeds the script arm
    /// of `withdraw_from_mask` and **nothing else** -- see the field's own doc
    /// comment on `TranslationConfig`.
    source_language: Option<Language>,
    /// Whether a tall kana-free free-text column is read a SECOND time, turned
    /// 90 CCW, and the turned read preferred. **OFF by default.**
    reread_rotated_columns: bool,
    /// Whether a watermark verdict is scoped to the site's own text rather than
    /// condemning the whole region. **OFF by default.**
    scope_watermark_refusals: bool,
    /// How much MORE confident the upright read must be to beat the turned one.
    /// `None` disables the ranking and is the default.
    orientation_margin: Option<f64>,
    /// Pixels to grow an onomatopoeia crop by before reading it a SECOND time, to
    /// measure whether the read survives a small change of geometry. `None`
    /// disables it and is the default. **Reported, never acted on.**
    perturb_grow_px: Option<NonZeroU32>,
    /// Whether a free-standing region whose read came back EMPTY is re-read
    /// through the sidecar's rotation sweep and `upright_select`.
    /// **OFF by default.** HTTP-sidecar route only.
    upright_pass: bool,
    /// Whether a SYNTHESISED bubble read (`--read-textless-bubbles`) is read a
    /// SECOND time with the crop turned 180 degrees, keeping the flipped read
    /// only when it clearly beats the upright one. **OFF by default.**
    flip_reread_bubbles: bool,
    /// Whether a sparse or decline-carrying page may buy ONE spotting call and
    /// mint regions for display runs the detector never boxed.
    /// **OFF by default.** Sidecar route only.
    spot_rescue: bool,
    /// Whether a shipped spot rescue also joins the erase mask.
    /// **OFF by default**, adjudicated separately from the lettering.
    spot_rescue_erase: bool,
    /// Whether a WIDE scream-read mint becomes the mark-replacement device:
    /// erase scoped to the drawn mark's enclosed ink instead of the hull, one
    /// styled gradient replacement lettered along the ink's own axis.
    /// **OFF by default**, and inert unless `spot_rescue_erase` is
    /// also on -- the device is a modification of the mint's erase, and
    /// styling without the erase letters over the intact drawn mark.
    replace_scream_marks: bool,
    /// Whether a JOINED page (a seam composite) may buy the spot call too,
    /// regardless of its region count -- a composite exists because something
    /// crosses the cut, and a display run cut by the slice boundary is exactly
    /// the rescue's population while the sparse trigger structurally misses it
    /// (the composite carries its neighbours' ordinary regions).
    /// **OFF by default.**
    spot_rescue_joined: bool,
    /// Whether the script arm of `withdraw_from_mask` reaches DIALOGUE-role
    /// regions too. The pixel half of a paired lever; the lettering half reads
    /// the same flag in `birelate-server`. **Ships ON** (the default is
    /// `cli.rs`'s to set, not this struct's).
    leave_misread_bubbles: bool,
    /// Whether a declared-Korean read with no hangul counts as a script
    /// mismatch (`strict_korean_mismatch`). **Ships ON**, with its pair above.
    korean_script_strict: bool,
    /// Whether a DIALOGUE-role read the misread lever refuses on declared
    /// Korean buys ONE reserve-engine re-read on a padded crop, admitted iff
    /// hangul-majority. **Ships ON** (the default is `cli.rs`'s to set, not
    /// this struct's).
    reread_refused_dialogue: bool,
}

impl Processor {
    pub(super) fn new(
        config: OcrModel,
        device: koharu_ml::Device,
        large_crop_px: Option<u32>,
        implausible: ImplausibleRegions,
        withdraw_illegible_masks: bool,
        source_language: Option<Language>,
        reread_rotated_columns: bool,
        scope_watermark_refusals: bool,
        orientation_margin: Option<f64>,
        perturb_grow_px: Option<NonZeroU32>,
        upright_pass: bool,
        flip_reread_bubbles: bool,
        spot_rescue: bool,
        spot_rescue_erase: bool,
        replace_scream_marks: bool,
        spot_rescue_joined: bool,
        leave_misread_bubbles: bool,
        korean_script_strict: bool,
        reread_refused_dialogue: bool,
    ) -> Self {
        // Routing is meaningless when PaddleOCR-VL is already the primary: there
        // is no square to squash into, so every crop would route to the model
        // that is already running. Zero is accepted as "off" so a caller can
        // disable it without an Option round-trip.
        // MiniCPM-V is listed BESIDE PaddleOCR-VL rather than falling into the
        // wildcard, and it is the whole reason this match is spelled out. The `_`
        // arm silently switched the fallback ON for any engine added later, so
        // every crop >= `large_crop_px` would have been re-read by PaddleOCR-VL and
        // the arm under test would never have seen the biggest regions on the page
        // -- which on this corpus are exactly the drawn sound effects.
        let large_crop_px = match config {
            // HunyuanOCR joins the aspect-preserving side: its `smart_resize`
            // (Qwen2-VL-style, min 512^2, max 4096^2, snapped to multiples of 32)
            // never squashes a crop, so the oversized-crop reroute to PaddleOCR-VL
            // would only trade one aspect-preserving reader for another.
            OcrModel::PaddleOcrVl1_6 | OcrModel::OllamaVision | OcrModel::HunyuanOcr1_5 => None,
            OcrModel::MangaOcr | OcrModel::BaberuOcr => large_crop_px.filter(|limit| *limit > 0),
        };
        Self {
            config,
            device,
            model: ModelCell::new(),
            large_crop_px,
            implausible,
            withdraw_illegible_masks,
            source_language,
            reread_rotated_columns,
            scope_watermark_refusals,
            orientation_margin,
            perturb_grow_px,
            upright_pass,
            flip_reread_bubbles,
            spot_rescue,
            spot_rescue_erase,
            replace_scream_marks,
            spot_rescue_joined,
            leave_misread_bubbles,
            korean_script_strict,
            reread_refused_dialogue,
        }
    }

    /// The two misread levers as one value, so every call site carries both
    /// or neither -- the pairing is the point of the struct.
    const fn misread_levers(&self) -> MisreadLevers {
        MisreadLevers {
            leave_misread_bubbles: self.leave_misread_bubbles,
            korean_script_strict: self.korean_script_strict,
        }
    }
}

/// The upper bound on a detection box, and what to do with one that breaks it.
///
/// **Why an upper bound exists at all.** `large_crop_px` is a lower bound with
/// no ceiling, and RF-DETR does not only mis-*classify*, it mis-*segments*: on
/// one real page it returned a single `onomatopoeia` box of 1210x1247 on an
/// 844x1200 page -- **1.49x the area of the page it was detected on** -- at
/// confidence 0.43, over artwork. That box cleared the 448 lower bound, went to
/// PaddleOCR-VL, and returned zero characters.
///
/// It was not free. The crop is ~60x a typical text region, so libtorch's CUDA
/// caching allocator raised its high-water mark ~3.7 GB to serve it, and
/// libtorch never returns cached segments to the driver -- there is no
/// `empty_cache` anywhere in the FFI surface. On Windows admission control
/// budgets on DXGI `CurrentUsage`, which counts cached-but-free bytes, so
/// `available_bytes` fell under `safety_reserve + reservation` for the rest of
/// the process and `unload_idle` began evicting the 16.5 GiB translation model
/// to admit a 153 MiB detector. Measured cost: page median 5,189 ms before,
/// 12,790 ms after, permanently. **One bad box set the floor for the whole
/// run**, which is why the ceiling belongs here rather than in a size-aware
/// allocator we do not have.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ImplausibleRegions {
    /// Fraction of the page's own area at which a detection box stops being a
    /// credible text region. `None` is off, and is the state in which this file
    /// behaves exactly as it did before the bound existed.
    pub(super) max_area: Option<f32>,
    /// Whether a box over the bound is read by **no** engine, rather than merely
    /// being kept off PaddleOCR-VL. Off by default; see `implausible_region`.
    pub(super) skip: bool,
    /// Whether a box `skip` refused is also taken back out of the **erase mask**.
    ///
    /// # The interlock this exists to break
    ///
    /// `skip` and `withdraw_illegible_masks` both default ON and **starve each
    /// other on exactly this population**. `write_illegible_veto` walks `pending`,
    /// which holds OCR *results*; a box `skip` refused never becomes one, so
    /// `withdraw_from_mask` can never evaluate any of its arms for it. The
    /// reader's refusal deletes the very verdict the withdrawal needs.
    ///
    /// Measured on a 457-page benchmark carrying 22 such boxes:
    /// **18 are erased with nothing painted back.** Turning `skip` off instead
    /// rescues only **8** of them and costs a regression — it lets the same junk be
    /// read and lettered, painting `SONG` across intact artwork on one page. The 8
    /// it rescues are exactly those whose read fires `withdraw_from_mask`, which is
    /// the clue: the withdrawal is the right mechanism and the read is not needed
    /// to justify it. **A region no engine was allowed to read is a strictly
    /// stronger case for withdrawal than one that was read and came back
    /// unreadable**, which is what `withdraw_illegible_masks` already acts on.
    ///
    /// **ON by default**, after the render fixed 18 of 18 with 0
    /// broken and nothing lettered, and a blind panel agreed **15 of 15** — including
    /// a seat instructed to argue against it. See `ProcessorConfig` for the numbers.
    pub(super) veto_mask: bool,
}

/// The box the detector actually claimed, in page pixels.
///
/// Falls back to the crop only if the geometry is empty, which `crop` has
/// already refused by the time a target exists -- but the crop is the right
/// thing to fall back to, because it is what the model would be handed.
fn region_extent(target: &OcrTarget) -> (f64, f64) {
    geometry_extents(&target.geometry)
        .map(|(min_x, min_y, max_x, max_y)| (max_x - min_x, max_y - min_y))
        .unwrap_or_else(|| {
            (
                f64::from(target.image.width()),
                f64::from(target.image.height()),
            )
        })
}

/// PaddleOCR-VL, held beside the primary for crops the primary cannot resolve.
///
/// **It lives inside the primary's `ModelCell`, and that placement is the whole
/// design.** `Stages::unload` reaches exactly one `ModelState` per stage
/// (`ModelRef` wraps a single `&dyn ModelState`), so a second `ModelCell` here
/// would be invisible to residency and would strand ~1.9 GB that nothing could
/// free. Dropping the primary `Model` drops this with it.
///
/// Loaded on the first oversized crop rather than at stage load, because the
/// measured trigger rate on real pages is 0.7% of regions -- most sessions never
/// pay for it at all.
struct Fallback {
    device: koharu_ml::Device,
    threshold: Option<u32>,
    model: tokio::sync::OnceCell<Arc<Mutex<PaddleOCRVL>>>,
}

/// Whether this crop is too big for a fixed-square recogniser to resolve.
///
/// The LARGER side is the right term because the squash is to a SQUARE: a
/// 59x497 vertical column is downscaled on its long axis exactly as a 497x59
/// banner would be, and both were seen among the three oversized regions on 40
/// real pages. Compared with `>=`, so a threshold equal to the encoder input
/// reads as "anything that must be downscaled at all".
///
/// Free-standing rather than a method so it is testable without a `Device`.
fn oversized(width: u32, height: u32, threshold: Option<u32>) -> bool {
    threshold.is_some_and(|limit| width.max(height) >= limit)
}

impl Fallback {
    fn new(device: koharu_ml::Device, threshold: Option<u32>) -> Self {
        Self {
            device,
            threshold,
            model: tokio::sync::OnceCell::new(),
        }
    }

    fn claims(&self, image: &DynamicImage) -> bool {
        oversized(image.width(), image.height(), self.threshold)
    }

    async fn ensure(&self) -> Result<Arc<Mutex<PaddleOCRVL>>> {
        let model = self
            .model
            .get_or_try_init(|| async {
                // Loud on purpose. This can pull ~1.9 GB the first time, and
                // koharu's downloader publishes progress only on a channel the
                // GUI reads, so an unannounced fetch here would look like a hang.
                tracing::info!(
                    threshold = self.threshold,
                    "loading paddleocr-vl-1.6 for an oversized text region"
                );
                PaddleOCRVL::load(self.device.clone())
                    .await
                    .map(|model| Arc::new(Mutex::new(model)))
            })
            .await?;
        Ok(model.clone())
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> ModelRef<'_> {
        let name = match self.config {
            OcrModel::MangaOcr => "manga-ocr",
            OcrModel::BaberuOcr => "baberu-ocr",
            OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
            OcrModel::OllamaVision => OLLAMA_VISION_NAME,
            OcrModel::HunyuanOcr1_5 => HUNYUAN_NAME,
        };
        ModelRef::new(name, &self.model)
    }

    async fn load(&self) -> Result<()> {
        self.model
            .ensure(|| {
                Model::load(
                    self.device.clone(),
                    &self.config,
                    self.large_crop_px,
                    self.implausible,
                    self.withdraw_illegible_masks,
                    self.source_language,
                    self.reread_rotated_columns,
                    self.scope_watermark_refusals,
                    self.orientation_margin,
                    self.perturb_grow_px,
                    self.upright_pass,
                    self.flip_reread_bubbles,
                    self.spot_rescue,
                    self.spot_rescue_erase,
                    self.replace_scream_marks,
                    self.spot_rescue_joined,
                    self.misread_levers(),
                    self.reread_refused_dialogue,
                )
            })
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("OCR model is not loaded"))?
            .run(input)
            .await
    }
}

struct Model {
    primary: Primary,
    fallback: Fallback,
    implausible: ImplausibleRegions,
    /// Carried down here rather than read off `Processor` because `run` is a
    /// method on the loaded model, not on the stage.
    withdraw_illegible_masks: bool,
    /// Carried down for the same reason. Feeds the script arm of
    /// `withdraw_from_mask` and nothing else -- **not** the translator; see
    /// `TranslationConfig::source_language`.
    source_language: Option<Language>,
    /// Carried down for the same reason. **OFF by default.**
    reread_rotated_columns: bool,
    /// Carried down for the same reason. **OFF by default.**
    scope_watermark_refusals: bool,
    /// How much MORE confident the upright read must be for it to beat the turned
    /// one. `None` disables the ranking entirely and is the default; see
    /// `choose_orientation`.
    orientation_margin: Option<f64>,
    /// Carried down for the same reason. **OFF by default**, and
    /// reported rather than acted on.
    perturb_grow_px: Option<NonZeroU32>,
    /// Carried down for the same reason. **OFF by default.**
    upright_pass: bool,
    /// Carried down for the same reason. **OFF by default.**
    flip_reread_bubbles: bool,
    /// Carried down for the same reason. **OFF by default.**
    spot_rescue: bool,
    /// Carried down for the same reason. **OFF by default.**
    spot_rescue_erase: bool,
    /// Carried down for the same reason. **OFF by default.**
    replace_scream_marks: bool,
    /// Carried down for the same reason. **OFF by default.**
    spot_rescue_joined: bool,
    /// The paired misread levers, carried down together -- see `MisreadLevers`.
    /// The default is `cli.rs`'s to set.
    misread_levers: MisreadLevers,
    /// Carried down for the same reason. The default is `cli.rs`'s to set.
    reread_refused_dialogue: bool,
}

enum Primary {
    Manga(Arc<Mutex<MangaOcr>>),
    Baberu(Arc<Mutex<BaberuOcr>>),
    Paddle(Arc<Mutex<PaddleOCRVL>>),
    /// An Ollama-served vision model. Holds no weights in this process -- the
    /// `Mutex` exists only because `infer_text` is generic over `Arc<Mutex<M>>`.
    Ollama(Arc<Mutex<OllamaVision>>),
    /// HunyuanOCR behind the local sidecar. Same struct, same protocol, different
    /// tenant -- a separate variant only so `name()` stays lock-free and `&'static`.
    Hunyuan(Arc<Mutex<OllamaVision>>),
}

impl Model {
    async fn load(
        device: koharu_ml::Device,
        config: &OcrModel,
        large_crop_px: Option<u32>,
        implausible: ImplausibleRegions,
        withdraw_illegible_masks: bool,
        source_language: Option<Language>,
        reread_rotated_columns: bool,
        scope_watermark_refusals: bool,
        orientation_margin: Option<f64>,
        perturb_grow_px: Option<NonZeroU32>,
        upright_pass: bool,
        flip_reread_bubbles: bool,
        spot_rescue: bool,
        spot_rescue_erase: bool,
        replace_scream_marks: bool,
        spot_rescue_joined: bool,
        misread_levers: MisreadLevers,
        reread_refused_dialogue: bool,
    ) -> Result<Self> {
        let primary = Primary::load(device.clone(), config).await?;
        Ok(Self {
            primary,
            fallback: Fallback::new(device, large_crop_px),
            implausible,
            withdraw_illegible_masks,
            source_language,
            reread_rotated_columns,
            scope_watermark_refusals,
            orientation_margin,
            perturb_grow_px,
            upright_pass,
            flip_reread_bubbles,
            spot_rescue,
            spot_rescue_erase,
            replace_scream_marks,
            spot_rescue_joined,
            misread_levers,
            reread_refused_dialogue,
        })
    }
}

impl Primary {
    async fn load(device: koharu_ml::Device, config: &OcrModel) -> Result<Self> {
        match config {
            OcrModel::MangaOcr => Ok(Self::Manga(Arc::new(Mutex::new(
                MangaOcr::load(device).await?,
            )))),
            OcrModel::BaberuOcr => Ok(Self::Baberu(Arc::new(Mutex::new(
                BaberuOcr::load(device).await?,
            )))),
            OcrModel::PaddleOcrVl1_6 => Ok(Self::Paddle(Arc::new(Mutex::new(
                PaddleOCRVL::load(device).await?,
            )))),
            // `device` is ignored on purpose: the weights are Ollama's, on the same
            // card, and this process never sees them. Reachability is checked here
            // rather than on the first crop so a wrong endpoint fails at load with
            // a clear message instead of as 250 identical per-region errors.
            OcrModel::OllamaVision => {
                let mut engine = OllamaVision::new();
                engine.probe().await?;
                Ok(Self::Ollama(Arc::new(Mutex::new(engine))))
            }
            // Same probe-at-load rationale: a sidecar that is not running fails
            // here with one clear message naming the endpoint, not as ~250
            // identical per-region errors across a chapter.
            OcrModel::HunyuanOcr1_5 => {
                let mut engine = OllamaVision::hunyuan();
                engine.probe().await?;
                Ok(Self::Hunyuan(Arc::new(Mutex::new(engine))))
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Manga(_) => "manga-ocr",
            Self::Baberu(_) => "baberu-ocr",
            Self::Paddle(_) => "paddleocr-vl-1.6",
            Self::Ollama(_) => OLLAMA_VISION_NAME,
            Self::Hunyuan(_) => HUNYUAN_NAME,
        }
    }

    /// The HTTP client behind this engine, if it is one of the VLM tenants. The
    /// artwork-spotting gate exists only on that route: the in-process engines
    /// have no spotting task to ask, and their fabrication class (kana) is
    /// already caught by the script rules.
    fn vlm(&self) -> Option<Arc<Mutex<OllamaVision>>> {
        match self {
            Self::Ollama(model) | Self::Hunyuan(model) => Some(model.clone()),
            _ => None,
        }
    }

    async fn infer(
        &self,
        targets: Vec<OcrTarget>,
        orientation_margin: Option<f64>,
    ) -> Result<Vec<OcrResult>> {
        match self {
            // `None`, not `Some(0.0)`: these two engines expose no confidence, and
            // scoring an absent one as zero would hand every turn to the upright
            // read on no evidence. See `choose_orientation`.
            Self::Manga(model) => {
                infer_text(
                    model.clone(),
                    targets,
                    |model, image| Ok((model.inference(image)?, None)),
                    None::<RotateFn<MangaOcr>>,
                    orientation_margin,
                )
                .await
            }
            Self::Baberu(model) => {
                infer_text(
                    model.clone(),
                    targets,
                    |model, image| Ok((model.inference(image)?, None)),
                    None::<RotateFn<BaberuOcr>>,
                    orientation_margin,
                )
                .await
            }
            Self::Paddle(model) => {
                infer_text(
                    model.clone(),
                    targets,
                    |model, image| {
                        let result = model.inference(image, PaddleOCRVLTask::Ocr)?;
                        Ok((result.text, result.confidence))
                    },
                    // Scored but NOT rotate-capable: paddle has no server-side
                    // rotation, so its flip stays the single exact transpose.
                    None::<RotateFn<PaddleOCRVL>>,
                    orientation_margin,
                )
                .await
            }
            // `None` for confidence, for the same reason as `manga-ocr` above:
            // Ollama's `/api/chat` returns no logprobs, so there is nothing to
            // report, and scoring an absent confidence as 0.0 would hand every
            // orientation decision to the upright read on no evidence.
            Self::Ollama(model) => {
                infer_text(
                    model.clone(),
                    targets,
                    |model, image| Ok((model.inference(image)?, None)),
                    // Ollama's /api/chat has no rotate option; the capability
                    // probe already scopes every rotated path to the sidecar.
                    None::<RotateFn<OllamaVision>>,
                    orientation_margin,
                )
                .await
            }
            // The sidecar DOES return the decode's score, and
            // `inference_scored` surfaces it as the same `exp(mean_logprob)`
            // confidence PaddleOCR-VL reports -- with the margin unset (the
            // default) `choose_orientation` ignores it, so this is
            // byte-identical to an unscored read until the flag is used.
            Self::Hunyuan(model) => {
                infer_text(
                    model.clone(),
                    targets,
                    |model, image| model.inference_scored(image),
                    // The flip's refine grid rides the sidecar's own rotation --
                    // the exact instrument its refine cells were measured
                    // through -- and converts its raw `mean_logprob` to the
                    // same `exp` confidence every other read reports.
                    Some(
                        |model: &OllamaVision, image: &DynamicImage, angle: f64| {
                            let (text, mlp) = model.inference_rotated(image, angle)?;
                            Ok((text, mlp.map(|value| value.exp() as f32)))
                        },
                    ),
                    orientation_margin,
                )
                .await
            }
        }
    }
}

impl Model {
    async fn run(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let model_name = self.primary.name();
        let page = input.page;
        let mut targets = Vec::new();
        let source = input
            .images
            .get(&input.scene, page, "source")?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;
        for entity in input.scene.descendants(page)? {
            let region = entity.id();
            if !input.contains_entity(region)? {
                continue;
            }
            // Bind the whole component rather than testing `.kind` and dropping it.
            // `Region.label` is the ONLY carrier of the detector's own class here:
            // `region_kind` collapses onomatopoeia into `TextRegion` whenever
            // `translate_sfx` is on -- which is the default -- so a filter written
            // against `.kind` selects every text region and looks like it works.
            let region_component = input.scene.component::<Region>(region)?;
            let is_text_region = region_component
                .as_ref()
                .is_some_and(|value| value.kind == TextRegion::kind());
            if !is_text_region {
                continue;
            }
            let is_onomatopoeia = region_component
                .as_ref()
                .and_then(|value| value.label.as_deref())
                == Some(ONOMATOPOEIA_LABEL);
            let geometry = input
                .scene
                .component::<Geometry>(region)?
                .ok_or_else(|| anyhow!("text region {region} has no geometry"))?;
            let crop = crop(&source, &geometry)
                .with_context(|| format!("text region {region} is outside its source image"))?;
            // The grown twin, cut here because `source` does not cross the
            // `spawn_blocking` boundary into `infer_text` and `OcrTarget` carries
            // the cut image rather than the page.
            let grown = grown_twin(
                &source,
                &geometry,
                &crop,
                self.perturb_grow_px,
                is_onomatopoeia,
            )
            .with_context(|| format!("text region {region} is outside its source image"))?;
            if grown.is_none() && self.perturb_grow_px.is_some() && is_onomatopoeia {
                tracing::debug!(
                    region = %region,
                    width = crop.width(),
                    height = crop.height(),
                    "perturbation re-read skipped: the clamp refused to grow the box"
                );
            }
            for relation in input.scene.relations_to_as::<RecognizedFrom>(region) {
                let content = relation.value().source;
                let previous = input.scene.component::<SourceText>(content)?;
                if previous
                    .as_ref()
                    .is_some_and(|value| matches!(value.text.origin, Origin::User))
                {
                    continue;
                }
                // Read fallibly and outside a closure, for the reason
                // `write_illegible_veto` spells out: treating a scene-read error
                // as "not free-standing" turns a broken lookup into a quietly
                // disabled rule.
                let role = input.scene.component::<TextRole>(content)?;
                let reread_rotated = self.reread_rotated_columns
                    && rotation_candidate(
                        &geometry,
                        free_standing(role.as_ref().map(|value| value.role.as_str())),
                    );
                let reread_flipped = flip_candidate(
                    &input.scene,
                    region,
                    self.flip_reread_bubbles,
                );
                targets.push(OcrTarget {
                    content,
                    region,
                    geometry: geometry.clone(),
                    previous,
                    reread_rotated,
                    reread_flipped,
                    // Cut here, like `grown`, because the flip's second read
                    // happens past the `spawn_blocking` boundary.
                    flip_crop: if reread_flipped {
                        isolate_balloon(&crop)
                    } else {
                        None
                    },
                    image: crop.clone(),
                    grown: grown.clone(),
                });
            }
        }

        // The spot rescue's sparse-page trigger reads the count BEFORE the split
        // consumes the list: every missed-display page in the census
        // proposed 0-2 detector regions.
        let page_region_count = targets.len();
        // Split before inference, not after: a crop the primary cannot resolve
        // does not fail, it returns fluent invented prose, so there is nothing in
        // the output to route on afterwards.
        //
        // Three ways out rather than a `partition`, because the upper bound is
        // not the lower bound's negation. `implausible_region` is evaluated on
        // its own and not inside `Fallback::claims`: a box larger than its page
        // is a detector error whether or not routing is configured at all, so
        // `large-crop-ocr-px = 0` must not silently turn the ceiling off too.
        let page_size = (source.width(), source.height());
        let mut kept = Vec::new();
        let mut routed = Vec::new();
        let mut refused = 0usize;
        // Bounds of the boxes `skip` drops, so the erase mask can be told about a
        // region no engine was ever allowed to read. Collected here because this is
        // the only place that knows a target was refused for SIZE -- one step later
        // it is indistinguishable from a box the detector never produced.
        let mut unread = Vec::new();
        /* A page the caller ASSEMBLED so its text ends inside it. The size guard
         * below refuses a region reaching both edges because on an ordinary
         * slice that is a fragment of something taller -- but a webtoon seam is
         * cut from a run of slices precisely to hold such a name whole, so there
         * the same shape is the join having worked.
         *
         * Measured on a test chapter: the joined column is 1693px of a 1716px seam,
         * 0.987 against the 0.95 threshold, so every run-joined skill name was
         * refused and left in Chinese. Padding the crop does not help -- the
         * detector's box grows with it (1784 of 1816) -- which is why this is a
         * fact the caller has to state rather than one the geometry can infer.
         *
         * ONLY the cross-slice arm is disarmed. `max_area` still applies: a box
         * larger than its page is a detector error on any page, joined or not. */
        let joined_page = input.joined_page();
        // Refused free-standing targets held for the upright pass instead of
        // being dropped outright. THE LESSON OF THE FIRST RENDER:
        // the census's "never-read" boxes die HERE, in `refuse_region`, before
        // any read exists -- a pass triggered on empty READS swept nothing over
        // 81 slices, because on the slice pages the empty-source regions never
        // reach `pending` at all. Every target stashed here reaches a terminal
        // state in the pass block below: recovered into `pending`, or restored
        // to `unread` exactly as if the stash never happened.
        let mut upright_refused: Vec<OcrTarget> = Vec::new();
        // The bounds of every STASH target whose sweep SHIPPED a read.
        // Detection's mask never carried these boxes (they were refused before any
        // read existed), so an ordinary "stop withdrawing" cannot erase them --
        // their ink was never in the mask to withdraw from. A shipped recovery
        // publishes them through `write_rescue_mask` instead; a declined sweep
        // adds NOTHING here, so the refusal's veto is preserved byte-for-byte.
        let mut upright_rescued: Vec<(f64, f64, f64, f64)> = Vec::new();
        // Wide scream mints whose erase is scoped to the drawn
        // mark's own enclosed ink rather than the hull. Joined into the same
        // `text-mask-rescue` asset as `upright_rescued`, by the same writer.
        let mut scream_rescued: Vec<ScreamMarkInk> = Vec::new();
        // The bounds of every stash target whose sweep DECLINED --
        // the spot rescue's other trigger, and part of its overlap fence. The
        // declined target itself is untouched: it keeps its veto and its
        // refusal exactly as today, and a rescue is always a NEW entity.
        let mut upright_declined: Vec<(f64, f64, f64, f64)> = Vec::new();
        for target in targets {
            if refuse_region(
                region_extent(&target),
                page_size,
                self.implausible.max_area,
                joined_page,
            ) {
                refused += 1;
                let stash = self.implausible.skip && self.upright_pass && {
                    // Fallible and outside a closure, like every role read in
                    // this file: a scene error must not quietly disable the arm.
                    let role = input.scene.component::<TextRole>(target.content)?;
                    free_standing(role.as_ref().map(|value| value.role.as_str()))
                };
                if stash {
                    upright_refused.push(target);
                } else {
                    if self.implausible.skip && self.implausible.veto_mask {
                        if let Some(bounds) = geometry_extents(&target.geometry) {
                            unread.push(bounds);
                        }
                    }
                    // The conservative arm keeps reading it, just never with the
                    // engine whose allocator high-water mark is the defect. It is
                    // the default because the failure it avoids is measured and the
                    // failure it risks is not: a box this large is *usually* large
                    // because its glyphs are, and then the primary's 224 square is
                    // benign. `skip` is the arm to measure -- see the flag's doc.
                    if !self.implausible.skip {
                        kept.push(target);
                    }
                }
                continue;
            }
            if self.fallback.claims(&target.image) {
                routed.push(target);
            } else {
                kept.push(target);
            }
        }
        if refused > 0 {
            /* `warn` rather than `info`: unlike the routing below, this fires
             * only when the detector has produced a box that cannot be right.
             *
             * The submitted image's own dimensions are here because the
             * denominator is whatever was POSTed, not a canonical page, and that
             * distinction decides whether a refusal is the rule working or the
             * rule misapplied. The threshold was tuned on full pages, where the
             * separation is wide: over 14,877 regions of an 844x1200 volume the
             * largest legitimate box is 0.30 of its page and the one mis-segmented
             * outlier is 1.49, with nothing between. But the extension's webtoon
             * seam posts a third image per cut that is CROPPED AROUND ITS OWN
             * SUBJECT, so the bubble it exists to rejoin is a large fraction of it
             * by construction -- measured, 6 of 28 real seam crops carry a region
             * at or above the 0.5 default, two of them real dialogue. Those crops
             * are 1280x1000 and larger, so they cannot be told from a page by area
             * alone; only the shape and the count can, and only if they are
             * logged.
             *
             * The old message said "larger than the page they were found on",
             * which was true only at a ceiling of 1.0 and has been the wrong
             * sentence since the default became 0.5. `targets` rather than
             * `regions` because that is what the loop counts -- one per
             * `RecognizedFrom` relation -- which is one per region today and need
             * not stay that way. */
            tracing::warn!(
                targets = refused,
                max_area = ?self.implausible.max_area,
                skipped = self.implausible.skip,
                image_width = page_size.0,
                image_height = page_size.1,
                page = %page,
                "refusing detection regions too large a share of the image they were found in"
            );
        }

        let generation = generation(PRODUCER, model_name)?;
        // Each result carries the provenance of the model that actually read it,
        // so the scene records the routing rather than crediting every region to
        // the configured engine. The edit's own author stays the configured
        // engine, which is what `verify_applied` and `/status` report.
        let mut pending: Vec<(OcrResult, Generation)> = self
            .primary
            .infer(kept, self.orientation_margin)
            .await?
            .into_iter()
            .map(|result| (result, generation.clone()))
            .collect();

        if !routed.is_empty() {
            tracing::info!(
                regions = routed.len(),
                threshold = self.fallback.threshold,
                primary = model_name,
                "routing oversized text regions to paddleocr-vl-1.6"
            );
            let routed_generation = super::generation(PRODUCER, FALLBACK_MODEL)?;
            let model = self.fallback.ensure().await?;
            /* THE FALLBACK IS A SECOND, TEXTUALLY SEPARATE PADDLE CALL SITE, and it
             * is the one the large display regions actually go through. A change
             * made only at the primary above would leave every routed region on
             * `None` while the primaries reported a real number -- half-wired, and
             * invisible to any test that exercises only the primary path. */
            let results = infer_text(
                model,
                routed,
                |model, image| {
                    let result = model.inference(image, PaddleOCRVLTask::Ocr)?;
                    Ok((result.text, result.confidence))
                },
                // Same as the primary paddle arm: scored, not rotate-capable.
                None::<RotateFn<PaddleOCRVL>>,
                self.orientation_margin,
            )
            .await?;
            pending.extend(
                results
                    .into_iter()
                    .map(|result| (result, routed_generation.clone())),
            );
        }

        /* THE UPRIGHT PASS. A refused box's text is usually rotated far outside
         * the ±45° the pipeline can represent; stood upright, the shipping
         * engine reads 14/17 of that population against 1/17 native. Two
         * populations, one rule:
         *
         *   1. The `upright_refused` stash -- free-standing targets the size/
         *      fragment ceiling refused before any engine read them. THIS is
         *      where the census's "never-read" boxes actually die (the first
         *      render proved it: a pass triggered on empty reads swept nothing
         *      over 81 slices). A recovered read becomes an ordinary
         *      `OcrResult`; a declined or failed sweep restores the target to
         *      `unread` byte-for-byte as if the stash never existed -- the
         *      refusal's veto (it is load-bearing for the eraser) is
         *      preserved exactly for everything the sweep does not rescue.
         *   2. Free-standing regions in `pending` whose read came back EMPTY
         *      (the paddle-era "OCR returned nothing" class).
         *
         * It runs BEFORE the spotting gate, the veto writer and the lettering
         * rules on purpose: a recovered read is an ordinary read, and every
         * downstream gate keeps its jurisdiction over it. The measured spot
         * coverage on the wired pages is 16/17 text boxes >= the floor (the one
         * below it carries a 5-glyph read the gate never checks) and 0/8 artwork
         * boxes, so the gate that refuses junk cannot refuse these recoveries.
         *
         * Fails OPEN per region: a sidecar hiccup mid-sweep leaves the region
         * exactly as refused as it is today. */
        {
            let engine_if_capable = if self.upright_pass {
                match self.primary.vlm() {
                    Some(engine) => {
                        let capable = engine
                            .lock()
                            .map(|e| e.rotate_capable)
                            .map_err(|_| anyhow!("OCR model lock is poisoned"))?;
                        if !capable && !upright_refused.is_empty() {
                            tracing::warn!(
                                "upright pass is on but the OCR endpoint does not \
                                 advertise rotate_ccw; pass disabled -- a sidecar \
                                 ignoring the option would fake a sweep of sixteen \
                                 identical reads"
                            );
                        }
                        capable.then_some(engine)
                    }
                    None => None,
                }
            } else {
                None
            };
            match engine_if_capable {
                Some(engine) => {
                    for target in upright_refused.drain(..) {
                        let crop_image = target.image.clone();
                        let sweep_engine = engine.clone();
                        let swept = tokio::task::spawn_blocking(move || {
                            upright_sweep(&sweep_engine, &crop_image)
                        })
                        .await
                        .context("upright sweep panicked")?;
                        let recovered = match swept {
                            Ok(rows) => match upright_select(&rows) {
                                Some(pick) => {
                                    let row = &rows[pick];
                                    // Non-optional: a region carries no
                                    // provenance on the wire, so without this
                                    // line the pass is unfalsifiable.
                                    tracing::info!(
                                        region = %target.region,
                                        angle = row.angle,
                                        read = %row.text,
                                        mean_logprob = row.mlp.unwrap_or(f64::NAN),
                                        reads = rows.len(),
                                        "upright pass recovered a read from a refused box"
                                    );
                                    // The row's `mlp` is the same quantity every
                                    // scored engine reports, one `exp` away --
                                    // carried so the recovery's own confidence
                                    // reaches the wire like any other read's.
                                    Some((row.text.clone(), row.mlp.map(|value| value.exp() as f32)))
                                }
                                None => {
                                    tracing::info!(
                                        region = %target.region,
                                        reads = rows.len(),
                                        "upright pass swept the refused box and declined every candidate"
                                    );
                                    None
                                }
                            },
                            Err(error) => {
                                tracing::warn!(
                                    region = %target.region,
                                    error = %error,
                                    "upright pass failed open; the refused box stays refused"
                                );
                                None
                            }
                        };
                        match recovered {
                            Some((text, recovered_confidence)) => {
                                // The recovery LETTERS, so its source ink must
                                // ERASE -- otherwise the recovered title letters
                                // beside a fully preserved column -- but ONLY
                                // when the box is NARROW: `rescue_narrow`, and
                                // the render is the authority for both arms.
                                if let Some(bounds) = geometry_extents(&target.geometry) {
                                    if rescue_narrow(bounds) {
                                        upright_rescued.push(bounds);
                                    }
                                }
                                pending.push((
                                    OcrResult {
                                        content: target.content,
                                        region: target.region,
                                        geometry: target.geometry,
                                        previous: target.previous,
                                        text,
                                        confidence: recovered_confidence,
                                        minted: false,
                                    },
                                    generation.clone(),
                                ));
                            }
                            None => {
                                if let Some((min_x, min_y, max_x, max_y)) =
                                    geometry_extents(&target.geometry)
                                {
                                    upright_declined.push((
                                        min_x,
                                        min_y,
                                        max_x - min_x,
                                        max_y - min_y,
                                    ));
                                }
                                if self.implausible.veto_mask {
                                    if let Some(bounds) = geometry_extents(&target.geometry) {
                                        unread.push(bounds);
                                    }
                                }
                            }
                        }
                    }
                    for (result, _) in pending.iter_mut() {
                        if !result.text.trim().is_empty() {
                            continue;
                        }
                        let role = input.scene.component::<TextRole>(result.content)?;
                        if !free_standing(role.as_ref().map(|value| value.role.as_str())) {
                            continue;
                        }
                        let Ok(crop_image) = crop(&source, &result.geometry) else {
                            continue;
                        };
                        let sweep_engine = engine.clone();
                        let swept = tokio::task::spawn_blocking(move || {
                            upright_sweep(&sweep_engine, &crop_image)
                        })
                        .await
                        .context("upright sweep panicked")?;
                        match swept {
                            Ok(rows) => {
                                if let Some(pick) = upright_select(&rows) {
                                    let row = &rows[pick];
                                    tracing::info!(
                                        region = %result.region,
                                        angle = row.angle,
                                        read = %row.text,
                                        mean_logprob = row.mlp.unwrap_or(f64::NAN),
                                        reads = rows.len(),
                                        "upright pass recovered a read from a turned crop"
                                    );
                                    result.text = row.text.clone();
                                    // The replacement read's own score replaces
                                    // the empty read's -- the same pairing the
                                    // stash arm above preserves.
                                    result.confidence = row.mlp.map(|value| value.exp() as f32);
                                } else {
                                    tracing::info!(
                                        region = %result.region,
                                        reads = rows.len(),
                                        "upright pass swept the crop and declined every candidate"
                                    );
                                }
                            }
                            Err(error) => {
                                tracing::warn!(
                                    region = %result.region,
                                    error = %error,
                                    "upright pass failed open; the region keeps its empty read"
                                );
                            }
                        }
                    }
                }
                None => {
                    // No capable engine: every stashed target is restored to the
                    // exact path it would have taken without the stash. Losing a
                    // veto here would erase artwork the refusal was protecting.
                    for target in upright_refused.drain(..) {
                        if self.implausible.veto_mask {
                            if let Some(bounds) = geometry_extents(&target.geometry) {
                                unread.push(bounds);
                            }
                        }
                    }
                }
            }
        }

        /* THE RESERVE RE-READ. A dialogue balloon whose declared-ko read came
         * back wrong-script is REFUSED by the misread lever downstream
         * (`withdraw_from_mask` + `labels.rs`), so the reader keeps the drawn
         * ink -- correct, and still short of the bar: the balloon should letter
         * English. The READ is the recoverable half: on a 19-region refusal
         * population the reserve engine on padded crops recovered 4 of the 6
         * genuine dialogue targets EXACTLY, where same-engine re-reads (0 and
         * 180 degrees) recovered none. See `REREAD_PAD_PX`'s doc for the
         * numbers.
         *
         * It runs AFTER the upright pass and BEFORE the veto writer and the
         * spotting gate, for the same reason the upright pass does: an admitted
         * read is an ordinary read, and every downstream gate keeps its
         * jurisdiction over it -- the erase, the translation and the lettering
         * all follow from the replaced text with no second flag. A declined
         * re-read changes nothing: the refusal+withdrawal ships the drawn ink
         * exactly as today. Fails OPEN per region and per pass. */
        if self.reread_refused_dialogue {
            let mut reread_targets: Vec<OcrTarget> = Vec::new();
            for (result, _) in pending.iter() {
                // Fallible and outside a closure, like every role read in this
                // file: a scene error must not quietly disable the pass.
                let role = input.scene.component::<TextRole>(result.content)?;
                let region_component = input.scene.component::<Region>(result.region)?;
                let is_onomatopoeia = region_component
                    .as_ref()
                    .and_then(|value| value.label.as_deref())
                    == Some(ONOMATOPOEIA_LABEL);
                if !reread_candidate(
                    &result.text,
                    self.source_language,
                    free_standing(role.as_ref().map(|value| value.role.as_str())),
                    is_onomatopoeia,
                    self.misread_levers,
                ) {
                    continue;
                }
                let padded = match crop_grown(&source, &result.geometry, REREAD_PAD_PX) {
                    Ok(image) => image,
                    Err(error) => {
                        tracing::warn!(
                            region = %result.region,
                            error = %error,
                            "reserve re-read skipped: the padded crop failed; the refusal stands"
                        );
                        continue;
                    }
                };
                let previous = input.scene.component::<SourceText>(result.content)?;
                reread_targets.push(OcrTarget {
                    content: result.content,
                    region: result.region,
                    geometry: result.geometry.clone(),
                    previous,
                    reread_rotated: false,
                    reread_flipped: false,
                    flip_crop: None,
                    image: padded,
                    grown: None,
                });
            }
            if !reread_targets.is_empty() {
                tracing::info!(
                    regions = reread_targets.len(),
                    pad_px = REREAD_PAD_PX,
                    "re-reading lever-refused dialogue regions through the reserve engine"
                );
                let reread_generation = super::generation(PRODUCER, FALLBACK_MODEL)?;
                let reread = match self.fallback.ensure().await {
                    Ok(model) => {
                        infer_text(
                            model,
                            reread_targets,
                            |model, image| {
                                let result = model.inference(image, PaddleOCRVLTask::Ocr)?;
                                Ok((result.text, result.confidence))
                            },
                            // Same as the routed arm: scored, not rotate-capable.
                            None::<RotateFn<PaddleOCRVL>>,
                            self.orientation_margin,
                        )
                        .await
                    }
                    Err(error) => Err(error),
                };
                match reread {
                    Ok(results) => {
                        for recovered in results {
                            let Some(entry) = pending
                                .iter_mut()
                                .find(|(existing, _)| existing.content == recovered.content)
                            else {
                                continue;
                            };
                            // Non-optional: a region carries no
                            // provenance on the wire, so without these lines
                            // the pass is unfalsifiable from a run log.
                            if admit_hangul_reread(&recovered.text) {
                                tracing::info!(
                                    region = %recovered.region,
                                    first = %entry.0.text,
                                    read = %recovered.text,
                                    "reserve re-read recovered a hangul read; the region letters normally"
                                );
                                entry.0.text = recovered.text;
                                entry.0.confidence = recovered.confidence;
                                entry.1 = reread_generation.clone();
                            } else {
                                tracing::info!(
                                    region = %recovered.region,
                                    read = %recovered.text,
                                    "reserve re-read declined by script membership; the refusal stands"
                                );
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            error = %error,
                            "reserve re-read failed open; every refusal stands"
                        );
                    }
                }
            }
        }

        let mut edit = input.scene.edit_as(generation.clone());
        edit.observe_assets(page)?;
        for (result, _) in &pending {
            edit.observe::<Region>(result.region)?;
            edit.observe::<Geometry>(result.region)?;
            edit.observe::<SourceText>(result.content)?;
        }

        /* THE ARTWORK-SPOTTING GATE. A VLM reader answers
         * confident Han singles on petals and speed lines, which no script rule can
         * touch on a Chinese page (the paddle-era fabrications were kana, and the
         * shipped wrong-script arm caught those). The tell that DOES separate is the
         * same engine's own official spotting task, asked once per page: over the
         * whole chapter, all twelve fabrications sit at literally ZERO spotting
         * coverage while 24 of 26 real short reads sit at >= 0.51.
         *
         * This is NOT PaddleOCR-VL's spotting, which was rejected for its 83%
         * empty base rate over real drawn glyphs; HunyuanOCR's spotting boxed
         * every real drawn effect it was scored on. Nor is it a confidence,
         * geometry-perturbation, second-engine or string-composition rule.
         *
         * Costs one extra sidecar round-trip ONLY on pages that carry a short
         * (<= 4 glyph) non-empty read -- about one page in eight on the measured
         * chapter. Fails OPEN on any error: a sidecar hiccup must never erase a
         * real effect. The read stays in `SourceText` either way -- the record is
         * kept, the LETTERING and the ERASE are what the verdict withdraws. */
        let mut spot_refused: std::collections::HashSet<EntityId> =
            std::collections::HashSet::new();
        /* FREE-STANDING ONLY, the same scope the wrong-script arm ships with and
         * for the same measured reason: a balloon is a text container and its
         * contents are set dialogue, while every fabrication this gate exists for
         * is a detector proposal on open artwork -- all twelve carry
         * `onomatopoeia`/`free-text`, zero carry `dialogue`. Scoping here is what
         * returns the page-time cost: short in-balloon interjections are the
         * COMMON short read, and without this they bought a spotting call per
         * page whose answer was always "covered". A fallible loop, not a closure,
         * for the reason the veto writer records: a `?` cannot escape `any()`,
         * and swallowing a scene-read error would quietly disable the gate. */
        let mut has_spot_candidate = false;
        if spot_gate_enabled() {
            for (result, _) in &pending {
                let role = input.scene.component::<TextRole>(result.content)?;
                if spot_eligible(
                    &result.text,
                    self.source_language,
                    role.as_ref().map(|value| value.role.as_str()),
                ) {
                    has_spot_candidate = true;
                    break;
                }
            }
        }
        // ONE spot call per page, shared between the artwork gate and the spot
        // rescue -- the +18% page-time the gate's own scoping bought back must
        // not be re-spent by a second caller asking the same question.
        let mut spotted_boxes: Option<Vec<[f64; 4]>> = None;
        if has_spot_candidate {
            if let Some(engine) = self.primary.vlm() {
                let page_image = source.clone();
                let spotted = tokio::task::spawn_blocking(move || {
                    let engine = engine
                        .lock()
                        .map_err(|_| anyhow!("OCR model lock is poisoned"))?;
                    engine.spot_page(&page_image)
                })
                .await
                .context("spotting task panicked")
                .and_then(|inner| inner);
                match spotted {
                    Ok(boxes) => {
                        let (width, height) = (source.width(), source.height());
                        spotted_boxes = Some(boxes.clone());
                        for (result, _) in &pending {
                            let role =
                                input.scene.component::<TextRole>(result.content)?;
                            if spot_refuses(
                                &result.text,
                                self.source_language,
                                role.as_ref().map(|value| value.role.as_str()),
                                &result.geometry,
                                &boxes,
                                width,
                                height,
                            ) {
                                let coverage = geometry_extents(&result.geometry)
                                    .map_or(0.0, |extents| {
                                        spot_coverage(extents, &boxes, width, height)
                                    });
                                // Non-optional: a region carries no
                                // provenance on the wire, so without this line the
                                // gate is unfalsifiable.
                                tracing::info!(
                                    region = %result.region,
                                    read = %result.text,
                                    coverage,
                                    spotted = boxes.len(),
                                    "spotting finds no text here; the read is not lettered"
                                );
                                spot_refused.insert(result.region);
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            page = %page,
                            error = %error,
                            "artwork-spotting gate failed open; nothing refused"
                        );
                    }
                }
            }
        }
        /* THE ILLEGIBLE-READ FLOOR -- the leak class: a sub-floor
         * free-standing read lettered over artwork ("SELECT" on the vortex at
         * 0.167) with the drawn glyph it misread erased from under it. Its own
         * loop, deliberately NOT inside the spotting arm above: it needs no
         * spot call, and it must reach reads longer than the 4-glyph spot
         * eligibility. It writes into the same `spot_refused` set so the
         * LETTERING and the ERASE move together through the veto writer -- the
         * drawn effect stands, un-erased, exactly as a spot refusal leaves it.
         * A refusal added downstream in the server would be right and too
         * late: the erase mask is already committed, and the page renders a
         * BLANK where the artwork was. */
        for (result, _) in &pending {
            if spot_refused.contains(&result.region) {
                continue;
            }
            let role = input.scene.component::<TextRole>(result.content)?;
            if illegible_read(
                result.confidence,
                role.as_ref().map(|value| value.role.as_str()),
            ) {
                // Non-optional: without this line the floor is
                // unfalsifiable from the wire.
                tracing::info!(
                    region = %result.region,
                    read = %result.text,
                    confidence = result.confidence.unwrap_or(f32::NAN),
                    "the read is too illegible to letter; the drawn effect stands"
                );
                spot_refused.insert(result.region);
            }
        }
        /* THE SPOT RESCUE -- the display runs the detector never boxed. On a
         * test chapter one page's banner has NO covering box (93.6% of its ink
         * outside every proposal) and another page's two display runs have
         * none either, so no reading fix can reach them: the failure is
         * upstream of OCR. HunyuanOCR's own page-level spotting DOES box them,
         * with excellent geometry -- but spotting used as the DETECTOR floods
         * on watermarks (61.65%), which is why every box here passes the
         * geometric admission predicate first and a 16-read sweep second.
         * A shipped rescue MINTS a
         * new region/content/layer trio: every component is new, so scene
         * authorship passes, and the declined stash target keeps its refusal
         * and its veto untouched. Reader-visible, so OFF by default; the render
         * is the authority, and the erase is a separate flag adjudicated on its
         * own render. */
        /* The JOINED trigger -- the display run the slice boundary cut. A
         * composite page exists BECAUSE something crosses the cut, and the
         * sparse trigger structurally misses it: the composite carries its
         * neighbours' ordinary regions (one measured composite proposes 3, one
         * over the threshold) while the cut column is undetected on the slice
         * AND on the composite. Probed at debug level: the
         * spotting DOES box the whole cut column, and on the composite nothing
         * overlaps it, so the ordinary admission predicate admits it -- the
         * only missing piece was this trigger. ~26 composites and so ~26 extra
         * spot calls per 179-slice chapter, behind its own flag until a render
         * earns the default. */
        if self.spot_rescue
            && (page_region_count <= SPOT_RESCUE_SPARSE_REGIONS
                || !upright_declined.is_empty()
                || (self.spot_rescue_joined && joined_page))
        {
            if let Some(engine) = self.primary.vlm() {
                let capable = engine
                    .lock()
                    .map_err(|_| anyhow!("OCR model lock is poisoned"))?
                    .rotate_capable;
                if capable {
                    if spotted_boxes.is_none() {
                        let page_image = source.clone();
                        let spot_engine = engine.clone();
                        let spotted = tokio::task::spawn_blocking(move || {
                            let engine = spot_engine
                                .lock()
                                .map_err(|_| anyhow!("OCR model lock is poisoned"))?;
                            engine.spot_page(&page_image)
                        })
                        .await
                        .context("spot-rescue spotting task panicked")
                        .and_then(|inner| inner);
                        match spotted {
                            Ok(boxes) => spotted_boxes = Some(boxes),
                            Err(error) => {
                                tracing::warn!(
                                    page = %page,
                                    error = %error,
                                    "spot rescue failed open; nothing minted"
                                );
                            }
                        }
                    }
                    if let Some(boxes) = spotted_boxes.as_ref() {
                        let (page_width, page_height) = (source.width(), source.height());
                        let page_rgb = source.to_rgb8();
                        // The overlap fence: everything already on the page, in
                        // (x, y, w, h) page pixels -- read results, never-read
                        // vetoed boxes, and the declined stashes themselves.
                        let mut existing: Vec<(f64, f64, f64, f64)> = Vec::new();
                        for (result, _) in &pending {
                            if let Some((min_x, min_y, max_x, max_y)) =
                                geometry_extents(&result.geometry)
                            {
                                existing.push((min_x, min_y, max_x - min_x, max_y - min_y));
                            }
                        }
                        existing.extend(unread.iter().map(
                            |&(min_x, min_y, max_x, max_y)| {
                                (min_x, min_y, max_x - min_x, max_y - min_y)
                            },
                        ));
                        existing.extend(upright_declined.iter().copied());
                        // The ARTWORK fence for the whole-mark growth: settled
                        // panels and bubbles. Growth-added ink must lie on
                        // paper -- artwork-side ink is claimable only through
                        // the rescue's own validated box (the tail glyphs
                        // live ON the panel, but the spotting model proposed
                        // that box itself). This is what keeps the growth off
                        // a character's face: eyes and brows are dark,
                        // glyph-scale and enclosed by bright skin --
                        // topological ink to the letter, and inside the
                        // settled panel to the fence.
                        let mut artwork: Vec<(f64, f64, f64, f64)> = Vec::new();
                        for entity in input.scene.descendants(page)? {
                            let id = entity.id();
                            let Some(region) = input.scene.component::<Region>(id)? else {
                                continue;
                            };
                            if region.kind != koharu_scene::BubbleRegion::kind()
                                && region.kind != koharu_scene::PanelRegion::kind()
                            {
                                continue;
                            }
                            let Some(geometry) = input.scene.component::<Geometry>(id)? else {
                                continue;
                            };
                            if let Some((min_x, min_y, max_x, max_y)) =
                                geometry_extents(&geometry)
                            {
                                artwork.push((min_x, min_y, max_x - min_x, max_y - min_y));
                            }
                        }
                        let mut minted: Vec<(OcrResult, Generation)> = Vec::new();
                        for raw in boxes {
                            let Some(bounds) =
                                denormalize_spot_box(*raw, page_width, page_height)
                            else {
                                continue;
                            };
                            let Some(ink_frac) = spot_ink_fraction(&page_rgb, bounds) else {
                                continue;
                            };
                            if !spot_rescue_candidate(
                                bounds,
                                (page_width, page_height),
                                &existing,
                                ink_frac,
                            ) {
                                tracing::debug!(
                                    page = %page,
                                    spot_box = ?raw,
                                    ink_frac,
                                    "spot box dropped by the admission predicate"
                                );
                                continue;
                            }
                            /* A composite mint must SPAN one of the cuts the
                             * band was composed for. Two adjacent bands see
                             * the same cut column -- one slice carries the
                             * tail of its composite's `三重防御` -- and without
                             * this rule each band mints its own fragment and
                             * the paint-backs stomp each other; rendered,
                             * "TRIPLE DEFENSE!" came back half
                             * overwritten by the neighbour band's lettering.
                             * A box that crosses no cut is a
                             * single slice's own business, and the base page's
                             * rescue owns that population. Empty boundaries --
                             * a caller that did not say where the cuts are --
                             * keeps the old behaviour. */
                            if joined_page
                                && !input.joined_boundaries().is_empty()
                                && !mint_spans_a_cut(bounds, input.joined_boundaries())
                            {
                                tracing::debug!(
                                    page = %page,
                                    spot_box = ?raw,
                                    "composite spot box spans no cut; left to the base pages"
                                );
                                continue;
                            }
                            let geometry =
                                Geometry::rectangle(bounds.0, bounds.1, bounds.2, bounds.3);
                            let Ok(crop_image) = crop(&source, &geometry) else {
                                continue;
                            };
                            let sweep_engine = engine.clone();
                            let swept = tokio::task::spawn_blocking(move || {
                                upright_sweep(&sweep_engine, &crop_image)
                            })
                            .await
                            .context("spot-rescue sweep panicked")?;
                            let rows = match swept {
                                Ok(rows) => rows,
                                Err(error) => {
                                    tracing::warn!(
                                        page = %page,
                                        error = %error,
                                        "spot-rescue sweep failed open; the box stays unread"
                                    );
                                    continue;
                                }
                            };
                            let Some(pick) = upright_select(&rows) else {
                                tracing::info!(
                                    page = %page,
                                    spot_box = ?raw,
                                    reads = rows.len(),
                                    "spot rescue swept the box and declined every candidate"
                                );
                                continue;
                            };
                            let row = &rows[pick];
                            let text = row.text.clone();
                            // The furniture / illegibility / wrong-script belt,
                            // INSIDE the pipeline where it still keeps the box
                            // off the page entirely -- the server's
                            // `Refusal::Watermark` runs after inpainting and
                            // spares only the lettering, never the art.
                            if withdraw_from_mask(
                                &text,
                                self.source_language,
                                true,
                                self.scope_watermark_refusals,
                                self.misread_levers,
                            ) {
                                tracing::info!(
                                    page = %page,
                                    read = %text,
                                    "spot rescue withdrew its own read; nothing minted"
                                );
                                continue;
                            }
                            let confidence = row.mlp.map(|value| value.exp() as f32);
                            // A WIDE scream-read mint is the
                            // mark-replacement device, and the device replaces
                            // the WHOLE drawn mark -- the licensed page's own
                            // treatment. Both flags, then the composed
                            // predicate, then the growth and its re-read; an
                            // abstention or failure at any step ships the
                            // behaviour before it, down to the plain mint.
                            // Gated on `spot_rescue_erase` because the device
                            // is a modification of the mint's ERASE: styling
                            // without the erase would letter over the intact
                            // drawn mark.
                            // The fallback windows below must be the ORIGINAL
                            // box even after an adoption path mutates the
                            // shadows -- captured once, structurally, rather
                            // than relying on which arm assigns what.
                            let original_bounds = bounds;
                            let mut bounds = bounds;
                            let mut geometry = geometry;
                            let mut text = text;
                            let confidence = confidence;
                            let mut scream_grown = false;
                            let scream: Option<ScreamMarkInk> = if self.replace_scream_marks
                                && self.spot_rescue_erase
                                && scream_mark(
                                    (
                                        bounds.0,
                                        bounds.1,
                                        bounds.0 + bounds.2,
                                        bounds.1 + bounds.3,
                                    ),
                                    &text,
                                ) {
                                // The growth is pure pixel work over a window
                                // that can reach page scale -- off the
                                // executor, like every other heavy step here.
                                let grow_image = page_rgb.clone();
                                let grow_existing = existing.clone();
                                let grow_artwork = artwork.clone();
                                let grow_bounds = bounds;
                                let grown = tokio::task::spawn_blocking(move || {
                                    grow_scream_mark(
                                        &grow_image,
                                        grow_bounds,
                                        &grow_existing,
                                        &grow_artwork,
                                    )
                                })
                                .await;
                                match grown {
                                    Err(error) => {
                                        tracing::warn!(
                                            page = %page,
                                            error = %error,
                                            "scream growth analysis panicked; the mint ships unstyled"
                                        );
                                        None
                                    }
                                    Ok(None) => None,
                                    Ok(Some((false, _, ink))) => Some(ink),
                                    Ok(Some((true, grown_bounds, grown_ink))) => {
                                        // A grown mark owes a re-read -- of
                                        // its dominant NEW BAND, not the whole
                                        // crop: the whole crop mixes two glyph
                                        // scales and per-glyph orientations
                                        // and read as belt-withdrawn Han
                                        // garbage on the test exhibit. The band
                                        // is the tight single-run crop the
                                        // sidecar reads; the composition with
                                        // the validated read is gated by
                                        // `scream_compose_band`, and any
                                        // failure falls open to the tail-only
                                        // device on the original box.
                                        let mut adopted = None;
                                        match scream_head_band(
                                            grown_bounds,
                                            original_bounds,
                                            24.0,
                                        ) {
                                            None => tracing::info!(
                                                page = %page,
                                                grown_box = ?grown_bounds,
                                                "scream growth added only slivers; the tail-only device stands"
                                            ),
                                            Some((band, side)) => {
                                        let band_geometry = Geometry::rectangle(
                                            band.0, band.1, band.2, band.3,
                                        );
                                        match crop(&source, &band_geometry) {
                                            Err(error) => tracing::warn!(
                                                page = %page,
                                                error = %error,
                                                "scream growth could not crop its band; the tail-only device stands"
                                            ),
                                            Ok(band_crop) => {
                                            let sweep_engine = engine.clone();
                                            let swept = tokio::task::spawn_blocking(move || {
                                                upright_sweep(&sweep_engine, &band_crop)
                                            })
                                            .await;
                                            match swept {
                                                Ok(Ok(band_rows)) => {
                                                    if let Some(band_pick) =
                                                        upright_select(&band_rows)
                                                    {
                                                        let band_row =
                                                            &band_rows[band_pick];
                                                        let withdrawn = withdraw_from_mask(
                                                            &band_row.text,
                                                            self.source_language,
                                                            true,
                                                            self.scope_watermark_refusals,
                                                            self.misread_levers,
                                                        );
                                                        if let Some(composed) =
                                                            scream_compose_band(
                                                                &band_row.text,
                                                                &text,
                                                                side,
                                                                withdrawn,
                                                            )
                                                        {
                                                            // The band's own sweep
                                                            // numbers, so the log
                                                            // reconciles; the wire
                                                            // confidence stays the
                                                            // VALIDATED read's --
                                                            // the composition is
                                                            // anchored on it.
                                                            tracing::info!(
                                                                page = %page,
                                                                grown_box = ?grown_bounds,
                                                                band = ?band,
                                                                band_read = %band_row.text,
                                                                original_read = %text,
                                                                composed = %composed,
                                                                angle = band_row.angle,
                                                                mean_logprob = band_row
                                                                    .mlp
                                                                    .unwrap_or(f64::NAN),
                                                                reads = band_rows.len(),
                                                                "scream mark grew to its whole drawn extent"
                                                            );
                                                            bounds = grown_bounds;
                                                            geometry = Geometry::rectangle(
                                                                grown_bounds.0,
                                                                grown_bounds.1,
                                                                grown_bounds.2,
                                                                grown_bounds.3,
                                                            );
                                                            text = composed;
                                                            scream_grown = true;
                                                            adopted = Some(grown_ink);
                                                        } else {
                                                            tracing::info!(
                                                                page = %page,
                                                                grown_box = ?grown_bounds,
                                                                band = ?band,
                                                                band_read = %band_row.text,
                                                                withdrawn,
                                                                "scream growth declined its band read; the tail-only device stands"
                                                            );
                                                        }
                                                    } else {
                                                        tracing::info!(
                                                            page = %page,
                                                            grown_box = ?grown_bounds,
                                                            band = ?band,
                                                            "scream growth swept its band and declined every candidate; the tail-only device stands"
                                                        );
                                                    }
                                                }
                                                Ok(Err(error)) => tracing::warn!(
                                                    page = %page,
                                                    error = %error,
                                                    "scream growth band sweep failed open; the tail-only device stands"
                                                ),
                                                Err(error) => tracing::warn!(
                                                    page = %page,
                                                    error = %error,
                                                    "scream growth band sweep panicked; the tail-only device stands"
                                                ),
                                            }
                                            }
                                        }
                                            }
                                        }
                                        // Declined or failed: the shipped
                                        // tail-only device, ORIGINAL box, with
                                        // the same existing-region fence every
                                        // other arm honours.
                                        adopted.or_else(|| {
                                            scream_ink_analysis(
                                                &page_rgb,
                                                original_bounds,
                                                &existing,
                                                &artwork,
                                                original_bounds,
                                            )
                                            .map(|growth| growth.ink)
                                        })
                                    }
                                }
                            } else {
                                None
                            };
                            // Mint the trio. Every component is NEW, so scene
                            // authorship passes; re-aiming an existing region is
                            // a hard cross-producer error AND inert for the
                            // lettering, which follows the LAYER's frame.
                            let region_entity = edit.add_entity(page, At::End)?;
                            edit.set(region_entity, &geometry)?;
                            edit.set(
                                region_entity,
                                &Region {
                                    origin: Origin::Generated(generation.clone()),
                                    kind: RegionKind::new(TextRegion::KIND)?,
                                    label: Some("text".to_owned()),
                                },
                            )?;
                            // The validator demands a real confidence, so the
                            // decode's own is reported rather than a sentinel
                            // somebody later reads as a detector score.
                            let label_confidence = wire_confidence(confidence).unwrap_or(0.5);
                            edit.set(
                                region_entity,
                                &DetectionAnalysis {
                                    origin: Origin::Generated(generation.clone()),
                                    labels: vec![
                                        DetectionLabel {
                                            kind: RegionKind::new(TextRegion::KIND)?,
                                            confidence: label_confidence,
                                        },
                                        DetectionLabel {
                                            kind: RegionKind::new(SPOT_RESCUED_REGION_KIND)?,
                                            confidence: label_confidence,
                                        },
                                    ],
                                },
                            )?;
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
                            edit.set(
                                content,
                                &TextRole {
                                    origin: Origin::Generated(generation.clone()),
                                    role: "dev.koharu.text.free-text".to_owned(),
                                },
                            )?;
                            edit.relate::<RecognizedFrom>(content, region_entity)?;
                            edit.relate::<FitsTo>(layer, region_entity)?;
                            let inferred = spot_typography(&page_rgb, bounds);
                            // A tall minted column letters along its own axis,
                            // exactly as a detector-found free-text column does.
                            // Without this the horizontal English wraps inside
                            // the column's WIDTH and the solver collapses:
                            // probed on a seam composite, the
                            // minted 三重防御 lettered at 16.7 px in a 137 px
                            // column while the detector's neighbouring column
                            // lettered at 89.5 px, turned. Same predicate, same
                            // function, so it claims exactly the population
                            // detection would have turned.
                            if let Some(mark) = &scream {
                                // The replacement letters as ONE band along the
                                // ink's own principal axis, centred on its
                                // centroid -- the frame the drawn cascade set,
                                // not the hull and not the upright sweep's
                                // read angle (the two differ by 70 degrees on
                                // the test exhibit). Rotation is allowed here
                                // by design for sound effects; the scream_mark
                                // gate is what proves the class.
                                edit.set(
                                    layer,
                                    &rotated_rectangle_geometry(
                                        scream_band_cell(bounds, mark),
                                        mark.angle_degrees,
                                    ),
                                )?;
                            } else if let Some(turned) = free_text_column_geometry(
                                [
                                    bounds.0 as f32,
                                    bounds.1 as f32,
                                    (bounds.0 + bounds.2) as f32,
                                    (bounds.1 + bounds.3) as f32,
                                ],
                                inferred,
                            ) {
                                edit.set(layer, &turned)?;
                            }
                            let stroke = text_stroke(inferred, false);
                            if let Some(mark) = &scream {
                                // Fill runs the drawn mark's own ramp: the
                                // eroded core's bright end at the layout's
                                // top, its dark end at the bottom, tilting
                                // with the frame. Halo from the bright ring
                                // the mark itself wears, so the replacement
                                // stays legible on the panel exactly where
                                // the original was.
                                edit.set(
                                    layer,
                                    &Typography {
                                        origin: Origin::Generated(generation.clone()),
                                        preferred_font: None,
                                        font_weight: Some(HEAVY_INK_WEIGHT),
                                        size: None,
                                        auto_fit: true,
                                        color: Some([
                                            mark.head[0],
                                            mark.head[1],
                                            mark.head[2],
                                            u8::MAX,
                                        ]),
                                        stroke_color: stroke.map(|_| {
                                            [mark.halo[0], mark.halo[1], mark.halo[2], u8::MAX]
                                        }),
                                        stroke_width: stroke.map(|(_, width)| width),
                                        alignment: None,
                                        writing_mode: Some(WritingMode::Horizontal),
                                        extensions: [(
                                            FILL_GRADIENT_TO_EXTENSION.to_owned(),
                                            format!(
                                                "{},{},{},255",
                                                mark.tail[0], mark.tail[1], mark.tail[2]
                                            ),
                                        )]
                                        .into_iter()
                                        .collect(),
                                    },
                                )?;
                            } else {
                                edit.set(
                                    layer,
                                    &Typography {
                                        origin: Origin::Generated(generation.clone()),
                                        preferred_font: None,
                                        font_weight: None,
                                        size: inferred.map(|value| value.font_size),
                                        auto_fit: true,
                                        color: inferred.map(|value| {
                                            [
                                                value.color[0],
                                                value.color[1],
                                                value.color[2],
                                                u8::MAX,
                                            ]
                                        }),
                                        stroke_color: stroke.map(|(color, _)| {
                                            [color[0], color[1], color[2], u8::MAX]
                                        }),
                                        stroke_width: stroke.map(|(_, width)| width),
                                        alignment: None,
                                        writing_mode: inferred.map(|value| value.writing_mode),
                                        extensions: Default::default(),
                                    },
                                )?;
                            }
                            // Non-optional: a region carries no provenance
                            // on the wire, so without this line a rescue is
                            // invisible in `format=json` and unfalsifiable.
                            tracing::info!(
                                page = %page,
                                region = %region_entity,
                                spot_box = ?raw,
                                angle = row.angle,
                                read = %text,
                                mean_logprob = row.mlp.unwrap_or(f64::NAN),
                                reads = rows.len(),
                                erase = self.spot_rescue_erase,
                                "spot rescue minted a region for a display run the detector never boxed"
                            );
                            if let Some(mark) = scream {
                                // The same rule again: the device's decision and
                                // its sampled style must be visible from the
                                // log, or the render is unfalsifiable.
                                tracing::info!(
                                    page = %page,
                                    region = %region_entity,
                                    angle_degrees = mark.angle_degrees,
                                    head = ?mark.head,
                                    tail = ?mark.tail,
                                    halo = ?mark.halo,
                                    ink_px = mark.ink_px,
                                    grown = scream_grown,
                                    "scream mark replaced: one styled scream letters in the drawn mark's place"
                                );
                                scream_rescued.push(mark);
                            } else if self.spot_rescue_erase {
                                upright_rescued.push((
                                    bounds.0,
                                    bounds.1,
                                    bounds.0 + bounds.2,
                                    bounds.1 + bounds.3,
                                ));
                            }
                            existing.push(bounds);
                            minted.push((
                                OcrResult {
                                    content,
                                    region: region_entity,
                                    geometry: geometry.clone(),
                                    previous: None,
                                    text,
                                    confidence,
                                    minted: true,
                                },
                                generation.clone(),
                            ));
                        }
                        pending.extend(minted);
                    }
                }
            }
        }
        // The rescue mask is written BEFORE the veto and independently
        // of every veto flag -- a shipped recovery's erase must not be switchable
        // off by a flag about withdrawal. Empty unless the upright pass recovered
        // something NARROW (`rescue_narrow`), so every other page is
        // byte-identical.
        if !upright_rescued.is_empty() || !scream_rescued.is_empty() {
            write_rescue_mask(&mut edit, page, &source, &upright_rescued, &scream_rescued)?;
        }
        // Two arms, deliberately independent. The illegible arm reads `pending` and
        // is gated by its own flag; the unread arm carries boxes that never reached
        // `pending` at all and must not be switched off by the other flag -- that
        // coupling is the interlock this whole change exists to break. `unread` is
        // empty unless `veto_mask` is on, so the shipping arm is unchanged.
        if self.withdraw_illegible_masks || !unread.is_empty() || !spot_refused.is_empty() {
            write_illegible_veto(
                &mut edit,
                &input.scene,
                page,
                &source,
                if self.withdraw_illegible_masks || !spot_refused.is_empty() {
                    &pending
                } else {
                    &[]
                },
                self.source_language,
                &unread,
                self.scope_watermark_refusals,
                &spot_refused,
                self.misread_levers,
            )?;
        }
        /* SYNTHESISED REGIONS WHOSE READ IS MEANINGLESS ARE NOT LETTERED.
         *
         * The erase and the lettering are decided in two different places, and
         * `withdraw_from_mask` only reaches the first. On the eight ellipsis pages
         * `--read-textless-bubbles` fires on, the mask was correctly withdrawn and
         * the English `......` was then drawn HORIZONTALLY ACROSS the Japanese
         * vertical ellipsis it had just spared, leaving a cross of dots.
         *
         * **The obvious fix -- exempting punctuation from `hide_implausible`'s
         * free-standing gate -- is refuted by measurement**: 1,901 in-bubble
         * punctuation-only regions across the run corpus would stop being lettered,
         * and they include the `3`, `2`, `!!` and `......` balloons that gate's own
         * doc comment names as having become BLANK HOLES when a build refused them.
         *
         * This is narrow instead, and the argument is what makes it safe: refusing
         * to letter a SYNTHESISED region leaves the page exactly as the flag-OFF arm
         * renders it, so no hole can appear that is not already there today. It
         * cannot touch an ordinary balloon at all.
         *
         * The read is still written to `SourceText` above, so the payload reports
         * what OCR actually returned. Losing the LETTERING is the point; losing the
         * RECORD would lose information for no gain. */
        let hidden: Vec<EntityId> = {
            let snapshot = &input.scene;
            let mut hidden = Vec::new();
            for (result, _) in &pending {
                // A spotting-refused read is hidden whatever its region kind:
                // the gate's verdict is "there is no text here", which subsumes
                // the synthesised-region question entirely.
                if !spot_refused.contains(&result.region) {
                    // A mint is neither synthesised nor spot-refused, and its
                    // read passed `withdraw_from_mask` before minting. Stated
                    // here rather than inherited: without this line the same
                    // `continue` happens only because `is_synthesised`
                    // swallows the snapshot's `EntityNotFound` for an
                    // in-edit region -- the right answer by accident, and one
                    // `.ok()` removal away from a whole-page 500.
                    if result.minted {
                        continue;
                    }
                    if !is_synthesised(&input.scene, result.region) {
                        continue;
                    }
                    let role = input.scene.component::<TextRole>(result.content)?;
                    if !withdraw_from_mask(
                        &result.text,
                        self.source_language,
                        free_standing(role.as_ref().map(|value| value.role.as_str())),
                        self.scope_watermark_refusals,
                        self.misread_levers,
                    ) {
                        continue;
                    }
                }
                // content -> layer, the direction the scene does not index.
                if let Ok(layers) = snapshot.text_layers() {
                    for layer in layers {
                        if layer.content().map(|c| c.id()).ok() == Some(result.content) {
                            hidden.push(layer.id());
                        }
                    }
                }
            }
            hidden
        };
        for layer in &hidden {
            edit.set(
                *layer,
                &Visibility {
                    origin: Origin::Generated(generation.clone()),
                    visible: false,
                    opacity: 1.0,
                },
            )?;
        }

        for (result, generation) in pending {
            let language = stamped_language(
                result.previous.and_then(|value| value.language),
                self.source_language,
            );
            edit.set(
                result.content,
                &SourceText {
                    text: Authored::generated(result.text, generation.clone()),
                    language,
                },
            )?;
            let (min_x, min_y, max_x, max_y) = geometry_extents(&result.geometry)
                .ok_or_else(|| anyhow!("text region {} has empty geometry", result.region))?;
            edit.set(
                result.region,
                &OcrAnalysis {
                    origin: Origin::Generated(generation.clone()),
                    direction: if max_y - min_y >= (max_x - min_x) * 1.15 {
                        TextDirection::Vertical
                    } else {
                        TextDirection::Horizontal
                    },
                    confidence: wire_confidence(result.confidence),
                    line_boundaries: Vec::new(),
                },
            )?;
        }
        finish(edit)
    }
}

struct OcrTarget {
    content: EntityId,
    region: EntityId,
    geometry: Geometry,
    previous: Option<SourceText>,
    image: DynamicImage,
    /// Whether this target is ELIGIBLE for the turned second read -- a tall
    /// free-standing column, on a run with the flag on. Eligibility is decided on
    /// the box; whether the turn actually happens is decided on the first read's
    /// text by `wants_rotated_reread`.
    reread_rotated: bool,
    /// Whether this target is ELIGIBLE for the 180-degree second read -- a
    /// SYNTHESISED bubble region, on a run with the flag on; `flip_candidate`
    /// decides at build time, where the scene is still in reach. Whether the
    /// flip actually happens is decided on the first read: only a read that
    /// carries a score can be beaten by one.
    reread_flipped: bool,
    /// The BALLOON-ISOLATED twin of `image`, cut at build time for the same
    /// reason as `grown`: the page does not cross into `infer_text`. `Some`
    /// only for flip candidates whose crop has a credible white body
    /// (`isolate_balloon`); the flipped second read uses it, because the grid
    /// measured the surrounding ART -- not the tilt -- as what breaks the read
    /// (raw crop 0/13 angles exact, isolated crop exact at 175/180/185). `None`
    /// falls back to the raw crop, which is the behaviour that shipped when the
    /// flip landed. The UPRIGHT read never uses it: when the flip loses, the
    /// page must stay byte-identical to the flag-off arm.
    flip_crop: Option<DynamicImage>,
    /// The SAME box, cut again with a few pixels of extra background on every
    /// side, for the perturbation-stability measurement. `Some` only when the flag
    /// is on, the region is an onomatopoeia, **and the grown crop actually came
    /// back a different size** -- a box the clamp refused to grow is dropped here
    /// rather than carried, because an unperturbed pair scores as "stable" and
    /// would be a false negative. Reported, never acted on.
    grown: Option<DynamicImage>,
}

struct OcrResult {
    content: EntityId,
    region: EntityId,
    geometry: Geometry,
    previous: Option<SourceText>,
    text: String,
    /// The decode's own length-normalised sequence probability,
    /// `exp(mean log p)`, when the engine reports one: PaddleOCR-VL computes it
    /// and the Hunyuan sidecar returns it; `manga-ocr`, `baberu-ocr` and Ollama
    /// report `None`. `None` means NOT MEASURED, never zero -- the distinction
    /// every caller in this file already preserves. When a second read was
    /// taken, this is the confidence of the read that was KEPT.
    confidence: Option<f32>,
    /// This result's entities were MINTED by this same stage call and exist
    /// only in the in-flight `Edit` -- the frozen `input.scene` answers
    /// `EntityNotFound` for them, never `None`, because `Snapshot::component`
    /// resolves through the entity's page (the fact detection's
    /// `layer_geometry_is_writable` guards). A reader walking `pending`
    /// against the snapshot must branch on this instead of looking up: an
    /// unguarded `?` at the veto writer's role read once failed every page that
    /// minted with a whole-page 500.
    minted: bool,
}

/// Publish the regions OCR could not read, so the inpainter can spare them.
///
/// **The mask is written during DETECTION, before a character has been read**, so
/// by the time anything knows a box held no text the pixels are already promised
/// to the inpainter. This is the earliest point at which the answer exists.
///
/// **It writes a VETO rather than editing `text-mask`, and that is not a
/// stylistic choice.** `koharu-scene` enforces authorship: `text-mask` belongs to
/// `dev.koharu.pipeline.detection`, and an attempt to overwrite it from this
/// stage fails the whole page with `component authorship conflict`. So this
/// stage owns a mask of its own -- `text-mask-veto`, the pixels that must NOT be
/// erased -- and `stages::inpainting::prepare` subtracts it. That also keeps the
/// two stages honest: detection still says what it found, OCR says what it could
/// not read, and neither rewrites the other.
///
/// **The veto is `refused MINUS kept`, and the subtraction is what makes it
/// safe.** Boxes overlap -- the containment case is exactly a small box inside
/// a large one -- so vetoing a refused box wholesale would un-erase a real
/// region's glyphs where the two meet. Anything a kept region also covers is
/// removed from the veto, so the inpainter's behaviour there is unchanged.
///
/// Bounds are grown by the same radius the detection stage dilated by, because
/// the pixels to spare are the dilated ones, not the raw box.
/// Whether this region's pixels must be taken back OUT of the erase mask.
///
/// The composed predicate, named rather than spelled inline at the call site,
/// **because a test of the halves separately does not test this**. Both
/// `illegible_text` and `watermark_text` had passing unit tests while the `||`
/// between them was deleted from `write_illegible_veto`, and all 128 stayed
/// green -- the erase would have gone right back to destroying every watermark
/// plate on the page with nothing to show it. Test the thing the caller calls.
///
/// It is now a THREE-way `||` with an `&&` inside the third arm, so there are two
/// more ways to ship it unwired than there were. Assert on **this** function.
///
/// **Why the third arm is gated on `free_standing`.** It is the same gate
/// `labels::hide_implausible` applies, and for the same measured reason: a
/// balloon is a text CONTAINER. Detection has already erased it to flat white, so
/// sparing a bubble's pixels after the fact cannot put the original back -- it
/// would leave the flat fill and no English, which is the blank-hole defect that
/// gate exists to avoid. Artwork was never a container, so there the erase is the
/// whole of the damage. Keeping the two gates identical is also what stops a
/// region being spared by one and lettered by the other.
fn free_standing(role: Option<&str>) -> bool {
    role.is_some_and(|role| !role.ends_with("dialogue"))
}

/// The kept read's confidence as the scene schema will accept it, or `None`.
///
/// `OcrAnalysis::validate` REJECTS a non-finite or out-of-`0.0..=1.0` value and
/// that error fails the WHOLE PAGE, so an engine transport defect must degrade to
/// "not measured" rather than become a 500. In-range values pass untouched:
/// `exp(mean log p)` is in `(0, 1]` by construction, so anything outside is
/// already a defect of the transport, never a measurement.
fn wire_confidence(confidence: Option<f32>) -> Option<f32> {
    confidence.filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
}

/// The per-region language tag the scene carries and the wire's
/// `regions[].source_language` serialises. DECLARED, never detected: no OCR
/// engine here reports a language -- the engine call signature carries text and
/// an optional score, nothing else -- so the truthful arm is the language the
/// caller declared for the request, and `ja-JP` survives only as the undeclared
/// default. An existing tag wins outright, so a re-run never overwrites what
/// an earlier pass or a project file already stamped.
///
/// **Wire metadata, not the translator's language.** `declared` is the same
/// `Model` field that feeds `withdraw_from_mask`'s script arm; it still never
/// reaches the prompt -- that is `TranslationConfig::source_language`.
fn stamped_language(
    previous: Option<LanguageTag>,
    declared: Option<Language>,
) -> Option<LanguageTag> {
    previous
        .or_else(|| declared.and_then(|language| LanguageTag::new(language.tag()).ok()))
        .or_else(|| LanguageTag::new("ja-JP").ok())
}

/// Was this region invented from a bubble holding no text, rather than returned by
/// the detector?
///
/// Reads the SECOND `DetectionLabel` that `write_region` appends -- see
/// [`super::detection::SYNTHESISED_REGION_KIND`] for why the marker lives there and
/// not on `Region.label`. Any-match rather than index-1, so the answer does not
/// depend on label order.
fn is_synthesised(scene: &Snapshot, region: EntityId) -> bool {
    scene
        .component::<DetectionAnalysis>(region)
        .ok()
        .flatten()
        .is_some_and(|analysis| {
            analysis
                .labels
                .iter()
                .any(|label| label.kind.as_str() == SYNTHESISED_REGION_KIND)
        })
}

/// The misread levers, carried together because they are one decision made in
/// two places: `leave_misread_bubbles` widens WHO the script arm reaches
/// (dialogue-role regions, not only free-standing ones), `korean_script_strict`
/// widens WHAT counts as a mismatch (a declared-ko read with no hangul). The
/// lettering side reads the same two flags in `birelate-server`'s
/// `hide_implausible` -- reader and mask move together or neither does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MisreadLevers {
    pub leave_misread_bubbles: bool,
    pub korean_script_strict: bool,
}

impl MisreadLevers {
    /// Both levers off: the predicate without either lever, byte for byte. Test-only --
    /// production always builds the pair from the two flags, so a non-test
    /// build has no caller and rustc's dead-code lint said so (seen compiling
    /// the lib for `birelate-server`, where the tests that use it are absent).
    #[cfg(test)]
    pub(super) const OFF: Self = Self {
        leave_misread_bubbles: false,
        korean_script_strict: false,
    };
}

fn withdraw_from_mask(
    text: &str,
    source_language: Option<Language>,
    free_standing: bool,
    scoped: bool,
    levers: MisreadLevers,
) -> bool {
    /* The script term is composed INSIDE the free-standing term on purpose: the
     * strict arm must obey the same audience rule as the ratio arm, so with
     * `leave_misread_bubbles` off both arms stay free-standing-only, and with it
     * on both reach dialogue. A strict arm outside the parenthesis would have
     * withdrawn dialogue masks with the lettering half still gated -- the exact
     * one-half failure the levers struct exists to prevent. */
    illegible_text(text)
        || only_site_furniture(text, scoped)
        || ((free_standing || levers.leave_misread_bubbles)
            && (script_mismatch(text, source_language)
                || (levers.korean_script_strict && strict_korean_mismatch(text, source_language))))
}

/// The whole decision the reserve re-read makes about a FIRST read.
///
/// Extracted so the composed predicate is what a test can call (testing the
/// halves leaves the `&&`s between them unexercised). The
/// caller does exactly one thing with a `true`: cuts a `REREAD_PAD_PX` crop and
/// buys one read from the reserve engine.
///
/// The scope is the misread lever's own dialogue arm, narrowed three ways:
/// - **Declared KOREAN only.** The measurement is Korean; the admission test
///   below is hangul membership and would be meaningless elsewhere.
/// - **Dialogue role, not an onomatopoeia.** The censused refusal population
///   carries 3 SFX the official release ships as drawn; re-admitting them
///   REGRESSES (pixel-checked against the official English pages). Free-standing regions never
///   qualify -- their refusals belong to other rules.
/// - **Not punctuation-only or empty** (`illegible_text`): the drawn `?!` IS
///   the correct lettering, and 8 of the 19 censused refusals are this class.
///
/// `leave_misread_bubbles` gates candidacy because it is the same flag that
/// makes the dialogue refusal reachable in `withdraw_from_mask` at all -- a
/// re-read of a region nothing would have refused buys nothing.
fn reread_candidate(
    text: &str,
    source_language: Option<Language>,
    free_standing: bool,
    is_onomatopoeia: bool,
    levers: MisreadLevers,
) -> bool {
    if free_standing || is_onomatopoeia || !levers.leave_misread_bubbles {
        return false;
    }
    if !matches!(source_language, Some(Language::Korean)) {
        return false;
    }
    let text = text.trim();
    if text.is_empty() || illegible_text(text) {
        return false;
    }
    script_mismatch(text, source_language)
        || (levers.korean_script_strict && strict_korean_mismatch(text, source_language))
}

/// Admission for a reserve re-read: SCRIPT MEMBERSHIP, never a confidence
/// comparison -- score-based candidate picking was measured unreliable.
/// Hangul-majority: at least one hangul letter, and hangul is no less than half
/// of the scripted letters. A re-read that comes back wrong-script again
/// (repeated Han garbage, or an all-Latin romanization) stays refused.
fn admit_hangul_reread(text: &str) -> bool {
    let counts = Scripts::of(text);
    counts.hangul > 0 && counts.hangul * 2 >= counts.scripted()
}

/// How much taller than wide a box must be to be a candidate for the turn.
///
/// **Shape alone cannot decide orientation, and this constant does not pretend
/// to.** An upright vertical column of square glyphs and a horizontal line laid
/// on its side are the same shape -- which is why the pre-OCR pixel classifier
/// was declined at AUC 0.491. This is only a cheap
/// upper bound on the population that is asked the real question, which is
/// `wants_rotated_reread` on the first read's TEXT.
const COLUMN_ASPECT: f64 = 3.0;

/// Whether this box is eligible for a turned second read.
///
/// Free-standing for the same measured reason every other gate in this file is:
/// a balloon is a text container, and its contents are ordinary set text. A
/// display line turned on its side is artwork.
fn rotation_candidate(geometry: &Geometry, free_standing: bool) -> bool {
    free_standing
        && geometry_extents(geometry).is_some_and(|(min_x, min_y, max_x, max_y)| {
            let (width, height) = (max_x - min_x, max_y - min_y);
            width > 0.0 && height >= width * COLUMN_ASPECT
        })
}

/// Whether the FIRST read looks like it came off sideways glyphs.
///
/// **Kana is the whole of the test, and it is what keeps ordinary vertical
/// Japanese out.** Upright vertical Japanese is 38.3% of manga-ja free-text and
/// turning it would break pages that work today; essentially all of it carries
/// kana, so a kana-free Han-only read excludes it without asking about shape.
///
/// **This is a FLOOR, not a judgement, and the difference is the risk.** It does
/// not adjudicate between two well-formed Chinese strings -- only
/// `choose_orientation`'s margin can, and only when both reads carry a score
/// (`manga-ocr` and `baberu-ocr` report none, and three other OCR-derived signals
/// were measured and rejected). So the population genuinely at risk is **upright vertical CHINESE**,
/// which is kana-free by construction and which this test cannot separate from
/// the sideways case. That is why the flag ships OFF: the control render, not
/// this predicate, is what decides whether it may go on.
fn wants_rotated_reread(text: &str) -> bool {
    let scripts = Scripts::of(text);
    scripts.kana == 0 && scripts.han > 0
}

/// Which of the two reads to keep.
///
/// Strict dominance, deliberately: the turned read wins unless it produced
/// nothing at all. There is no quality comparison here because none is
/// implementable -- see `wants_rotated_reread`. Naming it as its own function is
/// what lets a test assert the pick without a model, and what stops the rule
/// being buried in an `if` where deleting it would leave every test green.
/// Which of the two reads of one crop to keep.
///
/// **The historical rule is "the turned read wins unless it is empty", and it had
/// no alternative**: nothing could rank two reads, so preferring the turn was the
/// only way to make the re-read worth doing at all.
///
/// `margin` turns that into a ranking. `None` is byte-identical to the old
/// behaviour and is the default. `Some(m)` keeps the turn UNLESS the upright read
/// beats it by at least `m` -- deliberately asymmetric, because the turn exists to
/// rescue sideways columns and a tie must not undo it.
///
/// **Measured over all 179 slices of a test chapter, every pair adjudicated
/// against the drawn page rather than against a counter.** Six upright/turned
/// pairs, five differing:
///
/// (strings are invented stand-ins; scores are as measured)
///
/// ```text
/// page  upright                      turned                      a margin
/// p1    凌霄境界 0.8424 RIGHT     陵霄境界 0.6044 wrong     FIXES
/// p2    风之王   0.7209 RIGHT     空白的K  0.1760 garbage   FIXES
/// p1    感       0.5708 wrong     金       0.0610 wrong     wash
/// p3    玄冰真诀 0.9508 RIGHT     玄水真诀 0.9906 wrong     MISSES
/// p4    玄水真诀 0.8843 wrong     玄冰真诀 0.9921 RIGHT     keeps
/// ```
///
/// Two fixes, one correct keep, one wash, and **one MISS that matters more than the
/// count**: on `p3` the model is MORE CONFIDENT IN THE WRONG READ, so no margin
/// recovers that pair. The turn already wins it today, which makes it a missed fix
/// rather than a regression. **Confidence is not a reliable ranker -- it is a ranker
/// that is right more often than the unconditional rule it replaces.**
///
/// `p3` and `p4` are the SAME skill name on two pages and the correct orientation
/// is opposite on each, which is the population the turn exists for and why nothing
/// keyed on orientation alone wins both.
///
/// A margin anywhere in `(0.11, 0.23)` gives that outcome. **Do not read a
/// recommended value out of that interval** -- five pairs on one chapter, which is
/// why the flag ships off.
///
/// **The first version of this table was WRONG on two rows, both the same way.**
/// `p2` and `p3`'s upright reads are multi-line; `tracing` writes them across
/// several physical lines and the instrument reading the log kept only the first, so
/// both were published as one-glyph fragments. The log's own `chars=` field
/// contradicted it and nothing checked. `p3`'s verdict INVERTED when corrected.
///
/// A missing confidence disables the ranking for that pair rather than treating it
/// as zero: `manga-ocr` and `baberu-ocr` return `None`, and scoring them as 0.0
/// would hand every one of their turns to the upright read on no evidence.
fn choose_orientation(
    upright: String,
    upright_confidence: Option<f32>,
    rotated: String,
    rotated_confidence: Option<f32>,
    margin: Option<f64>,
) -> String {
    if rotated.trim().is_empty() {
        return upright;
    }
    if let (Some(margin), Some(up), Some(down)) = (margin, upright_confidence, rotated_confidence)
        && f64::from(up) - f64::from(down) >= margin
    {
        return upright;
    }
    rotated
}

/// The balloon alone: the crop's largest white region, grown a stroke's width,
/// everything else painted white, cut to the kept extent.
///
/// **Measured before being wired**: the
/// flipped RAW bubble crop never reads `冥霜之王！` at any of thirteen angles
/// (nearest miss `冥霜之王..` at 195/200), while the flipped ISOLATED crop reads
/// it BYTE-EXACT at 175, 180 and 185 -- the surrounding art, not the residual
/// tilt, is what breaks the last glyph. Isolation without the flip is still
/// junk (an unrelated four-glyph read upright), so this composes with the flip rather than
/// replacing it. All twelve control cells stayed exact.
///
/// `None` when the crop has no credible white body (under `MIN_BODY_FRACTION`
/// of its pixels): a dark or borderless balloon falls back to the raw crop and
/// the branch behaves exactly as it shipped -- fail-open, like every gate in
/// this file. The threshold constants mirror the measured instrument (white >
/// 200 on all channels, ~12 px cross dilation, 8 px margin).
fn isolate_balloon(image: &DynamicImage) -> Option<DynamicImage> {
    const WHITE_FLOOR: u8 = 200;
    const MIN_BODY_FRACTION: f32 = 0.10;
    const DILATE_STEPS: u32 = 4; // 4 x 3 px cross reach, the instrument's 12 px
    const MARGIN: u32 = 8;
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return None;
    }
    let idx = |x: u32, y: u32| (y * w + x) as usize;
    let white: Vec<bool> = rgb
        .pixels()
        .map(|p| p.0.iter().all(|&c| c > WHITE_FLOOR))
        .collect();
    // Largest 4-connected white component, iterative so a page-sized region
    // cannot overflow the stack.
    let mut visited = vec![false; (w * h) as usize];
    let mut best: Vec<usize> = Vec::new();
    let mut stack: Vec<(u32, u32)> = Vec::new();
    for sy in 0..h {
        for sx in 0..w {
            if !white[idx(sx, sy)] || visited[idx(sx, sy)] {
                continue;
            }
            let mut component = Vec::new();
            visited[idx(sx, sy)] = true;
            stack.push((sx, sy));
            while let Some((x, y)) = stack.pop() {
                component.push(idx(x, y));
                let visit = |nx: u32, ny: u32, visited: &mut Vec<bool>, stack: &mut Vec<(u32, u32)>| {
                    let j = idx(nx, ny);
                    if white[j] && !visited[j] {
                        visited[j] = true;
                        stack.push((nx, ny));
                    }
                };
                if x > 0 {
                    visit(x - 1, y, &mut visited, &mut stack);
                }
                if x + 1 < w {
                    visit(x + 1, y, &mut visited, &mut stack);
                }
                if y > 0 {
                    visit(x, y - 1, &mut visited, &mut stack);
                }
                if y + 1 < h {
                    visit(x, y + 1, &mut visited, &mut stack);
                }
            }
            if component.len() > best.len() {
                best = component;
            }
        }
    }
    if (best.len() as f32) < (w * h) as f32 * MIN_BODY_FRACTION {
        return None;
    }
    let mut mask = vec![false; (w * h) as usize];
    for &i in &best {
        mask[i] = true;
    }
    for _ in 0..DILATE_STEPS {
        let previous = mask.clone();
        for y in 0..h {
            for x in 0..w {
                if previous[idx(x, y)] {
                    continue;
                }
                let reached = (1..=3u32).any(|d| {
                    (x >= d && previous[idx(x - d, y)])
                        || (x + d < w && previous[idx(x + d, y)])
                        || (y >= d && previous[idx(x, y - d)])
                        || (y + d < h && previous[idx(x, y + d)])
                });
                if reached {
                    mask[idx(x, y)] = true;
                }
            }
        }
    }
    let mut out = rgb;
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (w, h, 0u32, 0u32);
    for y in 0..h {
        for x in 0..w {
            if mask[idx(x, y)] {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            } else {
                out.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
    }
    let left = min_x.saturating_sub(MARGIN);
    let top = min_y.saturating_sub(MARGIN);
    let right = (max_x + 1 + MARGIN).min(w);
    let bottom = (max_y + 1 + MARGIN).min(h);
    Some(DynamicImage::ImageRgb8(out).crop_imm(left, top, right - left, bottom - top))
}

/// The marker a spot-rescued region carries as its SECOND `DetectionAnalysis`
/// label, the `SYNTHESISED_REGION_KIND` pattern exactly: the first label stays
/// the real `text` kind so the wire's `detection_confidence` convention holds,
/// and the marker is the only provenance a rescued region has.
const SPOT_RESCUED_REGION_KIND: &str = "dev.birelate.region.spot-rescued";

/// The spot rescue's geometric filters, measured on a test chapter: the site
/// watermark's plate is ~0.9% of its page while the missed display class runs
/// 24.6%, so an AREA FLOOR separates them with no string rule at all -- the
/// plate's read came back MANGLED on 14 pages, so a string filter is a belt
/// here, never the load-bearing gate. The ceiling mirrors `refuse_region`'s
/// 0.5: a box the pipeline would refuse from its own detector must not enter
/// through the side door. The overlap bound is against the SMALLER box, so
/// containment in either direction counts -- one measured page's third spot
/// box duplicates an already-lettered effect and dies here.
const SPOT_RESCUE_MIN_AREA: f64 = 0.025;
const SPOT_RESCUE_MAX_AREA: f64 = 0.5;
const SPOT_RESCUE_MAX_OVERLAP: f64 = 0.25;

/// The INK FLOOR on a minted box: a HUD-chrome mint once destroyed a 406x126
/// band of foreground artwork (`erased_px=51156`) for a box that is **0.00996
/// ink** by `spot_typography`'s own mask, because no gate on the mint path ever
/// looks at the pixels (`latin_on_cjk` structurally cannot refuse a mint: a
/// mint's spot coverage against its own box is 1.0). Measured over every mint
/// in the stored server logs (15 distinct (box, read) pairs): the lowest
/// genuine keeper is a vertical hangul column at **0.260**; then a romanized
/// read at 0.323, a title column at 0.329, then 0.374, 0.413, 0.450, a banner
/// at 0.456, `아아아` at 0.678, and two composite mints at 0.55-0.60 across
/// every plausible band offset. 0.05 sits 5x above the defect and 5x below the
/// tightest keeper, and anywhere in [0.02, 0.25] separates every row. The
/// detector-box ink minimum (0.063 over 2,441 boxes) is a DIFFERENT instrument
/// on a different population and does not price this one. The floor matters more now,
/// not less: the rescue trigger has three arms and `--spot-rescue-joined`
/// ships ON, so the mint population is wider than any stored run.
const SPOT_RESCUE_MIN_INK: f64 = 0.05;

/// Pages the rescue may buy a spot call for: every census page a display run
/// was missed on proposed 0-2 detector regions (in a 37-page census the 6
/// missed-display pages all proposed ZERO), so a sparse page is the trigger,
/// alongside any page where a refused box's sweep declined.
const SPOT_RESCUE_SPARSE_REGIONS: usize = 2;

/// A spot box in page pixels, from the sidecar's norm1000-against-the-original
/// convention (confirmed twice against Tencent's own `denormalize_coordinates`).
fn denormalize_spot_box(raw: [f64; 4], width: u32, height: u32) -> Option<(f64, f64, f64, f64)> {
    let (w, h) = (f64::from(width), f64::from(height));
    let [x1, y1, x2, y2] = raw;
    let (left, top) = (x1 * w / 1000.0, y1 * h / 1000.0);
    let (right, bottom) = (x2 * w / 1000.0, y2 * h / 1000.0);
    (right > left && bottom > top).then_some((left, top, right - left, bottom - top))
}

/// The COMPOSED admission predicate for a spot box -- area floor, area ceiling,
/// the overlap test and the ink floor in one place, because a test of two
/// halves does not test the `||` between them and this project has shipped
/// that mistake. `ink_frac` is the caller's measurement through
/// `spot_ink_fraction`, taken as a number so the tests can assert on the
/// measured population directly (`SPOT_RESCUE_MIN_INK`'s table).
fn spot_rescue_candidate(
    bounds: (f64, f64, f64, f64),
    page: (u32, u32),
    existing: &[(f64, f64, f64, f64)],
    ink_frac: f64,
) -> bool {
    let (x, y, w, h) = bounds;
    let page_area = f64::from(page.0) * f64::from(page.1);
    if page_area <= 0.0 || w <= 0.0 || h <= 0.0 {
        return false;
    }
    let share = (w * h) / page_area;
    if !(SPOT_RESCUE_MIN_AREA..=SPOT_RESCUE_MAX_AREA).contains(&share) {
        return false;
    }
    if ink_frac < SPOT_RESCUE_MIN_INK {
        return false;
    }
    for &(ex, ey, ew, eh) in existing {
        let ix = (x + w).min(ex + ew) - x.max(ex);
        let iy = (y + h).min(ey + eh) - y.max(ey);
        if ix <= 0.0 || iy <= 0.0 {
            continue;
        }
        let smaller = (w * h).min(ew * eh);
        if smaller > 0.0 && (ix * iy) / smaller > SPOT_RESCUE_MAX_OVERLAP {
            return false;
        }
    }
    true
}

/// Whether a composite mint's box crosses one of the cuts its band was
/// composed for. The half-open test is deliberate: a box that merely TOUCHES
/// a cut from one side still lies wholly in one slice's part of the band.
fn mint_spans_a_cut(bounds: (f64, f64, f64, f64), cuts: &[f64]) -> bool {
    cuts.iter()
        .any(|&cut| bounds.1 < cut && bounds.1 + bounds.3 > cut)
}

/// Typography for a rescued box, measured the way detection measures it --
/// `infer_typography` over a coarse ink mask synthesised INSIDE the box only:
/// pixels whose colour sits far from the box's own median. Without this the
/// renderer falls back to opaque theme black, which on a black banner is
/// black on black. Coarse is the contract: size, colour and writing mode come
/// from the same measured constants every detected region uses, and the render
/// is the authority on whether coarse was enough.
/// The coarse ink mask `spot_typography` measures from -- pixels whose colour
/// sits far (L1 > 180) from the box's own median, in a full-page-sized buffer.
/// Extracted so the typography and `SPOT_RESCUE_MIN_INK`'s floor share ONE
/// definition of ink that cannot drift. Returns the buffer and the clamped
/// box `(left, top, right, bottom)`, or `None` for a degenerate box.
fn spot_ink_mask(
    image: &RgbImage,
    bounds: (f64, f64, f64, f64),
) -> Option<(Vec<u8>, (u32, u32, u32, u32))> {
    const INK_DISTANCE: i32 = 180;
    let (width, height) = image.dimensions();
    let left = bounds.0.max(0.0) as u32;
    let top = bounds.1.max(0.0) as u32;
    let right = ((bounds.0 + bounds.2).ceil() as u32).min(width);
    let bottom = ((bounds.1 + bounds.3).ceil() as u32).min(height);
    if right <= left || bottom <= top {
        return None;
    }
    let mut channel = [Vec::new(), Vec::new(), Vec::new()];
    for y in (top..bottom).step_by(2) {
        for x in (left..right).step_by(2) {
            let p = image.get_pixel(x, y).0;
            for (slot, value) in channel.iter_mut().zip(p) {
                slot.push(value);
            }
        }
    }
    let median = {
        let mut m = [0u8; 3];
        for (slot, out) in channel.iter_mut().zip(m.iter_mut()) {
            slot.sort_unstable();
            *out = slot[slot.len() / 2];
        }
        m
    };
    let mut pixels = vec![0u8; width as usize * height as usize];
    for y in top..bottom {
        let row = y as usize * width as usize;
        for x in left..right {
            let p = image.get_pixel(x, y).0;
            let distance: i32 = p
                .iter()
                .zip(median)
                .map(|(&a, b)| (i32::from(a) - i32::from(b)).abs())
                .sum();
            if distance > INK_DISTANCE {
                pixels[row + x as usize] = 1;
            }
        }
    }
    Some((pixels, (left, top, right, bottom)))
}

/// The share of a spot box that is ink by `spot_ink_mask`'s definition -- the
/// number `SPOT_RESCUE_MIN_INK` gates. `None` means the box clamps to nothing
/// on this page, which the caller treats as a refusal.
fn spot_ink_fraction(image: &RgbImage, bounds: (f64, f64, f64, f64)) -> Option<f64> {
    let (pixels, (left, top, right, bottom)) = spot_ink_mask(image, bounds)?;
    let area = u64::from(right - left) * u64::from(bottom - top);
    if area == 0 {
        return None;
    }
    let ink: u64 = pixels.iter().map(|&p| u64::from(p)).sum();
    Some(ink as f64 / area as f64)
}

fn spot_typography(
    image: &RgbImage,
    bounds: (f64, f64, f64, f64),
) -> Option<InferredTypography> {
    let (width, height) = image.dimensions();
    let (pixels, (left, top, right, bottom)) = spot_ink_mask(image, bounds)?;
    let detection = KoharuLayoutDetection {
        label_id: 0,
        label: "text".to_owned(),
        score: 0.5,
        bbox: [left as f32, top as f32, right as f32, bottom as f32],
        area: (right - left) * (bottom - top),
        mask: KoharuLayoutMask {
            width,
            height,
            pixels,
        },
    };
    infer_typography(image, &detection)
}

/// Is this target read a second time with the crop turned 180 degrees?
///
/// SYNTHESISED regions only, and the justification is containment, not symmetry with
/// `rotation_candidate`: keeping the upright read of a synthesised region leaves
/// the page exactly as it renders today, so the flip can only ever change a
/// population that is already invented -- it cannot touch an ordinary balloon.
/// The neighbouring class, an upside-down balloon WITH a detected text region
/// inside, is deliberately NOT reached: it is an ordinary dialogue read, and
/// widening to it inherits the refuted 1,901-region in-bubble population.
///
/// Takes the SNAPSHOT rather than a precomputed flag, so a test can build a scene
/// with and without the `SYNTHESISED_REGION_KIND` label and catch an
/// implementation that reads the wrong carrier -- `Region.label` does not hold
/// this marker; the second `DetectionAnalysis` label does.
fn flip_candidate(scene: &Snapshot, region: EntityId, enabled: bool) -> bool {
    enabled && is_synthesised(scene, region)
}

/// How much MORE confident the flipped read must be to take a synthesised bubble
/// from its upright read.
///
/// **Unfitted, and carried as a constant rather than a flag on purpose.** The
/// stored score corpora cannot price this population in either direction -- the
/// upright sweeps' good-label set is near-circular (its "truth" is an earlier
/// native read), and the target slice itself appears in no scored corpus.
/// `0.17` is borrowed from the orientation margin adopted after a render as a
/// conservative starting point; the flip's own render is the authority, and
/// the log line below carries both scores precisely so that render can move
/// this number on evidence.
const FLIP_CONFIDENCE_MARGIN: f64 = 0.17;

/// The flip's refine grid. `180.0` is the flip itself; `175.0` and `185.0` are
/// its measured neighbours: on the instrument's crop bytes all three read the
/// target balloon BYTE-EXACT, while the pipeline's own crop bytes at `180.0`
/// landed on the wrong side of the model's `王`/`用` decision boundary (the
/// model is sensitive to the exact crop bytes). The neighbours are independent
/// rolls of the same knife edge, and the best-scoring candidate ships -- in
/// every observed pair the exact read outscored its near-miss. Three reads on a
/// ~1-region-per-chapter population; unscored and rotate-incapable engines never
/// pay any of it.
const FLIP_REFINE_ANGLES: [f64; 3] = [175.0, 180.0, 185.0];

/// The best-scoring NON-EMPTY candidate of the flip's refine sweep, with its
/// angle for the log line. All-empty (or no candidates) returns the empty
/// string, which `choose_flip`'s first guard turns into "keep upright" -- the
/// fail-safe every arm of this mechanism funnels to. A `None` confidence never
/// beats a `Some`: scoring absent as `-inf` is exactly the rule
/// `choose_orientation`'s doc demands for missing scores.
fn best_flip(candidates: Vec<(f64, String, Option<f32>)>) -> (f64, String, Option<f32>) {
    candidates
        .into_iter()
        .filter(|(_, text, _)| !text.trim().is_empty())
        .max_by(|a, b| {
            a.2.unwrap_or(f32::NEG_INFINITY)
                .partial_cmp(&b.2.unwrap_or(f32::NEG_INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((180.0, String::new(), None))
}

/// Which read of a flipped synthesised bubble to keep, WITH its confidence.
///
/// **The asymmetry is the exact INVERSE of `choose_orientation`'s, and reusing
/// that function here was the reviewed design's one fatal defect.** The column
/// turn exists to rescue sideways columns, so there the ROTATED read is the
/// default winner and a missing confidence still hands it the pair. Here the
/// upright read is today's shipping behaviour and eleven of the twelve genuine
/// textless-bubble reads in the 2,519-bubble census are upright, so the
/// UPRIGHT read is the default winner, and the flip must clearly beat it:
///
/// - an empty flipped read never wins;
/// - a missing confidence on EITHER side keeps the upright read -- `manga-ocr`,
///   `baberu-ocr` and Ollama report `None`, and an unconditional flip there
///   would replace every genuine read on no evidence at all;
/// - otherwise the flipped read wins only by `FLIP_CONFIDENCE_MARGIN` or more.
///
/// Fluent junk can outscore the truth (`choose_orientation`'s table shows it on
/// this mechanism's sibling), so a miss here is expected to be POSSIBLE: the
/// failure mode is "the balloon stays wrong as it ships today", never "a right
/// balloon goes wrong". Returns the kept read's own confidence beside it,
/// because the wire reports one and pairing one read's text with the other's
/// number would be worse than reporting none.
fn choose_flip(
    upright: String,
    upright_confidence: Option<f32>,
    flipped: String,
    flipped_confidence: Option<f32>,
) -> (String, Option<f32>) {
    if flipped.trim().is_empty() {
        return (upright, upright_confidence);
    }
    if let (Some(up), Some(down)) = (upright_confidence, flipped_confidence)
        && f64::from(down) - f64::from(up) >= FLIP_CONFIDENCE_MARGIN
    {
        return (flipped, flipped_confidence);
    }
    (upright, upright_confidence)
}

/// The upright pass's angle grid: 12 coarse angles on a 30-degree
/// grid, then 4 refine angles around the best-scoring non-empty coarse read.
/// Coarse-only was measured and is NOT enough -- the fit population drops from
/// 13/17 to 9/17 without the refine rows, so the refine pass is load-bearing,
/// not decoration.
const UPRIGHT_COARSE: [f64; 12] = [
    0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
];
const UPRIGHT_REFINE: [f64; 4] = [-15.0, -7.5, 7.5, 15.0];

/// One swept read: the sidecar-rotated crop's text after the standard
/// normalise-strip-refuse chain, and the decode's length-normalised log-prob.
struct UprightRow {
    angle: f64,
    text: String,
    mlp: Option<f64>,
}

fn upright_score(row: &UprightRow) -> f64 {
    row.mlp.unwrap_or(f64::NEG_INFINITY)
}

/// The adjudication normaliser of the angle-sweep instrument, ported:
/// strip quote/asterisk/period edges, drop every `！`/`!`, collapse dash-runs
/// (`—`, `-`, `–`, `ー`, `─`) to one em dash, unify `・` to `·`, drop a trailing
/// `。`, remove all whitespace. The selection rule below ranks NORMALISED reads;
/// what ships is always a RAW read -- the normaliser decides between candidates
/// and never rewrites one.
fn upright_norm(s: &str) -> String {
    let edge: &[char] = &['。', '．', '.', '"', '\'', '`', '*', '\n', ' '];
    let s = s.trim().trim_matches(|c| edge.contains(&c));
    let s: String = s.chars().filter(|c| *c != '！' && *c != '!').collect();
    let mut out = String::new();
    let mut in_dash = false;
    for c in s.trim().chars() {
        if matches!(c, '—' | '-' | '–' | 'ー' | '─') {
            if !in_dash {
                out.push('—');
            }
            in_dash = true;
        } else {
            in_dash = false;
            out.push(if c == '・' { '·' } else { c });
        }
    }
    out.trim_end_matches('。')
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

/// Alphanumeric skeleton of a normalised read. CJK ideographs count (`轰` is
/// alphabetic to Unicode); punctuation, dashes and arrows do not.
fn upright_key(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).collect()
}

/// Which swept read ships, if any -- the `silence-then-climb` rule plus a
/// ship floor. Measured on a 25-crop refused-box population:
/// the bare rule reads **13/17 text with 0/8 artwork fabrications surviving the
/// downstream gates**; with the floor it ships **11/17**, giving up exactly the
/// two single-glyph EXACTs (`快`, `轰`) to silence every junk lettering the
/// rendered chapter showed (the two text residuals are Pareto-proven
/// unreachable by any monotone function of the score, and the two head-cut
/// composites fail at every angle).
///
/// Two layers, each answering a different question:
///
/// **H0 -- is there text at all.** If punctuation-only/empty reads outnumber the
/// best-supported non-empty alphanumeric skeleton across the sweep, the sweep
/// mostly saw marks: ship nothing. On the fit population this fires on 5 of 8
/// artwork crops and 0 of 17 text crops (every text crop has literally zero
/// empty rows in its sweep).
///
/// **Then the corroborated climb -- which read carries the most of it.** Anchor
/// on the argmax-score read, then climb the containment order of normalised
/// reads: a climb is admitted only if the longer read scores within 0.40 of the
/// anchor AND every character it adds was seen by some other angle (junk
/// extensions are single-witness; real recoveries are corroborated). Candidate
/// climbs rank by how many rows of the sweep they explain, so a plateau's read
/// beats a lone confident one. The shipped read's characters are a superset of
/// the anchor's by construction -- the project's don't-lose-information rule as
/// an algorithm.
///
/// The engine's meta-sentences (`图片中...`) are deliberately NOT excluded here:
/// if one wins, it ships and `labels.rs`'s `Refusal::ImageDescription` refuses
/// the lettering downstream -- that gate staying in the path is load-bearing,
/// and the test naming it pins the contract.
fn upright_select(rows: &[UprightRow]) -> Option<usize> {
    let norms: Vec<String> = rows.iter().map(|r| upright_norm(&r.text)).collect();
    let keys: Vec<String> = norms.iter().map(|n| upright_key(n)).collect();

    // H0: silence.
    let n_empty = keys.iter().filter(|k| k.is_empty()).count();
    let mut support: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for key in &keys {
        if !key.is_empty() {
            *support.entry(key.as_str()).or_default() += 1;
        }
    }
    let best_support = support.values().copied().max().unwrap_or(0);
    if n_empty > best_support {
        return None;
    }

    // Anchor: argmax mean_logprob over rows with non-empty RAW text; ties go to
    // the smaller angle. An anchor whose normalised form is empty has no defence
    // in the containment order ("" is a substring of everything) -- ship nothing.
    let cands: Vec<usize> = (0..rows.len())
        .filter(|&i| !rows[i].text.trim().is_empty())
        .collect();
    let &anchor = cands.iter().max_by(|&&a, &&b| {
        upright_score(&rows[a])
            .total_cmp(&upright_score(&rows[b]))
            .then(rows[b].angle.total_cmp(&rows[a].angle))
    })?;
    if norms[anchor].is_empty() {
        return None;
    }

    let subsumption = |s: &str| -> usize {
        norms
            .iter()
            .filter(|n| !n.is_empty() && s.contains(n.as_str()))
            .count()
    };
    // Angle support of a normalised read -- how many distinct swept angles
    // produced exactly it. Feeds the two climb guards below.
    let support_of = |text: &str| -> usize {
        let angles: std::collections::HashSet<u64> = (0..rows.len())
            .filter(|&i| norms[i] == text)
            .map(|i| rows[i].angle.to_bits())
            .collect();
        angles.len()
    };
    // Corroboration, with the DUPLICATION GUARD an adversarial review added:
    // set-difference corroboration is vacuous when a climb only
    // REPEATS characters already present -- the added set is empty and the
    // witness test constrains nothing, which shipped a hallucinated duplicated
    // glyph (裂裂风之王！ over a nine-angle 裂风之王！ plateau) on a held-out
    // crop. A duplication climb must witness ITSELF: its exact normalised
    // text at two or more angles.
    let corroborated = |cur: &str, cand: &str| -> bool {
        let cur_chars: std::collections::HashSet<char> = cur.chars().collect();
        let added: Vec<char> = cand.chars().filter(|c| !cur_chars.contains(c)).collect();
        if added.is_empty() {
            return support_of(cand) >= 2;
        }
        let mut pool: std::collections::HashSet<char> = std::collections::HashSet::new();
        for &i in &cands {
            if norms[i] != cand {
                pool.extend(norms[i].chars());
            }
        }
        added.iter().all(|c| pool.contains(c))
    };

    let floor = upright_score(&rows[anchor]) - 0.40;
    let mut cur = norms[anchor].clone();
    for _ in 0..4 {
        // The PLATEAU GUARD rides beside corroboration (same adversarial pass):
        // a climb may not leave a plateau for a read with materially less angle
        // support -- the -1 slack is the earlier rule's, measured there. Neither guard
        // fires on the fit population (all three fit climbs add genuinely new
        // characters from wider-or-equal support); only held-out data could
        // catch the hole.
        let climbs: Vec<usize> = cands
            .iter()
            .copied()
            .filter(|&i| {
                norms[i].chars().count() > cur.chars().count()
                    && norms[i].contains(cur.as_str())
                    && upright_score(&rows[i]) >= floor
                    && support_of(&norms[i]) + 1 >= support_of(&cur)
                    && corroborated(&cur, &norms[i])
            })
            .collect();
        let Some(&best) = climbs.iter().max_by(|&&a, &&b| {
            subsumption(&norms[a])
                .cmp(&subsumption(&norms[b]))
                .then(upright_score(&rows[a]).total_cmp(&upright_score(&rows[b])))
                .then(norms[a].chars().count().cmp(&norms[b].chars().count()))
                .then(rows[b].angle.total_cmp(&rows[a].angle))
        }) else {
            break;
        };
        if norms[best] == cur {
            break;
        }
        cur = norms[best].clone();
    }

    let pick = cands
        .iter()
        .copied()
        .filter(|&i| norms[i] == cur)
        .max_by(|&a, &b| {
            upright_score(&rows[a])
                .total_cmp(&upright_score(&rows[b]))
                .then(rows[b].angle.total_cmp(&rows[a].angle))
        })?;
    // The SHIP FLOOR: a pick under three glyphs does not ship. Counted on the
    // RAW text, non-whitespace -- the normaliser drops `！`, so a normalised
    // count would kill a three-glyph keeper (`快跑！` in the tests) that the floor's
    // pricing explicitly kept. Reproduced on the fit population before wiring:
    // 13/17 -> 11/17, the two lost cells the priced single-glyph EXACTs (`快`,
    // `轰`), post-gate fabrications still 0/8 -- and every junk lettering the
    // rendered chapter showed (`地产` "REAL ESTATE", `舞` "DANCE", `1舞` on a
    // seam join) is at most two glyphs, so all of them decline to silence.
    if rows[pick]
        .text
        .chars()
        .filter(|c| !c.is_whitespace())
        .count()
        < UPRIGHT_SHIP_FLOOR
    {
        return None;
    }
    Some(pick)
}

/// The minimum RAW non-whitespace glyph count an upright pick
/// needs to ship. Three is the smallest value that removes every junk lettering
/// observed on the rendered chapter while costing it nothing it actually
/// lettered -- the only sub-three recovery the chapter kept (a three-glyph
/// exclamation, `快跑！` in the tests) is exactly
/// three, which is also why the count is raw and not normalised.
const UPRIGHT_SHIP_FLOOR: usize = 3;

/// Sweep one crop through the rotation grid. Runs on a blocking thread (the
/// engine's `request` uses `block_on`, same contract as `infer_text`). Every
/// read passes the IDENTICAL normalise-strip-refuse chain the primary read
/// went through, so the selected text is comparable with -- and substitutable
/// for -- an ordinary read. Any read failing mid-sweep fails the whole sweep,
/// and the caller fails OPEN.
fn upright_sweep(
    engine: &Arc<Mutex<OllamaVision>>,
    image: &DynamicImage,
) -> Result<Vec<UprightRow>> {
    let engine = engine
        .lock()
        .map_err(|_| anyhow!("OCR model lock is poisoned"))?;
    let mut rows: Vec<UprightRow> = Vec::new();
    let read_at = |angle: f64, rows: &mut Vec<UprightRow>| -> Result<()> {
        let (raw, mlp) = engine.inference_rotated(image, angle)?;
        let text = clean_read(raw);
        rows.push(UprightRow { angle, text, mlp });
        Ok(())
    };
    for angle in UPRIGHT_COARSE {
        read_at(angle, &mut rows)?;
    }
    let best = rows
        .iter()
        .filter(|r| !r.text.trim().is_empty())
        .max_by(|a, b| upright_score(a).total_cmp(&upright_score(b)))
        .map(|r| r.angle);
    if let Some(center) = best {
        for offset in UPRIGHT_REFINE {
            let angle = (center + offset).rem_euclid(360.0);
            if rows.iter().any(|r| (r.angle - angle).abs() < 1e-6) {
                continue;
            }
            read_at(angle, &mut rows)?;
        }
    }
    Ok(rows)
}

/// `unread` is the bounds of boxes no engine was ever allowed to read, because
/// `ImplausibleRegions::skip` dropped them for size before they could become an
/// `OcrResult`. They are withdrawn **unconditionally** -- there is no text to run
/// `withdraw_from_mask` over, and its absence is the point: a region nothing read
/// is a stronger case for sparing the artwork than one that was read and came back
/// unreadable. Empty unless `ImplausibleRegions::veto_mask` is on, so the shipping
/// arm is byte-identical to what it was before this parameter existed.
fn write_illegible_veto(
    edit: &mut koharu_scene::Edit,
    scene: &koharu_scene::Snapshot,
    page: EntityId,
    source: &DynamicImage,
    pending: &[(OcrResult, Generation)],
    source_language: Option<Language>,
    unread: &[(f64, f64, f64, f64)],
    scope_watermark_refusals: bool,
    spot_refused: &std::collections::HashSet<EntityId>,
    levers: MisreadLevers,
) -> Result<()> {
    // A fallible loop rather than `filter_map`, because reading the role can
    // fail and a `?` inside a closure cannot escape it. Silently treating a
    // scene-read error as "not free-standing" would turn a broken lookup into a
    // quietly disabled rule -- the shape of failure this file already has a
    // warning about two functions up.
    let mut refused: Vec<(bool, (f64, f64, f64, f64))> =
        unread.iter().map(|bounds| (true, *bounds)).collect();
    let mut kept = Vec::new();
    for (result, _) in pending {
        let Some(bounds) = geometry_extents(&result.geometry) else {
            continue;
        };
        // A mint joins NEITHER partition. The role read below cannot resolve
        // it (the frozen snapshot answers `EntityNotFound`, which once failed
        // every minting page with a 500), and it does not belong here even resolved:
        // its read already passed `withdraw_from_mask` free-standing before
        // minting, with the same inputs this loop would recompute, so it can
        // never join `refused`; and joining `kept` would punch a hole in an
        // overlapping refusal's veto -- re-enabling an erase under the
        // lettering that `--spot-rescue-erase` OFF promised not to make. The
        // mint's erase is its own flag's decision, through
        // `write_rescue_mask`, which unions AFTER this veto.
        if result.minted {
            continue;
        }
        let role = scene.component::<TextRole>(result.content)?;
        // The spotting verdict joins the composed predicate here so one veto
        // writer covers both: a read the gate refused withdraws its erase exactly
        // as an illegible one does.
        let withdraw = spot_refused.contains(&result.region)
            || withdraw_from_mask(
                &result.text,
                source_language,
                free_standing(role.as_ref().map(|value| value.role.as_str())),
                scope_watermark_refusals,
                levers,
            );
        if withdraw {
            refused.push((withdraw, bounds));
        } else {
            kept.push((withdraw, bounds));
        }
    }
    if refused.is_empty() {
        return Ok(());
    }
    let (width, height) = (source.width(), source.height());
    if width == 0 || height == 0 {
        return Ok(());
    }
    // The rule `write_mask` dilates by, shared pre-clamp; this site's tail
    // stays here (`super::dilation_radius`'s doc owns the why).
    let radius = super::dilation_radius(width.max(height)).max(0.0) as i64;
    let grow = |bounds: (f64, f64, f64, f64)| {
        let (min_x, min_y, max_x, max_y) = bounds;
        (
            (min_x.floor() as i64 - radius).clamp(0, i64::from(width)),
            (min_y.floor() as i64 - radius).clamp(0, i64::from(height)),
            (max_x.ceil() as i64 + radius).clamp(0, i64::from(width)),
            (max_y.ceil() as i64 + radius).clamp(0, i64::from(height)),
        )
    };
    let refused: Vec<_> = refused.into_iter().map(|(_, b)| grow(b)).collect();
    let kept: Vec<_> = kept.into_iter().map(|(_, b)| grow(b)).collect();
    let covered = |boxes: &[(i64, i64, i64, i64)], x: i64, y: i64| {
        boxes
            .iter()
            .any(|(x0, y0, x1, y1)| x >= *x0 && x < *x1 && y >= *y0 && y < *y1)
    };

    let mut veto = image::GrayImage::new(width, height);
    let mut spared = 0_u64;
    for (x0, y0, x1, y1) in &refused {
        for y in *y0..*y1 {
            for x in *x0..*x1 {
                if covered(&kept, x, y) {
                    continue;
                }
                veto.get_pixel_mut(x as u32, y as u32).0[0] = u8::MAX;
                spared += 1;
            }
        }
    }
    if spared == 0 {
        return Ok(());
    }
    tracing::info!(
        target: "koharu_pipeline::ocr",
        refused = refused.len(),
        kept = kept.len(),
        spared_px = spared,
        "vetoed unreadable regions from the erase mask"
    );

    let mut bytes = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageLuma8(veto).write_to(&mut bytes, image::ImageFormat::Png)?;
    edit.set_asset(
        page,
        &koharu_scene::AssetRole::new("text-mask-veto")?,
        koharu_scene::AssetInput::new(
            Arc::<[u8]>::from(bytes.into_inner()),
            "image/png",
            koharu_scene::AssetMetadata {
                width: Some(width),
                height: Some(height),
                attributes: std::collections::BTreeMap::new(),
            },
        ),
    )?;
    Ok(())
}

/// The pixels a STASH RECOVERY must erase, published as `text-mask-rescue` --
/// the additive counterpart of `text-mask-veto`.
///
/// Detection's `text-mask` deliberately excludes the boxes the size/fragment
/// ceiling refuses, and scene authorship forbids this stage editing that asset
/// -- the exact constraint that created `text-mask-veto`. So a recovered
/// target's erase travels the same way: a role of this stage's own, which
/// `stages::inpainting::prepare` UNIONS in AFTER subtracting the veto, so a
/// recovery erases even where an overlapping refusal is spared.
///
/// Bounds are grown by the same `round(max_dim / 1024 * 6)` radius the
/// detection mask dilates by and the veto grows by -- the pixels to erase are
/// the dilated ones, or the glyphs' anti-aliased skirts survive as the exact
/// ghost ink this exists to remove.
/// Whether a recovered target is NARROW enough for its whole hull to join the
/// erase mask. The RENDER is the authority for both arms (every page looked
/// at):
///
/// - narrow columns hull-erased BEAUTIFULLY -- short sides 124, 122, 197 and
///   193: the column vanishes, LaMa reconstructs a clean band, the title
///   letters in its place;
/// - wide effect boxes hull-erased DESTRUCTIVELY -- one with short side 784
///   lost its background scene to flat slabs, 56% of the page's pixels.
///
/// The population separates on a measured hole: over all 25 refused-box targets
/// of a test chapter the short sides run ... 193, 197, 200, 203 | 309, 335, 428
/// ... (runtime bounds). 256 sits inside it.
/// A wide recovery still LETTERS -- beside its preserved source, exactly as
/// before this constant existed; erasing wide drawn effects needs an INK-scoped
/// mask, and all three simple ink rules were refuted by looking at their masks.
const UPRIGHT_RESCUE_MAX_SHORT_SIDE: f64 = 256.0;

/// The geometry gate, named so the composed behaviour is testable at
/// the predicate the caller calls.
fn rescue_narrow((min_x, min_y, max_x, max_y): (f64, f64, f64, f64)) -> bool {
    (max_x - min_x).min(max_y - min_y) <= UPRIGHT_RESCUE_MAX_SHORT_SIDE
}

/// Luma floor above which a pixel counts as the mark's BRIGHT surround --
/// burst paper and halo ring both clear it, panel streaks and glyph ink both
/// miss it. Measured on the test exhibit: the halo
/// medians 241 and the panel's brightest streaks stay under ~140.
const SCREAM_BRIGHT_FLOOR: f32 = 180.0;
/// Window padding around the hull, so a glyph whose enclosure crosses the
/// hull edge either becomes a border-touching component (spared, erased by
/// nobody) or a fully-enclosed one -- never a half-claimed one.
const SCREAM_WINDOW_PAD: i64 = 32;
/// An enclosed dark component below this is a sweat drop or a screentone
/// speck, not a glyph -- the exhibit's glyph cores run thousands of pixels each.
const SCREAM_MIN_COMPONENT_PX: usize = 256;
/// Below this much total enclosed ink the device abstains entirely and the
/// mint ships today's behaviour.
const SCREAM_MIN_TOTAL_INK_PX: usize = 1024;
/// Erosion before colour sampling: the outermost pixels of
/// a drawn stroke are an anti-aliased blend, and a median over them ships mud.
const SCREAM_CORE_EROSION: usize = 3;
/// The replacement band's height as a fraction of the hull's short side.
const SCREAM_BAND_FRACTION: f64 = 0.35;
/// How far past the dilated ink the erase expands THROUGH BRIGHT PIXELS ONLY.
/// The drawn glyphs bleed a soft pink glow into the surrounding paper; the
/// first render left that ring as LaMa's context and it rebuilt every erased
/// glyph as a grey-brown ghost smear (looked at).
/// Expanding through bright claims the glow and the paper around it -- both
/// rebuilt as clean white -- while every dark pixel stays a wall: the burst
/// outline, the panel streaks, the face, and the KEPT drawn head glyph's own
/// ink (whose glow reaches 200 px into this window and must survive with it).
const SCREAM_GLOW_RADIUS: usize = 48;
/// Radius of the morphological CLOSING applied to the bright set before the
/// flood. The exhibit's `악` sends a descender through the lobe boundary onto the
/// panel, and its taper tip dissolves into the border-connected dark -- one
/// sub-8 px isthmus opened the whole stroke, and both first renders shipped
/// its ghost (looked at). Sealing gaps up to
/// `2 * SCREAM_SEAL_RADIUS` makes such a stroke enclosed and claimable, and
/// as the same operation drops dark features THINNER than the gap -- the
/// burst outline foremost -- out of the dark set entirely: neither ink nor
/// wall, so the outline is never erased and never opens what it touches.
const SCREAM_SEAL_RADIUS: usize = 4;
/// A bright pixel with channel spread at or under this is PAPER, not glow --
/// the exhibit's paper is [242,240,240] (spread 2) while its glow runs pink
/// (spread 25+). Paper's median is the flat-paint colour.
const SCREAM_PAPER_SATURATION_MAX: i32 = 15;
/// Each whole-mark growth step expands the analysis window this far on the
/// sides where cropped-glyph evidence stands. Larger than a glyph's stroke
/// but smaller than a panel, so growth converges in a few steps on a real
/// cascade and the cap bounds a pathological one.
const SCREAM_GROW_STEP: f64 = 192.0;
/// Growth iterations are capped -- a cascade that has not converged by here
/// is not a cascade, and the fallback is the shipped tail-only device.
const SCREAM_GROW_MAX_STEPS: usize = 8;
/// An OPEN component larger than this is art (the panel, the face), never a
/// window-cropped glyph -- the exhibit's head glyphs' strokes run 15-25k px.
const SCREAM_MAX_GLYPH_PX: usize = 40_000;
/// An erased pixel whose ORIGINAL sits within this of the paper median per
/// channel is painted flat instead of inpainted. LaMa given the large
/// paper-side hole invented a grey smoke cloud on both refinement renders
/// (looked at in crops); for pixels that were
/// provably paper, flat paper IS the right answer, not an approximation --
/// the same argument `flat_fill_regions` records, applied per pixel because
/// this region's hull spans paper AND panel and so fails the whole-region
/// uniformity guard structurally.
const SCREAM_PAPER_TOLERANCE: i32 = 16;
/// Principal-axis angles inside this of zero snap to zero -- an upright mark
/// letters upright, without a spurious tilt from asymmetric glyphs.
const SCREAM_ANGLE_SNAP_DEGREES: f32 = 8.0;
/// Beyond this the axis estimate is a vertical cascade this build does not
/// place; the device keeps the styled fill but letters unrotated.
const SCREAM_ANGLE_LIMIT_DEGREES: f32 = 60.0;

/// The content gate: a read is a SCREAM when one glyph repeats into
/// at least half of a run of three or more. `아아아` and `AAAH!` fire;
/// `（众）`, `숙...`, `HUD INTERFACE`, `别过来！` and every dialogue line
/// checked do not. Deliberately script-blind -- the drawn-scream class is not
/// a property of one language.
fn scream_read(text: &str) -> bool {
    let glyphs: Vec<char> = text.chars().filter(|c| c.is_alphanumeric()).collect();
    if glyphs.len() < 3 {
        return false;
    }
    let mut counts = std::collections::HashMap::new();
    for glyph in &glyphs {
        *counts.entry(*glyph).or_insert(0_usize) += 1;
    }
    let repeated = counts.values().copied().max().unwrap_or(0);
    repeated * 2 >= glyphs.len()
}

/// The composed predicate the mint calls: a WIDE recovery
/// (`!rescue_narrow` -- the same 256 px bound that already separates the test
/// exhibit from the narrow `숙...` and `（众）` mints) whose read is a scream.
/// Named and tested as one function because a test of two halves does not
/// test the `&&` between them.
fn scream_mark(bounds: (f64, f64, f64, f64), read: &str) -> bool {
    !rescue_narrow(bounds) && scream_read(read)
}

/// What `scream_mark_ink` measured off the drawn mark: the ink-scoped erase
/// mask and the style the replacement letters with.
struct ScreamMarkInk {
    /// The erase mask, in window coordinates -- `u8::MAX` over the dilated
    /// enclosed ink plus the bright-only glow expansion around it
    /// (`SCREAM_GLOW_RADIUS`), so LaMa rebuilds from clean paper.
    mask: GrayImage,
    /// The subset of `mask` whose ORIGINAL pixels are paper
    /// (`SCREAM_PAPER_TOLERANCE` of `paper`) -- painted flat by the
    /// inpainting stage instead of being handed to the model.
    flat: GrayImage,
    /// The paper median the flat subset is painted with.
    paper: [u8; 3],
    /// The window's page-space origin.
    origin: (u32, u32),
    /// The eroded core's bright end, the gradient's top stop.
    head: [u8; 3],
    /// The eroded core's dark end, the gradient's bottom stop.
    tail: [u8; 3],
    /// The bright ring around the glyphs, the halo stroke's colour.
    halo: [u8; 3],
    /// The ink's principal axis in degrees, y-down, positive = down-right.
    /// Zero when the mark is upright or the estimate is out of range.
    angle_degrees: f32,
    /// The ink centroid in PAGE coordinates -- the band centres here.
    centroid: (f64, f64),
    /// Enclosed ink pixels before dilation, for the log.
    ink_px: u64,
}

/// One pass of the topological analysis plus what the whole-mark GROWTH
/// reads off it: where the enclosed ink actually is, and whether a
/// glyph-scale OPEN component presses against an expandable analysis border
/// -- the signature of a glyph the window has cropped (the exhibit's drawn head:
/// the spot box covered only the tail, and the head's glyphs cross the
/// window edge one after another as it grows).
struct ScreamGrowth {
    ink: ScreamMarkInk,
    /// The kept components' union bbox, PAGE coordinates (x, y, w, h).
    ink_bounds: (f64, f64, f64, f64),
    /// Cropped-glyph evidence per expandable border side
    /// (left, top, right, bottom).
    candidates: [bool; 4],
}

/// Segment the drawn mark's ink inside the mint's hull, topologically: the
/// mark's glyphs -- red heads on burst paper, near-black tails inside their
/// own halos -- are dark components fully ENCLOSED by bright pixels, while
/// panel streaks, the burst outline and the surrounding art all reach the
/// window border and survive the flood.
///
/// This is deliberately NOT one of the refuted colour-distance trio
/// (bg-dist / glyph-dist / otsu): all three are colour-distance rules, and on
/// a multi-tone window (white burst above, near-black panel below -- exactly
/// the test exhibit) each one claims a background band as ink. Verified on the
/// exhibit's own strip: bg-dist claims 77% of the
/// crop and inverts, the topological rule claims 17.7% and it is the glyphs.
/// Test-only unfenced wrapper -- every production caller goes through
/// `grow_scream_mark` or `scream_ink_analysis` WITH the existing-region
/// fence; the tests use this to measure the fence's own effect.
#[cfg(test)]
fn scream_mark_ink(image: &RgbImage, bounds: (f64, f64, f64, f64)) -> Option<ScreamMarkInk> {
    scream_ink_analysis(image, bounds, &[], &[], bounds).map(|growth| growth.ink)
}

/// The analysis behind `scream_mark_ink`, with the growth signals exposed
/// and two fences on what may become ink or evidence:
/// - components inside EXISTING region boxes are excluded -- a neighbour's
///   balloon glyphs are enclosed dark in bright too, and neither the erase
///   nor the growth may claim them;
/// - components inside settled ARTWORK regions (panels, bubbles) are
///   excluded UNLESS they intersect the `validated` box -- the rescue's own
///   model-proposed extent. A character's eyes and brows are dark,
///   glyph-scale and enclosed by bright skin: topological ink to the letter,
///   and exactly what the first growth render swallowed before this fence
///   (the exhibit's own face, hull 827k px against the 552k ceiling).
///   The tail glyphs live ON the panel, which is why the validated box is
///   exempt rather than the fence absolute.
fn scream_ink_analysis(
    image: &RgbImage,
    bounds: (f64, f64, f64, f64),
    existing: &[(f64, f64, f64, f64)],
    artwork: &[(f64, f64, f64, f64)],
    validated: (f64, f64, f64, f64),
) -> Option<ScreamGrowth> {
    let (page_w, page_h) = image.dimensions();
    if page_w == 0 || page_h == 0 {
        return None;
    }
    let x0 = ((bounds.0.floor() as i64) - SCREAM_WINDOW_PAD).clamp(0, i64::from(page_w)) as u32;
    let y0 = ((bounds.1.floor() as i64) - SCREAM_WINDOW_PAD).clamp(0, i64::from(page_h)) as u32;
    let x1 = (((bounds.0 + bounds.2).ceil() as i64) + SCREAM_WINDOW_PAD)
        .clamp(0, i64::from(page_w)) as u32;
    let y1 = (((bounds.1 + bounds.3).ceil() as i64) + SCREAM_WINDOW_PAD)
        .clamp(0, i64::from(page_h)) as u32;
    let (w, h) = ((x1.checked_sub(x0)?) as usize, (y1.checked_sub(y0)?) as usize);
    if w < 8 || h < 8 {
        return None;
    }
    let index = |x: usize, y: usize| y * w + x;
    let mut bright = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let pixel = image.get_pixel(x0 + x as u32, y0 + y as u32).0;
            let luma = 0.299 * f32::from(pixel[0])
                + 0.587 * f32::from(pixel[1])
                + 0.114 * f32::from(pixel[2]);
            bright[index(x, y)] = luma >= SCREAM_BRIGHT_FLOOR;
        }
    }
    // Seal the bright surround (morphological closing, `SCREAM_SEAL_RADIUS`)
    // before the flood: a sub-8 px gap where a stroke's taper meets open dark
    // no longer opens the whole stroke, and dark features thinner than the
    // gap -- the burst outline -- drop out of the dark set entirely.
    let mut bright_closed = bright.clone();
    for _ in 0..SCREAM_SEAL_RADIUS {
        let previous = bright_closed.clone();
        for y in 0..h {
            for x in 0..w {
                let i = index(x, y);
                if !previous[i]
                    && ((x > 0 && previous[index(x - 1, y)])
                        || (x + 1 < w && previous[index(x + 1, y)])
                        || (y > 0 && previous[index(x, y - 1)])
                        || (y + 1 < h && previous[index(x, y + 1)]))
                {
                    bright_closed[i] = true;
                }
            }
        }
    }
    for _ in 0..SCREAM_SEAL_RADIUS {
        let previous = bright_closed.clone();
        for y in 0..h {
            for x in 0..w {
                let i = index(x, y);
                if previous[i] {
                    // Out-of-window counts as BRIGHT: eroding against the
                    // window edge would turn the whole border ring dark,
                    // fuse every border-touching sliver into one giant open
                    // component, and blind the growth's cropped-glyph
                    // evidence behind the area ceiling.
                    let interior = (x == 0 || previous[index(x - 1, y)])
                        && (x + 1 >= w || previous[index(x + 1, y)])
                        && (y == 0 || previous[index(x, y - 1)])
                        && (y + 1 >= h || previous[index(x, y + 1)]);
                    bright_closed[i] = interior;
                }
            }
        }
    }
    let sealed_dark: Vec<bool> = bright_closed.iter().map(|&value| !value).collect();
    // Flood the SEALED dark set from the window border; what the flood cannot
    // reach is enclosed by bright.
    let mut open = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    let seed = |i: usize, open: &mut Vec<bool>, stack: &mut Vec<usize>| {
        if sealed_dark[i] && !open[i] {
            open[i] = true;
            stack.push(i);
        }
    };
    for x in 0..w {
        seed(index(x, 0), &mut open, &mut stack);
        seed(index(x, h - 1), &mut open, &mut stack);
    }
    for y in 0..h {
        seed(index(0, y), &mut open, &mut stack);
        seed(index(w - 1, y), &mut open, &mut stack);
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        for (nx, ny) in [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ] {
            if nx < w && ny < h {
                let n = index(nx, ny);
                if sealed_dark[n] && !open[n] {
                    open[n] = true;
                    stack.push(n);
                }
            }
        }
    }
    // A component bbox in window coords, and whether its PAGE-space box
    // overlaps any existing region -- those pixels belong to someone else.
    let component_bbox = |component: &[usize]| {
        let (mut bx0, mut by0, mut bx1, mut by1) = (usize::MAX, usize::MAX, 0_usize, 0_usize);
        for &i in component {
            let (x, y) = (i % w, i / w);
            bx0 = bx0.min(x);
            by0 = by0.min(y);
            bx1 = bx1.max(x);
            by1 = by1.max(y);
        }
        (bx0, by0, bx1, by1)
    };
    let overlaps = |(bx0, by0, bx1, by1): (usize, usize, usize, usize),
                    boxes: &[(f64, f64, f64, f64)]| {
        let (px0, py0) = (f64::from(x0) + bx0 as f64, f64::from(y0) + by0 as f64);
        let (px1, py1) = (f64::from(x0) + bx1 as f64 + 1.0, f64::from(y0) + by1 as f64 + 1.0);
        boxes
            .iter()
            .any(|&(ex, ey, ew, eh)| px0 < ex + ew && ex < px1 && py0 < ey + eh && ey < py1)
    };
    let fenced = |bbox: (usize, usize, usize, usize)| {
        overlaps(bbox, existing)
            || (overlaps(bbox, artwork) && !overlaps(bbox, std::slice::from_ref(&validated)))
    };
    // Enclosed components, kept only at glyph scale and only where no
    // existing region already owns the pixels.
    let mut ink = vec![false; w * h];
    let mut visited = vec![false; w * h];
    let mut ink_px = 0_u64;
    let (mut ink_x0, mut ink_y0, mut ink_x1, mut ink_y1) =
        (usize::MAX, usize::MAX, 0_usize, 0_usize);
    for start in 0..w * h {
        if !sealed_dark[start] || open[start] || visited[start] {
            continue;
        }
        let mut component = vec![start];
        visited[start] = true;
        let mut cursor = 0;
        while cursor < component.len() {
            let (x, y) = (component[cursor] % w, component[cursor] / w);
            cursor += 1;
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < w && ny < h {
                    let n = index(nx, ny);
                    if sealed_dark[n] && !open[n] && !visited[n] {
                        visited[n] = true;
                        component.push(n);
                    }
                }
            }
        }
        if component.len() < SCREAM_MIN_COMPONENT_PX {
            continue;
        }
        let bbox = component_bbox(&component);
        if fenced(bbox) {
            continue;
        }
        ink_px += component.len() as u64;
        for i in component {
            ink[i] = true;
        }
        ink_x0 = ink_x0.min(bbox.0);
        ink_y0 = ink_y0.min(bbox.1);
        ink_x1 = ink_x1.max(bbox.2);
        ink_y1 = ink_y1.max(bbox.3);
    }
    if (ink_px as usize) < SCREAM_MIN_TOTAL_INK_PX {
        return None;
    }
    // Cropped-glyph evidence: an OPEN component at glyph scale, outside every
    // existing region, pressed against 1-2 ADJACENT border sides where the
    // window can still expand. The panel and the face are open too but fail
    // the area ceiling; the burst outline was dropped by the sealing pass, so
    // it is no component at all; and a border the window shares with the page
    // edge has nothing beyond it to grow into.
    let expandable = [
        x0 > 0,
        y0 > 0,
        (x1 as i64) < i64::from(page_w),
        (y1 as i64) < i64::from(page_h),
    ];
    let mut candidates = [false; 4];
    let mut open_visited = vec![false; w * h];
    for start in 0..w * h {
        if !open[start] || open_visited[start] {
            continue;
        }
        let mut component = vec![start];
        open_visited[start] = true;
        let mut cursor = 0;
        while cursor < component.len() {
            let (x, y) = (component[cursor] % w, component[cursor] / w);
            cursor += 1;
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < w && ny < h {
                    let n = index(nx, ny);
                    if open[n] && !open_visited[n] {
                        open_visited[n] = true;
                        component.push(n);
                    }
                }
            }
        }
        if !(SCREAM_MIN_COMPONENT_PX..=SCREAM_MAX_GLYPH_PX).contains(&component.len()) {
            continue;
        }
        let bbox = component_bbox(&component);
        if fenced(bbox) {
            continue;
        }
        let touches = [
            bbox.0 == 0,
            bbox.1 == 0,
            bbox.2 + 1 == w,
            bbox.3 + 1 == h,
        ];
        let touched = touches.iter().filter(|&&side| side).count();
        // One side, or a corner (two adjacent); a component spanning
        // opposite borders is a runner, not a glyph.
        let corner = touched == 2
            && !((touches[0] && touches[2]) || (touches[1] && touches[3]));
        if touched == 1 || corner {
            for side in 0..4 {
                if touches[side] && expandable[side] {
                    candidates[side] = true;
                }
            }
        }
    }
    // Eroded core for every colour sample, so the anti-aliased rim never
    // reaches a median.
    let mut core = ink.clone();
    for _ in 0..SCREAM_CORE_EROSION {
        let previous = core.clone();
        for y in 0..h {
            for x in 0..w {
                let i = index(x, y);
                if previous[i] {
                    let interior = x > 0
                        && x + 1 < w
                        && y > 0
                        && y + 1 < h
                        && previous[index(x - 1, y)]
                        && previous[index(x + 1, y)]
                        && previous[index(x, y - 1)]
                        && previous[index(x, y + 1)];
                    core[i] = interior;
                }
            }
        }
    }
    if !core.iter().any(|&value| value) {
        core = ink.clone();
    }
    let sample = |select: &dyn Fn(usize, usize) -> bool| -> Option<[u8; 3]> {
        let mut channels: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for y in 0..h {
            for x in 0..w {
                if select(x, y) {
                    let pixel = image.get_pixel(x0 + x as u32, y0 + y as u32).0;
                    for (channel, value) in channels.iter_mut().zip(pixel) {
                        channel.push(value);
                    }
                }
            }
        }
        if channels[0].is_empty() {
            return None;
        }
        Some(channels.map(|mut channel| {
            let mid = channel.len() / 2;
            *channel.select_nth_unstable(mid).1
        }))
    };
    let core_rows: Vec<usize> = (0..h)
        .filter(|&y| (0..w).any(|x| core[index(x, y)]))
        .collect();
    let (row_lo, row_hi) = (*core_rows.first()?, *core_rows.last()?);
    let third = ((row_hi - row_lo) / 3).max(1);
    let head = sample(&|x, y| core[index(x, y)] && y <= row_lo + third)?;
    let tail = sample(&|x, y| core[index(x, y)] && y + third >= row_hi + 1)?;
    // Principal axis from the core's central second moments -- y-down, so a
    // positive angle is the down-right cascade (+42.7 on the test exhibit).
    let (mut sx, mut sy, mut n) = (0.0_f64, 0.0_f64, 0.0_f64);
    for y in 0..h {
        for x in 0..w {
            if core[index(x, y)] {
                sx += x as f64;
                sy += y as f64;
                n += 1.0;
            }
        }
    }
    let (mx, my) = (sx / n, sy / n);
    let (mut mu20, mut mu02, mut mu11) = (0.0_f64, 0.0_f64, 0.0_f64);
    for y in 0..h {
        for x in 0..w {
            if core[index(x, y)] {
                let (dx, dy) = (x as f64 - mx, y as f64 - my);
                mu20 += dx * dx;
                mu02 += dy * dy;
                mu11 += dx * dy;
            }
        }
    }
    let mut angle_degrees =
        (0.5 * (2.0 * mu11).atan2(mu20 - mu02)).to_degrees() as f32;
    if angle_degrees.abs() < SCREAM_ANGLE_SNAP_DEGREES
        || angle_degrees.abs() > SCREAM_ANGLE_LIMIT_DEGREES
        || !angle_degrees.is_finite()
    {
        angle_degrees = 0.0;
    }
    // Dilate for the erase, same radius rule as the hull writer, and sample
    // the halo from the bright ring the dilation swallows.
    let radius = super::dilation_radius(page_w.max(page_h)).max(0.0) as usize;
    let mut dilated = ink.clone();
    for _ in 0..radius {
        let previous = dilated.clone();
        for y in 0..h {
            for x in 0..w {
                let i = index(x, y);
                if !previous[i]
                    && ((x > 0 && previous[index(x - 1, y)])
                        || (x + 1 < w && previous[index(x + 1, y)])
                        || (y > 0 && previous[index(x, y - 1)])
                        || (y + 1 < h && previous[index(x, y + 1)]))
                {
                    dilated[i] = true;
                }
            }
        }
    }
    let halo = sample(&|x, y| {
        let i = index(x, y);
        dilated[i] && !ink[i] && bright[i]
    })
    .unwrap_or([255, 255, 255]);
    // Glow pass: expand from the dilated ink through BRIGHT pixels only, so
    // the glyphs' glow ring joins the erase and LaMa rebuilds from clean
    // paper -- while every dark pixel (outline, panel, art, the kept head
    // glyph's ink) stays a wall the expansion cannot claim or cross.
    let mut erase = dilated.clone();
    let mut ring: Vec<usize> = (0..w * h).filter(|&i| erase[i]).collect();
    for _ in 0..SCREAM_GLOW_RADIUS {
        let mut next: Vec<usize> = Vec::new();
        for &i in &ring {
            let (x, y) = (i % w, i / w);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < w && ny < h {
                    let n = index(nx, ny);
                    if bright[n] && !erase[n] {
                        erase[n] = true;
                        next.push(n);
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        ring = next;
    }
    let mut mask = GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            if erase[index(x, y)] {
                mask.get_pixel_mut(x as u32, y as u32).0[0] = u8::MAX;
            }
        }
    }
    // Paper for the flat paint: the median of the LOW-SPREAD bright pixels --
    // glow is bright but pink, and must go to the model, not the bucket.
    let paper = sample(&|x, y| {
        if !bright[index(x, y)] {
            return false;
        }
        let pixel = image.get_pixel(x0 + x as u32, y0 + y as u32).0;
        let (min, max) = pixel.iter().fold((255_i32, 0_i32), |(min, max), &value| {
            (min.min(i32::from(value)), max.max(i32::from(value)))
        });
        max - min <= SCREAM_PAPER_SATURATION_MAX
    });
    let mut flat = GrayImage::new(w as u32, h as u32);
    if let Some(paper) = paper {
        for y in 0..h {
            for x in 0..w {
                if !erase[index(x, y)] {
                    continue;
                }
                let pixel = image.get_pixel(x0 + x as u32, y0 + y as u32).0;
                if pixel
                    .iter()
                    .zip(paper)
                    .all(|(&value, target)| (i32::from(value) - i32::from(target)).abs()
                        <= SCREAM_PAPER_TOLERANCE)
                {
                    flat.get_pixel_mut(x as u32, y as u32).0[0] = u8::MAX;
                }
            }
        }
    }
    Some(ScreamGrowth {
        ink: ScreamMarkInk {
            mask,
            flat,
            paper: paper.unwrap_or([255, 255, 255]),
            origin: (x0, y0),
            head,
            tail,
            halo,
            angle_degrees,
            centroid: (f64::from(x0) + mx, f64::from(y0) + my),
            ink_px,
        },
        ink_bounds: (
            f64::from(x0) + ink_x0 as f64,
            f64::from(y0) + ink_y0 as f64,
            (ink_x1 - ink_x0 + 1) as f64,
            (ink_y1 - ink_y0 + 1) as f64,
        ),
        candidates,
    })
}

/// Grow the mint's extent to the WHOLE drawn mark: expand the analysis
/// window while cropped-glyph evidence stands at an expandable border, then
/// return `(grown, hull, ink)` -- `grown` false means the mark was already
/// whole and the hull is the caller's own bounds, byte-identical to the
/// tail-only device; `grown` true means the hull is the union of the
/// caller's bounds and the full ink extent, and the caller owes the mark a
/// RE-READ before adopting it (`scream_growth_adopted`).
///
/// The test exhibit: the spot box covered only the `아아아` tail while
/// the drawn `크아` head stood above it -- never detected (RF-DETR proposes
/// the whole mark at 0.12, sub-floor, every run) and never read. The head's
/// glyphs cross the growing window's top edge one after another, each step
/// converting the next from border-cropped evidence into enclosed ink,
/// exactly as the licensed edition's whole-mark lettering demands.
fn grow_scream_mark(
    image: &RgbImage,
    bounds: (f64, f64, f64, f64),
    existing: &[(f64, f64, f64, f64)],
    artwork: &[(f64, f64, f64, f64)],
) -> Option<(bool, (f64, f64, f64, f64), ScreamMarkInk)> {
    let (page_w, page_h) = image.dimensions();
    let mut window = bounds;
    let mut expanded = false;
    let mut growth = scream_ink_analysis(image, window, existing, artwork, bounds)?;
    for _ in 0..SCREAM_GROW_MAX_STEPS {
        let [left, top, right, bottom] = growth.candidates;
        if !(left || top || right || bottom) {
            break;
        }
        let mut next = window;
        if left {
            next.0 -= SCREAM_GROW_STEP;
            next.2 += SCREAM_GROW_STEP;
        }
        if top {
            next.1 -= SCREAM_GROW_STEP;
            next.3 += SCREAM_GROW_STEP;
        }
        if right {
            next.2 += SCREAM_GROW_STEP;
        }
        if bottom {
            next.3 += SCREAM_GROW_STEP;
        }
        let clamped_x0 = next.0.max(0.0);
        let clamped_y0 = next.1.max(0.0);
        let clamped_x1 = (next.0 + next.2).min(f64::from(page_w));
        let clamped_y1 = (next.1 + next.3).min(f64::from(page_h));
        let next = (
            clamped_x0,
            clamped_y0,
            clamped_x1 - clamped_x0,
            clamped_y1 - clamped_y0,
        );
        if (next.0 - window.0).abs() < 1.0
            && (next.1 - window.1).abs() < 1.0
            && (next.2 - window.2).abs() < 1.0
            && (next.3 - window.3).abs() < 1.0
        {
            break;
        }
        let Some(regrown) = scream_ink_analysis(image, next, existing, artwork, bounds) else {
            break;
        };
        window = next;
        growth = regrown;
        expanded = true;
    }
    if !expanded {
        return Some((false, bounds, growth.ink));
    }
    // Never shrink below what the rescue itself claimed: the hull is the
    // union of the caller's box and the full ink extent.
    let (ix, iy, iw, ih) = growth.ink_bounds;
    let hull_x0 = bounds.0.min(ix);
    let hull_y0 = bounds.1.min(iy);
    let hull_x1 = (bounds.0 + bounds.2).max(ix + iw);
    let hull_y1 = (bounds.1 + bounds.3).max(iy + ih);
    let hull = (hull_x0, hull_y0, hull_x1 - hull_x0, hull_y1 - hull_y0);
    let grown = (hull.0 - bounds.0).abs() >= 1.0
        || (hull.1 - bounds.1).abs() >= 1.0
        || (hull.2 - bounds.2).abs() >= 1.0
        || (hull.3 - bounds.3).abs() >= 1.0;
    // The hull inherits the spot admission's own area fence: the original box
    // was bounded by `SPOT_RESCUE_MAX_AREA` and a grown mint must not slip
    // past it -- this is also what keeps the re-read's crop inside the size
    // class the sweep has always been fed (the sidecar's context ceiling).
    let over_ceiling = grown
        && hull.2 * hull.3 > SPOT_RESCUE_MAX_AREA * f64::from(page_w) * f64::from(page_h);
    if !grown || over_ceiling {
        // A window that expanded and came back empty-handed (or absurd) must
        // NOT ship the grown window's ink -- its mask origin, samples and
        // glow scope all differ from the tail-only device's. Re-measure on
        // the caller's own bounds so "not grown" means exactly the shipped
        // behaviour.
        if over_ceiling {
            tracing::info!(
                hull = ?hull,
                "scream growth hit the area ceiling; the tail-only device stands"
            );
        }
        let original = scream_ink_analysis(image, bounds, existing, artwork, bounds)?;
        return Some((false, bounds, original.ink));
    }
    Some((true, hull, growth.ink))
}

/// The most glyphs a HEAD-BAND read may add -- a drawn scream's lead-in is a
/// couple of glyphs (`크아`), and a hangul-shaped hallucination that rambles
/// past this is refused rather than composed.
const SCREAM_HEAD_MAX_GLYPHS: usize = 8;

/// A band's position relative to the validated box in reading order --
/// above/left prepends, below/right appends.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ScreamBandSide {
    Before,
    After,
}

/// The dominant rectangle of the grown extent MINUS the validated box, with
/// its reading-order side. The whole grown crop mixes two glyph scales and
/// per-glyph orientations and read as Han garbage on the exhibit
/// (belt-withdrawn); the band alone is
/// the tight single-run crop the sidecar reads well. `None` when no band
/// clears `min_side` -- growth that added only slivers has nothing to read.
fn scream_head_band(
    outer: (f64, f64, f64, f64),
    inner: (f64, f64, f64, f64),
    min_side: f64,
) -> Option<((f64, f64, f64, f64), ScreamBandSide)> {
    let bands = [
        (
            (outer.0, outer.1, outer.2, inner.1 - outer.1),
            ScreamBandSide::Before,
        ),
        (
            (
                outer.0,
                inner.1 + inner.3,
                outer.2,
                (outer.1 + outer.3) - (inner.1 + inner.3),
            ),
            ScreamBandSide::After,
        ),
        (
            (outer.0, outer.1, inner.0 - outer.0, outer.3),
            ScreamBandSide::Before,
        ),
        (
            (
                inner.0 + inner.2,
                outer.1,
                (outer.0 + outer.2) - (inner.0 + inner.2),
                outer.3,
            ),
            ScreamBandSide::After,
        ),
    ];
    bands
        .into_iter()
        .filter(|((_, _, w, h), _)| w.min(*h) >= min_side)
        .max_by(|((_, _, aw, ah), _), ((_, _, bw, bh), _)| {
            (aw * ah).total_cmp(&(bw * bh))
        })
}

/// Compose the band's read with the validated one, in reading order, as ONE
/// predicate: the band must have survived the belt (the caller passes its
/// verdict), must add a plausible lead-in or tail-out (1..=`SCREAM_HEAD_MAX_
/// GLYPHS` glyphs), and the COMPOSITION must still be a scream. Containment
/// of the validated read holds by construction -- which is what retired the
/// count-only adoption gate's "가가가가 for 아아아" trap: a band that turns
/// the composition into mostly-foreign glyphs fails `scream_read` here.
fn scream_compose_band(
    band_read: &str,
    original_read: &str,
    side: ScreamBandSide,
    withdrawn: bool,
) -> Option<String> {
    if withdrawn {
        return None;
    }
    let band_glyphs = band_read.chars().filter(|c| c.is_alphanumeric()).count();
    if !(1..=SCREAM_HEAD_MAX_GLYPHS).contains(&band_glyphs) {
        return None;
    }
    let composed = match side {
        ScreamBandSide::Before => format!("{band_read}{original_read}"),
        ScreamBandSide::After => format!("{original_read}{band_read}"),
    };
    scream_read(&composed).then_some(composed)
}

/// The replacement's frame: one band of `SCREAM_BAND_FRACTION` of the hull's
/// short side, along the ink's axis through its centroid, as long as the hull
/// admits at that tilt -- so the rotated band's own hull stays inside the
/// mint's (a hull is not a box, and the inscribing is the point).
fn scream_band_cell(bounds: (f64, f64, f64, f64), mark: &ScreamMarkInk) -> [f32; 4] {
    let (hull_w, hull_h) = (bounds.2.max(1.0), bounds.3.max(1.0));
    let band = (hull_w.min(hull_h) * SCREAM_BAND_FRACTION).max(48.0);
    let (sin, cos) = f64::from(mark.angle_degrees).to_radians().sin_cos();
    let (sin, cos) = (sin.abs().max(1e-3), cos.abs().max(1e-3));
    let length = ((hull_w - band * sin) / cos)
        .min((hull_h - band * cos) / sin)
        .min(hull_w.hypot(hull_h))
        .max(band);
    // Centre on the ink, then CLAMP the centre so the rotated band's own
    // hull stays inside the mint's -- an off-centre centroid (routine once
    // the hull is union(spot box, grown ink)) must slide the band inward,
    // never letter past the hull onto un-erased art.
    let half_w = (length * cos + band * sin) * 0.5;
    let half_h = (length * sin + band * cos) * 0.5;
    let clamp_centre = |centre: f64, low: f64, high: f64, half: f64| {
        // Strictly wider than the band, with an epsilon: an exact inscribe
        // makes `low + half` and `high - half` equal up to float error, and
        // `clamp` panics on min > max.
        if high - low - 2.0 * half > 1e-6 {
            centre.clamp(low + half, high - half)
        } else {
            (low + high) * 0.5
        }
    };
    let cx = clamp_centre(mark.centroid.0, bounds.0, bounds.0 + hull_w, half_w);
    let cy = clamp_centre(mark.centroid.1, bounds.1, bounds.1 + hull_h, half_h);
    [
        (cx - length * 0.5) as f32,
        (cy - band * 0.5) as f32,
        (cx + length * 0.5) as f32,
        (cy + band * 0.5) as f32,
    ]
}

fn write_rescue_mask(
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    source: &DynamicImage,
    rescued: &[(f64, f64, f64, f64)],
    marks: &[ScreamMarkInk],
) -> Result<()> {
    let (width, height) = (source.width(), source.height());
    if width == 0 || height == 0 || (rescued.is_empty() && marks.is_empty()) {
        return Ok(());
    }
    // Same shared rule and same kept tail as `write_illegible_veto` above.
    let radius = super::dilation_radius(width.max(height)).max(0.0) as i64;
    let mut mask = image::GrayImage::new(width, height);
    let mut erased = 0_u64;
    for (min_x, min_y, max_x, max_y) in rescued {
        let x0 = (min_x.floor() as i64 - radius).clamp(0, i64::from(width));
        let y0 = (min_y.floor() as i64 - radius).clamp(0, i64::from(height));
        let x1 = (max_x.ceil() as i64 + radius).clamp(0, i64::from(width));
        let y1 = (max_y.ceil() as i64 + radius).clamp(0, i64::from(height));
        for y in y0..y1 {
            for x in x0..x1 {
                let pixel = &mut mask.get_pixel_mut(x as u32, y as u32).0[0];
                if *pixel == 0 {
                    *pixel = u8::MAX;
                    erased += 1;
                }
            }
        }
    }
    // A scream mark joins ink-scoped and already dilated -- its
    // mask is blitted where the hull writer above would have filled the box.
    for mark in marks {
        let (origin_x, origin_y) = mark.origin;
        for (x, y, value) in mark.mask.enumerate_pixels() {
            if value.0[0] == 0 {
                continue;
            }
            let (page_x, page_y) = (origin_x + x, origin_y + y);
            if page_x < width && page_y < height {
                let pixel = &mut mask.get_pixel_mut(page_x, page_y).0[0];
                if *pixel == 0 {
                    *pixel = u8::MAX;
                    erased += 1;
                }
            }
        }
    }
    if erased == 0 {
        return Ok(());
    }
    tracing::info!(
        target: "koharu_pipeline::ocr",
        rescued = rescued.len(),
        erased_px = erased,
        "recovered regions joined the erase mask"
    );
    let mut bytes = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageLuma8(mask).write_to(&mut bytes, image::ImageFormat::Png)?;
    edit.set_asset(
        page,
        &koharu_scene::AssetRole::new("text-mask-rescue")?,
        koharu_scene::AssetInput::new(
            Arc::<[u8]>::from(bytes.into_inner()),
            "image/png",
            koharu_scene::AssetMetadata {
                width: Some(width),
                height: Some(height),
                attributes: std::collections::BTreeMap::new(),
            },
        ),
    )?;
    // The flat-paint companion: an RGB paint-by-value image -- non-black
    // pixels ARE the paper colour the inpainting stage paints there instead
    // of asking the model. Written only when some mark proved paper.
    let mut flat = image::RgbImage::new(width, height);
    let mut flat_px = 0_u64;
    for mark in marks {
        let (origin_x, origin_y) = mark.origin;
        for (x, y, value) in mark.flat.enumerate_pixels() {
            if value.0[0] == 0 {
                continue;
            }
            let (page_x, page_y) = (origin_x + x, origin_y + y);
            if page_x < width && page_y < height {
                flat.put_pixel(page_x, page_y, image::Rgb(mark.paper));
                flat_px += 1;
            }
        }
    }
    if flat_px > 0 {
        let mut bytes = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(flat).write_to(&mut bytes, image::ImageFormat::Png)?;
        edit.set_asset(
            page,
            &koharu_scene::AssetRole::new("text-mask-rescue-flat")?,
            koharu_scene::AssetInput::new(
                Arc::<[u8]>::from(bytes.into_inner()),
                "image/png",
                koharu_scene::AssetMetadata {
                    width: Some(width),
                    height: Some(height),
                    attributes: std::collections::BTreeMap::new(),
                },
            ),
        )?;
    }
    Ok(())
}

/// The `rotate` slot for engines that cannot ask for a server-side rotation.
/// `None` keeps the flip on the exact Rust `rotate180` and the refine grid
/// never runs -- a plain `fn` pointer type so a call site can name `None`
/// without inventing a closure.
type RotateFn<M> = fn(&M, &DynamicImage, f64) -> Result<(String, Option<f32>)>;

async fn infer_text<M: Send + 'static>(
    model: Arc<Mutex<M>>,
    targets: Vec<OcrTarget>,
    inference: impl Fn(&M, &DynamicImage) -> Result<(String, Option<f32>)> + Send + Sync + 'static,
    rotate: Option<impl Fn(&M, &DynamicImage, f64) -> Result<(String, Option<f32>)> + Send + Sync + 'static>,
    orientation_margin: Option<f64>,
) -> Result<Vec<OcrResult>> {
    tokio::task::spawn_blocking(move || {
        let model = model
            .lock()
            .map_err(|_| anyhow!("OCR model lock is poisoned"))?;
        // Both reads go through the IDENTICAL normalise-then-strip-then-refuse
        // chain. If the turned read skipped any of it, the two strings would not be
        // comparable and the empty test in `choose_orientation` would stop catching
        // a degenerate turn -- the failure would be silent and would look like a
        // good read.
        //
        // ONE chain for every read, however it was produced -- primary, turned,
        // flipped, or the refine grid's rotated candidates. A read that skipped
        // any of it would not be comparable to its sibling and the empty tests
        // in the choosers would stop catching a degenerate result. The chain
        // itself is `clean_read`, named and shared with `upright_sweep`'s
        // `read_at` so the two sites cannot drift.
        let read = |image: &DynamicImage| -> Result<(String, Option<f32>)> {
            // The confidence rides ALONGSIDE the normalise chain, never through it:
            // the chain is `String -> String` and the number is a property of the
            // decode, not of the cleaned string.
            let (text, confidence) = inference(&model, image)?;
            Ok((clean_read(text), confidence))
        };
        targets
            .into_iter()
            .map(|target| {
                let (upright, upright_confidence) = read(&target.image)?;
                // PERTURBATION STABILITY, reported and never acted on. The grown
                // read is computed, logged beside the upright one, and then
                // DROPPED -- it does not reach `text`, the mask, the eraser or the
                // translator.
                //
                // It runs through the SAME `read` closure, so both strings pass the
                // identical normalise-strip-refuse chain and are comparable; and
                // both therefore emit the `paddleocr-vl read` debug line,
                // whose crop dimensions are what let a parser tell them apart.
                if let Some(grown) = target.grown.as_ref() {
                    let (grown_text, grown_confidence) = read(grown)?;
                    // Non-optional, for the reason the turned read's own line
                    // gives: a region carries no provenance on the wire, so
                    // without this the measurement is unfalsifiable.
                    tracing::info!(
                        region = %target.region,
                        upright = %upright,
                        upright_confidence = upright_confidence.unwrap_or(f32::NAN),
                        upright_width = target.image.width(),
                        upright_height = target.image.height(),
                        grown = %grown_text,
                        grown_confidence = grown_confidence.unwrap_or(f32::NAN),
                        grown_width = grown.width(),
                        grown_height = grown.height(),
                        agreed = upright == grown_text,
                        "perturbation re-read of a grown onomatopoeia crop"
                    );
                }
                let (text, confidence) = if target.reread_rotated && wants_rotated_reread(&upright) {
                    // `rotate270` is 270 CLOCKWISE, i.e. 90 counter-clockwise --
                    // the turn measured to read this population, not its mirror.
                    let turned = target.image.rotate270();
                    let (rotated, rotated_confidence) = read(&turned)?;
                    // Non-optional. `stages[].model` reports the CONFIGURED engine
                    // and a region carries no provenance, so without this line a
                    // turned read is invisible in `format=json` and every A/B of
                    // this flag is unfalsifiable.
                    tracing::info!(
                        region = %target.region,
                        upright = %upright,
                        upright_confidence = upright_confidence.unwrap_or(f32::NAN),
                        rotated = %rotated,
                        rotated_confidence = rotated_confidence.unwrap_or(f32::NAN),
                        margin = ?orientation_margin,
                        "second OCR read of a turned free-text column"
                    );
                    let chosen = choose_orientation(
                        upright.clone(),
                        upright_confidence,
                        rotated.clone(),
                        rotated_confidence,
                        orientation_margin,
                    );
                    // The kept read's own confidence travels with it. When the two
                    // strings are equal the pick is ambiguous and the upright score
                    // is reported -- same text, and the tie cannot matter to a
                    // caller that only ever sees the pair as one field.
                    let confidence = if chosen == upright {
                        upright_confidence
                    } else {
                        rotated_confidence
                    };
                    (chosen, confidence)
                } else if target.reread_flipped && upright_confidence.is_some() {
                    // 180 degrees is an exact pixel transpose in this crate AND in
                    // the sidecar's PIL (`Transpose.ROTATE_180` fast path), so
                    // rotating HERE is byte-identical to asking the sidecar and
                    // engine-agnostic. The `is_some` guard is the engine gate:
                    // `choose_flip` keeps upright on any missing score, so an
                    // unscored engine's second read could never ship -- skipping
                    // it entirely is the same answer without the wasted call.
                    // The flip reads the BALLOON-ISOLATED twin when one exists:
                    // the grid measured the surrounding ART, not the residual
                    // tilt, as what breaks the read -- `isolate_balloon`'s doc
                    // carries the numbers. The raw crop is the fail-open arm,
                    // and it is the exact behaviour that shipped when the flip
                    // landed.
                    let flip_source = target.flip_crop.as_ref().unwrap_or(&target.image);
                    let (flip_angle, flipped, flipped_confidence) = match &rotate {
                        // The refine grid, through the sidecar's OWN rotation --
                        // the instrument that produced the exact-match cells.
                        // Pad-then-rotate is the order every selection number
                        // was measured under, and at 180.0 PIL's fast path is
                        // the same exact transpose as `rotate180`.
                        Some(rotate_read) => {
                            let mut candidates = Vec::new();
                            for angle in FLIP_REFINE_ANGLES {
                                let (text, confidence) =
                                    rotate_read(&model, flip_source, angle)?;
                                candidates.push((angle, clean_read(text), confidence));
                            }
                            best_flip(candidates)
                        }
                        None => {
                            let (text, confidence) = read(&flip_source.rotate180())?;
                            (180.0, text, confidence)
                        }
                    };
                    // Non-optional: a region carries no provenance on the
                    // wire, so without this line the flip is invisible in
                    // `format=json` and every A/B of the flag is unfalsifiable.
                    tracing::info!(
                        region = %target.region,
                        upright = %upright,
                        upright_confidence = upright_confidence.unwrap_or(f32::NAN),
                        flipped = %flipped,
                        flipped_confidence = flipped_confidence.unwrap_or(f32::NAN),
                        margin = FLIP_CONFIDENCE_MARGIN,
                        flip_angle,
                        isolated = target.flip_crop.is_some(),
                        "second OCR read of a flipped synthesised bubble"
                    );
                    choose_flip(upright, upright_confidence, flipped, flipped_confidence)
                } else {
                    (upright, upright_confidence)
                };
                Ok(OcrResult {
                    content: target.content,
                    region: target.region,
                    geometry: target.geometry,
                    previous: target.previous,
                    text,
                    confidence,
                    minted: false,
                })
            })
            .collect()
    })
    .await
    .context("OCR task panicked")?
}

/// Non-whitespace characters below which a periodic read is a real sound effect.
///
/// **The empty band is measured, not chosen.** Over the 2,534 distinct OCR
/// sources on disk (27,100 region instances), bucketing *every* string that
/// reaches `DEGENERATE_MIN_COVER` by its core length gives 1,207 instances at
/// **2-6** characters, **zero** from 7 all the way to 511, and 10 at 512. The
/// floor sits 5.3x above the largest legitimate periodic read -- which is 6
/// characters (`ええええええええー`, at 9, covers only 0.875 and so is not
/// periodic at all) -- and 16x below the only defect.
///
/// The band is wider than the periodic population alone needs, and that is the
/// point. Sound effects are routed through OCR, so the reads at risk are
/// `onomatopoeia` regions, and the two longest in the corpus are 23 and 18 core
/// characters (`LONGEST_ONOMATOPOEIA_LENGTHS` in the tests). The floor clears
/// both, so the rule never measures a real effect rather than merely scoring it
/// low.
const DEGENERATE_MIN_CORE: usize = 32;

/// The longest repeating unit a loop is measured against.
///
/// An OCR decoder that falls into a loop repeats a token or two, not a clause:
/// the one real instance is a 2-character unit. Four is generous enough to cover
/// a kana-plus-mark unit and short enough that a whole repeated *phrase* --
/// which is language, not a decoder fault -- is never described by it.
const DEGENERATE_MAX_UNIT: usize = 4;

/// Share of positions that must match at the lag before the string is a loop.
///
/// **This is the second, independent empty band.** Above 9 core characters the
/// highest cover any legitimate source on disk reaches is 0.67 (`．．．．．．．これは`,
/// 10 chars); at 16-23 it is 0.50, at 24-31 it is 0.27, at 32-47 it is 0.21 and
/// at 48-95 it is 0.26 -- a chemistry block of structural formulas, which is
/// the most periodic long *real* read in the corpus. The defect scores 1.00.
/// So length and cover each separate the two populations on their own, and the
/// rule only fires where both agree.
const DEGENERATE_MIN_COVER: f64 = 0.90;

/// Refuse a read the recogniser looped on, and hand back an empty source.
///
/// **The defect, measured.** On one page of a Japanese test volume an
/// `onomatopoeia` region routed to PaddleOCR-VL returns `ピー` **256 times** --
/// 512 characters, 2 distinct characters, no whitespace. It is the *only* such
/// string in 27,100 region instances, and it is expensive out of all proportion:
/// against the same page read without routing it costs OCR +5,849 ms and
/// translation +14,060 ms (24,898 ms against 4,964 ms end to end). It burns most
/// of the page's reply budget, so `truncated` is set and the page's *other*
/// bubbles come back untranslated -- **all 8 truncated pages in the corpus are
/// this one page**, against 0 of the other 4,332 -- and it lettered 1,705
/// characters of `BEEP` at font size 9.0 into a 572x216 box.
///
/// **It returns an EMPTY source rather than dropping the result, and the
/// difference is the artwork.** An empty read is already `illegible_text`, so it
/// flows into the `text-mask-veto` and the pixels are taken back out of
/// the erase mask: the drawn effect stays on the page, untranslated. Dropping
/// the `OcrResult` instead would remove the region from `pending`, so
/// `write_illegible_veto` would never see it, detection's mask would erase the
/// box anyway, and the effect would be replaced by a blank fill with nothing
/// lettered into it. Empty is also not a new state on this path -- the 1210x1247
/// box in `ImplausibleRegions` above returned zero characters from the same
/// model -- so nothing downstream meets a shape it has not already handled.
///
/// Applied inside `infer_text`, so it covers the primary and the routed
/// recogniser alike: the fault is a property of the string, and nothing
/// guarantees only PaddleOCR-VL can produce one.
fn refuse_degenerate_repetition(text: String) -> String {
    if !degenerate_repetition(&text) {
        return text;
    }
    tracing::warn!(
        target: "koharu_pipeline::ocr",
        chars = text.chars().count(),
        sample = %text.chars().take(12).collect::<String>(),
        "refused a degenerate OCR read"
    );
    String::new()
}

/// Whitespace is stripped before measuring, so a loop the decoder happened to
/// space out is described the same way as one it did not.
fn degenerate_repetition(text: &str) -> bool {
    let core: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if core.len() < DEGENERATE_MIN_CORE {
        return false;
    }
    (1..=DEGENERATE_MAX_UNIT).any(|unit| {
        // `core.len() >= DEGENERATE_MIN_CORE > DEGENERATE_MAX_UNIT >= unit`, so
        // the comparable count is always positive and this never divides by zero.
        let comparable = core.len() - unit;
        let matched = core[unit..]
            .iter()
            .zip(core.iter())
            .filter(|(later, earlier)| later == earlier)
            .count();
        matched as f64 >= comparable as f64 * DEGENERATE_MIN_COVER
    })
}

/// The formulas HunyuanOCR uses to talk ABOUT a crop instead of reading it,
/// byte-identical to `birelate-server`'s `ENGINE_META_PREFIXES` (`labels.rs`).
/// The lettering side refuses these -- but lettering runs after the pipeline
/// returns, so the ERASE could not see the rule: on one test chapter BOTH arms
/// wiped 36,829 px of falling petals on one page (eleven petals at one box
/// alone) under reads of `图片中没有文字。` -- the engine SAYING no text is
/// here while the pipeline erased the artwork under the sentence anyway.
///
/// The `图中` locative is the PROMPT'S OWN wording -- `HUNYUAN_PROMPT` says
/// `请识别图中的所有文字` and never `图片` -- so a negative answer echoing the
/// question back lands on the shorter locative at least as readily as on the
/// bookish `图片中`. It did: `图中没有文字` shipped unrefused eight times
/// across the stored corpus, lettering "There is no text in the image." on
/// artwork in two test chapters. Across 29,826 stored results exactly
/// eighteen distinct reads open with `图` and all eighteen are the engine;
/// the one genuine `图` on any page is mid-compound (a word for a
/// DIAGRAM). `图中的文字` stays deliberately ABSENT: `文字` is ordinary
/// vocabulary and `图中的文字…` is exactly how that diagram line would open,
/// while `文本` has zero genuine occurrences anywhere in the corpus.
const ENGINE_META_PREFIXES: [&str; 5] =
    ["图片中没有", "图片中的文本", "图片中的文字", "图中没有", "图中的文本"];

/// Strip the engine's own conversational preamble, keeping what it quotes.
///
/// `图片中的文本内容是：X` is the engine asserting the text is `X`, so `X`
/// survives and is judged on its merits -- a SHRINK, never a drop, so no
/// information is lost. A matched prefix with no colon anywhere is a bare assertion
/// of absence (`图片中没有文字。`) and maps to empty. Prefix-exact on the
/// model's own formulas, never a `contains`, for the reason `labels.rs`
/// records: a story sentence would have to OPEN in the narrator's voice about
/// the image it is inside to be touched. Runs BEFORE `normalize_ocr_text` so a
/// quoted all-placeholder run is still seen whole by that rule. The `info`
/// line is what keeps the class countable once the string is gone from the
/// wire.
fn strip_engine_preamble(text: String) -> String {
    let trimmed = text.trim_start();
    if !ENGINE_META_PREFIXES
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
    {
        return text;
    }
    let kept = trimmed
        .find(['：', ':'])
        .map(|at| trimmed[at..].chars().skip(1).collect::<String>())
        .unwrap_or_default()
        .trim()
        .to_owned();
    tracing::info!(
        raw = %trimmed,
        kept = %kept,
        "stripped an engine preamble from a read"
    );
    kept
}

/// ONE name for the whole read-cleaning chain, so its call sites cannot drift:
/// `infer_text`'s primary/turned/flipped reads and `upright_sweep`'s `read_at`
/// each had a hand-written copy, and a rule added to one would have shipped
/// half-wired -- the failure this file already warns about twice. The
/// preamble strip sits FIRST (see `strip_engine_preamble`); `strip_latex` sits
/// before `refuse_degenerate_repetition` for the reason the `infer_text`
/// comment gives -- markup is not part of what the decoder looped on.
fn clean_read(text: String) -> String {
    refuse_degenerate_repetition(strip_latex(normalize_ocr_text(strip_engine_preamble(text))))
}

// Manga OCR can emit replacement-box glyphs for an isolated Japanese ellipsis.
// Normalize only an all-placeholder sequence so ordinary OCR output is preserved.
fn normalize_ocr_text(text: String) -> String {
    let visible = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<Vec<_>>();
    if visible.len() >= 2
        && visible
            .iter()
            .all(|character| matches!(character, '☐' | '□' | '▢' | '▣' | '�'))
    {
        "…".to_owned()
    } else {
        text
    }
}

/// LaTeX control sequences worth a character rather than a deletion.
///
/// Deliberately short. The general rule below handles an *unknown* command by
/// dropping the command and keeping its braced argument, so this table only needs
/// the commands whose whole meaning IS a character and which would otherwise
/// vanish. `\times` is the one the corpus actually contains; the rest are its
/// immediate neighbours in the same arithmetic, and each is unambiguous.
const LATEX_SYMBOLS: &[(&str, char)] = &[
    ("times", '×'),
    ("div", '÷'),
    ("pm", '±'),
    ("cdot", '·'),
    ("leq", '≤'),
    ("geq", '≥'),
    ("neq", '≠'),
    ("approx", '≈'),
    ("infty", '∞'),
    ("ldots", '…'),
    ("dots", '…'),
    ("cdots", '…'),
];

/// Does this read carry a LaTeX **wrapper**, as opposed to a stray backslash?
///
/// **This is the whole safety argument, so it is deliberately narrow.** The gate
/// is not "contains a backslash" — a read of a Windows path (`C:\Users\...`) or a
/// drawn `\(^o^)/` emoticon has one, and stripping either would be text taken off
/// the page. It is not "contains a brace" either; braces occur in ordinary
/// punctuation.
///
/// It is: an inline-math opener (`\(` or `\[`), an environment (`\begin{`), or a
/// pair of `$` **with a control sequence between them**. That last clause matters:
/// `$100 and $200` is a paired `$` and must not qualify, and it is exactly the
/// shape a price on a shop sign would take.
fn latex_wrapped(text: &str) -> bool {
    // BOTH halves of the pair, and that is what excludes the emoticon `\(^o^)/` --
    // which opens with a literal `\(` and closes with a bare `)`. An opener-only
    // test admits it, and the first draft of this function did: the damage-set
    // test went red on `\(^o^)/ -> (o)/`, three characters of a drawn face gone.
    if (text.contains("\\(") && text.contains("\\)"))
        || (text.contains("\\[") && text.contains("\\]"))
        || (text.contains("\\begin{") && text.contains("\\end{"))
    {
        return true;
    }
    let mut dollars = text.match_indices('$');
    let Some((open, _)) = dollars.next() else {
        return false;
    };
    dollars.next().is_some_and(|(close, _)| {
        text.get(open..close)
            .is_some_and(|inner| inner.contains('\\'))
    })
}

/// Strip LaTeX markup an OCR engine emitted, keeping every character of content.
///
/// **The defect.** `paddleocr-vl-1.6` is a *document* VLM, not a manga OCR, and on
/// anything it reads as mathematics or as formatted text it emits LaTeX. Nothing
/// between the engine and the letterer removed it, so the markup was translated
/// and drawn on the artwork. Measured over every run JSON on disk: **6 regions
/// carry it in the OCR source, 3 distinct strings, over 6 run files, and 4 of them
/// reach the page** — small, real, and the kind of defect a reader notices at once,
/// because it is Latin punctuation soup in the middle of a drawn panel.
///
///     page A   \(\underline{\text{the}}\)   -> lettered VERBATIM
///     page B   a credit line with \(^{④}\)
///     page C   a blackboard of times tables, \(\begin{array}{l} ...
///
/// **Why here and not in the translator prompt.** The page-A pair is the
/// measurement that settles it: the same source string appears in both arms of one
/// A/B, and the OFF arm's translator happened to clean it to `the` while the ON arm
/// passed it through untouched — **the same input, two answers, in one run**, and
/// it reproduces across two independent runs.
/// On page C the translator did not even try to strip: it **rewrote the
/// delimiters**, `\(...\)` in and `$...$` out. That layer already attempts this and
/// fails non-deterministically.
///
/// **Why STRIP and not REFUSE.** A refusal in `labels.rs` would treat the read as
/// "not dialogue" and drop it — but page A's markup wraps the real word *the* and
/// page B's wraps *④*. Stripping preserves them; refusing deletes them, and not
/// losing information is essential. The markup is an
/// artefact of the ENGINE, not a property of the page, which is what makes the OCR
/// stage its home.
///
/// **Why inside `infer_text`.** Same argument `refuse_degenerate_repetition`
/// makes: the fault is a property of the string and nothing guarantees only
/// PaddleOCR-VL can produce one, so it covers the primary and the routed
/// recogniser alike — and both the upright and the turned read go through the
/// identical chain, which is what keeps the two comparable in `choose_orientation`.
///
/// **An all-markup read comes back EMPTY, not as its original.** Empty is already
/// `illegible_text`, so it flows into the text-mask veto and the drawn glyphs stay
/// on the page untranslated. Handing back the original would letter the markup,
/// which is the defect. This is the same disposal `refuse_degenerate_repetition`
/// argues for, and for the same reason.
fn strip_latex(text: String) -> String {
    if !latex_wrapped(&text) {
        return text;
    }
    let stripped = unwrap_latex(&text);
    if stripped.trim().is_empty() {
        tracing::warn!(
            target: "koharu_pipeline::ocr",
            sample = %text.chars().take(24).collect::<String>(),
            "OCR read was LaTeX markup with no content"
        );
        return String::new();
    }
    if stripped != text {
        tracing::info!(
            target: "koharu_pipeline::ocr",
            before = %text,
            after = %stripped,
            "stripped LaTeX markup from an OCR read"
        );
    }
    stripped
}

/// The scanner. Structural rules only, so a command nobody listed still degrades
/// to its content rather than to markup on the page.
fn unwrap_latex(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            // `\\` is a row separator, and the rows of the one real example are
            // separate sums -- joining them without a gap would read as one number.
            '\\' if i + 1 < chars.len() && chars[i + 1] == '\\' => {
                out.push(' ');
                i += 2;
            }
            '\\' if i + 1 < chars.len() && chars[i + 1].is_ascii_alphabetic() => {
                let start = i + 1;
                let mut end = start;
                while end < chars.len() && chars[end].is_ascii_alphabetic() {
                    end += 1;
                }
                let name: String = chars[start..end].iter().collect();
                i = end;
                if name == "begin" || name == "end" {
                    // The braced argument is the ENVIRONMENT NAME, not content, so
                    // it goes -- unlike every other command, where it is the
                    // content and stays.
                    i = skip_braced_group(&chars, i);
                    // `\begin{array}{l}` carries a column spec too. Dropped only
                    // when it really looks like one, so a second group that is
                    // content is never eaten.
                    if let Some(next) = braced_group(&chars, i)
                        && next.1.len() <= 4
                        && next.1.iter().all(|c| matches!(c, 'l' | 'c' | 'r' | '|'))
                    {
                        i = next.0;
                    }
                } else if let Some((_, symbol)) =
                    LATEX_SYMBOLS.iter().find(|(command, _)| *command == name)
                {
                    out.push(*symbol);
                }
                // Anything else: the command is markup and disappears, while a
                // braced group after it is content and is unwrapped by the plain
                // brace arms below. That is what turns `\underline{\text{the}}`
                // into `the` without either command being named here.
            }
            // The inline-math delimiters themselves. These are markup, unlike the
            // escaped literals below -- `\(` is an opener, `\%` is a percent sign.
            '\\' if i + 1 < chars.len() && matches!(chars[i + 1], '(' | ')' | '[' | ']') => {
                i += 2;
            }
            // A backslash before a non-letter is an ESCAPED LITERAL -- `\%`, `\$`,
            // `\{`. The character is real; only the escape is markup.
            '\\' if i + 1 < chars.len() => {
                out.push(chars[i + 1]);
                i += 2;
            }
            // A trailing lone backslash. Nothing to escape, so nothing to keep.
            '\\' => i += 1,
            // Superscript and subscript markers. `\(^{④}\)` is a ④ with a hat on
            // it; the ④ is the content and the hat is not.
            '^' | '_' => i += 1,
            '{' | '}' | '$' => i += 1,
            character => {
                out.push(character);
                i += 1;
            }
        }
    }
    collapse_whitespace(&out)
}

/// The extent of a braced group starting at `i`, and its contents.
fn braced_group(chars: &[char], i: usize) -> Option<(usize, &[char])> {
    if chars.get(i) != Some(&'{') {
        return None;
    }
    let mut depth = 0;
    for (offset, character) in chars[i..].iter().enumerate() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((i + offset + 1, &chars[i + 1..i + offset]));
                }
            }
            _ => {}
        }
    }
    None
}

fn skip_braced_group(chars: &[char], i: usize) -> usize {
    braced_group(chars, i).map_or(i, |(end, _)| end)
}

/// Stripping leaves the gaps the markup used to occupy. A page reads better
/// without them, and a translator prompt is shorter.
fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for character in text.chars() {
        if character.is_whitespace() {
            space = !out.is_empty();
        } else {
            if space {
                out.push(' ');
            }
            space = false;
            out.push(character);
        }
    }
    out
}

fn crop(source: &DynamicImage, geometry: &Geometry) -> Result<DynamicImage> {
    crop_grown(source, geometry, 0)
}

/// The whole decision the OCR stage makes about a perturbation re-read: flag on,
/// region is an onomatopoeia, **and the growth actually took**.
///
/// **Extracted so the composed predicate is what a test can call.** Testing
/// `crop_grown` and the label test separately would leave the `&&` between them
/// unexercised, which is how a fix once shipped entirely unwired with 128
/// tests green. The caller does exactly one thing with this: puts the result in
/// `OcrTarget::grown`.
///
/// `None` means no second read, and the three reasons are deliberately
/// indistinguishable to the caller because it treats them identically. The one
/// that matters for correctness is the third: a box the clamp refused to grow --
/// already flush against a page edge -- comes back at its original size, and an
/// unperturbed pair ALWAYS agrees. Scoring that as "stable" would be a false
/// negative pointing the wrong way, so it is dropped instead.
fn grown_twin(
    source: &DynamicImage,
    geometry: &Geometry,
    base: &DynamicImage,
    grow_px: Option<NonZeroU32>,
    is_onomatopoeia: bool,
) -> Result<Option<DynamicImage>> {
    let Some(grow) = grow_px.filter(|_| is_onomatopoeia) else {
        return Ok(None);
    };
    let grown = crop_grown(source, geometry, grow.get())?;
    if grown.width() == base.width() && grown.height() == base.height() {
        return Ok(None);
    }
    Ok(Some(grown))
}

/// The same crop, with the box pushed out by `grow_px` on every side first.
///
/// **GROW ONLY, never shrink**, and the asymmetry is the whole point. Growing adds
/// background and cannot clip a stroke; shrinking clips glyphs and would make real
/// text look unstable by construction -- manufacturing the very signal the caller
/// is trying to measure.
///
/// **The clamp is what makes this fallible in a way the caller MUST check.** Lower
/// edges are pushed out and then floored at zero; upper edges are pushed out and
/// then capped at the image. A box already touching an edge -- and three of the
/// four measured ones overrun their own slice -- therefore comes back at or near
/// its original size however much growth was asked for. A caller that assumes the
/// growth happened reads "the crop did not change" as "the read is stable", which
/// is a false negative pointing the wrong way. **Compare the returned dimensions
/// against the ungrown crop and discard the pair when they match.**
fn crop_grown(source: &DynamicImage, geometry: &Geometry, grow_px: u32) -> Result<DynamicImage> {
    let (min_x, min_y, max_x, max_y) =
        geometry_extents(geometry).ok_or_else(|| anyhow!("geometry is empty"))?;
    let grow = f64::from(grow_px);
    // Subtract in f64 BEFORE the cast. Growing `x` by subtracting from the u32
    // would underflow to ~4 billion on any box within `grow_px` of the left edge,
    // and `crop_imm` would return an empty image rather than fail.
    let x = (min_x - grow).floor().max(0.0) as u32;
    let y = (min_y - grow).floor().max(0.0) as u32;
    let right = (max_x + grow)
        .ceil()
        .max(0.0)
        .min(f64::from(source.width())) as u32;
    let bottom = (max_y + grow)
        .ceil()
        .max(0.0)
        .min(f64::from(source.height())) as u32;
    if right <= x || bottom <= y {
        bail!("geometry does not overlap the image");
    }
    Ok(source.crop_imm(x, y, right - x, bottom - y))
}

#[cfg(test)]
mod tests {
    /// Adapts a test closure that only produces TEXT to the shape `infer_text` now
    /// takes. Every one of these fixtures stands in for an engine with no
    /// confidence -- which is not a shortcut: `manga-ocr` and `baberu-ocr` really do
    /// report `None`, so this is the majority arm in production too.
    fn text_only<M>(
        inference: impl Fn(&M, &DynamicImage) -> anyhow::Result<String> + Send + Sync + 'static,
    ) -> impl Fn(&M, &DynamicImage) -> anyhow::Result<(String, Option<f32>)> + Send + Sync + 'static
    {
        move |model, image| Ok((inference(model, image)?, None))
    }

    use super::NonZeroU32;
    use super::{
        DEGENERATE_MIN_CORE, OcrResult, OcrTarget, SPOT_COVERAGE_FLOOR, choose_orientation, crop, crop_grown,
        clean_read, degenerate_repetition, grown_twin, illegible_text,
        RotateFn, SYNTHESISED_REGION_KIND, UprightRow, best_flip, choose_flip,
        denormalize_spot_box, flip_candidate, isolate_balloon,
        illegible_read, infer_text, latex_wrapped, latin_on_cjk, normalize_ocr_text, oversized,
        parse_spot_boxes,
        mint_spans_a_cut, spot_ink_fraction, spot_rescue_candidate,
        refuse_degenerate_repetition, refuse_region, rescue_narrow, rotation_candidate, wire_confidence,
        script_mismatch, spot_candidate, spot_coverage, spot_eligible, spot_refuses, spot_suspect,
        strip_latex, upright_norm,
        MisreadLevers, SPOT_MAX_PIXELS, spot_scale_dimensions, stamped_language, upright_select,
        wants_rotated_reread, withdraw_from_mask,
        admit_hangul_reread, reread_candidate,
        ScreamBandSide, ScreamMarkInk, grow_scream_mark, scream_band_cell, scream_compose_band,
        scream_head_band, scream_mark, scream_mark_ink, scream_read,
    };
    use image::{DynamicImage, GrayImage, RgbImage};
    use koharu_scene::{
        DetectionAnalysis, EntityId, Geometry, LanguageTag, Origin, RegionSpec, TextRegion,
    };
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    // Reached through `crate::stages` rather than `super`, because the OCR router
    // now calls the composed `refuse_region` and imports neither of these -- the
    // tests below still pin the AREA arm and the two-arm predicate on their own,
    // beside the composed one the caller actually uses.
    // `watermark_text` joins these for the same reason they are here: the OCR
    // router now calls the composed `only_site_furniture` and no longer imports
    // the bare rule, but the tests below still pin it on its own beside the
    // composed one the caller actually uses.
    use crate::stages::{
        implausible_region, only_site_furniture, refuse_before_reading, site_address, story_text,
        watermark_text,
    };
    use koharu_translator::Language;

    /// The defect, verbatim: `ピー` 256 times, as PaddleOCR-VL returned it for the
    /// `onomatopoeia` region on one test page.
    fn the_beep_loop() -> String {
        "ピー".repeat(256)
    }

    /// **Every** distinct periodic source in the corpus whose core is under the
    /// floor, with its instance count. This is the population the rule exists to
    /// leave alone, and it is all 1,207 of the 1,217 periodic instances on disk
    /// -- the remaining 10 are the loop.
    ///
    /// The list is exhaustive on purpose, and it was not: it carried 24 of these
    /// 51 sources and claimed the 1,207 anyway, so the aggregate overstated its
    /// own table by 25%. Re-derived over the stored run results -- 4,342 JSON
    /// replies, 28,594 region rows of which 1,494 have an empty source, leaving
    /// **27,100** instances over **2,534** distinct non-empty sources, which is
    /// where `refuse_degenerate_repetition`'s two headline numbers come from.
    /// Bucketing every source that reaches `DEGENERATE_MIN_COVER` by core
    /// length: 365 at 2, 584 at 3, 177 at 4, 29 at 5, 52 at 6, **nothing at all
    /// from 7 to 511**, and 10 at 512. The 25 entries added here are mostly
    /// digits, Latin fragments and Chinese reduplication -- unglamorous, but
    /// they are two-fifths of the distinct population the negative direction is
    /// supposed to cover. Two dialogue rows (`空だぞ空`, `川よ川`) are invented
    /// stand-ins with the measured reads' length and repeat structure.
    const REAL_SHORT_PERIODIC: &[(&str, u32)] = &[
        ("．．．", 447),
        ("！！", 144),
        ("．．．．．．", 52),
        ("～～", 33),
        ("ええ", 31),
        ("いい", 28),
        ("ポワポワ", 26),
        ("！！！", 26),
        ("・ｒ・", 21),
        ("．．", 18),
        ("that", 17),
        ("ハハハ", 15),
        ("ワイワイ", 15),
        ("!!", 15),
        ("ゴツゴツ", 14),
        ("まぁまぁ", 14),
        ("あらあら", 14),
        ("わっ、わっ", 13),
        ("もっとも", 13),
        ("５５", 13),
        ("４４", 13),
        ("ノノ", 13),
        ("哈哈", 12),
        ("スース", 11),
        ("ゴリゴ", 11),
        ("◇◇◇", 11),
        ("ババ", 11),
        ("ああ", 11),
        ("空だぞ空", 10),
        ("いやいや", 10),
        ("川よ川", 10),
        ("√2√", 10),
        ("はは", 9),
        ("なるほどな", 8),
        ("やってやっ", 8),
        ("モグモグ", 8),
        ("バカバカ", 8),
        ("やれやれ", 8),
        ("しかし", 8),
        ("～～～～", 7),
        ("附附附附", 6),
        ("许许许", 6),
        ("——", 6),
        ("TNT", 5),
        ("谢谢", 4),
        ("“”、“", 3),
        ("0070", 3),
        ("000", 3),
        ("00", 3),
        ("い、いい", 1),
        ("／／", 1),
    ];

    /// The closest thing on disk to a legitimate refusal, and it is **not** in
    /// the table above because it is not periodic by the rule: `ええええええええー`
    /// is 8 kana and a trailing bar, so unit 1 matches 7 of 8 comparable
    /// positions for a cover of **0.875** -- the highest any real sub-floor read
    /// reaches, and still under the 0.90 bar. It sat in `REAL_SHORT_PERIODIC`
    /// with 7 instances beside 24 genuinely periodic sources, which is where 7
    /// of the 242 phantom instances in the old aggregate came from.
    const THE_CLOSEST_NEAR_MISS: (&str, u32) = ("ええええええええー", 7);

    /// The core lengths of the two longest `onomatopoeia` reads in the whole
    /// corpus, with the number of rows carrying that label and the highest
    /// cover either reaches at any unit in `1..=DEGENERATE_MAX_UNIT`. The
    /// Japanese row is an invented stand-in with the measured read's core
    /// length; its cover is its own.
    ///
    /// **This is the population `DEGENERATE_MIN_CORE` is actually placed
    /// against.** Sound effects are sent through OCR, so an
    /// `onomatopoeia` region is the only thing the rule can plausibly take by
    /// mistake, and these two are its ceiling at 23 and 18 core characters.
    /// Everything in `REAL_SHORT_PERIODIC` is 6 characters or fewer, so the
    /// negative table on its own leaves every floor from 7 upward unpinned.
    const LONGEST_ONOMATOPOEIA_LENGTHS: &[(&str, usize, u32, f64)] = &[
        ("http://www.example.com/", 23, 6, 0.182),
        ("そのボタンだけは絶対に押さないで！！", 18, 7, 0.059),
    ];

    #[test]
    fn repeated_placeholder_glyphs_are_an_ellipsis() {
        assert_eq!(normalize_ocr_text("☐ ☐ ☐".to_owned()), "…");
        assert_eq!(normalize_ocr_text("□\n□".to_owned()), "…");
    }

    #[test]
    fn ordinary_text_and_single_boxes_are_unchanged() {
        assert_eq!(normalize_ocr_text("待って…".to_owned()), "待って…");
        assert_eq!(normalize_ocr_text("☐".to_owned()), "☐");
    }

    /// Unset means the routing is off entirely, which is the shipped default and
    /// the only state in which no second model can ever be loaded.
    #[test]
    fn no_threshold_routes_nothing() {
        assert!(!oversized(4000, 4000, None));
    }

    /// The three real oversized regions found on 40 pages, at the 448 default:
    /// one wide caption that manga-ocr fabricated, one wide balloon it read
    /// correctly, and one tall vertical column. All three route -- the false
    /// positives cost time, never accuracy, which is what lets the rule be this
    /// blunt.
    #[test]
    fn the_larger_side_decides_so_a_vertical_column_routes_too() {
        assert!(oversized(465, 198, Some(448)), "wide caption");
        assert!(oversized(502, 332, Some(448)), "wide balloon");
        assert!(oversized(59, 497, Some(448)), "vertical column");
    }

    /// The median real region is 120px and p90 is 229px, so ordinary bubble text
    /// must never route. `narrow-32` from the dense fixture is the closest real
    /// miss at 418px, and manga-ocr reads it perfectly.
    #[test]
    fn ordinary_bubbles_stay_on_the_primary() {
        assert!(!oversized(120, 90, Some(448)));
        assert!(!oversized(229, 140, Some(448)));
        assert!(!oversized(418, 60, Some(448)), "narrow-32");
    }

    /// Boundary, spelled out because `>=` versus `>` decides whether a threshold
    /// equal to the encoder input means "never downscaled" or "always".
    #[test]
    fn the_threshold_is_inclusive() {
        assert!(oversized(448, 10, Some(448)));
        assert!(!oversized(447, 10, Some(448)));
    }

    /// **The exact geometry that latched the run**, and the pair that has to
    /// stay on opposite sides of the bound. The 1210x1247 `onomatopoeia` box on
    /// one test page covers 1.49x its own 844x1200 page; it cleared the 448 lower
    /// bound, went to PaddleOCR-VL, returned zero characters, and left the
    /// caching allocator ~3.7 GB higher for the remaining 26 pages. A 500x300
    /// region on the same page is 0.15x and must still route -- that is the
    /// population the oversized-crop routing was measured on, where the largest
    /// real region was 502x332.
    #[test]
    fn a_region_larger_than_its_page_is_refused_while_an_ordinary_large_one_still_routes() {
        let page = (844, 1200);
        assert!(
            implausible_region((1210.0, 1247.0), page, Some(0.5)),
            "the onomatopoeia box, 1.49x the page"
        );
        assert!(
            !implausible_region((500.0, 300.0), page, Some(0.5)),
            "0.15x the page"
        );
        assert!(
            oversized(500, 300, Some(448)),
            "and it is still over the lower bound, so it routes"
        );
    }

    /// Unset is the state in which this file behaves exactly as it did before
    /// the ceiling existed, which is the arm the A/B needs. A non-positive or
    /// NaN fraction reads the same way rather than being clamped.
    #[test]
    fn no_upper_bound_refuses_nothing() {
        let page = (844, 1200);
        assert!(!implausible_region((4000.0, 4000.0), page, None));
        assert!(!implausible_region((4000.0, 4000.0), page, Some(0.0)));
        assert!(!implausible_region((4000.0, 4000.0), page, Some(-1.0)));
        assert!(!implausible_region((4000.0, 4000.0), page, Some(f32::NAN)));
    }

    /// The gap the default sits in. Of the 24 routed regions in the run that
    /// latched, the largest legitimate one is 0.30x of the page and the outlier
    /// is 1.49x; 0.5 is 1.67x above the first and a third of the second, with
    /// nothing observed in between.
    #[test]
    fn the_default_clears_the_largest_real_region_by_a_wide_margin() {
        let page = (844, 1200);
        // 0.30x: the largest region in the run that was not the detector error.
        assert!(!implausible_region((712.0, 427.0), page, Some(0.5)));
        // 0.16x: the largest of the three the routing was measured on, 502x332.
        assert!(!implausible_region((502.0, 332.0), page, Some(0.5)));
    }

    /// A fraction, not a pixel count, because the two pages this project
    /// measures on are different sizes. 650x210 is the dense fixture's wide box
    /// -- the case the oversized-crop routing exists for -- and it must keep routing on its own
    /// 1492x1118 page, where it is 0.08x. The same box is 0.13x of a manga page;
    /// an absolute threshold would have to be two different numbers.
    #[test]
    fn the_bound_is_a_fraction_so_it_transfers_between_page_sizes() {
        assert!(!implausible_region((650.0, 210.0), (1492, 1118), Some(0.5)));
        assert!(!implausible_region((650.0, 210.0), (844, 1200), Some(0.5)));
        assert!(oversized(650, 210, Some(448)), "so both still route");
    }

    /// Inclusive, and pinned because the flag's most defensible value is the
    /// impossibility line itself: at 1.0 the rule refuses only a box that is at
    /// least the whole page, which is a statement no page can contradict.
    #[test]
    fn the_area_bound_is_inclusive() {
        assert!(implausible_region((844.0, 1200.0), (844, 1200), Some(1.0)));
        assert!(!implausible_region((844.0, 1199.0), (844, 1200), Some(1.0)));
    }

    /// A degenerate page cannot divide, and must not refuse everything by
    /// accident -- `mask_geometry` can hand back a zero-extent box, and a
    /// zero-area page would make every comparison `>= 0` and true.
    #[test]
    fn a_degenerate_page_or_region_refuses_nothing() {
        assert!(!implausible_region((1210.0, 1247.0), (0, 1200), Some(0.5)));
        assert!(!implausible_region((1210.0, 1247.0), (844, 0), Some(0.5)));
        assert!(!implausible_region((0.0, 1247.0), (844, 1200), Some(0.5)));
        assert!(!implausible_region((-10.0, -10.0), (844, 1200), Some(0.5)));
    }

    // ---- cross-slice fragments ------------------------------------------------

    /// The four regions of a test chapter that reach both cuts, with the slice
    /// each was measured on. Three are skill names whose uncut controls are in
    /// the same chapter, and all three were lettered as confidently wrong
    /// English (the labels below are invented stand-ins for the reads).
    ///
    /// **Asserted on `refuse_before_reading`, not on `cross_slice_fragment`.** Both
    /// arms of that `||` have their own tests below and above, and this project has
    /// already shipped a fix whose `||` was deleted with all 128 tests still green.
    /// This is the predicate the OCR router and the erase mask actually call.
    #[test]
    fn a_region_reaching_both_cuts_is_refused_before_any_engine_reads_it() {
        for (label, region, page) in [
            ("A 冰真诀 -> Bing Zhenjue", (124.0, 906.0), (1200u32, 919u32)),
            ("B 迎接之王 -> King of Welcomes", (124.0, 905.0), (1200, 919)),
            ("C 1号云守！ -> No. 1: Kumomori!", (197.0, 901.0), (1200, 908)),
            ("D narration, cut mid-word", (56.0, 933.0), (1200, 935)),
        ] {
            assert!(
                refuse_before_reading(region, page, Some(0.5)),
                "{label} reaches both cuts and must not be read"
            );
        }
    }

    /// A page the caller ASSEMBLED keeps the AREA arm and loses the FRAGMENT arm.
    ///
    /// **Asserted on `refuse_region`, the predicate the OCR router actually
    /// calls**, and not on the two branches separately -- for the same reason
    /// `refuse_before_reading` is asserted on rather than its own two arms. A
    /// test of the branches leaves the choice between them unasserted, and the
    /// failure is silent: every seam stays refused and the suite stays green.
    #[test]
    fn a_joined_page_may_hold_text_that_reaches_both_of_its_edges() {
        // Slice C joined to the next slice: one column, 1693px of a 1716px seam.
        // 0.987 of the page, past the 0.95 the fragment arm refuses at.
        let joined_column = (204.0, 1693.0);
        let seam = (1200u32, 1716u32);

        // On an ordinary slice this is a fragment of something taller, and
        // reading it is how `1号云守！` became `No. 1: Kumomori!`.
        assert!(
            refuse_region(joined_column, seam, Some(0.5), false),
            "an ordinary page must still refuse a region reaching both edges"
        );
        // On a page assembled to hold exactly that column, it is the whole name.
        assert!(
            !refuse_region(joined_column, seam, Some(0.5), true),
            "a joined page must read the very region the join went to fetch"
        );
    }

    /// A box larger than its own page is a detector error whether the caller
    /// assembled the page or not, so the IMPOSSIBILITY survives everywhere --
    /// and the configured plausibility bound survives on a joined page too.
    /// Raising the bound on joined pages was measured at zero wins and three
    /// regressions once the upright pass existed.
    #[test]
    fn a_joined_page_still_refuses_a_box_larger_than_itself() {
        // 1.18x of the page: an impossibility, refused on ANY page.
        let impossible = (1300.0, 1000.0);
        assert!(refuse_region(impossible, (1200, 917), Some(0.5), true));
        assert!(refuse_region(impossible, (1200, 917), Some(0.5), false));

        // 0.68 of the page: past the configured bound on EVERY kind of page.
        // On a joined page that refusal is not a discard -- it is the handoff
        // to the upright pass, which reads what the ordinary path fabricates.
        let large_but_possible = (920.0, 813.0);
        assert!(refuse_region(large_but_possible, (1200, 917), Some(0.5), false));
        assert!(
            refuse_region(large_but_possible, (1200, 917), Some(0.5), true),
            "the configured area bound must hold on a joined page too"
        );

        // With no ceiling configured a joined page refuses nothing, unchanged.
        assert!(!refuse_region(large_but_possible, (1200, 917), None, true));

        // A ceiling the caller set higher than the box is kept.
        assert!(
            !refuse_region(impossible, (1200, 917), Some(2.0), true),
            "an explicit higher ceiling is the caller's and stands"
        );
    }

    /// The balloon a join went to fetch is REFUSED into the upright pass, and
    /// that is how it gets read correctly.
    ///
    /// Measured on a seam composite: refused -> stashed -> 16-angle sweep ->
    /// a three-glyph exclamation (stand-in `快跑！`) at 0.9831 -> "Run!", the
    /// correct read; admitted instead ->
    /// ordinary single-orientation read -> a wrong two-glyph word at 0.5417 ->
    /// a mistranslation lettered into the balloon. Reader and mask still
    /// move together: `detection.rs`'s erase-mask call passes the same
    /// `joined_page`, and the balloon's flat-fill erase is identical either way.
    #[test]
    fn the_balloon_a_join_went_to_fetch_is_refused_into_the_upright_pass() {
        let balloon = (766.0, 927.0);
        let composite = (1200u32, 908u32);
        assert!(
            refuse_region(balloon, composite, Some(0.5), false),
            "on an ordinary slice this is still a fragment and still refused"
        );
        assert!(
            refuse_region(balloon, composite, Some(0.5), true),
            "on the joined page the area bound hands it to the upright pass"
        );
    }

    /// The full census of one test chapter's 26 composites: every region at or
    /// above either ceiling, plus the two wins the joined flag genuinely buys.
    /// The three at/over 0.5 area are refused into the upright pass (which
    /// reads the balloon and the SFX correctly and declines the drawn phrase,
    /// sparing 801,784 px of drawn art); the two columns and the narration keep
    /// their ordinary reads; the impossibility is refused on any page. A minted
    /// column is absent deliberately: the mint path never calls
    /// `refuse_region` at all.
    #[test]
    fn the_censused_composites_split_exactly_at_the_area_bound() {
        for (label, region, page, refused) in [
            ("balloon", (766.1007433263608, 926.5131228363222), (1200u32, 908u32), true),
            ("drawn phrase", (698.2026599027422, 1109.1809870159523), (1200, 1088), true),
            ("SFX", (1099.8262990712296, 1394.4960303031858), (1200, 1384), true),
            ("impossibility", (1219.0, 1747.0), (1200, 1386), true),
            ("first column", (199.9, 903.9), (1200, 908), false),
            ("second column", (203.0, 906.0), (1200, 908), false),
            ("narration", (56.25, 1211.0), (1200, 1211), false),
        ] {
            assert_eq!(
                refuse_region(region, page, Some(0.5), true),
                refused,
                "{label}: the joined page's verdict moved"
            );
        }
    }

    /// The ordinary page is untouched by any of this. Pinned against the same
    /// four regions the fragment arm was built on, through the new predicate,
    /// so the guard cannot be relaxed for everyone by accident.
    #[test]
    fn an_ordinary_page_refuses_exactly_what_it_did_before() {
        for (label, region, page) in [
            ("A", (124.0, 906.0), (1200u32, 919u32)),
            ("B", (124.0, 905.0), (1200, 919)),
            ("C", (197.0, 901.0), (1200, 908)),
            ("D", (56.0, 933.0), (1200, 935)),
        ] {
            assert!(
                refuse_region(region, page, Some(0.5), false),
                "{label} must still be refused on an ordinary slice"
            );
            assert_eq!(
                refuse_region(region, page, Some(0.5), false),
                refuse_before_reading(region, page, Some(0.5)),
                "{label}: the ordinary arm must BE `refuse_before_reading`"
            );
        }
    }

    /// The floor of the measured plateau. Every threshold from 0.90 to 0.98 picks
    /// the same four regions on the test chapter, and the tallest region that is NOT a
    /// fragment sits at 0.886 of its page -- so the rule must keep taking it.
    #[test]
    fn the_tallest_region_that_is_not_a_fragment_is_still_read() {
        // A narrow column at 0.886 of its page -- the plateau's floor. Deliberately
        // NOT the 920x813 region, which reads as the same height share but is
        // refused by the AREA arm at 0.68 of the page, so it would have proved
        // nothing about this one. The first draft of this test used it and passed
        // for the wrong reason.
        assert!(!refuse_before_reading((124.0, 813.0), (1200, 917), Some(0.5)));
        // A tall balloon that is comfortably inside its slice.
        assert!(!refuse_before_reading((300.0, 700.0), (1200, 1000), Some(0.5)));
    }

    /// A height share is not scale-free, and the erase-mask fixtures next door are
    /// one pixel tall on purpose. Without the floor this rule refuses every
    /// detection in every one of them.
    #[test]
    fn a_page_too_short_for_a_cut_line_of_text_has_no_fragments() {
        assert!(!refuse_before_reading((4.0, 1.0), (4, 1), None));
        assert!(!refuse_before_reading((10.0, 40.0), (200, 40), None));
        // And the floor does not reach real material.
        assert!(refuse_before_reading((56.0, 933.0), (1200, 935), None));
    }

    /// The height rule must not depend on the area flag, because it answers a
    /// different question -- a narrow column can reach both cuts while occupying a
    /// small fraction of the page. Slice D's narration is 56x933 on a 1200x935 slice: **4.7%** of
    /// the area, and every bit a fragment.
    #[test]
    fn a_narrow_column_is_a_fragment_even_with_the_area_rule_off() {
        assert!(refuse_before_reading((56.0, 933.0), (1200, 935), None));
        assert!(refuse_before_reading((56.0, 933.0), (1200, 935), Some(0.5)));
    }

    /// A zero-height page cannot divide, and must not refuse everything.
    #[test]
    fn a_degenerate_page_is_not_all_fragments() {
        assert!(!refuse_before_reading((100.0, 900.0), (1200, 0), None));
        assert!(!refuse_before_reading((100.0, -5.0), (1200, 935), None));
    }

    // ---- degenerate OCR reads -------------------------------------------------

    /// The one string in 27,100 region instances that the rule exists for.
    #[test]
    fn the_beep_loop_is_refused_and_comes_back_empty() {
        let loop_read = the_beep_loop();
        assert_eq!(loop_read.chars().count(), 512);
        assert!(degenerate_repetition(&loop_read));
        assert_eq!(refuse_degenerate_repetition(loop_read), "");
    }

    /// **The direction that matters.** A rule that fires too often is invisible
    /// in a positive-only test, so every real short periodic source on disk is
    /// asserted silent here by name. Together these are 1,207 of the 1,217
    /// periodic region instances in the corpus; the remaining 10 are the loop.
    ///
    /// The instance counts are asserted to sum to that 1,207 rather than being
    /// decoration beside the strings, because the aggregate is the only part of
    /// this table a reader cannot check by eye -- and it was wrong by 242.
    #[test]
    fn real_repetitive_sound_effects_are_not_refused() {
        let mut total = 0;
        for (text, instances) in REAL_SHORT_PERIODIC {
            total += instances;
            assert!(
                !degenerate_repetition(text),
                "{text:?} ({instances} instances on disk) must survive"
            );
            assert_eq!(
                refuse_degenerate_repetition((*text).to_owned()),
                *text,
                "{text:?} must be handed through unchanged"
            );
        }
        assert_eq!(
            (REAL_SHORT_PERIODIC.len(), total),
            (51, 1_207),
            "the table must stay the whole sub-floor periodic population, not a sample of it"
        );

        // The one real read that comes closest to the cover bar without
        // reaching it. It is not periodic, so it never depended on the floor --
        // which is exactly why it must not be counted as if it were.
        let (near_miss, instances) = THE_CLOSEST_NEAR_MISS;
        assert_eq!(near_miss.chars().count(), 9);
        assert!(
            !degenerate_repetition(near_miss),
            "{near_miss:?} ({instances} instances on disk) covers 0.875, under the 0.90 bar"
        );
    }

    /// The long end of the legitimate distribution, quoted from the corpus. The
    /// chemistry block is the most periodic *real* read at 48-95 characters
    /// (cover 0.26) and the volume's copyright notice is the longest real
    /// non-fixture read at 129 (both stood in for by invented text of the same
    /// length and comparable cover). Neither is near the bar.
    #[test]
    fn long_real_reads_are_not_refused() {
        let chemistry = "H-N-N-N-N-N-H\nH-N-N-H-N-H\nH-N-H-N-H\nBoiling pt = 41.82 deg C\nVolume = 0.913 L mol-1";
        assert_eq!(chemistry.chars().count(), 83);
        assert!(!degenerate_repetition(chemistry));

        let copyright = "この冊子に収められた文章と挿絵の権利は、すべて制作者に帰属します。事前の許可なく複製、転載、配布、改変（ウェブサイトや動画への掲載を含む）を行うことは固くお断りしています。また、私的な利用であっても、保護のための仕組みを外して写しを作ることはご遠慮ください。";
        assert_eq!(copyright.chars().count(), 129);
        assert!(!degenerate_repetition(copyright));

        let dense = "今日は朝から雨が降っていた。駅前の商店街はいつもより静かで、傘を差した人影がまばらに動いていた。彼女は改札の前で立ち止まり、時計を見上げてから小さくため息をついた。";
        assert!(!degenerate_repetition(dense));
    }

    /// Nothing short can be refused whatever its shape, which is what keeps the
    /// rule off the entire short band without inspecting it.
    ///
    /// **This test used to pass for every value of the floor it names.** Written
    /// purely in terms of `DEGENERATE_MIN_CORE - 1` and `DEGENERATE_MIN_CORE`,
    /// the two run assertions below are a tautology about a run of one
    /// character: they hold at 5, at 32 and at 5,000. Setting the floor to 5 left
    /// this test -- *the* test named after the floor -- green, and only the
    /// negative table went red, and that table tops out at 6 characters. So
    /// every floor from 7 to 32 was held in place by nothing at all.
    ///
    /// Two things pin it now: the literal value, and the real ceiling of the
    /// population at risk.
    #[test]
    fn the_length_floor_is_what_protects_the_sound_effects() {
        assert_eq!(
            DEGENERATE_MIN_CORE, 32,
            "every measured claim in this module is about this value; \
             changing it must break a test rather than slide through one"
        );

        // `．` is the single most repeated character in the corpus; below the
        // floor even a pure run of it is kept.
        let just_under = "．".repeat(DEGENERATE_MIN_CORE - 1);
        assert!(!degenerate_repetition(&just_under));
        let at_floor = "．".repeat(DEGENERATE_MIN_CORE);
        assert!(degenerate_repetition(&at_floor));

        // The floor has to clear the longest sound-effect reads that actually
        // exist, or the rule starts measuring them and only the cover bar
        // stands between a real effect and an empty source.
        for (text, core_len, rows, cover) in LONGEST_ONOMATOPOEIA_LENGTHS {
            assert_eq!(
                text.chars().filter(|c| !c.is_whitespace()).count(),
                *core_len,
                "{text:?}"
            );
            assert!(
                DEGENERATE_MIN_CORE > *core_len,
                "the floor is {DEGENERATE_MIN_CORE}, which does not clear {text:?} \
                 ({core_len} core chars, {rows} onomatopoeia rows on disk, cover {cover:.3}) \
                 -- the longest genuine sound-effect reads in the corpus must sit \
                 below the floor, not merely under the cover bar"
            );
            assert!(!degenerate_repetition(text), "{text:?} must survive");
            assert_eq!(
                refuse_degenerate_repetition((*text).to_owned()),
                *text,
                "{text:?} must be handed through unchanged"
            );
        }
    }

    /// Whitespace is not text, so a decoder that spaces its loop out must not
    /// escape by inflating the character count.
    #[test]
    fn whitespace_does_not_hide_a_loop_or_create_one() {
        let spaced: String = the_beep_loop()
            .chars()
            .flat_map(|c| [c, ' '])
            .collect::<String>();
        assert!(degenerate_repetition(&spaced));
        // ...and padding a legitimate short effect past the floor with spaces
        // still leaves it below it.
        let padded = format!("{:64}", "ポワポワ");
        assert!(!degenerate_repetition(&padded));
    }

    /// The cover bar, from the side that decides. A string that repeats a unit
    /// for most of its length but then says something else is language; the
    /// corpus's most periodic long real read scores 0.26 and the loop scores
    /// 1.00, so a run has to be nearly the whole string before it counts.
    #[test]
    fn a_long_read_that_merely_starts_with_a_run_is_kept() {
        let run_then_prose = format!("{}{}", "ピー".repeat(8), "って警報が鳴り止まないんだけど誰か止めてくれない");
        assert!(run_then_prose.chars().count() >= DEGENERATE_MIN_CORE);
        assert!(!degenerate_repetition(&run_then_prose));
    }

    /// A unit longer than the bound is a repeated *phrase*, which is a thing
    /// people write. Pinned so widening `DEGENERATE_MAX_UNIT` cannot happen by
    /// accident.
    #[test]
    fn a_repeated_phrase_longer_than_the_unit_bound_is_kept() {
        let refrain = "がんばって".repeat(8);
        assert_eq!(refrain.chars().count(), 40);
        assert!(refrain.chars().count() >= DEGENERATE_MIN_CORE);
        assert!(!degenerate_repetition(&refrain));
        // The same phrase clipped to four characters IS within the bound, which
        // is what the constant is trading away.
        assert!(degenerate_repetition(&"がんばっ".repeat(10)));
    }

    /// Empty is the shape a refusal hands downstream, and it has to be the shape
    /// `write_illegible_veto` partitions on -- otherwise the effect's pixels are
    /// erased with nothing lettered back into them. This is the coupling that
    /// makes "empty" the right answer rather than "drop the result".
    /// The watermark half of the veto, which is what stops 25 manhua pages
    /// coming back with a smear where a clean site plate had been. Every string
    /// here is shaped like a read from the 60-page corpus; site names are
    /// invented stand-ins, and the upper-case address entry now matches only
    /// through its generic phrase (no ledger entry carries case).
    #[test]
    fn a_site_watermark_is_withdrawn_from_the_erase_mask() {
        for mark in [
            "最新免費漫畫 paperleaf.com",
            "最新免费漫画 www.paperleaf.com",
            "本漫畫由紙葉漫畫收集整理",
            "WWW.PAPERLEAF.COM 最新免费漫画",
            "腾讯动漫",
        ] {
            assert!(watermark_text(mark), "{mark:?} should be a watermark");
        }
    }

    /// The rule must not eat dialogue: ordinary lines of the kind that corpus
    /// carries (invented stand-ins). A watermark test that
    /// refused ordinary speech would take the page off the page.
    #[test]
    fn ordinary_text_is_not_a_watermark() {
        for text in [
            "我的一盏旧灯",
            "A Lone Boat Crosses the Cold River by Night",
            "喵",
            "",
            "manga",
        ] {
            assert!(!watermark_text(text), "{text:?} is not a watermark");
        }
    }

    /// The two halves of the veto are independent: a watermark is perfectly
    /// legible text, so `illegible_text` says nothing about it, and that is
    /// exactly why it needed its own rule rather than a looser threshold on the
    /// old one.
    #[test]
    fn the_two_veto_rules_do_not_subsume_each_other() {
        let mark = "最新免费漫画 www.paperleaf.com";
        assert!(watermark_text(mark));
        assert!(!illegible_text(mark));
        // The composed predicate the veto actually calls. Without this line the
        // `||` can be deleted from the call site and every test stays green.
        assert!(withdraw_from_mask(mark, None, false, false, MisreadLevers::OFF));
        assert!(withdraw_from_mask("↓", None, false, false, MisreadLevers::OFF));
        assert!(!withdraw_from_mask(
            "A Lone Boat Crosses the Cold River by Night",
            None,
            false,
            false,
            MisreadLevers::OFF,
        ));
        assert!(illegible_text("↓"));
        assert!(!watermark_text("↓"));
    }

    /// A spot-rescue mint exists only in the in-flight `Edit`, so the frozen
    /// `input.scene` answers `EntityNotFound` for its content -- the same fact
    /// `layer_geometry_is_writable` guards in detection -- and the veto writer
    /// walked `pending` into it with an unguarded `?`. The render measured the
    /// cost: 4 mints on a test chapter, 4 whole-page 500s -- 100% of the mint
    /// path.
    ///
    /// Driven through `write_illegible_veto` itself -- the function the run
    /// calls -- and asserted twice, because there are two wrong fixes and each
    /// passes one half. (1) `Ok`: the crash itself. (2) the veto asset is
    /// byte-identical with and without the mint in `pending` (`BlobId` is
    /// blake3 of the bytes): a mint belongs to NEITHER partition. Its read
    /// already passed `withdraw_from_mask` free-standing before minting, with
    /// the same inputs the veto would recompute, so it can never join
    /// `refused`; and joining `kept` would punch a hole in an overlapping
    /// refusal's veto -- re-enabling an erase under the lettering that
    /// `--spot-rescue-erase` OFF promised not to make. The mint's erase is its
    /// own flag's decision, through `write_rescue_mask`, which unions AFTER
    /// this veto. Computing the mint's verdict from a carried role fails (2)
    /// exactly as swallowing the lookup with `.ok()` does.
    #[test]
    fn a_minted_result_neither_fails_nor_reshapes_the_veto() {
        let generation = super::super::generation(super::PRODUCER, "hunyuan-ocr-1.5").unwrap();
        // The refused-unread box the veto must paint in both arms; the mint's
        // box sits strictly inside it, so a mint wrongly landing in `kept`
        // punches a visible hole.
        let unread = [(100.0_f64, 100.0, 400.0, 400.0)];
        let arm = |with_mint: bool| {
            let mut session = koharu_scene::Session::memory().unwrap();
            let mut page_id = None;
            let patch = session
                .snapshot()
                .patch(|edit| {
                    page_id = Some(edit.add_page(
                        koharu_scene::PageDraft::new("page", 1000.0, 1000.0),
                        koharu_scene::At::End,
                    )?);
                    Ok(())
                })
                .unwrap();
            session.commit(patch).unwrap();
            let page = page_id.unwrap();
            let scene = session.snapshot();
            let mut edit = scene.edit_as(generation.clone());
            let mut pending: Vec<(OcrResult, koharu_scene::Generation)> = Vec::new();
            if with_mint {
                // The trio exactly as the rescue mints it.
                let geometry = Geometry::rectangle(150.0, 150.0, 200.0, 200.0);
                let region = edit.add_entity(page, koharu_scene::At::End).unwrap();
                edit.set(region, &geometry).unwrap();
                edit.set(
                    region,
                    &koharu_scene::Region {
                        origin: Origin::Generated(generation.clone()),
                        kind: koharu_scene::RegionKind::new(TextRegion::KIND).unwrap(),
                        label: Some("text".to_owned()),
                    },
                )
                .unwrap();
                let content = edit.add_text_content(page, koharu_scene::At::End).unwrap();
                edit.set(
                    content,
                    &super::TextRole {
                        origin: Origin::Generated(generation.clone()),
                        role: "dev.koharu.text.free-text".to_owned(),
                    },
                )
                .unwrap();
                pending.push((
                    OcrResult {
                        content,
                        region,
                        geometry,
                        previous: None,
                        text: "别过来！".to_owned(),
                        confidence: Some(0.9),
                        minted: true,
                    },
                    generation.clone(),
                ));
            }
            let outcome = super::write_illegible_veto(
                &mut edit,
                &scene,
                page,
                &DynamicImage::new_rgb8(1000, 1000),
                &pending,
                None,
                &unread,
                false,
                &std::collections::HashSet::new(),
                MisreadLevers::OFF,
            );
            let patch = edit.finish().unwrap();
            session.commit(patch).unwrap();
            let blob = session
                .snapshot()
                .asset(
                    page,
                    &koharu_scene::AssetRole::new("text-mask-veto").unwrap(),
                )
                .unwrap()
                .map(|asset| asset.blob);
            (outcome, blob)
        };
        let (control_outcome, control_blob) = arm(false);
        control_outcome.unwrap();
        assert!(
            control_blob.is_some(),
            "the control arm must write a veto for the unread box"
        );
        let (minted_outcome, minted_blob) = arm(true);
        assert!(
            minted_outcome.is_ok(),
            "a result minted in this same stage call must not fail the veto writer: {minted_outcome:?}"
        );
        assert_eq!(
            minted_blob, control_blob,
            "a mint joins neither partition: the veto must be byte-identical with and without it"
        );
    }

    /// **A mouth line read as a prolongation mark.**
    ///
    /// Driven through `withdraw_from_mask` with the arguments that region actually
    /// carried in a test render: role `dialogue`, so
    /// `free_standing` is FALSE and the script arm is structurally unreachable, and
    /// `scoped` false. Asserting `illegible_text` alone would pass with the whole
    /// call deleted from the veto -- the halves-not-the-`||` trap this file has
    /// already caught twice.
    ///
    /// The erase is written before OCR runs, so this predicate withdrawing the mask
    /// is what pays for the 8,535 px of her face the lettering rule alone would
    /// have left destroyed.
    #[test]
    fn a_read_that_is_only_prolongation_marks_is_withdrawn() {
        // Built by `parse`, not from a literal -- the trap pinned a few tests below,
        // where a `Some("zh-CN")` literal was green over a feature dead in the render.
        let ja: Language = "ja".parse().expect("ja parses");
        // What OCR actually returned for the spurious bubble, verbatim.
        assert!(withdraw_from_mask("ー", Some(ja), false, false, MisreadLevers::OFF));
        assert!(illegible_text("ー"));
        // Halfwidth twin, and a run of them: same reasoning, no preceding vowel.
        assert!(withdraw_from_mask("ｰ", Some(ja), false, false, MisreadLevers::OFF));
        assert!(withdraw_from_mask("ーー", Some(ja), false, false, MisreadLevers::OFF));
        // Not language-dependent -- the region carried ja-JP, but a mouth line is
        // a mouth line whatever the page is written in.
        assert!(withdraw_from_mask("ー", None, false, false, MisreadLevers::OFF));
    }

    /// **The reads that must SURVIVE it, each shaped like a real read from the
    /// corpora the population render covered (the two longer lines are invented
    /// stand-ins).**
    ///
    /// `ん…` is the load-bearing one: a genuine balloon whose entire read is ONE
    /// character. Character count was one of only three fields
    /// that separated the false positive from the true positive, so a rule keyed on
    /// length would have thrown this away -- which is why the population was
    /// rendered BEFORE the rule was written rather than after.
    #[test]
    fn ordinary_reads_carrying_a_prolongation_are_kept() {
        let ja: Language = "ja".parse().expect("ja parses");
        for text in [
            "すーっと？",   // dialogue, carries the mark mid-word
            "そうだよー",   // the mark is TRAILING and still real
            "ん…",          // a whole balloon of one character
            "あー",         // the shortest legitimate use: one vowel, lengthened
            "冥霜之王！",   // the true positive this feature exists for
        ] {
            assert!(
                !withdraw_from_mask(text, Some(ja), false, false, MisreadLevers::OFF),
                "{text:?} is a real read and must not be withdrawn"
            );
            assert!(!illegible_text(text), "{text:?} must not be illegible");
        }
    }

    /// The reserve re-read's candidate gate, driven through the composed
    /// predicate with reads shaped like the censused refusal population's
    /// (invented stand-ins) -- fire and not-fire both, because testing the
    /// halves leaves the `&&`s between them unexercised.
    ///
    /// The levers are built as `resolve` ships them (both ON), and the language
    /// from a `parse`, per the trap pinned two tests below.
    #[test]
    fn the_reserve_reread_fires_on_the_lever_refused_dialogue_population_only() {
        let ko: Language = "ko".parse().expect("ko parses");
        let shipping = MisreadLevers {
            leave_misread_bubbles: true,
            korean_script_strict: true,
        };
        // The exhibit: drawn `어…` misread `他….` at 0.317, role dialogue.
        // The whole reason the pass exists.
        assert!(reread_candidate("他….", Some(ko), false, false, shipping));
        // Drawn `씨익` misread "BOLT" -- an all-Latin read, a candidate BY
        // TEXT; the onomatopoeia label is what must exclude it, so both sides
        // of that label test are driven here.
        assert!(reread_candidate("BOLT", Some(ko), false, false, shipping));
        assert!(!reread_candidate("BOLT", Some(ko), false, true, shipping));
        // Punctuation-only refusals are `illegible_text`'s population -- the
        // drawn `?!` IS the correct lettering (8 of the 19 censused refusals),
        // and the pupil's `·` is the same class.
        assert!(!reread_candidate("?!", Some(ko), false, false, shipping));
        assert!(!reread_candidate("·", Some(ko), false, false, shipping));
        // A prolongation-only read is the one illegible class the script tests
        // do NOT refuse on their own: `ー` counts as kana, so on declared ko
        // the ratio arm scores it 1.0 and only the `illegible_text` guard
        // keeps a mouth-line misread from buying a pointless re-read. This
        // line is what proves that guard is live -- deleting it went green
        // through every case above.
        assert!(!reread_candidate("ーー", Some(ko), false, false, shipping));
        // Free-standing regions never qualify; their refusals belong to other
        // rules (the upright pass owns the recoverable half of that class).
        assert!(!reread_candidate("他….", Some(ko), true, false, shipping));
        // Declared Japanese, or no declaration: not this pass's population.
        let ja: Language = "ja".parse().expect("ja parses");
        assert!(!reread_candidate("他….", Some(ja), false, false, shipping));
        assert!(!reread_candidate("他….", None, false, false, shipping));
        // A hangul read was never refused; there is nothing to recover.
        assert!(!reread_candidate("어…", Some(ko), false, false, shipping));
        // Levers off: dialogue is never refused
        // there, so a re-read would buy nothing and must not fire.
        assert!(!reread_candidate(
            "他….",
            Some(ko),
            false,
            false,
            MisreadLevers::OFF
        ));
    }

    /// Admission is SCRIPT MEMBERSHIP -- confidence comparisons were measured
    /// unreliable. Recoveries shaped like the four measured ones admit; garbage
    /// shaped like the measured garbage declines (all invented stand-ins).
    #[test]
    fn a_reserve_reread_is_admitted_by_hangul_majority_alone() {
        // Stand-ins for the four EXACT recoveries the reserve engine measured
        // on padded crops, each with the measured read's shape.
        for text in ["어…", "그만해라…", "저, 저것은", "꽤나재밌는"] {
            assert!(admit_hangul_reread(text), "{text:?} is a recovery");
        }
        // What a wrong-script re-read looks like: the elongated-kana Han
        // garble, a romanization, an empty read, bare punctuation, and the
        // original misread itself.
        for text in ["木木木木木木木月", "HAN-GYUL", "", "?!", "他…."] {
            assert!(!admit_hangul_reread(text), "{text:?} must stay refused");
        }
        // The majority bar exactly: one hangul beside one han is half and
        // admits; one hangul beside two han is under half and declines.
        assert!(admit_hangul_reread("어他"));
        assert!(!admit_hangul_reread("어他他"));
    }

    /// The THIRD arm, asserted through the composed predicate rather than through
    /// `script_mismatch`, and with both halves of its `&&` driven independently.
    ///
    /// A test of `script_mismatch` alone passes with the whole arm deleted from
    /// `withdraw_from_mask`; that is exactly how a fix once shipped unwired with
    /// 128 green, and the comment on `withdraw_from_mask` says so.
    #[test]
    fn the_script_arm_fires_only_free_standing_and_only_when_told_the_language() {
        // Kana on a Chinese page: a bad read, and the whole point of the arm.
        let kana_on_manhua = "ソレデ";
        // Neither of the two older arms sees it -- so if this text is withdrawn,
        // it can only be the third one that did it.
        assert!(!illegible_text(kana_on_manhua));
        assert!(!watermark_text(kana_on_manhua));

        /* THE LANGUAGE COMES FROM A `parse`, NOT FROM A LITERAL, AND THAT IS THE
         * POINT OF THIS LINE. The first version of this test passed the literal
         * `Some("zh-CN")` into a `&str` parameter and was green while the feature
         * was DEAD in the render: production had a `Language` and passed
         * `to_string()`, which strum renders as "Simplified Chinese", so the tag
         * parser matched nothing and the arm never fired. Twenty pages of GPU
         * proved it -- the veto counts were identical with and without the flag.
         * Building the value the way the caller builds it is what closes that
         * gap; the signature takes `Language` now, so a string cannot be smuggled
         * in here again. */
        let zh: Language = "zh".parse().expect("zh parses");
        let ja: Language = "ja".parse().expect("ja parses");
        let ko: Language = "ko".parse().expect("ko parses");
        assert_eq!(zh.to_string(), "Simplified Chinese", "the trap, pinned");

        // Told the language, free-standing: withdrawn.
        assert!(withdraw_from_mask(kana_on_manhua, Some(zh), true, false, MisreadLevers::OFF));

        // Both halves of the `&&`, each flipped alone.
        assert!(
            !withdraw_from_mask(kana_on_manhua, Some(zh), false, false, MisreadLevers::OFF),
            "in a bubble the erase already happened, so sparing pixels leaves a blank hole"
        );
        assert!(
            !withdraw_from_mask(kana_on_manhua, None, true, false, MisreadLevers::OFF),
            "an undeclared language must fire nothing, or genuine Japanese gets dropped"
        );

        // Japanese is the case that must never fire: kana and Han both belong.
        assert!(!withdraw_from_mask(kana_on_manhua, Some(ja), true, false, MisreadLevers::OFF));
        assert!(!withdraw_from_mask("それは、", Some(ja), true, false, MisreadLevers::OFF));

        // Korean: Han and kana are both foreign, Hangul is not.
        assert!(withdraw_from_mask("ソレデ", Some(ko), true, false, MisreadLevers::OFF));
        assert!(!withdraw_from_mask("안녕하세요", Some(ko), true, false, MisreadLevers::OFF));

        // And Chinese Han on a Chinese page is a GOOD read, not a mismatch.
        assert!(!withdraw_from_mask("孤舟夜渡寒江去", Some(zh), true, false, MisreadLevers::OFF));

        // The shape of the region the GPU run found on the manhua corpus:
        // kana and Han mixed, free-text, on a page declared Chinese.
        assert!(withdraw_from_mask("」ト土？", Some(zh), true, false, MisreadLevers::OFF));
    }

    /// The ordering trap `Scripts::of` inherits from `labels.rs`: `・` (U+30FB)
    /// lives inside the kana block, so a test that reaches the block before the
    /// punctuation check reads a row of interpuncts as kana and calls a Korean
    /// page 100% foreign script.
    #[test]
    fn interpuncts_are_punctuation_before_they_are_kana() {
        let ko: Language = "ko".parse().unwrap();
        let zh: Language = "zh".parse().unwrap();
        assert!(!script_mismatch("・・・", Some(ko)));
        assert!(!script_mismatch("・・・", Some(zh)));
    }

    #[test]
    fn a_refused_read_is_illegible_so_the_erase_mask_is_withdrawn() {
        let refused = refuse_degenerate_repetition(the_beep_loop());
        assert_eq!(refused, "");
        assert!(illegible_text(&refused));
        // and the loop itself is NOT illegible, so nothing else in the pipeline
        // would have spared it.
        assert!(!illegible_text(&the_beep_loop()));
    }

    /// `normalize_ocr_text` runs first and must not manufacture a loop: an
    /// all-placeholder read collapses to a single ellipsis, far under the floor.
    #[test]
    fn normalization_runs_first_and_cannot_create_a_refusal() {
        let boxes = "☐".repeat(64);
        let normalized = normalize_ocr_text(boxes);
        assert_eq!(normalized, "…");
        assert!(!degenerate_repetition(&normalized));
    }

    // ---- the turned second read of a sideways display column ----
    //
    // The strings below have the shape of one PaddleOCR-VL call on a whole
    // joined column (crop x 44..293 y 18..3175), taken with no detection in the
    // loop so it is the read this code path actually produces; the names and
    // the site are invented stand-ins:
    //
    //   turned 90 CCW : 苍炎之王·冥霜之王·裂风之王最新免费漫画 www.paperleef.com
    //   upright       : 仓炎岛王王·冥雪岛王王 / 最新免费漫画 / www.papcrleaf.com
    //
    // The reference line is `苍炎之王 · 冥霜之王 · 裂风之王` -- twelve glyphs, two
    // interpuncts. Upright gets two names and both are wrong; turned gets all
    // three, byte-exact.

    /// A crop carrying a marker in its TOP-LEFT corner.
    ///
    /// **The marker is why this tests the DIRECTION and not merely the turn.**
    /// `rotate90` and `rotate270` both take a portrait crop to a landscape one, so
    /// a stub keying on aspect ratio would pass under either -- and the two are not
    /// interchangeable: the CLOCKWISE turn of this very column was measured
    /// returning nothing from every engine.
    fn column(width: u32, height: u32) -> DynamicImage {
        let mut buffer = image::RgbImage::new(width, height);
        buffer.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        DynamicImage::ImageRgb8(buffer)
    }

    /// Whether this crop was turned 90 COUNTER-clockwise, read off where the
    /// marker landed: CCW sends the top-left corner to the bottom-left, CW sends
    /// it to the top-right.
    fn turned_ccw(image: &DynamicImage) -> bool {
        image.width() > image.height()
            && image.to_rgb8().get_pixel(0, image.height() - 1) == &image::Rgb([255, 0, 0])
    }

    fn target(image: DynamicImage, reread_rotated: bool) -> OcrTarget {
        let (width, height) = (f64::from(image.width()), f64::from(image.height()));
        OcrTarget {
            content: EntityId::new(),
            region: EntityId::new(),
            geometry: Geometry::rectangle(0.0, 0.0, width, height),
            previous: None,
            image,
            reread_rotated,
            reread_flipped: false,
            flip_crop: None,
            grown: None,
        }
    }

    fn flip_target(image: DynamicImage) -> OcrTarget {
        OcrTarget {
            reread_flipped: true,
            ..target(image, false)
        }
    }

    /// Whether this crop was turned exactly 180 degrees: the corner marker lands
    /// in the BOTTOM-RIGHT, where neither the CCW turn (bottom-left) nor the
    /// upright crop (top-left) puts it, and the dimensions do not change.
    fn flipped_180(image: &DynamicImage) -> bool {
        image.to_rgb8().get_pixel(image.width() - 1, image.height() - 1) == &image::Rgb([255, 0, 0])
    }

    /// **THE TEST THAT MATTERS.** It drives `infer_text` -- the function the stage
    /// actually calls -- so it asserts the rotate, the gate, the normalise pair and
    /// the pick as ONE unit. The unit tests below cannot do that: this file's own
    /// warning above `withdraw_from_mask` records 128 green tests over a deleted
    /// `||`, and the same hole exists here in three places.
    ///
    /// Prove it red by flipping `rotate270` to `rotate90`, or by deleting the
    /// `choose_orientation` call -- only this test goes red for either.
    #[tokio::test]
    async fn a_sideways_column_is_read_from_the_turned_crop() {
        let reads = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&reads);
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(248, 3157), true)],
            text_only(move |(), image: &DynamicImage| {
                counter.fetch_add(1, Ordering::SeqCst);
                // The three names come back only from a genuine COUNTER-clockwise
                // turn, which is what the measurement used.
                Ok(if turned_ccw(image) {
                    "苍炎之王·冥霜之王·裂风之王".to_string()
                } else {
                    "仓炎岛王王·冥雪岛王王".to_string()
                })
            }),
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "苍炎之王·冥霜之王·裂风之王");
        assert_eq!(reads.load(Ordering::SeqCst), 2, "both reads must be taken");
    }

    /// The instrumentation contract: the confidence an engine reports rides the
    /// SAME result as its text, rather than being computed and dropped inside
    /// `infer_text`.
    #[tokio::test]
    async fn the_engine_confidence_reaches_the_result() {
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(100, 40), false)],
            |(): &(), _: &DynamicImage| Ok(("寒霜".to_string(), Some(0.73))),
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].confidence, Some(0.73));
    }

    /// "Not measured" survives as `None`, never as zero -- the distinction every
    /// caller of a confidence in this file is built on.
    #[tokio::test]
    async fn a_missing_confidence_stays_missing() {
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(100, 40), false)],
            text_only(|(), _: &DynamicImage| Ok("寒霜".to_string())),
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].confidence, None);
    }

    /// The KEPT read's confidence, not the first read's. A turned pair the margin
    /// resolves must report the score of the read it actually shipped -- both
    /// directions, or the wire pairs one read's text with the other's number.
    #[tokio::test]
    async fn the_kept_reads_confidence_travels_with_a_turned_pair() {
        let upright_kept = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(248, 3157), true)],
            |(): &(), image: &DynamicImage| {
                Ok(if turned_ccw(image) {
                    ("玄水真诀".to_string(), Some(0.2))
                } else {
                    ("玄冰真诀".to_string(), Some(0.9))
                })
            },
            None::<RotateFn<()>>,
            Some(0.17),
        )
        .await
        .unwrap();
        assert_eq!(upright_kept[0].text, "玄冰真诀");
        assert_eq!(
            upright_kept[0].confidence,
            Some(0.9),
            "the upright read shipped, so its score must ship with it"
        );

        let turned_kept = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(248, 3157), true)],
            |(): &(), image: &DynamicImage| {
                Ok(if turned_ccw(image) {
                    ("玄冰真诀".to_string(), Some(0.2))
                } else {
                    ("玄水真诀".to_string(), Some(0.9))
                })
            },
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(turned_kept[0].text, "玄冰真诀");
        assert_eq!(
            turned_kept[0].confidence,
            Some(0.2),
            "the turned read shipped, so its score must ship with it"
        );
    }

    /// The four arms of the composed predicate, over a REAL snapshot rather than
    /// a precomputed flag, so an implementation reading the wrong carrier fails
    /// here rather than in a render: `Region.label` does not hold the marker,
    /// the second `DetectionAnalysis` label does.
    #[test]
    fn flip_candidate_is_true_only_for_a_synthesised_region_with_the_flag_on() {
        let mut session = koharu_scene::Session::memory().unwrap();
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(
                    koharu_scene::PageDraft::new("page", 1200.0, 1800.0),
                    koharu_scene::At::End,
                )?;
                let synthesised = edit.add_entity(page, koharu_scene::At::End)?;
                edit.set(synthesised, &Geometry::rectangle(114.0, 457.0, 1075.0, 439.0))?;
                edit.set(
                    synthesised,
                    &koharu_scene::Region {
                        origin: Origin::User,
                        kind: koharu_scene::RegionKind::new(TextRegion::KIND).unwrap(),
                        label: None,
                    },
                )?;
                edit.set(
                    synthesised,
                    &DetectionAnalysis {
                        origin: Origin::User,
                        labels: vec![
                            koharu_scene::DetectionLabel {
                                kind: koharu_scene::RegionKind::new(TextRegion::KIND).unwrap(),
                                confidence: 0.5078125,
                            },
                            koharu_scene::DetectionLabel {
                                kind: koharu_scene::RegionKind::new(SYNTHESISED_REGION_KIND)
                                    .unwrap(),
                                confidence: 0.5078125,
                            },
                        ],
                    },
                )?;
                let ordinary = edit.add_entity(page, koharu_scene::At::End)?;
                edit.set(ordinary, &Geometry::rectangle(10.0, 10.0, 200.0, 100.0))?;
                edit.set(
                    ordinary,
                    &koharu_scene::Region {
                        origin: Origin::User,
                        kind: koharu_scene::RegionKind::new(TextRegion::KIND).unwrap(),
                        label: None,
                    },
                )?;
                edit.set(
                    ordinary,
                    &DetectionAnalysis {
                        origin: Origin::User,
                        labels: vec![koharu_scene::DetectionLabel {
                            kind: koharu_scene::RegionKind::new(TextRegion::KIND).unwrap(),
                            confidence: 0.9,
                        }],
                    },
                )?;
                ids = Some((synthesised, ordinary));
                Ok(())
            })
            .unwrap();
        session.commit(patch).unwrap();
        let (synthesised, ordinary) = ids.unwrap();
        let scene = session.snapshot();
        assert!(flip_candidate(&scene, synthesised, true));
        assert!(
            !flip_candidate(&scene, ordinary, true),
            "an ordinary detection must never flip"
        );
        assert!(
            !flip_candidate(&scene, synthesised, false),
            "the flag off must gate the synthesised region too"
        );
        assert!(!flip_candidate(&scene, ordinary, false));
    }

    /// **THE TEST THAT MATTERS for the flip** -- it drives `infer_text`, the
    /// function the stage actually calls, so the flip, the second read, the
    /// normalise chain and the pick are asserted as ONE unit. Prove it red by
    /// deleting the `reread_flipped` branch, or by wiring `choose_orientation`
    /// in place of `choose_flip` -- the margin case below catches the swap.
    #[tokio::test]
    async fn a_flipped_synthesised_bubble_ships_the_flipped_read_only_when_it_clearly_wins() {
        let reads = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&reads);
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![flip_target(column(1075, 439))],
            move |(): &(), image: &DynamicImage| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(if flipped_180(image) {
                    ("冥霜之王！".to_string(), Some(0.9))
                } else {
                    ("丁木妈石图".to_string(), Some(0.3))
                })
            },
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "冥霜之王！", "0.6 clears the 0.17 margin");
        assert_eq!(
            results[0].confidence,
            Some(0.9),
            "the flipped read shipped, so its score must ship with it"
        );
        assert_eq!(reads.load(Ordering::SeqCst), 2, "both orientations must be read");
    }

    /// Within the margin the upright read stays, even when the flip scores
    /// HIGHER. Eleven of twelve genuine textless bubbles are upright; a
    /// tie must not take one from them -- and this is exactly the case that goes
    /// wrong if anyone swaps `choose_orientation` in: its default winner is the
    /// rotated read.
    #[tokio::test]
    async fn a_flip_inside_the_margin_keeps_the_upright_read() {
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![flip_target(column(1075, 439))],
            |(): &(), image: &DynamicImage| {
                Ok(if flipped_180(image) {
                    ("冥霜之王！".to_string(), Some(0.5))
                } else {
                    ("ん…".to_string(), Some(0.4))
                })
            },
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "ん…");
        assert_eq!(results[0].confidence, Some(0.4));
    }

    /// The arm `choose_orientation` INVERTS, driven independently on each side.
    /// On `manga-ocr`, `baberu-ocr` and Ollama every confidence is `None`; if a
    /// missing score handed the pair to the flip, every genuine textless-bubble
    /// read on the shipping Japanese arm would be replaced on no evidence.
    #[test]
    fn choose_flip_keeps_upright_when_either_confidence_is_missing() {
        let arms: [(Option<f32>, Option<f32>); 3] =
            [(None, Some(0.99)), (Some(0.01), None), (None, None)];
        for (up, down) in arms {
            let (text, confidence) =
                choose_flip("ん…".to_string(), up, "junk".to_string(), down);
            assert_eq!(text, "ん…", "up={up:?} down={down:?}");
            assert_eq!(confidence, up, "the kept read keeps its own score");
        }
    }

    /// An empty flipped read never wins, whatever it scores.
    #[test]
    fn an_empty_flip_never_wins() {
        let (text, confidence) =
            choose_flip("ん…".to_string(), Some(0.01), "  ".to_string(), Some(0.99));
        assert_eq!(text, "ん…");
        assert_eq!(confidence, Some(0.01));
    }

    /// An unscored engine pays NOTHING: the second read is skipped entirely,
    /// not taken and discarded.
    #[tokio::test]
    async fn an_unscored_engine_never_takes_the_flipped_read_at_all() {
        let reads = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&reads);
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![flip_target(column(1075, 439))],
            text_only(move |(), _: &DynamicImage| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok("ん…".to_string())
            }),
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "ん…");
        assert_eq!(reads.load(Ordering::SeqCst), 1, "no score, no second read");
    }

    /// A bubble the flag did not elect is read once -- the OFF arm is
    /// byte-identical to today, whatever the engine would have scored.
    #[tokio::test]
    async fn an_unelected_bubble_is_read_once() {
        let reads = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&reads);
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(1075, 439), false)],
            move |(): &(), image: &DynamicImage| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(if flipped_180(image) {
                    ("wrong side".to_string(), Some(0.99))
                } else {
                    ("ん…".to_string(), Some(0.1))
                })
            },
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "ん…");
        assert_eq!(reads.load(Ordering::SeqCst), 1);
    }

    /// A synthetic balloon: white body, a dark glyph inside it, a dark border
    /// ring around it, and coloured "art" filling the rest of the crop.
    fn balloon_fixture() -> DynamicImage {
        let mut buffer = image::RgbImage::from_pixel(400, 200, image::Rgb([120, 40, 160]));
        for y in 24..176 {
            for x in 54..346 {
                buffer.put_pixel(x, y, image::Rgb([10, 10, 10]));
            }
        }
        for y in 30..170 {
            for x in 60..340 {
                buffer.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        for y in 90..110 {
            for x in 190..210 {
                buffer.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        DynamicImage::ImageRgb8(buffer)
    }

    /// The measured contract of `isolate_balloon`: the art dies, the glyph
    /// survives. The grid's exact-match cells all sit on the isolated crop, so
    /// an isolation that leaks art -- or eats the glyph -- reproduces the miss.
    #[test]
    fn isolation_keeps_the_balloon_and_paints_the_art_white() {
        let isolated = isolate_balloon(&balloon_fixture())
            .expect("a white body this large must isolate");
        let rgb = isolated.to_rgb8();
        assert_eq!(
            rgb.get_pixel(0, 0),
            &image::Rgb([255, 255, 255]),
            "the cut's corner was art and must come back white"
        );
        // The dilation deliberately keeps a stroke-width RING around the body --
        // the measured instrument kept the same halo and the exact-match cells
        // were read through it -- so some rim art survives by design. What must
        // die is the art FIELD beyond the rim.
        let original = balloon_fixture()
            .to_rgb8()
            .pixels()
            .filter(|p| p.0 == [120, 40, 160])
            .count();
        let art = rgb.pixels().filter(|p| p.0 == [120, 40, 160]).count();
        assert!(
            art * 5 < original,
            "the art field must be painted white: {art} of {original} art pixels survive"
        );
        let dark = rgb
            .pixels()
            .filter(|p| p.0.iter().all(|&c| c < 90))
            .count();
        assert!(
            dark >= 400,
            "the glyph's ink must survive isolation, got {dark} dark pixels"
        );
    }

    /// Fail-open: a crop with no credible white body ships its raw crop -- the
    /// behaviour the flip had when it landed. A dark balloon must not be
    /// whited out by its own isolation.
    #[test]
    fn a_crop_without_a_white_body_fails_open() {
        let dark = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            300,
            200,
            image::Rgb([30, 20, 40]),
        ));
        assert!(
            isolate_balloon(&dark).is_none(),
            "no body, no isolation -- the raw crop must ship"
        );
    }

    /// The flipped read must come from the ISOLATED twin when one exists. The
    /// grid's exact-match cells are all on the isolated crop, so a branch that
    /// quietly reads the raw crop reproduces the old miss with every other
    /// test green. Prove it red by reading `target.image` in the flip branch.
    #[tokio::test]
    async fn the_flip_reads_the_isolated_twin_when_one_exists() {
        let reads = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&reads);
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![OcrTarget {
                flip_crop: Some(column(500, 300)),
                ..flip_target(column(1075, 439))
            }],
            move |(): &(), image: &DynamicImage| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(if image.width() == 500 && flipped_180(image) {
                    ("冥霜之王！".to_string(), Some(0.9))
                } else if flipped_180(image) {
                    ("the raw crop, the miss the grid measured".to_string(), Some(0.9))
                } else {
                    ("丁木妈石图".to_string(), Some(0.3))
                })
            },
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            results[0].text,
            "冥霜之王！",
            "the isolated twin must be the crop that flips"
        );
        assert_eq!(reads.load(Ordering::SeqCst), 2);
    }

    /// The refine grid ships the best-scoring angle, through the real caller.
    /// The rotate stub keys on the ANGLE, so a branch that never asks for the
    /// neighbours -- or picks by position instead of score -- goes red here and
    /// nowhere else. The knife-edge this exists for: the pipeline's own crop
    /// bytes at 180 read the wrong glyph while a neighbour reads the truth.
    #[tokio::test]
    async fn the_refine_ships_the_best_scoring_angle() {
        let upright_reads = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&upright_reads);
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![flip_target(column(1075, 439))],
            move |(): &(), _: &DynamicImage| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(("丁木妈石图".to_string(), Some(0.3)))
            },
            Some(|(): &(), _: &DynamicImage, angle: f64| {
                Ok(match angle {
                    a if a == 175.0 => ("冥霜之王！".to_string(), Some(0.9)),
                    a if a == 180.0 => ("冥霜之用！".to_string(), Some(0.7)),
                    _ => ("junk".to_string(), Some(0.2)),
                })
            }),
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "冥霜之王！", "the 175 candidate outscores 180");
        assert_eq!(results[0].confidence, Some(0.9));
        assert_eq!(
            upright_reads.load(Ordering::SeqCst),
            1,
            "the refine replaces the flip's read, it does not add an upright one"
        );
    }

    /// An all-empty refine keeps the upright read: `best_flip` returns the
    /// empty string and `choose_flip`'s first guard funnels it to upright --
    /// the same fail-safe every other arm of the mechanism lands on.
    #[tokio::test]
    async fn an_all_empty_refine_keeps_the_upright_read() {
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![flip_target(column(1075, 439))],
            |(): &(), _: &DynamicImage| Ok(("ん…".to_string(), Some(0.4))),
            Some(|(): &(), _: &DynamicImage, _: f64| Ok(("  ".to_string(), Some(0.99)))),
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "ん…");
        assert_eq!(results[0].confidence, Some(0.4));
    }

    /// A refined candidate is cleaned on the SAME chain as every other read --
    /// a raw candidate would not be comparable to the upright read, and a
    /// degenerate one would dodge the empty guard.
    #[tokio::test]
    async fn the_refined_read_is_cleaned_on_the_same_chain() {
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![flip_target(column(1075, 439))],
            |(): &(), _: &DynamicImage| Ok(("丁木妈石图".to_string(), Some(0.3))),
            Some(|(): &(), _: &DynamicImage, _: f64| {
                Ok((r"\(冥霜之王！\)".to_string(), Some(0.9)))
            }),
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            results[0].text,
            "冥霜之王！",
            "the LaTeX wrapper must be stripped before the pick"
        );
    }

    /// `best_flip`'s three contracts, as data: highest score wins, `None`
    /// never beats `Some`, and empty candidates cannot win at any score.
    #[test]
    fn best_flip_picks_the_highest_scored_non_empty_candidate() {
        let (angle, text, confidence) = best_flip(vec![
            (175.0, "王".to_string(), Some(0.9)),
            (180.0, "用".to_string(), Some(0.7)),
            (185.0, "junk".to_string(), Some(0.2)),
        ]);
        assert_eq!((angle, text.as_str(), confidence), (175.0, "王", Some(0.9)));

        let (_, text, _) = best_flip(vec![
            (175.0, "scored".to_string(), Some(0.01)),
            (180.0, "unscored".to_string(), None),
        ]);
        assert_eq!(text, "scored", "None must lose to any Some");

        let (_, text, confidence) = best_flip(vec![
            (175.0, "  ".to_string(), Some(0.99)),
            (180.0, String::new(), Some(0.98)),
        ]);
        assert_eq!(
            text, "",
            "all-empty returns empty, which choose_flip turns into upright"
        );
        assert_eq!(confidence, None);
    }

    /// The spot rescue's COMPOSED admission predicate on the measured numbers:
    /// a banner box survives against everything really on that page, and the
    /// watermark plate dies on the area floor -- with no string rule, which is
    /// the point: the plate's read came back MANGLED on 14 pages.
    #[test]
    fn spot_box_survives_the_filters() {
        // norm1000 [135,544,845,890] on 1200x1041 -> the banner box
        let banner = denormalize_spot_box([135.0, 544.0, 845.0, 890.0], 1200, 1041).unwrap();
        // the page's two real proposals: the misplaced off-canvas box and the
        // speed-line box, both from the wire
        let existing = [
            (-66.798, -117.215, 881.253, 735.614),
            (866.577, 313.637, 376.220, 401.527),
        ];
        assert!(
            spot_rescue_candidate(banner, (1200, 1041), &existing, 0.456),
            "the banner box must clear the floor, the ceiling and the fence"
        );
        // the watermark plate, norm1000 [828,936,987,986] on the same page --
        // ink passed HIGH so the refusal is pinned to the AREA floor alone
        let plate = denormalize_spot_box([828.0, 936.0, 987.0, 986.0], 1200, 1041).unwrap();
        assert!(
            !spot_rescue_candidate(plate, (1200, 1041), &existing, 0.456),
            "the plate is ~0.8% of the page and must die on the area floor"
        );
    }

    /// The ink floor on the measured population. FIRE: a HUD-chrome mint --
    /// norm1000 [668,648,998,755] on 1200x1045, area share 0.0353 CLEARS the
    /// area floor, ink 0.00996 -- destroyed a 406x126 band of artwork; and a
    /// chrome badge whose hull is refused by BOTH floors (area 0.00434, ink
    /// ~0.0), pinned separately so its safety stops being an accident of size.
    /// NOT-FIRE: every measured keeper down to the tightest, at 0.260.
    #[test]
    fn a_low_ink_mint_is_refused_and_the_keepers_survive() {
        let hud = denormalize_spot_box([668.0, 648.0, 998.0, 755.0], 1200, 1045).unwrap();
        assert!(
            !spot_rescue_candidate(hud, (1200, 1045), &[], 0.00996),
            "the HUD chrome clears area (0.0353) and must die on the ink floor"
        );
        // The badge's real hull (165x33 at ~mid-page) dies on area...
        let badge = (520.0, 830.0, 165.0, 33.0);
        assert!(
            !spot_rescue_candidate(badge, (1200, 1045), &[], 0.0),
            "the badge's hull is 0.4% of the page: the area floor refuses it"
        );
        // ...and an area-clearing variant of it still dies on ink, which is
        // the rule that makes its safety a property instead of an accident.
        let badge_grown = (420.0, 780.0, 330.0, 132.0);
        assert!(
            !spot_rescue_candidate(badge_grown, (1200, 1045), &[], 0.0),
            "an area-clearing chrome box with no ink must still be refused"
        );
        // The keepers: a hangul column at 0.260 (the tightest), then 0.323, a
        // title column at 0.329, 0.374, 0.413, 0.450, a banner at 0.456, and
        // 아아아 at 0.678.
        for ink in [0.260, 0.323, 0.329, 0.374, 0.413, 0.450, 0.456, 0.678] {
            assert!(
                spot_rescue_candidate(hud, (1200, 1045), &[], ink),
                "a genuine mint at ink {ink} must survive the floor"
            );
        }
    }

    /// The illegible floor on the measured population, both poles from the
    /// same chapter and run. FIRE: every lettered member of the leak class --
    /// `选` at 0.167 ("SELECT" over a drawn `轰`), the seam's `选` at 0.194,
    /// and three short name or romanized reads at 0.334, 0.440 and 0.456 --
    /// plus the second chapter's 0.5228 leak. NOT-FIRE: the chapter's lowest correct free-text read
    /// (0.7308); the lowest correct DIALOGUE read (0.5043) and a sub-floor
    /// dialogue score, because a balloon is never in scope however low it
    /// scores; the mislabelled cut balloon (0.5417, free-text label on a real
    /// balloon -- above the floor by design); and `None`, because NOT MEASURED
    /// is never zero -- manga-ocr's whole arm reports `None` and must be
    /// untouched.
    #[test]
    fn the_illegible_floor_refuses_the_leak_class_and_nothing_real() {
        for (confidence, role, label) in [
            (0.16704, "onomatopoeia", "SELECT on the vortex"),
            (0.19449939, "free-text", "a seam's SELECT over brush art"),
            (0.3338, "free-text", "a three-letter romanization"),
            (0.4400, "onomatopoeia", "a two-glyph name with a bang"),
            (0.4561, "free-text", "a two-glyph romanized name"),
            (0.5227619, "onomatopoeia", "a romanized two-word name, the leak 0.50 shipped"),
        ] {
            assert!(
                illegible_read(Some(confidence), Some(role)),
                "{label} must be refused by the floor"
            );
        }
        for (confidence, role, label) in [
            (Some(0.7308), "free-text", "the lowest correct free-text read, both chapters"),
            (Some(0.6642), "free-text", "a meta-answer dies by CLASS, not here"),
            (Some(0.5043), "dialogue", "the lowest correct dialogue read"),
            (Some(0.16), "dialogue", "a sub-floor DIALOGUE read is out of scope"),
            (None, "free-text", "an unscored engine never fires the floor"),
        ] {
            assert!(
                !illegible_read(confidence, Some(role)),
                "{label} must not be refused"
            );
        }
    }

    /// `spot_ink_fraction` measures what `spot_typography`'s mask measures:
    /// distance from the box's own median colour. A flat box is 0.0 ink; a
    /// box with a tenth of its rows in far-from-median black is 0.10; a box
    /// clamped off the canvas is None, which the caller refuses.
    #[test]
    fn spot_ink_fraction_measures_the_typography_mask() {
        let mut image = image::RgbImage::from_pixel(100, 100, image::Rgb([200, 200, 200]));
        assert_eq!(
            spot_ink_fraction(&image, (0.0, 0.0, 100.0, 100.0)),
            Some(0.0),
            "a flat box has no ink"
        );
        for y in 0..10 {
            for x in 0..100 {
                image.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        let ink = spot_ink_fraction(&image, (0.0, 0.0, 100.0, 100.0)).unwrap();
        assert!(
            (ink - 0.10).abs() < 1e-9,
            "ten black rows of a hundred are 0.10 ink, got {ink}"
        );
        assert_eq!(
            spot_ink_fraction(&image, (200.0, 200.0, 50.0, 50.0)),
            None,
            "a box clamped off the canvas has no measurement"
        );
    }

    /// The composite span gate on the measured numbers: one seam's mint
    /// (y 23.2, h 654.8, the whole cut column) crosses that band's cut at 134
    /// and ships; the next seam's mint (y 105, h 434 -- the
    /// same column's tail seen from the NEXT band, cut at 972) crosses
    /// nothing and must not -- rendered, the two paint-backs stomped each
    /// other. A box merely touching a cut from one side does not span it.
    #[test]
    fn a_composite_mint_must_span_a_cut() {
        assert!(
            mint_spans_a_cut((141.6, 23.226, 136.8, 654.752), &[134.0]),
            "the whole cut column crosses its band's cut and ships"
        );
        assert!(
            !mint_spans_a_cut((138.0, 105.0, 144.0, 434.0), &[972.0]),
            "the neighbour band's fragment view crosses nothing"
        );
        assert!(
            !mint_spans_a_cut((100.0, 100.0, 100.0, 100.0), &[200.0]),
            "a box whose bottom TOUCHES the cut still lies in one slice"
        );
        assert!(
            !mint_spans_a_cut((100.0, 100.0, 100.0, 100.0), &[]),
            "no cuts, nothing spanned -- the caller gates on non-empty first"
        );
        assert!(
            mint_spans_a_cut((100.0, 100.0, 100.0, 400.0), &[972.0, 150.0]),
            "any one cut of a multi-slice run is enough"
        );
    }

    /// One measured page's third spot box duplicates the already-lettered effect; the
    /// overlap fence -- intersection over the SMALLER box, so containment in
    /// either direction counts -- is load-bearing, not hygiene.
    #[test]
    fn spot_box_overlapping_a_lettered_region_is_dropped() {
        let crossed = denormalize_spot_box([115.0, 394.0, 462.0, 524.0], 1200, 1041).unwrap();
        // an existing lettered region covering most of the same ink
        let existing = [(150.0, 400.0, 380.0, 130.0)];
        assert!(
            !spot_rescue_candidate(crossed, (1200, 1041), &existing, 0.323),
            "a box duplicating an existing region must be dropped"
        );
        assert!(
            spot_rescue_candidate(crossed, (1200, 1041), &[], 0.323),
            "the same box on an empty page is a legitimate candidate"
        );
    }

    /// `upright_select` ships a banner from the PRODUCTION grid -- the measured
    /// sweep's angles and scores, coarse thirty-degree angles plus the measured
    /// refine rows around the best coarse. The measured read matched the
    /// official edition's lettering, so the shipped read is verified truth, not
    /// just the confident pick. The strings are invented stand-ins under one
    /// consistent glyph-for-glyph substitution, so every equality and
    /// containment between rows -- all the selection rule sees -- is preserved.
    #[test]
    fn upright_select_ships_a_banner_from_the_production_grid() {
        let rows: Vec<UprightRow> = [
            (0.0, "甲乙丙丁", Some(-0.3689)),
            (30.0, "甲戊己庚", Some(-1.1682)),
            (60.0, "！辛壬丁", Some(-0.7510)),
            (90.0, "别过来！", Some(-0.0580)),
            (120.0, "别过来！", Some(-0.0203)),
            (150.0, "癸过来！", Some(-0.1068)),
            (180.0, "山水来...", Some(-1.1234)),
            (210.0, "癸火木甲", Some(-0.5213)),
            (240.0, "癸金土！", Some(-0.3995)),
            (270.0, "日过来！", Some(-0.0977)),
            (300.0, "别过来！", Some(-0.0459)),
            (330.0, "甲辰江！", Some(-0.8577)),
            // measured refine rows around the best coarse angle
            (105.0, "别过来！", Some(-0.0107)),
            (135.0, "河过来！", Some(-0.1189)),
        ]
        .into_iter()
        .map(|(angle, text, mlp)| UprightRow {
            angle,
            text: text.to_string(),
            mlp,
        })
        .collect();
        let pick = upright_select(&rows).expect("a corroborated 4-glyph read must ship");
        assert_eq!(rows[pick].text, "别过来！");
    }

    /// Same, for a top display row -- a crossed-out formation name the official
    /// edition letters with the authored strike-through kept. Invented stand-ins
    /// under the same kind of consistent substitution.
    #[test]
    fn upright_select_ships_a_crossed_out_top_row() {
        let rows: Vec<UprightRow> = [
            (0.0, "甲乙丙丁戊己", Some(-1.0923)),
            (30.0, "庚辛庚壬庚壬", Some(-1.1969)),
            (60.0, "青云九宫星阵", Some(-0.0796)),
            (90.0, "青云九癸山阵", Some(-0.4573)),
            (120.0, "水火水木金土", Some(-0.7074)),
            (150.0, "日火辰江河湖", Some(-1.2622)),
            (180.0, "海火松竹梅兰", Some(-0.8308)),
            (210.0, "菊荷火 桥火 火 火", Some(-1.4061)),
            (240.0, "楼台亭阁 书台琴棋", Some(-0.7502)),
            (270.0, "画茶舟车马牛", Some(-1.2763)),
            (300.0, "羊鸡丙犬鱼己", Some(-0.6869)),
            (330.0, "庚鸟庚虫花草", Some(-0.8439)),
            (45.0, "叶根叶枝果雪", Some(-1.1283)),
            (75.0, "青云九宫星阵", Some(-0.0076)),
        ]
        .into_iter()
        .map(|(angle, text, mlp)| UprightRow {
            angle,
            text: text.to_string(),
            mlp,
        })
        .collect();
        let pick = upright_select(&rows).expect("a corroborated 6-glyph read must ship");
        assert_eq!(rows[pick].text, "青云九宫星阵");
    }

    /// The sidecar's norm1000-against-the-original convention, on the exact
    /// banner numbers.
    #[test]
    fn denormalize_spot_box_lands_on_the_banner() {
        let (x, y, w, h) =
            denormalize_spot_box([135.0, 544.0, 845.0, 890.0], 1200, 1041).unwrap();
        assert!((x - 162.0).abs() < 0.5, "{x}");
        assert!((y - 566.3).abs() < 0.5, "{y}");
        assert!((w - 852.0).abs() < 0.5, "{w}");
        assert!((h - 360.2).abs() < 0.5, "{h}");
        assert!(denormalize_spot_box([500.0, 500.0, 400.0, 600.0], 1200, 1041).is_none());
    }

    /// The stash path's erase gate still refuses the banner-sized box: the spot
    /// rescue's erase is its OWN flag, and this pins that nothing turned the
    /// big-box erase on silently -- a hull erase of a wide box destroys the art
    /// under it.
    #[test]
    fn rescue_narrow_still_refuses_the_spot_banner() {
        assert!(
            !rescue_narrow((162.0, 566.0, 1014.0, 927.0)),
            "361 px short side is over the 256 bound and must stay refused"
        );
    }

    /// The content gate on the reads that measured it: the exhibit's scream
    /// fires; every stored non-scream mint read refuses, including the ones a
    /// looser rule would restyle by accident.
    #[test]
    fn scream_read_accepts_repeated_glyph_screams_and_refuses_everything_else() {
        assert!(scream_read("아아아"), "the test exhibit");
        assert!(scream_read("AAAH!"), "an English scream is the same class");
        assert!(scream_read("ギャアアア"), "a kana scream with a lead-in glyph");
        assert!(!scream_read("숙..."), "one glyph plus ellipsis is not a run");
        assert!(!scream_read("（众）"), "a narrow mint the official edition letters (Crowd)");
        assert!(!scream_read("HUD INTERFACE"), "a wide display mint that must not restyle");
        assert!(!scream_read("别过来！"), "three distinct glyphs are dialogue");
        assert!(!scream_read("三重防御"), "a skill name is not a scream");
        assert!(!scream_read("Wave"), "a lettered SFX word stays as lettered");
        assert!(!scream_read(""), "empty never fires");
    }

    /// The COMPOSED scream predicate, asserted exactly as the mint calls it
    /// -- a test of the two halves does not test the `&&` between them. The
    /// three mints with stored wire geometry are the fixture: the wide scream
    /// fires, and BOTH narrow mints refuse whatever their read.
    #[test]
    fn scream_mark_is_the_composed_predicate_and_only_the_exhibit_fires() {
        // The exhibit's 아아아: hull x251.16 y587.2, 418.83x547.2 -- wide AND a scream.
        assert!(scream_mark((251.16, 587.2, 669.99, 1134.4), "아아아"));
        // A 숙... mint: 271.17x227.2 -- narrow, refused before the read matters.
        assert!(!scream_mark((373.29, 0.0, 644.46, 227.2), "숙..."));
        // A （众） mint: 483.6x190.68 -- narrow, and the read refuses too.
        assert!(!scream_mark((362.4, 602.0, 846.0, 792.68), "（众）"));
        // A wide box with a non-scream read: geometry alone must not fire.
        assert!(!scream_mark((0.0, 0.0, 500.0, 500.0), "HUD INTERFACE"));
        // A narrow box with a scream read: the read alone must not fire.
        assert!(!scream_mark((0.0, 0.0, 100.0, 500.0), "아아아"));
    }

    /// The topological ink rule on a synthetic multi-tone window shaped like
    /// the exhibit: glyph-scale dark components fully enclosed by bright are
    /// claimed; border-touching dark (panel, outline ends) and sub-glyph
    /// specks are spared; the sampled ramp and axis read off the drawn ink.
    #[test]
    fn scream_mark_ink_claims_enclosed_glyphs_and_spares_border_art() {
        let mut image = RgbImage::from_pixel(600, 700, image::Rgb([230, 228, 226]));
        // A dark stripe entering from the left border -- open, must survive.
        for y in 300..340 {
            for x in 0..200 {
                image.put_pixel(x, y, image::Rgb([40, 20, 20]));
            }
        }
        // Three 48x48 enclosed glyph blocks stepping down-right, bright head
        // to near-black tail.
        for (index, colour) in [[190, 40, 40], [110, 40, 40], [45, 44, 44]].iter().enumerate() {
            let (gx, gy) = (150 + index as u32 * 120, 120 + index as u32 * 160);
            for y in gy..gy + 48 {
                for x in gx..gx + 48 {
                    image.put_pixel(x, y, image::Rgb(*colour));
                }
            }
        }
        // A 6x6 enclosed speck -- under the component floor, must be spared.
        for y in 600..606 {
            for x in 500..506 {
                image.put_pixel(x, y, image::Rgb([30, 30, 30]));
            }
        }
        let mark = scream_mark_ink(&image, (60.0, 60.0, 480.0, 560.0))
            .expect("three glyph-scale components clear both floors");
        // The sealing pass's diamond kernel may shave a few corner pixels off
        // a square block; the claim is "the three glyphs and nothing else",
        // not corner-exact geometry.
        assert!(
            mark.ink_px <= 3 * 48 * 48 && mark.ink_px >= 3 * 48 * 48 - 400,
            "the three glyphs and nothing else, got {}",
            mark.ink_px
        );
        let claimed = |x: u32, y: u32| {
            let (ox, oy) = mark.origin;
            x >= ox
                && y >= oy
                && x - ox < mark.mask.width()
                && y - oy < mark.mask.height()
                && mark.mask.get_pixel(x - ox, y - oy).0[0] != 0
        };
        assert!(claimed(174, 144), "the head glyph is erased");
        assert!(claimed(438, 490), "the tail glyph is erased");
        assert!(!claimed(100, 320), "the border-touching stripe is art, spared");
        assert!(!claimed(502, 602), "the speck is under the floor, spared");
        assert!(
            mark.head[0] > mark.tail[0] + 50,
            "the ramp's bright end is the head: {:?} vs {:?}",
            mark.head,
            mark.tail
        );
        assert!(
            (mark.angle_degrees - 45.0).abs() < 15.0,
            "a down-right cascade reads as a positive axis, got {}",
            mark.angle_degrees
        );
        assert!(
            mark.halo.iter().all(|&channel| channel > 180),
            "the halo samples the bright surround, got {:?}",
            mark.halo
        );
        // The flat-paint subset: the paper ring around the glyphs is claimed
        // for direct painting, the glyph cores are not (they go to the model),
        // and the sampled paper is the field colour.
        assert!(
            mark.paper
                .iter()
                .zip([230_u8, 228, 226])
                .all(|(&got, expected)| got.abs_diff(expected) <= 2),
            "paper is the bright low-spread field, got {:?}",
            mark.paper
        );
        let flat_at = |x: u32, y: u32| {
            let (ox, oy) = mark.origin;
            mark.flat.get_pixel(x - ox, y - oy).0[0] != 0
        };
        assert!(
            flat_at(174, 100),
            "the paper ring above the head glyph is painted flat"
        );
        assert!(
            !flat_at(174, 144),
            "the glyph core goes to the model, never the bucket"
        );
        assert!(
            mark.flat
                .as_raw()
                .iter()
                .zip(mark.mask.as_raw())
                .all(|(&flat, &mask)| flat == 0 || mask != 0),
            "flat is a subset of the erase mask"
        );
    }

    /// A five-glyph diagonal cascade whose window covers only the last two:
    /// the third glyph crosses the window's top-left corner, and the growth
    /// steps over it glyph by glyph until the whole mark is enclosed ink.
    fn cascade_image() -> RgbImage {
        let mut image = RgbImage::from_pixel(700, 1000, image::Rgb([230, 228, 226]));
        // Steps larger than the block, so consecutive glyphs are SEPARATE
        // components -- connected glyphs would chain to the border and open
        // the whole mark, which is a different page, not this fixture.
        for step in 0..5_u32 {
            let (gx, gy) = (100 + step * 80, 100 + step * 70);
            for y in gy..gy + 70 {
                for x in gx..gx + 70 {
                    image.put_pixel(x, y, image::Rgb([70, 35, 35]));
                }
            }
        }
        image
    }

    #[test]
    fn grow_scream_mark_walks_the_cascade_to_its_whole_extent() {
        let image = cascade_image();
        let (grown, hull, ink) = grow_scream_mark(&image, (340.0, 310.0, 150.0, 140.0), &[], &[])
            .expect("the visible tail clears both floors");
        assert!(grown, "the cropped third glyph is the evidence that grows the window");
        assert!(
            (hull.0 - 100.0).abs() <= 1.5
                && (hull.1 - 100.0).abs() <= 1.5
                && (hull.0 + hull.2 - 490.0).abs() <= 1.5
                && (hull.1 + hull.3 - 450.0).abs() <= 1.5,
            "the hull is the whole cascade's ink, got {hull:?}"
        );
        assert!(
            ink.ink_px <= 5 * 70 * 70 && ink.ink_px >= 5 * 70 * 70 - 400,
            "all five glyphs and nothing else (corner shaving tolerated), got {}",
            ink.ink_px
        );
    }

    #[test]
    fn grow_scream_mark_never_grows_toward_an_existing_region() {
        let mut image = cascade_image();
        // A neighbour's glyph pressed against the initial window's right
        // border -- inside an existing region box, so neither evidence nor ink.
        for y in 340..410 {
            for x in 500..570 {
                image.put_pixel(x, y, image::Rgb([40, 40, 40]));
            }
        }
        let existing = [(495.0, 335.0, 80.0, 80.0)];
        let (grown, hull, ink) =
            grow_scream_mark(&image, (340.0, 310.0, 150.0, 140.0), &existing, &[])
                .expect("the cascade still clears both floors");
        assert!(grown, "growth up the cascade is unaffected");
        assert!(
            hull.0 + hull.2 <= 491.5,
            "the hull never reaches the neighbour, got {hull:?}"
        );
        assert!(
            ink.ink_px <= 5 * 70 * 70 && ink.ink_px >= 5 * 70 * 70 - 400,
            "the neighbour's ink is never claimed, got {}",
            ink.ink_px
        );
    }

    #[test]
    fn grow_scream_mark_is_the_identity_on_a_whole_mark() {
        let image = cascade_image();
        let bounds = (80.0, 80.0, 440.0, 400.0);
        let (grown, hull, _) = grow_scream_mark(&image, bounds, &[], &[])
            .expect("the whole cascade clears both floors");
        assert!(!grown, "nothing crosses the border, so nothing grows");
        assert_eq!(hull, bounds, "the caller's own bounds, byte-identical");
    }

    /// The band geometry: the exhibit's grown extent minus its validated box
    /// is dominated by the head band ABOVE it, and reading order says a
    /// Before band prepends.
    #[test]
    fn scream_head_band_picks_the_exhibits_head_and_its_reading_order() {
        let grown = (195.0, 100.0, 475.0, 1034.4);
        let validated = (251.16, 587.2, 418.83, 547.2);
        let (band, side) = scream_head_band(grown, validated, 24.0)
            .expect("the head band clears the side floor");
        assert_eq!(side, ScreamBandSide::Before, "above the validated box: prepend");
        assert!(
            (band.1 - 100.0).abs() < 1.0 && (band.1 + band.3 - 587.2).abs() < 1.0,
            "the band is the extent above the validated box, got {band:?}"
        );
        assert!(
            scream_head_band(grown, grown, 24.0).is_none(),
            "no growth, no band"
        );
    }

    /// The composition gate, named as the mint calls it: belt verdict, a
    /// plausible band size, and the COMPOSITION still being a scream -- which
    /// is what retired the count trap ("가가가가 for 아아아"): containment of
    /// the validated read holds by construction, and a band that swamps the
    /// composition with foreign glyphs fails the scream test.
    #[test]
    fn scream_compose_band_composes_in_reading_order_and_refuses_the_traps() {
        assert_eq!(
            scream_compose_band("크아", "아아아", ScreamBandSide::Before, false).as_deref(),
            Some("크아아아아"),
            "the exhibit's head prepends"
        );
        assert_eq!(
            scream_compose_band("악", "아아아", ScreamBandSide::After, false).as_deref(),
            Some("아아아악"),
            "a tail-out appends"
        );
        assert!(
            scream_compose_band("크아", "아아아", ScreamBandSide::Before, true).is_none(),
            "a belt-withdrawn band never composes"
        );
        assert!(
            scream_compose_band("", "아아아", ScreamBandSide::Before, false).is_none(),
            "an empty band adds nothing"
        );
        assert!(
            scream_compose_band(
                "가나다라마바사아자차",
                "아아아",
                ScreamBandSide::Before,
                false
            )
            .is_none(),
            "a rambling band is over the glyph cap"
        );
        assert!(
            scream_compose_band("가나다라", "아아아", ScreamBandSide::Before, false)
                .is_none(),
            "a band that swamps the scream shape is refused"
        );
    }

    /// A window that EXPANDED chasing open evidence and found no new enclosed
    /// ink must ship the tail-only device exactly -- re-measured on the
    /// caller's own bounds, never the grown window's samples and mask origin.
    #[test]
    fn grow_scream_mark_re_measures_on_the_original_bounds_when_growth_finds_nothing() {
        // ONLY the two tail glyphs -- reusing the full cascade would grow
        // genuinely and test the wrong arm.
        let mut image = RgbImage::from_pixel(700, 1000, image::Rgb([230, 228, 226]));
        for (gx, gy) in [(340_u32, 310_u32), (420, 380)] {
            for y in gy..gy + 70 {
                for x in gx..gx + 70 {
                    image.put_pixel(x, y, image::Rgb([70, 35, 35]));
                }
            }
        }
        // An open runner crossing the initial window's top border and running
        // to the page edge: candidate evidence every step, enclosed never.
        for y in 0..300 {
            for x in 420..460 {
                image.put_pixel(x, y, image::Rgb([50, 30, 30]));
            }
        }
        let bounds = (340.0, 310.0, 150.0, 140.0);
        let (grown, hull, ink) = grow_scream_mark(&image, bounds, &[], &[])
            .expect("the two enclosed glyphs clear both floors");
        assert!(!grown, "the runner never becomes ink, so nothing is grown");
        assert_eq!(hull, bounds, "the caller's own bounds");
        assert_eq!(
            ink.origin,
            (308, 278),
            "the ink is re-measured in the ORIGINAL window, not the expanded one"
        );
        assert!(
            ink.ink_px <= 2 * 70 * 70 && ink.ink_px >= 2 * 70 * 70 - 200,
            "exactly the two enclosed glyphs, got {}",
            ink.ink_px
        );
    }

    /// The ARTWORK fence: enclosed dark inside a settled panel is a
    /// character's face, not a glyph -- growth must neither adopt it as ink
    /// nor chase it as evidence -- while the rescue's own validated box stays
    /// exempt, because a scream's tail legitimately lives ON the panel.
    #[test]
    fn the_artwork_fence_keeps_growth_off_the_face_and_spares_the_validated_tail() {
        let mut image = cascade_image();
        // A face feature: glyph-scale dark enclosed by bright skin, sitting
        // inside the panel box, up-left of the cascade where growth walks.
        for y in 120..170 {
            for x in 250..310 {
                image.put_pixel(x, y, image::Rgb([35, 30, 30]));
            }
        }
        let bounds = (340.0, 310.0, 150.0, 140.0);
        // The panel covers the feature AND the cascade's tail glyphs; the
        // validated box exemption is what keeps the tail claimable.
        let panel = [(255.0, 110.0, 70.0, 70.0), (330.0, 300.0, 170.0, 160.0)];
        let (grown, hull, ink) = grow_scream_mark(&image, bounds, &[], &panel)
            .expect("the cascade still clears both floors");
        assert!(grown, "growth up the cascade still happens on paper");
        assert!(
            ink.ink_px < 5 * 70 * 70 + 1_000,
            "the face feature is never adopted as ink, got {}",
            ink.ink_px
        );
        assert!(
            hull.1 >= 95.0,
            "the hull covers the cascade but is not dragged by the fenced feature: {hull:?}"
        );
        // And the validated exemption: the tail glyphs sit inside the second
        // panel box yet stay ink.
        assert!(
            ink.ink_px >= 5 * 70 * 70 - 3_000,
            "the tail on the panel is exempt through the validated box, got {}",
            ink.ink_px
        );
    }

    /// The existing-region fence reaches the INK, not only the growth: the
    /// plain wrapper (no fence) claims a neighbour's enclosed glyph that the
    /// fenced analysis spares -- the difference the mint's fallback arm relies
    /// on when it passes `existing` explicitly.
    #[test]
    fn the_existing_fence_spares_a_neighbours_enclosed_glyphs() {
        let mut image = cascade_image();
        for y in 340..410 {
            for x in 500..570 {
                image.put_pixel(x, y, image::Rgb([40, 40, 40]));
            }
        }
        let bounds = (340.0, 310.0, 230.0, 140.0);
        let unfenced = scream_mark_ink(&image, bounds).expect("everything clears the floors");
        let fenced = super::scream_ink_analysis(
            &image,
            bounds,
            &[(495.0, 335.0, 80.0, 80.0)],
            &[],
            bounds,
        )
            .expect("the cascade glyphs still clear the floors")
            .ink;
        assert!(
            unfenced.ink_px >= fenced.ink_px + 4_000,
            "the fence spares the neighbour: unfenced {} vs fenced {}",
            unfenced.ink_px,
            fenced.ink_px
        );
    }

    /// An off-centre ink centroid slides the band INWARD: the rotated band's
    /// own hull must stay inside the mint's hull wherever the ink sits.
    #[test]
    fn scream_band_cell_clamps_an_off_centre_centroid_into_the_hull() {
        let mark = ScreamMarkInk {
            mask: GrayImage::new(1, 1),
            flat: GrayImage::new(1, 1),
            paper: [242, 240, 240],
            origin: (0, 0),
            head: [78, 32, 31],
            tail: [43, 42, 43],
            halo: [241, 238, 238],
            angle_degrees: 40.0,
            centroid: (10.0, 10.0),
            ink_px: 40_000,
        };
        let bounds = (0.0, 0.0, 400.0, 600.0);
        let [left, top, right, bottom] = scream_band_cell(bounds, &mark);
        let (cx, cy) = (f64::from(left + right) * 0.5, f64::from(top + bottom) * 0.5);
        let (width, height) = (f64::from(right - left), f64::from(bottom - top));
        let (sin, cos) = f64::from(mark.angle_degrees).to_radians().sin_cos();
        let half_w = (width * cos.abs() + height * sin.abs()) * 0.5;
        let half_h = (width * sin.abs() + height * cos.abs()) * 0.5;
        assert!(
            cx - half_w >= -1.0
                && cx + half_w <= 401.0
                && cy - half_h >= -1.0
                && cy + half_h <= 601.0,
            "the rotated band stays inside the hull: centre ({cx:.1},{cy:.1}), \
             half extents ({half_w:.1},{half_h:.1})"
        );
    }

    /// The device abstains -- `None`, shipping today's behaviour -- when the
    /// window holds no glyph-scale enclosed ink at all.
    #[test]
    fn scream_mark_ink_abstains_on_a_window_with_no_enclosed_ink() {
        let image = RgbImage::from_pixel(400, 400, image::Rgb([235, 233, 231]));
        assert!(scream_mark_ink(&image, (50.0, 50.0, 300.0, 300.0)).is_none());
    }

    /// The band inscribes: its rotated hull stays inside the mint's hull --
    /// a hull is not a box, and the inscribing is what keeps the replacement
    /// where the erased mark was.
    #[test]
    fn scream_band_cell_inscribes_the_rotated_band_in_the_hull() {
        let mark = ScreamMarkInk {
            mask: GrayImage::new(1, 1),
            flat: GrayImage::new(1, 1),
            paper: [242, 240, 240],
            origin: (0, 0),
            head: [78, 32, 31],
            tail: [43, 42, 43],
            halo: [241, 238, 238],
            angle_degrees: 42.7,
            centroid: (460.0, 860.0),
            ink_px: 40_000,
        };
        let bounds = (251.16, 587.2, 418.83, 547.2);
        let [left, top, right, bottom] = scream_band_cell(bounds, &mark);
        let (width, height) = (f64::from(right - left), f64::from(bottom - top));
        let (sin, cos) = f64::from(mark.angle_degrees).to_radians().sin_cos();
        let hull_w = width * cos.abs() + height * sin.abs();
        let hull_h = width * sin.abs() + height * cos.abs();
        assert!(
            hull_w <= bounds.2 + 1.0 && hull_h <= bounds.3 + 1.0,
            "rotated hull {hull_w:.1}x{hull_h:.1} must fit {}x{}",
            bounds.2,
            bounds.3
        );
        assert!(width > height, "the band is a line of text, longer than tall");
    }

    /// `OcrAnalysis::validate` fails the WHOLE PAGE on a bad confidence, so the
    /// guard degrades a transport defect to "not measured" and touches nothing
    /// in range.
    #[test]
    fn wire_confidence_drops_only_what_the_schema_would_fail_the_page_on() {
        assert_eq!(wire_confidence(Some(f32::NAN)), None);
        assert_eq!(wire_confidence(Some(f32::INFINITY)), None);
        assert_eq!(wire_confidence(Some(-0.001)), None);
        assert_eq!(wire_confidence(Some(1.5)), None);
        assert_eq!(wire_confidence(Some(0.0)), Some(0.0));
        assert_eq!(wire_confidence(Some(1.0)), Some(1.0));
        assert_eq!(wire_confidence(Some(0.73)), Some(0.73));
        assert_eq!(wire_confidence(None), None);
    }

    /// The control population pays NOTHING. Ordinary vertical Japanese carries
    /// kana, so the gate never fires and the model is asked exactly once -- which
    /// is the whole cost argument for the flag.
    #[tokio::test]
    async fn an_upright_japanese_column_is_never_read_twice() {
        let reads = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&reads);
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(196, 766), true)],
            text_only(move |(), _: &DynamicImage| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok("僕たちは、放課後に、図書室の前で".to_string())
            }),
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "僕たちは、放課後に、図書室の前で");
        assert_eq!(reads.load(Ordering::SeqCst), 1, "kana must cost one read");
    }

    /// A target the flag did not elect is untouched however sideways it looks.
    /// This is what makes the flag OFF byte-identical to the old behaviour.
    #[tokio::test]
    async fn an_ineligible_target_is_read_once_even_when_kana_free() {
        let reads = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&reads);
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(248, 3157), false)],
            text_only(move |(), _: &DynamicImage| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok("仓炎岛王王·冥雪岛王王".to_string())
            }),
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "仓炎岛王王·冥雪岛王王");
        assert_eq!(reads.load(Ordering::SeqCst), 1);
    }

    /// **THE TEST THAT MATTERS for the LaTeX strip.** It drives `infer_text`, so
    /// it asserts that the strip is WIRED INTO THE CHAIN THE STAGE CALLS -- not
    /// that `strip_latex` works, which the unit tests below cover and which is
    /// exactly what tests of a fix once proved while the fix was entirely
    /// unwired: 128 green, zero behaviour.
    ///
    /// Prove it red by deleting the `strip_latex` call in `infer_text`'s `read`
    /// closure. Only this test and its sibling go red for that; every unit test
    /// below stays green.
    #[tokio::test]
    async fn latex_is_stripped_on_the_path_the_stage_actually_calls() {
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(196, 766), false)],
            text_only(move |(), _: &DynamicImage| Ok("\\(\\underline{\\text{the}}\\)".to_string())),
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            results[0].text, "the",
            "the word survives; only the markup around it goes"
        );
    }

    /// And on the TURNED read too. Both reads share one closure, so a strip that
    /// covered only the upright one would make the two strings incomparable in
    /// `choose_orientation` -- the failure this file's own comment warns is silent
    /// and looks like a good read.
    #[tokio::test]
    async fn the_turned_read_is_stripped_on_the_same_chain() {
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(248, 3157), true)],
            text_only(move |(), image: &DynamicImage| {
                Ok(if turned_ccw(image) {
                    "\\(\\text{苍炎之王}\\)".to_string()
                } else {
                    // Kana-free, so the turn is elected -- mirroring the upright
                    // read the real measurement got from this column.
                    "仓炎岛王王".to_string()
                })
            }),
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "苍炎之王");
    }

    /// The three strings that are actually on disk (the credit line's names and
    /// number are invented stand-ins), and what each must become. Measured over
    /// every run JSON: 6 regions carry LaTeX in the OCR source, 3 distinct, over
    /// 6 run files, 4 of them reaching the page.
    #[test]
    fn the_three_reads_on_disk_keep_every_character_of_content() {
        // Page A -- lettered VERBATIM in one arm and silently
        // cleaned by the translator in the other, which is why this cannot live in
        // the prompt.
        assert_eq!(strip_latex("\\(\\underline{\\text{the}}\\)".into()), "the");

        // Page B -- a fan chat group credit line. The ④ is content and the
        // superscript hat is not; the QQ number must come through untouched.
        assert_eq!(
            strip_latex("云海官方粉丝\\(^{④}\\)群 QQ群号：246813579".into()),
            "云海官方粉丝④群 QQ群号：246813579"
        );

        // Page C -- a blackboard of times tables. This one really IS mathematics, so
        // the arithmetic has to survive as arithmetic: rows separated, `\times`
        // becoming a multiplication sign rather than vanishing into `88=4`.
        assert_eq!(
            strip_latex(
                "\\(\\begin{array}{l} 8\\times8=4 \\\\ 8\\times8=32 \\\\ 9\\times9=40 \\\\ 19\\times19= \\end{array}\\)".into()
            ),
            "8×8=4 8×8=32 9×9=40 19×19="
        );
    }

    /// **The damage set is what this rule is judged on**, so the gate is asserted
    /// on the things it must NOT touch. A backslash is not a LaTeX wrapper: a
    /// drawn emoticon has one, a Windows path has several, and a price sign is a
    /// paired `$`. Every one of these is page text and must come back byte-identical.
    #[test]
    fn a_stray_backslash_or_dollar_is_not_markup() {
        for untouched in [
            "\\(^o^)/",                       // an emoticon, and it opens with `\(`
            "C:\\Users\\someone\\koharu",     // a path on a drawn screen
            "$100 and $200",                  // paired dollars, no control sequence
            "H-N-N-N-N-N-H Boiling pt = 41.82 deg C", // a chemistry read
            "僕たちは、放課後に、図書室の前で",
            "３",
            "．．．．．．",
            "",
        ] {
            assert_eq!(
                strip_latex(untouched.into()),
                untouched,
                "must be left exactly as read"
            );
        }
    }

    /// The emoticon is the sharpest of those, because it defeats the obvious gate.
    /// `\(^o^)/` opens with a literal `\(` -- so the wrapper test alone admits it,
    /// and only the closing-delimiter requirement keeps it out. Pinned separately
    /// so the reason cannot be refactored away.
    #[test]
    fn the_gate_is_narrow_enough_to_refuse_an_emoticon() {
        assert!(!latex_wrapped("$100 and $200"));
        assert!(!latex_wrapped("C:\\Users\\someone"));
        assert!(latex_wrapped("\\(\\underline{\\text{the}}\\)"));
        assert!(latex_wrapped("$\\begin{array}{l}1\\end{array}$"));
    }

    /// A command nobody listed still degrades to its content rather than to markup
    /// on the page. That is what keeps this rule from needing a table that grows
    /// every time the engine surprises us.
    #[test]
    fn an_unlisted_command_keeps_its_argument() {
        assert_eq!(strip_latex("\\(\\mathbb{R}\\)".into()), "R");
        assert_eq!(strip_latex("\\(\\textcolor{blue}\\)".into()), "blue");
    }

    /// A read that is markup and nothing else comes back EMPTY, not as itself.
    /// Empty is already `illegible_text`, so the erase is withdrawn and the drawn
    /// glyphs stay on the page untranslated -- the same disposal
    /// `refuse_degenerate_repetition` argues for. Handing back the original would
    /// letter the markup, which is the defect.
    #[test]
    fn an_all_markup_read_comes_back_empty() {
        assert_eq!(strip_latex("\\(\\)".into()), "");
        assert_eq!(strip_latex("\\(\\quad\\)".into()), "");
    }

    /// A turn that produces nothing must not take the page down with it: the
    /// upright read stands. Without this the gate would trade a wrong string for
    /// an empty one, which reads downstream as "no text here".
    #[tokio::test]
    async fn a_turn_that_reads_nothing_keeps_the_upright_string() {
        let results = infer_text(
            Arc::new(Mutex::new(())),
            vec![target(column(248, 3157), true)],
            text_only(move |(), image: &DynamicImage| {
                Ok(if turned_ccw(image) {
                    String::new()
                } else {
                    "仓炎岛王王·冥雪岛王王".to_string()
                })
            }),
            None::<RotateFn<()>>,
            None,
        )
        .await
        .unwrap();
        assert_eq!(results[0].text, "仓炎岛王王·冥雪岛王王");
    }

    /// The gate, on measured strings only. Kana is the whole of it.
    #[test]
    fn only_a_kana_free_han_read_asks_for_the_turn() {
        assert!(wants_rotated_reread("仓炎岛王王·冥雪岛王王"));
        assert!(wants_rotated_reread("苍炎之王·冥霜之王·裂风之王"));
        // Ordinary vertical Japanese, shaped like manga-ocr's real upright read
        // of a control column. Turning this population would break working pages.
        assert!(!wants_rotated_reread("僕たちは、放課後に、図書室の前で"));
        assert!(!wants_rotated_reread("これは、"));
        // Nothing to turn.
        assert!(!wants_rotated_reread(""));
        assert!(!wants_rotated_reread("   "));
        // Latin-only: a site address carries no Han and asks for nothing.
        assert!(!wants_rotated_reread("www.paperleaf.com"));
        // Interpuncts are punctuation, counted before the kana block -- a row of
        // them is not a kana read. `Scripts::of` documents why the order matters.
        assert!(!wants_rotated_reread("・・・"));
    }

    /// The pick with the ranking OFF, which is what ships: the turned read wins
    /// unless it is empty. Every confidence here is `None` AND the margin is
    /// `None`, because both are what production passes today.
    #[test]
    fn the_turned_read_wins_unless_it_produced_nothing() {
        let cangyan = "\u{82cd}\u{708e}\u{4e4b}\u{738b}";
        let wrong = "\u{4ed3}\u{708e}\u{5c9b}\u{738b}\u{738b}";
        assert_eq!(
            choose_orientation(wrong.to_string(), None, cangyan.to_string(), None, None),
            cangyan
        );
        assert_eq!(
            choose_orientation(wrong.to_string(), None, String::new(), None, None),
            wrong
        );
        assert_eq!(
            choose_orientation(wrong.to_string(), None, "  \n ".to_string(), None, None),
            wrong
        );
    }

    /// **The measured pairs, pinned as data**, over all 179 slices of a test
    /// chapter (invented stand-in strings, measured scores). Every pair was
    /// adjudicated against the DRAWN page and not against a
    /// counter, so the expected value below is a reading of the artwork rather than
    /// a restatement of the arithmetic.
    ///
    /// Asserts on `choose_orientation` -- the function the caller calls -- with both
    /// halves of the new condition driven independently. A test of the margin
    /// comparison alone would stay green with the whole arm deleted from the
    /// function, which is how a fix once shipped unwired with 128 tests passing.
    #[test]
    fn the_margin_recovers_the_upright_read_only_when_it_is_clearly_better() {
        /* (upright, upright conf, rotated, rotated conf, WHAT THE FUNCTION MUST
         * RETURN -- which is NOT always what is right on the page).
         *
         * That distinction is the point of this comment. On p3 the correct
         * reading is the UPRIGHT one and the model is more confident in the wrong
         * turned one, so no margin recovers it and the function correctly returns
         * the wrong string. Pinning the RETURN keeps the assertion honest; the
         * page's own answer lives in `choose_orientation`'s doc comment, where it
         * can disagree with the arithmetic without making a test lie.
         *
         * The multi-line reads are written whole here. An earlier version carried
         * p2's and p3's uprights as single glyphs, because the instrument that
         * read them out of the log stopped at the first physical line. */
        let measured = [
            ("\u{51cc}\u{9704}\u{5883}\u{754c}", 0.8424_f32,
             "\u{9675}\u{9704}\u{5883}\u{754c}", 0.6044_f32,
             "\u{51cc}\u{9704}\u{5883}\u{754c}"),
            // upright correct AND WHOLE; the turn is garbage. The margin FIXES it.
            ("\u{98ce}\u{4e4b}\u{738b}", 0.7209,
             "\u{7a7a}\u{767d}\u{7684}K", 0.1760,
             "\u{98ce}\u{4e4b}\u{738b}"),
            // THE MISS. Upright is RIGHT, the turn is WRONG, and the model prefers
            // the turn -- so the function returns the wrong read, and must keep
            // doing so until something better than confidence ranks them.
            ("\u{7384}\u{51b0}\u{771f}\u{8bc0}", 0.9508,
             "\u{7384}\u{6c34}\u{771f}\u{8bc0}", 0.9906,
             "\u{7384}\u{6c34}\u{771f}\u{8bc0}"),
            ("\u{7384}\u{6c34}\u{771f}\u{8bc0}", 0.8843,
             "\u{7384}\u{51b0}\u{771f}\u{8bc0}", 0.9921,
             "\u{7384}\u{51b0}\u{771f}\u{8bc0}"),
        ];
        // Inside (0.11, 0.23): above the two margins the TURN legitimately wins by,
        // below the two the UPRIGHT wins by. The shipping value, adopted after a
        // render -- see the flag.
        let margin = Some(0.17);
        for (upright, up, rotated, down, better) in measured {
            assert_eq!(
                choose_orientation(
                    upright.to_string(),
                    Some(up),
                    rotated.to_string(),
                    Some(down),
                    margin
                ),
                better,
                "{upright:?}@{up} vs {rotated:?}@{down} must return {better:?}"
            );
            // OFF is the shipping arm and must be untouched by any of this.
            assert_eq!(
                choose_orientation(
                    upright.to_string(),
                    Some(up),
                    rotated.to_string(),
                    Some(down),
                    None
                ),
                rotated,
                "with no margin the turn must still win unconditionally"
            );
        }
    }

    /// The ways the ranking must decline to act, each driven alone.
    ///
    /// The absent-confidence case is load-bearing rather than defensive:
    /// `manga-ocr` and `baberu-ocr` report no confidence at all, so scoring a
    /// missing value as 0.0 would hand every one of their turns to the upright read
    /// on no evidence whatever.
    #[test]
    fn an_absent_confidence_disables_the_ranking_rather_than_scoring_zero() {
        let margin = Some(0.17);
        let feng = "\u{98ce}\u{4e4b}\u{738b}";
        let garbage = "\u{7a7a}\u{767d}\u{7684}K";
        // The upright would win by 0.54 if it had a number -- but it has none.
        assert_eq!(
            choose_orientation(feng.to_string(), None, garbage.to_string(), Some(0.1760), margin),
            garbage
        );
        assert_eq!(
            choose_orientation(feng.to_string(), Some(0.7209), garbage.to_string(), None, margin),
            garbage
        );
        // And the empty test still runs FIRST, ahead of all of it.
        assert_eq!(
            choose_orientation(feng.to_string(), Some(0.0), String::new(), Some(1.0), margin),
            feng
        );
        // A tie leaves the turn alone: the margin is asymmetric on purpose.
        assert_eq!(
            choose_orientation("a".to_string(), Some(0.5), "b".to_string(), Some(0.5), margin),
            "b"
        );
        assert_eq!(
            choose_orientation("a".to_string(), Some(0.5), "b".to_string(), Some(0.5), Some(0.0)),
            "a",
            "a zero margin makes a tie enough, which is why 0.0 is not the OFF value"
        );
        // The OFF value is `inf`, now that the CLI defaults the margin and `None`
        // is unreachable from a flag: no finite gap clears it, so the turn wins
        // unconditionally -- the unranked behaviour, expressed as a number.
        assert_eq!(
            choose_orientation(
                "a".to_string(),
                Some(1.0),
                "b".to_string(),
                Some(0.0),
                Some(f64::INFINITY)
            ),
            "b",
            "an infinite margin must be the off arm even at the maximum possible gap"
        );
    }

    /// Eligibility is a cheap upper bound on the population, not a decision.
    /// Shape cannot tell an upright column from a turned one, so this only has
    /// to keep balloons and squat boxes out.
    #[test]
    fn only_a_tall_free_standing_box_is_eligible() {
        let tall = Geometry::rectangle(44.5, 18.6, 248.4, 3156.4);
        assert!(rotation_candidate(&tall, true));
        // A balloon is a text container; its contents are ordinary set text.
        assert!(!rotation_candidate(&tall, false));
        // The natural control: the same name set HORIZONTALLY, read
        // correctly today at 431x210. Nothing here may touch it.
        assert!(!rotation_candidate(
            &Geometry::rectangle(0.0, 0.0, 431.0, 210.0),
            true
        ));
        // Square-ish, and under the aspect floor either way.
        assert!(!rotation_candidate(
            &Geometry::rectangle(0.0, 0.0, 100.0, 200.0),
            true
        ));
    }

    /// **The erase mask must not move.** Both orientations of this region carry
    /// `最新免费漫画`, so the watermark arm fires either way and the watermark
    /// refusal is untouched by the turned re-read. If this ever diverges, turning a column would
    /// silently start erasing watermark plates again -- the exact damage
    /// `watermark_text` was moved into the pipeline to stop.
    #[test]
    fn turning_a_column_does_not_change_whether_its_pixels_are_spared() {
        let upright = "仓炎岛王王·冥雪岛王王\n最新免费漫画\nwww.papcrleaf.com";
        let turned = "苍炎之王·冥霜之王·裂风之王最新免费漫画 www.paperleef.com";
        assert_eq!(
            withdraw_from_mask(upright, None, true, false, MisreadLevers::OFF),
            withdraw_from_mask(turned, None, true, false, MisreadLevers::OFF)
        );
        assert!(withdraw_from_mask(turned, None, true, false, MisreadLevers::OFF));
    }

    // ---- a watermark condemning the story text beside it ----

    /// The measured region, both ways round. Upright the site's text is on its own
    /// two lines; turned it is glued to the last name with no separator at all,
    /// which is what rules out scoping by line.
    const UPRIGHT_REGION: &str = "仓炎岛王王·冥雪岛王王\n最新免费漫画\nwww.papcrleaf.com";
    const TURNED_REGION: &str = "苍炎之王·冥霜之王·裂风之王最新免费漫画 www.paperleef.com";

    /// **THE POINT OF THE WHOLE ITEM.** The skill name survives its watermark, in
    /// both orientations, and what survives is exactly the reference line.
    #[test]
    fn a_skill_name_survives_the_watermark_stamped_across_it() {
        assert_eq!(story_text(TURNED_REGION), "苍炎之王·冥霜之王·裂风之王");
        assert_eq!(story_text(UPRIGHT_REGION), "仓炎岛王王·冥雪岛王王");
    }

    /// **A LINE RULE WOULD HAVE THROWN THE NAME AWAY.** Turned, the whole read is
    /// one line and every character of it would go. This is the test that records
    /// why the scope is substrings; delete it and the line version gets
    /// re-derived.
    #[test]
    fn scoping_by_line_would_lose_the_turned_read_entirely() {
        let by_line: Vec<&str> = TURNED_REGION
            .lines()
            .filter(|line| !watermark_text(line))
            .collect();
        assert!(
            by_line.is_empty(),
            "the turned read is ONE line, so a line rule keeps nothing"
        );
        assert!(!story_text(TURNED_REGION).is_empty());
    }

    /// Pure furniture is still refused, which is the ordinary case and the great
    /// majority of watermark refusals on disk. Nothing here rescues a plate.
    #[test]
    fn a_region_holding_only_the_sites_own_text_is_still_refused() {
        for furniture in [
            "最新免费漫画 www.paperleaf.com",
            "本漫畫由",
            "最新免费漫画",
            "腾讯动漫",
            "最新免费漫画\nwww.papcrleaf.com",
            "本漫畫由紙葉漫畫收集整理，更多免費漫畫請訪問",
        ] {
            assert!(
                only_site_furniture(furniture, true),
                "{furniture:?} is furniture and must stay refused"
            );
        }
    }

    /// The site-address shape, which is what stops a garbled URL being lettered
    /// onto the artwork beside a rescued name. A literal list cannot do it: the
    /// same plate reads `papcrleaf` one way and `paperleef` the other.
    #[test]
    fn a_mangled_site_address_is_furniture_however_it_was_misread() {
        for address in ["www.papcrleaf.com", "www.paperleef.com", "Pagerleaf.com", "paper-leaf.net"] {
            assert!(site_address(address), "{address:?} is shaped like a site");
        }
        // And it must never fire on story text. CJK cannot satisfy it at all,
        // which makes the false-positive risk on dialogue structurally zero.
        for story in ["苍炎之王", "冥霜之王·裂风之王", "早く行こう！", "Ka-boom!!", "Mr. Smith"] {
            assert!(!site_address(story), "{story:?} is not a site address");
        }
    }

    /// **The off arm is the shipped behaviour, op for op.** Every region that is
    /// refused today is still refused with the flag off, including the one this
    /// item exists to rescue -- so shipping this dark changes nothing.
    #[test]
    fn the_off_arm_refuses_exactly_what_it_refuses_today() {
        for text in [UPRIGHT_REGION, TURNED_REGION, "最新免费漫画", "最新免费漫画 www.paperleaf.com"] {
            assert_eq!(
                only_site_furniture(text, false),
                watermark_text(text),
                "{text:?} must be unchanged with the flag off"
            );
            assert!(only_site_furniture(text, false));
        }
        // Ordinary dialogue is untouched by either arm.
        assert!(!only_site_furniture("早く行こう！", false));
        assert!(!only_site_furniture("早く行こう！", true));
    }

    /// The composed predicate the veto calls, on the region this item is about.
    /// The erase follows the letter: once the name survives, the box is no longer
    /// withdrawn from the mask, and its artwork -- plate included -- is erased.
    /// That is the trade, pinned here so a render can be argued against it.
    #[test]
    fn a_rescued_region_stops_being_spared_from_the_erase() {
        assert!(withdraw_from_mask(TURNED_REGION, None, true, false, MisreadLevers::OFF));
        assert!(!withdraw_from_mask(TURNED_REGION, None, true, true, MisreadLevers::OFF));
        // Pure furniture is spared under both arms; only the mixed box moves.
        assert!(withdraw_from_mask("最新免费漫画", None, true, false, MisreadLevers::OFF));
        assert!(withdraw_from_mask("最新免费漫画", None, true, true, MisreadLevers::OFF));
    }

    /// **What the TRANSLATOR is handed, which is not what the gate decides on.**
    /// Scoping the verdict alone let a joined column through refusal and
    /// then sent the whole raw string to the model, so the page lettered
    /// `BLUE FLAME KING • FROST KING • GALE KING LATEST FREE MANGA WWW.PAPERLEEF.COM`
    /// over its own artwork while the plate was still underneath -- caught in the
    /// render, not here, which is why the string is pinned here now.
    ///
    /// There is no later place to fix it: source and translated line counts do not
    /// correspond, so nothing downstream can subtract the site's half of a reply.
    /// Furniture that reaches the model comes back translated and gets lettered.
    #[test]
    fn what_reaches_the_translator_carries_no_furniture() {
        assert_eq!(story_text(TURNED_REGION), "苍炎之王·冥霜之王·裂风之王");
        assert!(!story_text(TURNED_REGION).contains("最新免费漫画"));
        assert!(!story_text(TURNED_REGION).contains("paperleef"));
        // The shape of the chapter's one genuine loss of story text.
        assert_eq!(story_text("最新免費漫畫\nPagerleaf.com\n雪落城"), "雪落城");
    }

    /// **The refusal is defeated by the plate being read CORRECTLY.**
    ///
    /// `story_text` strips every [`WATERMARKS`] entry before it filters site
    /// addresses. When an entry is itself the head of a domain, a plate reading
    /// `<mark> <mark>.com` loses BOTH marks and leaves the orphan `.com`, which
    /// [`site_address`] cannot recognise: it trims the leading dot, finds `com`,
    /// and `rsplit_once` then returns `None` for want of a second dot. Without
    /// the orphan-tail rule the non-empty residue makes `only_site_furniture`
    /// return false, so the region is translated and lettered.
    ///
    /// A MISREAD plate is refused anyway, because its address survives mark
    /// removal intact and is well-formed. That inversion was the signature:
    /// pages refused on a bad read while others were lettered on a good one.
    ///
    /// Measured over a whole 180-page chapter: **all 45** reads of the generic
    /// watermark phrase were refused, and of the 44 reads of the site's own name
    /// **32 were lettered**. 31 of those 32 were the bare plate and were the
    /// defect; one must NOT move, because it carries genuine story text, and the
    /// test above already pins its shape. The worst of the 31 sent the orphan
    /// `.com` to the translator and got invented dialogue back. The plates below
    /// are built from the generic mark, so the orphan arm is exercised without
    /// naming any site.
    #[test]
    fn a_correctly_read_site_plate_is_still_only_furniture() {
        // The three spacings the recogniser returns for such a plate.
        for plate in [
            "最新免费漫画 最新免费漫画.com",
            "最新免费漫画\n最新免费漫画.com",
            "最新免费漫画最新免费漫画.com",
        ] {
            assert_eq!(story_text(plate), "", "{plate:?} left a residue");
            assert!(only_site_furniture(plate, true), "{plate:?} escaped refusal");
        }
        // The control: a misread address beside the mark is refused today, and
        // must stay refused. If this half ever goes red the fix has overshot.
        assert!(only_site_furniture("最新免费漫画\npaperleef.com", true));
        // The control that matters more -- the skill-name rescue must survive. A
        // skill name sharing a box with the plate is NOT furniture.
        assert!(!only_site_furniture("最新免费漫画 最新免费漫画.com\n苍炎之王", true));
    }

    /// **The refusal is defeated by DECORATION around a correctly-read plate.**
    ///
    /// The erase-side twin of `labels.rs`'s
    /// `a_decorated_site_plate_is_refused_on_the_scoped_arm`. The orphan-tail rule
    /// fixed the neighbouring orphan-`.com` residue and is a real fix; it does not reach
    /// this one. `👇最新免费漫画👇` strips to a residue of two pointing hands, so
    /// `only_site_furniture`'s second half was false and the plate was erased-and-
    /// lettered rather than left alone.
    ///
    /// Asserted at BOTH levels on purpose. `only_site_furniture` is the rule that
    /// changed; `withdraw_from_mask` is what the OCR stage actually calls, and
    /// it is an `||` of three terms. Testing only the rule would not prove the
    /// caller reaches it, and testing only the caller would let `illegible_text`
    /// carry the assertion — a fix once shipped entirely unwired that way, with
    /// 128 tests green.
    #[test]
    fn a_decorated_site_plate_is_still_only_furniture() {
        for plate in [
            "\u{1F447}最新免费漫画\u{1F447}",
            "↓最新免费漫画↓",
            "◇最新免费漫画◇",
            "\u{1F447}最新免费漫画\u{1F447}\nwww.paperleaf.com",
        ] {
            assert_eq!(story_text(plate), "", "{plate:?} left a residue");
            assert!(only_site_furniture(plate, true), "{plate:?} escaped refusal");
            assert!(
                withdraw_from_mask(plate, None, true, true, MisreadLevers::OFF),
                "{plate:?} still reached the erase mask"
            );
        }
        // The rescues, at both levels. Decoration is dropped; script is not.
        assert!(!only_site_furniture("最新免費漫畫\nPagerleaf.com\n雪落城", true));
        assert!(!only_site_furniture(
            "\u{1F447}最新免费漫画\u{1F447}\n这是……！？",
            true
        ));
        // The unscoped arm never consulted the residue and must not start.
        assert!(only_site_furniture("↓最新免费漫画↓", false));
    }

    /// **A FULLWIDTH address is not recognised, so the plate survives.**
    ///
    /// Every character is non-ASCII, so [`site_address`]'s trim eats the whole token
    /// and the `.` test finds nothing to split on. Of the 8 distinct fullwidth
    /// garbles across 93,588 regions in 16,918 run JSONs, exactly **one** folds into
    /// a well-formed address, twice; the plate below is an invented stand-in of
    /// the same shape. **Claim two, not nineteen.**
    #[test]
    fn a_fullwidth_site_address_is_still_only_furniture() {
        assert_eq!(story_text("最新免费漫画ｐａｐｅｒｌｅｆ．ｃｏｎ"), "");
        assert!(only_site_furniture("最新免费漫画ｐａｐｅｒｌｅｆ．ｃｏｎ", true));
        assert!(withdraw_from_mask("最新免费漫画ｐａｐｅｒｌｅｆ．ｃｏｎ", None, true, true, MisreadLevers::OFF));
        // The halfwidth twin was already refused and must stay so.
        assert!(only_site_furniture("最新免费漫画paperlef.con", true));
        // `８ｍ` folds to `8m`, which has no dot, so the fold does not sweep it up
        // and the story text beside it survives.
        assert!(!only_site_furniture("最新免费漫画\n８ｍ\n苍炎之王", true));
    }

    /// **The decoration rule must not reach a region with no watermark in it.**
    ///
    /// [`story_text`] has **two** consumers and only one is the refusal: `targets()`
    /// in `stages/translation.rs` applies it to EVERY region's source before
    /// translation. Ungated, the decoration rule takes punctuation off ordinary
    /// pages — measured at **30 occurrences, 9 distinct, every one NON-watermark**
    /// across 93,588 regions, a Korean bubble
    /// losing its `!` among them. Gated: **0**, refusal unchanged.
    ///
    /// Asserted on `story_text` directly, deliberately: the consumer at risk is the
    /// translator, and `only_site_furniture` cannot see a non-watermark region.
    #[test]
    fn decoration_is_only_dropped_from_a_watermark_region() {
        // Shaped like the real corpus string that exposed this.
        assert_eq!(story_text("冬\n나\n무\n길\n로\n!"), "冬\n나\n무\n길\n로\n!");
        assert_eq!(story_text("1\nf"), "1\nf");
        assert_eq!(story_text("！？"), "！？");
        // With a watermark present the rule still fires.
        assert_eq!(story_text("\u{1F447}最新免费漫画\u{1F447}"), "");
        assert_eq!(
            story_text("\u{1F447}最新免费漫画\u{1F447}\n这是……！？"),
            "这是……！？"
        );
    }

    /// **A TRADITIONAL plate escapes the rule entirely.**
    ///
    /// [`WATERMARKS`] carried `最新免费漫画` and `腾讯动漫` in SIMPLIFIED form only,
    /// while every other CJK entry carried both. `watermark_text` is a literal
    /// `contains`, so a traditional read never reached `story_text`, and
    /// `site_prose` — which would otherwise catch `漫畫` — is only reachable from
    /// inside it. Prophylactic: no test corpus holds either string.
    #[test]
    fn a_traditional_site_plate_is_still_only_furniture() {
        for plate in ["最新免費漫畫", "騰訊動漫", "↓最新免費漫畫↓"] {
            assert!(only_site_furniture(plate, true), "{plate:?} escaped refusal");
        }
    }

    /// Removal is repeated, and it does not eat the text around it. No entry
    /// carries cased letters today, so case folding cannot change a result here.
    #[test]
    fn furniture_is_removed_wherever_it_appears() {
        assert_eq!(story_text("苍炎之王最新免费漫画苍炎之王"), "苍炎之王苍炎之王");
        assert_eq!(story_text("腾讯动漫苍炎之王腾讯动漫"), "苍炎之王");
        assert_eq!(story_text(""), "");
        assert_eq!(story_text("   \n  "), "");
    }

    /// A 400x300 page, so every case below has room to grow on at least one side.
    fn perturb_page() -> DynamicImage {
        DynamicImage::new_rgb8(400, 300)
    }

    fn perturb_grow(px: u32) -> Option<NonZeroU32> {
        NonZeroU32::new(px)
    }

    /// Zero growth is byte-identical to `crop`, which is what makes the flag's OFF
    /// arm provably the behaviour that shipped before it existed.
    /// The wire shape the spotting task actually returns, fenced and bare, plus
    /// the failure shapes that must parse to NOTHING -- the gate fails open.
    #[test]
    fn spot_boxes_parse_the_real_shapes_and_fail_open() {
        let bare = r#"[{"box": [62, 36, 230, 74], "text": "最新"}, {"box": [21, 79, 270, 108], "text": "www"}]"#;
        assert_eq!(parse_spot_boxes(bare).len(), 2);
        let fenced = format!("```json
{bare}
```");
        assert_eq!(parse_spot_boxes(&fenced).len(), 2);
        assert!(parse_spot_boxes("").is_empty());
        assert!(parse_spot_boxes("图片中没有文字。").is_empty());
        assert!(parse_spot_boxes(r#"[{"box": [1, 2, 3], "text": "short"}]"#).is_empty());
    }

    /// The measured chapter separation this gate rests on, as arithmetic: a
    /// region inside a spotted line scores high, a region nowhere near any
    /// spotted line scores exactly zero, and the floor sits between them.
    #[test]
    fn spot_coverage_separates_covered_from_uncovered() {
        // Page 1200x919; one spotted line over x 740..787, y 269..474.
        let boxes = [[617.0, 291.0, 660.0, 520.0]];
        let column = (740.6, 269.2, 787.5, 473.9);
        let covered = spot_coverage(column, &boxes, 1200, 919);
        assert!(covered > SPOT_COVERAGE_FLOOR, "{covered}");
        // A petal box in the page's middle, nowhere near the line: exactly zero.
        let petal = (300.0, 600.0, 360.0, 660.0);
        assert_eq!(spot_coverage(petal, &boxes, 1200, 919), 0.0);
        // No boxes at all: zero, so an empty spotting page refuses every suspect.
        assert_eq!(spot_coverage(column, &[], 1200, 919), 0.0);
    }

    /// Length picks who is CHECKED, never who is refused: the real effects are
    /// inside the suspect band and must rely on coverage alone to survive.
    #[test]
    fn the_suspect_band_holds_the_fabrications_and_the_real_effects_alike() {
        for fabricated in ["我", "李雪", "低等", "小口鱼"] {
            assert!(spot_suspect(fabricated), "{fabricated}");
        }
        for real in ["彭", "彭！", "轰", "逃命"] {
            assert!(spot_suspect(real), "{real}");
        }
        assert!(!spot_suspect(""));
        assert!(!spot_suspect("这是一段完整的对话。"));
    }

    /// The COMPOSED predicate both call sites use -- the halves alone are not the
    /// behaviour. Junk that other gates refuse must not buy a spotting call;
    /// every measured fabrication and every real effect must still be checked;
    /// and the BOM-carrying bare mark -- which slips the punctuation rules in
    /// both copies -- must STAY a candidate, because the gate is what catches it.
    #[test]
    fn junk_reads_do_not_buy_a_spotting_call_and_the_bom_mark_still_does() {
        let zh = Some(Language::ChineseSimplified);
        for junk in ["1", "H", "？", "……", "↗", "1 2 3", "00 0", "."] {
            assert!(!spot_candidate(junk, zh), "{junk}");
        }
        for checked in ["我", "李雪", "低等", "彭", "彭！", "轰", "逃命", "曾"] {
            assert!(spot_candidate(checked, zh), "{checked}");
        }
        assert!(spot_candidate("\u{FEFF}！", zh), "the BOM mark must stay gated");
    }

    /// The all-Latin arm, asserted on the predicate the refusal loop actually
    /// calls -- a real page's geometry, empty spotting boxes, so eligibility is
    /// the only thing under test. Refused at any length on a CJK page; silence
    /// fires nothing; dialogue stays out; a non-CJK source fires nothing.
    #[test]
    fn a_latin_fabrication_is_checked_at_any_length() {
        let geometry = Geometry::rectangle(829.758, 31.494, 234.234, 203.028);
        for language in [
            Language::ChineseSimplified,
            Language::Japanese,
            Language::Korean,
        ] {
            assert!(
                spot_refuses(
                    "Ren Hao",
                    Some(language),
                    Some("free-text"),
                    &geometry,
                    &[],
                    1200,
                    908
                ),
                "{language:?}"
            );
        }
        assert!(!spot_refuses(
            "Ren Hao",
            None,
            Some("free-text"),
            &geometry,
            &[],
            1200,
            908
        ));
        assert!(!spot_refuses(
            "Ren Hao",
            Some(Language::ChineseSimplified),
            Some("dialogue"),
            &geometry,
            &[],
            1200,
            908
        ));
        assert!(!latin_on_cjk("Ren Hao", Some(Language::English)));
    }

    /// Every genuine lettered free-standing read in the measured corpus stays
    /// OUT of the gate. This is the test that makes a raised length ceiling
    /// (rejected candidate B) unavailable: replace the all-Latin arm with
    /// `glyphs <= 12` and the two long display columns here go red, in code,
    /// permanently. Those two rows are invented stand-ins with the measured
    /// glyph counts.
    #[test]
    fn every_real_display_read_stays_out_of_the_gate() {
        let zh = Some(Language::ChineseSimplified);
        for real in [
            "凌霄境界.",
            "裂风之王！",
            "灵光识·苍炎之王",
            "轰隆隆隆---！",
            "炎之王·冥霜之王·",
            "雾起东城，灯落南桥……",
            "各位别再犹豫，快拦住他！",
        ] {
            assert!(!spot_eligible(real, zh, Some("free-text")), "{real}");
        }
        // Mixed-script blocks hold Han, so the all-Latin arm cannot see them.
        assert!(!spot_eligible(
            "万事如意福满门\nNew Year's Eve\n全体员工来拜年",
            zh,
            Some("free-text"),
        ));
    }

    /// The preamble strip through the ONE named chain both call sites use,
    /// remainders judged on their own merits -- a shrink, never a drop.
    #[test]
    fn the_engine_preamble_is_stripped_wherever_a_read_is_cleaned() {
        for (raw, kept) in [
            ("图片中没有文字。", ""),
            ("图片中的文本内容是：", ""),
            ("图片中的文本内容是：☆", "☆"),
            ("图片中的文本内容是：- -", "- -"),
            ("图片中的文本内容是：叶", "叶"),
            ("图片中的文本内容是：Ren Hao", "Ren Hao"),
            ("图片中的文字是：V", "V"),
            // The shorter locative -- the prompt's own wording. The
            // first row is VERBATIM the read that lettered on a test page
            // (note: no trailing 。, which a whole-sentence rule would need).
            ("图中没有文字", ""),
            ("图中没有任何文字", ""),
            ("图中的文本内容是：叶", "叶"),
        ] {
            assert_eq!(clean_read(raw.to_owned()), kept, "{raw}");
        }
        // A read merely CONTAINING a formula away from its start is untouched,
        // for the new prefixes exactly as for the old.
        assert_eq!(
            clean_read("他说图片中没有文字。".to_owned()),
            "他说图片中没有文字。"
        );
        assert_eq!(
            clean_read("看图中没有文字。".to_owned()),
            "看图中没有文字。"
        );
        // The DIAGRAM sense is real page text and deliberately ungated:
        // `文字` is ordinary vocabulary, unlike `文本`.
        assert_eq!(
            clean_read("图中的文字是山门旧训".to_owned()),
            "图中的文字是山门旧训"
        );
    }

    /// The order inside `clean_read` is pinned at the point of the decision:
    /// the strip runs BEFORE the placeholder rule, so a quoted all-placeholder
    /// run is still seen whole and becomes an ellipsis. Swapping the two calls
    /// fails this single assertion.
    #[test]
    fn the_preamble_is_stripped_before_the_placeholder_rule() {
        assert_eq!(clean_read("图片中的文本内容是：□□".to_owned()), "…");
    }

    /// Every stripped remainder measured in the corpus lands in an
    /// existing gate, by one of two convergent paths: illegible (and
    /// `withdraw_from_mask` spares the artwork), or a single Han glyph that is
    /// spot-eligible and coverage-refused on clean artwork. Asserted on the
    /// predicates the veto writer actually calls, so the convergence is a
    /// checked claim rather than a design hope.
    #[test]
    fn a_stripped_remainder_lands_in_an_existing_gate() {
        let zh = Some(Language::ChineseSimplified);
        for illegible in ["", "D", "cē", "- -", "☆", "V", "y"] {
            assert!(withdraw_from_mask(illegible, zh, true, false, MisreadLevers::OFF), "{illegible:?}");
        }
        let geometry = Geometry::rectangle(57.0, 522.0, 323.0, 169.0);
        for han_single in ["叶", "非"] {
            assert!(
                !withdraw_from_mask(han_single, zh, true, false, MisreadLevers::OFF),
                "{han_single}"
            );
            assert!(
                spot_refuses(han_single, zh, Some("free-text"), &geometry, &[], 1200, 908),
                "{han_single}"
            );
        }
    }

    #[test]
    fn the_ungrown_crop_is_the_crop() {
        let source = perturb_page();
        let region = Geometry::rectangle(50.0, 40.0, 120.0, 90.0);
        let plain = crop(&source, &region).unwrap();
        let zero = crop_grown(&source, &region, 0).unwrap();
        assert_eq!(
            (plain.width(), plain.height()),
            (zero.width(), zero.height())
        );
        assert_eq!(plain.to_rgb8().into_raw(), zero.to_rgb8().into_raw());
    }

    /// An interior box grows by `grow_px` on ALL FOUR sides, so each dimension
    /// gains twice the growth.
    #[test]
    fn an_interior_box_grows_on_every_side() {
        let source = perturb_page();
        let region = Geometry::rectangle(50.0, 40.0, 120.0, 90.0);
        let plain = crop(&source, &region).unwrap();
        let grown = crop_grown(&source, &region, 8).unwrap();
        assert_eq!(grown.width(), plain.width() + 16);
        assert_eq!(grown.height(), plain.height() + 16);
    }

    /// **The u32 underflow guard.** Growing `x` by subtracting from the cast u32
    /// would wrap to ~4 billion for any box within `grow_px` of the left edge, and
    /// `crop_imm` returns an EMPTY image rather than failing -- so the bug would
    /// look like a maximally unstable read instead of a crash.
    #[test]
    fn a_box_against_the_origin_does_not_underflow() {
        let source = perturb_page();
        let region = Geometry::rectangle(2.0, 3.0, 60.0, 60.0);
        let grown = crop_grown(&source, &region, 32).unwrap();
        assert!(
            grown.width() > 0 && grown.height() > 0,
            "crop came back empty"
        );
        // Clamped at the origin, so it gains only what was available on the left
        // (2 px) or top (3 px), plus the full growth on the far side.
        assert_eq!(grown.width(), 60 + 2 + 32);
        assert_eq!(grown.height(), 60 + 3 + 32);
    }

    /// **The false-negative guard, and the reason `grown_twin` exists.** A box
    /// already covering the whole page cannot grow in any direction, so the pair
    /// would be two reads of the IDENTICAL crop -- already measured at 0
    /// disagreements in 2,740 regions. It must be dropped, not scored as stable.
    #[test]
    fn a_box_the_clamp_cannot_grow_is_dropped_rather_than_scored() {
        let source = perturb_page();
        let whole = Geometry::rectangle(0.0, 0.0, 400.0, 300.0);
        let base = crop(&source, &whole).unwrap();
        let grown = crop_grown(&source, &whole, 16).unwrap();
        assert_eq!(
            (grown.width(), grown.height()),
            (base.width(), base.height()),
            "the clamp should have refused this growth"
        );
        assert!(
            grown_twin(&source, &whole, &base, perturb_grow(16), true)
                .unwrap()
                .is_none(),
            "an unperturbed pair always agrees and must not be scored"
        );
    }

    /// The composed predicate, exercised through the ONE function the caller calls
    /// rather than through its halves. A fix once shipped completely
    /// unwired with 128 tests green because each half was tested and the `&&`
    /// between them was not.
    #[test]
    fn the_second_read_is_offered_only_to_onomatopoeia_with_the_flag_on() {
        let source = perturb_page();
        let region = Geometry::rectangle(50.0, 40.0, 120.0, 90.0);
        let base = crop(&source, &region).unwrap();

        // Flag on and the region IS an onomatopoeia: the only arm that reads twice.
        let offered = grown_twin(&source, &region, &base, perturb_grow(8), true)
            .unwrap()
            .expect("an onomatopoeia with the flag on should be re-read");
        assert_eq!(offered.width(), base.width() + 16);

        // Flag on, region is ordinary text: no second read. This is the arm a test
        // written against `Region.kind` would silently PASS while selecting every
        // region in production, because `region_kind` collapses onomatopoeia into
        // `TextRegion` whenever `translate_sfx` is on -- which is the default.
        assert!(
            grown_twin(&source, &region, &base, perturb_grow(8), false)
                .unwrap()
                .is_none(),
            "ordinary text must not be re-read"
        );

        // Flag off: no second read for anyone, onomatopoeia included.
        assert!(
            grown_twin(&source, &region, &base, None, true)
                .unwrap()
                .is_none(),
            "the flag is OFF by default and must gate the onomatopoeia arm too"
        );
        assert!(
            grown_twin(&source, &region, &base, None, false)
                .unwrap()
                .is_none()
        );
    }

    // ------------------------------------------------------------------------
    // The upright pass's selection rule. Every fixture below is a real failure
    // shape from the measured sweep, with the real scores rounded -- these are
    // the cells the rule was measured on; reads that came off a page are
    // invented stand-ins of the same shape. The assertions call
    // `upright_select` WHOLE: testing its layers separately is how a fix once
    // shipped unwired.

    fn row(angle: f64, text: &str, mlp: f64) -> UprightRow {
        UprightRow {
            angle,
            text: text.to_string(),
            mlp: Some(mlp),
        }
    }

    /// The adjudication normaliser, pinned: dash-runs collapse, `！` drops,
    /// `・` unifies, whitespace goes. The selection rule ranks THESE strings.
    #[test]
    fn upright_norm_matches_the_adjudication_normaliser() {
        assert_eq!(upright_norm("隆隆----！"), "隆隆—");
        assert_eq!(upright_norm("轰隆隆隆\n纸叶漫画\npaperleaf.com"), "轰隆隆隆纸叶漫画paperleaf.com");
        assert_eq!(upright_norm("霜之王・"), "霜之王·");
        assert_eq!(upright_norm("！"), "");
    }

    /// A measured crop's shape: short confident junk (`隆隆！！`, the dash dropped, score
    /// -0.08) must lose to the dash-carrying plateau (`隆隆----！` at
    /// 90/120/150, scores near -0.27). Length-normalised argmax alone picks the
    /// junk; the climb must recover the dash.
    ///
    /// The `i----柳隆` row is NOT decoration: it is the real cell's independent
    /// dash witness, and the corroboration gate requires one -- the first
    /// version of this fixture trimmed it away and the climb was correctly
    /// refused, which is the gate working, not the rule failing.
    #[test]
    fn upright_select_recovers_the_dash_the_score_talked_the_engine_out_of() {
        let rows = vec![
            row(90.0, "隆隆----！", -0.29),
            row(120.0, "隆隆---！", -0.27),
            row(150.0, "隆隆---！", -0.25),
            row(165.0, "隆隆！！", -0.08),
            row(180.0, "隆隆！！", -0.09),
            row(240.0, "一一一鳥", -1.09),
            row(270.0, "i----柳隆", -0.61),
        ];
        let pick = upright_select(&rows).expect("a read must ship");
        assert!(
            upright_norm(&rows[pick].text).contains('—'),
            "shipped {:?}; the dash-carrying superset explains the 隆隆 rows and must win",
            rows[pick].text
        );
    }

    /// A measured seam crop's shape: the `快` plateau against a junk one-angle
    /// extension (`少快`). Under the ship floor the bare `快` is a single glyph
    /// and DECLINES ENTIRELY -- the priced cost, and free on the rendered
    /// chapter because its `快跑！` recovery comes from the seam composite, not
    /// from this crop.
    #[test]
    fn upright_select_refuses_an_uncorroborated_extension() {
        let rows = vec![
            row(90.0, "快", -0.08),
            row(150.0, "快", -0.18),
            row(240.0, "少\n快", -0.11),
            row(300.0, "快", -0.94),
            row(315.0, "快", -0.31),
            row(345.0, "快", -0.07),
            row(0.0, "弃", -0.08),
        ];
        assert!(
            upright_select(&rows).is_none(),
            "the 快 plateau is one glyph: under the ship floor it must decline, \
             and the uncorroborated 少快 must certainly not ship instead"
        );
    }

    /// The corroboration gate, re-pinned on a floor-passing base so the floor
    /// cannot make the test above vacuous: a junk one-angle extension
    /// (`少快跑`) must NOT swallow the `快跑！` plateau -- its added character
    /// was seen by no other angle. Without a discriminating fixture above the
    /// floor, the corroboration gate could be deleted and every test would
    /// stay green.
    #[test]
    fn upright_select_still_refuses_an_uncorroborated_extension_above_the_floor() {
        let rows = vec![
            row(90.0, "快跑！", -0.08),
            row(150.0, "快跑！", -0.18),
            row(240.0, "少\n快跑！", -0.11),
            row(300.0, "快跑！", -0.94),
            row(315.0, "快跑！", -0.31),
            row(345.0, "快跑！", -0.07),
            row(0.0, "弃跑！", -0.09),
        ];
        let pick = upright_select(&rows).expect("a read must ship");
        assert_eq!(
            upright_norm(&rows[pick].text),
            "快跑",
            "少快跑's 少 appears at exactly one angle; an uncorroborated climb is junk"
        );
    }

    /// The geometry gate, pinned on the measured hole's own edges: the widest
    /// column that hull-erased WELL (short side 203.3) must rescue, and the
    /// narrowest wide box (308.8) must not -- wide recoveries letter beside
    /// their preserved source, because a wide hull erase replaced a background
    /// scene with flat slabs.
    #[test]
    fn only_a_narrow_recovery_joins_the_erase_mask() {
        assert!(
            rescue_narrow((0.0, 0.0, 203.3, 1844.0)),
            "the widest good column must still rescue"
        );
        assert!(
            !rescue_narrow((0.0, 0.0, 308.8, 341.0)),
            "the narrowest wide box must preserve its artwork"
        );
        // Orientation must not matter: a horizontal band is as narrow as a column.
        assert!(rescue_narrow((0.0, 0.0, 1844.0, 203.3)));
    }

    /// The ship floor, both arms pinned. A rendered crop's shape: a `舞`
    /// plateau wins the climb and must NOT ship (the true `轰` is unreachable
    /// by score, and the pick lettered "DANCE" across drawn artwork). And
    /// the count is RAW, non-whitespace: `快跑！` normalises to two characters
    /// (`！` drops) but its raw three must ship, or the chapter loses the one
    /// sub-four recovery it actually letters.
    #[test]
    fn the_ship_floor_counts_raw_glyphs_and_refuses_a_sub_three_pick() {
        let junk = vec![
            row(60.0, "舞", -0.10),
            row(90.0, "舞", -0.12),
            row(120.0, "舞", -0.20),
        ];
        assert!(
            upright_select(&junk).is_none(),
            "a single-glyph pick is under the floor and must decline to silence"
        );
        let keeper = vec![
            row(90.0, "快跑！", -0.10),
            row(120.0, "快跑！", -0.15),
            row(150.0, "快跑！", -0.20),
        ];
        let pick = upright_select(&keeper).expect("three raw glyphs must ship");
        assert_eq!(
            keeper[pick].text, "快跑！",
            "the floor counts the RAW text: a normalised count drops `！` and kills this keeper"
        );
    }

    /// A measured crop's shape: the corroborated superset climbs. `丁霜之王·`'s added `·`
    /// is witnessed by another angle's `可霜之王·`, so the climb is admitted and
    /// the middle dot -- which the truth carries -- ships.
    #[test]
    fn upright_select_admits_a_corroborated_superset() {
        let rows = vec![
            row(75.0, "丁霜之王！", -0.14),
            row(67.5, "丁霜之王·！", -0.16),
            row(90.0, "丁霜之王·", -0.22),
            row(270.0, "可霜之王·", -0.33),
            row(150.0, "一、石木那堆", -1.92),
        ];
        let pick = upright_select(&rows).expect("a read must ship");
        assert!(
            upright_norm(&rows[pick].text).contains('·'),
            "shipped {:?}; the corroborated · must not be dropped",
            rows[pick].text
        );
    }

    /// H0: a sweep that mostly saw marks ships NOTHING. Artwork crops answer
    /// punctuation and empties at most angles; a plurality of empty-key reads
    /// over the best-supported real key is the sweep saying "no text here".
    #[test]
    fn upright_select_ships_nothing_when_the_sweep_mostly_saw_marks() {
        let rows = vec![
            row(0.0, "！", -0.1),
            row(30.0, "", -0.2),
            row(60.0, "！！", -0.15),
            row(90.0, "↗", -0.3),
            row(120.0, "战", -0.05),
        ];
        assert!(
            upright_select(&rows).is_none(),
            "four empty-key rows against a single-witness 战: silence wins"
        );
    }

    /// The engine's meta-sentence is deliberately NOT excluded from selection:
    /// when it wins, it ships, and `Refusal::ImageDescription` downstream is the
    /// gate that keeps it off the page. This test pins that contract -- if
    /// selection ever starts eating meta-sentences, the downstream gate loses
    /// its machine-catchable signal and single-glyph junk ships instead.
    #[test]
    fn upright_select_ships_the_catchable_meta_sentence_rather_than_junk() {
        let rows = vec![
            row(0.0, "图片中没有文字。", -0.05),
            row(30.0, "图片中没有文字。", -0.06),
            row(60.0, "图片中没有文字。", -0.07),
            row(90.0, "叶", -0.30),
            row(120.0, "花", -0.40),
        ];
        let pick = upright_select(&rows).expect("the meta-sentence ships and is caught downstream");
        assert!(
            rows[pick].text.starts_with("图片中"),
            "shipped {:?}; the meta-sentence is the machine-catchable outcome",
            rows[pick].text
        );
    }

    /// A held-out crop's shape, found by an adversarial review: nine
    /// angles read `裂风之王！` exactly; a single angle reads `裂裂风之王！`
    /// with a duplicated `裂`. Set-difference corroboration is BLIND to
    /// duplication -- the added set is empty, nothing needs a witness -- so
    /// only the self-witness rule (a duplication climb must appear at two or
    /// more angles) and the plateau guard stop the hallucinated glyph.
    #[test]
    fn upright_select_refuses_a_single_angle_duplication_climb() {
        let rows = vec![
            row(0.0, "裂风之王！", -0.0004),
            row(30.0, "裂风之王！", -0.001),
            row(90.0, "裂风之王！", -0.002),
            row(180.0, "裂风之王！", -0.003),
            row(210.0, "裂风之王！", -0.004),
            row(270.0, "裂裂风之王！", -0.0227),
        ];
        let pick = upright_select(&rows).expect("a read must ship");
        assert_eq!(
            upright_norm(&rows[pick].text),
            "裂风之王",
            "the duplicated 裂 was witnessed at one angle only and must not ship"
        );
    }

    /// An all-empty sweep, and a sweep whose best read normalises empty, both
    /// ship nothing -- "" is a substring of everything, so an empty anchor has
    /// no defence in the containment order.
    #[test]
    fn upright_select_ships_nothing_for_empty_sweeps() {
        assert!(upright_select(&[]).is_none());
        let rows = vec![row(0.0, "", -0.1), row(30.0, "", -0.1)];
        assert!(upright_select(&rows).is_none());
        let rows = vec![row(0.0, "！", -0.1), row(30.0, "", -0.2)];
        assert!(upright_select(&rows).is_none());
    }

    /// The misread levers, on the predicate the three production sites actually call.
    /// The levers change WHO the script term reaches and WHAT counts as a
    /// mismatch; every other arm must be untouched by them.
    #[test]
    fn the_misread_levers_compose_into_withdraw_from_mask() {
        let ko = Language::Korean;
        const A: MisreadLevers = MisreadLevers { leave_misread_bubbles: true, korean_script_strict: false };
        const B: MisreadLevers = MisreadLevers { leave_misread_bubbles: false, korean_script_strict: true };
        const AB: MisreadLevers = MisreadLevers { leave_misread_bubbles: true, korean_script_strict: true };

        // The exhibit: a dialogue-role single-Han read on declared ko.
        // Today's arms leave it; lever A withdraws it.
        assert!(!withdraw_from_mask("我", Some(ko), false, false, MisreadLevers::OFF));
        assert!(!withdraw_from_mask("我", Some(ko), false, false, B), "B alone must not widen the audience");
        assert!(withdraw_from_mask("我", Some(ko), false, false, A));

        // The Class B population, free-standing: no predicate today, the
        // strict lever supplies one -- for declared Korean only.
        for leak in ["Gyul", "UEFA", "KFC", "BOLT", "yo 出"] {
            assert!(!withdraw_from_mask(leak, Some(ko), true, false, MisreadLevers::OFF), "{leak}");
            assert!(withdraw_from_mask(leak, Some(ko), true, false, B), "{leak}");
            assert!(
                !withdraw_from_mask(leak, Some(Language::ChineseSimplified), true, false, B),
                "{leak}: latin is ordinary on a manhua page"
            );
            assert!(!withdraw_from_mask(leak, None, true, false, B), "{leak}: undeclared fires nothing");
        }

        // A dialogue-role Class B leak needs BOTH levers, exactly as the
        // lettering side does.
        assert!(!withdraw_from_mask("Gyul", Some(ko), false, false, B));
        assert!(!withdraw_from_mask("Gyul", Some(ko), false, false, A));
        assert!(withdraw_from_mask("Gyul", Some(ko), false, false, AB));

        // Real Korean is untouchable under every combination.
        for levers in [MisreadLevers::OFF, A, B, AB] {
            assert!(!withdraw_from_mask("안녕하세요", Some(ko), true, false, levers));
            assert!(!withdraw_from_mask("안녕하세요", Some(ko), false, false, levers));
        }
    }

    /// The pupil's mask half needs no lever at all: with U+00B7 in the
    /// punctuation table, `·` is `letters == 0` and the unconditional
    /// illegible arm withdraws it -- the same soundness ProlongationOnly
    /// already leans on.
    #[test]
    fn the_interpunct_withdraws_through_the_illegible_arm() {
        assert!(illegible_text("·"), "U+00B7 must not count as a letter");
        assert!(withdraw_from_mask("·", None, false, false, MisreadLevers::OFF));
        // The katakana middle dot, its long-listed twin, behaves identically.
        assert!(withdraw_from_mask("・", None, false, false, MisreadLevers::OFF));
    }

    /// The stamp is a three-arm `.or_else` composite, asserted here on the one
    /// function the write site calls -- testing the arms separately would not
    /// test the chain. The declared language echoes
    /// back; `ja-JP` is only the undeclared default.
    #[test]
    fn the_stamp_echoes_the_declared_language_and_defaults_to_ja_jp() {
        for (declared, tag) in [
            (Language::Korean, "ko-KR"),
            (Language::ChineseSimplified, "zh-CN"),
            (Language::Japanese, "ja-JP"),
        ] {
            assert_eq!(stamped_language(None, Some(declared)).unwrap().as_str(), tag);
        }
        // Undeclared keeps the historical default, `ja-JP`. Omitting the field
        // instead would break any consumer that keys on its presence.
        assert_eq!(stamped_language(None, None).unwrap().as_str(), "ja-JP");
    }

    /// An existing tag beats the declaration: a re-run must not overwrite what an
    /// earlier pass already stamped (the not-losing-information arm).
    #[test]
    fn an_existing_tag_survives_the_stamp_whatever_is_declared() {
        let previous = LanguageTag::new("ko-KR").ok();
        assert_eq!(
            stamped_language(previous, Some(Language::Japanese)).unwrap().as_str(),
            "ko-KR"
        );
    }

    /// The spot-image budget. The 2130x8000 pole is the MEASURED
    /// refusal -- 16,712 visual tokens against serve.ps1's `-c 10240`,
    /// `request ... exceeds the available context size` -- and every page size
    /// the corpora actually ship must pass through untouched, or the scale
    /// would silently change spot reads on ordinary pages.
    #[test]
    fn the_spot_image_is_bounded_and_ordinary_pages_pass_untouched() {
        for (width, height) in [(1200u32, 1700u32), (2130, 1350), (2130, 2700), (800, 8000)] {
            assert_eq!(spot_scale_dimensions(width, height), None, "{width}x{height}");
        }
        let (width, height) = spot_scale_dimensions(2130, 8000).unwrap();
        assert!(u64::from(width) * u64::from(height) <= SPOT_MAX_PIXELS);
        let aspect_in = 2130.0 / 8000.0;
        let aspect_out = f64::from(width) / f64::from(height);
        assert!((aspect_in - aspect_out).abs() < 0.01, "{width}x{height}");
        assert!(width < 2130 && height < 8000, "a bound is a shrink, never a stretch");
        // Degenerate zero-size input is left alone rather than divided by.
        assert_eq!(spot_scale_dimensions(0, 0), None);
    }
}
