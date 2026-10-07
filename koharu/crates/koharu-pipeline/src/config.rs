use std::num::NonZeroU32;

use anyhow::{Result, bail};
use koharu_translator::{GenerationConfig, Language};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use specta::Type;

use crate::stages::{Flux2KleinConfig, KoharuLayoutRFDetrSeg2XLConfig, RoremMixedConfig};

#[derive(Clone, Debug, PartialEq, Type)]
pub struct PipelineConfig {
    pub detection: DetectionModel,
    pub ocr: OcrModel,
    pub translation: TranslationConfig,
    pub inpainting: InpaintingModel,
    /// Settings for every model are kept independently of the active model.
    /// The active stage fields above only select which profile is used.
    pub processor: ProcessorConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct PipelineFile {
    detection: ModelSelection,
    ocr: ModelSelection,
    translation: TranslationConfig,
    inpainting: ModelSelection,
    #[serde(default)]
    processor: ProcessorConfig,
}

impl Default for PipelineFile {
    fn default() -> Self {
        Self {
            detection: ModelSelection {
                model: "koharu-layout-rfdetr-seg-2xl".to_owned(),
            },
            // The SECOND copy of the OCR default -- `PipelineConfig::default()`
            // below is the first, and `missing_slots_use_defaults` asserts the two
            // agree, so changing one copy without the other fails that test.
            ocr: ModelSelection {
                model: "hunyuan-ocr-1.5".to_owned(),
            },
            translation: TranslationConfig::default(),
            inpainting: ModelSelection {
                model: "lama".to_owned(),
            },
            processor: ProcessorConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct ModelSelection {
    model: String,
}

impl Serialize for PipelineConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let detection = match &self.detection {
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_) => "koharu-layout-rfdetr-seg-2xl",
        };
        let ocr = match &self.ocr {
            OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
            OcrModel::MangaOcr => "manga-ocr",
            OcrModel::BaberuOcr => "baberu-ocr",
            OcrModel::OllamaVision => "ollama-vision",
            OcrModel::HunyuanOcr1_5 => "hunyuan-ocr-1.5",
        };
        let inpainting = match &self.inpainting {
            InpaintingModel::LaMa {} => "lama",
            InpaintingModel::AotInpainting {} => "aot-inpainting",
            InpaintingModel::Flux2Klein(_) => "flux2-klein",
            InpaintingModel::RoremMixed(_) => "rorem-mixed",
        };
        let mut processor = self.processor.clone();
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(config) = &self.detection;
        processor
            .koharu_layout_rfdetr_seg_2xl
            .get_or_insert_with(|| config.clone());
        match &self.inpainting {
            InpaintingModel::Flux2Klein(config) => {
                processor.flux2_klein.get_or_insert_with(|| config.clone());
            }
            InpaintingModel::RoremMixed(config) => {
                processor.rorem_mixed.get_or_insert_with(|| config.clone());
            }
            InpaintingModel::LaMa {} | InpaintingModel::AotInpainting {} => {}
        }
        PipelineFile {
            detection: ModelSelection {
                model: detection.to_owned(),
            },
            ocr: ModelSelection {
                model: ocr.to_owned(),
            },
            translation: self.translation.clone(),
            inpainting: ModelSelection {
                model: inpainting.to_owned(),
            },
            processor,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PipelineConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let file = PipelineFile::deserialize(deserializer)?;
        let detection = match file.detection.model.as_str() {
            "koharu-layout-rfdetr-seg-2xl" => DetectionModel::KoharuLayoutRFDetrSeg2XL(
                file.processor
                    .koharu_layout_rfdetr_seg_2xl
                    .clone()
                    .unwrap_or_default(),
            ),
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported detection model {model}"
                )));
            }
        };
        let ocr = match file.ocr.model.as_str() {
            "paddleocr-vl-1.6" => OcrModel::PaddleOcrVl1_6,
            "manga-ocr" => OcrModel::MangaOcr,
            "baberu-ocr" => OcrModel::BaberuOcr,
            "ollama-vision" => OcrModel::OllamaVision,
            "hunyuan-ocr-1.5" => OcrModel::HunyuanOcr1_5,
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported OCR model {model}"
                )));
            }
        };
        let inpainting = match file.inpainting.model.as_str() {
            "lama" => InpaintingModel::LaMa {},
            "aot-inpainting" => InpaintingModel::AotInpainting {},
            "flux2-klein" => {
                InpaintingModel::Flux2Klein(file.processor.flux2_klein.clone().unwrap_or_default())
            }
            "rorem-mixed" => {
                InpaintingModel::RoremMixed(file.processor.rorem_mixed.clone().unwrap_or_default())
            }
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported inpainting model {model}"
                )));
            }
        };
        Ok(Self {
            detection,
            ocr,
            translation: file.translation,
            inpainting,
            processor: file.processor,
        })
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(
                KoharuLayoutRFDetrSeg2XLConfig::default(),
            ),
            // HunyuanOCR is the default. PaddleOCR-VL is the reserve: still wired
            // and selectable, no longer default.
            ocr: OcrModel::HunyuanOcr1_5,
            translation: TranslationConfig::default(),
            inpainting: InpaintingModel::LaMa {},
            processor: ProcessorConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct TranslationConfig {
    pub model: koharu_translator::ModelSelection,
    pub generation: GenerationConfig,
    #[specta(type = String)]
    pub target_language: Language,
    /// What the page is written in, when the caller knows.
    ///
    /// **This is NOT for the translator; wiring it there was measured and rejected.**
    /// `TranslationRequest::with_source_language` has no consumer on the local
    /// path but the prompt -- `prompt.rs` swaps the named language into the system
    /// prompt and serialises it into the user JSON -- and naming the source
    /// language in the prompt measured **-0.154 +/-0.071** over 511 segments.
    /// `replay.rs` builds its `named` arm with exactly that call.
    /// **`stages/translation.rs` must not read this field**, or the shipping
    /// page becomes the arm that lost.
    ///
    /// It exists for the ERASE MASK. `stages/ocr.rs` withdraws a region from the
    /// mask when OCR could not read it, and until this field existed the pipeline
    /// could not ask the question that separates a bad read from a real one: is
    /// this text in the script the page is written in? `illegible_text` says so
    /// in its own comment. That half is pixels rather than prompt, and the
    /// prompt measurement above says nothing about it.
    ///
    /// `None` means nothing trustworthy said, and then no script rule fires --
    /// the same default `labels::SourceScript::Unknown` takes, for the same
    /// reason: inferring "not Japanese" from silence would fire the kana rule on
    /// genuine Japanese dialogue and drop it, which is the one failure this gate
    /// must not have.
    #[serde(default)]
    #[specta(type = Option<String>)]
    pub source_language: Option<Language>,
    pub instructions: Option<String>,
    /// Whether the translator prompt carries the containment sentence -- each
    /// segment's translation covers its own source only, even where one
    /// sentence spans several segments. The sentence and its reasoning live in
    /// `koharu-translator/src/prompt.rs`.
    ///
    /// `serde(default)` so a pipeline config written before the field existed
    /// still loads under `deny_unknown_fields`; the default is OFF.
    #[serde(default)]
    pub containment_clause: bool,

    /// Whether each segment is described to the model -- what KIND of text it is,
    /// and whether the artwork STRIKES IT THROUGH. **OFF by default**;
    /// `--segment-context`.
    ///
    /// The model has always received a bare `{id, text}` list, so a name the page
    /// visibly cancels arrived as an ordinary word and came back invented. The
    /// facts were already measured -- the strike by `strike_ink`, the kind by
    /// `region_kind` -- and both died before the translator.
    ///
    /// `serde(default)` for `containment_clause`'s reason: a config written
    /// before the field existed still loads under `deny_unknown_fields`.
    #[serde(default)]
    pub segment_context: bool,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            model: koharu_translator::ModelSelection::default(),
            generation: GenerationConfig::default(),
            target_language: Language::English,
            source_language: None,
            instructions: None,
            containment_clause: false,
            segment_context: false,
        }
    }
}

impl PipelineConfig {
    pub fn load() -> anyhow::Result<koharu_config::Config<Self>> {
        koharu_config::load("pipeline")
    }

    pub fn detection(&self) -> Result<DetectionModel> {
        match &self.detection {
            DetectionModel::KoharuLayoutRFDetrSeg2XL(config) => {
                Ok(DetectionModel::KoharuLayoutRFDetrSeg2XL(
                    self.processor
                        .koharu_layout_rfdetr_seg_2xl
                        .clone()
                        .unwrap_or_else(|| config.clone()),
                ))
            }
        }
    }

    pub fn inpainting(&self) -> Result<InpaintingModel> {
        match &self.inpainting {
            InpaintingModel::LaMa {} => Ok(InpaintingModel::LaMa {}),
            InpaintingModel::AotInpainting {} => Ok(InpaintingModel::AotInpainting {}),
            InpaintingModel::Flux2Klein(config) => Ok(InpaintingModel::Flux2Klein(
                self.processor
                    .flux2_klein
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
            InpaintingModel::RoremMixed(config) => Ok(InpaintingModel::RoremMixed(
                self.processor
                    .rorem_mixed
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let _ = self.detection()?;
        let _ = self.inpainting()?;
        if !matches!(
            self.ocr,
            OcrModel::PaddleOcrVl1_6
                | OcrModel::MangaOcr
                | OcrModel::BaberuOcr
                | OcrModel::OllamaVision
                | OcrModel::HunyuanOcr1_5
        ) {
            bail!("unsupported OCR model")
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default, deny_unknown_fields)]
pub struct ProcessorConfig {
    #[serde(rename = "koharu-layout-rfdetr-seg-2xl")]
    pub koharu_layout_rfdetr_seg_2xl: Option<KoharuLayoutRFDetrSeg2XLConfig>,
    #[serde(rename = "flux2-klein")]
    pub flux2_klein: Option<Flux2KleinConfig>,
    #[serde(rename = "rorem-mixed")]
    pub rorem_mixed: Option<RoremMixedConfig>,

    /// Read a text region whose larger side is at least this many pixels with
    /// PaddleOCR-VL, whatever `ocr` selects. `None` disables the routing.
    ///
    /// The server defaults this to 448 (`--large-crop-ocr`). It stays an
    /// `Option` here rather than carrying that default, because this crate has
    /// no opinion about it: `PipelineConfig` is also built from a config file and
    /// by `run.rs`, and neither should acquire a second model implicitly.
    ///
    /// **Why a second model rather than a bigger token budget.** `manga-ocr` and
    /// `baberu-ocr` both preprocess by resizing the crop to a fixed SQUARE --
    /// 224 for manga-ocr -- without preserving aspect ratio, so what reaches the
    /// encoder is `font_px * 224 / crop_width`. A 30px bubble in a 200px crop
    /// gives ~34px and reads perfectly; a 14px caption in a 465px box gives
    /// ~6.7px and comes back as fluent invented prose. Raising `max_length`
    /// cannot help: a blind recogniser emits EOS early, so the cap is never the
    /// binding constraint. PaddleOCR-VL's `smart_resize` scales both axes by one
    /// factor against a pixel budget, so it keeps the glyphs and reads the crop.
    ///
    /// **The threshold is deliberately crude, and it can afford to be.** The
    /// 66-region benchmark found manga-ocr and paddleocr-vl-1.6 disagree on
    /// kanji in zero regions, so a crop routed unnecessarily costs time and some
    /// punctuation normalisation, never accuracy. Measured over 453 regions of
    /// 40 real pages, 448 -- twice the 224 input -- fires on 0.7% of them.
    ///
    /// Ignored when `ocr` is already `paddleocr-vl-1.6`, which has no square to
    /// squash into.
    #[serde(rename = "large-crop-ocr-px")]
    pub large_crop_ocr_px: Option<u32>,

    /// The ceiling on `large_crop_ocr_px`, as a fraction of the page's own area.
    /// A text region at or above it is **not** routed. `None` disables it, which
    /// is this crate's default and reproduces the pre-ceiling behaviour exactly.
    /// The server defaults it to 0.5 (`--large-crop-ocr-max-area`).
    ///
    /// **Why a lower bound alone is not enough.** RF-DETR does not only
    /// mis-classify, it mis-segments: on one real page it returned a single
    /// `onomatopoeia` box of 1210x1247 on an 844x1200 page -- 1.49x the area of
    /// the page it was detected on -- over artwork, at confidence 0.43. It
    /// cleared the 448 lower bound, went to PaddleOCR-VL, and returned zero
    /// characters. That crop is ~60x a typical text region, so torch's caching
    /// allocator grew its high-water mark ~3.7 GB to serve it and, having no
    /// automatic return path, kept them -- which is the input side of the same
    /// latch `release_cached_vram` addresses from the other end. Measured: page
    /// median 5,189 ms before, 12,790 ms after, for the rest of the process.
    ///
    /// **It is a geometric test, not a content heuristic**, which is what
    /// separates it from ink density, detection confidence and text-segmenter
    /// fraction -- all three measured on this same class of mis-segmented box,
    /// all three rejected. A region larger than the page it was found on is a
    /// detector error by construction, whatever it contains.
    #[serde(rename = "large-crop-ocr-max-area")]
    pub large_crop_ocr_max_area: Option<f32>,

    /// Whether a region over `large_crop_ocr_max_area` is read by **no** engine,
    /// rather than merely being kept off PaddleOCR-VL.
    ///
    /// Off by default, and the default is the conservative arm on purpose. The
    /// measured half is that PaddleOCR-VL read the offending box as the empty
    /// string, so nothing is lost by not reading it at all; the unmeasured half
    /// is what the primary returns for the same box, and manga-ocr's failures
    /// are fluent rather than empty. Leaving it on the primary also keeps a
    /// legitimately huge region readable -- a full-page sign has huge glyphs for
    /// the same reason it has a huge box, and the 224 square only punishes
    /// glyphs that are small relative to the crop.
    ///
    /// Decided independently of `large_crop_ocr_px`: a box larger than its page
    /// is a detector error whether or not routing is configured, so turning
    /// routing off must not turn the ceiling off with it.
    #[serde(rename = "skip-implausible-regions")]
    pub skip_implausible_regions: bool,

    /// Turn a tall vertical free-standing column into its own ink instead of
    /// widening it, so the English runs DOWN the column the way the artist set
    /// the original.
    ///
    /// **ON by default.** It changes what a page looks like, so it was adopted
    /// only after a render.
    ///
    /// **Its blast radius is bounded by the caller, not by this flag.** The
    /// widening it replaces fires on 366 of 1,281 free-text regions on a
    /// baseline run -- 38.3% of manga-ja free-text against 10.3% of
    /// manhua, a 17x spread a pooled figure would have hidden -- but the turn is
    /// gated on `joined_page`, and a manga page is never joined. Measured: a real
    /// manga page carrying two regions that QUALIFY for the turn renders
    /// **byte-identical** with it on and off.
    ///
    /// It exists because the box, not the size cap, is what holds a skill name
    /// small: raising the cap measured 2.10x in the JSON and ZERO pixels in the
    /// render. Turning the box swaps the binding constraint
    /// from the column's width to its height. See `free_text_column_geometry`
    /// for why it is gated to an inferred angle of exactly zero.
    #[serde(rename = "rotate-free-text-columns")]
    pub rotate_free_text_columns: bool,

    /// Whether a box the detector truncated at a slice edge is grown along its
    /// own ink to span the slice, and whether a sub-floor box against an edge is
    /// reported as an edge hint. **OFF by default.**
    ///
    /// The webtoon seam is the consumer of both: it needs a box at each end of a
    /// run to plan a join, and on one test chapter the display column has no
    /// usable box on three slices of five. With this off, that stays true and the
    /// seam joins the pair it has always joined.
    ///
    /// **Off is a choice, not a gap in measurement.** Growing a column to span its
    /// slice pushes it past `CROSS_SLICE_HEIGHT_SHARE`, so on a page read UNJOINED
    /// the cross-slice fragment rule then refuses it: the artwork survives and the
    /// original glyphs stay, where today a wrong string is lettered over them.
    /// That is probably better, and it is still a default that changes what a
    /// reader sees.
    /// Whether the free-text column TURN is also offered on an UNJOINED page.
    /// **ON by default**, on the rendered cost.
    ///
    /// On a joined page the turn already hands back the ink's own bbox and the
    /// exemplar measures 1.00x. Unjoined columns still take the widened cell,
    /// whose width is driven by HEIGHT (`max(own_width, own_height * 0.6)`), so a
    /// 145x626 column is handed 375.6 px and the English lands on artwork the
    /// eraser never cleaned -- both mask builders key off the raw detection.
    ///
    /// On manga the same predicate also claims ordinary upright vertical
    /// Japanese, and detection is never told the source language, so the cost is
    /// real: sound EFFECTS are lettered sideways. Measured at 0.169% and 0.046% of
    /// two pages, and no balloon changes -- dialogue takes the bubble path.
    #[serde(rename = "turn-unjoined-columns")]
    pub turn_unjoined_columns: bool,

    #[serde(rename = "repair-clipped-columns")]
    pub repair_clipped_columns: bool,

    /// `Some(floor)` opens a replacement-only `text` score band on JOINED pages
    /// only. **`None` by default** -- the server resolves
    /// `--joined-page-text-floor`. A composite squashes a cut
    /// column's own score below the 0.25 floor exactly where the column matters
    /// most, while a fragment survives. A band box is never a region in its own
    /// right; it survives only by replacing a kept box through the tie-break.
    #[serde(rename = "joined-page-text-floor")]
    pub joined_page_text_floor: Option<f32>,

    /// Whether NMS prefers the column-shaped box of a contained pair -- same
    /// width and much taller, or same height and much narrower -- over the
    /// higher-scored one. **ON by default**, adopted on a rendered census, and
    /// **SCOPED to declared Chinese/Korean** in the server's `desired_config` --
    /// on Japanese it was measured harmful (10 of 91 pages of a Japanese test
    /// volume degraded). `--axis-aware-nms`. This ships with
    /// `joined-page-text-floor` as a pair: the floor alone is
    /// inert by construction, because a band box survives only through the
    /// replace outcome this flag enables.
    #[serde(rename = "axis-aware-nms")]
    pub axis_aware_nms: bool,

    /// Whether a box the axis tie-break EVICTS leaves its uncovered residue
    /// behind as a region of its own, instead of going whole. **ON by default**,
    /// adopted on a whole-chapter render; `--nms-residue-regions`.
    ///
    /// Rides on `axis-aware-nms` and cannot act without it -- a residue exists
    /// only where a `ReplaceKept` happened -- but carries its own switch so the
    /// tie-break keeps a byte-exact control arm, which the Japanese census of
    /// `axis-aware-nms` needed and would not have had otherwise.
    #[serde(rename = "nms-residue-regions")]
    pub nms_residue_regions: bool,

    /// Whether a read the pipeline ALREADY refuses to letter is kept out of the
    /// translation request, instead of being translated and thrown away. **ON by
    /// default**, adopted on a one-flag render; `--skip-unlettered-reads`.
    ///
    /// **Measured on a 179-slice Chinese test chapter (with an earlier, longer
    /// watermark list): 188 of 331 regions are refused, and NOT ONE of them is
    /// ever lettered.**
    /// So the cost is not the wasted tokens, which would be reason enough at 57%
    /// of the page's regions. It is that junk riding in a request re-rolls the
    /// GOOD text beside it: the chapter's translation drift begins at a seam
    /// composite where an admitted `！` sits in the same request as a live
    /// one-character region, whose English the model then changed to a different
    /// word -- and *that* pair is not refused, so it enters the story window and
    /// compounds.
    ///
    /// **The filter never empties a page.** If nothing would survive it, every
    /// target is passed through exactly as today, so `has_work` is unchanged on
    /// every page in both arms. That is not tidiness: `has_work` false skips the
    /// stage, which means `Progress::Untranslated` never fires, which means
    /// `nothing_translated`'s 502 cannot fire -- and the renderer's
    /// `fallback_to_source_text` then paints the raw source in a translation
    /// font. A silent untranslated page is precisely the lie that guard exists
    /// to catch, so the filter is not allowed to reach it.
    ///
    /// Carries its own switch for `nms-residue-regions`' reason: a byte-exact
    /// control arm. Reply ids are safe by construction -- `prompt.rs` indexes
    /// into the segments THIS request sent, so a shorter list shrinks the id
    /// space uniformly and `process`'s `zip` stays sound.
    #[serde(rename = "skip-unlettered-reads")]
    pub skip_unlettered_reads: bool,

    /// Whether a strike-through mark the author drew across a region is measured
    /// and RE-DRAWN over the replacement lettering. **ON by default**, adopted
    /// on a whole-chapter render; `--strike-through-devices`.
    ///
    /// Unlike its `nms-residue-regions` neighbour it is **not script-scoped and
    /// not transitively scoped either**, so shipping it on arms it for Japanese
    /// too. That is deliberate -- the detector measures ink geometry and never
    /// asks what the text says -- but the validation set behind it is n=2
    /// positives, both Chinese, so it is the flag to watch first if a ja page
    /// ever grows a line through its lettering.
    ///
    /// Re-drawn rather than preserved, and that is forced rather than chosen: the
    /// English is composited over the inpainted raster, so a surviving mark sits
    /// UNDER it. Sparing the eraser cannot put a line across the words.
    #[serde(rename = "strike-through-devices")]
    pub strike_through_devices: bool,

    /// Whether free-standing lettering takes its fill, weight and
    /// contrast-picked outline from the drawn ink's sampled colour rather than
    /// the contrast-ranked sample. `--sampled-ink-lettering`, and
    /// the server's `cli.rs` is the authority for the shipping default.
    #[serde(rename = "sampled-ink-lettering")]
    pub sampled_ink_lettering: bool,

    /// Whether a detected BUBBLE holding no text region of its own is read as one.
    /// **OFF by default.**
    ///
    /// The detector finds a balloon's shape and its text independently, and on
    /// one Chinese test slice it found the shape at **0.5078** and none of the text --
    /// so the balloon reached a reader in Chinese while ten counters read zero and
    /// every one of them was correct. Every gate this project has tuned lives
    /// downstream of a detection that never happened, which is why nothing else
    /// reaches that page.
    ///
    /// Measured over **four populations, 750 pages, 1,565 bubbles**: the rule fires
    /// **4 times**. Three are real untranslated text -- that balloon and two short
    /// Japanese balloons (an exclamation and a lone `ん`), defects nobody had
    /// reported -- and one is a spurious bubble on flat skin. **A
    /// false-positive rate of 1 in 1,565, 0.064%**, and manga carries 1,330 of
    /// those bubbles with two hits, both true.
    ///
    /// **`withdraw_unread_masks` does NOT mitigate that false positive, and a
    /// render showed it.** `withdraw_unread_masks` only covers a region OCR reads as
    /// NOTHING. On a webtoon test page the spurious bubble was read as `ー` -- the
    /// character's own mouth line -- so the veto never fired, the region was erased,
    /// and a white dash was lettered across her face. A read that is wrong is not a
    /// read that is absent.
    ///
    /// **The region does not grow between the detection and OCR.** An earlier
    /// version handed the bubble's axis-aligned bbox to a
    /// `text` region, and `build_region` then TURNED it to the balloon's angle --
    /// and turning an already-axis-aligned rectangle can only inflate its hull, so
    /// the balloon arrived at OCR at **1145 x 662 = 0.6957 of the page**, over the 0.5
    /// `large_crop_ocr_max_area` ceiling. `skip_implausible_regions` refused it
    /// (`no OCR engine ever read it`) and the render came back byte-identical to
    /// the OFF arm, on a bubble detection measuring only 0.4342.
    ///
    /// `oriented_ink_box` in `stages/detection.rs` is the repair and carries the
    /// arithmetic: the region is turned from the balloon's extent along ITS OWN
    /// axes, so the hull lands back on the ink at **0.3713** instead of inflating
    /// past it. **Shrinking to the ink's HULL would not have worked** -- 0.3713 is
    /// the hull of a balloon already turned, and feeding it back axis-aligned
    /// applies the angle twice, for 0.6073 and the same refusal.
    ///
    /// The false positive above is unchanged by that, and is still unpaid for.
    #[serde(rename = "read-textless-bubbles")]
    pub read_textless_bubbles: bool,

    /// Whether a tall kana-free free-text column is read a SECOND time with the
    /// crop turned 90 CCW, and the turned read preferred. **OFF by default.**
    ///
    /// The artist sets a skill name as a HORIZONTAL line and turns the whole line
    /// sideways, so its glyphs are rotated 90 clockwise on the page. Nothing tells
    /// the recogniser, which reads them upright and returns plausible wrong
    /// characters. Measured on one joined test column, one paddle call on
    /// the whole strip: turned it returns `苍炎之王·冥霜之王·裂风之王`, byte-exact
    /// against the reference; upright it returns `仓炎岛王王·冥雪岛王王`.
    ///
    /// **Off because the population at risk cannot be separated by the gate.**
    /// The kana census keeps ordinary vertical JAPANESE out, but upright vertical
    /// CHINESE is kana-free too and would be turned wrongly. Only a control render
    /// can say how often that happens. See `stages::ocr::wants_rotated_reread`.
    #[serde(rename = "reread-rotated-columns")]
    pub reread_rotated_columns: bool,

    /// Whether a free-standing region whose read came back EMPTY is re-read
    /// through the OCR sidecar's rotation sweep, with `stages::ocr::
    /// upright_select` choosing among the sixteen angle candidates or declining.
    /// **A bare config still deserialises to `false`, but the server ships it
    /// ON**, adopted on a rendered chapter, and `cli.rs`'s resolve is the
    /// authority for the shipping value.
    /// HTTP-sidecar route only -- the in-process engines have no rotation option
    /// to ask, and the pass restores the refusal byte-for-byte when the sidecar
    /// lacks the rotate capability.
    ///
    /// The population it exists for is the refused boxes: text rotated 47-130
    /// degrees on the page, unrepresentable in the pipeline's ±45 window.
    /// Measured on a 25-crop test population: 13/17 text reads
    /// recovered (native floor 1/17, angle-oracle ceiling 14/17) with 0/8
    /// artwork fabrications surviving the downstream gates; the selection rule's
    /// two text residuals are Pareto-proven unreachable from the decoder score.
    /// `upright_select` also carries a three-RAW-glyph ship floor (11/17
    /// offline, zero rendered cost, every observed junk lettering silenced);
    /// `stages/ocr.rs::UPRIGHT_SHIP_FLOOR`.
    /// Scope guard: the pass fires only where today's pipeline letters NOTHING,
    /// so its worst case is lettering junk where there was silence -- which is
    /// exactly what the spotting gate, `Refusal::ImageDescription` and the
    /// script rules already refuse, and they all stay in its path.
    #[serde(rename = "upright-pass")]
    pub upright_pass: bool,

    /// Whether a SYNTHESISED bubble read (`--read-textless-bubbles`) is read a
    /// SECOND time with the crop turned 180 degrees, keeping the flipped read
    /// only when its confidence beats the upright read's by
    /// `stages::ocr::FLIP_CONFIDENCE_MARGIN`. **A bare config still
    /// deserialises to `false`, but the server ships it ON**, adopted on a
    /// rendered chapter, and `cli.rs`'s resolve is the authority for the
    /// shipping value.
    ///
    /// The population is the synthesised regions ONLY: keeping the upright read
    /// leaves the page exactly as it renders without the flag, so the flip
    /// cannot touch an ordinary balloon. It exists for the ~180-degree rotation
    /// class, which no other orientation lever can
    /// represent -- `mask_angle` is ±45, and the upright pass sweeps refused
    /// FREE-STANDING targets while a synthesised region is dialogue by
    /// construction. An upside-down balloon WITH a detected text region inside
    /// is deliberately NOT reached. An engine reporting no confidence never
    /// flips and never pays the second read: `manga-ocr`, `baberu-ocr` and
    /// Ollama are structurally outside it.
    #[serde(rename = "flip-reread-bubbles")]
    pub flip_reread_bubbles: bool,

    /// Whether a sparse or decline-carrying page may buy ONE HunyuanOCR
    /// spotting call and MINT regions for display runs the detector never
    /// boxed. **OFF by default.**
    ///
    /// The population: display runs the detector never boxed -- a diagonal
    /// banner (no covering box on the host's own bytes, 93.6% of its ink outside
    /// every proposal), two display runs on another page, and six more
    /// missed-display pages in a census -- all of them sparse (0-2 detector
    /// regions). Admission is GEOMETRIC first (area floor 2.5% of page, ceiling
    /// 0.5, overlap fence against every existing box) because the watermark plate
    /// was measured MANGLED on 14 pages, so a string filter can only ever be a
    /// belt. A surviving box is swept at 16 angles and `upright_select` decides;
    /// a shipped read still passes `withdraw_from_mask` before anything is
    /// minted. Sidecar route only; paged Japanese structurally never reaches it.
    #[serde(rename = "spot-rescue")]
    pub spot_rescue: bool,

    /// Whether a shipped spot rescue also joins the ERASE mask
    /// (`text-mask-rescue`). **OFF by default, and judged separately from the
    /// lettering** -- a rescued display box is far over `rescue_narrow`'s 256 px
    /// bound, and a big-box erase destroys a lot of artwork when it is wrong.
    #[serde(rename = "spot-rescue-erase")]
    pub spot_rescue_erase: bool,

    /// Whether a WIDE scream-read spot mint becomes the
    /// mark-replacement device: the erase scoped to the drawn mark's enclosed
    /// ink instead of the rotated box's hull, and ONE styled gradient
    /// replacement lettered along the ink's own principal axis. **OFF by
    /// default** (the server's `cli.rs` is the authority for the shipping
    /// default), and inert unless `spot-rescue-erase` is also on.
    #[serde(rename = "replace-scream-marks")]
    pub replace_scream_marks: bool,

    /// Whether a JOINED page (a seam composite) may buy the spot call too,
    /// regardless of how many regions its detector proposed. **OFF by
    /// default.**
    ///
    /// The population: a display run CUT by a slice boundary. Per-slice the
    /// fragments are undetected or unreadable, and the sparse trigger misses
    /// the composite because a composite carries its neighbours' ordinary
    /// regions -- one test composite proposes 3, one over the
    /// sparse threshold, while HunyuanOCR's spotting boxes the whole
    /// `凌霄境界・三重防御` column on it (probed at debug level:
    /// the column box `[112, 14, 235, 998]` was spotted on the base slice and
    /// dropped only by the overlap fence; on the composite nothing overlaps
    /// it). A composite exists BECAUSE something crosses the cut, so it is
    /// exactly the rescue's population, ~26 pages and ~26 spot calls per
    /// 179-slice chapter.
    #[serde(rename = "spot-rescue-joined")]
    pub spot_rescue_joined: bool,

    /// How much MORE confident the UPRIGHT read must be before it beats the turned
    /// one, when both orientations were read. **`None` disables the ranking and is
    /// the default**, which is byte-identical to the behaviour that shipped before
    /// it existed.
    ///
    /// `choose_orientation` kept the turned read whenever it was non-empty, because
    /// nothing could rank two reads. The decoder's own length-normalised sequence
    /// probability is now available, so something can.
    ///
    /// **Asymmetric on purpose.** The turn exists to rescue sideways columns, so a
    /// tie -- or anything short of `margin` -- must leave it alone. Only a clearly
    /// more confident upright read takes it back.
    ///
    /// Measured over all 179 slices of a Chinese test chapter, six
    /// upright/turned pairs, every one adjudicated against the drawn page: the
    /// turn wins its two GOOD pairs by 0.0398 and 0.1078, the upright wins its
    /// two by 0.2380 and 0.5449. Any value
    /// in `(0.11, 0.23)` is right or neutral on all five differing pairs. The
    /// HunyuanOCR re-measurement (thirteen pairs) widened that band to
    /// `(0.03, 0.44)`, and **the server ships `0.17`**, adopted on a rendered
    /// chapter. A bare config
    /// that omits the key still deserialises to `None` (do not rank); `cli.rs`'s
    /// resolve is the authority for the shipping value, and its off arm is `inf`.
    #[serde(rename = "orientation-confidence-margin")]
    pub orientation_confidence_margin: Option<f64>,

    /// Pixels to grow an ONOMATOPOEIA crop by before reading it a SECOND time, so
    /// the two reads can be compared. **`None` disables it and is the default**,
    /// which is byte-identical to the behaviour that shipped before it existed.
    ///
    /// **It is REPORTED and never acted on.** The second read is logged beside the
    /// first and then dropped: it does not reach the region's text, the mask, the
    /// eraser or the translator. Nothing is gated on it, because nothing yet knows
    /// whether it separates anything.
    ///
    /// **What it is for.** A real glyph should survive a few pixels of extra
    /// background; a shape matched off artwork should not. Re-running the
    /// IDENTICAL crop tells nothing -- both engines are byte-deterministic,
    /// confirmed at 49 of 49 across two server processes -- so a tell has to vary
    /// the GEOMETRY, and this one does. It is keyed to neither the string's
    /// content nor its script, nor ink density, detection confidence or
    /// text-segmenter fraction (rejected in `implausible_region`'s own doc
    /// comment), nor a format classifier, a second-engine tell, read repetition
    /// or OCR confidence, all of which were tried and rejected.
    ///
    /// **GROW ONLY, never shrink** -- see `stages::ocr::crop_grown`, which also
    /// documents why a box the clamp REFUSES to grow must be discarded rather than
    /// scored as stable.
    ///
    /// **Scoped to onomatopoeia**, which is 18 regions across the 179 slices of
    /// a Chinese test chapter, so the cost is one extra inference on a
    /// population of that size rather than on every region.
    #[serde(rename = "perturb-reread-grow-px")]
    pub perturb_reread_grow_px: Option<NonZeroU32>,

    /// Whether a watermark verdict is scoped to the site's OWN TEXT rather than
    /// condemning every character the detector merged with it. **OFF by
    /// default.**
    ///
    /// The site stamps its plate horizontally coincident with a display column, so
    /// any correctly-drawn box around that column contains it -- nothing merged
    /// anything, and there is no upstream split to make. One `最新免费漫画` then
    /// condemns the skill name beside it and the column is thrown away whole,
    /// which breaks the design rule that no text is taken off the page.
    ///
    /// Scoped to SUBSTRINGS, not lines: with `reread_rotated_columns` on the
    /// recogniser returns the name and the plate on ONE line, so a line rule would
    /// refuse the lot. See `stages::story_text`.
    ///
    /// **Off because it trades artwork for text.** A region that survives is
    /// erased and inpainted WHOLE -- plate included -- which is the smearing that
    /// moving `watermark_text` into this crate was meant to stop. Bounded to the
    /// few regions carrying both.
    #[serde(rename = "scope-watermark-refusals")]
    pub scope_watermark_refusals: bool,

    /// Whether a DIALOGUE-role read that the script rules refuse is also taken
    /// out of the erase mask -- the pixel half of a paired lever.
    ///
    /// `withdraw_from_mask`'s script arm requires `free_standing` today, and the
    /// lettering side's free-standing gate skips the same regions -- reader and
    /// mask move together, so BOTH halves skip a misread balloon. On Korean test
    /// chapters that is a hangul groan misread as a Han character, erased, and
    /// lettered with a wrong English word, and a drawn red sound effect erased
    /// for a wrong translation. With this on,
    /// the script arm fires for dialogue too: the balloon keeps its drawn ink
    /// and the lettering side (behind the SAME flag, `labels.rs`) refuses the
    /// wrong word. **Never enable one half without the other**: refusal alone
    /// is a blank hole, withdrawal alone letters onto un-erased ink.
    ///
    /// The struct default is `false`; the SHIPPING default lives in `cli.rs`
    /// and is **ON**, adopted on a census.
    #[serde(rename = "leave-misread-bubbles")]
    pub leave_misread_bubbles: bool,

    /// Whether a declared-Korean read with NO hangul at all (and at least three
    /// scripted letters) counts as a script mismatch.
    ///
    /// The ratio arm divides foreign counts by `scripted()`, which INCLUDES
    /// latin, so an all-Latin misread of drawn hangul ("Hwak", "OUBA", "ZGM")
    /// can never reach `FOREIGN_SHARE`, and `ga 大` scores 1/3, under the 0.34
    /// bar. Korean is written in hangul; a read of three or more scripted
    /// letters carrying none of it is not a Korean read. Authentic English
    /// display art on a Korean page is REFUSED-AS-DRAWN under this rule, which
    /// for a Latin target language is lossless -- the art already says it.
    ///
    /// The struct default is `false`; the SHIPPING default lives in `cli.rs`
    /// and is **ON**, with its pair.
    #[serde(rename = "korean-script-strict")]
    pub korean_script_strict: bool,

    /// Whether a DIALOGUE-role read the misread lever refuses on a declared
    /// KOREAN page buys ONE re-read from the reserve engine (PaddleOCR-VL) on a
    /// +24 px padded crop, admitted iff the re-read is hangul-majority --
    /// script membership, never a confidence comparison (score-monotone
    /// selection rules were measured and rejected).
    ///
    /// The lever above keeps the drawn ink instead of lettering a wrong word,
    /// which is correct and still short of the bar: the balloon should letter
    /// English. The read is the recoverable half -- on the censused 19-region
    /// refusal population the reserve engine on padded crops recovered 4 of
    /// the 6 genuine dialogue targets EXACTLY where same-engine re-reads
    /// recovered none. SFX/onomatopoeia and
    /// punctuation-only refusals are excluded: the official release ships
    /// those as drawn, and re-admitting them regresses.
    ///
    /// The struct default is `false`; the shipping default lives in `cli.rs`
    /// and is **ON**, adopted on the rendered evidence (one page changed across
    /// both censused chapters, zero re-admissions).
    #[serde(rename = "reread-refused-dialogue")]
    pub reread_refused_dialogue: bool,

    /// Whether a region over `large_crop_ocr_max_area` also contributes **no
    /// pixels to the erase mask**, rather than only being refused by OCR.
    ///
    /// **This is the arm that can actually save the artwork, and it is a
    /// different decision from `skip_implausible_regions`.** That one is "do not
    /// read it"; this one is "do not erase it". The erase mask is written by the
    /// detection stage from the raw detections, before OCR has run at all, so a
    /// box refused by the reader is still erased and inpainted -- on one manga
    /// test page a 1210x1247 `onomatopoeia` box on an 844x1200 page covers a
    /// whole drawn panel, and every arm measured before this flag destroyed it.
    /// Skipping the read only stopped English being lettered on top of the damage.
    ///
    /// Kept as its own field precisely because they are separable: an A/B has to
    /// be able to name each end, and "refuse to read but keep erasing" is a
    /// legitimate configuration (a huge box whose glyphs really are huge reads
    /// badly on a 224 square but still needs erasing).
    ///
    /// # ON by default, and that panel is saved
    ///
    /// `cli.rs` is the authority for the shipping default. With this on, the
    /// panel is **preserved, max|d| = 1 across the whole image** -- a JPEG
    /// round-trip, not an erase.
    ///
    /// The risk it carries is the mirror of the OCR one: a legitimately enormous
    /// region -- a full-page sign, a splash-page effect -- would stop being erased
    /// and would render with the source kana still under the English. Nothing like
    /// it has been seen in ~2,441 measured regions, where the largest legitimate
    /// box is 0.30x of its page, but "not seen" is not "measured".
    ///
    /// # It saves far less than it looks like it should
    ///
    /// Over a full baseline benchmark run, 23 regions were refused by the
    /// reader for size and **19 were erased anyway**. This gate spared only 4 --
    /// and those 4 are the ones with the largest *reported* boxes. It is not
    /// misbehaving: `mask_includes` is handed `detection.bbox` **raw**, while the
    /// reader is handed the same box **rotated and re-boxed**, which can only be
    /// larger. So the two stages disagree about the same 23 boxes, and the other
    /// 19 are genuinely under the ceiling as far as this gate can see.
    ///
    /// **The lever that should have caught those 19 is `withdraw_illegible_masks`,
    /// and `skip_implausible_regions` starves it**: `write_illegible_veto` walks
    /// OCR *results*, and a region refused for size never becomes one. Turning the
    /// reader gate on removed the only mechanism that still worked on this
    /// population. Both flags default ON, so this interlock is the shipping state.
    ///
    /// Only the erase mask (`text-mask`) is gated. `bubble-mask` is a balloon
    /// polygon the GUI reads for layout and nothing inpaints from, so refusing a
    /// detection there would change editing behaviour to buy nothing.
    #[serde(rename = "skip-implausible-masks")]
    pub skip_implausible_masks: bool,

    /// Whether a region **OCR could not read** is taken back out of the erase
    /// mask before the inpainter runs.
    ///
    /// **This is the same question `skip_implausible_masks` asks, asked with the
    /// answer instead of a proxy.** That one refuses a box for being implausibly
    /// large; this one refuses it for containing nothing that could be text in
    /// any script. They stay separate fields because they catch different
    /// populations: a huge box may hold real glyphs, and a small one may hold a
    /// bird.
    ///
    /// Measured on three Chinese webtoon slices with this off: **15 regions, 0
    /// lettered, 15 erased.** RF-DETR labelled five birds `onomatopoeia`,
    /// PaddleOCR-VL read them as `↓ Y V √ 1`, `labels::hide_implausible` refused
    /// every one -- and the birds were erased out of the sky anyway, because that
    /// refusal happens after `execute` returns while the mask is written during
    /// detection. 0.901% of one page destroyed for zero text.
    ///
    /// **On by default.** The rule this project sets for a pixel-changing default
    /// is a render showing the artwork preserved with nothing else regressing,
    /// and unlike `skip_implausible_masks` that render exists. The residual risk
    /// is the mirror image: a real effect whose OCR came back as one Latin letter
    /// would stop being erased and would render with its source glyph under the
    /// English. On the measured pages every genuine effect read as a CJK
    /// character (`米 共 大 明 鸣 嫩`) and every false one did not, so the two
    /// populations separate on script rather than on length.
    ///
    /// Only `text-mask` is affected. `bubble-mask` is a balloon polygon the GUI
    /// reads for layout and nothing inpaints from.
    ///
    /// **Inert without OCR.** `clean_only` runs detection and inpainting with no
    /// OCR stage, so there is no verdict to act on and the mask is unfiltered --
    /// a `clean_only` measurement does NOT show this working. Measure it on the
    /// ordinary path.
    #[serde(rename = "withdraw-illegible-masks")]
    pub withdraw_illegible_masks: bool,

    /// Whether a region `skip_implausible_regions` refused for SIZE is also taken
    /// back out of the erase mask, rather than only being kept away from OCR.
    ///
    /// **This breaks an interlock between two flags that both default ON.**
    /// `skip_implausible_regions` drops the box before it can become an OCR result;
    /// `withdraw_illegible_masks` only walks results. So the reader's refusal
    /// deletes the very verdict the withdrawal needs, and the box is erased with
    /// nothing painted back. Measured over a benchmark run, 22 such
    /// boxes across 457 pages of manhua, manhwa and webtoon: **18 erased.**
    ///
    /// **Turning the reader gate off instead is NOT the fix, and was measured.** It
    /// rescues only **8 of the 18** — exactly those whose read happens to fire
    /// `withdraw_from_mask` — leaves 10 erased and slightly worse, and newly breaks
    /// a page that was pixel-perfect, painting a wrong English word across intact
    /// artwork to render one character. That the 8 are precisely the withdrawal's own
    /// population is the clue this field acts on: **the withdrawal is the right
    /// mechanism, and the read is not needed to justify it.**
    ///
    /// **ON by default.** The render fixed **18 of 18 with 0 broken and nothing
    /// lettered**. Collateral is five pages at max|d| = 1 with 0 pixels over
    /// threshold — the byte-identical control arm's own floor.
    ///
    /// The deciding argument: **the erase buys nothing.**
    /// Neither arm letters anything into these boxes, so the question is the
    /// original drawing against whatever LaMa invents — and the fill is a visible
    /// defect in 4 of 5, turning half a page into a white smear on one and spilling
    /// past the panel border into the blank gutter on another.
    ///
    /// The cost is an untranslated source effect left visible, which is what the
    /// pipeline already does with these boxes: it has never translated them. So this
    /// decides whether the *drawing* survives, not whether the reader gets a
    /// translation. `false` is the other arm.
    #[serde(rename = "withdraw-unread-masks")]
    pub withdraw_unread_masks: bool,

    /// Whether a masked blob cut by the inpaint tile grid is shown to the model
    /// as one shape rather than as two half-shapes.
    ///
    /// **The residue this removes is LaMa's own output, not un-erased ink.** The
    /// tiler walks a fixed 512 grid, and each tile's crop reaches
    /// `TILE_CONTEXT = 128` past its core. `crop_tile_mask` used to mark only the
    /// core, so the half of a blob belonging to the NEIGHBOURING core arrived as
    /// unmasked context touching the hole -- real image, as far as the model is
    /// concerned -- and it continued the stroke inward. The next tile then read
    /// those hallucinated pixels back out of `output` and continued them again.
    ///
    /// Measured on one test webtoon page, residue as a share of
    /// the effect's own ink by distance to the nearest grid line: **42.5% at
    /// 0-32px, 45.9% at 32-64, 15.5% at 64-128, 1.4% beyond 128** -- a cliff
    /// exactly at `TILE_CONTEXT`, which is the reach of the mechanism.
    ///
    /// **The erased footprint is bit-identical either way.** The flood only sets
    /// pixels already `>= 127` in the page mask, and `composite_generated` writes
    /// only inside the core and only where that mask is set. It cannot erase one
    /// pixel of artwork that is not already erased -- which matters here, because
    /// `withdraw_illegible_masks` above exists to STOP artwork being erased and a
    /// residue fix that gave that back would be a net regression.
    #[serde(rename = "seam-safe-erase")]
    pub seam_safe_erase: bool,

    /// Ask libtorch to return its cached CUDA segments to the driver after the
    /// residency sweep unloads a stage. `None` leaves the cache alone, which is
    /// upstream's behaviour and what every caller had before this existed.
    ///
    /// **What it repairs.** Unloading a stage drops its weights, but torch's
    /// caching allocator keeps every segment it ever grew: it has no automatic
    /// return path, and until now this tree had no way to ask for one -- there
    /// was no `emptyCache` anywhere in it. On Windows `resources::windows`
    /// budgets on DXGI `QueryVideoMemoryInfo`'s `CurrentUsage`, a *driver*
    /// figure, so those cached-but-free bytes count as used. One outsized input
    /// -- a mis-segmented region bigger than the page, a webtoon seam crop --
    /// raises the high-water mark by gigabytes that never come back, and
    /// `admission_plan` then sees a card too full to leave anything resident.
    /// From there `reservation` charges an unloaded stage its `peak_bytes`
    /// rather than its `workspace_bytes`, so the demand to re-admit it is
    /// several times what freeing it released and the eviction latches on.
    ///
    /// **Why it is optional rather than always on.** `emptyCache` releases a
    /// segment only when no live block remains inside it, so a fragmented cache
    /// can hand back a fraction of what it holds; and it synchronizes the device
    /// on the way. Whether that trade is worth it is a measurement, not an
    /// argument, so both arms have to be runnable. The server defaults it on
    /// (`--release-cached-vram`); this crate stays opinionless for the same
    /// reason `large_crop_ocr_px` does.
    ///
    /// Ignored on a device torch addresses as CPU -- there is no cache to empty.
    #[serde(rename = "release-cached-vram")]
    pub release_cached_vram: Option<bool>,

    /// Ask a stage whether the page has anything for it *before* paging its
    /// weights in, and skip the stage outright when the answer is no. `false`
    /// keeps the historical order -- load, then discover -- which is what every
    /// caller had before this existed.
    ///
    /// Only a stage that overrides `StageProcessor::has_work` can answer, and
    /// today that is translation alone; every other stage answers `true` and is
    /// unaffected either way.
    ///
    /// **Why it is worth anything at all.** `stage_runner` loads unconditionally
    /// and `Translator::translate` returns early on an empty segment list, so a
    /// page with no text pays a full cold load of the local LLM -- 16.5 GiB --
    /// to hand it nothing. That is free when the weights happen to be resident,
    /// and costs a cold start every time residency has evicted them, which under
    /// memory pressure is every page.
    ///
    /// **What it is worth is a property of the page set, not of the code.**
    /// Counted from two measured runs: 80 of 219 slices of a webtoon chapter
    /// (36.5%) yield no region at all, 66 of them paid the load, and those 66
    /// cost 686 s of a 2351 s run. The same count over 213 pages of a manga
    /// volume is 8 (3.8%), all of which happened to find the weights resident,
    /// for 0.6 s of 1386 s. Whole pages are almost never empty; a strip sliced
    /// into fixed-height images is a third empty. Expect a webtoon-shaped win.
    ///
    /// The server defaults it on (`--no-skip-empty-stages` turns it off). This
    /// crate stays opinionless, for the same reason `large_crop_ocr_px` does.
    #[serde(rename = "skip-empty-stages")]
    pub skip_empty_stages: bool,

    /// Write the settled per-region detection masks and the final assembled
    /// inpaint masks as PNGs under this directory, one file per object, named
    /// by page, region index, label and page-space window. `None` — the
    /// default, and the only shipped state — writes nothing and changes
    /// nothing.
    ///
    /// A debug instrument, not a feature. The sampler's and the eraser's inputs
    /// are invisible from the wire (`KoharuLayoutDetection.mask` is
    /// `skip_serializing`), and a design that depends on what the RF-DETR mask
    /// covers has to be able to SEE it. A dump failure is a `tracing::warn!`,
    /// never a stage
    /// error: an instrument must not kill the run it is instrumenting.
    #[serde(rename = "debug-mask-dir")]
    pub debug_mask_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model", deny_unknown_fields)]
pub enum DetectionModel {
    #[serde(rename = "koharu-layout-rfdetr-seg-2xl")]
    KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model", deny_unknown_fields)]
pub enum OcrModel {
    #[serde(rename = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[serde(rename = "manga-ocr")]
    MangaOcr,
    #[serde(rename = "baberu-ocr")]
    BaberuOcr,
    /// Any vision model Ollama serves, over its NATIVE `/api/chat`.
    ///
    /// **Names the ROUTE, not a model** -- deliberately, so the variant does not
    /// go stale when the model behind it changes.
    ///
    /// The only engine here with no weights in this process. Koharu cannot send an
    /// image to a remote model through the OpenAI-shaped path in
    /// `koharu-translator` -- its `Message` is `{ role, content:
    /// &str }` with no content-parts array. **That limit does not bind here:** that struct is
    /// private and file-local, and Ollama's native API carries images in a SEPARATE
    /// `images: [base64]` field on the message rather than in content parts. This
    /// variant writes its own request type and never touches the translator's.
    #[serde(rename = "ollama-vision")]
    OllamaVision,
    /// HunyuanOCR v1.5 behind a local sidecar, over the same Ollama-native
    /// protocol as `ollama-vision`. **The default, with `paddleocr-vl-1.6` as
    /// the reserve** -- still wired, still selectable, no longer default. Unlike
    /// `ollama-vision` this names the MODEL, not the route: the sidecar serves
    /// exactly one model and the version is part of the measurement record.
    #[serde(rename = "hunyuan-ocr-1.5")]
    HunyuanOcr1_5,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model", deny_unknown_fields)]
pub enum InpaintingModel {
    #[serde(rename = "lama")]
    LaMa {},
    #[serde(rename = "aot-inpainting")]
    AotInpainting {},
    #[serde(rename = "flux2-klein")]
    Flux2Klein(Flux2KleinConfig),
    #[serde(rename = "rorem-mixed")]
    RoremMixed(RoremMixedConfig),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_select_one_processor_for_each_phase() {
        let config = PipelineConfig::default();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::HunyuanOcr1_5));
        assert!(matches!(config.inpainting, InpaintingModel::LaMa {}));
    }

    /// Pins BOTH arms of the OCR default, so neither can silently
    /// revert: the default IS HunyuanOCR, and the demoted engine still parses --
    /// "reserve" means wired and selectable, not removed.
    #[test]
    fn hunyuan_is_the_default_and_paddle_stays_wired() {
        assert!(matches!(
            PipelineConfig::default().ocr,
            OcrModel::HunyuanOcr1_5
        ));

        let hunyuan: PipelineConfig =
            toml::from_str("[ocr]\nmodel = \"hunyuan-ocr-1.5\"").unwrap();
        assert!(matches!(hunyuan.ocr, OcrModel::HunyuanOcr1_5));

        let reserve: PipelineConfig =
            toml::from_str("[ocr]\nmodel = \"paddleocr-vl-1.6\"").unwrap();
        assert!(matches!(reserve.ocr, OcrModel::PaddleOcrVl1_6));
    }

    #[test]
    fn parses_phase_keyed_processor_configuration() {
        let config: PipelineConfig = toml::from_str(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [ocr]
                model = "baberu-ocr"

                [inpainting]
                model = "rorem-mixed"

                [processor."rorem-mixed"]
                prompt = "Remove the lettering."
                negative_prompt = "letters, words"
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::BaberuOcr));
        assert!(matches!(
            config.inpainting(),
            Ok(InpaintingModel::RoremMixed(config))
                if config.prompt == "Remove the lettering."
                    && config.negative_prompt == "letters, words"
        ));
    }

    #[test]
    fn missing_slots_use_defaults() {
        let config = toml::from_str::<PipelineConfig>("").unwrap();

        assert_eq!(config, PipelineConfig::default());
    }

    #[test]
    fn rejects_legacy_processor_configuration() {
        let result = toml::from_str::<PipelineConfig>(
            r#"
                [[processors]]
                model = "comic_layout_yolo26s"
                enabled = false

                [[processors]]
                model = "mask_fusion"
            "#,
        );

        assert!(result.is_err());
    }

    #[test]
    fn rejects_unknown_model_configuration_fields() {
        let result = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"
                legacy_threshold = 0.5

                [ocr]
                model = "paddleocr-vl-1.6"
                legacy_language = "ja"

                [inpainting]
                model = "lama"
                legacy_resolution = 1024
            "#,
        );

        assert!(result.is_err());
    }

    #[test]
    fn parses_detection_and_generative_inpainting_options() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting]
                model = "flux2-klein"

                [processor."koharu-layout-rfdetr-seg-2xl"]
                text_threshold = 0.25
                bubble_threshold = 0.45
                panel_threshold = 0.55

                [processor."flux2-klein"]
                prompt = "Reconstruct the illustration without text."
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection().unwrap(),
            DetectionModel::KoharuLayoutRFDetrSeg2XL(config)
                if config.text_threshold == Some(0.25)
                    && config.bubble_threshold == Some(0.45)
                    && config.panel_threshold == Some(0.55)
        ));
        assert!(matches!(
            config.inpainting().unwrap(),
            InpaintingModel::Flux2Klein(config)
                if config.prompt == "Reconstruct the illustration without text."
        ));
    }

    #[test]
    fn keeps_profiles_separate_from_active_stage_selection() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting]
                model = "flux2-klein"

                [processor."flux2-klein"]
                prompt = "saved prompt"
            "#,
        )
        .unwrap();

        let InpaintingModel::Flux2Klein(config) = config.inpainting().unwrap() else {
            panic!("expected FLUX profile")
        };
        assert_eq!(config.prompt, "saved prompt");
    }

    #[test]
    fn serializes_model_profiles_under_processor() {
        let config = PipelineConfig {
            detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig {
                text_threshold: Some(0.25),
                ..Default::default()
            }),
            ocr: OcrModel::PaddleOcrVl1_6,
            translation: TranslationConfig::default(),
            inpainting: InpaintingModel::Flux2Klein(Flux2KleinConfig {
                prompt: "Keep the line art.".to_owned(),
                ..Default::default()
            }),
            processor: ProcessorConfig::default(),
        };
        let document = toml::to_string(&config).unwrap();
        assert!(document.contains("[detection]\nmodel = \"koharu-layout-rfdetr-seg-2xl\""));
        assert!(document.contains("[processor.koharu-layout-rfdetr-seg-2xl]"));
        assert!(document.contains("[processor.flux2-klein]"));
        assert!(document.contains("[translation]"));
        assert!(!document.contains("prompt = \"Keep the line art.\"\n[inpainting]"));

        let restored = toml::from_str::<PipelineConfig>(&document).unwrap();
        assert!(matches!(
            restored.inpainting().unwrap(),
            InpaintingModel::Flux2Klein(config) if config.prompt == "Keep the line art."
        ));
    }

    #[test]
    fn rejects_removed_inpainting_options() {
        for (name, source) in [
            (
                "lama",
                r#"
                [inpainting]
                model = "lama"
                hd_strategy = "resize"
            "#,
            ),
            (
                "aot-inpainting",
                r#"
                [inpainting]
                model = "aot-inpainting"
                max_side = 1024
            "#,
            ),
            (
                "flux2-klein",
                r#"
                [inpainting]
                model = "flux2-klein"
                strength = 0.5
            "#,
            ),
            (
                "rorem-mixed",
                r#"
                [inpainting]
                model = "rorem-mixed"
                resolution = 1024
            "#,
            ),
        ] {
            assert!(
                toml::from_str::<PipelineConfig>(source).is_err(),
                "{name} accepted a removed option"
            );
        }
    }
}
