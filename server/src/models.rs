//! Runtime parsing of the model-selection fields the extension sends.
//!
//! `run.rs` gets these from clap `ValueEnum`, which is no help here: our values
//! arrive as strings at request time. The accepted spellings are kept
//! byte-identical to koharu-pipeline's own `Deserialize` impl for
//! `PipelineConfig`, so the server and the config file never disagree.

use koharu_pipeline::{
    DetectionModel, Flux2KleinConfig, InpaintingModel, KoharuLayoutRFDetrSeg2XLConfig, OcrModel,
    RoremMixedConfig,
};
use koharu_renderer::HyphenationPolicy;
use koharu_translator::{Language, Provider};

use crate::error::{ApiError, clip};

/// How much of a rejected value is echoed back. Keeps the accepted-value list
/// inside the extension's 120-byte window, and stops a hostile page reflecting
/// a long string through the server.
const ECHO: usize = 16;

/// The extension's provider vocabulary, which is not Koharu's.
pub const PROVIDER_LOCAL: &str = "local";
pub const PROVIDER_OLLAMA: &str = "ollama";

/// The local translator this server pins unless `--llm` says otherwise.
///
/// `ModelSelection::default()` names a different model again and sets a
/// quantization, so it is deliberately not used. This no longer agrees with
/// `run.rs`'s own default either -- that is still `gemma4-31b-it`, and the two
/// are allowed to differ because only this one is what a reader's page goes
/// through.
///
/// **Chosen over the previous default, `gemma4-31b-it`**, after a five-arm sweep
/// over 168 pages, every arm run twice. The incumbent stays fully reachable: `--llm
/// gemma4-31b-it`, the popup's Model field, or the `llm` form field on a single
/// request.
///
/// **Four independent instruments FAILED to separate the two on quality**, and
/// that is the whole basis of the switch:
///
/// * a blind LLM-judged panel over 36 items -- paired **+0.09**, bootstrap CI
///   **[-0.15, +0.33]**;
/// * a third-party human translation, read against all four sources -- every
///   gap's CI spans zero;
/// * a name-consistency metric;
/// * a pixel comparison of 16 rendered pages -- identical fit behaviour,
///   identical lettering, only the wording differs.
///
/// **Stated fairly, because "indistinguishable" is not "equal":** the MoE sits a
/// hair *below* the incumbent on all four sources, so the sign is consistent
/// even though no magnitude is separable. That is the honest case against. It
/// lost to three measured wins:
///
/// * **Speed.** 78 / 70 s against 221 / 242 s on the corpus -- but that is
///   TRANSLATOR-ONLY, because `replay.rs` runs no detection, OCR or inpainting.
///   End to end it is **0.69x the incumbent's page time, 0.64x warm**. Never
///   repeat the ~3x as a page-time claim.
/// * **VRAM.** 13.26 GB of weights against 16.09 GB, and a KV cache of
///   **220 KiB per token against 880** -- printed by llama.cpp at load and
///   reproduced exactly from the GGUF headers (30 blocks, 25 sliding at 8 KV
///   heads and 5 global at 2, against 60 / 50 at 16 and 10 at 4). That factor of
///   four is what pays for `MAX_STORY_PAIRS` in `cli.rs`.
/// * **Fit.** Its output is **2.6% shorter** over 511 segments (p90 63 chars
///   against 67), which is mildly kinder to auto-fit.
///
/// Sampling is not a confound: the catalog gives both descriptors the same
/// temperature 1.0 / top_k 64 / top_p 0.95.
pub const DEFAULT_LOCAL_MODEL: &str = "gemma4-26b-a4b-it";

/// `OcrModel` is a plain serde enum tagged on "model" with no `FromStr`, so the
/// spellings are matched by hand.
pub fn parse_ocr(value: &str) -> Result<OcrModel, ApiError> {
    match value {
        "paddleocr-vl-1.6" => Ok(OcrModel::PaddleOcrVl1_6),
        "manga-ocr" => Ok(OcrModel::MangaOcr),
        "baberu-ocr" => Ok(OcrModel::BaberuOcr),
        "ollama-vision" => Ok(OcrModel::OllamaVision),
        "hunyuan-ocr-1.5" => Ok(OcrModel::HunyuanOcr1_5),
        other => Err(ApiError::bad_request(format!(
            "bad ocr \"{}\"; use hunyuan-ocr-1.5, paddleocr-vl-1.6, manga-ocr, baberu-ocr, ollama-vision",
            clip(other, ECHO)
        ))),
    }
}

/// `parse_ocr` plus `--hunyuan-substitute`: a request for `hunyuan-ocr-1.5` is
/// served by `substitute` when one is set. The ONE place a wire name becomes an
/// engine -- the process default (`Cli::resolve`) and every request
/// (`routes::selection_from`) both call it, so the config that loads, and the
/// stage telemetry read back from it, name the engine that actually runs. The
/// one exception is on purpose: `Cli::resolve` reads the substitute's own value
/// with `parse_ocr` directly, because the substitute must not be substituted.
pub fn resolve_ocr(value: &str, substitute: Option<&OcrModel>) -> Result<OcrModel, ApiError> {
    Ok(match (parse_ocr(value)?, substitute) {
        (OcrModel::HunyuanOcr1_5, Some(engine)) => engine.clone(),
        (model, _) => model,
    })
}

/// `LaMa` and `AotInpainting` are struct variants with no fields, so the braces
/// are mandatory. The two generative variants carry a config whose `Default`
/// supplies a non-empty prompt; the inpainting stage only ever checks those
/// prompts for NUL, so `::default()` is always accepted.
/// `auto` | `normal` | `last-resort` | `disabled`.
///
/// `auto` is `None`: the renderer's own per-layer rule, which is what shipped
/// before this flag existed. Kept reachable so the old behaviour can be
/// reproduced, not because it is good -- it leaves free-standing text on
/// `normal`, which is the misfiring case.
pub fn parse_hyphenation(value: &str) -> Result<Option<HyphenationPolicy>, ApiError> {
    match value {
        "auto" => Ok(None),
        "normal" => Ok(Some(HyphenationPolicy::Normal)),
        "last-resort" => Ok(Some(HyphenationPolicy::LastResort)),
        "disabled" => Ok(Some(HyphenationPolicy::Disabled)),
        other => Err(ApiError::bad_request(format!(
            "bad hyphenation \"{}\"; use auto, normal, last-resort, disabled",
            clip(other, ECHO)
        ))),
    }
}

/// `rorem_steps` must be the SAME value everywhere this is called. It is part of
/// the config the pipeline compares to decide whether a request needs a reload,
/// so a per-request parse that disagreed with the process default would rebuild
/// the `StageRunner` on every page -- discarding the residency profiles and
/// making any timing taken from it meaningless.
pub fn parse_inpainting(
    value: &str,
    rorem_steps: Option<i32>,
    flux_strength: Option<f64>,
) -> Result<InpaintingModel, ApiError> {
    match value {
        "lama" => Ok(InpaintingModel::LaMa {}),
        "aot-inpainting" => Ok(InpaintingModel::AotInpainting {}),
        "flux2-klein" => Ok(InpaintingModel::Flux2Klein(Flux2KleinConfig {
            strength: flux_strength
                .filter(|value| *value > 0.0 && *value <= 1.0)
                .unwrap_or(Flux2KleinConfig::default().strength),
            ..Flux2KleinConfig::default()
        })),
        "rorem-mixed" => Ok(InpaintingModel::RoremMixed(RoremMixedConfig {
            num_inference_steps: rorem_steps
                .filter(|steps| *steps > 0)
                .unwrap_or(RoremMixedConfig::default().num_inference_steps),
            ..RoremMixedConfig::default()
        })),
        other => Err(ApiError::bad_request(format!(
            "bad inpainting \"{}\"; use lama, aot-inpainting, flux2-klein, rorem-mixed",
            clip(other, ECHO)
        ))),
    }
}

/// The caller-stated FORMAT axis of the profile split: what kind of publication
/// the site serves. Deliberately not a language and never derived from one -- a
/// Japanese webtoon was measured at exactly the 690px width of a Korean manhwa
/// test corpus, so neither axis may imply the other.
///
/// Inert beyond validation and a log line until a lever consumes it; the
/// change that wires a consumer must add `profile` to the extension's
/// `settingsFingerprint` in the same patch (`extension/cache.js` says why).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageProfile {
    Manga,
    Webtoon,
}

/// Refuses garbage the way every declared field here does: an ignored
/// declaration is the quiet failure this axis exists to remove.
pub fn parse_profile(value: &str) -> Result<PageProfile, ApiError> {
    match value {
        "manga" => Ok(PageProfile::Manga),
        "webtoon" => Ok(PageProfile::Webtoon),
        other => Err(ApiError::bad_request(format!(
            "bad profile \"{}\"; use manga or webtoon",
            clip(other, ECHO)
        ))),
    }
}

/// A boolean form field, named in its own error message.
///
/// Strict like `parse_provider`, and for the same reason -- the extension is the
/// only caller and sends exactly one of these two spellings, so anything else is
/// a mistake worth a 400 rather than a guess. Reading an unexpected value as
/// `false` would silently do the opposite of what was asked, which for
/// `skip_inpainting` means erasing the art the caller wanted kept.
pub fn parse_flag(field: &str, value: &str) -> Result<bool, ApiError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(ApiError::bad_request(format!(
            "bad {} \"{}\"; use true or false",
            field,
            clip(other, ECHO)
        ))),
    }
}

/// `Language` derives strum's `EnumString` with `ascii_case_insensitive`, so
/// this accepts the BCP-47 tag, the bare subtag and the English name alike.
/// Its own error says only "Matching variant not found", so it is replaced --
/// and 42 variants will not fit the window, so the message names a few.
pub fn parse_language(value: &str) -> Result<Language, ApiError> {
    value.parse::<Language>().map_err(|_| {
        ApiError::bad_request(format!(
            "bad target_language \"{}\"; use a BCP-47 tag: en-US, ja-JP, zh-CN, es-ES, fr-FR, etc.",
            clip(value, ECHO)
        ))
    })
}

/// Never `value.parse::<Provider>()`. That is case-sensitive, rejects the
/// extension's literal "ollama", and would silently accept eleven cloud
/// backends the popup never offers.
pub fn parse_provider(value: &str) -> Result<Provider, ApiError> {
    match value {
        PROVIDER_LOCAL => Ok(Provider::Local),
        PROVIDER_OLLAMA => Ok(Provider::OpenAiCompatible),
        other => Err(ApiError::bad_request(format!(
            "bad provider \"{}\"; use ollama or local",
            clip(other, ECHO)
        ))),
    }
}

/// Canonicalises the wire spelling so a pinned provider can be reported back
/// using the same vocabulary the popup uses.
pub fn provider_wire_name(value: &str) -> Option<&'static str> {
    match value {
        PROVIDER_LOCAL => Some(PROVIDER_LOCAL),
        PROVIDER_OLLAMA => Some(PROVIDER_OLLAMA),
        _ => None,
    }
}

/// Not wire-driven: `DetectionModel` has exactly one variant and the extension
/// sends no detection field.
///
/// `translate_sfx` is a process-wide default rather than a per-request field for
/// the same reason: it changes what detection *writes into the scene*, so a
/// request that flipped it would need a pipeline reload, and the reload would
/// discard the residency profiles.
#[must_use]
pub fn detection(
    translate_sfx: bool,
    refine_text_mask: bool,
    onomatopoeia_threshold: Option<f32>,
    mask_scale: Option<f32>,
    ink_mask: bool,
) -> DetectionModel {
    DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig {
        translate_sfx: Some(translate_sfx),
        // `None` and `Some(false)` mean the same thing to the stage; sending the
        // explicit value keeps `/status` and a config dump honest about which
        // arm this process is running.
        refine_text_mask: Some(refine_text_mask),
        onomatopoeia_threshold,
        mask_scale,
        ink_mask: Some(ink_mask),
        ..KoharuLayoutRFDetrSeg2XLConfig::default()
    })
}

/// The model name a stage reports through `Progress`, used to verify after the
/// fact that the reload actually landed.
#[must_use]
pub fn ocr_model_name(model: &OcrModel) -> &'static str {
    match model {
        OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
        OcrModel::MangaOcr => "manga-ocr",
        OcrModel::BaberuOcr => "baberu-ocr",
        OcrModel::OllamaVision => "ollama-vision",
        OcrModel::HunyuanOcr1_5 => "hunyuan-ocr-1.5",
    }
}

#[must_use]
pub fn inpainting_model_name(model: &InpaintingModel) -> &'static str {
    match model {
        InpaintingModel::LaMa {} => "lama",
        InpaintingModel::AotInpainting {} => "aot-inpainting",
        InpaintingModel::Flux2Klein(_) => "flux2-klein",
        InpaintingModel::RoremMixed(_) => "rorem-mixed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::BUDGET;

    /// `ollama-vision` is not just another spelling: it is the first engine on
    /// this path with NO weights in this process, reached over HTTP. The two
    /// things a rename could silently break are the wire spelling the extension
    /// sends and the name reported to `/status`, so both are pinned here.
    ///
    /// **This asserts the round trip, not the two halves separately.** A test
    /// that only checked `parse_ocr` would pass with `ocr_model_name` still
    /// returning a stale string, and `/status` would then disagree with what the
    /// caller asked for -- which is how a reload fires on every first translate.
    #[test]
    fn the_ollama_vision_engine_survives_the_round_trip_the_caller_actually_makes() {
        let parsed = parse_ocr("ollama-vision").expect("the popup offers this value");
        assert_eq!(parsed, OcrModel::OllamaVision);
        assert_eq!(ocr_model_name(&parsed), "ollama-vision");
    }

    /// The rejection message is the only place a caller learns what IS accepted,
    /// so it has to list the new engine too.
    /// Same round-trip pin for the DEFAULT engine. If either half goes
    /// stale, /status disagrees with the caller and a reload fires on every
    /// first translate -- for the DEFAULT engine, that is every session.
    #[test]
    fn the_hunyuan_default_survives_the_same_round_trip() {
        let parsed = parse_ocr("hunyuan-ocr-1.5").expect("the default must parse");
        assert_eq!(parsed, OcrModel::HunyuanOcr1_5);
        assert_eq!(ocr_model_name(&parsed), "hunyuan-ocr-1.5");
    }

    #[test]
    fn the_rejection_message_names_every_accepted_engine() {
        let Err(error) = parse_ocr("no-such-engine") else {
            panic!("an unknown engine must be refused");
        };
        let text = format!("{error:?}");
        for engine in [
            "hunyuan-ocr-1.5",
            "paddleocr-vl-1.6",
            "manga-ocr",
            "baberu-ocr",
            "ollama-vision",
        ] {
            assert!(text.contains(engine), "{engine} missing from: {text}");
        }
    }

    #[test]
    fn accepts_every_value_the_popup_offers() {
        for value in [
            "hunyuan-ocr-1.5",
            "paddleocr-vl-1.6",
            "manga-ocr",
            "baberu-ocr",
            "ollama-vision",
        ] {
            assert!(parse_ocr(value).is_ok(), "{value}");
        }
        for value in ["lama", "aot-inpainting", "flux2-klein", "rorem-mixed"] {
            assert!(parse_inpainting(value, None, None).is_ok(), "{value}");
        }
        for value in [
            "en-US", "es-ES", "fr-FR", "de-DE", "pt-BR", "ru-RU", "zh-CN", "ko-KR",
        ] {
            assert!(parse_language(value).is_ok(), "{value}");
        }
        assert_eq!(
            parse_provider("ollama").unwrap(),
            Provider::OpenAiCompatible
        );
        assert_eq!(parse_provider("local").unwrap(), Provider::Local);
        assert!(parse_flag("skip_inpainting", "true").unwrap());
        assert!(!parse_flag("skip_inpainting", "false").unwrap());
    }

    #[test]
    fn an_unreadable_flag_is_refused_not_read_as_false() {
        // Defaulting would erase the artwork the caller asked to keep, which is
        // the one outcome this switch exists to prevent.
        for value in ["1", "0", "yes", "TRUE", "", "on"] {
            assert!(parse_flag("skip_inpainting", value).is_err(), "{value}");
        }
    }

    #[test]
    fn a_flag_error_names_the_field_that_was_wrong() {
        // One parser serves several fields, so the message has to say which one
        // -- "use true or false" alone does not tell a caller where to look.
        let message = parse_flag("clean_only", "maybe").unwrap_err().message;
        assert!(message.contains("clean_only"), "{message}");
    }

    #[test]
    fn a_profile_is_one_of_two_words_and_garbage_is_refused() {
        // The two spellings the extension's `resolveSite` can emit,
        // and nothing else: an ignored declaration is the quiet failure the
        // axis exists to remove, so "Webtoon", "strip" and "" are 400s rather
        // than silently-undeclared.
        assert_eq!(parse_profile("manga").unwrap(), PageProfile::Manga);
        assert_eq!(parse_profile("webtoon").unwrap(), PageProfile::Webtoon);
        for value in ["Webtoon", "MANGA", "strip", "paged", "", "true"] {
            assert!(parse_profile(value).is_err(), "{value}");
        }
        let message = parse_profile("comic").unwrap_err().message;
        assert!(message.contains("manga or webtoon"), "{message}");
    }

    #[test]
    fn maps_values_to_the_right_variant() {
        assert!(matches!(parse_ocr("manga-ocr").unwrap(), OcrModel::MangaOcr));
        assert!(matches!(
            parse_inpainting("lama", None, None).unwrap(),
            InpaintingModel::LaMa {}
        ));
        assert!(matches!(
            parse_inpainting("flux2-klein", None, None).unwrap(),
            InpaintingModel::Flux2Klein(_)
        ));
        assert_eq!(parse_language("en-US").unwrap(), Language::English);
    }

    #[test]
    fn language_accepts_the_bare_subtag_in_any_case() {
        assert_eq!(parse_language("en").unwrap(), Language::English);
        assert_eq!(parse_language("EN-us").unwrap(), Language::English);
        assert_eq!(
            parse_language("pt-BR").unwrap(),
            Language::BrazilianPortuguese
        );
    }

    #[test]
    fn rejects_a_provider_koharu_knows_but_the_popup_does_not() {
        // Guards against anyone "simplifying" this into parse::<Provider>().
        assert!(parse_provider("openai-compatible").is_err());
        assert!(parse_provider("claude").is_err());
        assert!(parse_provider("Local").is_err());
    }

    #[test]
    fn errors_stay_inside_the_extensions_window() {
        let hostile = "x".repeat(4096);
        let multibyte = "あ".repeat(4096);
        for value in [hostile.as_str(), multibyte.as_str()] {
            for message in [
                parse_ocr(value).unwrap_err().message,
                parse_inpainting(value, None, None).unwrap_err().message,
                parse_language(value).unwrap_err().message,
                parse_provider(value).unwrap_err().message,
                parse_flag("skip_inpainting", value).unwrap_err().message,
                parse_flag("clean_only", value).unwrap_err().message,
            ] {
                assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
            }
        }
    }

    #[test]
    fn errors_name_the_accepted_values_before_the_cutoff() {
        let message = parse_inpainting("nope", None, None).unwrap_err().message;
        assert!(message[..message.len().min(BUDGET)].contains("rorem-mixed"));
    }

    #[test]
    fn model_names_match_the_wire_spellings() {
        assert_eq!(ocr_model_name(&OcrModel::BaberuOcr), "baberu-ocr");
        assert_eq!(
            inpainting_model_name(&InpaintingModel::AotInpainting {}),
            "aot-inpainting"
        );
    }
}
