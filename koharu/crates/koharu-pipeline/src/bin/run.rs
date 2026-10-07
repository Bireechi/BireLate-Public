use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, bail};
use clap::{Parser, ValueEnum};
use koharu_config::Config;
use koharu_pipeline::{
    Committer, DetectionModel, Flux2KleinConfig, InpaintingModel, KoharuLayoutRFDetrSeg2XLConfig,
    OcrModel, Operation, Pipeline, PipelineConfig, Progress, Request, RoremMixedConfig, Scope,
    StageOutput, TranslationConfig,
};
use koharu_renderer::{Compositor, RasterOptions, Rasterizer, RenderRequest, SceneRenderer};
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session};
use koharu_translator::{
    GenerationConfig, Language, ModelSelection, OpenAiCompatibleConfig, Provider, ProvidersConfig,
};
use url::Url;

/// Default when translating through Koharu's own bundled llama.cpp engine.
/// Model ids come from Koharu's registry, not from Ollama's tag namespace.
const DEFAULT_LOCAL_MODEL: &str = "gemma4-31b-it";

#[derive(Debug, Parser)]
#[command(version, about = "Run Koharu's complete in-process pipeline")]
struct Arguments {
    #[arg(short, long, value_name = "INPUT")]
    input: PathBuf,

    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    #[arg(long, value_enum, default_value = "koharu-layout-rfdetr-seg-2xl")]
    detection: DetectionChoice,

    #[arg(long, value_enum, default_value = "paddleocr-vl-1.6")]
    ocr: OcrChoice,

    #[arg(long = "font-family", value_name = "FAMILY")]
    font_families: Vec<String>,

    /// Sharpen the inpainting mask with manga-text-segmentation-2025.
    /// Off by default -- see `KoharuLayoutRFDetrSeg2XLConfig::refine_text_mask`
    /// for the measurements. `--refine-text-mask true` enables it, which is what
    /// makes an A/B of the mask move exactly one variable.
    #[arg(long, value_name = "BOOL")]
    refine_text_mask: Option<bool>,

    /// Probability above which a pixel is text in that refined mask.
    #[arg(long, value_name = "PROBABILITY")]
    text_mask_threshold: Option<f32>,

    /// Dilation applied to the refined mask, in pixels.
    #[arg(long, value_name = "PIXELS")]
    text_mask_padding_iterations: Option<u32>,

    #[arg(long, value_enum, default_value = "lama")]
    inpainting: InpaintingChoice,

    #[arg(long, default_value = "en-US")]
    target_language: Language,

    #[arg(long)]
    translation_instructions: Option<String>,

    /// Which backend runs the translation LLM.
    #[arg(long, value_enum, default_value = "local")]
    provider: ProviderChoice,

    /// Model name. For `--provider local` this is a Koharu registry id
    /// (e.g. gemma4-31b-it). For `--provider ollama` it is an Ollama tag
    /// (e.g. qwen3:8b) -- run `ollama list` to see what is available.
    #[arg(long)]
    llm: Option<String>,

    /// Override the OpenAI-compatible endpoint. Defaults to Ollama's
    /// http://localhost:11434/v1.
    #[arg(long, value_name = "URL")]
    base_url: Option<Url>,

    #[arg(long)]
    cpu: bool,
}

struct SessionCommitter<'a>(&'a mut Session);

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> Result<koharu_scene::Snapshot> {
        Ok(self.0.commit(output.patch)?.snapshot)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DetectionChoice {
    #[value(name = "koharu-layout-rfdetr-seg-2xl")]
    KoharuLayoutRFDetrSeg2XL,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OcrChoice {
    #[value(name = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[value(name = "manga-ocr")]
    MangaOcr,
    #[value(name = "baberu-ocr")]
    BaberuOcr,
    #[value(name = "ollama-vision")]
    OllamaVision,
    #[value(name = "hunyuan-ocr-1.5")]
    HunyuanOcr1_5,
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum ProviderChoice {
    /// Koharu's bundled llama.cpp engine, which loads its own weights.
    #[value(name = "local")]
    Local,
    /// An already-running Ollama server, via its OpenAI-compatible API.
    /// Avoids holding a second copy of a model in VRAM.
    #[value(name = "ollama")]
    Ollama,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InpaintingChoice {
    #[value(name = "lama")]
    LaMa,
    #[value(name = "aot-inpainting")]
    AotInpainting,
    #[value(name = "flux2-klein")]
    Flux2Klein,
    #[value(name = "rorem-mixed")]
    RoremMixed,
}

impl Arguments {
    /// Resolve the model name, which is namespaced differently per provider.
    fn model(&self) -> Result<String> {
        match (&self.llm, self.provider) {
            (Some(model), _) => Ok(model.clone()),
            (None, ProviderChoice::Local) => Ok(DEFAULT_LOCAL_MODEL.to_owned()),
            // Ollama tags are user-specific, so there is no sensible default.
            (None, ProviderChoice::Ollama) => bail!(
                "--llm is required with --provider ollama: pass an Ollama model tag \
                 (run `ollama list` to see them). Koharu registry ids such as \
                 {DEFAULT_LOCAL_MODEL} are only valid with --provider local."
            ),
        }
    }

    fn providers_config(&self) -> ProvidersConfig {
        let mut providers = ProvidersConfig::default();
        // Default already points at Ollama; only override when asked.
        if let Some(base_url) = self.base_url.clone() {
            providers.openai_compatible = OpenAiCompatibleConfig {
                base_url: Some(base_url),
            };
        }
        providers
    }

    fn pipeline_config(&self) -> Result<PipelineConfig> {
        Ok(PipelineConfig {
            detection: match self.detection {
                DetectionChoice::KoharuLayoutRFDetrSeg2XL => {
                    DetectionModel::KoharuLayoutRFDetrSeg2XL(
                        KoharuLayoutRFDetrSeg2XLConfig {
                            refine_text_mask: self.refine_text_mask,
                            text_mask_threshold: self.text_mask_threshold,
                            text_mask_padding_iterations: self.text_mask_padding_iterations,
                            ..KoharuLayoutRFDetrSeg2XLConfig::default()
                        },
                    )
                }
            },
            ocr: match self.ocr {
                OcrChoice::PaddleOcrVl1_6 => OcrModel::PaddleOcrVl1_6,
                OcrChoice::MangaOcr => OcrModel::MangaOcr,
                OcrChoice::BaberuOcr => OcrModel::BaberuOcr,
                OcrChoice::OllamaVision => OcrModel::OllamaVision,
                OcrChoice::HunyuanOcr1_5 => OcrModel::HunyuanOcr1_5,
            },
            translation: TranslationConfig {
                model: ModelSelection {
                    provider: match self.provider {
                        ProviderChoice::Local => Provider::Local,
                        ProviderChoice::Ollama => Provider::OpenAiCompatible,
                    },
                    model: Some(self.model()?),
                    quantization: None,
                },
                generation: GenerationConfig::default(),
                target_language: self.target_language,
                /* Nothing trustworthy said, which is exactly what `None` means
                 * here -- see the field's own doc. This CLI takes one image with
                 * no page context, so it cannot know the script and must not
                 * guess: inferring "not Japanese" from silence would fire the
                 * kana rule on pages it was never measured against. */
                source_language: None,
                instructions: self.translation_instructions.clone(),
                // OFF, matching the shipping default. This CLI is a
                // single-image probe, not an A/B harness, so levers it does
                // not expose stay at their defaults.
                containment_clause: false,
                segment_context: false,
            },
            inpainting: match self.inpainting {
                InpaintingChoice::LaMa => InpaintingModel::LaMa {},
                InpaintingChoice::AotInpainting => InpaintingModel::AotInpainting {},
                InpaintingChoice::Flux2Klein => {
                    InpaintingModel::Flux2Klein(Flux2KleinConfig::default())
                }
                InpaintingChoice::RoremMixed => {
                    InpaintingModel::RoremMixed(RoremMixedConfig::default())
                }
            },
            processor: Default::default(),
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    initialize_with_retry().await;

    let source = fs::read(&arguments.input)
        .with_context(|| format!("failed to read {}", arguments.input.display()))?;
    let decoded = image::load_from_memory(&source).context("failed to decode input image")?;
    let name = arguments
        .input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("input");
    let mut session = Session::memory()?;
    let mut page = None;
    let patch = session.snapshot().patch(|edit| {
        let id = edit.add_page(
            PageDraft::new(
                name,
                f64::from(decoded.width()),
                f64::from(decoded.height()),
            ),
            At::End,
        )?;
        edit.set_asset(
            id,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(source),
                image_media_type(&arguments.input),
                AssetMetadata {
                    width: Some(decoded.width()),
                    height: Some(decoded.height()),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        page = Some(id);
        Ok(())
    })?;
    session.commit(patch)?;
    let page = page.expect("page ID is assigned by the edit");

    let device = koharu_ml::device(arguments.cpu);
    let pipeline = Pipeline::from_config(
        Config::memory(arguments.pipeline_config()?),
        Config::memory(arguments.providers_config()),
        device,
    )?;
    let snapshot = session.snapshot();
    let mut committer = SessionCommitter(&mut session);
    let report = pipeline
        .execute(
            snapshot,
            Request {
                operation: Operation::Full,
                scope: Scope::Pages(vec![page]),
                progress: Some(Arc::new(|event| match event {
                    Progress::Finished { stage, elapsed, .. } => {
                        eprintln!("{stage} finished in {:.2}s", elapsed.as_secs_f64());
                    }
                    // Loud on purpose. These regions render in the source
                    // language, so with nothing printed the page just looks as
                    // though OCR missed them.
                    Progress::Untranslated {
                        entities,
                        segments,
                        truncated,
                        duplicate_ids,
                        out_of_range_ids,
                        ..
                    } => {
                        // No missed ids and still truncated means the cut landed
                        // inside the final segment's text: every region carries a
                        // translation, one of them just stops mid-word. Saying
                        // "0 of N came back untranslated" would read as an
                        // all-clear.
                        //
                        // `truncated` is now tested rather than assumed. This
                        // event also fires for a reply that merely mis-addressed
                        // an id, which can leave no missed ids at all, and
                        // blaming the token cap for that would send a reader
                        // after the wrong number entirely.
                        if entities.is_empty() && truncated {
                            eprintln!(
                                "warning: the model hit its token cap; the last of \
                                 {segments} regions is cut off mid-text"
                            );
                        } else if !entities.is_empty() {
                            eprintln!(
                                "warning: {} of {segments} regions came back untranslated{}",
                                entities.len(),
                                if truncated {
                                    " after the model hit its token cap"
                                } else {
                                    ""
                                }
                            );
                        }
                        /* Printed beside the count above rather than folded into
                         * it, because it answers the question that count cannot:
                         * an id answered twice steals another id's slot, so the
                         * victim appears in the line above as an ordinary miss
                         * and reads as a reply that stopped early. One says
                         * "raise the token budget", the other says the opposite. */
                        if duplicate_ids > 0 || out_of_range_ids > 0 {
                            eprintln!(
                                "warning: the model answered {duplicate_ids} id(s) more than once \
                                 and named {out_of_range_ids} id(s) no region has; those replies \
                                 were dropped"
                            );
                        }
                    }
                    _ => {}
                })),
                ..Request::default()
            },
            &mut committer,
        )
        .await?;
    eprintln!("pipeline finished in {:.2}s", report.elapsed.as_secs_f64());

    let compositor = Compositor::new();
    let scene_renderer = SceneRenderer::new();
    let rasterizer = Rasterizer::new()?;
    let mut request = RenderRequest::new(page);
    request.theme.font_families = arguments.font_families;
    let render_started = Instant::now();
    let snapshot = session.snapshot();
    let composition = compositor.compile(&snapshot, &request)?;
    let frame = scene_renderer.render(&snapshot, &composition)?;
    let raster = rasterizer.rasterize(&frame, RasterOptions::default())?;
    let render_elapsed = render_started.elapsed();
    raster
        .image
        .save(&arguments.output)
        .with_context(|| format!("failed to write {}", arguments.output.display()))?;
    eprintln!(
        "rendered {} in {:.2}s",
        arguments.output.display(),
        render_elapsed.as_secs_f64()
    );
    Ok(())
}

async fn initialize_with_retry() {
    let mut delay = Duration::from_secs(1);
    let mut attempt = 0_u64;
    loop {
        attempt += 1;
        match koharu_ml::init().await {
            Ok(()) => return,
            Err(error) => {
                let jitter = Duration::from_millis((attempt.wrapping_mul(137)) % 251);
                let wait = delay + jitter;
                eprintln!(
                    "runtime initialization attempt {attempt} failed: {error}; retrying in {:.1}s",
                    wait.as_secs_f64()
                );
                tokio::time::sleep(wait).await;
                delay = delay.saturating_mul(2).min(Duration::from_secs(30));
            }
        }
    }
}

fn image_media_type(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_flags_select_models() {
        let arguments = Arguments::try_parse_from([
            "run",
            "--input",
            "input.png",
            "--output",
            "output.png",
            "--ocr",
            "manga-ocr",
            "--inpainting",
            "flux2-klein",
        ])
        .unwrap();
        assert!(matches!(
            arguments.pipeline_config().unwrap().ocr,
            OcrModel::MangaOcr
        ));
        assert!(matches!(
            arguments.pipeline_config().unwrap().inpainting,
            InpaintingModel::Flux2Klein(_)
        ));
    }

    fn args_from(extra: &[&str]) -> Arguments {
        let mut argv = vec!["run", "--input", "in.png", "--output", "out.png"];
        argv.extend_from_slice(extra);
        Arguments::try_parse_from(argv).unwrap()
    }

    #[test]
    fn local_provider_defaults_to_koharu_registry_model() {
        let arguments = args_from(&[]);
        assert_eq!(arguments.model().unwrap(), DEFAULT_LOCAL_MODEL);
        let translation = arguments.pipeline_config().unwrap().translation;
        assert_eq!(translation.model.provider, Provider::Local);
    }

    #[test]
    fn ollama_provider_selects_openai_compatible() {
        let arguments = args_from(&["--provider", "ollama", "--llm", "qwen3:8b"]);
        let translation = arguments.pipeline_config().unwrap().translation;
        assert_eq!(translation.model.provider, Provider::OpenAiCompatible);
        assert_eq!(translation.model.model.as_deref(), Some("qwen3:8b"));
    }

    #[test]
    fn ollama_provider_requires_an_explicit_model() {
        // Ollama tags are user-specific, so defaulting would silently pick a
        // model the server does not have.
        assert!(args_from(&["--provider", "ollama"]).model().is_err());
    }

    #[test]
    fn base_url_overrides_the_ollama_default() {
        let arguments = args_from(&[
            "--provider",
            "ollama",
            "--llm",
            "qwen3:8b",
            "--base-url",
            "http://192.0.2.50:11434/v1",
        ]);
        let base_url = arguments.providers_config().openai_compatible.base_url;
        assert_eq!(base_url.unwrap().as_str(), "http://192.0.2.50:11434/v1");
    }

    #[test]
    fn default_providers_config_points_at_local_ollama() {
        let base_url = args_from(&[]).providers_config().openai_compatible.base_url;
        assert_eq!(
            base_url.unwrap().as_str(),
            "http://localhost:11434/v1",
            "koharu-translator's built-in default should already target Ollama"
        );
    }
}
