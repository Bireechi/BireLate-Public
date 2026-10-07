//! Replay frozen page segments through one translator, with no pipeline attached.
//!
//! WHY A RUST BIN AND NOT A PYTHON DRIVER. The incumbent is decoded under an
//! llguidance grammar built from a per-page JSON schema, and nothing outside
//! this workspace can reproduce that: llama-server and Ollama both constrain
//! with a different implementation over a different token set. Four smaller
//! gaps compound it -- the seed, `top_k`/`min_p`/`repeat_penalty` (absent from
//! any wire schema), the chat prompt (rendered in-process with minijinja and
//! tokenized `AddBos::Never`, where an HTTP server re-templates it), and
//! `n_ctx = prompt + max_tokens + 1`, which changes CUDA attention blocking and
//! therefore the logits. So an HTTP driver would measure models through two
//! different decoders and call the difference quality.
//!
//! This calls `Translator::translate` -- the same method
//! `stages/translation.rs` calls, with a request built the same way -- so the
//! prompt, the schema and the sampler are the shipped ones by construction
//! rather than by careful copying.
//!
//! WHAT IT DELIBERATELY DOES NOT DO is run detection, OCR or inpainting. Those
//! are deterministic and shared, so every candidate is fed byte-identical
//! segments out of a frozen corpus and the only thing that varies between
//! arms is the model. A 168-page corpus is then a few minutes per arm
//! instead of most of an hour, and no arm can win by getting easier text.
//!
//! ID SHUFFLE IS INVISIBLE IN THE RETURN VALUE, which is why this installs a
//! tracing layer. `prompt::translations` seeds its result from the *source*
//! segments and fills in by id, so by the time `Translated` exists the ids have
//! already been put back in order -- a page the model answered out of order is
//! indistinguishable from one it answered in order. The translator counts it
//! and logs it; the layer below is the only way to attach that count to the
//! page it belongs to. `gemma4-31b-it` shuffles on ~5.3% of pages, so a harness that
//! reproduces roughly that is working and one that reports zero is not.
//!
//!     cargo run --release -p birelate-server --bin replay -- \
//!         --freeze corpus\frozen --llm gemma4-31b-it --out corpus\replay
//!
//! Output holds page text, so keep the --out directory out of version control.
//!
//! It is also where a *prompt* arm is measurable at all, and for the same
//! reason: `--source-language-prompt off|named|medium` feeds the freeze
//! manifest's per-source language into the request, which the pipeline has no
//! honest way to supply -- no OCR backend reports a language, and the region
//! stamp in `stages/ocr.rs` merely echoes the request's own declaration back.
//! Replay bypasses the pipeline, so no plumbing is needed to try it.
//! `--style-clause` is the second such arm and varies a different sentence,
//! so the two compose: a run may name both, and
//! the output filename records whichever of them is not the default.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use koharu_translator::{
    GenerationConfig, Language, ModelSelection, Provider, ProvidersConfig, StyleClause,
    TranslationRequest, Translator,
};
use serde::{Deserialize, Serialize};
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context as LayerContext, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Set by the layer, read and reset around each `translate`. A single
/// `translate` is in flight at a time by construction -- the corpus is walked
/// sequentially because the story window makes page N+1 depend on page N -- so
/// one slot is enough and needs no keying.
static OUT_OF_ORDER: AtomicI64 = AtomicI64::new(-1);

struct OrderProbe;

struct OrderVisitor(Option<i64>);

impl Visit for OrderVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "out_of_order" {
            self.0 = Some(value as i64);
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if field.name() == "out_of_order" {
            self.0 = Some(value);
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

impl<S: tracing::Subscriber> Layer<S> for OrderProbe {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: LayerContext<'_, S>) {
        let mut visitor = OrderVisitor(None);
        event.record(&mut visitor);
        if let Some(count) = visitor.0 {
            OUT_OF_ORDER.store(count, Ordering::Relaxed);
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Replay frozen segments through one translation model")]
struct Args {
    // A frozen corpus directory: summary.json plus one subdirectory of page
    // JSON per source.
    /// Frozen corpus directory: summary.json plus one subdirectory of page JSON
    /// per source.
    #[arg(long)]
    freeze: PathBuf,

    /// Directory to write the replay results to.
    #[arg(long)]
    out: PathBuf,

    // Tracks the server's own default rather than naming a model here, so a
    // bare replay always measures whatever a reader's page would go through.
    /// Translation model id. Defaults to the server's own default model.
    #[arg(long, default_value = birelate_server::models::DEFAULT_LOCAL_MODEL)]
    llm: String,

    #[arg(long, default_value = "local")]
    provider: String,

    /// Model quantization to load, where the model offers more than one.
    #[arg(long)]
    quantization: Option<String>,

    // PINNED, and not left to the descriptor. The catalog gives
    // `gemma4-31b-it` 1.0 and `ministral-3-14b-instruct` 0.05, so an arm run on
    // descriptor defaults differs from the incumbent in two variables at once
    // and neither can be attributed. 0 also makes the determinism check mean
    // something: two identical runs should agree byte for byte.
    /// Sampling temperature, pinned rather than taken from the model descriptor. 0
    /// makes repeat runs comparable byte for byte.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    // Left unset by default so the per-page token budget still scales it.
    /// Maximum tokens per reply. Unset lets the per-page budget scale it.
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Language to translate into, as a BCP 47 tag such as en-US.
    #[arg(long, default_value = "en-US")]
    target_language: String,

    // Story pairs carried into the next page's prompt. 0 turns the window off,
    // which is the no-story arm.
    /// Story pairs carried into the next page's prompt. 0 turns the story window off.
    #[arg(long, default_value_t = birelate_server::story::DEFAULT_PAIRS)]
    story_pairs: usize,

    // Whole-corpus repeats. Two is the determinism check.
    /// Number of whole-corpus repeats. 2 checks determinism.
    #[arg(long, default_value_t = 1)]
    repeat: usize,

    // Base sampler seed. ABSENT keeps upstream's fixed constant, under which
    // every repeat is byte-identical at any temperature -- that is the
    // determinism check, and it is also why a bare `--repeat N` can never
    // produce independent draws (measured, not assumed).
    // With it, run N samples with `seed + N`, so `--seed 1 --repeat 8` is
    // eight independent draws on one model load.
    /// Base sampler seed; run N uses seed + N. Unset keeps the fixed upstream seed, so
    /// every repeat is identical.
    #[arg(long)]
    seed: Option<u32>,

    // Only these source keys, comma separated. Default is every source.
    /// Only these source keys, comma separated. Default: every source.
    #[arg(long)]
    sources: Option<String>,

    // `off` (default), `named`, or `medium` -- how much the system prompt is
    // told about the language the page is written in.
    //
    // `off` is the shipped prompt: `stages/translation.rs` never calls
    // `with_source_language`, so every page ever translated here has been told
    // to translate "from the detected source language". `named` names it;
    // `medium` names it and calls the page manga / Korean manhwa / Chinese
    // manhua as well.
    //
    // **The language comes from the freeze manifest, which is the honest
    // producer here.** No OCR backend reports a language: the region stamp in
    // `stages/ocr.rs` echoes the request's declared language -- so reading the
    // region field back out yields a declaration, never a fact about the page.
    // The freeze manifest records what each source actually is, per source,
    // beside the pages. Shipping a winning arm to the live pipeline stays a
    // deliberate change: the live server already takes a per-request
    // `source_language`, but the translation stage never feeds it to the
    // prompt, because naming the language scored worse than leaving it out
    // over 511 segments, and adding the medium term scored worse still.
    //
    // A String parsed into a typed value, like `--hyphenation`, because that
    // flag exists for the same reason -- three arms that had to be compared.
    /// How much the system prompt is told about the page's language: off (default),
    /// named, or medium.
    #[arg(long, value_name = "ARM", default_value = "off")]
    source_language_prompt: String,

    // `shipped` (default), `no-sfx`, `none`, or `sized` -- which style sentence
    // the system prompt carries between the task sentence and the JSON
    // contract.
    //
    // The shipped one is "Preserve character voice, emotional tone,
    // relationship nuance, emphasis, and sound effects while keeping wording
    // concise enough for speech bubbles." -- the densest instruction in the
    // prompt, three separable demands in one sentence, and the only one that has
    // never been varied. The arms take it apart:
    //
    // - `no-sfx` strikes sound effects off the preserve list. In this field
    //   "preserve the sound effects" is idiom for leaving them in the source
    //   script, which is the opposite of what this server does: it translates
    //   sound effects.
    // - `none` deletes the sentence outright, and is the control: without it a
    //   win for either other arm cannot be told from the prompt merely being
    //   shorter.
    // - `sized` keeps the preserve list and replaces only the concision tail,
    //   with what a balloon does to a long line rather than with a character
    //   count -- a number invites trading meaning for the number, and would bind
    //   unevenly (dialogue p90 is 64 characters on a Japanese manga test
    //   corpus, 96 on a Chinese webtoon one).
    //
    // **Orthogonal to `--source-language-prompt`, and that is asserted rather
    // than assumed**: the two vary different sentences, all twelve combinations
    // render, and `the_style_clause_is_identical_across_the_three_language_arms`
    // in `prompt.rs` pins that the style sentence does not move when the
    // language arm does.
    //
    // A String parsed into `koharu_translator::StyleClause`, exactly like the
    // flag above: the four *values* are named here, the four *sentences* are
    // not. Every word the model reads stays in `prompt.rs`, so this bin selects
    // an arm and never composes one.
    /// Which style sentence the system prompt carries: shipped (default), no-sfx, none,
    /// or sized.
    #[arg(long, value_name = "ARM", default_value = "shipped")]
    style_clause: String,

    /// Run on the CPU instead of the GPU.
    #[arg(long)]
    cpu: bool,
}

/// How much the system prompt says about the source language.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceLanguagePrompt {
    /// What ships: "from the detected source language".
    Off,
    /// The language, named.
    Named,
    /// The language, named, plus the medium noun beside it.
    Medium,
}

impl SourceLanguagePrompt {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "named" => Ok(Self::Named),
            "medium" => Ok(Self::Medium),
            other => anyhow::bail!(
                "bad --source-language-prompt {other:?}; use off, named, or medium"
            ),
        }
    }

    fn needs_language(self) -> bool {
        self != Self::Off
    }
}

/// The `--style-clause` value, mapped onto the translator crate's own arm type.
///
/// Not a `FromStr` on `StyleClause` itself: the four spellings are this bin's CLI
/// vocabulary, while the enum is a library type that a server request or a GUI
/// would reach by name. Keeping the mapping here means the library never grows a
/// parser for strings only a harness types.
///
/// Refuses before `koharu_ml::init()`, like every other arm parse in this file,
/// so a typo costs nothing -- the alternative is discovering it after the GPU
/// has been claimed and ~20 GB of weights are resident.
fn parse_style_clause(value: &str) -> Result<StyleClause> {
    match value {
        "shipped" => Ok(StyleClause::Shipped),
        "no-sfx" => Ok(StyleClause::NoSfx),
        "none" => Ok(StyleClause::None),
        "sized" => Ok(StyleClause::Sized),
        other => anyhow::bail!(
            "bad --style-clause {other:?}; use shipped, no-sfx, none, or sized"
        ),
    }
}

/// Where one run's JSON lands, with each **non-default** arm named in the
/// filename and each default one silent.
///
/// The rule extends the single-flag naming to two flags without moving any
/// existing file: with both arms at their defaults the name is the plain
/// `<llm>-run<N>.json`, and `--source-language-prompt named` alone still writes
/// `<llm>-named-run<N>.json`.
///
/// The suffixes cannot be confused for each other even though both are bare
/// words: the two vocabularies are disjoint (`named`/`medium` against
/// `no-sfx`/`none`/`sized`) and the order is fixed, language then style. What
/// matters more is that no two *arms* share a name -- four style arms times three
/// language arms into one `--out` would otherwise be twelve overwrites of at most
/// three files.
fn output_name(llm: &str, language: Option<&str>, style: Option<&str>, run_index: usize) -> String {
    let mut name = llm.replace([':', '/'], "_");
    for arm in [language, style].into_iter().flatten() {
        name.push('-');
        name.push_str(arm);
    }
    format!("{name}-run{run_index}.json")
}

/// The slice of the frozen corpus's summary.json this needs.
#[derive(Deserialize)]
struct FreezeSummary {
    sources: Vec<FreezeSource>,
}

#[derive(Deserialize)]
struct FreezeSource {
    key: String,
    language: String,
    engine: String,
    pages: Vec<FreezePage>,
}

#[derive(Deserialize)]
struct FreezePage {
    file: String,
    #[serde(default)]
    error: Option<String>,
}

/// One frozen page, as much of it as replay reads.
#[derive(Deserialize)]
struct FrozenPage {
    #[serde(default)]
    regions: Vec<FrozenRegion>,
}

#[derive(Deserialize)]
struct FrozenRegion {
    #[serde(default)]
    source: String,
    /// The corpus's per-segment label. Carried through untouched so the report
    /// can score dialogue separately from watermarks without re-deciding, and
    /// NOT used to filter what is sent: the pipeline sends every region, so a
    /// faithful replay must too or the prompt is a different length.
    #[serde(default)]
    label: Option<String>,
}

#[derive(Serialize)]
struct PageResult {
    file: String,
    segments: usize,
    ms: u128,
    untranslated: Vec<usize>,
    truncated: bool,
    /// How many segments arrived out of input order, or null when the provider
    /// reported nothing -- which is every provider but the local one.
    out_of_order: Option<i64>,
    /// Reply entries addressed to an id already answered, and to an id no
    /// segment has. Both were dropped.
    ///
    /// Read straight off the return value, unlike `out_of_order` above, which
    /// needs the tracing layer because the ids have already been put back in
    /// order by the time `Translated` exists. These two survive the trip, which
    /// is the point of carrying them there: `untranslated` reports a duplicate
    /// only as its victim, so an arm that loses ids and an arm that runs out of
    /// tokens score identically without this column.
    duplicate_ids: usize,
    out_of_range_ids: usize,
    story_context: usize,
    labels: Vec<Option<String>>,
    outputs: Vec<String>,
}

#[derive(Serialize)]
struct SourceResult {
    key: String,
    language: String,
    engine: String,
    story: String,
    pages: Vec<PageResult>,
}

#[derive(Serialize)]
struct RunResult {
    llm: String,
    provider: String,
    quantization: Option<String>,
    temperature: f32,
    max_tokens: Option<u32>,
    target_language: String,
    story_pairs: usize,
    /// Which prompt arm produced this file. Recorded because the arms are
    /// invisible in the output otherwise -- the segments are the same, the model
    /// is the same, and only the system prompt differed.
    source_language_prompt: String,
    /// Which style-sentence arm produced this file, recorded for the same reason:
    /// the arms are invisible in the output otherwise.
    style_clause: String,
    run_index: usize,
    freeze: String,
    sources: Vec<SourceResult>,
    total_ms: u128,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // The probe carries no filter of its own, so it sees the translator's
    // out-of-order debug event whatever BIRELATE_LOG says. The printing layer keeps its own filter
    // so a replay is not drowned in llama.cpp's load chatter.
    let filter = EnvFilter::try_from_env("BIRELATE_LOG")
        .unwrap_or_else(|_| EnvFilter::new("warn,birelate_server=info"));
    tracing_subscriber::registry()
        .with(OrderProbe)
        .with(tracing_subscriber::fmt::layer().with_filter(filter))
        .init();

    let provider = match args.provider.as_str() {
        "local" => Provider::Local,
        "ollama" | "openai-compatible" => Provider::OpenAiCompatible,
        other => anyhow::bail!("unknown provider {other:?}; use local or ollama"),
    };
    let selection = ModelSelection {
        provider,
        model: Some(args.llm.clone()),
        quantization: args.quantization.clone(),
    };
    let target: Language = args
        .target_language
        .parse()
        .map_err(|_| anyhow::anyhow!("bad target language {:?}", args.target_language))?;
    let arm = SourceLanguagePrompt::parse(&args.source_language_prompt)?;
    let style = parse_style_clause(&args.style_clause)?;

    let mut generation = GenerationConfig::default();
    generation.temperature = Some(args.temperature);
    if let Some(cap) = args.max_tokens {
        generation.max_tokens = Some(cap);
    }

    // Without this the first translate fails with "llama.cpp backend is not
    // initialized" -- the server does it at startup (`lib.rs`) and a bin
    // that builds a Translator directly inherits none of that. Retried for the
    // same reason the server retries: a cold start can lose a race with a
    // runtime file still in use and legitimately fail once or twice.
    let mut delay = std::time::Duration::from_secs(1);
    for attempt in 1..=6u32 {
        match koharu_ml::init().await {
            Ok(()) => break,
            Err(error) if attempt == 6 => return Err(error).context("runtime never initialized"),
            Err(error) => {
                eprintln!("runtime init failed ({error}); retrying in {:.0}s", delay.as_secs_f64());
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(std::time::Duration::from_secs(30));
            }
        }
    }

    let device = koharu_ml::device(args.cpu);
    let translator = Translator::from_config(
        device,
        koharu_config::Config::memory(ProvidersConfig::default()),
    )?;

    let summary: FreezeSummary = read_json(&args.freeze.join("summary.json"))?;
    let wanted: Option<Vec<String>> = args
        .sources
        .as_deref()
        .map(|list| list.split(',').map(|s| s.trim().to_owned()).collect());

    fs::create_dir_all(&args.out)?;

    for run_index in 0..args.repeat.max(1) {
        // Per-run, not per-process: the whole point of the knob is that the
        // repeats differ. Absent stays absent, preserving the determinism arm.
        generation.seed = args.seed.map(|base| base + run_index as u32);
        let started = Instant::now();
        let mut sources = Vec::new();

        for source in &summary.sources {
            if let Some(keys) = wanted.as_ref() {
                if !keys.contains(&source.key) {
                    continue;
                }
            }
            // Parsed once per source, and only when an arm asks for it: the
            // `off` arm never reads the field, so an unreadable one must not
            // fail it. When an arm *does* need it, failing loudly is the point
            // -- quietly falling back to "the detected source language" would
            // make the run a relabelled control and nothing in the output would
            // say so.
            let source_language = if arm.needs_language() {
                Some(source.language.parse::<Language>().map_err(|_| {
                    anyhow::anyhow!(
                        "source {:?} records language {:?}, which koharu cannot parse; \
                         --source-language-prompt {} needs one it can",
                        source.key,
                        source.language,
                        args.source_language_prompt,
                    )
                })?)
            } else {
                None
            };

            // A fresh window per source AND per run: a story carried across
            // runs would make run two a different experiment from run one, and
            // the determinism check would fail for a reason that is not the
            // model's.
            let stories = birelate_server::story::Stories::new(args.story_pairs);
            let story_id = format!("replay-{}-{}", source.key, run_index);
            let mut pages = Vec::new();

            for page in &source.pages {
                if page.error.is_some() {
                    continue;
                }
                let path = args.freeze.join(&source.key).join(format!("{}.json", page.file));
                let frozen: FrozenPage = read_json(&path)?;
                let segments: Vec<String> =
                    frozen.regions.iter().map(|r| r.source.clone()).collect();
                let labels: Vec<Option<String>> =
                    frozen.regions.iter().map(|r| r.label.clone()).collect();
                if segments.is_empty() {
                    continue;
                }

                let context = stories.context(&story_id);
                // Unconditional, unlike the language arm: the style sentence
                // needs nothing out of the manifest, so every arm of it is
                // meaningful on a corpus whose language koharu cannot parse.
                let mut request =
                    TranslationRequest::new(segments.clone(), target).with_style_clause(style);
                if let Some(language) = source_language {
                    request = request
                        .with_source_language(language)
                        .with_medium_term(arm == SourceLanguagePrompt::Medium);
                }
                if !context.is_empty() {
                    request = request.with_context(context.iter().cloned());
                }

                OUT_OF_ORDER.store(-1, Ordering::Relaxed);
                let at = Instant::now();
                let (_provider_id, translated) = translator
                    .translate(&selection, generation, request)
                    .await
                    .with_context(|| format!("translating {}/{}", source.key, page.file))?;
                let ms = at.elapsed().as_millis();
                let seen = OUT_OF_ORDER.swap(-1, Ordering::Relaxed);

                // Recorded exactly as the server does it, so the next page's
                // prompt is the one the pipeline would have built.
                stories.record(
                    &story_id,
                    segments
                        .iter()
                        .cloned()
                        .zip(translated.segments.iter().cloned()),
                );

                let dropped = translated.duplicate_ids + translated.out_of_range_ids;
                println!(
                    "  {:<10} {:>3} seg {:>6} ms  ctx {:>3}{}{}{}",
                    page.file,
                    segments.len(),
                    ms,
                    context.len(),
                    if translated.truncated { "  TRUNCATED" } else { "" },
                    if seen > 0 { format!("  out-of-order {seen}") } else { String::new() },
                    // Printed as the pair, never as the sum: a duplicate is a
                    // copying slip inside the id set the model was given, an
                    // out-of-range id is an invented one, and under the local
                    // grammar the second should be impossible.
                    if dropped > 0 {
                        format!(
                            "  dropped-ids dup {} oor {}",
                            translated.duplicate_ids, translated.out_of_range_ids
                        )
                    } else {
                        String::new()
                    },
                );

                pages.push(PageResult {
                    file: page.file.clone(),
                    segments: segments.len(),
                    ms,
                    untranslated: translated.untranslated.clone(),
                    truncated: translated.truncated,
                    out_of_order: (seen >= 0).then_some(seen),
                    duplicate_ids: translated.duplicate_ids,
                    out_of_range_ids: translated.out_of_range_ids,
                    story_context: context.len(),
                    labels,
                    outputs: translated.segments.clone(),
                });
            }

            println!(
                "{} ({}, {}): {} pages",
                source.key,
                source.language,
                source.engine,
                pages.len()
            );
            sources.push(SourceResult {
                key: source.key.clone(),
                language: source.language.clone(),
                engine: source.engine.clone(),
                story: story_id,
                pages,
            });
        }

        let result = RunResult {
            llm: args.llm.clone(),
            provider: args.provider.clone(),
            quantization: args.quantization.clone(),
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            target_language: args.target_language.clone(),
            story_pairs: args.story_pairs,
            source_language_prompt: args.source_language_prompt.clone(),
            style_clause: args.style_clause.clone(),
            run_index,
            freeze: args.freeze.display().to_string(),
            sources,
            total_ms: started.elapsed().as_millis(),
        };
        // Every default arm stays silent in the filename and every non-default
        // one names itself -- so a run with both flags at their defaults still
        // writes the plain name. See `output_name`.
        let path = args.out.join(output_name(
            &args.llm,
            (arm != SourceLanguagePrompt::Off).then_some(args.source_language_prompt.as_str()),
            (style != StyleClause::Shipped).then_some(args.style_clause.as_str()),
            run_index,
        ));
        fs::write(&path, serde_json::to_string_pretty(&result)?)?;

        let shuffled: usize = result
            .sources
            .iter()
            .flat_map(|s| s.pages.iter())
            .filter(|p| p.out_of_order.unwrap_or(0) > 0)
            .count();
        let answered: usize = result
            .sources
            .iter()
            .flat_map(|s| s.pages.iter())
            .count();
        let truncated: usize = result
            .sources
            .iter()
            .flat_map(|s| s.pages.iter())
            .filter(|p| p.truncated)
            .count();
        let mut per_language: BTreeMap<&str, (usize, u128)> = BTreeMap::new();
        for source in &result.sources {
            let entry = per_language.entry(source.language.as_str()).or_default();
            for page in &source.pages {
                entry.0 += 1;
                entry.1 += page.ms;
            }
        }

        println!(
            "\n--- run {run_index}: {} (source-language-prompt {}, style-clause {}) ---",
            args.llm, args.source_language_prompt, args.style_clause
        );
        println!(
            "  {answered} pages, {:.1}s total, {shuffled} shuffled ({:.1}%), {truncated} truncated",
            result.total_ms as f64 / 1000.0,
            if answered > 0 {
                100.0 * shuffled as f64 / answered as f64
            } else {
                0.0
            }
        );
        for (language, (pages, ms)) in &per_language {
            println!(
                "  {language}: {pages} pages, {:.0} ms mean",
                *ms as f64 / *pages.max(&1) as f64
            );
        }
        println!("  wrote {}", path.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default is the shipped prompt, so a bare replay is still the control
    /// and every recorded run predating this flag is still reproducible by
    /// leaving it off.
    #[test]
    fn the_default_arm_is_the_shipped_prompt() {
        let args = Args::parse_from(["replay", "--freeze", "f", "--out", "o"]);
        assert_eq!(args.source_language_prompt, "off");
        assert_eq!(
            SourceLanguagePrompt::parse(&args.source_language_prompt).unwrap(),
            SourceLanguagePrompt::Off
        );
        // The same guarantee for the second arm, and both in one test because
        // what has to hold is that a *bare* replay is the control on every lever
        // at once, not that each lever has a sensible default in isolation.
        assert_eq!(args.style_clause, "shipped");
        assert_eq!(
            parse_style_clause(&args.style_clause).unwrap(),
            StyleClause::Shipped
        );
    }

    #[test]
    fn exactly_four_style_arms_are_accepted_and_the_refusal_names_them() {
        for (value, expected) in [
            ("shipped", StyleClause::Shipped),
            ("no-sfx", StyleClause::NoSfx),
            ("none", StyleClause::None),
            ("sized", StyleClause::Sized),
        ] {
            assert_eq!(parse_style_clause(value).unwrap(), expected, "{value}");
        }
        // The near-misses a hand would actually type: the underscore spelling,
        // the plural, and the capitalised form clap does not fold.
        for bad in ["no_sfx", "nosfx", "off", "Shipped", ""] {
            let error = parse_style_clause(bad).unwrap_err().to_string();
            assert!(
                error.contains("shipped, no-sfx, none, or sized"),
                "{bad}: {error}"
            );
        }
    }

    /// The filename rule, over the four corners of the two-flag grid.
    ///
    /// Default-on-both must keep the plain single-flag name, or an existing
    /// command silently writes somewhere new. The other three must differ from it
    /// and from each other, or a sweep silently reports one arm four times.
    #[test]
    fn each_pair_of_arms_writes_a_distinct_file_and_the_default_pair_keeps_its_name() {
        let plain = output_name("gemma4-31b-it", None, None, 0);
        assert_eq!(plain, "gemma4-31b-it-run0.json");

        let language_only = output_name("gemma4-31b-it", Some("named"), None, 0);
        assert_eq!(language_only, "gemma4-31b-it-named-run0.json");

        let style_only = output_name("gemma4-31b-it", None, Some("no-sfx"), 0);
        assert_eq!(style_only, "gemma4-31b-it-no-sfx-run0.json");

        let both = output_name("gemma4-31b-it", Some("medium"), Some("sized"), 1);
        assert_eq!(both, "gemma4-31b-it-medium-sized-run1.json");

        let names = [&plain, &language_only, &style_only, &both];
        for (i, a) in names.iter().enumerate() {
            for b in &names[i + 1..] {
                assert_ne!(a, b);
            }
        }

        // A provider tag still loses its colon and slash, which is the whole
        // reason that replacement exists: `qwen3:8b` is not a filename.
        assert_eq!(
            output_name("qwen3:8b", None, Some("none"), 0),
            "qwen3_8b-none-run0.json"
        );
    }

    /// Every one of the twelve arm pairs names a different file. Enumerated
    /// rather than argued from the disjoint vocabularies, because that argument
    /// is exactly what a fifth arm value would quietly break.
    #[test]
    fn all_twelve_arm_pairs_are_distinct_filenames() {
        let mut names = std::collections::BTreeSet::new();
        for language in [None, Some("named"), Some("medium")] {
            for style in [None, Some("no-sfx"), Some("none"), Some("sized")] {
                assert!(
                    names.insert(output_name("m", language, style, 0)),
                    "{language:?} + {style:?} collided"
                );
            }
        }
        assert_eq!(names.len(), 12);
    }

    /// The two flags are independent at the CLI as well as in the prompt: naming
    /// one must not disturb the other's default.
    #[test]
    fn naming_one_arm_leaves_the_other_at_its_default() {
        let styled = Args::parse_from([
            "replay", "--freeze", "f", "--out", "o", "--style-clause", "sized",
        ]);
        assert_eq!(styled.style_clause, "sized");
        assert_eq!(styled.source_language_prompt, "off");

        let named = Args::parse_from([
            "replay",
            "--freeze",
            "f",
            "--out",
            "o",
            "--source-language-prompt",
            "medium",
        ]);
        assert_eq!(named.source_language_prompt, "medium");
        assert_eq!(named.style_clause, "shipped");
    }

    #[test]
    fn exactly_three_arms_are_accepted_and_the_refusal_names_them() {
        assert_eq!(
            SourceLanguagePrompt::parse("named").unwrap(),
            SourceLanguagePrompt::Named
        );
        assert_eq!(
            SourceLanguagePrompt::parse("medium").unwrap(),
            SourceLanguagePrompt::Medium
        );
        let error = SourceLanguagePrompt::parse("Named").unwrap_err().to_string();
        assert!(error.contains("off, named, or medium"), "{error}");
    }

    /// Only `off` may run against a manifest whose language it cannot read. The
    /// other two would silently become `off` if this were ever relaxed, and the
    /// output file would still be labelled with the arm that did not run.
    #[test]
    fn only_the_off_arm_can_ignore_the_manifests_language() {
        assert!(!SourceLanguagePrompt::Off.needs_language());
        assert!(SourceLanguagePrompt::Named.needs_language());
        assert!(SourceLanguagePrompt::Medium.needs_language());
    }

    /// The tags a freeze manifest actually records, in the shape it records them,
    /// so a corpus of `ja` / `zh` / `ko` cannot fail the arm at page one after the
    /// models have already loaded.
    #[test]
    fn every_language_the_freeze_manifest_records_parses() {
        for (tag, expected) in [
            ("ja", Language::Japanese),
            ("zh", Language::ChineseSimplified),
            ("ko", Language::Korean),
        ] {
            assert_eq!(tag.parse::<Language>().unwrap(), expected);
        }
    }
}
