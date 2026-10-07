use std::{sync::Arc, time::Instant};

use anyhow::Context;
use koharu_ml::llm::{ChatMessage, ChatTemplateOptions, FinishReason, Input, Llm, LoadOptions};

mod catalog;

pub use catalog::LocalConfig;
use catalog::LocalModelDescriptor;
pub(crate) use catalog::{DEFAULT_MODEL, DEFAULT_QUANTIZATION};

use crate::{
    Device, Error, GenerationConfig, Model, ModelSelection, Provider, Quantization, Result,
    Translated, TranslationRequest, prompt,
};

/// Whole-reply overhead: the `{"translations":[` head, the `]}` tail, the stop
/// token, and enough slack for three-digit ids and a pretty-printed envelope.
const TOKEN_ENVELOPE: u32 = 96;
/// Per segment. Measured against gemma4's own 262k vocabulary: the `{"id":N,
/// "text":"` / `"},` scaffolding costs 8-9 tokens and a typical bubble's English
/// about 12, so a compact reply runs ~19 tokens a segment and a pretty-printed
/// one ~34. Nothing in the JSON-schema grammar forbids the model indenting, so
/// this is sized against the pretty-printed rate with room over it.
const TOKENS_PER_SEGMENT: u32 = 48;
/// Where the budget stops growing, and the reason it is a *budget* rather than
/// simply a large number. See [`token_budget`].
const TOKEN_BUDGET_CAP: u32 = 2048;

/// How many tokens one page's reply is allowed, given how many segments it has.
///
/// Every descriptor in the catalog pins `max_tokens` at 1000 no matter how much
/// text the page holds, and overflowing it is silent: the reply is cut off
/// mid-array, `crate::json::from_str` repairs it into valid JSON, and
/// `prompt::translations` seeds its output from the source, so the bubbles past
/// the cut render as untranslated Japanese with nothing anywhere reporting it.
/// A page dense enough to overflow 1000 tokens is an ordinary page: the crossover
/// is somewhere between 30 and 51 segments.
///
/// The cap exists because this is a VRAM decision, not a latency one.
/// `context_values` in koharu-ml sizes `n_ctx` as `prompt + max_tokens + 1`
/// whenever no explicit context length is given, and none is, so the KV cache
/// grows one for one with this number -- ~880 KiB per token for
/// gemma4-31b-it-Q4_K_XL. Raising it never errors; it silently allocates. Spend
/// enough and the pipeline's own admission control drops below
/// `safety_reserve + reservation`, which is the branch on which it evicts every
/// other stage's model on *every* request. 2048 costs 0.88 GiB against 1000 and
/// carries a page of ~60 segments; the observed peak is remembered for the life
/// of the process, so a generous cap is charged to every later page too.
///
/// Never returns less than the descriptor asks for, and never clips a
/// descriptor that already wants more than the cap.
fn token_budget(expected: usize, descriptor: Option<u32>) -> u32 {
    // Mirrors `ModelGeneration::options`' own fallback, so a descriptor that
    // states no budget still cannot be lowered by this.
    let floor = descriptor.unwrap_or(1000);
    let expected = u32::try_from(expected).unwrap_or(u32::MAX);
    let want = TOKEN_ENVELOPE.saturating_add(TOKENS_PER_SEGMENT.saturating_mul(expected));
    want.clamp(floor, TOKEN_BUDGET_CAP.max(floor))
}

#[derive(Debug)]
pub struct LocalTranslator {
    descriptor: LocalModelDescriptor,
    llm: Arc<Llm>,
}

impl LocalTranslator {
    pub async fn load(device: Device, selection: &ModelSelection) -> Result<Self> {
        let model = selection
            .model
            .as_deref()
            .context("local translation requires a selected model")?;
        let descriptor = catalog::MODELS
            .iter()
            .copied()
            .find(|descriptor| descriptor.id == model)
            .with_context(|| format!("unknown local translator '{model}'"))?;
        let model_path = descriptor.resolve(selection).await?;
        let llm = Llm::load_with_options(device, model_path, LoadOptions::default())
            .await
            .context("failed to load local translation model")?;
        Ok(Self {
            descriptor,
            llm: Arc::new(llm),
        })
    }

    pub(crate) async fn translate(
        &self,
        request: TranslationRequest,
        mut generation: GenerationConfig,
    ) -> Result<Translated> {
        let expected = request.segments.len();
        if expected == 0 {
            return Ok(Translated::default());
        }
        if !self
            .descriptor
            .target_languages
            .contains(request.target_language)
        {
            return Err(Error::UnsupportedLanguage {
                provider: "local",
                language: request.target_language,
            });
        }

        let prompt = self.render_prompt(&request)?;
        // `local_output_schema`, not `output_schema`: the llguidance override that
        // stops `\rightarrow` decoding to U+000D rides on this schema, and it must
        // not reach a remote provider that validates its input.
        let schema = prompt::local_output_schema(expected);
        let llm = Arc::clone(&self.llm);
        // Only fills a gap. An explicit caller value still wins outright, which
        // is the contract `ModelGeneration::options` states by asking
        // `overrides.max_tokens.or(self.max_tokens)`.
        if generation.max_tokens.is_none() {
            generation.max_tokens = Some(token_budget(
                expected,
                self.descriptor.generation.max_tokens,
            ));
        }
        // Kept for the split-fragment retry below: the retry re-resolves the
        // same caller overrides with only the seed changed, budget included.
        let overrides = generation;
        let generation = self.descriptor.generation.options(generation);
        let budget = generation.max_tokens;
        // The `Instant` starts *inside* the closure, not around the `await`, so
        // `spawn_blocking`'s own queueing latency is excluded. A translation
        // runs behind the server's one-permit GPU gate, so that queue is
        // normally empty -- but "normally" is the sort of assumption that turns
        // a measurement into a story, and the cost of not making it is one line.
        let (output, inference) = tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let output = llm.inference_with_json_schema(&Input::new(&prompt), &generation, &schema);
            (output, started.elapsed())
        })
        .await
        .context("local translation task panicked")?;
        let output = output?;

        // Splits one page's translation into the three terms it actually has.
        //
        // `Generation` has carried `prompt_duration` and `generation_duration`
        // since upstream, and this call site dropped both on the floor -- so the
        // only per-page number anyone could quote was the whole stage, and every
        // attempt to say where it went had to be inferred by regressing stage
        // totals against region counts across whole runs. That works (it is how
        // the ~15 ms/token decode rate was established) but it cannot see a
        // fixed cost at all, because a fixed cost is exactly what a regression
        // intercept confounds with everything else fixed.
        //
        // `setup` is the term that has no field anywhere and is the reason this
        // logs a total rather than just forwarding the two: `prompt_duration`'s
        // clock starts *after* `new_context` and `build_sampler`
        // (`koharu-ml/src/llm/model.rs`), so the difference is precisely
        // context creation plus sampler construction. That is where the
        // llguidance token trie is rebuilt over the whole 262,144-entry
        // vocabulary on every single page, and it is the one candidate whose
        // whole case rests on a number nobody has ever printed. Saturating
        // subtraction because three `Instant`s on two clocks can disagree by a
        // tick, and a panic here would cost a page that had already been paid
        // for on the GPU.
        //
        // `info` rather than `debug`: the server's default filter is `info`, so
        // this rides the next ordinary run with no environment variable and no
        // second measurement pass. One line per page.
        let setup = inference
            .saturating_sub(output.prompt_duration)
            .saturating_sub(output.generation_duration);
        /* Rides the same event as the token counts on purpose. The open question
         * these four answer is *why* a 24-pair story window emits 79% more
         * tokens for the same characters than a 96-pair one (0.98 chars/token
         * against 1.75) while being 41% slower per page entirely in decode. The
         * hypothesis is constrained decoding: with a thinner prompt the model's
         * preferred token is more often disallowed by the JSON grammar, and the
         * legal substitute is a shorter piece. Testing that needs
         * `grammar_displaced` next to `generated_tokens` for the *same* page, in
         * one line, or the join is by timestamp across two events and every
         * concurrent page corrupts it.
         *
         * `unwrap_or_default` collapses "decoding was unconstrained" to zeros.
         * That is not a silent lie: `grammar_steps` is the denominator of every
         * other field here and is itself zero in that case, so a reader sees
         * "no grammar ran" rather than "the grammar allowed nothing". This
         * backend always passes a schema, so a zero here is a bug signal.
         *
         * See `koharu-llama`'s `GrammarStats` for why the mean rank of the
         * chosen token is NOT among these: it is settled downstream by `dist`
         * after temperature, so it would mix grammar pressure with ordinary
         * stochastic sampling, and the sampler cannot see it anyway. */
        let grammar = output.grammar.unwrap_or_default();
        tracing::info!(
            model = self.descriptor.id,
            segments = expected,
            prompt_tokens = output.prompt_tokens,
            generated_tokens = output.generated_tokens,
            setup_ms = setup.as_millis(),
            prompt_ms = output.prompt_duration.as_millis(),
            generation_ms = output.generation_duration.as_millis(),
            inference_ms = inference.as_millis(),
            prefill_tps = output.prompt_tokens_per_second(),
            decode_tps = output.generated_tokens_per_second(),
            grammar_steps = grammar.steps,
            grammar_displaced = grammar.displaced,
            grammar_allowed_mean = grammar.allowed_per_step(),
            grammar_allowed_pct = grammar.allowed_percent(),
            grammar_gap_mean = grammar.mean_displaced_gap(),
            "local translation timing"
        );

        // `prompt::translations` reports *that* segments were missed; the finish
        // reason is the only thing that says why, and it exists on this path
        // alone -- every remote backend discards its provider's own before the
        // reply is parsed. Worth a line even when the repair happened to recover
        // every id, because it is the one warning that names an actionable cap.
        let truncated = output.finish_reason == FinishReason::Length;
        if truncated {
            tracing::warn!(
                model = self.descriptor.id,
                segments = expected,
                generated_tokens = output.generated_tokens,
                max_tokens = budget,
                "local translation stopped on the token cap; the reply was cut off"
            );
        }
        let mut translated = prompt::translations("local", &output.text, &request.segments)?;
        translated.truncated = truncated;

        /* The SEED-VARIED RETRY.
         *
         * A reply entry ending in trailing whitespace is the fingerprint of a
         * split fragment -- one source's translation across two reply slots,
         * every later slot displaced, every counter clean (the grammar pins
         * the slot count, so nothing else can see it). A static rejoin was
         * validated against every adjudicated test site and REFUTED: no
         * wire feature separates a displaced page from one the model
         * recovered on its own, and gluing a recovered page corrupts it.
         *
         * The retry needs no such judgement. The cut is a stochastic minority
         * outcome -- 1-3 of 8 independent draws at the worst measured sites --
         * so ONE re-ask with a different seed usually returns a clean page,
         * and a clean full-page draw repairs displaced and recovered pages
         * alike, lost tails included. With the sampler seed a fixed constant
         * this retry was measured a byte-identical no-op; the seed knob is
         * what makes it work.
         *
         * Bounded to ONE retry, fail-open on every error, and the first reply
         * is kept unless the retry is STRICTLY cleaner -- fewer fragments and
         * not truncated. A truncated first reply is the token cap's business,
         * not this pass's. Skipped entirely when
         * `KOHARU_RETRY_SPLIT_FRAGMENTS=0`, the A/B arm, matching its sibling
         * `KOHARU_REASK_UNTRANSLATED`. */
        if !truncated && retry_split_fragments_enabled() {
            let fragments = split_fragments(&translated);
            if !fragments.is_empty() {
                let first_seed = overrides
                    .seed
                    .unwrap_or_else(|| koharu_ml::llm::GenerationOptions::default().seed);
                let mut retry_overrides = overrides;
                retry_overrides.seed = Some(first_seed.wrapping_add(1));
                let retry_generation = self.descriptor.generation.options(retry_overrides);
                let retry_prompt = self.render_prompt(&request)?;
                let retry_schema = prompt::local_output_schema(expected);
                let retry_llm = Arc::clone(&self.llm);
                let retry = tokio::task::spawn_blocking(move || {
                    let started = Instant::now();
                    let output = retry_llm.inference_with_json_schema(
                        &Input::new(&retry_prompt),
                        &retry_generation,
                        &retry_schema,
                    );
                    (output, started.elapsed())
                })
                .await;
                match retry {
                    Ok((Ok(retry_output), retry_ms)) => {
                        let retry_truncated = retry_output.finish_reason == FinishReason::Length;
                        match prompt::translations("local", &retry_output.text, &request.segments)
                        {
                            Ok(mut retry_translated) => {
                                retry_translated.truncated = retry_truncated;
                                let retry_fragments = split_fragments(&retry_translated);
                                let keep_retry = retry_cleaner(
                                    fragments.len(),
                                    retry_fragments.len(),
                                    retry_truncated,
                                );
                                tracing::info!(
                                    model = self.descriptor.id,
                                    first_fragments = fragments.len(),
                                    retry_fragments = retry_fragments.len(),
                                    retry_seed = first_seed.wrapping_add(1),
                                    retry_ms = retry_ms.as_millis(),
                                    kept = if keep_retry { "retry" } else { "first" },
                                    "split-fragment retry"
                                );
                                if keep_retry {
                                    translated = retry_translated;
                                }
                            }
                            Err(error) => tracing::warn!(
                                %error,
                                "split-fragment retry reply unparsable; keeping the first"
                            ),
                        }
                    }
                    Ok((Err(error), _)) => tracing::warn!(
                        %error,
                        "split-fragment retry inference failed; keeping the first"
                    ),
                    Err(error) => tracing::warn!(
                        %error,
                        "split-fragment retry task panicked; keeping the first"
                    ),
                }
            }
        }
        Ok(translated)
    }

    fn render_prompt(&self, request: &TranslationRequest) -> Result<String> {
        let (system, payload) = prompt::prompts(request)?;
        Ok(self
            .llm
            .render_chat_prompt_with_options(
                &[ChatMessage::system(system), ChatMessage::user(payload)],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                },
            )
            .context("failed to render local translation prompt")?)
    }
}

/// Whether a reply carrying a split fragment buys one seed-varied retry. On
/// unless explicitly disabled -- `KOHARU_RETRY_SPLIT_FRAGMENTS=0` is the A/B
/// arm, the same contract as `KOHARU_REASK_UNTRANSLATED`.
fn retry_split_fragments_enabled() -> bool {
    !matches!(
        std::env::var("KOHARU_RETRY_SPLIT_FRAGMENTS").as_deref(),
        Ok("0") | Ok("false")
    )
}

/// The split-fragment fingerprint: an ANSWERED segment whose text
/// ends in trailing whitespace -- the stranded space before the quotation
/// mark (or clause) that terminated the string early. The core is
/// [`crate::is_cut`], shared with the server's `split_fragment_indices`
/// diagnostic; this layer's own arm stays here: an id in `untranslated` is
/// source-seeded, not answered, and never counts.
fn split_fragments(translated: &Translated) -> Vec<usize> {
    translated
        .segments
        .iter()
        .enumerate()
        .filter(|(index, text)| {
            !translated.untranslated.contains(index) && crate::is_cut(text)
        })
        .map(|(index, _)| index)
        .collect()
}

/// The keep rule, stated once so the test can break it: the retry replaces
/// the first reply only when it is STRICTLY cleaner -- fewer fragments and
/// not itself truncated. Equal is not cleaner; a truncated retry is never
/// kept, whatever its count, because truncation has its own machinery and a
/// short reply trivially carries fewer fragments.
fn retry_cleaner(first_fragments: usize, retry_fragments: usize, retry_truncated: bool) -> bool {
    !retry_truncated && retry_fragments < first_fragments
}

pub(crate) fn models() -> Vec<Model> {
    catalog::MODELS
        .iter()
        .map(|descriptor| Model {
            provider: Provider::Local,
            model: Some(descriptor.id.to_owned()),
            name: descriptor.name.to_owned(),
            quantizations: descriptor
                .quantizations
                .iter()
                .map(|quantization| Quantization {
                    id: quantization.id.to_owned(),
                    name: quantization.name.to_owned(),
                })
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelGeneration;

    /// What every descriptor in the catalog states today.
    const CATALOG: Option<u32> = Some(1000);

    /// The corpus string `"was a "` is the fixture the server's diagnostic
    /// uses too, so the two copies of the fingerprint stay provably aligned.
    #[test]
    fn the_fragment_fingerprint_fires_on_the_corpus_string_and_nothing_blank() {
        let fires = Translated {
            segments: vec!["was a ".into(), "clean.".into()],
            ..Translated::default()
        };
        assert_eq!(split_fragments(&fires), vec![0]);

        let trimmed = Translated {
            segments: vec!["was a".into(), "clean.".into()],
            ..Translated::default()
        };
        assert!(split_fragments(&trimmed).is_empty(), "no trailing whitespace, no fragment");

        let blank = Translated {
            segments: vec!["   ".into()],
            ..Translated::default()
        };
        assert!(split_fragments(&blank).is_empty(), "whitespace-only is dropped's business");

        let seeded = Translated {
            segments: vec!["\u{753b}\u{6570} ".into()],
            untranslated: vec![0],
            ..Translated::default()
        };
        assert!(
            split_fragments(&seeded).is_empty(),
            "an unanswered id carries seeded source, not a reply"
        );
    }

    /// Break `retry_cleaner`'s `<` to `<=` and the equal arm goes red; break
    /// the truncation guard and the truncated arm does. Each clause has a
    /// pole, so the composed predicate is what is asserted, not its halves.
    #[test]
    fn the_retry_keeps_the_first_reply_unless_strictly_cleaner() {
        assert!(retry_cleaner(2, 1, false), "strictly fewer fragments wins");
        assert!(retry_cleaner(1, 0, false), "a clean retry wins");
        assert!(!retry_cleaner(2, 2, false), "equal is not cleaner");
        assert!(!retry_cleaner(2, 3, false), "worse is certainly not");
        assert!(
            !retry_cleaner(1, 0, true),
            "a truncated retry is never kept: a short reply trivially has fewer fragments"
        );
    }

    #[test]
    fn a_small_page_keeps_the_descriptor_budget_byte_for_byte() {
        // 96 + 48n only passes 1000 at n = 19, so an ordinary page -- and the
        // test fixture's four bubbles -- generates exactly as it did before and
        // needs no re-profiling.
        for expected in [1, 4, 18] {
            assert_eq!(token_budget(expected, CATALOG), 1000, "{expected} segments");
        }
        assert_eq!(token_budget(19, CATALOG), 1008);
        assert_eq!(token_budget(20, CATALOG), 1056);
    }

    #[test]
    fn a_dense_page_is_capped_rather_than_left_to_grow() {
        // The cap first binds at 41 segments (96 + 48 * 41 = 2064). Beyond it
        // the KV cache, which tracks max_tokens one for one, would grow without
        // bound and take the pipeline's admission control down with it.
        assert_eq!(token_budget(40, CATALOG), 2016);
        assert_eq!(token_budget(41, CATALOG), TOKEN_BUDGET_CAP);
        assert_eq!(token_budget(usize::MAX, CATALOG), TOKEN_BUDGET_CAP);
    }

    #[test]
    fn the_budget_can_never_lower_what_the_descriptor_asked_for() {
        // A descriptor above the cap keeps its own number: this exists to raise
        // a too-small budget, never to impose one.
        assert_eq!(token_budget(1, Some(8192)), 8192);
        assert_eq!(token_budget(200, Some(8192)), 8192);
        // And a descriptor that states nothing falls back to the same 1000
        // `ModelGeneration::options` would have used.
        assert_eq!(token_budget(1, None), 1000);
        assert_eq!(token_budget(60, None), TOKEN_BUDGET_CAP);
    }

    #[test]
    fn an_explicit_caller_budget_still_wins() {
        // The budget only fills a `None`, so this is the value that reaches the
        // model however small or large it is.
        let descriptor = ModelGeneration {
            max_tokens: CATALOG,
            ..ModelGeneration::default()
        };
        for asked in [1, 250, 100_000] {
            let generation = GenerationConfig {
                max_tokens: Some(asked),
                ..GenerationConfig::default()
            };
            assert!(generation.max_tokens.is_some(), "the budget must not apply");
            assert_eq!(descriptor.options(generation).max_tokens, asked as usize);
        }
        // With no caller value the descriptor's own is what `options` falls back
        // to, which is precisely what `token_budget` is inserted ahead of.
        assert_eq!(
            descriptor.options(GenerationConfig::default()).max_tokens,
            1000
        );
    }
}
