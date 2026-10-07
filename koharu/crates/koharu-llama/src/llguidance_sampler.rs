//! Pure Rust llguidance sampler for constrained decoding.
//!
//! Implements a custom `llama_sampler` using the `llguidance` and `toktrie` Rust crates
//! to enforce grammar constraints (JSON schema, regex, Lark, etc.) during token sampling.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use llguidance::Matcher;
use toktrie::{ApproximateTokEnv, TokRxInfo, TokTrie};

use crate::GrammarError;
use crate::model::LlamaModel;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;

/// How hard the grammar actually pushed against the model, accumulated over one
/// inference.
///
/// **Why this exists.** A 24-pair story window is 41% slower per page than a
/// 96-pair one on the same corpus (1,470 vs 1,045 ms/page, reproducible to
/// within 2-4%), and it is entirely decode: prefill is 121 ms *cheaper* at 24
/// because the prompt is 1,425 tokens shorter. The model emits 79% more tokens
/// for the same amount of text -- 70,047 vs 39,082 tokens producing 68,368 vs
/// 68,309 characters, i.e. 0.98 chars/token against 1.75. The whole distribution
/// shifts (median 184 -> 341), so it is not a handful of runaway pages. The
/// hypothesis this counter exists to test is that with a thinner prompt the
/// model's preferred token is more often *disallowed* by the JSON grammar, and
/// the legal substitute is a shorter piece -- the same text spread over more
/// decode steps. Nobody had measured it.
///
/// # What is honestly countable here, and what is not
///
/// This is [`llg_apply`], which llama.cpp calls with the candidate array as it
/// reaches **chain position 0**: `koharu-ml`'s `build_sampler` pushes the
/// llguidance sampler before penalties, top-k, top-p, min-p, temperature and
/// `dist`, and `llama_sampler_sample` hands the first sampler the raw logits of
/// the entire vocabulary, unsorted, nothing filtered. So the argmax of what we
/// see really is *the model's own first choice*, before any sampler has had an
/// opinion. Two of the three quantities that would answer the question fall out
/// of the mask loop that already runs:
///
/// - **`displaced`** -- steps where that pre-mask argmax was masked out. This is
///   the direct form of "the grammar refused the model's first-choice token".
/// - **`allowed`** -- how many candidates the mask permits, summed over steps. A
///   thinner mask is less freedom, and the mean is comparable across models.
///
/// The third, *the mean rank of the token finally chosen*, is *not* countable
/// here and is deliberately not faked. Two independent reasons: the chosen token
/// is settled downstream by `dist` after temperature, so its rank would mix
/// grammar pressure with ordinary stochastic sampling; and [`llg_accept`]
/// receives the token id with no logits at all, so the rank cannot be
/// reconstructed after the fact. The nearest honest relative -- the rank of the
/// best *allowed* token -- is computable but needs a second full pass over
/// 262,144 candidates once the best allowed logit is known, roughly doubling the
/// per-token cost of this callback for a number that `displaced_gap_milli`
/// already answers in the model's own units. So it is not taken.
///
/// `displaced_gap_milli` is the sum, over displaced steps only, of
/// (best logit anywhere - best logit the grammar allows), in thousandths of a
/// logit. It says *how far down* the model's own scoring the grammar had to
/// reach, which is the part "was the top-1 refused?" cannot express: refusing a
/// near-tie costs nothing, refusing by four logits is a different token.
///
/// # Cost
///
/// Left on unconditionally, no flag. The per-candidate work is added to the
/// `is_allowed` loop that already existed: two float compares and one integer
/// add, no extra pass over the vocabulary and no allocation. The atomics are
/// touched **once per step**, not once per candidate -- the loop accumulates
/// into locals -- so a page pays five relaxed `fetch_add`s per generated token
/// against a decode step that is a forward pass through a 26B model.
///
/// Atomics rather than plain fields because the caller holds an `Arc` to the
/// same object in order to read the totals after generation, while the sampler
/// callbacks reach it through a raw pointer; `Relaxed` is sufficient because
/// nothing is published *through* these counters -- they are read once, after
/// the generation loop has finished, on the thread that ran it.
#[derive(Debug, Default)]
pub struct GrammarStats {
    steps: AtomicU64,
    candidates: AtomicU64,
    allowed: AtomicU64,
    displaced: AtomicU64,
    displaced_gap_milli: AtomicU64,
}

impl GrammarStats {
    fn record(&self, candidates: u64, allowed: u64, displaced: bool, gap_milli: u64) {
        self.steps.fetch_add(1, Ordering::Relaxed);
        self.candidates.fetch_add(candidates, Ordering::Relaxed);
        self.allowed.fetch_add(allowed, Ordering::Relaxed);
        if displaced {
            self.displaced.fetch_add(1, Ordering::Relaxed);
            self.displaced_gap_milli
                .fetch_add(gap_milli, Ordering::Relaxed);
        }
    }

    /// Reads the totals out. Call after the generation loop has finished.
    #[must_use]
    pub fn snapshot(&self) -> GrammarStatsSnapshot {
        GrammarStatsSnapshot {
            steps: self.steps.load(Ordering::Relaxed),
            candidates: self.candidates.load(Ordering::Relaxed),
            allowed: self.allowed.load(Ordering::Relaxed),
            displaced: self.displaced.load(Ordering::Relaxed),
            displaced_gap_milli: self.displaced_gap_milli.load(Ordering::Relaxed),
        }
    }
}

/// One inference's worth of [`GrammarStats`], read out flat so it can ride a
/// result struct.
///
/// `steps` counts sampling steps at which a mask was successfully computed, not
/// generated tokens. It is normally `generated_tokens` or one more than it (the
/// prefill samples once before the loop starts), and a step whose
/// `compute_mask_or_eos` errored is counted nowhere -- which is why the
/// denominator is published rather than assumed. `steps == 0` means no grammar
/// ran at all.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GrammarStatsSnapshot {
    /// Sampling steps at which the grammar computed a mask.
    pub steps: u64,
    /// Candidate entries seen, summed over steps. Effectively `steps * n_vocab`.
    pub candidates: u64,
    /// Candidates the grammar permitted, summed over steps.
    pub allowed: u64,
    /// Steps where the highest-logit candidate was **not** permitted.
    pub displaced: u64,
    /// Thousandths of a logit, summed over displaced steps: how far below the
    /// overall best logit the best permitted one sat.
    pub displaced_gap_milli: u64,
}

impl GrammarStatsSnapshot {
    /// Mean number of tokens the grammar left legal per step.
    #[must_use]
    pub fn allowed_per_step(&self) -> f64 {
        ratio(self.allowed, self.steps)
    }

    /// Percentage of the vocabulary the grammar left legal, averaged over steps.
    #[must_use]
    pub fn allowed_percent(&self) -> f64 {
        ratio(self.allowed, self.candidates) * 100.0
    }

    /// Fraction of steps where the model's first-choice token was refused.
    #[must_use]
    pub fn displaced_share(&self) -> f64 {
        ratio(self.displaced, self.steps)
    }

    /// Mean logit drop on the steps that were displaced. Zero when none were.
    #[must_use]
    pub fn mean_displaced_gap(&self) -> f64 {
        ratio(self.displaced_gap_milli, self.displaced) / 1000.0
    }
}

/// Zero rather than NaN on an empty denominator: these land in a `tracing` line
/// meant to be parsed by tools, and a NaN there is a parse failure rather than a
/// visible zero.
fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Internal state for the llguidance sampler.
struct LlgContext {
    matcher: Matcher,
    tok_env: Arc<ApproximateTokEnv>,
    grammar_kind: String,
    grammar_data: String,
    stats: Arc<GrammarStats>,
}

/// Build a [`toktrie::TokEnv`] from a [`LlamaModel`]'s vocabulary.
///
/// This mirrors the logic in upstream `llguidance.cpp` â€” for each token:
/// - Try normal detokenize (special=false)
/// - If empty, detokenize with special=true and prefix with 0xFF marker byte
fn build_tok_env(model: &LlamaModel) -> Arc<ApproximateTokEnv> {
    let n_vocab = model.n_vocab().cast_unsigned();
    let tok_eos = {
        let eot = unsafe { koharu_llama_sys::llama_vocab_eot(model.vocab_ptr()) };
        if eot == -1 {
            model.token_eos().0.cast_unsigned()
        } else {
            eot.cast_unsigned()
        }
    };
    let info = TokRxInfo::new(n_vocab, tok_eos);

    let mut words = Vec::with_capacity(n_vocab as usize);
    for i in 0..n_vocab.cast_signed() {
        let token = LlamaToken(i);
        let bytes = model
            .token_to_piece_bytes(token, 32, false, None)
            .unwrap_or_default();
        if bytes.is_empty() {
            let special_bytes = model
                .token_to_piece_bytes(token, 32, true, None)
                .unwrap_or_default();
            if special_bytes.is_empty() {
                words.push(vec![]);
            } else {
                let mut marked = Vec::with_capacity(special_bytes.len() + 1);
                marked.push(0xFF);
                marked.extend(special_bytes);
                words.push(marked);
            }
        } else {
            words.push(bytes);
        }
    }

    let trie = TokTrie::from(&info, &words);
    Arc::new(ApproximateTokEnv::new(trie))
}

// --- extern "C" vtable callbacks ---

unsafe extern "C" fn llg_name(
    _smpl: *const koharu_llama_sys::llama_sampler,
) -> *const std::os::raw::c_char {
    c"llguidance".as_ptr()
}

unsafe extern "C" fn llg_accept(
    smpl: *mut koharu_llama_sys::llama_sampler,
    token: koharu_llama_sys::llama_token,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let _ = ctx.matcher.consume_token(token.cast_unsigned());
}

unsafe extern "C" fn llg_apply(
    smpl: *mut koharu_llama_sys::llama_sampler,
    cur_p: *mut koharu_llama_sys::llama_token_data_array,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let cur_p = unsafe { &mut *cur_p };

    let Ok(mask) = ctx.matcher.compute_mask_or_eos() else {
        /* Counted nowhere on purpose. Nothing was masked on this step, so it is
         * neither a displaced step nor an un-displaced one, and folding it into
         * either would bias the very ratio this instrumentation exists to
         * measure. `steps` is published so the denominator is the observed one
         * rather than an assumed `generated_tokens`. */
        return;
    };

    let data = unsafe { std::slice::from_raw_parts_mut(cur_p.data, cur_p.size) };

    /* Both maxima come out of the loop that was already here. `best` is the
     * model's own first choice -- this sampler sits at chain position 0, so no
     * penalty, top-k or temperature has touched these logits yet -- and
     * `best_allowed` is the best the grammar will let through. The array is not
     * sorted (llama.cpp says outright not to assume it is), which costs nothing:
     * a maximum does not care about order. */
    let mut allowed: u64 = 0;
    let mut best = f32::NEG_INFINITY;
    let mut best_allowed = f32::NEG_INFINITY;
    for item in data.iter_mut() {
        let logit = item.logit;
        if logit > best {
            best = logit;
        }
        if mask.is_allowed(item.id.cast_unsigned()) {
            allowed += 1;
            if logit > best_allowed {
                best_allowed = logit;
            }
        } else {
            item.logit = f32::NEG_INFINITY;
        }
    }

    /* Equality, not identity: if some allowed token carries the *same* logit as
     * the argmax, the model was indifferent between them and the grammar cost it
     * nothing, so that is not a displacement. Comparing floats exactly is right
     * here because both sides are maxima over the same untouched values -- when
     * the argmax itself is allowed the two are literally the same float, not two
     * numbers that happen to be close.
     *
     * The finite guard covers the degenerate mask that allows nothing (which
     * `compute_mask_or_eos` should never produce, since EOS is in the candidate
     * array) and would otherwise turn the gap into an infinity. */
    let displaced = best_allowed < best;
    let gap_milli = if displaced && best.is_finite() && best_allowed.is_finite() {
        // `f64::from` first: the difference of two logits is small, but the cast
        // to integer is a truncation and doing it in f32 would round twice.
        ((f64::from(best) - f64::from(best_allowed)) * 1000.0).max(0.0) as u64
    } else {
        0
    };
    ctx.stats
        .record(data.len() as u64, allowed, displaced, gap_milli);
}

unsafe extern "C" fn llg_reset(smpl: *mut koharu_llama_sys::llama_sampler) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let _ = ctx.matcher.reset();
}

unsafe extern "C" fn llg_clone(
    smpl: *const koharu_llama_sys::llama_sampler,
) -> *mut koharu_llama_sys::llama_sampler {
    let ctx = unsafe { &*(*smpl).ctx.cast::<LlgContext>() };
    let new_ctx = Box::new(LlgContext {
        matcher: ctx.matcher.deep_clone(),
        tok_env: Arc::clone(&ctx.tok_env),
        grammar_kind: ctx.grammar_kind.clone(),
        grammar_data: ctx.grammar_data.clone(),
        /* Shared, not copied. A clone of this sampler is still sampling the same
         * inference, so its steps belong in the same totals; a fresh counter
         * would silently drop them. */
        stats: Arc::clone(&ctx.stats),
    });
    unsafe {
        koharu_llama_sys::llama_sampler_init(
            &raw mut LLG_SAMPLER_I,
            Box::into_raw(new_ctx).cast::<c_void>(),
        )
    }
}

unsafe extern "C" fn llg_free(smpl: *mut koharu_llama_sys::llama_sampler) {
    let ctx_ptr = unsafe { (*smpl).ctx.cast::<LlgContext>() };
    if !ctx_ptr.is_null() {
        drop(unsafe { Box::from_raw(ctx_ptr) });
    }
}

static mut LLG_SAMPLER_I: koharu_llama_sys::llama_sampler_i = koharu_llama_sys::llama_sampler_i {
    name: Some(llg_name),
    accept: Some(llg_accept),
    apply: Some(llg_apply),
    reset: Some(llg_reset),
    clone: Some(llg_clone),
    free: Some(llg_free),
    backend_init: None,
    backend_accept: None,
    backend_apply: None,
    backend_set_input: None,
};

/// Everything an llguidance sampler needs that depends on the **model** rather
/// than on the grammar, so it can be built once and reused for every page.
///
/// **This is a pure function of the vocabulary, and it was being rebuilt on
/// every inference.** `build_tok_env` walks all 262,144 tokens doing an FFI
/// `llama_token_to_piece` and a heap allocation each, then builds a `TokTrie`
/// over the result; `ParserFactory::new_simple` then runs `SlicedBiasComputer`,
/// which scans that whole vocabulary again per regex node. Neither depends on
/// the schema. Upstream says so outright -- `ParserFactory`'s own doc comment
/// reads that one is "typically created once per model/tokenizer and reused
/// across requests" -- and koharu was violating that contract once per page.
///
/// **Measured, not assumed: 510 ms median** of fixed per-inference cost on eight
/// real pages (495-721), read off the translator's `setup_ms` timing, which is
/// `inference - prompt_duration - generation_duration` and therefore covers
/// exactly context creation plus this. It is 8.4% of the inference and about 11%
/// of a warm page.
///
/// What is NOT cached is the parser: `output_schema` is parameterised by the
/// page's segment count (`minItems`/`maxItems`/`maximum` all carry it), so the
/// grammar genuinely differs page to page and `create_parser` has to run each
/// time. That is the cheap half.
///
/// Held by the caller for the life of the model. It borrows nothing from
/// `LlamaModel` -- the vocabulary is copied out during construction -- so it
/// cannot dangle if the model is later dropped, but a cache keyed on a model
/// pointer would, which is why this is a value the owner holds rather than a
/// global keyed on anything.
pub struct LlgEnv {
    tok_env: Arc<ApproximateTokEnv>,
    factory: llguidance::ParserFactory,
}

impl std::fmt::Debug for LlgEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlgEnv").finish_non_exhaustive()
    }
}

impl LlgEnv {
    /// Walks the model's whole vocabulary. Call once per model.
    ///
    /// # Errors
    ///
    /// Returns [`GrammarError`] if llguidance rejects the tokenizer.
    pub fn new(model: &LlamaModel) -> Result<Self, GrammarError> {
        let tok_env = build_tok_env(model);
        let tok_env_dyn: Arc<dyn toktrie::TokenizerEnv + Sync> = tok_env.clone();
        let factory = llguidance::ParserFactory::new_simple(&tok_env_dyn)
            .map_err(|_| GrammarError::NullGrammar)?;
        Ok(Self { tok_env, factory })
    }
}

/// Create an llguidance-based constrained decoding sampler from a prepared
/// environment.
///
/// Returns the sampler and a handle on its [`GrammarStats`]. The handle is an
/// `Arc` rather than a borrow because the sampler's own copy lives behind a raw
/// pointer that llama.cpp owns and frees, so the counters have to outlive it in
/// order to be read at all -- reading them *after* the generation loop is the
/// entire point.
pub(crate) fn create_llg_sampler(
    env: &LlgEnv,
    grammar_kind: &str,
    grammar_data: &str,
) -> Result<(LlamaSampler, Arc<GrammarStats>), GrammarError> {
    let tok_env = env.tok_env.clone();

    let grammar = llguidance::api::TopLevelGrammar::from_tagged_str(grammar_kind, grammar_data)
        .map_err(|_| GrammarError::NullGrammar)?;

    let parser = env
        .factory
        .create_parser(grammar)
        .map_err(|_| GrammarError::NullGrammar)?;

    let matcher = Matcher::new(Ok(parser));

    let stats = Arc::new(GrammarStats::default());
    let ctx = Box::new(LlgContext {
        matcher,
        tok_env,
        grammar_kind: grammar_kind.to_string(),
        grammar_data: grammar_data.to_string(),
        stats: Arc::clone(&stats),
    });

    let sampler = unsafe {
        koharu_llama_sys::llama_sampler_init(
            &raw mut LLG_SAMPLER_I,
            Box::into_raw(ctx).cast::<c_void>(),
        )
    };

    if sampler.is_null() {
        Err(GrammarError::NullGrammar)
    } else {
        Ok((LlamaSampler { sampler }, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /* These test the arithmetic, not the callback: `llg_apply` needs a live
     * llama.cpp candidate array and a compiled grammar, neither of which a
     * CPU-only unit test can build. What is worth pinning here is the shape of
     * the ratios, because every one of them has an empty denominator on some
     * real page -- a page whose grammar refused nothing has `displaced == 0`,
     * and a NaN there would break the log line rather than read as zero. */

    #[test]
    fn an_untouched_snapshot_reports_zeros_not_nan() {
        let snapshot = GrammarStats::default().snapshot();
        assert_eq!(snapshot.steps, 0);
        assert!(snapshot.allowed_per_step().is_finite());
        assert!(snapshot.allowed_percent().is_finite());
        assert!(snapshot.displaced_share().is_finite());
        assert!(snapshot.mean_displaced_gap().is_finite());
    }

    #[test]
    fn steps_with_no_displacement_leave_the_gap_at_zero() {
        let stats = GrammarStats::default();
        stats.record(1000, 40, false, 0);
        stats.record(1000, 60, false, 0);
        let snapshot = stats.snapshot();

        assert_eq!(snapshot.steps, 2);
        assert_eq!(snapshot.displaced, 0);
        assert!((snapshot.allowed_per_step() - 50.0).abs() < f64::EPSILON);
        assert!((snapshot.allowed_percent() - 5.0).abs() < f64::EPSILON);
        // Zero displaced steps means the mean is over nothing at all, and the
        // honest answer is 0.0 rather than a division by zero.
        assert!((snapshot.mean_displaced_gap() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn the_gap_is_averaged_over_displaced_steps_only() {
        let stats = GrammarStats::default();
        stats.record(1000, 10, true, 3_000);
        stats.record(1000, 10, false, 0);
        stats.record(1000, 10, true, 1_000);
        let snapshot = stats.snapshot();

        assert_eq!(snapshot.steps, 3);
        assert_eq!(snapshot.displaced, 2);
        // 4.0 logits over two displaced steps, not over all three -- a step the
        // grammar never touched would drag the mean toward zero and make a
        // strong constraint look mild.
        assert!((snapshot.mean_displaced_gap() - 2.0).abs() < 1e-9);
        assert!((snapshot.displaced_share() - 2.0 / 3.0).abs() < 1e-9);
    }
}
