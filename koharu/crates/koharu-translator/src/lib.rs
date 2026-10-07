//! Translation through local and hosted providers.

mod backend;
mod error;
mod json;
mod language;
mod local;
mod model;
mod prompt;
mod provider;
mod remote;

use std::sync::Arc;

use koharu_ml::Device;

use error::{Error, Result};
use local::LocalTranslator;

pub use backend::{
    SegmentContext, StyleClause, Translated, TranslationContext, TranslationRequest,
};
pub use language::Language;
pub use model::{GenerationConfig, Model, ModelSelection, Quantization};
pub(crate) use model::{ModelGeneration, QuantizationDefinition, display_name};
pub use provider::{Provider, ProviderConfig, ProvidersConfig};
// Needed to point the OpenAI-compatible provider at a specific endpoint
// (e.g. a local Ollama server) from outside this crate.
pub use remote::OpenAiCompatibleConfig;

/// A translation the model stopped in the middle of.
///
/// Trailing whitespace is the signature, and it is deliberately the only one.
/// Nothing legitimate ends a rendered line with a space -- the renderer would not
/// show it if it did -- so this has no false positives to trade against, and it
/// needs no reference arm, no second run and no model to evaluate.
///
/// It is a **lower bound** and must never be described as "the truncation rate":
/// a cut landing exactly on a word boundary carries no trailing space and is
/// invisible here. An empty string is excluded because the output schema's
/// `minLength` already makes that ungrammatical; `trim` would otherwise class it
/// as cut and send it down a retry that is not the right repair for it.
///
/// **This two-clause core is THE split-fragment fingerprint**, shared by three
/// places — here, `local::split_fragments` (the seed-varied retry trigger), and
/// the server's `regions::split_fragment_indices` diagnostic. The callers keep
/// their own third clauses (`!untranslated.contains` at the retry,
/// `refused.is_none()` on the wire), because those are different questions; a
/// change to THIS definition moves all three sites together instead of
/// silently desyncing the retry from the diagnostic.
pub fn is_cut(text: &str) -> bool {
    !text.trim().is_empty() && text != text.trim_end()
}

/// Whether a cut segment is re-asked. On unless explicitly disabled.
///
/// An environment variable rather than a config field because this is a repair
/// with no user-facing choice in it: the arm exists so the fix can be measured
/// against itself, not so a reader can pick worse output.
fn reask_enabled() -> bool {
    !matches!(
        std::env::var("KOHARU_REASK_CUT_SEGMENTS").as_deref(),
        Ok("0") | Ok("false")
    )
}

/// Whether a segment the provider never answered for is re-asked. On unless
/// explicitly disabled.
///
/// Its own flag and its own pass, NOT an arm of the cut repair: the cut
/// vector feeds `cascade_vote` and the tail/neighbour selectors, so unioning
/// missed ids into it would change which segments those arms select and
/// conflate the two populations in every counter. Re-asking a segment that is
/// KNOWN missed is the safe direction -- the same argument that defaults the
/// cut arm on -- and the accept guard is stricter than the cut path's: a reply
/// byte-equal to its source is still not a translation.
fn reask_untranslated_enabled() -> bool {
    !matches!(
        std::env::var("KOHARU_REASK_UNTRANSLATED").as_deref(),
        Ok("0") | Ok("false")
    )
}

/// What one untranslated re-ask reply did to the page.
enum UntranslatedRepair {
    /// The segment was written AND its index left `untranslated`.
    Repaired,
    /// The reply echoed its own source a second time; nothing changed. Writing
    /// it back would clear the index over an unchanged string -- the one
    /// outcome strictly worse than leaving the report honest.
    Echoed,
    /// Empty, unanswered, or cut; nothing changed.
    Rejected,
}

/// The accept guard AND the commit for one untranslated re-ask, in ONE
/// function, because the post-condition is composed: on accept the segment is
/// written and its index leaves `untranslated`, **both or neither**. A guard
/// tested apart from its commit passes with the clearing unwired, which is
/// exactly how a fix ships green and inert.
fn commit_untranslated_repair(
    translated: &mut Translated,
    index: usize,
    source: &str,
    reply: &Translated,
) -> UntranslatedRepair {
    let Some(text) = reply.segments.first() else {
        return UntranslatedRepair::Rejected;
    };
    if text.trim().is_empty() || !reply.untranslated.is_empty() || is_cut(text) {
        return UntranslatedRepair::Rejected;
    }
    if text == source {
        return UntranslatedRepair::Echoed;
    }
    translated.segments[index] = text.clone();
    translated.untranslated.retain(|&i| i != index);
    UntranslatedRepair::Repaired
}

/// Whether the segment FOLLOWING a cut is re-asked too. Off unless asked for.
///
/// Opposite default to [`reask_enabled`], and deliberately so. Re-asking a cut
/// segment touches something already known to be broken; re-asking its neighbour
/// touches a segment that may be perfectly correct, on the theory that it
/// received a spilled tail. The accept guard cannot tell a better answer from a
/// merely different one, so this arm ships off until it has been measured.
fn reask_neighbour_enabled() -> bool {
    matches!(
        std::env::var("KOHARU_REASK_NEIGHBOUR").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Whether the whole page tail after a cut is re-asked. **On** unless disabled.
///
/// Distinct from [`reask_neighbour_enabled`], which re-asks only `k+1`. That arm
/// measured as a clean null -- 17 replacements, identical residual -- and the
/// census explains why: it repaired one region of a displacement that runs to the
/// end of the page. 20 of 78 scoreable cuts displace their whole tail, ~147
/// regions across 14 pages, with shifts mostly 7-12 long.
///
/// # It shipped OFF first, and what changed
///
/// The benefit was never in doubt: 3 of 3 cascading pages fully repaired, ~28
/// regions realigned, every completeness counter unmoved, verified in pixels -- a
/// bubble holding one cut-off word became a whole sentence and the bubble wrongly
/// holding its spilled remainder got its own line back.
///
/// What held it off was a cost nothing could measure. A spurious firing cannot put
/// wrong content in a region, since the retry is handed that region's own source,
/// but a segment translated *alone* has no neighbours for pronouns, honorifics or
/// register to agree with -- and every counter reads identically either way. The
/// gate's 74% specificity predicts roughly a quarter of firings are spurious, so
/// that unmeasured cost was live on about a quarter of them.
///
/// # THE 60-ITEM PANEL THAT SHIPPED THIS WAS A BROKEN INSTRUMENT
///
/// **Do not quote the null this comment used to carry.** It reported a 60-item
/// blind panel at 22/20/18 per item, sign test **p = 0.878**, and concluded there
/// was no harm in re-asking alone. That panel was **page-blind**, and it does not
/// merely fail to see the effect -- **it reverses the sign, significantly, in both
/// directions.**
///
/// The decisive comparison: 184 context-dependent items, **the same items in both
/// arms**, same A/B side assignment, same five judges, the only difference being
/// whether the judge can see the page. Both control arms clean -- 80 votes on
/// byte-identical text, 0 preferences.
///
/// | arm | re-ask | original | tie | sign test |
/// |---|---|---|---|---|
/// | judged **with the page** | 35 | **103** | 46 | p = 0.000 -> ORIGINAL |
/// | judged **page-blind** | **69** | 40 | 75 | p = 0.007 -> RE-ASK |
///
/// Item by item, **29 items flipped original -> re-ask when the page was removed,
/// and exactly ONE flipped the other way.** That asymmetry is the mechanism: a
/// segment translated alone produces self-contained, fluent English -- it supplies
/// its own subject and closes its own sentence -- which reads BETTER with no page
/// to check against and is WRONG once there is one. Net direction **+68 with the
/// page, -29 without**, which fully explains the old 22/20/18: a real degradation
/// the judges could not see, plus a fluency preference they could, netting to a
/// coin flip.
///
/// Corroborated on a question of fact rather than taste -- 110 pairs where the two
/// arms **contradict each other about who is speaking or acting**, 5 judges,
/// control clean at 50 votes / 0 preferences: **person_conflict 6 re-ask against
/// 21 original, p = 0.006** (`name_conflict` is 7-8, p = 1.000).
///
/// # What this means for the flag
///
/// The old comment offered "turn it off if a larger panel finds a real effect".
/// **That panel has now been run and it found one.** What is recorded here is the
/// measurement, not a decision: the shipping default has not been flipped. Do not
/// read this comment as authority to flip it, and do not re-derive the null.
///
/// One fact from the old comment survives and is worth keeping: **36% of re-asks
/// return a byte-identical string**, so a large share of firings change nothing.
fn reask_tail_enabled() -> bool {
    !matches!(
        std::env::var("KOHARU_REASK_TAIL").as_deref(),
        Ok("0") | Ok("false")
    )
}

/// Whether a re-ask is shown the rest of its page as reference context.
///
/// # The defect this exists to remove, rather than gate around
///
/// A re-ask sends ONE segment, so the model answering it cannot see the page.
/// That is measurably where the damage comes from: on a page whose request held a
/// single segment -- where the re-ask prompt is byte-identical to the page prompt
/// -- 136 of 136 probes came back byte-identical, so there is no sampling noise
/// and every divergence is caused by removing the page. Judged with the page
/// visible, 5.7-7.2% of re-asked segments come back net worse, concentrated
/// entirely in segments whose English depends on page context and absent outside
/// them (37.0% net, p = 0.000, against a null at p = 0.696).
///
/// Raising [`CASCADE_VOTE_THRESHOLD`] worked around that by firing less often.
/// This attacks the cause: put the page back, as `context` rather than as
/// `segments`.
///
/// # Why this is NOT the batching the repair loop refuses
///
/// The loop's own comment rejects "one request for all of them", because a batch
/// of cut segments would hand the model neighbours to spill into -- reproducing
/// the very condition being repaired. That objection is about what goes in
/// `segments`, and this puts nothing there. `TranslationContext` entries are
/// reference only, and `prompt.rs` tells the model so in the system prompt: use
/// them for terminology, voice and continuity, and do not translate or return
/// them. The reply is still one segment, so the id-alignment failure mode has
/// nothing to align badly.
///
/// It is also the mechanism the story window already rides on, so it is proven at
/// 96 pairs a page rather than novel.
///
/// # Off by default until it is measured
///
/// The prediction is sharp and could be wrong: if losing the page is the whole of
/// the cause, a re-ask carrying its page should reproduce the page-pass answer
/// far more often, and the ~63% divergence rate should collapse toward the 0% the
/// single-segment band shows. An arm that cannot be run against itself cannot
/// establish that, so this ships OFF and is switched on for the measurement.
fn reask_page_context_enabled() -> bool {
    matches!(
        std::env::var("KOHARU_REASK_PAGE_CONTEXT").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// The page's other segments, as reference-only context for a re-ask of `index`.
///
/// Appended to whatever context the template already carries -- the story window
/// -- rather than replacing it, because the two answer different questions: the
/// window preserves naming across pages, this preserves reference within one.
///
/// The target itself is excluded, or the model would be handed the answer it is
/// being asked for. A segment whose translation is empty is skipped: an empty
/// pair teaches nothing and spends prompt on it.
fn page_context(
    template: &TranslationRequest,
    sources: &[String],
    translated: &[String],
    index: usize,
) -> Vec<TranslationContext> {
    let mut out = template.context.clone();
    out.extend(
        sources
            .iter()
            .zip(translated.iter())
            .enumerate()
            .filter(|(i, (_, t))| *i != index && !t.trim().is_empty())
            .map(|(_, (s, t))| TranslationContext::new(s.clone(), t.clone())),
    );
    out
}

/// How many healthy segments a page re-asks purely to be measured. 0 = off.
///
/// The panel this feeds needs pairs where the ORIGINAL was already correct, and
/// waiting for the cascade gate to misfire yields about three a volume: reaching
/// fifty would take seventeen volumes. Inducing them instead takes one.
fn probe_count() -> usize {
    std::env::var("KOHARU_REASK_PROBE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// Segments worth probing: already translated, and translated properly.
///
/// Everything excluded here is excluded because including it would answer a
/// different question than the panel is asking:
///
/// - a **cut** segment is the population the cut repair already handles;
/// - a translation **equal to its source** was never translated, so re-asking it
///   measures "translating beats not translating" -- that contamination reached a
///   built panel once and three of its seven items were this;
/// - a **very short** source is punctuation or a lone interjection, where the two
///   renderings differ by a character and a judge is scoring noise.
///
/// Evenly spaced rather than the first N, because the first segments of a page are
/// systematically different -- titles, narration boxes -- and probing only those
/// would sample one register of the page.
fn probe_indices(sources: &[String], translations: &[String], want: usize) -> Vec<usize> {
    if want == 0 {
        return Vec::new();
    }
    let healthy: Vec<usize> = (0..sources.len().min(translations.len()))
        .filter(|&index| {
            let source = sources[index].trim();
            let translated = translations[index].trim();
            !translated.is_empty()
                && !is_cut(&translations[index])
                && translated != source
                && source.chars().count() >= 4
        })
        .collect();
    if healthy.is_empty() {
        return Vec::new();
    }
    let take = want.min(healthy.len());
    (0..take)
        .map(|n| healthy[n * healthy.len() / take])
        .collect()
}

/// How much of the tail after a cut reads as belonging to the PREVIOUS region.
///
/// Under a correct alignment a translation's length tracks its own source; under a
/// one-off shift it tracks the previous one. This is the fraction of tail regions
/// fitting the previous source better, and it is the same statistic the offline
/// measurement computes -- deliberately, so the shipped gate and the offline
/// instrument cannot drift apart and disagree about the same page.
///
/// Measured against 88 hand-adjudicated cases: **AUC 0.914**, leave-one-out
/// sensitivity 90% and specificity 74% at [`CASCADE_VOTE_THRESHOLD`]. A *vote*
/// rather than a mean because a mean is dominated by two wildly mismatched regions
/// while a vote asks the same question of every one; the mean version of this
/// signal caught 6 of 20 and its published rate was wrong by 4x.
///
/// Log ratio because Japanese-to-English expansion is multiplicative: a
/// 4-character source becoming 20 characters is ordinary, and a linear difference
/// would call every short region a mismatch.
///
/// # Its false-positive mode, which is real and cost a test
///
/// Where source lengths fall steeply across a page and the English runs longer
/// than the Japanese, the previous and longer source fits better by arithmetic
/// alone and the vote climbs with nothing wrong. That is most of the 26% the
/// specificity gives away. It is why the gate is a *filter on top of an existing
/// cut* rather than a page-level detector on its own: a page with no cut never
/// reaches this code.
///
/// Returns 0.0 for a tail of fewer than two scorable regions, which is not
/// "aligned" but "no evidence" -- the caller must treat it as a refusal to fire.
fn cascade_vote(sources: &[String], translations: &[String], cut_index: usize) -> f64 {
    fn err(translated: &str, source: &str) -> f64 {
        ((translated.chars().count() + 1) as f64 / (source.chars().count() + 1) as f64)
            .ln()
            .abs()
    }
    let mut wins = 0usize;
    let mut total = 0usize;
    for index in (cut_index + 1)..translations.len().min(sources.len()) {
        let translated = &translations[index];
        if translated.trim().is_empty() || index == 0 {
            continue;
        }
        total += 1;
        if err(translated, &sources[index - 1]) < err(translated, &sources[index]) {
            wins += 1;
        }
    }
    if total < 2 {
        return 0.0;
    }
    wins as f64 / total as f64
}

/// Fitted on the 88 census-labelled cases as the most specific threshold still
/// reaching 90% sensitivity. Not a natural constant: refit it if the corpus moves.
///
/// # Raised from 0.35 to 0.50, because a firing has a measured cost
///
/// 0.35 was chosen when the only thing being traded was recall against a cost
/// nobody could measure. It is measured now: on the segments a firing actually
/// rewrites, ~10% come back worse, and 70 of 77 rewrites addressed no
/// displacement at all. Under a real cost the gate should buy precision.
///
/// The census sweep says it can:
///
/// | thr | caught | missed | false+ | recall | prec |
/// |---|---|---|---|---|---|
/// | 0.35 | 16 | 1 | 12 | 94% | 57% |
/// | 0.45 | 11 | 6 | 4 | 65% | 73% |
/// | 0.50 | 10 | 7 | **0** | 59% | **100%** |
///
/// That table is FITTED -- it picks 0.50 by looking at the cases the detector was
/// tuned on -- so it is the shape and not the number. The number comes from a
/// held-out corpus that was never in the census: 77 tail segments rewritten on a
/// Japanese test volume, judged by a five-judge panel with the page in front of
/// it, split by their page's vote.
///
/// | vote band | rewritten | repaired | harmed | net | true displacements |
/// |---|---|---|---|---|---|
/// | 0.00-0.40 | 17 | 1 | 4 | **-3** | 0 |
/// | 0.40-0.50 | 41 | 9 | 3 | +6 | 0 |
/// | above 0.50 | 19 | 8 | 1 | **+7** | **7 of 7** |
///
/// Below 0.40 the repair is net NEGATIVE. Every genuine displacement the panel
/// could identify sits above 0.50. So 0.50 keeps all 7 real cascades and +7 of
/// the +10 net benefit while cutting rewrites from 77 to 19 -- a 75% drop in
/// exposure for 30% of the benefit.
///
/// What this costs, stated plainly: recall falls from 94% to 59% on the census,
/// so roughly four cascading pages a volume go unrepaired. That is the trade, and
/// it is the right way round only because an unrepaired cascade leaves text that
/// was already wrong, while a spurious firing damages text that was right.
const CASCADE_VOTE_THRESHOLD: f64 = 0.50;

/// Every region from the cut to the end of the page.
///
/// Individually, and NOT by applying an assumed offset, because the offset is not
/// always one: one census page is displaced by two, its cut's tail having
/// consumed two ids with no resync. Re-asking each region on its own makes the
/// repair independent of how far the page slipped.
fn tail_indices(cut: &[usize], from: usize, len: usize) -> Vec<usize> {
    (from..len).filter(|index| !cut.contains(index)).collect()
}

/// The segments following a cut one, which is where a spilled tail lands.
///
/// Pure and separate from the retry loop so the selection can be tested without a
/// model. Three rules, each of which is a case that occurs in the corpus:
///
/// - a neighbour past the end of the page is dropped, because the last region on
///   a page has nowhere to spill to, and the corpus has that case;
/// - a neighbour that is ITSELF cut is dropped, because it is already in the cut
///   list and re-asking it twice would waste a call -- one corpus page has
///   adjacent cuts at ids 2 and 3;
/// - duplicates are dropped, so two cuts in a row cannot queue the same follower.
fn neighbour_indices(cut: &[usize], len: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for index in cut.iter().map(|index| index + 1) {
        if index < len && !cut.contains(&index) && !out.contains(&index) {
            out.push(index);
        }
    }
    out
}

#[derive(Clone)]
pub struct Translator {
    providers: koharu_config::Config<ProvidersConfig>,
    local: Arc<tokio::sync::Mutex<Option<LoadedLocal>>>,
    client: reqwest::Client,
    device: Device,
}

struct LoadedLocal {
    selection: ModelSelection,
    translator: Arc<LocalTranslator>,
}

impl Translator {
    pub fn from_config(
        device: Device,
        providers: koharu_config::Config<ProvidersConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            providers,
            local: Arc::new(tokio::sync::Mutex::new(None)),
            client: koharu_runtime::http_client()?,
            device,
        })
    }

    #[must_use]
    pub fn model(selection: &ModelSelection) -> &'static str {
        selection.provider.into()
    }

    #[must_use]
    pub fn loaded(&self, selection: &ModelSelection) -> bool {
        if selection.provider != Provider::Local {
            return true;
        }
        self.local
            .try_lock()
            .map(|loaded| {
                loaded
                    .as_ref()
                    .is_some_and(|loaded| loaded.selection == *selection)
            })
            .unwrap_or(true)
    }

    pub fn unload(&self) -> bool {
        self.local
            .try_lock()
            .map(|mut loaded| loaded.take().is_some())
            .unwrap_or(false)
    }

    pub async fn load_model(&self, selection: &ModelSelection) -> anyhow::Result<()> {
        if selection.provider == Provider::Local {
            self.local(selection).await?;
        }
        Ok(())
    }

    /// Translates a page's segments, reporting what came back as well as the
    /// text.
    ///
    /// [`Translated::untranslated`] is the part a caller must not ignore. It is
    /// never an error -- the page is still readable and mostly right -- but the
    /// segments it names render in the *source* language, which on a manga page
    /// is indistinguishable from a stage that never ran. The segment-count guard
    /// below cannot stand in for it: a reply is padded back to full length from
    /// the source before it ever gets here, so the count always matches.
    pub async fn translate(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        request: TranslationRequest,
    ) -> anyhow::Result<(&'static str, Translated)> {
        let provider = selection.provider;
        let provider_id: &'static str = provider.into();
        if request.segments.is_empty() {
            return Ok((provider_id, Translated::complete(request.segments)));
        }

        let expected = request.segments.len();
        // Kept for the re-ask below, which needs the SOURCE text of the segments
        // it retries; `request` is moved into the provider call.
        let sources = request.segments.clone();
        let retry_template = request.clone();
        let retry_generation = generation.clone();
        let mut translated = self
            .dispatch(selection, provider, generation, request)
            .await?;
        if translated.segments.len() != expected {
            return Err(Error::SegmentCount {
                provider: provider_id,
                expected,
                actual: translated.segments.len(),
            }
            .into());
        }

        self.repair_cut_segments(
            selection,
            provider,
            &retry_generation,
            &retry_template,
            &sources,
            &mut translated,
        )
        .await;

        self.repair_untranslated_segments(
            selection,
            provider,
            &retry_generation,
            &retry_template,
            &sources,
            &mut translated,
        )
        .await;

        self.probe_reask(
            selection,
            provider,
            &retry_generation,
            &retry_template,
            &sources,
            &translated,
        )
        .await;

        Ok((provider_id, translated))
    }

    /// Re-ask healthy segments purely to measure what asking alone costs.
    ///
    /// **It never writes to `translated`, and that is the whole point.** The
    /// output of a probed run is byte-identical to an unprobed one, so this can be
    /// switched on over any corpus without the measurement changing the thing being
    /// measured, and a probed arm stays comparable to every arm already recorded.
    ///
    /// The question it exists for cannot be answered by the counters: a re-ask is
    /// handed the region's own source and so cannot put the wrong content in it,
    /// but a segment translated *without its page* has no neighbours for pronouns,
    /// honorifics or register to agree with. The first panel caught exactly that --
    /// a contemplative grunt came back as "Mmph..." (physical strain) where the
    /// page-context rendering was "Hmm...", and all five judges
    /// preferred the original. One observed case is a mechanism, not a rate.
    ///
    /// Waiting for the cascade gate to misfire yields ~3 usable pairs a volume.
    /// This yields one per page.
    async fn probe_reask(
        &self,
        selection: &ModelSelection,
        provider: Provider,
        generation: &GenerationConfig,
        template: &TranslationRequest,
        sources: &[String],
        translated: &Translated,
    ) {
        let want = probe_count();
        if want == 0 {
            return;
        }
        for index in probe_indices(sources, &translated.segments, want) {
            let mut retry = template.clone();
            retry.segments = vec![sources[index].clone()];
            if reask_page_context_enabled() {
                retry.context = page_context(template, sources, &translated.segments, index);
            }
            let Ok(result) = self
                .dispatch(selection, provider, generation.clone(), retry)
                .await
            else {
                continue;
            };
            let Some(text) = result.segments.first() else {
                continue;
            };
            if text.trim().is_empty() || !result.untranslated.is_empty() {
                continue;
            }
            tracing::info!(
                target: "koharu_translator",
                role = "probe",
                index,
                source = sources[index].as_str(),
                before = translated.segments[index].as_str(),
                after = text.as_str(),
                "probed a healthy segment"
            );
        }
    }

    /// One provider call, whichever family the selection names.
    ///
    /// Extracted so the initial request and the cut-segment re-ask go down the
    /// *same* path. A retry that took a different route would be measuring the
    /// route.
    async fn dispatch(
        &self,
        selection: &ModelSelection,
        provider: Provider,
        generation: GenerationConfig,
        request: TranslationRequest,
    ) -> anyhow::Result<Translated> {
        if provider == Provider::Local {
            Ok(self
                .local(selection)
                .await?
                .translate(request, generation)
                .await?)
        } else {
            let providers = self.providers.read()?.clone();
            Ok(remote::translate(&self.client, &providers, selection, &generation, &request).await?)
        }
    }

    /// Re-ask for segments the model cut off mid-sentence, one at a time.
    ///
    /// # The defect
    ///
    /// A reply can be valid, complete, correctly addressed -- and still stop in
    /// the middle of a sentence. Measured on a 213-page volume: 11-28 segments a
    /// run, 5-12 per 1,000 translations, against 3 a run on pages the token cap
    /// already reports. **Nothing sees it.** The text is not empty, so the
    /// dropped report cannot; it does not equal its source, so `untranslated`
    /// cannot; the page is under budget, so `truncated` is false.
    ///
    /// Two shapes were read off real pages. The remainder is simply lost -- a
    /// whole line came back as its first word and a trailing space.
    /// Or it *spills* into the next id, which then loses its own content and
    /// renders confident English over the wrong artwork: verified in pixels, a
    /// bubble rendered only the first word of its line while an unrelated
    /// bubble rendered the rest.
    ///
    /// # Why a re-ask, and why it should work
    ///
    /// Three explanations were eliminated with numbers before this was written.
    /// Not the token budget: excluding pages the cap already flags, the AUC of
    /// segment count on "this page has a cut" is 0.550, chance. Not the repairing
    /// parser: 63 of 71 cuts have a LATER id on the same page answered, which a
    /// truncated reply cannot do. Not the prompt, which already says "Never
    /// merge, split, omit, or add segments" -- and five reworded system prompts
    /// across two panels have never beaten the shipped one.
    ///
    /// What is left is that the grammar constrains the *shape* of the JSON and
    /// not its content, so a string may legally end anywhere, and the model ends
    /// one where it believes the sentence continues into the next segment.
    ///
    /// The fix follows from the measurement that matters: cuts are **not**
    /// deterministic. At a fixed configuration, two runs differing only by an
    /// unrelated patch share 2 of ~21 cut segments -- Jaccard 0.10 -- and 39 of
    /// 54 cut segments across four arms are cut in exactly one of them. A segment
    /// re-asked under different conditions overwhelmingly comes back whole. And a
    /// segment asked **on its own** has nowhere to spill to.
    ///
    /// # What it deliberately does not do
    ///
    /// It does not touch a segment whose retry is also cut, so a genuinely
    /// hard segment keeps the better of two answers rather than looping. It does
    /// not re-ask empty segments -- the output schema's `minLength` already makes
    /// those ungrammatical, and an empty reply here would mean something else is
    /// wrong.
    /// It never lengthens `untranslated`: a retry that fails leaves the original
    /// text exactly as it was.
    ///
    /// Set `KOHARU_REASK_CUT_SEGMENTS=0` to disable, which is the A/B arm.
    async fn repair_cut_segments(
        &self,
        selection: &ModelSelection,
        provider: Provider,
        generation: &GenerationConfig,
        template: &TranslationRequest,
        sources: &[String],
        translated: &mut Translated,
    ) {
        let cut: Vec<usize> = translated
            .segments
            .iter()
            .enumerate()
            .filter(|(index, text)| is_cut(text) && sources.get(*index).is_some())
            .map(|(index, _)| index)
            .collect();
        /* The wire's verdict pair, and the detection runs BEFORE
         * the flag check on purpose: an off arm that reported nothing would
         * reproduce the exact ambiguity the telemetry comment below records.
         * `still_cut` here is what SHIPS if no repair runs; every later exit
         * recounts it from the final text, so a failed retry is included --
         * unlike the tracing line's local counter, which names only the
         * retried-alone-and-still-cut subset. */
        translated.cut_found = cut.len();
        translated.still_cut = cut.len();
        if !reask_enabled() {
            return;
        }
        if cut.is_empty() {
            return;
        }

        /* The segment AFTER a cut is where a spilled tail lands, and it is not
         * itself cut -- so nothing above selects it and its own content stays
         * lost. Measured on a re-baseline: 4 of 6 surviving cuts are this
         * shape, e.g. one segment rendering "It's " while the next carries the
         * whole correct translation of the first one's source.
         *
         * Off by default because it is the one arm here that can touch a segment
         * that is not itself faulty: a neighbour may be perfectly correct, and
         * the accept guard below tests only that a reply is non-empty, uncut and
         * answered -- not that it is better. Measure before shipping it on. */
        let len = sources.len().min(translated.segments.len());
        /* Gated on the cascade vote rather than fired on every cut, and the
         * neighbour arm is why. That arm re-asked k+1 unconditionally, accepted 17
         * replacements and moved nothing measurable -- it rewrote correct segments
         * on pages that were never displaced. Only ~26% of cuts cascade, so firing
         * the tail repair on all of them would rewrite three correct pages for
         * every damaged one it fixed. The vote costs nothing but string lengths. */
        let vote = cascade_vote(sources, &translated.segments, cut[0]);
        let cascading = reask_tail_enabled() && vote > CASCADE_VOTE_THRESHOLD;
        let neighbours: Vec<usize> = if cascading {
            tail_indices(&cut, cut[0] + 1, len)
        } else if reask_neighbour_enabled() {
            neighbour_indices(&cut, len)
        } else {
            Vec::new()
        };

        // One segment per request rather than one request for all of them. A
        // batch of the cut segments would reproduce the very condition being
        // repaired -- neighbours to spill into -- and the population is small
        // enough (a handful a page at most) that the extra calls are cheap
        // beside a page that is already ~1.4s of translation.
        let cut_found = cut.len();
        let neighbour_found = neighbours.len();
        let mut repaired_cut = 0usize;
        let mut repaired_neighbour = 0usize;
        let mut still_cut = 0usize;
        let mut failed = 0usize;

        for (index, is_neighbour) in cut
            .iter()
            .map(|index| (*index, false))
            .chain(neighbours.iter().map(|index| (*index, true)))
        {
            let mut retry = template.clone();
            retry.segments = vec![sources[index].clone()];
            // The page as reference, never as segments -- see
            // `reask_page_context_enabled`. On a cascading page the neighbours
            // shown here may themselves be displaced, which is a real limit on
            // what this can repair and is why the probe arm, whose neighbours are
            // healthy by construction, is the population it gets measured on.
            if reask_page_context_enabled() {
                retry.context = page_context(template, sources, &translated.segments, index);
            }
            let Ok(result) = self
                .dispatch(selection, provider, generation.clone(), retry)
                .await
            else {
                failed += 1;
                continue;
            };
            let Some(text) = result.segments.first() else {
                failed += 1;
                continue;
            };
            // Reject an answer that is empty, unanswered, or still cut. A retry
            // failing any of those leaves the page exactly as it was.
            //
            // THAT IS THE WHOLE OF WHAT THIS CHECKS, and this comment used to
            // claim it "can improve a segment and never degrade one". It cannot
            // make that promise and never could: nothing here compares the answer
            // to the page. A re-ask that comes back fluent, complete, and wrong
            // about who is speaking passes every test above and overwrites correct
            // text. Measured on a five-judge panel with the page visible, ~10% of
            // the segments a firing rewrites come back worse. The block comment
            // below already said the page is what a lone re-ask loses; the guard's
            // own claim was simply never narrowed to match it.
            if text.trim().is_empty() || !result.untranslated.is_empty() {
                failed += 1;
                continue;
            }
            if is_cut(text) {
                // Asked ALONE and still cut. That is the interesting residual --
                // it cannot be a spill, because there was nothing to spill into.
                still_cut += 1;
                continue;
            }
            /* The BEFORE and AFTER of every accepted replacement, which is the
             * only way to panel this repair honestly.
             *
             * The counters say "17 replaced" and name none of them, so the obvious
             * comparison -- diff two arms -- cannot separate a replacement from
             * ordinary sampling drift, and every segment differs between two runs
             * of a sampled model. Emitting the pair from inside the one run that
             * made it removes the confound entirely.
             *
             * It matters because of what a spurious firing costs. A re-ask cannot
             * put the wrong content in a region: it is handed that region's own
             * source. What it can lose is the PAGE -- a segment translated alone
             * has no neighbours, so pronouns, honorifics and continuity have
             * nothing to agree with, and those are exactly what the story window
             * exists to preserve. No counter in this project can see that, which
             * is the same blindness that made the neighbour arm look harmless.
             * Judging it needs the two strings side by side.
             *
             * `role` separates the two populations, and they are not comparable: a
             * `cut` replacement is repairing text known to be broken, while a
             * `neighbour` replacement may be overwriting text that was perfectly
             * correct. Only the second is evidence about the cost of firing. */
            tracing::info!(
                target: "koharu_translator",
                role = if is_neighbour { "neighbour" } else { "cut" },
                index,
                source = sources[index].as_str(),
                before = translated.segments[index].as_str(),
                after = text.as_str(),
                "replaced a segment"
            );
            translated.segments[index] = text.clone();
            if is_neighbour {
                repaired_neighbour += 1;
            } else {
                repaired_cut += 1;
            }
        }

        /* Logged rather than returned, and the reason is worth stating: adding a
         * field to `Translated` means editing koharu-pipeline's `Progress` and
         * the server's response shape too, three crates for a number whose only
         * consumer is measurement. `koharu-translator` already logs per-page
         * telemetry at this level, so this is the established route.
         *
         * It exists because the cut repair first shipped WITHOUT it, and that
         * made 4 surviving cuts unattributable: "the retry fired and came
         * back cut" and "the retry never fired" are opposite faults and looked
         * identical from outside. `still_cut` is the field that separates them. */
        tracing::info!(
            target: "koharu_translator",
            cut_found,
            neighbour_found,
            repaired_cut,
            repaired_neighbour,
            still_cut,
            failed,
            // The gate's own input, so a page that did NOT get the tail repair can
            // be told from one that was never eligible. Without it a zero in
            // `neighbour_found` is ambiguous between "not cascading" and "the arm
            // is off", which is the same silence this telemetry exists to end.
            cascade_vote = format!("{vote:.2}"),
            cascading,
            /* Enough to NAME the page in the response JSON, which this line could
             * not do and which cost a real analysis: the story-96 arm fired the
             * gate on two pages and neither could be located afterwards, because a
             * log line saying only `cascading=true` identifies nothing.
             *
             * This crate has no page id -- it is handed segments, not a scene. But
             * `segments` is the page's region count and `cut_at` is the index list,
             * and in practice those two pick one page out of a 213-page run.
             * Threading a real id through `TranslationRequest` is a wider change
             * than the problem needs. */
            segments = sources.len(),
            cut_at = format!("{cut:?}"),
            "re-asked segments the model cut off"
        );

        /* The wire's `still_cut` is the FINAL state -- what ships. Recounted
         * from the text itself so a retry that failed, was rejected, or came
         * back cut is included; the local `still_cut` counter above stays as
         * the tracing line's narrower retried-alone diagnostic. */
        translated.still_cut = translated
            .segments
            .iter()
            .filter(|text| is_cut(text))
            .count();
    }

    /// Re-ask every segment the provider never answered for.
    ///
    /// # The defect this repairs, measured on a test page
    ///
    /// The reply is SEEDED from the source (`prompt.rs`), so an id the model
    /// never answers ships as `translated == source` -- rendered as source
    /// text in the theme font, which for a CJK page means tofu wherever the
    /// resolved chain lacks a glyph. The wire reports it (`untranslated: [2]`,
    /// `duplicate_ids: 1` on that page -- llguidance pins the COUNT of reply
    /// entries but implements no `uniqueItems`, so a duplicated id always
    /// steals a segment from somebody), but nothing downstream acted on it.
    /// A segment re-asked ALONE cannot be mis-addressed: there is only one id
    /// to answer.
    ///
    /// # The accept guard, and why it is stricter than the cut path's
    ///
    /// A reply is accepted only if it is non-empty, answered, not cut, and
    /// **not byte-equal to its source** -- an echo is exactly the failure being
    /// repaired, so writing it back and clearing the index would turn a
    /// visible defect into an invisible one. On accept the segment is written
    /// AND its index leaves `untranslated`; both or neither, which is the
    /// composed post-condition the tests assert. A retry that fails any test
    /// leaves the page exactly as it was -- this pass never lengthens
    /// `untranslated` and never rewrites an answered segment.
    ///
    /// Set `KOHARU_REASK_UNTRANSLATED=0` to disable, which is the A/B arm.
    async fn repair_untranslated_segments(
        &self,
        selection: &ModelSelection,
        provider: Provider,
        generation: &GenerationConfig,
        template: &TranslationRequest,
        sources: &[String],
        translated: &mut Translated,
    ) {
        if !reask_untranslated_enabled() {
            return;
        }
        let missed: Vec<usize> = translated
            .untranslated
            .iter()
            .copied()
            .filter(|&index| {
                sources
                    .get(index)
                    .is_some_and(|source| !source.trim().is_empty())
            })
            .collect();
        if missed.is_empty() {
            return;
        }

        let untranslated_found = missed.len();
        let mut repaired_untranslated = 0usize;
        let mut still_echoed = 0usize;
        let mut failed = 0usize;

        for index in missed {
            let mut retry = template.clone();
            retry.segments = vec![sources[index].clone()];
            if reask_page_context_enabled() {
                retry.context = page_context(template, sources, &translated.segments, index);
            }
            let Ok(result) = self
                .dispatch(selection, provider, generation.clone(), retry)
                .await
            else {
                failed += 1;
                continue;
            };
            let before = translated.segments[index].clone();
            match commit_untranslated_repair(translated, index, &sources[index], &result) {
                UntranslatedRepair::Repaired => {
                    tracing::info!(
                        target: "koharu_translator",
                        role = "untranslated",
                        index,
                        source = sources[index].as_str(),
                        before = before.as_str(),
                        after = translated.segments[index].as_str(),
                        "replaced a segment"
                    );
                    repaired_untranslated += 1;
                }
                UntranslatedRepair::Echoed => still_echoed += 1,
                UntranslatedRepair::Rejected => failed += 1,
            }
        }

        tracing::info!(
            target: "koharu_translator",
            untranslated_found,
            repaired_untranslated,
            still_echoed,
            failed,
            segments = sources.len(),
            "re-asked segments the model never answered"
        );
    }

    pub async fn models() -> anyhow::Result<Vec<Model>> {
        let providers = ProvidersConfig::load()?;
        let providers = providers.read()?.clone();
        let client = koharu_runtime::http_client()?;
        let mut models = local::models();
        models.extend(remote::models(&client, &providers).await);
        Ok(models)
    }

    async fn local(&self, selection: &ModelSelection) -> Result<Arc<LocalTranslator>> {
        let mut loaded = self.local.lock().await;
        if loaded
            .as_ref()
            .is_none_or(|loaded| loaded.selection != *selection)
        {
            *loaded = Some(LoadedLocal {
                selection: selection.clone(),
                translator: Arc::new(LocalTranslator::load(self.device.clone(), selection).await?),
            });
        }
        Ok(Arc::clone(
            &loaded
                .as_ref()
                .expect("local translator was loaded")
                .translator,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        cascade_vote, commit_untranslated_repair, is_cut, neighbour_indices, page_context,
        probe_indices, tail_indices, Translated, UntranslatedRepair, CASCADE_VOTE_THRESHOLD,
    };
    use crate::{Language, TranslationContext, TranslationRequest};

    fn request_with_window() -> TranslationRequest {
        TranslationRequest::new(["ignored"], Language::English)
            .with_context([TranslationContext::new("前巻の台詞", "a line from last volume")])
    }

    /// The target must not appear in its own context, or the re-ask is handed the
    /// answer it is being asked for and every measurement of it is worthless.
    #[test]
    fn page_context_excludes_the_segment_being_re_asked() {
        let sources = vec!["あ".to_string(), "い".to_string(), "う".to_string()];
        let translated = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let ctx = page_context(&request_with_window(), &sources, &translated, 1);
        assert!(
            ctx.iter().all(|c| c.source != "い" && c.translation != "B"),
            "the re-asked segment leaked into its own context"
        );
        assert_eq!(ctx.iter().filter(|c| c.source == "あ").count(), 1);
        assert_eq!(ctx.iter().filter(|c| c.source == "う").count(), 1);
    }

    /// The story window is preserved and stays FIRST. The two contexts answer
    /// different questions -- naming across pages against reference within one --
    /// and dropping the window here would silently undo the story feature for
    /// every re-asked segment.
    #[test]
    fn page_context_keeps_the_story_window_ahead_of_the_page() {
        let sources = vec!["あ".to_string(), "い".to_string()];
        let translated = vec!["A".to_string(), "B".to_string()];
        let ctx = page_context(&request_with_window(), &sources, &translated, 0);
        assert_eq!(ctx.len(), 2, "window pair plus the one other segment");
        assert_eq!(ctx[0].source, "前巻の台詞");
        assert_eq!(ctx[1].source, "い");
    }

    /// An untranslated neighbour teaches nothing and costs prompt. It also must
    /// not be able to shorten the zip: `sources` and `translated` are the same
    /// length by construction here, and a blank must be filtered, not truncate.
    #[test]
    fn page_context_skips_neighbours_with_no_translation() {
        let sources = vec!["あ".to_string(), "い".to_string(), "う".to_string()];
        let translated = vec![String::new(), "B".to_string(), "   ".to_string()];
        let ctx = page_context(&request_with_window(), &sources, &translated, 1);
        assert_eq!(ctx.len(), 1, "only the window pair survives");
        assert_eq!(ctx[0].source, "前巻の台詞");
    }

    /// The fire case, asserted on the COMPOSED post-condition. On a test
    /// page the reply duplicated an id (llguidance pins entry count,
    /// not distinctness), region 2 shipped holding its source, and the wire
    /// read `untranslated: [2]`. A good re-ask must write the segment AND
    /// empty `untranslated` -- both or neither; a selector or guard asserted
    /// alone passes with the clearing unwired.
    #[test]
    fn a_good_reply_repairs_the_segment_and_clears_the_report() {
        let source = "妈妈让他赶紧记住这两个人的名字，免得明天在老师面前出错。";
        let mut page = Translated {
            segments: vec!["A".to_owned(), "B".to_owned(), source.to_owned()],
            untranslated: vec![2],
            ..Translated::default()
        };
        let reply = Translated {
            segments: vec![
                "Mom told him to learn those two names fast, so he won't slip up tomorrow."
                    .to_owned(),
            ],
            ..Translated::default()
        };
        assert!(matches!(
            commit_untranslated_repair(&mut page, 2, source, &reply),
            UntranslatedRepair::Repaired
        ));
        assert_ne!(page.segments[2], source, "the segment must hold the reply");
        assert!(
            page.untranslated.is_empty(),
            "the repaired index must leave the untranslated report"
        );
    }

    /// Every reply the guard refuses leaves the page BYTE-IDENTICAL: an echo
    /// (the failure being repaired -- writing it back would clear the index
    /// over an unchanged string), a cut reply, an unanswered reply, and an
    /// empty one. `untranslated` is never lengthened and never shortened.
    #[test]
    fn a_refused_reply_changes_nothing_at_all() {
        let source = "今日はもう帰りたい．．．";
        let make = || Translated {
            segments: vec!["X".to_owned(), source.to_owned()],
            untranslated: vec![1],
            ..Translated::default()
        };
        let reply = |segments: Vec<&str>, untranslated: Vec<usize>| Translated {
            segments: segments.into_iter().map(str::to_owned).collect(),
            untranslated,
            ..Translated::default()
        };

        let mut page = make();
        assert!(matches!(
            commit_untranslated_repair(&mut page, 1, source, &reply(vec![source], vec![])),
            UntranslatedRepair::Echoed
        ));
        assert_eq!(page, make(), "an echo must change nothing");

        for (broken, why) in [
            (reply(vec!["It's "], vec![]), "a cut reply"),
            (reply(vec![source], vec![0]), "an unanswered reply"),
            (reply(vec!["   "], vec![]), "a blank reply"),
            (reply(vec![], vec![]), "an empty reply"),
        ] {
            let mut page = make();
            assert!(
                matches!(
                    commit_untranslated_repair(&mut page, 1, source, &broken),
                    UntranslatedRepair::Rejected
                ),
                "{why} must be rejected"
            );
            assert_eq!(page, make(), "{why} must change nothing");
        }
    }

    /// The two shapes read off real pages, and the reason the signature is
    /// trailing whitespace rather than anything cleverer.
    #[test]
    fn a_translation_stopping_mid_sentence_is_cut() {
        // A line whose whole translation reads like "This road goes no further than--".
        assert!(is_cut("This "));
        // Cut before the last word of "Here, say ah~".
        assert!(is_cut("Here, say "));
        assert!(is_cut("It's not like I "));
        // A newline counts: the model stopped, it just stopped on a different
        // whitespace character.
        assert!(is_cut("And then\n"));
        assert!(is_cut("Three\t"));
    }

    #[test]
    fn an_ordinary_translation_is_not_cut() {
        assert!(!is_cut("What a mess!!!"));
        assert!(!is_cut("Geez..."));
        assert!(!is_cut("*Tremble*"));
        // Interior whitespace is every translation ever written.
        assert!(!is_cut("The bus to the harbour leaves at seven sharp."));
        // A LEADING space is not evidence of anything -- the cut is at the end.
        assert!(!is_cut(" Hah..."));
    }

    /// Empty is excluded on purpose, and this is not a detail.
    ///
    /// The output schema makes an empty translation ungrammatical, so one arriving
    /// here means something other than a cut went wrong, and a re-ask is not its
    /// repair. Without this arm `"   "` would trim to nothing, class as cut, and
    /// be sent down the wrong path.
    /// Everything the probe refuses, and each exclusion is a contamination that
    /// reached a real panel or would have.
    #[test]
    fn the_probe_only_takes_healthy_segments() {
        let sources = vec![
            "ちょうど週末は暇だから付き合おう".to_owned(), // healthy
            "まるで春の嵐のように".to_owned(),             // cut translation
            "今日はもう帰りたい．．．".to_owned(),         // untranslated: equals source
            "？".to_owned(),                               // too short to judge
            "青木春香は町内屈指の料理下手である".to_owned(), // healthy
        ];
        let translations = vec![
            "I happen to be free this weekend, so sure.".to_owned(),
            "storm. ".to_owned(),
            "今日はもう帰りたい．．．".to_owned(),
            "?".to_owned(),
            "Haruka Aoki is among the worst cooks in town".to_owned(),
        ];
        // Asking for more than exist yields only the healthy ones, in order.
        assert_eq!(probe_indices(&sources, &translations, 9), vec![0, 4]);
        assert_eq!(probe_indices(&sources, &translations, 0), Vec::<usize>::new());
    }

    /// Evenly spaced, not the first N: a page's opening segments are titles and
    /// narration boxes, so probing only those samples one register.
    #[test]
    fn the_probe_spreads_across_the_page() {
        let sources: Vec<String> = (0..10).map(|n| format!("これはテスト{n}です")).collect();
        let translations: Vec<String> = (0..10).map(|n| format!("This is test {n}.")).collect();
        let picked = probe_indices(&sources, &translations, 3);
        assert_eq!(picked.len(), 3);
        assert_eq!(picked[0], 0);
        assert!(picked[2] >= 6, "third probe at {} is not near the end", picked[2]);
        let mut sorted = picked.clone();
        sorted.dedup();
        assert_eq!(picked.len(), sorted.len(), "probe picked the same index twice");
    }

    /// A test page's cascade, confirmed by reading the page, in invented text
    /// that keeps every character count: the vote reads lengths and nothing else.
    ///
    /// Region 2 is cut; 3 carries 2's tail, 4 carries 3's line, 5 carries 4's. The
    /// vote must clear the threshold on this, or the shipped gate would decline to
    /// repair the one page everyone has looked at.
    #[test]
    fn the_confirmed_cascade_clears_the_gate() {
        let sources = vec![
            "手伝いはするけど料理対決って何なの．．．？".to_owned(),
            "青木春香（８歳）料理経験「無し".to_owned(),
            "第６話青木春香は逃げた".to_owned(),
            "あれ．．．．．何？".to_owned(),
            "なら明日の試合で本気を出してみれば".to_owned(),
        ];
        let translations = vec![
            "I'll help you out, but what do you mean by ".to_owned(),
            "a cooking battle...?".to_owned(),
            "Haruka Aoki (eight years old) | Cooking experience: None".to_owned(),
            "Chapter 6: Haruka Aoki ran away from home".to_owned(),
            "Wait... What?".to_owned(),
        ];
        let vote = cascade_vote(&sources, &translations, 0);
        assert!(
            vote > CASCADE_VOTE_THRESHOLD,
            "confirmed cascade scored {vote}, at or under the {CASCADE_VOTE_THRESHOLD} gate"
        );
    }

    /// A test page adjudicated as NOT a cascade: every region in the
    /// tail renders its own source. It must not fire, or the repair rewrites
    /// correct text -- which is how the neighbour arm spent 17 replacements for
    /// nothing.
    ///
    /// The text is invented but every character count is the real page's, and
    /// that is not a style preference. A synthetic "aligned" page written for
    /// this test with made-up lengths scored 0.75 and failed:
    /// its source lengths happened to fall 6, 11, 10, 5, 2 while English runs
    /// longer than Japanese, so the previous, longer source fit better by
    /// arithmetic alone. That is a genuine false-positive mode of the signal --
    /// see the note on [`cascade_vote`] -- and a hand-written fixture had walked
    /// straight into it while looking innocuous.
    #[test]
    fn an_aligned_page_does_not_clear_the_gate() {
        let sources = vec![
            "今日もまた３連敗かよ".to_owned(),
            "う．．．．．".to_owned(),
            "あの箱の中身はてっきり新しい靴か何かだと思っていたのに".to_owned(),
            "だんだん眠くなってきた．．．".to_owned(),
            "あと少し様子を見てみようか．．．".to_owned(),
            "これは．．．".to_owned(),
        ];
        let translations = vec![
            "Three ".to_owned(),
            "Hm...".to_owned(),
            "That's three ".to_owned(),
            "I'm getting sleepy.".to_owned(),
            "I'll wait a little longer and see how this plays out".to_owned(),
            "Is it...".to_owned(),
        ];
        let vote = cascade_vote(&sources, &translations, 0);
        assert!(
            vote <= CASCADE_VOTE_THRESHOLD,
            "aligned page scored {vote}, over the {CASCADE_VOTE_THRESHOLD} gate"
        );
    }

    /// A tail of under two scorable regions is "no evidence", not "aligned", and
    /// must return 0.0 so the caller declines rather than guesses.
    #[test]
    fn too_short_a_tail_returns_no_evidence() {
        let sources = vec!["あ".to_owned(), "い".to_owned()];
        let translations = vec!["a ".to_owned(), "b".to_owned()];
        assert_eq!(cascade_vote(&sources, &translations, 0), 0.0);
    }

    #[test]
    fn the_tail_runs_to_the_end_and_skips_regions_already_cut() {
        // Cut at 2, another cut at 5: the tail covers 3,4,6,7 and leaves 5 to the
        // cut list rather than queueing it twice.
        assert_eq!(tail_indices(&[2, 5], 3, 8), vec![3, 4, 6, 7]);
        // Nothing after the cut is not an error, just an empty tail.
        assert_eq!(tail_indices(&[7], 8, 8), Vec::<usize>::new());
    }

    /// The three rules, each drawn from a case that occurs in the corpus.
    #[test]
    fn a_neighbour_past_the_end_of_the_page_is_not_queued() {
        // The last region on its page: nothing to spill into.
        assert_eq!(neighbour_indices(&[4], 5), Vec::<usize>::new());
        assert_eq!(neighbour_indices(&[3], 5), vec![4]);
    }

    #[test]
    fn a_neighbour_that_is_itself_cut_is_not_queued_twice() {
        // Adjacent cuts at ids 2 and 3. Re-asking 3 as a
        // neighbour of 2 would spend a second call on a segment already in the
        // cut list.
        assert_eq!(neighbour_indices(&[2, 3], 10), vec![4]);
    }

    #[test]
    fn two_cuts_cannot_queue_the_same_follower() {
        // Not reachable from adjacent cuts alone, but the retry loop must never
        // be handed a duplicate index: the second pass would overwrite the
        // first result with another draw for no reason.
        let queued = neighbour_indices(&[1, 2, 5], 10);
        let mut sorted = queued.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(queued.len(), sorted.len());
        assert_eq!(queued, vec![3, 6]);
    }

    #[test]
    fn an_empty_or_blank_translation_is_not_a_cut() {
        assert!(!is_cut(""));
        assert!(!is_cut(" "));
        assert!(!is_cut("   \n\t "));
    }
}
